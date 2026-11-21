//! Phases I & II of Aroma (`search` and `rank`).
//!
//! Single-pass packed `(overlap << 16) | score` accumulation over per-language
//! postings computes BOTH Phase I distinct-feature overlap AND the exact
//! Phase II `prune` score simultaneously. `prune_with_fast_map` (4 KB L1 Bloom,
//! direct `qidx` lookup, exact signature-class greedy, and expected-score early
//! exit) then runs ONLY on surviving hits — strictly single-threaded when built
//! with `--threads 1`, or parallelized over `index.pool` when `threads > 1` and
//! pruning work exceeds the parallel threshold.

use std::cell::RefCell;

use rayon::prelude::*;

use crate::index::{
    AromaIndex, IndexedMethod, LanguageIndex, MethodLocation, Query, COUNT_OVERFLOW, COUNT_SHIFT,
    MID_MASK,
};
use crate::prune::{
    prune_fast, prune_with_fast_map, FastQueryMap, PruneScratch, PruneTarget, Pruned,
};
use crate::MethodRef;

pub const DEFAULT_SEARCH_LIMIT: usize = 1000;
const DEFAULT_RESULT_LIMIT: usize = 100;
const PAR_PRUNE_TOK_THRESHOLD: usize = 1536;

pub struct SearchOptions<'a> {
    pub search_limit: usize,
    pub limit: usize,
    pub min_similarity: f64,
    pub exclude: Option<&'a (dyn Fn(&MethodRef) -> bool + Sync)>,
}

impl Default for SearchOptions<'_> {
    fn default() -> Self {
        Self {
            search_limit: DEFAULT_SEARCH_LIMIT,
            limit: DEFAULT_RESULT_LIMIT,
            min_similarity: 0.0,
            exclude: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub method: MethodRef,
    pub similarity: f64,
    pub matched_lines: Vec<u32>,
}

pub(crate) struct ScoredCandidate<'a> {
    pub mid: u32,
    pub loc: &'a MethodLocation,
    pub score: u32,
}

pub(crate) struct RankedCandidate<'a> {
    pub mid: u32,
    pub loc: &'a MethodLocation,
    pub pruned: Pruned,
}

#[derive(Default)]
pub(crate) struct SearchScratch {
    acc32: Vec<u32>,
    acc64: Vec<u64>,
    hist: Vec<u32>,
    bucket_offsets: Vec<u32>,
    bucket_mids: Vec<u32>,
    pub fast_qmap: FastQueryMap,
    pub prune_target: PruneTarget,
    pub prune_scratch: PruneScratch,
}

thread_local! {
    pub(crate) static TL_SEARCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
    static TL_PAR_PRUNE: RefCell<PruneScratch> = RefCell::new(PruneScratch::default());
}

#[inline]
fn is_location_usable(
    index: &AromaIndex,
    lidx: &LanguageIndex,
    loc: &MethodLocation,
    origin: Option<&MethodRef>,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
) -> bool {
    if let Some(orig) = origin {
        let loc_path = &index.sources[loc.file_id as usize].path;
        if loc_path == &orig.path && loc.start_line <= orig.end_line && loc.end_line >= orig.start_line {
            return false;
        }
    }
    if let Some(ex) = exclude {
        let mref = index.make_ref(lidx, loc);
        if ex(&mref) {
            return false;
        }
    }
    true
}

#[inline]
fn first_usable_location<'a>(
    index: &AromaIndex,
    lidx: &'a LanguageIndex,
    m: &'a IndexedMethod,
    origin: Option<&MethodRef>,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
) -> Option<&'a MethodLocation> {
    if is_location_usable(index, lidx, &m.primary, origin, exclude) {
        return Some(&m.primary);
    }
    m.copies.iter().find(|c| is_location_usable(index, lidx, c, origin, exclude))
}

pub(crate) fn rank_scored<'a>(
    index: &'a AromaIndex,
    lidx: &'a LanguageIndex,
    query: &Query,
    search_limit: usize,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
    sc: &mut SearchScratch,
) -> Vec<ScoredCandidate<'a>> {
    let mcount = lidx.methods.len();
    if mcount == 0 || search_limit == 0 || query.features.is_empty() || query.size == 0 {
        return Vec::new();
    }

    let origin = query.origin.as_ref();
    let max_possible_ov = query.features.len();
    let use_packed_32 = query.size <= u16::MAX as usize && max_possible_ov <= u16::MAX as usize;

    let mut out: Vec<ScoredCandidate<'a>> = Vec::with_capacity(search_limit.min(mcount));

    if use_packed_32 {
        sc.acc32.clear();
        sc.acc32.resize(mcount, 0u32);
        let acc = &mut sc.acc32;

        for &(fid, cap) in &query.features {
            let fi = fid as usize;
            let ps = lidx.posting_offsets[fi] as usize;
            let pe = lidx.posting_offsets[fi + 1] as usize;
            let slice = &lidx.postings[ps..pe];
            if cap == 1 {
                for &p in slice {
                    let mid = (p & MID_MASK) as usize;
                    unsafe {
                        *acc.get_unchecked_mut(mid) += 0x0001_0001;
                    }
                }
            } else {
                for &p in slice {
                    let mid = (p & MID_MASK) as usize;
                    let raw = p >> COUNT_SHIFT;
                    let cnt = if raw < COUNT_OVERFLOW {
                        raw
                    } else {
                        lidx.posting_overflow[&(fid, mid as u32)]
                    };
                    let add = 0x0001_0000 + cnt.min(cap);
                    unsafe {
                        *acc.get_unchecked_mut(mid) += add;
                    }
                }
            }
        }

        let hist_len = max_possible_ov + 1;
        sc.hist.clear();
        sc.hist.resize(hist_len, 0u32);
        let mut max_ov = 0usize;
        let mut total_nonzero = 0usize;
        for &v in acc.iter() {
            if v != 0 {
                let ov = (v >> 16) as usize;
                unsafe { *sc.hist.get_unchecked_mut(ov) += 1 };
                if ov > max_ov {
                    max_ov = ov;
                }
                total_nonzero += 1;
            }
        }
        if total_nonzero == 0 {
            return Vec::new();
        }

        let need_buffer = if exclude.is_some() {
            total_nonzero
        } else {
            search_limit.saturating_add(8)
        };
        let mut cutoff = 1usize;
        let mut cum = 0usize;
        for ov in (1..=max_ov).rev() {
            cum += sc.hist[ov] as usize;
            if cum >= need_buffer {
                cutoff = ov;
                break;
            }
        }

        let range_len = max_ov - cutoff + 1;
        sc.bucket_offsets.clear();
        sc.bucket_offsets.resize(range_len + 1, 0u32);
        let mut off = 0u32;
        for b in 0..range_len {
            sc.bucket_offsets[b] = off;
            off += sc.hist[max_ov - b];
        }
        sc.bucket_offsets[range_len] = off;
        sc.bucket_mids.resize(off as usize, 0u32);

        for (mid, &v) in acc.iter().enumerate() {
            let ov = (v >> 16) as usize;
            if ov >= cutoff {
                let b = max_ov - ov;
                let p = unsafe { *sc.bucket_offsets.get_unchecked(b) } as usize;
                unsafe {
                    *sc.bucket_mids.get_unchecked_mut(p) = mid as u32;
                    *sc.bucket_offsets.get_unchecked_mut(b) = (p + 1) as u32;
                }
            }
        }

        for &mid in &sc.bucket_mids {
            if out.len() >= search_limit {
                break;
            }
            if let Some(loc) =
                first_usable_location(index, lidx, &lidx.methods[mid as usize], origin, exclude)
            {
                out.push(ScoredCandidate { mid, loc, score: acc[mid as usize] & 0xFFFF });
            }
        }

        if out.len() < search_limit && cutoff > 1 {
            let mut rest: Vec<(u32, u32, u32)> = Vec::new();
            for (mid, &v) in acc.iter().enumerate() {
                let ov = v >> 16;
                if ov > 0 && (ov as usize) < cutoff {
                    rest.push((ov, mid as u32, v & 0xFFFF));
                }
            }
            rest.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, mid, score) in rest {
                if out.len() >= search_limit {
                    break;
                }
                if let Some(loc) =
                    first_usable_location(index, lidx, &lidx.methods[mid as usize], origin, exclude)
                {
                    out.push(ScoredCandidate { mid, loc, score });
                }
            }
        }
    } else {
        sc.acc64.clear();
        sc.acc64.resize(mcount, 0u64);
        let acc = &mut sc.acc64;
        for &(fid, cap) in &query.features {
            let fi = fid as usize;
            let ps = lidx.posting_offsets[fi] as usize;
            let pe = lidx.posting_offsets[fi + 1] as usize;
            for &p in &lidx.postings[ps..pe] {
                let mid = (p & MID_MASK) as usize;
                let raw = p >> COUNT_SHIFT;
                let cnt = if raw < COUNT_OVERFLOW {
                    raw
                } else {
                    lidx.posting_overflow[&(fid, mid as u32)]
                };
                acc[mid] += (1u64 << 32) + cnt.min(cap) as u64;
            }
        }
        let mut cands: Vec<(u32, u32, u32)> = Vec::new();
        for (mid, &v) in acc.iter().enumerate() {
            if v != 0 {
                cands.push(((v >> 32) as u32, mid as u32, v as u32));
            }
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (_, mid, score) in cands {
            if out.len() >= search_limit {
                break;
            }
            if let Some(loc) =
                first_usable_location(index, lidx, &lidx.methods[mid as usize], origin, exclude)
            {
                out.push(ScoredCandidate { mid, loc, score });
            }
        }
    }

    // Stably sort by `score DESC` (preserving Phase I order on ties!)
    out.sort_by_key(|a| std::cmp::Reverse(a.score));
    out
}

fn matched_lines_of(lidx: &LanguageIndex, loc: &MethodLocation, tokens: &[usize]) -> Vec<u32> {
    let (starts, ends) = lidx.location_token_lines(loc);
    let mut lines: Vec<u32> = Vec::with_capacity(tokens.len());
    for &t in tokens {
        for ln in starts[t]..=ends[t] {
            lines.push(ln);
        }
    }
    lines.sort_unstable();
    lines.dedup();
    lines
}

pub fn search(index: &AromaIndex, query: &Query, options: &SearchOptions<'_>) -> Vec<SearchResult> {
    if options.limit == 0 || options.search_limit == 0 || query.size == 0 || query.features.is_empty() {
        return Vec::new();
    }
    let Some(lidx) = index.lang_index_by_id(&query.language) else {
        return Vec::new();
    };
    let qsize = query.size as f64;

    TL_SEARCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        let scored = rank_scored(index, lidx, query, options.search_limit, options.exclude, sc);

        // Slice strictly to the surviving hits (`similarity > min_similarity`, `<= limit`)
        let mut survive_end = 0usize;
        let mut total_toks = 0usize;
        let cap = scored.len().min(options.limit);
        while survive_end < cap {
            if (scored[survive_end].score as f64 / qsize) <= options.min_similarity {
                break;
            }
            total_toks += lidx.methods[scored[survive_end].mid as usize].primary.tok_count as usize;
            survive_end += 1;
        }
        let survivors = &scored[..survive_end];
        if survivors.is_empty() {
            return Vec::new();
        }

        let feature_count = index.stats.features;
        let use_fast = sc.fast_qmap.load(feature_count, &query.features);
        if !use_fast {
            sc.prune_target.build_from_pairs(&query.features);
        }
        let qmap = &sc.fast_qmap;
        let ptarget = &sc.prune_target;

        let results = if let Some(pool) = index.pool.as_ref()
            && survivors.len() >= 8
            && total_toks >= PAR_PRUNE_TOK_THRESHOLD
        {
            pool.install(|| {
                survivors
                    .par_iter()
                    .map(|cand| {
                        TL_PAR_PRUNE.with(|pcell| {
                            let mut ps = pcell.borrow_mut();
                            let m = &lidx.methods[cand.mid as usize];
                            let offsets = lidx.method_offsets(cand.mid as usize);
                            let pruned = if use_fast {
                                prune_with_fast_map(
                                    qmap,
                                    &lidx.all_features,
                                    offsets,
                                    m.feat_start as usize,
                                    Some(cand.score as usize),
                                    None,
                                    &mut ps,
                                )
                            } else {
                                prune_fast(
                                    ptarget,
                                    &lidx.all_features,
                                    offsets,
                                    m.feat_start as usize,
                                    Some(cand.score as usize),
                                    None,
                                    &mut ps,
                                )
                            };
                            debug_assert_eq!(pruned.score, cand.score as usize);
                            let matched_lines = matched_lines_of(lidx, cand.loc, &pruned.tokens);
                            SearchResult {
                                method: index.make_ref(lidx, cand.loc),
                                similarity: cand.score as f64 / qsize,
                                matched_lines,
                            }
                        })
                    })
                    .collect()
            })
        } else {
            let mut out = Vec::with_capacity(survivors.len());
            for cand in survivors {
                let m = &lidx.methods[cand.mid as usize];
                let offsets = lidx.method_offsets(cand.mid as usize);
                let pruned = if use_fast {
                    prune_with_fast_map(
                        qmap,
                        &lidx.all_features,
                        offsets,
                        m.feat_start as usize,
                        Some(cand.score as usize),
                        None,
                        &mut sc.prune_scratch,
                    )
                } else {
                    prune_fast(
                        ptarget,
                        &lidx.all_features,
                        offsets,
                        m.feat_start as usize,
                        Some(cand.score as usize),
                        None,
                        &mut sc.prune_scratch,
                    )
                };
                debug_assert_eq!(pruned.score, cand.score as usize);
                let matched_lines = matched_lines_of(lidx, cand.loc, &pruned.tokens);
                out.push(SearchResult {
                    method: index.make_ref(lidx, cand.loc),
                    similarity: cand.score as f64 / qsize,
                    matched_lines,
                });
            }
            out
        };

        if use_fast {
            sc.fast_qmap.clear();
        }
        results
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rank_for_recommend<'a>(
    index: &'a AromaIndex,
    lidx: &'a LanguageIndex,
    query: &Query,
    search_limit: usize,
    cluster_limit: usize,
    min_similarity: f64,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
    sc: &mut SearchScratch,
) -> Vec<RankedCandidate<'a>> {
    if cluster_limit == 0 || search_limit == 0 || query.size == 0 || query.features.is_empty() {
        return Vec::new();
    }
    let scored = rank_scored(index, lidx, query, search_limit, exclude, sc);
    let qsize = query.size as f64;
    let mut survive_end = 0usize;
    let mut total_toks = 0usize;
    let cap = scored.len().min(cluster_limit);
    while survive_end < cap {
        if (scored[survive_end].score as f64 / qsize) <= min_similarity {
            break;
        }
        total_toks += lidx.methods[scored[survive_end].mid as usize].primary.tok_count as usize;
        survive_end += 1;
    }
    let survivors = &scored[..survive_end];
    if survivors.is_empty() {
        return Vec::new();
    }

    let use_fast = sc.fast_qmap.load(index.stats.features, &query.features);
    if !use_fast {
        sc.prune_target.build_from_pairs(&query.features);
    }
    let qmap = &sc.fast_qmap;
    let ptarget = &sc.prune_target;

    let out = if let Some(pool) = index.pool.as_ref()
        && survivors.len() >= 8
        && total_toks >= PAR_PRUNE_TOK_THRESHOLD
    {
        pool.install(|| {
            survivors
                .par_iter()
                .map(|cand| {
                    TL_PAR_PRUNE.with(|pcell| {
                        let mut ps = pcell.borrow_mut();
                        let m = &lidx.methods[cand.mid as usize];
                        let offsets = lidx.method_offsets(cand.mid as usize);
                        let pruned = if use_fast {
                            prune_with_fast_map(
                                qmap,
                                &lidx.all_features,
                                offsets,
                                m.feat_start as usize,
                                Some(cand.score as usize),
                                None,
                                &mut ps,
                            )
                        } else {
                            prune_fast(
                                ptarget,
                                &lidx.all_features,
                                offsets,
                                m.feat_start as usize,
                                Some(cand.score as usize),
                                None,
                                &mut ps,
                            )
                        };
                        debug_assert_eq!(pruned.score, cand.score as usize);
                        RankedCandidate { mid: cand.mid, loc: cand.loc, pruned }
                    })
                })
                .collect()
        })
    } else {
        let mut out = Vec::with_capacity(survivors.len());
        for cand in survivors {
            let m = &lidx.methods[cand.mid as usize];
            let offsets = lidx.method_offsets(cand.mid as usize);
            let pruned = if use_fast {
                prune_with_fast_map(
                    qmap,
                    &lidx.all_features,
                    offsets,
                    m.feat_start as usize,
                    Some(cand.score as usize),
                    None,
                    &mut sc.prune_scratch,
                )
            } else {
                prune_fast(
                    ptarget,
                    &lidx.all_features,
                    offsets,
                    m.feat_start as usize,
                    Some(cand.score as usize),
                    None,
                    &mut sc.prune_scratch,
                )
            };
            debug_assert_eq!(pruned.score, cand.score as usize);
            out.push(RankedCandidate { mid: cand.mid, loc: cand.loc, pruned });
        }
        out
    };

    if use_fast {
        sc.fast_qmap.clear();
    }
    out
}
