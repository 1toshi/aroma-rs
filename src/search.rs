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
    AromaIndex, LanguageIndex, MethodLocation, Query, COUNT_OVERFLOW, COUNT_SHIFT, MID_MASK,
};
use crate::prune::{
    prune_fast, prune_with_fast_map, FastQueryMap, PruneScratch, PruneTarget, Pruned,
};
use crate::MethodRef;

pub const DEFAULT_SEARCH_LIMIT: usize = 1000;
const DEFAULT_RESULT_LIMIT: usize = 100;
const PAR_PRUNE_TOK_THRESHOLD: usize = 768;
const PAR_PRUNE_REC_TOK_THRESHOLD: usize = 1_000_000;

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
    raw_cands: Vec<(u64, u32)>,
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
    lidx: &LanguageIndex,
    loc: &MethodLocation,
    origin: Option<(Option<u32>, &MethodRef)>,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
    index: &AromaIndex,
) -> bool {
    if let Some((orig_fid, orig)) = origin
        && orig_fid == Some(loc.file_id)
        && loc.start_line <= orig.end_line
        && loc.end_line >= orig.start_line
    {
        return false;
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
    mid: usize,
    origin: Option<(Option<u32>, &MethodRef)>,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
) -> Option<&'a MethodLocation> {
    if exclude.is_none() {
        match origin {
            None => return Some(&unsafe { lidx.methods.get_unchecked(mid) }.primary),
            Some((None, _)) => return Some(&unsafe { lidx.methods.get_unchecked(mid) }.primary),
            Some((Some(ofid), _)) => {
                if unsafe { *lidx.canonical_file_id.get_unchecked(mid) } != ofid {
                    return Some(&unsafe { lidx.methods.get_unchecked(mid) }.primary);
                }
            }
        }
    }
    let m = unsafe { lidx.methods.get_unchecked(mid) };
    if is_location_usable(lidx, &m.primary, origin, exclude, index) {
        return Some(&m.primary);
    }
    m.copies
        .iter()
        .find(|c| is_location_usable(lidx, c, origin, exclude, index))
}

pub(crate) fn rank_scored<'a>(
    index: &'a AromaIndex,
    lidx: &'a LanguageIndex,
    query: &Query,
    search_limit: usize,
    top_k: usize,
    exclude: Option<&(dyn Fn(&MethodRef) -> bool + Sync)>,
    sc: &mut SearchScratch,
) -> Vec<ScoredCandidate<'a>> {
    let mcount = lidx.methods.len();
    if mcount == 0 || search_limit == 0 || top_k == 0 || query.features.is_empty() || query.size == 0 {
        return Vec::new();
    }

    let origin: Option<(Option<u32>, &MethodRef)> =
        query.origin.as_ref().map(|o| (index.path_to_id.get(&o.path).copied(), o));
    let max_possible_ov = query.features.len();
    let use_packed_32 = query.size <= u16::MAX as usize && max_possible_ov <= u16::MAX as usize;

    sc.raw_cands.clear();

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
                let (chunks, rem) = slice.as_chunks::<4>();
                for ch in chunks {
                    let m0 = (ch[0] & MID_MASK) as usize;
                    let m1 = (ch[1] & MID_MASK) as usize;
                    let m2 = (ch[2] & MID_MASK) as usize;
                    let m3 = (ch[3] & MID_MASK) as usize;
                    unsafe {
                        let a0 = *acc.get_unchecked(m0);
                        let a1 = *acc.get_unchecked(m1);
                        let a2 = *acc.get_unchecked(m2);
                        let a3 = *acc.get_unchecked(m3);
                        *acc.get_unchecked_mut(m0) = a0 + 0x0001_0001;
                        *acc.get_unchecked_mut(m1) = a1 + 0x0001_0001;
                        *acc.get_unchecked_mut(m2) = a2 + 0x0001_0001;
                        *acc.get_unchecked_mut(m3) = a3 + 0x0001_0001;
                    }
                }
                for &p in rem {
                    let mid = (p & MID_MASK) as usize;
                    unsafe {
                        *acc.get_unchecked_mut(mid) += 0x0001_0001;
                    }
                }
            } else if cap <= COUNT_OVERFLOW {
                let (chunks, rem) = slice.as_chunks::<4>();
                for ch in chunks {
                    let m0 = (ch[0] & MID_MASK) as usize;
                    let m1 = (ch[1] & MID_MASK) as usize;
                    let m2 = (ch[2] & MID_MASK) as usize;
                    let m3 = (ch[3] & MID_MASK) as usize;
                    let c0 = (ch[0] >> COUNT_SHIFT).min(cap);
                    let c1 = (ch[1] >> COUNT_SHIFT).min(cap);
                    let c2 = (ch[2] >> COUNT_SHIFT).min(cap);
                    let c3 = (ch[3] >> COUNT_SHIFT).min(cap);
                    unsafe {
                        let a0 = *acc.get_unchecked(m0);
                        let a1 = *acc.get_unchecked(m1);
                        let a2 = *acc.get_unchecked(m2);
                        let a3 = *acc.get_unchecked(m3);
                        *acc.get_unchecked_mut(m0) = a0 + 0x0001_0000 + c0;
                        *acc.get_unchecked_mut(m1) = a1 + 0x0001_0000 + c1;
                        *acc.get_unchecked_mut(m2) = a2 + 0x0001_0000 + c2;
                        *acc.get_unchecked_mut(m3) = a3 + 0x0001_0000 + c3;
                    }
                }
                for &p in rem {
                    let mid = (p & MID_MASK) as usize;
                    let cnt = (p >> COUNT_SHIFT).min(cap);
                    unsafe {
                        *acc.get_unchecked_mut(mid) += 0x0001_0000 + cnt;
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
        let (chunks, rem) = acc.as_chunks::<8>();
        for chunk in chunks {
            if (chunk[0] | chunk[1] | chunk[2] | chunk[3] | chunk[4] | chunk[5] | chunk[6] | chunk[7]) == 0 {
                continue;
            }
            for &v in chunk {
                if v != 0 {
                    let ov = (v >> 16) as usize;
                    unsafe { *sc.hist.get_unchecked_mut(ov) += 1 };
                    if ov > max_ov {
                        max_ov = ov;
                    }
                    total_nonzero += 1;
                }
            }
        }
        for &v in rem {
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

        let mut base_mid = 0usize;
        let (bchunks, brem) = acc.as_chunks::<8>();
        for chunk in bchunks {
            if (chunk[0] | chunk[1] | chunk[2] | chunk[3] | chunk[4] | chunk[5] | chunk[6] | chunk[7]) == 0 {
                base_mid += 8;
                continue;
            }
            for (k, &v) in chunk.iter().enumerate() {
                let ov = (v >> 16) as usize;
                if ov >= cutoff {
                    let b = max_ov - ov;
                    let p = unsafe { *sc.bucket_offsets.get_unchecked(b) } as usize;
                    unsafe {
                        *sc.bucket_mids.get_unchecked_mut(p) = (base_mid + k) as u32;
                        *sc.bucket_offsets.get_unchecked_mut(b) = (p + 1) as u32;
                    }
                }
            }
            base_mid += 8;
        }
        for (k, &v) in brem.iter().enumerate() {
            let ov = (v >> 16) as usize;
            if ov >= cutoff {
                let b = max_ov - ov;
                let p = unsafe { *sc.bucket_offsets.get_unchecked(b) } as usize;
                unsafe {
                    *sc.bucket_mids.get_unchecked_mut(p) = (base_mid + k) as u32;
                    *sc.bucket_offsets.get_unchecked_mut(b) = (p + 1) as u32;
                }
            }
        }

        for &mid in &sc.bucket_mids {
            if sc.raw_cands.len() >= search_limit {
                break;
            }
            if first_usable_location(index, lidx, mid as usize, origin, exclude).is_some() {
                let score = unsafe { *acc.get_unchecked(mid as usize) } & 0xFFFF;
                let idx = sc.raw_cands.len() as u64;
                let key = ((!score as u64) << 32) | idx;
                sc.raw_cands.push((key, mid));
            }
        }

        if sc.raw_cands.len() < search_limit && cutoff > 1 {
            let mut rest: Vec<(u32, u32, u32)> = Vec::new();
            for (mid, &v) in acc.iter().enumerate() {
                let ov = v >> 16;
                if ov > 0 && (ov as usize) < cutoff {
                    rest.push((ov, mid as u32, v & 0xFFFF));
                }
            }
            rest.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, mid, score) in rest {
                if sc.raw_cands.len() >= search_limit {
                    break;
                }
                if first_usable_location(index, lidx, mid as usize, origin, exclude).is_some() {
                    let idx = sc.raw_cands.len() as u64;
                    let key = ((!score as u64) << 32) | idx;
                    sc.raw_cands.push((key, mid));
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
            if sc.raw_cands.len() >= search_limit {
                break;
            }
            if first_usable_location(index, lidx, mid as usize, origin, exclude).is_some() {
                let idx = sc.raw_cands.len() as u64;
                let key = ((!score as u64) << 32) | idx;
                sc.raw_cands.push((key, mid));
            }
        }
    }

    // Partition and sort only the needed `top_k` candidates by `(score DESC, phase1_idx ASC)`.
    let keep = sc.raw_cands.len().min(top_k);
    if sc.raw_cands.len() > keep {
        sc.raw_cands.select_nth_unstable_by_key(keep - 1, |&(k, _)| k);
        sc.raw_cands.truncate(keep);
    }
    sc.raw_cands.sort_unstable_by_key(|&(k, _)| k);
    sc.raw_cands
        .iter()
        .map(|&(k, mid)| {
            let loc = first_usable_location(index, lidx, mid as usize, origin, exclude)
                .expect("candidate was already verified usable");
            ScoredCandidate { mid, loc, score: !((k >> 32) as u32) }
        })
        .collect()
}

fn matched_lines_of(lidx: &LanguageIndex, loc: &MethodLocation, tokens: &[usize]) -> Vec<u32> {
    let base = loc.start_line;
    let span = loc.end_line.saturating_sub(base) + 1;
    if span <= 512 {
        let mut bits = [0u64; 8];
        let mut overflow: Option<Vec<u32>> = None;
        let mut count = 0usize;
        for &t in tokens {
            let (a, b) = lidx.tok_line_span(loc, t);
            for ln in a..=b {
                let rel = ln.wrapping_sub(base);
                if rel < 512 {
                    let w = (rel >> 6) as usize;
                    let mask = 1u64 << (rel & 63);
                    if (bits[w] & mask) == 0 {
                        bits[w] |= mask;
                        count += 1;
                    }
                } else {
                    overflow.get_or_insert_with(Vec::new).push(ln);
                }
            }
        }
        if let Some(mut extra) = overflow {
            for (w, &word) in bits.iter().enumerate() {
                let mut x = word;
                let wbase = base + (w as u32) * 64;
                while x != 0 {
                    let tz = x.trailing_zeros();
                    extra.push(wbase + tz);
                    x &= x - 1;
                }
            }
            extra.sort_unstable();
            extra.dedup();
            return extra;
        }
        let mut out = Vec::with_capacity(count);
        for (w, &word) in bits.iter().enumerate() {
            let mut x = word;
            let wbase = base + (w as u32) * 64;
            while x != 0 {
                let tz = x.trailing_zeros();
                out.push(wbase + tz);
                x &= x - 1;
            }
        }
        out
    } else {
        let mut lines: Vec<u32> = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let (a, b) = lidx.tok_line_span(loc, t);
            for ln in a..=b {
                lines.push(ln);
            }
        }
        lines.sort_unstable();
        lines.dedup();
        lines
    }
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
        let scored = rank_scored(
            index,
            lidx,
            query,
            options.search_limit,
            options.limit,
            options.exclude,
            sc,
        );

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
            && survivors.len() >= 4
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
    let scored = rank_scored(index, lidx, query, search_limit, cluster_limit, exclude, sc);
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
        && survivors.len() >= 16
        && total_toks >= PAR_PRUNE_REC_TOK_THRESHOLD
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
