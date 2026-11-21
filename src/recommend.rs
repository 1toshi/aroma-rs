//! Phase III of Aroma (`recommend`): clustering, cluster intersection, and
//! zero-reparse line rendering using precomputed `StructSpan`s.

use rustc_hash::FxHashMap;

use crate::index::{AromaIndex, LanguageIndex, MethodLocation, Query};
use crate::prune::{prune_fast, PruneScratch, PruneTarget};
use crate::search::{rank_for_recommend, DEFAULT_SEARCH_LIMIT, TL_SEARCH};
use crate::text::{drop_chars, js_indent, js_trim, js_trim_end};
use crate::MethodRef;

pub struct RecommendOptions<'a> {
    pub search_limit: usize,
    pub cluster_limit: usize,
    pub min_similarity: f64,
    pub min_expansion: f64,
    pub min_cohesion: f64,
    pub max_cluster_overlap: f64,
    pub limit: usize,
    pub min_support: usize,
    pub exclude: Option<&'a (dyn Fn(&MethodRef) -> bool + Sync)>,
}

impl Default for RecommendOptions<'_> {
    fn default() -> Self {
        Self {
            search_limit: DEFAULT_SEARCH_LIMIT,
            cluster_limit: 100,
            min_similarity: 0.65,
            min_expansion: 1.5,
            min_cohesion: 0.9,
            max_cluster_overlap: 0.5,
            limit: 5,
            min_support: 2,
            exclude: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineRole {
    Query,
    Added,
    Structure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedLine {
    pub line: u32,
    pub text: String,
    pub role: LineRole,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub method: MethodRef,
    pub members: Vec<MethodRef>,
    pub similarity: f64,
    pub lines: Vec<RecommendedLine>,
    pub code: String,
}

/// Compact sorted multiset of `(feature_id, count)` with cached total weight.
struct SortedMultiset {
    pairs: Vec<(u32, u32)>,
    total: usize,
}

fn build_multiset_from_slice(feats: &[u32], tmp: &mut Vec<u32>) -> SortedMultiset {
    tmp.clear();
    tmp.extend_from_slice(feats);
    tmp.sort_unstable();
    rle_to_multiset(tmp)
}

fn build_multiset_from_tokens(
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    tokens: &[usize],
    tmp: &mut Vec<u32>,
) -> SortedMultiset {
    tmp.clear();
    for &t in tokens {
        let s = feat_base + offsets[t] as usize;
        let e = feat_base + offsets[t + 1] as usize;
        tmp.extend_from_slice(&all_features[s..e]);
    }
    tmp.sort_unstable();
    rle_to_multiset(tmp)
}

fn rle_to_multiset(sorted: &[u32]) -> SortedMultiset {
    let mut pairs = Vec::with_capacity(sorted.len());
    let mut i = 0;
    while i < sorted.len() {
        let f = sorted[i];
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == f {
            j += 1;
        }
        pairs.push((f, (j - i) as u32));
        i = j;
    }
    SortedMultiset { pairs, total: sorted.len() }
}

/// Two-pointer multiset intersection size with early exit when remaining weight
/// cannot exceed `min_needed` (`> min_needed` required).
#[inline]
fn intersection_size(a: &[(u32, u32)], b: &[(u32, u32)]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut total = 0usize;
    let al = a.len();
    let bl = b.len();
    while i < al && j < bl {
        let (fa, ca) = unsafe { *a.get_unchecked(i) };
        let (fb, cb) = unsafe { *b.get_unchecked(j) };
        if fa == fb {
            total += ca.min(cb) as usize;
            i += 1;
            j += 1;
        } else if fa < fb {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

fn intersect_multisets(a: &[(u32, u32)], b: &[(u32, u32)], out: &mut Vec<(u32, u32)>) -> usize {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    let mut total = 0usize;
    while i < a.len() && j < b.len() {
        let (fa, ca) = a[i];
        let (fb, cb) = b[j];
        if fa == fb {
            let m = ca.min(cb);
            out.push((fa, m));
            total += m as usize;
            i += 1;
            j += 1;
        } else if fa < fb {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// Sorted multiset sum: `a + b` (for `intersect`: `featuresOf(other) + query.features`).
fn add_multisets(a: &[(u32, u32)], b: &[(u32, u32)], out: &mut Vec<(u32, u32)>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let (fa, ca) = a[i];
        let (fb, cb) = b[j];
        if fa == fb {
            out.push((fa, ca + cb));
            i += 1;
            j += 1;
        } else if fa < fb {
            out.push((fa, ca));
            i += 1;
        } else {
            out.push((fb, cb));
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

#[inline]
fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    // Both `a` and `b` are strictly ascending candidate indices!
    let mut i = 0;
    let mut j = 0;
    let mut shared = 0usize;
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            shared += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    shared as f64 / (a.len() + b.len() - shared) as f64
}

fn cluster(
    whole: &[SortedMultiset],
    query_like: &[SortedMultiset],
    options: &RecommendOptions<'_>,
) -> Vec<Vec<usize>> {
    let n = whole.len();
    let mut raw_clusters: Vec<Vec<usize>> = Vec::new();
    let mut shared = Vec::new();
    let mut shared_ql = Vec::new();
    let mut next_shared = Vec::new();
    let mut next_ql = Vec::new();

    for seed in 0..n {
        let seed_size = query_like[seed].total;
        if seed_size == 0 || (whole[seed].total as f64 / seed_size as f64) <= options.min_expansion {
            continue;
        }
        let mut members = vec![seed];
        shared.clear();
        shared.extend_from_slice(&whole[seed].pairs);
        shared_ql.clear();
        shared_ql.extend_from_slice(&query_like[seed].pairs);
        let mut shared_ql_total = seed_size;
        let mut shared_total = whole[seed].total;
        let min_csq_f64 = options.min_cohesion * seed_size as f64;

        loop {
            let mut best_member: Option<usize> = None;
            let mut best_expansion = 0.0f64;
            let start = *members.last().unwrap() + 1;

            for member in start..n {
                // Fast upper-bound checks before scanning sorted slices
                if (shared_ql_total.min(query_like[member].total) as f64) <= min_csq_f64 {
                    continue;
                }
                let csq = intersection_size(&shared_ql, &query_like[member].pairs);
                if csq == 0 || (csq as f64 / seed_size as f64) <= options.min_cohesion {
                    continue;
                }
                let max_cs = shared_total.min(whole[member].total);
                let needed_exp = best_expansion.max(options.min_expansion);
                if (max_cs as f64 / csq as f64) <= needed_exp {
                    continue;
                }
                let cs = intersection_size(&shared, &whole[member].pairs);
                let expansion = cs as f64 / csq as f64;
                if expansion > options.min_expansion && expansion > best_expansion {
                    best_expansion = expansion;
                    best_member = Some(member);
                }
            }

            let Some(m) = best_member else { break };
            members.push(m);
            shared_total = intersect_multisets(&shared, &whole[m].pairs, &mut next_shared);
            std::mem::swap(&mut shared, &mut next_shared);
            shared_ql_total = intersect_multisets(&shared_ql, &query_like[m].pairs, &mut next_ql);
            std::mem::swap(&mut shared_ql, &mut next_ql);
        }
        raw_clusters.push(members);
    }

    // Dedup: keep `raw_clusters[pos]` iff jaccard with EVERY earlier cluster (including dropped ones) <= max_cluster_overlap
    let mut kept = Vec::new();
    for pos in 0..raw_clusters.len() {
        let cur = &raw_clusters[pos];
        let mut ok = true;
        for earlier in &raw_clusters[..pos] {
            if jaccard(earlier, cur) > options.max_cluster_overlap {
                ok = false;
                break;
            }
        }
        if ok {
            kept.push(cur.clone());
        }
    }
    kept
}

fn intersect_cluster(
    lidx: &LanguageIndex,
    members: &[usize],
    candidate_mids: &[u32],
    whole: &[SortedMultiset],
    query: &Query,
    target: &mut PruneTarget,
    scratch: &mut PruneScratch,
) -> Vec<usize> {
    let first_mid = candidate_mids[members[0]] as usize;
    let first_m = &lidx.methods[first_mid];
    let tok_count = first_m.primary.tok_count as usize;
    if members.len() == 1 {
        return (0..tok_count).collect();
    }
    let offsets = lidx.method_offsets(first_mid);
    let feat_base = first_m.feat_start as usize;
    let mut kept: Option<Vec<usize>> = None;
    let mut sum_pairs: Vec<(u32, u32)> = Vec::new();

    for &m_idx in &members[1..] {
        add_multisets(&whole[m_idx].pairs, &query.features, &mut sum_pairs);
        target.build_from_pairs(&sum_pairs);
        let p = prune_fast(
            target,
            &lidx.all_features,
            offsets,
            feat_base,
            None,
            kept.as_deref(),
            scratch,
        );
        kept = Some(p.tokens);
    }
    kept.unwrap()
}

/// Zero-reparse rendering using precomputed `StructSpan`s + `memchr` line extraction.
fn render_recommendation(
    index: &AromaIndex,
    lidx: &LanguageIndex,
    loc: &MethodLocation,
    kept: &[usize],
    query_like_tokens: &[usize],
) -> (Vec<RecommendedLine>, String) {
    let mut roles: FxHashMap<u32, LineRole> = FxHashMap::default();

    // Build fast membership bitset/sorted check for `query_like_tokens` (both `kept` and `query_like_tokens` are sorted ascending!)
    let mut ql_i = 0usize;
    for &tok in kept {
        while ql_i < query_like_tokens.len() && query_like_tokens[ql_i] < tok {
            ql_i += 1;
        }
        let is_ql = ql_i < query_like_tokens.len() && query_like_tokens[ql_i] == tok;
        let new_role = if is_ql { LineRole::Query } else { LineRole::Added };
        let (s_ln, e_ln) = lidx.tok_line_span(loc, tok);
        for ln in s_ln..=e_ln {
            match roles.get(&ln) {
                Some(LineRole::Added) => {}
                _ => {
                    roles.insert(ln, new_role);
                }
            }
        }
    }

    // Mark structure lines from precomputed `StructSpan`s:
    // A span `(tok_start, tok_end, line)` triggers iff some `t in kept` satisfies `tok_start <= t < tok_end`.
    if !kept.is_empty() {
        for sp in lidx.location_spans(loc) {
            if roles.contains_key(&sp.line) {
                continue;
            }
            let pos = kept.partition_point(|&t| (t as u32) < sp.tok_start);
            if pos < kept.len() && (kept[pos] as u32) < sp.tok_end {
                roles.insert(sp.line, LineRole::Structure);
            }
        }
    }

    let mut line_roles: Vec<(u32, LineRole)> = roles.into_iter().collect();
    line_roles.sort_unstable_by_key(|&(ln, _)| ln);

    if line_roles.is_empty() {
        return (Vec::new(), String::new());
    }

    // Extract only the needed lines from `source` using `memchr`
    let source = &index.sources[loc.file_id as usize].source;
    let bytes = source.as_bytes();
    let max_line = line_roles.last().unwrap().0 as usize;

    let mut lines_out: Vec<RecommendedLine> = Vec::with_capacity(line_roles.len());
    let mut lr_idx = 0usize;
    let mut cur_line = 1usize;
    let mut line_start = 0usize;

    for nl in memchr::memchr_iter(b'\n', bytes) {
        if cur_line == line_roles[lr_idx].0 as usize {
            let text = js_trim_end(&source[line_start..nl]).to_string();
            lines_out.push(RecommendedLine { line: cur_line as u32, text, role: line_roles[lr_idx].1 });
            lr_idx += 1;
            if lr_idx >= line_roles.len() {
                break;
            }
        }
        cur_line += 1;
        line_start = nl + 1;
        if cur_line > max_line {
            break;
        }
    }
    // Last line (no trailing `\n`) or out-of-bounds fallback
    while lr_idx < line_roles.len() {
        let target_ln = line_roles[lr_idx].0 as usize;
        let text = if target_ln == cur_line && line_start <= source.len() {
            js_trim_end(&source[line_start..]).to_string()
        } else {
            String::new()
        };
        lines_out.push(RecommendedLine { line: target_ln as u32, text, role: line_roles[lr_idx].1 });
        lr_idx += 1;
    }

    // Dedent for `code`
    let common = lines_out
        .iter()
        .filter(|l| !js_trim(&l.text).is_empty())
        .map(|l| js_indent(&l.text))
        .min()
        .unwrap_or(0);
    let mut code = String::new();
    for (i, l) in lines_out.iter().enumerate() {
        if i > 0 {
            code.push('\n');
        }
        code.push_str(drop_chars(&l.text, common));
    }

    (lines_out, code)
}

pub fn recommend(
    index: &AromaIndex,
    query: &Query,
    options: &RecommendOptions<'_>,
) -> Vec<Recommendation> {
    if options.limit == 0
        || options.cluster_limit == 0
        || options.search_limit == 0
        || query.size == 0
        || query.features.is_empty()
    {
        return Vec::new();
    }
    let Some(lidx) = index.lang_index_by_id(&query.language) else {
        return Vec::new();
    };
    let qsize = query.size as f64;

    TL_SEARCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        let candidates = rank_for_recommend(
            index,
            lidx,
            query,
            options.search_limit,
            options.cluster_limit,
            options.min_similarity,
            options.exclude,
            sc,
        );
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut tmp_sort: Vec<u32> = Vec::new();
        let mut whole: Vec<SortedMultiset> = Vec::with_capacity(candidates.len());
        let mut query_like: Vec<SortedMultiset> = Vec::with_capacity(candidates.len());
        let mut cand_mids: Vec<u32> = Vec::with_capacity(candidates.len());

        for c in &candidates {
            let mid = c.mid as usize;
            cand_mids.push(c.mid);
            whole.push(build_multiset_from_slice(lidx.method_features(mid), &mut tmp_sort));
            let m = &lidx.methods[mid];
            let offsets = lidx.method_offsets(mid);
            query_like.push(build_multiset_from_tokens(
                &lidx.all_features,
                offsets,
                m.feat_start as usize,
                &c.pruned.tokens,
                &mut tmp_sort,
            ));
        }

        let clusters = cluster(&whole, &query_like, options);
        let mut out = Vec::new();

        for members in clusters.into_iter().filter(|m| m.len() >= options.min_support).take(options.limit) {
            let first = &candidates[members[0]];
            let kept = intersect_cluster(
                lidx,
                &members,
                &cand_mids,
                &whole,
                query,
                &mut sc.prune_target,
                &mut sc.prune_scratch,
            );
            let (lines, code) =
                render_recommendation(index, lidx, first.loc, &kept, &first.pruned.tokens);
            let member_refs = members
                .iter()
                .map(|&idx| index.make_ref(lidx, candidates[idx].loc))
                .collect();
            out.push(Recommendation {
                method: index.make_ref(lidx, first.loc),
                members: member_refs,
                similarity: first.pruned.score as f64 / qsize,
                lines,
                code,
            });
        }
        out
    })
}
