//! Corpus indexing (`build_index`), 32-shard 64-bit feature interning,
//! per-language CSR inverted index with packed `(count << 24) | mid` postings,
//! and query construction (`query_from_snippet`, `queries_from_lines`).

use std::cell::RefCell;

use hashbrown::HashTable;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::features::{extract_location_and_spans, featurize_body_hashed, FeatScratch, StructSpan};
use crate::lang_py::{self, FastMethod};
use crate::lang_ts;
use crate::lezer;
use crate::simple::Body;
use crate::text::utf16_len;
use crate::{builtin_languages, Language, MethodRef, SnippetOptions, SourceFile};

pub(crate) const SHARDS: usize = 32;
const SHARD_MASK: usize = SHARDS - 1;
pub(crate) const COUNT_SHIFT: u32 = 24;
pub(crate) const MID_MASK: u32 = (1 << COUNT_SHIFT) - 1;
pub(crate) const COUNT_OVERFLOW: u32 = 255;

#[inline(always)]
fn shard_of(h: u64) -> usize {
    (h as usize) & SHARD_MASK
}

#[inline(always)]
fn shard_hash(h: u64) -> u64 {
    h.rotate_right(5)
}

pub(crate) struct ShardedFeatureTable {
    shards: Vec<HashTable<(u64, u32)>>,
}

impl ShardedFeatureTable {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(HashTable::new());
        }
        Self { shards }
    }

    #[inline]
    fn get_or_insert(&mut self, h: u64, next_id: &mut u32) -> u32 {
        let s = shard_of(h);
        let sh = shard_hash(h);
        let table = unsafe { self.shards.get_unchecked_mut(s) };
        if let Some(&(_, id)) = table.find(sh, |e| e.0 == h) {
            return id;
        }
        let id = *next_id;
        *next_id += 1;
        table.insert_unique(sh, (h, id), |e| shard_hash(e.0));
        id
    }

    #[inline]
    pub fn get(&self, h: u64) -> Option<u32> {
        let s = shard_of(h);
        let sh = shard_hash(h);
        unsafe { self.shards.get_unchecked(s) }
            .find(sh, |e| e.0 == h)
            .map(|&(_, id)| id)
    }
}

pub struct IndexOptions {
    pub languages: Option<Vec<Box<dyn Language>>>,
    pub max_file_length: usize,
    pub threads: Option<usize>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            languages: None,
            max_file_length: 1_000_000,
            threads: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IndexStats {
    pub files: usize,
    pub methods: usize,
    pub duplicate_methods: usize,
    pub features: usize,
    pub unsupported_files: usize,
    pub oversized_files: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct MethodLocation {
    pub file_id: u32,
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    /// Slice in `LanguageIndex::all_token_lines` / `all_token_end_lines`
    pub tok_line_start: u32,
    pub tok_count: u32,
    /// Slice in `LanguageIndex::all_struct_spans`
    pub span_start: u32,
    pub span_end: u32,
}

pub(crate) struct IndexedMethod {
    pub primary: MethodLocation,
    pub copies: Vec<MethodLocation>,
    /// Start index in `LanguageIndex::all_features`
    pub feat_start: u32,
    /// Start index in `LanguageIndex::all_offsets` (`tok_count + 1` entries, relative to `feat_start`)
    pub off_start: u32,
}

pub(crate) struct LanguageIndex {
    pub lang_id: String,
    pub methods: Vec<IndexedMethod>,
    /// Flat file_id of each canonical method (`primary.file_id`), enabling L1/L2 origin
    /// filtering in Phase I without dereferencing `IndexedMethod`.
    pub canonical_file_id: Vec<u32>,
    /// Flat CSR feature IDs for all canonical methods of this language
    pub all_features: Vec<u32>,
    /// Flat CSR per-token feature offsets (each method has `tok_count + 1` entries, relative to `feat_start`)
    pub all_offsets: Vec<u32>,
    /// Flat token start lines for all canonical + duplicate locations
    pub all_token_lines: Vec<u32>,
    /// 1-byte `end_line - line` delta per token (`255` means consult `token_end_overflow`)
    pub all_token_end_delta: Vec<u8>,
    pub token_end_overflow: FxHashMap<u32, u32>,
    /// Flat precomputed structural spans for zero-reparse rendering
    pub all_struct_spans: Vec<StructSpan>,
    /// Inverted index indexed by global `feature_id` (`0..=num_features`)
    pub posting_offsets: Vec<u32>,
    /// Packed `(count << 24) | local_mid` sorted ascending by `local_mid` within each feature
    pub postings: Vec<u32>,
    /// Exact counts for rare `(global_fid, local_mid)` where count >= 255
    pub posting_overflow: FxHashMap<(u32, u32), u32>,
}

impl LanguageIndex {
    fn new(lang_id: String) -> Self {
        Self {
            lang_id,
            methods: Vec::new(),
            canonical_file_id: Vec::new(),
            all_features: Vec::new(),
            all_offsets: Vec::new(),
            all_token_lines: Vec::new(),
            all_token_end_delta: Vec::new(),
            token_end_overflow: FxHashMap::default(),
            all_struct_spans: Vec::new(),
            posting_offsets: Vec::new(),
            postings: Vec::new(),
            posting_overflow: FxHashMap::default(),
        }
    }

    #[inline]
    pub fn method_features(&self, mid: usize) -> &[u32] {
        let m = &self.methods[mid];
        let off_end = m.off_start as usize + m.primary.tok_count as usize;
        let len = self.all_offsets[off_end] as usize;
        &self.all_features[m.feat_start as usize..m.feat_start as usize + len]
    }

    #[inline]
    pub fn method_offsets(&self, mid: usize) -> &[u32] {
        let m = &self.methods[mid];
        let s = m.off_start as usize;
        &self.all_offsets[s..s + m.primary.tok_count as usize + 1]
    }

    #[inline(always)]
    pub fn tok_line_span(&self, loc: &MethodLocation, tok: usize) -> (u32, u32) {
        let at = loc.tok_line_start as usize + tok;
        let line = unsafe { *self.all_token_lines.get_unchecked(at) };
        let d = unsafe { *self.all_token_end_delta.get_unchecked(at) };
        let end = if d < 255 {
            line + d as u32
        } else {
            line + self.token_end_overflow[&(at as u32)]
        };
        (line, end)
    }

    #[inline]
    pub fn location_spans(&self, loc: &MethodLocation) -> &[StructSpan] {
        &self.all_struct_spans[loc.span_start as usize..loc.span_end as usize]
    }
}

pub(crate) struct SourceEntry {
    pub path: String,
    pub source: String,
}

pub struct AromaIndex {
    pub(crate) languages: Vec<Box<dyn Language>>,
    pub(crate) lang_indices: Vec<LanguageIndex>,
    pub(crate) global_order: Vec<(u16, u32)>,
    pub(crate) stats: IndexStats,
    pub(crate) feature_table: ShardedFeatureTable,
    pub(crate) sources: Vec<SourceEntry>,
    pub(crate) path_to_id: FxHashMap<String, u32>,
    /// Reusable worker pool when built with `threads > 1` (`None` for `--threads 1`).
    pub(crate) pool: Option<rayon::ThreadPool>,
}

impl AromaIndex {
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    pub fn method_refs(&self) -> Vec<MethodRef> {
        let mut out = Vec::with_capacity(self.global_order.len());
        for &(li, mid) in &self.global_order {
            let lidx = &self.lang_indices[li as usize];
            let loc = &lidx.methods[mid as usize].primary;
            out.push(self.make_ref(lidx, loc));
        }
        out
    }

    pub fn method_count(&self) -> usize {
        self.global_order.len()
    }

    #[inline]
    pub(crate) fn make_ref(&self, lidx: &LanguageIndex, loc: &MethodLocation) -> MethodRef {
        MethodRef {
            path: self.sources[loc.file_id as usize].path.clone(),
            name: loc.name.clone(),
            language: lidx.lang_id.clone(),
            start_line: loc.start_line,
            end_line: loc.end_line,
        }
    }

    #[inline]
    pub(crate) fn lang_index_by_id(&self, id: &str) -> Option<&LanguageIndex> {
        self.lang_indices.iter().find(|li| li.lang_id == id)
    }
}

#[derive(Debug, Clone)]
pub struct Query {
    pub(crate) language: String,
    /// Sorted by `feature_id` ascending: `(feature_id, count)` for features present in index.
    pub(crate) features: Vec<(u32, u32)>,
    pub(crate) size: usize,
    pub origin: Option<MethodRef>,
    pub lines: Option<Vec<u32>>,
}

impl Query {
    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn origin(&self) -> Option<&MethodRef> {
        self.origin.as_ref()
    }

    pub fn lines(&self) -> Option<&[u32]> {
        self.lines.as_deref()
    }

    pub fn without_origin(&self) -> Query {
        let mut q = self.clone();
        q.origin = None;
        q
    }

    pub fn with_language(&self, language: &str) -> Query {
        let mut q = self.clone();
        q.language = language.to_string();
        q
    }
}

pub(crate) fn language_index_for_path(languages: &[Box<dyn Language>], path: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_len = 0usize;
    for (idx, lang) in languages.iter().enumerate() {
        for &ext in lang.extensions() {
            if path.ends_with(ext) && ext.len() > best_len {
                best = Some(idx);
                best_len = ext.len();
            }
        }
    }
    best
}

// ============================================================================
// Reusable workspace for zero-alloc parsing & featurization
// ============================================================================

#[derive(Default)]
struct ParseScratch {
    body: Body,
    methods: Vec<FastMethod>,
    feat: FeatScratch,
    hashes: Vec<u64>,
    offsets: Vec<u32>,
    tok_lines: Vec<u32>,
    tok_end_delta: Vec<u8>,
    end_overflow: Vec<(u32, u32)>,
}

thread_local! {
    static TL_SCRATCH: RefCell<ParseScratch> = RefCell::new(ParseScratch::default());
}

fn parse_file_into_body(
    lang: &dyn Language,
    file: &SourceFile,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    body.clear();
    out.clear();
    match lang.builtin_kind() {
        1 => lang_py::parse_methods_into(&file.source, body, out),
        2 => lang_ts::parse_methods_into(&file.source, &file.path, body, out),
        _ => {
            for m in lang.parse_methods(&file.source, &file.path) {
                let root = body.push_node(&m.body);
                out.push(FastMethod {
                    name: m.name,
                    start_line: m.start_line,
                    end_line: m.end_line,
                    body_root: root,
                });
            }
        }
    }
}

// ============================================================================
// `build_index` — Single-thread streaming + Multi-thread parallel sharded
// ============================================================================

pub fn build_index(files: &[SourceFile], options: IndexOptions) -> AromaIndex {
    let languages = options.languages.unwrap_or_else(builtin_languages);
    let threads = options
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);

    if threads <= 1 || files.len() < 8 {
        build_index_single(files, languages, options.max_file_length)
    } else {
        build_index_parallel(files, languages, options.max_file_length, threads)
    }
}

fn intern_source(
    path_to_id: &mut FxHashMap<String, u32>,
    sources: &mut Vec<SourceEntry>,
    file: &SourceFile,
) -> u32 {
    if let Some(&id) = path_to_id.get(&file.path) {
        sources[id as usize].source = file.source.clone();
        id
    } else {
        let id = sources.len() as u32;
        path_to_id.insert(file.path.clone(), id);
        sources.push(SourceEntry { path: file.path.clone(), source: file.source.clone() });
        id
    }
}

fn build_inverted_single(lidx: &mut LanguageIndex, feature_count: usize) {
    let mcount = lidx.methods.len();
    if mcount == 0 || feature_count == 0 {
        lidx.posting_offsets = vec![0u32; feature_count + 1];
        return;
    }
    let mut last_method = vec![u32::MAX; feature_count];
    let mut posting_offsets = vec![0u32; feature_count + 1];

    for mid in 0..mcount {
        let feats = lidx.method_features(mid);
        let mid_u32 = mid as u32;
        for &f in feats {
            let fi = f as usize;
            if unsafe { *last_method.get_unchecked(fi) } != mid_u32 {
                unsafe {
                    *last_method.get_unchecked_mut(fi) = mid_u32;
                    *posting_offsets.get_unchecked_mut(fi + 1) += 1;
                }
            }
        }
    }
    for f in 0..feature_count {
        posting_offsets[f + 1] += posting_offsets[f];
    }
    let total_postings = posting_offsets[feature_count] as usize;
    let mut postings = vec![0u32; total_postings];
    let mut cursor = posting_offsets[..feature_count].to_vec();
    last_method.fill(u32::MAX);

    let mut overflow = FxHashMap::default();

    for mid in 0..mcount {
        let feats = lidx.method_features(mid);
        let mid_u32 = mid as u32;
        for &f in feats {
            let fi = f as usize;
            if unsafe { *last_method.get_unchecked(fi) } != mid_u32 {
                unsafe {
                    *last_method.get_unchecked_mut(fi) = mid_u32;
                    let pos = *cursor.get_unchecked(fi) as usize;
                    *cursor.get_unchecked_mut(fi) = (pos + 1) as u32;
                    *postings.get_unchecked_mut(pos) = (1u32 << COUNT_SHIFT) | mid_u32;
                }
            } else {
                let pos = (unsafe { *cursor.get_unchecked(fi) } - 1) as usize;
                let cur = unsafe { *postings.get_unchecked(pos) } >> COUNT_SHIFT;
                if cur < COUNT_OVERFLOW - 1 {
                    unsafe { *postings.get_unchecked_mut(pos) += 1u32 << COUNT_SHIFT };
                } else if cur == COUNT_OVERFLOW - 1 {
                    unsafe { *postings.get_unchecked_mut(pos) = (COUNT_OVERFLOW << COUNT_SHIFT) | mid_u32 };
                    overflow.insert((f, mid_u32), COUNT_OVERFLOW);
                } else {
                    *overflow.get_mut(&(f, mid_u32)).unwrap() += 1;
                }
            }
        }
    }

    lidx.posting_offsets = posting_offsets;
    lidx.postings = postings;
    lidx.posting_overflow = overflow;
}

fn build_index_single(
    files: &[SourceFile],
    languages: Vec<Box<dyn Language>>,
    max_file_length: usize,
) -> AromaIndex {
    let mut lang_indices: Vec<LanguageIndex> =
        languages.iter().map(|l| LanguageIndex::new(l.id().to_string())).collect();
    if files.len() >= 16 {
        let mut bytes_by_lang = vec![0usize; languages.len()];
        for f in files {
            if let Some(li) = language_index_for_path(&languages, &f.path)
                && f.source.len() <= max_file_length
            {
                bytes_by_lang[li] += f.source.len();
            }
        }
        for (li, &b) in bytes_by_lang.iter().enumerate() {
            if b >= 65536 {
                lang_indices[li].all_features.reserve(b / 3);
                lang_indices[li].all_offsets.reserve(b / 10);
                lang_indices[li].all_token_lines.reserve(b / 10);
                lang_indices[li].all_token_end_delta.reserve(b / 10);
                lang_indices[li].all_struct_spans.reserve(b / 20);
            }
        }
    }
    let mut global_order: Vec<(u16, u32)> = Vec::new();
    let mut stats = IndexStats::default();
    let mut feature_table = ShardedFeatureTable::new();
    let mut next_fid = 0u32;
    let mut sources: Vec<SourceEntry> = Vec::new();
    let mut path_to_id: FxHashMap<String, u32> = FxHashMap::default();
    let mut seen: FxHashMap<(u16, u128), u32> = FxHashMap::default();

    let mut scratch = ParseScratch::default();

    for file in files {
        let Some(li) = language_index_for_path(&languages, &file.path) else {
            stats.unsupported_files += 1;
            continue;
        };
        if utf16_len(&file.source) > max_file_length {
            stats.oversized_files += 1;
            continue;
        }
        stats.files += 1;
        let lang = &*languages[li];
        parse_file_into_body(lang, file, &mut scratch.body, &mut scratch.methods);
        if scratch.methods.is_empty() {
            continue;
        }
        let base = file.source.as_bytes();
        let ma_labels = lang.member_access_labels();
        let mut file_id_opt: Option<u32> = None;

        for m in &scratch.methods {
            let digest = scratch.body.digest(base, m.body_root);
            let key = (li as u16, digest);
            let lidx = &mut lang_indices[li];

            if let Some(&earlier_mid) = seen.get(&key) {
                let tok_line_start = lidx.all_token_lines.len() as u32;
                let span_start = lidx.all_struct_spans.len() as u32;
                scratch.end_overflow.clear();
                let tok_count = extract_location_and_spans(
                    &scratch.body,
                    m.body_root,
                    &mut lidx.all_token_lines,
                    &mut lidx.all_token_end_delta,
                    &mut scratch.end_overflow,
                    &mut lidx.all_struct_spans,
                    &mut scratch.feat,
                );
                if tok_count == 0 {
                    continue;
                }
                for &(rel_tok, d) in &scratch.end_overflow {
                    lidx.token_end_overflow.insert(tok_line_start + rel_tok, d);
                }
                let span_end = lidx.all_struct_spans.len() as u32;
                let file_id = *file_id_opt
                    .get_or_insert_with(|| intern_source(&mut path_to_id, &mut sources, file));
                lidx.methods[earlier_mid as usize].copies.push(MethodLocation {
                    file_id,
                    name: m.name.clone(),
                    start_line: m.start_line,
                    end_line: m.end_line,
                    tok_line_start,
                    tok_count: tok_count as u32,
                    span_start,
                    span_end,
                });
                stats.duplicate_methods += 1;
                continue;
            }

            scratch.hashes.clear();
            scratch.offsets.clear();
            scratch.end_overflow.clear();
            let tok_line_start = lidx.all_token_lines.len() as u32;
            let span_start = lidx.all_struct_spans.len() as u32;
            let tok_count = featurize_body_hashed(
                &scratch.body,
                base,
                m.body_root,
                ma_labels,
                None,
                &mut scratch.feat,
                &mut scratch.hashes,
                &mut scratch.offsets,
                &mut lidx.all_token_lines,
                &mut lidx.all_token_end_delta,
                &mut scratch.end_overflow,
                Some(&mut lidx.all_struct_spans),
            );
            if tok_count == 0 {
                lidx.all_struct_spans.truncate(span_start as usize);
                continue;
            }
            for &(rel_tok, d) in &scratch.end_overflow {
                lidx.token_end_overflow.insert(tok_line_start + rel_tok, d);
            }
            let span_end = lidx.all_struct_spans.len() as u32;
            let file_id = *file_id_opt
                .get_or_insert_with(|| intern_source(&mut path_to_id, &mut sources, file));

            let feat_start = lidx.all_features.len() as u32;
            let off_start = lidx.all_offsets.len() as u32;
            lidx.all_offsets.extend_from_slice(&scratch.offsets);
            lidx.all_features.reserve(scratch.hashes.len());
            for &h in &scratch.hashes {
                let fid = feature_table.get_or_insert(h, &mut next_fid);
                lidx.all_features.push(fid);
            }

            let mid = lidx.methods.len() as u32;
            lidx.canonical_file_id.push(file_id);
            lidx.methods.push(IndexedMethod {
                primary: MethodLocation {
                    file_id,
                    name: m.name.clone(),
                    start_line: m.start_line,
                    end_line: m.end_line,
                    tok_line_start,
                    tok_count: tok_count as u32,
                    span_start,
                    span_end,
                },
                copies: Vec::new(),
                feat_start,
                off_start,
            });
            seen.insert(key, mid);
            global_order.push((li as u16, mid));
        }
    }

    drop(scratch);
    drop(seen);
    lezer::shrink_tl_workspace();
    lang_py::shrink_tl_tree();
    lang_ts::shrink_tl_tree();

    let feature_count = next_fid as usize;
    stats.methods = global_order.len();
    stats.features = feature_count;

    for lidx in &mut lang_indices {
        build_inverted_single(lidx, feature_count);
    }

    AromaIndex {
        languages,
        lang_indices,
        global_order,
        stats,
        feature_table,
        sources,
        path_to_id,
        pool: None,
    }
}

// ============================================================================
// Multi-threaded parallel indexing:
//   Stage 1: Byte-balanced chunks, parallel parse + intra-chunk dedup + local interning
//            followed by `pool.broadcast` release of worker parse workspaces!
//   Stage 2: Deterministic sequential dedup & CSR metadata assembly
//   Stage 3a: Parallel 32-shard interning of ONLY unique chunk-local 64-bit hashes
//   Stage 3b: Parallel zero-unsafe remap of local_fids -> LanguageIndex::all_features
//   Stage 3c: Parallel 16-bucket inverted index build with pre-sized buckets
// ============================================================================

struct ChunkMethod {
    file_idx: usize,
    lang_idx: u16,
    name: String,
    start_line: u32,
    end_line: u32,
    digest: u128,
    tok_count: u32,
    feat_len: u32,
    span_len: u32,
    overflow_len: u32,
}

struct ChunkBatch {
    methods: Vec<ChunkMethod>,
    local_hashes: Vec<u64>,
    local_fids: Vec<u32>,
    offsets: Vec<u32>,
    tok_lines: Vec<u32>,
    tok_end_delta: Vec<u8>,
    end_overflow: Vec<(u32, u32)>,
    spans: Vec<StructSpan>,
}

#[derive(Clone, Copy)]
struct CanonCopy {
    chunk_id: u32,
    src_start: u32,
    dst_start: u32,
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HashBucketEntry {
    hash: u64,
    chunk_id: u32,
    local_fid: u32,
}

type BucketPostings = (Vec<u32>, Vec<u32>, Vec<(u32, u32, u32)>);
type ShardInternResult = (HashTable<(u64, u32)>, u32, Vec<(u32, u32, u32)>);

fn build_inverted_parallel(lidx: &mut LanguageIndex, feature_count: usize, threads: usize) {
    let mcount = lidx.methods.len();
    if mcount == 0 || feature_count == 0 {
        lidx.posting_offsets = vec![0u32; feature_count + 1];
        return;
    }
    if mcount < 512 || feature_count < 4096 {
        build_inverted_single(lidx, feature_count);
        return;
    }

    let target_buckets = threads.next_power_of_two().clamp(8, 32);
    let bucket_width = feature_count.div_ceil(target_buckets).next_power_of_two().max(1);
    let shift = bucket_width.trailing_zeros();
    let num_buckets = feature_count.div_ceil(bucket_width);

    let num_blocks = threads.min(mcount).max(1);
    let block_size = mcount.div_ceil(num_blocks);
    let mut blocks: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);
    let mut s = 0;
    while s < mcount {
        let e = (s + block_size).min(mcount);
        blocks.push((s, e));
        s = e;
    }

    let est_per_bucket = (lidx.all_features.len() / (blocks.len() * num_buckets).max(1)).max(64);
    let use_packed_u32 = mcount <= u16::MAX as usize && bucket_width <= 65536;

    let bucket_results: Vec<BucketPostings> = if use_packed_u32 {
        let lf_mask = bucket_width - 1;
        let block_buckets: Vec<Vec<Vec<u32>>> = blocks
            .par_iter()
            .map(|&(m_start, m_end)| {
                let mut bks: Vec<Vec<u32>> =
                    (0..num_buckets).map(|_| Vec::with_capacity(est_per_bucket)).collect();
                for mid in m_start..m_end {
                    let mid_hi = (mid as u32) << 16;
                    for &fid in lidx.method_features(mid) {
                        let b = (fid as usize) >> shift;
                        let lf = (fid as usize) & lf_mask;
                        unsafe { bks.get_unchecked_mut(b) }.push(mid_hi | (lf as u32));
                    }
                }
                bks
            })
            .collect();

        let res = (0..num_buckets)
            .into_par_iter()
            .map(|b| {
                let f_base = b * bucket_width;
                let f_end = (f_base + bucket_width).min(feature_count);
                let width = f_end - f_base;

                let mut counts = vec![0u32; width];
                let mut last_mid = vec![u32::MAX; width];
                for blk in &block_buckets {
                    for &entry in &blk[b] {
                        let mid = entry >> 16;
                        let lf = (entry & 0xFFFF) as usize;
                        if unsafe { *last_mid.get_unchecked(lf) } != mid {
                            unsafe {
                                *last_mid.get_unchecked_mut(lf) = mid;
                                *counts.get_unchecked_mut(lf) += 1;
                            }
                        }
                    }
                }

                let mut cursors = vec![0u32; width];
                let mut total = 0u32;
                for i in 0..width {
                    cursors[i] = total;
                    total += counts[i];
                }
                let mut postings = vec![0u32; total as usize];
                let mut overflow: Vec<(u32, u32, u32)> = Vec::new();
                let mut ov_map: FxHashMap<(u32, u32), usize> = FxHashMap::default();
                last_mid.fill(u32::MAX);

                for blk in &block_buckets {
                    for &entry in &blk[b] {
                        let mid = entry >> 16;
                        let lf = (entry & 0xFFFF) as usize;
                        if unsafe { *last_mid.get_unchecked(lf) } != mid {
                            unsafe {
                                *last_mid.get_unchecked_mut(lf) = mid;
                                let pos = *cursors.get_unchecked(lf) as usize;
                                *cursors.get_unchecked_mut(lf) = (pos + 1) as u32;
                                *postings.get_unchecked_mut(pos) = (1u32 << COUNT_SHIFT) | mid;
                            }
                        } else {
                            let pos = (unsafe { *cursors.get_unchecked(lf) } - 1) as usize;
                            let cur = unsafe { *postings.get_unchecked(pos) } >> COUNT_SHIFT;
                            if cur < COUNT_OVERFLOW - 1 {
                                unsafe { *postings.get_unchecked_mut(pos) += 1u32 << COUNT_SHIFT };
                            } else if cur == COUNT_OVERFLOW - 1 {
                                let fid = (f_base + lf) as u32;
                                unsafe { *postings.get_unchecked_mut(pos) = (COUNT_OVERFLOW << COUNT_SHIFT) | mid };
                                let idx = overflow.len();
                                overflow.push((fid, mid, COUNT_OVERFLOW));
                                ov_map.insert((fid, mid), idx);
                            } else {
                                let fid = (f_base + lf) as u32;
                                let idx = ov_map[&(fid, mid)];
                                overflow[idx].2 += 1;
                            }
                        }
                    }
                }
                (counts, postings, overflow)
            })
            .collect();
        block_buckets.into_par_iter().for_each(drop);
        res
    } else {
        let block_buckets: Vec<Vec<Vec<(u32, u32)>>> = blocks
            .par_iter()
            .map(|&(m_start, m_end)| {
                let mut bks: Vec<Vec<(u32, u32)>> =
                    (0..num_buckets).map(|_| Vec::with_capacity(est_per_bucket)).collect();
                for mid in m_start..m_end {
                    let mid_u32 = mid as u32;
                    for &fid in lidx.method_features(mid) {
                        let b = (fid as usize) >> shift;
                        unsafe { bks.get_unchecked_mut(b) }.push((mid_u32, fid));
                    }
                }
                bks
            })
            .collect();

        let res = (0..num_buckets)
            .into_par_iter()
            .map(|b| {
                let f_base = b * bucket_width;
                let f_end = (f_base + bucket_width).min(feature_count);
                let width = f_end - f_base;

                let mut counts = vec![0u32; width];
                let mut last_mid = vec![u32::MAX; width];
                for blk in &block_buckets {
                    for &(mid, fid) in &blk[b] {
                        let lf = fid as usize - f_base;
                        if unsafe { *last_mid.get_unchecked(lf) } != mid {
                            unsafe {
                                *last_mid.get_unchecked_mut(lf) = mid;
                                *counts.get_unchecked_mut(lf) += 1;
                            }
                        }
                    }
                }

                let mut cursors = vec![0u32; width];
                let mut total = 0u32;
                for i in 0..width {
                    cursors[i] = total;
                    total += counts[i];
                }
                let mut postings = vec![0u32; total as usize];
                let mut overflow: Vec<(u32, u32, u32)> = Vec::new();
                let mut ov_map: FxHashMap<(u32, u32), usize> = FxHashMap::default();
                last_mid.fill(u32::MAX);

                for blk in &block_buckets {
                    for &(mid, fid) in &blk[b] {
                        let lf = fid as usize - f_base;
                        if unsafe { *last_mid.get_unchecked(lf) } != mid {
                            unsafe {
                                *last_mid.get_unchecked_mut(lf) = mid;
                                let pos = *cursors.get_unchecked(lf) as usize;
                                *cursors.get_unchecked_mut(lf) = (pos + 1) as u32;
                                *postings.get_unchecked_mut(pos) = (1u32 << COUNT_SHIFT) | mid;
                            }
                        } else {
                            let pos = (unsafe { *cursors.get_unchecked(lf) } - 1) as usize;
                            let cur = unsafe { *postings.get_unchecked(pos) } >> COUNT_SHIFT;
                            if cur < COUNT_OVERFLOW - 1 {
                                unsafe { *postings.get_unchecked_mut(pos) += 1u32 << COUNT_SHIFT };
                            } else if cur == COUNT_OVERFLOW - 1 {
                                unsafe { *postings.get_unchecked_mut(pos) = (COUNT_OVERFLOW << COUNT_SHIFT) | mid };
                                let idx = overflow.len();
                                overflow.push((fid, mid, COUNT_OVERFLOW));
                                ov_map.insert((fid, mid), idx);
                            } else {
                                let idx = ov_map[&(fid, mid)];
                                overflow[idx].2 += 1;
                            }
                        }
                    }
                }
                (counts, postings, overflow)
            })
            .collect();
        block_buckets.into_par_iter().for_each(drop);
        res
    };

    let mut posting_offsets = vec![0u32; feature_count + 1];
    let mut total_p = 0usize;
    for (b, (counts, postings, overflow)) in bucket_results.iter().enumerate() {
        let f_base = b * bucket_width;
        for (i, &c) in counts.iter().enumerate() {
            posting_offsets[f_base + i + 1] = c;
        }
        total_p += postings.len();
        for &(fid, mid, exact) in overflow {
            lidx.posting_overflow.insert((fid, mid), exact);
        }
    }
    for f in 0..feature_count {
        posting_offsets[f + 1] += posting_offsets[f];
    }
    let mut all_postings = vec![0u32; total_p];
    {
        let mut slices: Vec<(&[u32], &mut [u32])> = Vec::with_capacity(bucket_results.len());
        let mut rest: &mut [u32] = &mut all_postings;
        for (_, postings, _) in &bucket_results {
            let (head, tail) = rest.split_at_mut(postings.len());
            slices.push((postings.as_slice(), head));
            rest = tail;
        }
        slices.into_par_iter().for_each(|(src, dst)| {
            dst.copy_from_slice(src);
        });
    }
    bucket_results.into_par_iter().for_each(drop);
    lidx.posting_offsets = posting_offsets;
    lidx.postings = all_postings;
}

fn build_index_parallel(
    files: &[SourceFile],
    languages: Vec<Box<dyn Language>>,
    max_file_length: usize,
    threads: usize,
) -> AromaIndex {
    let mut stats = IndexStats::default();
    let mut work: Vec<(usize, u16)> = Vec::with_capacity(files.len());
    for (idx, file) in files.iter().enumerate() {
        let Some(li) = language_index_for_path(&languages, &file.path) else {
            stats.unsupported_files += 1;
            continue;
        };
        if utf16_len(&file.source) > max_file_length {
            stats.oversized_files += 1;
            continue;
        }
        stats.files += 1;
        work.push((idx, li as u16));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon thread pool");

    if work.is_empty() {
        let lang_indices = languages.iter().map(|l| LanguageIndex::new(l.id().to_string())).collect();
        return AromaIndex {
            languages,
            lang_indices,
            global_order: Vec::new(),
            stats,
            feature_table: ShardedFeatureTable::new(),
            sources: Vec::new(),
            path_to_id: FxHashMap::default(),
            pool: Some(pool),
        };
    }

    let target_chunks = (threads * 3).min(work.len()).max(1);
    let total_weight: usize = work.iter().map(|&(fi, _)| files[fi].source.len() + 512).sum();
    let target_w = (total_weight / target_chunks).max(1);
    let mut chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(target_chunks);
    let mut start = 0usize;
    while start < work.len() {
        let remaining_chunks = target_chunks.saturating_sub(chunk_ranges.len());
        if remaining_chunks <= 1 {
            chunk_ranges.push((start, work.len()));
            break;
        }
        let mut w = 0usize;
        let mut end = start;
        while end < work.len() {
            w += files[work[end].0].source.len() + 512;
            end += 1;
            if w >= target_w || work.len() - end < remaining_chunks - 1 {
                break;
            }
        }
        chunk_ranges.push((start, end));
        start = end;
    }

    let num_langs = languages.len();

    let (lang_indices, global_order, feature_table, sources, path_to_id) = pool.install(|| {
        // Stage 1: Parallel chunk parse + simplify + intra-chunk dedup + chunk-local interning
        let chunks: Vec<ChunkBatch> = chunk_ranges
            .par_iter()
            .map_init(ParseScratch::default, |sc, &(ws, we)| {
                let mut local_table: HashTable<(u64, u32)> = HashTable::new();
                let mut local_seen: FxHashSet<(u16, u128)> = FxHashSet::default();
                let mut batch = ChunkBatch {
                    methods: Vec::new(),
                    local_hashes: Vec::new(),
                    local_fids: Vec::new(),
                    offsets: Vec::new(),
                    tok_lines: Vec::new(),
                    tok_end_delta: Vec::new(),
                    end_overflow: Vec::new(),
                    spans: Vec::new(),
                };

                for &(file_idx, lang_idx) in &work[ws..we] {
                    let file = &files[file_idx];
                    let lang = &*languages[lang_idx as usize];
                    let base = file.source.as_bytes();
                    let ma_labels = lang.member_access_labels();
                    parse_file_into_body(lang, file, &mut sc.body, &mut sc.methods);

                    for m in &sc.methods {
                        let digest = sc.body.digest(base, m.body_root);
                        let key = (lang_idx, digest);
                        let s_start = batch.spans.len();
                        let ov_start = batch.end_overflow.len();

                        if local_seen.contains(&key) {
                            let tok_count = extract_location_and_spans(
                                &sc.body,
                                m.body_root,
                                &mut batch.tok_lines,
                                &mut batch.tok_end_delta,
                                &mut batch.end_overflow,
                                &mut batch.spans,
                                &mut sc.feat,
                            );
                            if tok_count == 0 {
                                batch.spans.truncate(s_start);
                                batch.end_overflow.truncate(ov_start);
                                continue;
                            }
                            batch.methods.push(ChunkMethod {
                                file_idx,
                                lang_idx,
                                name: m.name.clone(),
                                start_line: m.start_line,
                                end_line: m.end_line,
                                digest,
                                tok_count: tok_count as u32,
                                feat_len: 0,
                                span_len: (batch.spans.len() - s_start) as u32,
                                overflow_len: (batch.end_overflow.len() - ov_start) as u32,
                            });
                            continue;
                        }

                        sc.hashes.clear();
                        let o_start = batch.offsets.len();
                        let tok_count = featurize_body_hashed(
                            &sc.body,
                            base,
                            m.body_root,
                            ma_labels,
                            None,
                            &mut sc.feat,
                            &mut sc.hashes,
                            &mut batch.offsets,
                            &mut batch.tok_lines,
                            &mut batch.tok_end_delta,
                            &mut batch.end_overflow,
                            Some(&mut batch.spans),
                        );
                        if tok_count == 0 {
                            batch.offsets.truncate(o_start);
                            batch.spans.truncate(s_start);
                            batch.end_overflow.truncate(ov_start);
                            continue;
                        }
                        local_seen.insert(key);
                        batch.local_fids.reserve(sc.hashes.len());
                        for &h in &sc.hashes {
                            let lfid = if let Some(&(_, id)) = local_table.find(h, |e| e.0 == h) {
                                id
                            } else {
                                let id = batch.local_hashes.len() as u32;
                                batch.local_hashes.push(h);
                                local_table.insert_unique(h, (h, id), |e| e.0);
                                id
                            };
                            batch.local_fids.push(lfid);
                        }
                        batch.methods.push(ChunkMethod {
                            file_idx,
                            lang_idx,
                            name: m.name.clone(),
                            start_line: m.start_line,
                            end_line: m.end_line,
                            digest,
                            tok_count: tok_count as u32,
                            feat_len: sc.hashes.len() as u32,
                            span_len: (batch.spans.len() - s_start) as u32,
                            overflow_len: (batch.end_overflow.len() - ov_start) as u32,
                        });
                    }
                }

                batch
            })
            .collect();

        // Free all 15 workers' thread-local Lezer + SyntaxTree scratch BEFORE Stage 2/3 allocate!
        pool.broadcast(|_| {
            lezer::shrink_tl_workspace();
            lang_py::shrink_tl_tree();
            lang_ts::shrink_tl_tree();
        });

        // Stage 2: Exact pre-sizing + deterministic sequential deduplication & CSR assembly
        let num_chunks = chunks.len();
        let mut lang_indices: Vec<LanguageIndex> =
            languages.iter().map(|l| LanguageIndex::new(l.id().to_string())).collect();
        {
            let mut tok_totals = vec![0usize; num_langs];
            let mut span_totals = vec![0usize; num_langs];
            let mut off_totals = vec![0usize; num_langs];
            let mut meth_totals = vec![0usize; num_langs];
            for batch in &chunks {
                for m in &batch.methods {
                    let li = m.lang_idx as usize;
                    tok_totals[li] += m.tok_count as usize;
                    span_totals[li] += m.span_len as usize;
                    if m.feat_len > 0 {
                        off_totals[li] += m.tok_count as usize + 1;
                        meth_totals[li] += 1;
                    }
                }
            }
            for li in 0..num_langs {
                lang_indices[li].all_token_lines.reserve_exact(tok_totals[li]);
                lang_indices[li].all_token_end_delta.reserve_exact(tok_totals[li]);
                lang_indices[li].all_struct_spans.reserve_exact(span_totals[li]);
                lang_indices[li].all_offsets.reserve(off_totals[li]);
                lang_indices[li].methods.reserve(meth_totals[li]);
                lang_indices[li].canonical_file_id.reserve(meth_totals[li]);
            }
        }
        let mut global_order: Vec<(u16, u32)> = Vec::new();
        let mut sources: Vec<SourceEntry> = Vec::with_capacity(work.len());
        let mut source_file_indices: Vec<usize> = Vec::with_capacity(work.len());
        let mut path_to_id: FxHashMap<String, u32> = FxHashMap::default();
        let mut seen: FxHashMap<(u16, u128), u32> = FxHashMap::default();

        let mut all_chunk_hashes: Vec<Vec<u64>> = Vec::with_capacity(num_chunks);
        let mut all_chunk_fids: Vec<Vec<u32>> = Vec::with_capacity(num_chunks);
        let mut lang_copies: Vec<Vec<CanonCopy>> = (0..num_langs).map(|_| Vec::new()).collect();
        let mut lang_feat_totals: Vec<usize> = vec![0; num_langs];

        let mut last_file_idx = usize::MAX;
        let mut last_file_id = 0u32;

        for (cid, batch) in chunks.into_iter().enumerate() {
            let mut f_cur = 0usize;
            let mut o_cur = 0usize;
            let mut t_cur = 0usize;
            let mut s_cur = 0usize;
            let mut ov_cur = 0usize;

            for m in batch.methods {
                let file_id = if m.file_idx == last_file_idx {
                    last_file_id
                } else {
                    let f = &files[m.file_idx];
                    let fid = if let Some(&id) = path_to_id.get(&f.path) {
                        source_file_indices[id as usize] = m.file_idx;
                        id
                    } else {
                        let id = sources.len() as u32;
                        path_to_id.insert(f.path.clone(), id);
                        sources.push(SourceEntry { path: f.path.clone(), source: String::new() });
                        source_file_indices.push(m.file_idx);
                        id
                    };
                    last_file_idx = m.file_idx;
                    last_file_id = fid;
                    fid
                };

                let li = m.lang_idx as usize;
                let lidx = &mut lang_indices[li];
                let t_end = t_cur + m.tok_count as usize;
                let s_end = s_cur + m.span_len as usize;
                let ov_end = ov_cur + m.overflow_len as usize;

                let tok_line_start = lidx.all_token_lines.len() as u32;
                lidx.all_token_lines.extend_from_slice(&batch.tok_lines[t_cur..t_end]);
                lidx.all_token_end_delta.extend_from_slice(&batch.tok_end_delta[t_cur..t_end]);
                for &(rel_tok, d) in &batch.end_overflow[ov_cur..ov_end] {
                    lidx.token_end_overflow.insert(tok_line_start + rel_tok, d);
                }
                let span_start = lidx.all_struct_spans.len() as u32;
                lidx.all_struct_spans.extend_from_slice(&batch.spans[s_cur..s_end]);
                let span_end = lidx.all_struct_spans.len() as u32;

                let loc = MethodLocation {
                    file_id,
                    name: m.name,
                    start_line: m.start_line,
                    end_line: m.end_line,
                    tok_line_start,
                    tok_count: m.tok_count,
                    span_start,
                    span_end,
                };

                let key = (m.lang_idx, m.digest);
                if let Some(&earlier_mid) = seen.get(&key) {
                    lidx.methods[earlier_mid as usize].copies.push(loc);
                    stats.duplicate_methods += 1;
                    if m.feat_len > 0 {
                        f_cur += m.feat_len as usize;
                        o_cur += m.tok_count as usize + 1;
                    }
                } else {
                    debug_assert!(m.feat_len > 0);
                    let o_end = o_cur + m.tok_count as usize + 1;
                    let off_start = lidx.all_offsets.len() as u32;
                    lidx.all_offsets.extend_from_slice(&batch.offsets[o_cur..o_end]);

                    let feat_start = lang_feat_totals[li] as u32;
                    lang_feat_totals[li] += m.feat_len as usize;
                    lang_copies[li].push(CanonCopy {
                        chunk_id: cid as u32,
                        src_start: f_cur as u32,
                        dst_start: feat_start,
                        len: m.feat_len,
                    });

                    let mid = lidx.methods.len() as u32;
                    lidx.canonical_file_id.push(file_id);
                    lidx.methods.push(IndexedMethod {
                        primary: loc,
                        copies: Vec::new(),
                        feat_start,
                        off_start,
                    });
                    seen.insert(key, mid);
                    global_order.push((m.lang_idx, mid));

                    f_cur += m.feat_len as usize;
                    o_cur = o_end;
                }

                t_cur = t_end;
                s_cur = s_end;
                ov_cur = ov_end;
            }

            all_chunk_hashes.push(batch.local_hashes);
            all_chunk_fids.push(batch.local_fids);
        }
        drop(seen);

        // Clone retained file sources in parallel!
        sources
            .par_iter_mut()
            .zip(source_file_indices.par_iter())
            .for_each(|(entry, &fi)| {
                entry.source = files[fi].source.clone();
            });
        drop(source_file_indices);

        // Stage 3a: Parallel 32-shard interning of unique chunk-local 64-bit hashes
        let chunk_lens: Vec<usize> = all_chunk_hashes.iter().map(|v| v.len()).collect();
        let per_chunk_buckets: Vec<Vec<Vec<HashBucketEntry>>> = all_chunk_hashes
            .par_iter()
            .enumerate()
            .map(|(cid, hashes)| {
                let est = (hashes.len() / SHARDS).max(16);
                let mut bks: Vec<Vec<HashBucketEntry>> =
                    (0..SHARDS).map(|_| Vec::with_capacity(est)).collect();
                for (lfid, &h) in hashes.iter().enumerate() {
                    let s = shard_of(h);
                    unsafe { bks.get_unchecked_mut(s) }.push(HashBucketEntry {
                        hash: h,
                        chunk_id: cid as u32,
                        local_fid: lfid as u32,
                    });
                }
                bks
            })
            .collect();
        drop(all_chunk_hashes);

        let shard_interned: Vec<ShardInternResult> = (0..SHARDS)
            .into_par_iter()
            .map(|s| {
                let est: usize = per_chunk_buckets.iter().map(|b| b[s].len()).sum();
                let mut table: HashTable<(u64, u32)> = HashTable::with_capacity(est / 2);
                let mut mapped: Vec<(u32, u32, u32)> = Vec::with_capacity(est);
                let mut next_id = 0u32;
                for blk in &per_chunk_buckets {
                    for &e in &blk[s] {
                        let sh = shard_hash(e.hash);
                        let id = if let Some(&(_, id)) = table.find(sh, |x| x.0 == e.hash) {
                            id
                        } else {
                            let id = next_id;
                            next_id += 1;
                            table.insert_unique(sh, (e.hash, id), |x| shard_hash(x.0));
                            id
                        };
                        mapped.push((e.chunk_id, e.local_fid, id));
                    }
                }
                (table, next_id, mapped)
            })
            .collect();
        per_chunk_buckets.into_par_iter().for_each(drop);

        let mut shard_base_fid = [0u32; SHARDS];
        let mut total_features = 0u32;
        for s in 0..SHARDS {
            shard_base_fid[s] = total_features;
            total_features += shard_interned[s].1;
        }
        let feature_count = total_features as usize;

        let mut shards_tables: Vec<HashTable<(u64, u32)>> = Vec::with_capacity(SHARDS);
        let mut all_mapped: Vec<Vec<(u32, u32, u32)>> = Vec::with_capacity(SHARDS);
        for (table, _, mapped) in shard_interned {
            shards_tables.push(table);
            all_mapped.push(mapped);
        }
        // Adjust per-shard feature IDs to global IDs in parallel
        shards_tables.par_iter_mut().enumerate().for_each(|(s, table)| {
            let base = shard_base_fid[s];
            if base > 0 {
                for entry in table.iter_mut() {
                    entry.1 += base;
                }
            }
        });
        let mut chunk_remap: Vec<Vec<u32>> = chunk_lens.iter().map(|&n| vec![0u32; n]).collect();
        for (s, mapped) in all_mapped.into_iter().enumerate() {
            let base = shard_base_fid[s];
            for (cid, lfid, lid) in mapped {
                unsafe {
                    *chunk_remap.get_unchecked_mut(cid as usize).get_unchecked_mut(lfid as usize) =
                        base + lid;
                }
            }
        }
        let feature_table = ShardedFeatureTable { shards: shards_tables };

        // Stage 3b: Parallel remap `all_chunk_fids` -> `lidx.all_features`
        for li in 0..num_langs {
            let total = lang_feat_totals[li];
            if total == 0 {
                continue;
            }
            let copies = &lang_copies[li];
            let mut all_f = vec![0u32; total];
            let n_slices = (threads * 2).min(copies.len()).max(1);
            let per = copies.len().div_ceil(n_slices);
            let mut slice_jobs: Vec<(&[CanonCopy], &mut [u32], usize)> =
                Vec::with_capacity(n_slices);
            let mut rest: &mut [u32] = &mut all_f;
            let mut cur_off = 0usize;
            let mut ci = 0usize;
            while ci < copies.len() {
                let ce = (ci + per).min(copies.len());
                let last = copies[ce - 1];
                let end_off = last.dst_start as usize + last.len as usize;
                let take = end_off - cur_off;
                let (head, tail) = rest.split_at_mut(take);
                slice_jobs.push((&copies[ci..ce], head, cur_off));
                rest = tail;
                cur_off = end_off;
                ci = ce;
            }
            slice_jobs.into_par_iter().for_each(|(jobs, dst, base_off)| {
                for job in jobs {
                    let src = &all_chunk_fids[job.chunk_id as usize]
                        [job.src_start as usize..job.src_start as usize + job.len as usize];
                    let remap = &chunk_remap[job.chunk_id as usize];
                    let d_start = job.dst_start as usize - base_off;
                    let d_slice = &mut dst[d_start..d_start + job.len as usize];
                    for (d, &lfid) in d_slice.iter_mut().zip(src.iter()) {
                        *d = unsafe { *remap.get_unchecked(lfid as usize) };
                    }
                }
            });
            lang_indices[li].all_features = all_f;
        }
        drop(all_chunk_fids);
        drop(chunk_remap);
        drop(lang_copies);

        // Stage 3c: Parallel inverted index build
        for lidx in &mut lang_indices {
            build_inverted_parallel(lidx, feature_count, threads);
        }

        stats.methods = global_order.len();
        stats.features = feature_count;

        (lang_indices, global_order, feature_table, sources, path_to_id)
    });

    AromaIndex {
        languages,
        lang_indices,
        global_order,
        stats,
        feature_table,
        sources,
        path_to_id,
        pool: Some(pool),
    }
}

// ============================================================================
// Queries: `query_from_snippet` and `queries_from_lines`
// ============================================================================

fn hashes_to_query(
    index: &AromaIndex,
    language: String,
    hashes: &[u64],
    origin: Option<MethodRef>,
    lines: Option<Vec<u32>>,
) -> Query {
    let size = hashes.len();
    let mut matched: Vec<u32> = Vec::with_capacity(size);
    for &h in hashes {
        if let Some(fid) = index.feature_table.get(h) {
            matched.push(fid);
        }
    }
    matched.sort_unstable();
    let mut features: Vec<(u32, u32)> = Vec::new();
    let mut i = 0;
    while i < matched.len() {
        let f = matched[i];
        let mut j = i + 1;
        while j < matched.len() && matched[j] == f {
            j += 1;
        }
        features.push((f, (j - i) as u32));
        i = j;
    }
    Query { language, features, size, origin, lines }
}

pub fn query_from_snippet(
    index: &AromaIndex,
    source: &str,
    options: &SnippetOptions,
) -> Option<Query> {
    let lang_idx = options
        .language
        .as_deref()
        .and_then(|id| index.languages.iter().position(|l| l.id() == id))
        .or_else(|| {
            options
                .path
                .as_deref()
                .and_then(|p| language_index_for_path(&index.languages, p))
        })?;
    let lang = &*index.languages[lang_idx];
    let ma_labels = lang.member_access_labels();

    TL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        sc.body.clear();
        sc.hashes.clear();
        sc.offsets.clear();
        sc.tok_lines.clear();
        sc.tok_end_delta.clear();
        sc.end_overflow.clear();

        let owned_dedent: String;
        let (root, base): (u32, &[u8]) = match lang.builtin_kind() {
            1 => {
                let (r, ded) = lang_py::parse_snippet_into(source, options, &mut sc.body)?;
                owned_dedent = ded;
                (r, owned_dedent.as_bytes())
            }
            2 => {
                let r = lang_ts::parse_snippet_into(source, options, &mut sc.body)?;
                (r, source.as_bytes())
            }
            _ => {
                let node = lang.parse_snippet(source, options)?;
                let r = sc.body.push_node(&node);
                (r, &[])
            }
        };

        let tok_count = featurize_body_hashed(
            &sc.body,
            base,
            root,
            ma_labels,
            None,
            &mut sc.feat,
            &mut sc.hashes,
            &mut sc.offsets,
            &mut sc.tok_lines,
            &mut sc.tok_end_delta,
            &mut sc.end_overflow,
            None,
        );
        if tok_count == 0 {
            return None;
        }
        Some(hashes_to_query(index, lang.id().to_string(), &sc.hashes, None, None))
    })
}

pub fn queries_from_lines(index: &AromaIndex, file: &SourceFile, lines: &[u32]) -> Vec<Query> {
    let Some(lang_idx) = language_index_for_path(&index.languages, &file.path) else {
        return Vec::new();
    };
    let lang = &*index.languages[lang_idx];
    let ma_labels = lang.member_access_labels();
    let wanted: FxHashSet<u32> = lines.iter().copied().collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    let base = file.source.as_bytes();

    TL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        parse_file_into_body(lang, file, &mut sc.body, &mut sc.methods);
        let mut queries = Vec::new();

        for m in &sc.methods {
            sc.hashes.clear();
            sc.offsets.clear();
            sc.tok_lines.clear();
            sc.tok_end_delta.clear();
            sc.end_overflow.clear();
            let mut covered: Vec<u32> = Vec::new();

            let mut pred = |line: u32, end_line: u32| -> bool {
                let mut sel = false;
                for ln in line..=end_line {
                    if wanted.contains(&ln) {
                        covered.push(ln);
                        sel = true;
                    }
                }
                sel
            };

            let tok_count = featurize_body_hashed(
                &sc.body,
                base,
                m.body_root,
                ma_labels,
                Some(&mut pred),
                &mut sc.feat,
                &mut sc.hashes,
                &mut sc.offsets,
                &mut sc.tok_lines,
                &mut sc.tok_end_delta,
                &mut sc.end_overflow,
                None,
            );
            if tok_count > 0 {
                covered.sort_unstable();
                covered.dedup();
                let origin = MethodRef {
                    path: file.path.clone(),
                    name: m.name.clone(),
                    language: lang.id().to_string(),
                    start_line: m.start_line,
                    end_line: m.end_line,
                };
                queries.push(hashes_to_query(
                    index,
                    lang.id().to_string(),
                    &sc.hashes,
                    Some(origin),
                    Some(covered),
                ));
            }
        }
        queries
    })
}
