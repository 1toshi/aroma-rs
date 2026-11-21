//! Corpus indexing (`build_index`), 32-shard 64-bit feature interning,
//! multi-wave pipelined parallel indexing for bounded peak RSS,
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
use crate::text::utf16_len_exceeds;
use crate::{builtin_languages, Language, MethodRef, SnippetOptions, SourceFile};

pub(crate) const SHARDS: usize = 32;
const SHARD_MASK: usize = SHARDS - 1;
const SHARD_ID_SHIFT: u32 = 26;
const SHARD_LID_MASK: u32 = (1 << SHARD_ID_SHIFT) - 1;
pub(crate) const COUNT_SHIFT: u32 = 24;
pub(crate) const MID_MASK: u32 = (1 << COUNT_SHIFT) - 1;
pub(crate) const COUNT_OVERFLOW: u32 = 255;

/// Shard index extracted from middle bits (`32..37`), completely independent
/// of both `hashbrown`'s low bucket-index bits (`0..20`) and top 7-bit H2 tag
/// (`57..64`), preserving full 128-way H2 filtering in every shard!
#[inline(always)]
fn shard_of(h: u64) -> usize {
    ((h >> 32) as usize) & SHARD_MASK
}

#[inline(always)]
fn shard_hash(h: u64) -> u64 {
    h
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

    fn with_shard_capacity(cap_per_shard: usize) -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(HashTable::with_capacity(cap_per_shard));
        }
        Self { shards }
    }

    #[inline(always)]
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

    #[inline(always)]
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
    /// Slice in `LanguageIndex::all_token_lines` / `all_token_end_delta`
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

#[derive(Clone, Copy)]
struct SourceSpan {
    path_start: u32,
    path_end: u32,
    src_start: u32,
    src_end: u32,
}

/// Concatenated path & source blob eliminating thousands of per-file `String`
/// heap allocations and copies.
#[derive(Default)]
pub(crate) struct SourceStore {
    paths: String,
    sources: String,
    entries: Vec<SourceSpan>,
}

impl SourceStore {
    fn with_capacity(num_files: usize) -> Self {
        Self {
            paths: String::new(),
            sources: String::new(),
            entries: Vec::with_capacity(num_files),
        }
    }

    #[inline(always)]
    pub fn path(&self, id: u32) -> &str {
        let e = unsafe { self.entries.get_unchecked(id as usize) };
        &self.paths[e.path_start as usize..e.path_end as usize]
    }

    #[inline(always)]
    pub fn source(&self, id: u32) -> &str {
        let e = unsafe { self.entries.get_unchecked(id as usize) };
        &self.sources[e.src_start as usize..e.src_end as usize]
    }

    #[inline]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn push_path_only(&mut self, path: &str) -> u32 {
        let id = self.entries.len() as u32;
        let ps = self.paths.len() as u32;
        self.paths.push_str(path);
        let pe = self.paths.len() as u32;
        self.entries.push(SourceSpan { path_start: ps, path_end: pe, src_start: 0, src_end: 0 });
        id
    }

    fn push_full(&mut self, path: &str, source: &str) -> u32 {
        let id = self.entries.len() as u32;
        let ps = self.paths.len() as u32;
        self.paths.push_str(path);
        let pe = self.paths.len() as u32;
        let ss = self.sources.len() as u32;
        self.sources.push_str(source);
        let se = self.sources.len() as u32;
        self.entries.push(SourceSpan { path_start: ps, path_end: pe, src_start: ss, src_end: se });
        id
    }

    fn set_source(&mut self, id: u32, source: &str) {
        let ss = self.sources.len() as u32;
        self.sources.push_str(source);
        let se = self.sources.len() as u32;
        let e = &mut self.entries[id as usize];
        e.src_start = ss;
        e.src_end = se;
    }

    /// Populates `self.sources` in one contiguous allocation, copying files in parallel
    /// with zero redundant zero-fill.
    fn fill_sources_parallel(&mut self, files: &[SourceFile], file_indices: &[usize]) {
        let mut total = 0usize;
        for (e, &fi) in self.entries.iter_mut().zip(file_indices.iter()) {
            let len = files[fi].source.len();
            e.src_start = total as u32;
            total += len;
            e.src_end = total as u32;
        }
        let mut blob: Vec<u8> = Vec::with_capacity(total);
        {
            let spare = &mut blob.spare_capacity_mut()[..total];
            if total >= 262_144 && file_indices.len() >= 32 {
                let n_jobs = 32usize.min(file_indices.len());
                let target = (total / n_jobs).max(1);
                type CopyChunk<'a> = (&'a [usize], &'a mut [std::mem::MaybeUninit<u8>]);
                let mut jobs: Vec<CopyChunk<'_>> = Vec::with_capacity(n_jobs);
                let mut rest = spare;
                let mut fi_start = 0usize;
                while fi_start < file_indices.len() {
                    let mut acc = 0usize;
                    let mut fi_end = fi_start;
                    while fi_end < file_indices.len() {
                        acc += files[file_indices[fi_end]].source.len();
                        fi_end += 1;
                        if acc >= target && jobs.len() + 1 < n_jobs {
                            break;
                        }
                    }
                    let (head, tail) = rest.split_at_mut(acc);
                    jobs.push((&file_indices[fi_start..fi_end], head));
                    rest = tail;
                    fi_start = fi_end;
                }
                jobs.into_par_iter().for_each(|(fis, dst)| {
                    let mut pos = 0usize;
                    for &fi in fis {
                        let src = files[fi].source.as_bytes();
                        // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` have identical layout.
                        let src_uninit = unsafe {
                            std::slice::from_raw_parts(
                                src.as_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                                src.len(),
                            )
                        };
                        dst[pos..pos + src.len()].copy_from_slice(src_uninit);
                        pos += src.len();
                    }
                });
            } else {
                let mut pos = 0usize;
                for &fi in file_indices {
                    let bytes = files[fi].source.as_bytes();
                    let src_uninit = unsafe {
                        std::slice::from_raw_parts(
                            bytes.as_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                            bytes.len(),
                        )
                    };
                    spare[pos..pos + bytes.len()].copy_from_slice(src_uninit);
                    pos += bytes.len();
                }
            }
        }
        // SAFETY: every byte in `0..total` was initialized above.
        unsafe {
            blob.set_len(total);
        }
        // SAFETY: every copied slice came from a valid UTF-8 `&str`.
        self.sources = unsafe { String::from_utf8_unchecked(blob) };
    }
}

pub struct AromaIndex {
    pub(crate) languages: Vec<Box<dyn Language>>,
    pub(crate) lang_indices: Vec<LanguageIndex>,
    pub(crate) global_order: Vec<(u16, u32)>,
    pub(crate) stats: IndexStats,
    pub(crate) feature_table: ShardedFeatureTable,
    pub(crate) sources: SourceStore,
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
            path: self.sources.path(loc.file_id).to_string(),
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
    local_table: HashTable<(u64, u32)>,
    local_seen: FxHashSet<(u16, u128)>,
}

impl ParseScratch {
    #[inline]
    fn cap_after_large_file(&mut self) {
        if self.body.items.capacity() > 16384 {
            self.body.items.clear();
            self.body.items.shrink_to(4096);
        }
        if self.body.kids.capacity() > 16384 {
            self.body.kids.clear();
            self.body.kids.shrink_to(4096);
        }
        if self.hashes.capacity() > 32768 {
            self.hashes.clear();
            self.hashes.shrink_to(8192);
        }
    }
}

thread_local! {
    static TL_SCRATCH: RefCell<ParseScratch> = RefCell::new(ParseScratch::default());
}

fn parse_file_into_body(
    lang: &dyn Language,
    file: &SourceFile,
    line_range: Option<(u32, u32)>,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    body.clear();
    out.clear();
    match lang.builtin_kind() {
        1 => lang_py::parse_methods_into(&file.source, line_range, body, out),
        2 => lang_ts::parse_methods_into(&file.source, &file.path, line_range, body, out),
        _ => {
            for m in lang.parse_methods(&file.source, &file.path) {
                if let Some((min_ln, max_ln)) = line_range
                    && (m.end_line < min_ln || m.start_line > max_ln)
                {
                    continue;
                }
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
// `build_index` — Single-thread streaming + Multi-wave parallel pipelined
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
    sources: &mut SourceStore,
    file: &SourceFile,
) -> u32 {
    if let Some(&id) = path_to_id.get(&file.path) {
        sources.set_source(id, &file.source);
        id
    } else {
        let id = sources.push_full(&file.path, &file.source);
        path_to_id.insert(file.path.clone(), id);
        id
    }
}

fn build_inverted_single(lidx: &mut LanguageIndex, feature_count: usize) {
    let mcount = lidx.methods.len();
    if mcount == 0 || feature_count == 0 {
        lidx.posting_offsets = vec![0u32; feature_count + 1];
        return;
    }
    let mut posting_offsets = vec![0u32; feature_count + 1];
    let mut overflow = FxHashMap::default();
    let postings = if mcount < u16::MAX as usize {
        let mut last_method = vec![u16::MAX; feature_count];
        for mid in 0..mcount {
            let feats = lidx.method_features(mid);
            let mid16 = mid as u16;
            for &f in feats {
                let fi = f as usize;
                if unsafe { *last_method.get_unchecked(fi) } != mid16 {
                    unsafe {
                        *last_method.get_unchecked_mut(fi) = mid16;
                        *posting_offsets.get_unchecked_mut(fi + 1) += 1;
                    }
                }
            }
        }
        for f in 0..feature_count {
            posting_offsets[f + 1] += posting_offsets[f];
        }
        let total_postings = posting_offsets[feature_count] as usize;
        let mut postings: Vec<u32> = Vec::with_capacity(total_postings);
        let mut cursor = posting_offsets[..feature_count].to_vec();
        last_method.fill(u16::MAX);
        {
            let p_uninit = &mut postings.spare_capacity_mut()[..total_postings];
            for mid in 0..mcount {
                let feats = lidx.method_features(mid);
                let mid16 = mid as u16;
                let mid_u32 = mid as u32;
                for &f in feats {
                    let fi = f as usize;
                    if unsafe { *last_method.get_unchecked(fi) } != mid16 {
                        unsafe {
                            *last_method.get_unchecked_mut(fi) = mid16;
                            let pos = *cursor.get_unchecked(fi) as usize;
                            *cursor.get_unchecked_mut(fi) = (pos + 1) as u32;
                            p_uninit.get_unchecked_mut(pos).write((1u32 << COUNT_SHIFT) | mid_u32);
                        }
                    } else {
                        let pos = (unsafe { *cursor.get_unchecked(fi) } - 1) as usize;
                        let val = unsafe { p_uninit.get_unchecked(pos).assume_init() };
                        let cur = val >> COUNT_SHIFT;
                        if cur < COUNT_OVERFLOW - 1 {
                            unsafe {
                                p_uninit
                                    .get_unchecked_mut(pos)
                                    .write(val + (1u32 << COUNT_SHIFT));
                            }
                        } else if cur == COUNT_OVERFLOW - 1 {
                            unsafe {
                                p_uninit
                                    .get_unchecked_mut(pos)
                                    .write((COUNT_OVERFLOW << COUNT_SHIFT) | mid_u32);
                            }
                            overflow.insert((f, mid_u32), COUNT_OVERFLOW);
                        } else {
                            *overflow.get_mut(&(f, mid_u32)).unwrap() += 1;
                        }
                    }
                }
            }
        }
        unsafe {
            postings.set_len(total_postings);
        }
        postings
    } else {
        let mut last_method = vec![u32::MAX; feature_count];
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
                        unsafe {
                            *postings.get_unchecked_mut(pos) =
                                (COUNT_OVERFLOW << COUNT_SHIFT) | mid_u32;
                        }
                        overflow.insert((f, mid_u32), COUNT_OVERFLOW);
                    } else {
                        *overflow.get_mut(&(f, mid_u32)).unwrap() += 1;
                    }
                }
            }
        }
        postings
    };

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
    let mut total_bytes = 0usize;
    if files.len() >= 16 {
        let mut bytes_by_lang = vec![0usize; languages.len()];
        for f in files {
            if let Some(li) = language_index_for_path(&languages, &f.path)
                && f.source.len() <= max_file_length
            {
                bytes_by_lang[li] += f.source.len();
                total_bytes += f.source.len();
            }
        }
        for (li, &b) in bytes_by_lang.iter().enumerate() {
            if b >= 65536 {
                lang_indices[li].all_features.reserve(b / 5 + b / 25);
                lang_indices[li].all_offsets.reserve(b / 24);
                lang_indices[li].all_token_lines.reserve(b / 18);
                lang_indices[li].all_token_end_delta.reserve(b / 18);
                lang_indices[li].all_struct_spans.reserve(b / 52);
            }
        }
    }
    let mut global_order: Vec<(u16, u32)> = Vec::new();
    let mut stats = IndexStats::default();
    let mut feature_table = if total_bytes >= 65536 {
        ShardedFeatureTable::with_shard_capacity((total_bytes / (SHARDS * 35)).max(128))
    } else {
        ShardedFeatureTable::new()
    };
    let mut next_fid = 0u32;
    let mut sources = SourceStore::with_capacity(files.len());
    if total_bytes >= 65536 {
        sources.sources.reserve(total_bytes);
    }
    let mut path_to_id: FxHashMap<String, u32> = FxHashMap::default();
    let mut seen: FxHashMap<(u16, u128), u32> = if total_bytes >= 65536 {
        FxHashMap::with_capacity_and_hasher(total_bytes / 1000, Default::default())
    } else {
        FxHashMap::default()
    };

    let mut scratch = ParseScratch::default();

    for file in files {
        let Some(li) = language_index_for_path(&languages, &file.path) else {
            stats.unsupported_files += 1;
            continue;
        };
        if utf16_len_exceeds(&file.source, max_file_length) {
            stats.oversized_files += 1;
            continue;
        }
        stats.files += 1;
        let lang = &*languages[li];
        parse_file_into_body(lang, file, None, &mut scratch.body, &mut scratch.methods);
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
        scratch.cap_after_large_file();
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
// Multi-wave pipelined parallel indexing
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

#[derive(Default)]
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

impl ChunkBatch {
    fn clear(&mut self) {
        self.methods.clear();
        self.local_hashes.clear();
        self.local_fids.clear();
        self.offsets.clear();
        self.tok_lines.clear();
        self.tok_end_delta.clear();
        self.end_overflow.clear();
        self.spans.clear();
    }
}

#[derive(Clone, Copy)]
struct CanonCopy {
    chunk_id: u32,
    src_start: u32,
    dst_start: u32,
    len: u32,
}

#[derive(Default)]
struct ShardBucket {
    hashes: Vec<u64>,
    ids: Vec<u32>, // packed `(cid << 24) | lfid` when compact, else `[cid, lfid]` pairs
}

impl ShardBucket {
    fn clear(&mut self) {
        self.hashes.clear();
        self.ids.clear();
    }
}

/// Wrapper for disjoint parallel writes into `chunk_remap[cid][lfid]`.
/// SAFETY: Every `(cid, lfid)` belongs to a unique shard `s = shard_of(hash)`,
/// so distinct shards never alias the same element.
struct RemapPtrs(Vec<*mut u32>);
unsafe impl Sync for RemapPtrs {}
impl RemapPtrs {
    #[inline(always)]
    unsafe fn write(&self, cid: usize, lfid: usize, val: u32) {
        unsafe {
            *(*self.0.get_unchecked(cid)).add(lfid) = val;
        }
    }
}

fn build_inverted_parallel(
    lidx: &mut LanguageIndex,
    feature_count: usize,
    threads: usize,
    reusable_u32_buckets: &mut Vec<Vec<Vec<u32>>>,
) {
    let mcount = lidx.methods.len();
    if mcount == 0 || feature_count == 0 {
        lidx.posting_offsets = vec![0u32; feature_count + 1];
        return;
    }
    if mcount < 512 || feature_count < 4096 {
        build_inverted_single(lidx, feature_count);
        return;
    }

    let target_buckets = (threads * 3).next_power_of_two().clamp(16, 64);
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

    let avg_per_bucket = lidx.all_features.len() / (blocks.len() * num_buckets).max(1);
    let est_per_bucket = (avg_per_bucket + avg_per_bucket / 4).max(64);
    let use_packed_u32 = mcount < u16::MAX as usize && bucket_width <= 65536;

    if use_packed_u32 {
        let lf_mask = bucket_width - 1;
        if reusable_u32_buckets.len() < blocks.len() {
            reusable_u32_buckets.resize_with(blocks.len(), Vec::new);
        }

        // Phase 1: Single branch-free parallel scatter into reusable `block_buckets`.
        reusable_u32_buckets[..blocks.len()]
            .par_iter_mut()
            .zip(blocks.par_iter())
            .for_each(|(bks, &(m_start, m_end))| {
                if bks.len() < num_buckets {
                    bks.resize_with(num_buckets, || Vec::with_capacity(est_per_bucket));
                }
                for bk in &mut bks[..num_buckets] {
                    bk.clear();
                    if bk.capacity() < est_per_bucket {
                        bk.reserve(est_per_bucket);
                    }
                }
                for mid in m_start..m_end {
                    let mid_hi = (mid as u32) << 16;
                    for &fid in lidx.method_features(mid) {
                        let b = (fid as usize) >> shift;
                        let lf = (fid as usize) & lf_mask;
                        unsafe { bks.get_unchecked_mut(b) }.push(mid_hi | (lf as u32));
                    }
                }
            });
        let block_buckets = &reusable_u32_buckets[..blocks.len()];

        // Phase 2a: Parallel zero-init + count distinct methods per feature directly
        // into `posting_offsets[1..]` using reused L1-resident `u16` `last_mid`.
        let mut posting_offsets: Vec<u32> = Vec::with_capacity(feature_count + 1);
        posting_offsets.push(0);
        let bucket_lens: Vec<usize> = {
            let spare = &mut posting_offsets.spare_capacity_mut()[..feature_count];
            spare
                .par_chunks_mut(bucket_width)
                .enumerate()
                .map_init(
                    || vec![u16::MAX; bucket_width],
                    |last_mid, (b, counts_uninit)| {
                        let width = counts_uninit.len();
                        for c in counts_uninit.iter_mut() {
                            c.write(0);
                        }
                        // SAFETY: all `width` elements were just initialized to `0`.
                        let counts_out = unsafe {
                            std::slice::from_raw_parts_mut(
                                counts_uninit.as_mut_ptr().cast::<u32>(),
                                width,
                            )
                        };
                        last_mid[..width].fill(u16::MAX);
                        let mut bsum = 0usize;
                        for blk in block_buckets {
                            for &entry in &blk[b] {
                                let mid = (entry >> 16) as u16;
                                let lf = (entry & 0xFFFF) as usize;
                                if unsafe { *last_mid.get_unchecked(lf) } != mid {
                                    unsafe {
                                        *last_mid.get_unchecked_mut(lf) = mid;
                                        *counts_out.get_unchecked_mut(lf) += 1;
                                    }
                                    bsum += 1;
                                }
                            }
                        }
                        bsum
                    },
                )
                .collect()
        };
        // SAFETY: index 0 was pushed and indices `1..=feature_count` were initialized above.
        unsafe {
            posting_offsets.set_len(feature_count + 1);
        }

        // Parallel prefix sum over `posting_offsets[1..=feature_count]`.
        let mut bucket_bases: Vec<u32> = Vec::with_capacity(num_buckets);
        let mut running = 0u32;
        for &blen in &bucket_lens {
            bucket_bases.push(running);
            running += blen as u32;
        }
        let total_p = running as usize;
        posting_offsets[1..=feature_count]
            .par_chunks_mut(bucket_width)
            .zip(bucket_bases.par_iter())
            .for_each(|(chunk, &base)| {
                let mut acc = base;
                for x in chunk.iter_mut() {
                    acc += *x;
                    *x = acc;
                }
            });
        let mut all_postings: Vec<u32> = Vec::with_capacity(total_p);

        // Phase 2b: Scatter packed postings directly into `all_postings` in parallel
        // with reused per-worker `(cursors, last_mid)` scratch.
        let overflows: Vec<Vec<(u32, u32, u32)>> = {
            let mut slices: Vec<(usize, &mut [std::mem::MaybeUninit<u32>])> =
                Vec::with_capacity(num_buckets);
            let mut rest = &mut all_postings.spare_capacity_mut()[..total_p];
            for (b, &blen) in bucket_lens.iter().enumerate() {
                let (head, tail) = rest.split_at_mut(blen);
                slices.push((b, head));
                rest = tail;
            }
            let po_ref = &posting_offsets;
            slices
                .into_par_iter()
                .map_init(
                    || (vec![0u32; bucket_width], vec![u16::MAX; bucket_width]),
                    |(cursors, last_mid), (b, postings)| {
                        let f_base = b * bucket_width;
                        let f_end = (f_base + bucket_width).min(feature_count);
                        let width = f_end - f_base;
                        let b_base = po_ref[f_base];
                        for i in 0..width {
                            cursors[i] = po_ref[f_base + i] - b_base;
                        }
                        last_mid[..width].fill(u16::MAX);
                        let mut overflow: Vec<(u32, u32, u32)> = Vec::new();
                        let mut ov_map: FxHashMap<(u32, u32), usize> = FxHashMap::default();

                        for blk in block_buckets {
                            for &entry in &blk[b] {
                                let mid16 = (entry >> 16) as u16;
                                let mid = mid16 as u32;
                                let lf = (entry & 0xFFFF) as usize;
                                if unsafe { *last_mid.get_unchecked(lf) } != mid16 {
                                    unsafe {
                                        *last_mid.get_unchecked_mut(lf) = mid16;
                                        let pos = *cursors.get_unchecked(lf) as usize;
                                        *cursors.get_unchecked_mut(lf) = (pos + 1) as u32;
                                        postings
                                            .get_unchecked_mut(pos)
                                            .write((1u32 << COUNT_SHIFT) | mid);
                                    }
                                } else {
                                    let pos = (unsafe { *cursors.get_unchecked(lf) } - 1) as usize;
                                    let val = unsafe { postings.get_unchecked(pos).assume_init() };
                                    let cur = val >> COUNT_SHIFT;
                                    if cur < COUNT_OVERFLOW - 1 {
                                        unsafe {
                                            postings
                                                .get_unchecked_mut(pos)
                                                .write(val + (1u32 << COUNT_SHIFT));
                                        }
                                    } else if cur == COUNT_OVERFLOW - 1 {
                                        let fid = (f_base + lf) as u32;
                                        unsafe {
                                            postings
                                                .get_unchecked_mut(pos)
                                                .write((COUNT_OVERFLOW << COUNT_SHIFT) | mid);
                                        }
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
                        overflow
                    },
                )
                .collect()
        };
        // SAFETY: every slot in `0..total_p` was written above.
        unsafe {
            all_postings.set_len(total_p);
        }

        for ov in overflows {
            for (fid, mid, exact) in ov {
                lidx.posting_overflow.insert((fid, mid), exact);
            }
        }
        lidx.posting_offsets = posting_offsets;
        lidx.postings = all_postings;
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

        let bucket_counts: Vec<Vec<u32>> = (0..num_buckets)
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
                counts
            })
            .collect();

        let mut posting_offsets = vec![0u32; feature_count + 1];
        let mut bucket_lens: Vec<usize> = Vec::with_capacity(num_buckets);
        for (b, counts) in bucket_counts.iter().enumerate() {
            let f_base = b * bucket_width;
            let mut bsum = 0usize;
            for (i, &c) in counts.iter().enumerate() {
                posting_offsets[f_base + i + 1] = c;
                bsum += c as usize;
            }
            bucket_lens.push(bsum);
        }
        drop(bucket_counts);
        for f in 0..feature_count {
            posting_offsets[f + 1] += posting_offsets[f];
        }
        let total_p = posting_offsets[feature_count] as usize;
        let mut all_postings: Vec<u32> = Vec::with_capacity(total_p);

        let overflows: Vec<Vec<(u32, u32, u32)>> = {
            let mut slices: Vec<(usize, &mut [std::mem::MaybeUninit<u32>])> =
                Vec::with_capacity(num_buckets);
            let mut rest = &mut all_postings.spare_capacity_mut()[..total_p];
            for (b, &blen) in bucket_lens.iter().enumerate() {
                let (head, tail) = rest.split_at_mut(blen);
                slices.push((b, head));
                rest = tail;
            }
            let po_ref = &posting_offsets;
            slices
                .into_par_iter()
                .map(|(b, postings)| {
                    let f_base = b * bucket_width;
                    let f_end = (f_base + bucket_width).min(feature_count);
                    let width = f_end - f_base;
                    let b_base = po_ref[f_base];
                    let mut cursors = vec![0u32; width];
                    for i in 0..width {
                        cursors[i] = po_ref[f_base + i] - b_base;
                    }
                    let mut last_mid = vec![u32::MAX; width];
                    let mut overflow: Vec<(u32, u32, u32)> = Vec::new();
                    let mut ov_map: FxHashMap<(u32, u32), usize> = FxHashMap::default();

                    for blk in &block_buckets {
                        for &(mid, fid) in &blk[b] {
                            let lf = fid as usize - f_base;
                            if unsafe { *last_mid.get_unchecked(lf) } != mid {
                                unsafe {
                                    *last_mid.get_unchecked_mut(lf) = mid;
                                    let pos = *cursors.get_unchecked(lf) as usize;
                                    *cursors.get_unchecked_mut(lf) = (pos + 1) as u32;
                                    postings
                                        .get_unchecked_mut(pos)
                                        .write((1u32 << COUNT_SHIFT) | mid);
                                }
                            } else {
                                let pos = (unsafe { *cursors.get_unchecked(lf) } - 1) as usize;
                                let val = unsafe { postings.get_unchecked(pos).assume_init() };
                                let cur = val >> COUNT_SHIFT;
                                if cur < COUNT_OVERFLOW - 1 {
                                    unsafe {
                                        postings
                                            .get_unchecked_mut(pos)
                                            .write(val + (1u32 << COUNT_SHIFT));
                                    }
                                } else if cur == COUNT_OVERFLOW - 1 {
                                    unsafe {
                                        postings
                                            .get_unchecked_mut(pos)
                                            .write((COUNT_OVERFLOW << COUNT_SHIFT) | mid);
                                    }
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
                    overflow
                })
                .collect()
        };
        // SAFETY: every slot in `0..total_p` was written above.
        unsafe {
            all_postings.set_len(total_p);
        }

        block_buckets.into_par_iter().for_each(drop);
        for ov in overflows {
            for (fid, mid, exact) in ov {
                lidx.posting_overflow.insert((fid, mid), exact);
            }
        }
        lidx.posting_offsets = posting_offsets;
        lidx.postings = all_postings;
    }
}

fn build_index_parallel(
    files: &[SourceFile],
    languages: Vec<Box<dyn Language>>,
    max_file_length: usize,
    threads: usize,
) -> AromaIndex {
    let num_langs = languages.len();
    let mut stats = IndexStats::default();
    let mut work: Vec<(usize, u16)> = Vec::with_capacity(files.len());
    let mut total_weight = 0usize;
    let mut total_bytes = 0usize;
    let mut bytes_by_lang = vec![0usize; num_langs];
    let is_ts_lang: Vec<bool> = languages.iter().map(|l| l.builtin_kind() == 2).collect();

    for (idx, file) in files.iter().enumerate() {
        let Some(li) = language_index_for_path(&languages, &file.path) else {
            stats.unsupported_files += 1;
            continue;
        };
        if utf16_len_exceeds(&file.source, max_file_length) {
            stats.oversized_files += 1;
            continue;
        }
        stats.files += 1;
        let slen = file.source.len();
        total_bytes += slen;
        bytes_by_lang[li] += slen;
        total_weight += if is_ts_lang[li] { (slen * 3) / 2 + 768 } else { slen + 512 };
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
            sources: SourceStore::default(),
            path_to_id: FxHashMap::default(),
            pool: Some(pool),
        };
    }

    // Single parallel wave with `threads * 3` dynamic LPT chunks maximizes 15T throughput
    // while keeping peak RSS bounded (~334 MB on 29 MB corpus under macOS libmalloc).
    let num_waves = 1usize;
    let target_wave_w = (total_weight / num_waves).max(1);
    let mut wave_ranges: Vec<(usize, usize, usize)> = Vec::with_capacity(num_waves);
    {
        let mut ws = 0usize;
        for w_idx in 0..num_waves {
            if w_idx + 1 == num_waves {
                let mut w = 0usize;
                for &(fi, li) in &work[ws..] {
                    let slen = files[fi].source.len();
                    w += if is_ts_lang[li as usize] { (slen * 3) / 2 + 768 } else { slen + 512 };
                }
                if ws < work.len() {
                    wave_ranges.push((ws, work.len(), w));
                }
                break;
            }
            let mut w = 0usize;
            let mut we = ws;
            while we < work.len() {
                let (fi, li) = work[we];
                let slen = files[fi].source.len();
                w += if is_ts_lang[li as usize] { (slen * 3) / 2 + 768 } else { slen + 512 };
                we += 1;
                if w >= target_wave_w {
                    break;
                }
            }
            if we > ws {
                wave_ranges.push((ws, we, w));
            }
            ws = we;
        }
    }

    let (lang_indices, global_order, feature_table, sources, path_to_id) = pool.install(|| {
        let mut lang_indices: Vec<LanguageIndex> =
            languages.iter().map(|l| LanguageIndex::new(l.id().to_string())).collect();
        for (li, &b) in bytes_by_lang.iter().enumerate() {
            if b >= 65536 {
                lang_indices[li].all_features.reserve(b / 5 + b / 25);
                lang_indices[li].all_offsets.reserve(b / 24);
                lang_indices[li].all_token_lines.reserve(b / 18);
                lang_indices[li].all_token_end_delta.reserve(b / 18);
                lang_indices[li].all_struct_spans.reserve(b / 52);
            }
        }

        let mut global_order: Vec<(u16, u32)> = Vec::new();
        let mut sources = SourceStore::with_capacity(work.len());
        let mut source_file_indices: Vec<usize> = Vec::with_capacity(work.len());
        let mut path_to_id: FxHashMap<String, u32> =
            FxHashMap::with_capacity_and_hasher(work.len(), Default::default());
        let mut seen: FxHashMap<(u16, u128), u32> = if total_bytes >= 65536 {
            FxHashMap::with_capacity_and_hasher(total_bytes / 1000, Default::default())
        } else {
            FxHashMap::default()
        };

        let est_shard_cap = if total_bytes >= 65536 {
            (total_bytes / (SHARDS * 35)).max(128)
        } else {
            64
        };
        let mut shards_tables: Vec<HashTable<(u64, u32)>> =
            (0..SHARDS).map(|_| HashTable::with_capacity(est_shard_cap)).collect();
        let mut shard_next_id = [0u32; SHARDS];

        let mut last_file_idx = usize::MAX;
        let mut last_file_id = 0u32;

        // Reusable wave pools: allocated once in wave 0 and reused across all waves
        // to eliminate cross-wave mimalloc segment stacking and page-fault churn.
        // Use `threads * 3` chunks per wave so P-cores dynamically steal more work
        // than E-cores with minimal wave-barrier tail stall.
        let max_chunks = (threads * 3).min(work.len()).clamp(1, 250);
        let mut chunk_pool: Vec<ChunkBatch> = (0..max_chunks).map(|_| ChunkBatch::default()).collect();
        let mut per_chunk_buckets: Vec<Vec<ShardBucket>> = (0..max_chunks)
            .map(|_| (0..SHARDS).map(|_| ShardBucket::default()).collect())
            .collect();
        let mut chunk_remap: Vec<Vec<u32>> = (0..max_chunks).map(|_| Vec::new()).collect();
        let mut wave_copies: Vec<Vec<CanonCopy>> = (0..num_langs).map(|_| Vec::new()).collect();

        for &(wave_start, wave_end, wave_w) in &wave_ranges {
            let wave_len = wave_end - wave_start;
            let target_chunks = (threads * 3).min(wave_len).clamp(1, 250);
            let target_cw = (wave_w / target_chunks).max(1);
            let mut chunk_ranges: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(target_chunks);
            let mut s = wave_start;
            while s < wave_end {
                let rem_c = target_chunks.saturating_sub(chunk_ranges.len());
                if rem_c <= 1 {
                    let mut w = 0usize;
                    let mut b = 0usize;
                    for &(fi, li) in &work[s..wave_end] {
                        let slen = files[fi].source.len();
                        b += slen;
                        w += if is_ts_lang[li as usize] { (slen * 3) / 2 + 768 } else { slen + 512 };
                    }
                    chunk_ranges.push((s, wave_end, w, b));
                    break;
                }
                let mut w = 0usize;
                let mut b = 0usize;
                let mut e = s;
                while e < wave_end {
                    let (fi, li) = work[e];
                    let slen = files[fi].source.len();
                    b += slen;
                    w += if is_ts_lang[li as usize] { (slen * 3) / 2 + 768 } else { slen + 512 };
                    e += 1;
                    if w >= target_cw || wave_end - e < rem_c - 1 {
                        break;
                    }
                }
                chunk_ranges.push((s, e, w, b));
                s = e;
            }

            let num_chunks = chunk_ranges.len();
            // Wave Stage 1: LPT-scheduled parallel parse + simplify + intra-chunk dedup + local intern
            // directly into the reusable `chunk_pool[..num_chunks]`.
            let mut work_pairs: Vec<(usize, &mut ChunkBatch)> =
                chunk_pool[..num_chunks].iter_mut().enumerate().collect();
            work_pairs.sort_unstable_by(|(a, _), (b, _)| chunk_ranges[*b].2.cmp(&chunk_ranges[*a].2));

            work_pairs
                .into_par_iter()
                .for_each(|(cid, batch)| {
                    TL_SCRATCH.with(|cell| {
                        let sc = &mut *cell.borrow_mut();
                        let (ws, we, _, chunk_bytes) = chunk_ranges[cid];
                        sc.local_table.clear();
                        sc.local_seen.clear();
                        batch.clear();
                        if batch.local_fids.capacity() == 0 {
                            batch.methods.reserve((chunk_bytes / 1200).max(16));
                            batch.local_hashes.reserve((chunk_bytes / 22).max(64));
                            batch.local_fids.reserve((chunk_bytes / 4).max(64));
                            batch.offsets.reserve((chunk_bytes / 24).max(32));
                            batch.tok_lines.reserve((chunk_bytes / 18).max(32));
                            batch.tok_end_delta.reserve((chunk_bytes / 18).max(32));
                            batch.spans.reserve((chunk_bytes / 52).max(16));
                        }

                        for &(file_idx, lang_idx) in &work[ws..we] {
                            let file = &files[file_idx];
                            let lang = &*languages[lang_idx as usize];
                            let base = file.source.as_bytes();
                            let ma_labels = lang.member_access_labels();
                            parse_file_into_body(lang, file, None, &mut sc.body, &mut sc.methods);

                            for m in &sc.methods {
                                let digest = sc.body.digest(base, m.body_root);
                                let key = (lang_idx, digest);
                                let s_start = batch.spans.len();
                                let ov_start = batch.end_overflow.len();

                                if sc.local_seen.contains(&key) {
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
                                sc.local_seen.insert(key);
                                batch.local_fids.reserve(sc.hashes.len());
                                for &h in &sc.hashes {
                                    let lfid = if let Some(&(_, id)) =
                                        sc.local_table.find(h, |e| e.0 == h)
                                    {
                                        id
                                    } else {
                                        let id = batch.local_hashes.len() as u32;
                                        batch.local_hashes.push(h);
                                        sc.local_table.insert_unique(h, (h, id), |e| e.0);
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
                            sc.cap_after_large_file();
                        }
                    });
                });

            // Wave Stage 2: Deterministic sequential dedup & metadata assembly
            for v in &mut wave_copies {
                v.clear();
            }
            let mut wave_feat_base = vec![0usize; num_langs];
            let mut wave_feat_added = vec![0usize; num_langs];
            for li in 0..num_langs {
                wave_feat_base[li] = lang_indices[li].all_features.len();
            }

            for (cid, batch) in chunk_pool[..num_chunks].iter_mut().enumerate() {
                let mut f_cur = 0usize;
                let mut o_cur = 0usize;
                let mut t_cur = 0usize;
                let mut s_cur = 0usize;
                let mut ov_cur = 0usize;

                for m in batch.methods.drain(..) {
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
                            sources.push_path_only(&f.path);
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

                        let dst_rel = wave_feat_added[li] as u32;
                        let feat_start = (wave_feat_base[li] + wave_feat_added[li]) as u32;
                        wave_feat_added[li] += m.feat_len as usize;
                        wave_copies[li].push(CanonCopy {
                            chunk_id: cid as u32,
                            src_start: f_cur as u32,
                            dst_start: dst_rel,
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
            }

            // Wave Stage 3a: Parallel 32-shard interning with direct disjoint write into `chunk_remap`
            let compact_ids = num_chunks < 256
                && chunk_pool[..num_chunks].iter().all(|c| c.local_hashes.len() < (1 << 24));
            per_chunk_buckets[..num_chunks]
                .par_iter_mut()
                .enumerate()
                .for_each(|(cid, bks)| {
                    let hashes = &chunk_pool[cid].local_hashes;
                    let est = (hashes.len() / SHARDS).max(16);
                    for bk in bks.iter_mut() {
                        bk.clear();
                        if bk.hashes.capacity() == 0 {
                            bk.hashes.reserve(est);
                            bk.ids.reserve(est);
                        }
                    }
                    let cid_hi = (cid as u32) << 24;
                    for (lfid, &h) in hashes.iter().enumerate() {
                        let s = shard_of(h);
                        let bk = unsafe { bks.get_unchecked_mut(s) };
                        bk.hashes.push(h);
                        if compact_ids {
                            bk.ids.push(cid_hi | (lfid as u32));
                        } else {
                            bk.ids.push(cid as u32);
                            bk.ids.push(lfid as u32);
                        }
                    }
                });

            let mut ptrs: Vec<*mut u32> = Vec::with_capacity(num_chunks);
            for cid in 0..num_chunks {
                let need = chunk_pool[cid].local_hashes.len();
                let rv = &mut chunk_remap[cid];
                rv.clear();
                rv.resize(need, 0);
                ptrs.push(rv.as_mut_ptr());
            }
            let remap_ptrs = RemapPtrs(ptrs);
            let bks_ref = &per_chunk_buckets[..num_chunks];

            shards_tables
                .par_iter_mut()
                .zip(shard_next_id.par_iter_mut())
                .enumerate()
                .for_each(|(s, (table, next_id))| {
                    let s_tag = (s as u32) << SHARD_ID_SHIFT;
                    for blk in bks_ref {
                        let bk = &blk[s];
                        if compact_ids {
                            for (&h, &packed) in bk.hashes.iter().zip(bk.ids.iter()) {
                                let sh = shard_hash(h);
                                let lid = if let Some(&(_, id)) = table.find(sh, |x| x.0 == h) {
                                    id
                                } else {
                                    let id = *next_id;
                                    *next_id += 1;
                                    table.insert_unique(sh, (h, id), |x| shard_hash(x.0));
                                    id
                                };
                                let cid = (packed >> 24) as usize;
                                let lfid = (packed & 0x00FF_FFFF) as usize;
                                unsafe {
                                    remap_ptrs.write(cid, lfid, s_tag | lid);
                                }
                            }
                        } else {
                            for (k, &h) in bk.hashes.iter().enumerate() {
                                let sh = shard_hash(h);
                                let lid = if let Some(&(_, id)) = table.find(sh, |x| x.0 == h) {
                                    id
                                } else {
                                    let id = *next_id;
                                    *next_id += 1;
                                    table.insert_unique(sh, (h, id), |x| shard_hash(x.0));
                                    id
                                };
                                let cid = bk.ids[k * 2] as usize;
                                let lfid = bk.ids[k * 2 + 1] as usize;
                                unsafe {
                                    remap_ptrs.write(cid, lfid, s_tag | lid);
                                }
                            }
                        }
                    }
                });

            // Wave Stage 3b: Parallel remap `chunk_pool[cid].local_fids` directly into `lang_indices[li].all_features`
            let chunks_ref = &chunk_pool[..num_chunks];
            let remap_ref = &chunk_remap[..num_chunks];
            for li in 0..num_langs {
                let added = wave_feat_added[li];
                if added == 0 {
                    continue;
                }
                let copies = &wave_copies[li];
                let old_len = lang_indices[li].all_features.len();
                lang_indices[li].all_features.reserve(added);
                {
                    let dst_all = &mut lang_indices[li].all_features.spare_capacity_mut()[..added];
                    let n_slices = (threads * 4).min(copies.len()).max(1);
                    let per = copies.len().div_ceil(n_slices);
                    type RemapJob<'a> =
                        (&'a [CanonCopy], &'a mut [std::mem::MaybeUninit<u32>], usize);
                    let mut slice_jobs: Vec<RemapJob<'_>> = Vec::with_capacity(n_slices);
                    let mut rest = dst_all;
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
                            let src = &chunks_ref[job.chunk_id as usize].local_fids
                                [job.src_start as usize..job.src_start as usize + job.len as usize];
                            let remap = &remap_ref[job.chunk_id as usize];
                            let d_start = job.dst_start as usize - base_off;
                            let d_slice = &mut dst[d_start..d_start + job.len as usize];
                            for (d, &lfid) in d_slice.iter_mut().zip(src.iter()) {
                                d.write(unsafe { *remap.get_unchecked(lfid as usize) });
                            }
                        }
                    });
                }
                // SAFETY: every element in `old_len..old_len + added` was written above.
                unsafe {
                    lang_indices[li].all_features.set_len(old_len + added);
                }
            }
        }

        // Release all temporary wave pools & thread-local scratch BEFORE Stage 3c & `sources`!
        chunk_pool.into_par_iter().for_each(drop);
        per_chunk_buckets.into_par_iter().for_each(drop);
        drop(chunk_remap);
        drop(wave_copies);
        drop(seen);
        pool.broadcast(|_| {
            lezer::shrink_tl_workspace();
            lang_py::shrink_tl_tree();
            TL_SCRATCH.with(|cell| *cell.borrow_mut() = ParseScratch::default());
        });

        // Finalize global dense feature IDs (`shard_base_fid[s] + lid`) in-place!
        let mut shard_base_fid = [0u32; SHARDS];
        let mut total_features = 0u32;
        for s in 0..SHARDS {
            shard_base_fid[s] = total_features;
            total_features += shard_next_id[s];
        }
        let feature_count = total_features as usize;
        shards_tables.par_iter_mut().enumerate().for_each(|(s, table)| {
            let base = shard_base_fid[s];
            if base > 0 {
                for entry in table.iter_mut() {
                    entry.1 += base;
                }
            }
        });
        for lidx in &mut lang_indices {
            lidx.all_features.par_chunks_mut(65536).for_each(|chunk| {
                for fid in chunk {
                    let s = (*fid >> SHARD_ID_SHIFT) as usize;
                    let lid = *fid & SHARD_LID_MASK;
                    *fid = unsafe { *shard_base_fid.get_unchecked(s) } + lid;
                }
            });
        }
        let feature_table = ShardedFeatureTable { shards: shards_tables };

        // Stage 3c: Parallel zero-copy-postings inverted index build (reusing `block_buckets` across languages)
        let mut reusable_u32_buckets: Vec<Vec<Vec<u32>>> = Vec::new();
        for lidx in &mut lang_indices {
            build_inverted_parallel(lidx, feature_count, threads, &mut reusable_u32_buckets);
        }
        reusable_u32_buckets.into_par_iter().for_each(drop);

        // Fill retained file sources in one contiguous blob
        sources.fill_sources_parallel(files, &source_file_indices);
        drop(source_file_indices);

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
    if lines.is_empty() {
        return Vec::new();
    }
    let Some(lang_idx) = language_index_for_path(&index.languages, &file.path) else {
        return Vec::new();
    };
    let lang = &*index.languages[lang_idx];
    let ma_labels = lang.member_access_labels();

    let mut min_ln = u32::MAX;
    let mut max_ln = 0u32;
    for &ln in lines {
        min_ln = min_ln.min(ln);
        max_ln = max_ln.max(ln);
    }
    let wanted_set: Option<FxHashSet<u32>> = if lines.len() > 16 {
        Some(lines.iter().copied().collect())
    } else {
        None
    };

    let base = file.source.as_bytes();

    TL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        parse_file_into_body(lang, file, Some((min_ln, max_ln)), &mut sc.body, &mut sc.methods);
        let mut queries = Vec::new();

        for m in &sc.methods {
            if m.end_line < min_ln || m.start_line > max_ln {
                continue;
            }
            sc.hashes.clear();
            sc.offsets.clear();
            sc.tok_lines.clear();
            sc.tok_end_delta.clear();
            sc.end_overflow.clear();
            let mut covered: Vec<u32> = Vec::new();

            let mut pred = |line: u32, end_line: u32| -> bool {
                if end_line < min_ln || line > max_ln {
                    return false;
                }
                let mut sel = false;
                if let Some(ref set) = wanted_set {
                    for ln in line..=end_line {
                        if set.contains(&ln) {
                            covered.push(ln);
                            sel = true;
                        }
                    }
                } else {
                    for ln in line..=end_line {
                        if lines.contains(&ln) {
                            covered.push(ln);
                            sel = true;
                        }
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
