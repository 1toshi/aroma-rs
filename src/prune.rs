//! Algorithm 1 of the Aroma paper (`prune`) with two exact fast paths:
//! 1. `FastQueryMap`: 4 KB L1 Bloom filter + direct L1-hot `qidx[fid]` table.
//! 2. Adaptive exact greedy descent:
//!    - Zero-greedy fast path: when `sum_t gain_0(t) == expected_score`, every
//!      token with `gain_0(t) > 0` is necessarily kept with no greedy descent.
//!    - `max_gain == 1` fast path: single linear scan over hitting tokens.
//!    - Small/medium hitting sets (`tok_hits.len() < SIG_THRESHOLD` or explicit
//!      `candidates`): direct bucket descent with dirty-bucket skip-sort.
//!    - Large hitting sets (`tok_hits.len() >= SIG_THRESHOLD`, `candidates: None`):
//!      exact signature-equivalence-class greedy (`run_greedy_sig`).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
pub struct TokenTable {
    pub features: Vec<i32>,
    pub offsets: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pruned {
    pub tokens: Vec<usize>,
    pub score: usize,
}

/// Public Algorithm 1 implementation operating on an arbitrary `HashMap<i32, usize>`
/// and `TokenTable`.
pub fn prune(
    target: &HashMap<i32, usize>,
    table: &TokenTable,
    candidates: Option<&[usize]>,
) -> Pruned {
    if target.is_empty() || table.offsets.len() < 2 {
        return Pruned { tokens: Vec::new(), score: 0 };
    }
    let mut pt = PruneTarget::default();
    pt.build_from_i32_map(target);
    let mut scratch = PruneScratch::default();
    let feats_u32: Vec<u32> = table.features.iter().map(|&x| x as u32).collect();
    prune_fast(&pt, &feats_u32, &table.offsets, 0, None, candidates, &mut scratch)
}

// ============================================================================
// 4 KB L1 Bloom + compact L1-resident open-addressed `QSlot` map (`FastQueryMap`)
// ============================================================================

const BLOOM_WORDS: usize = 512; // 32,768 bits = 4 KB
const BLOOM_MASK: usize = BLOOM_WORDS - 1;
const EMPTY_KEY: u32 = u32::MAX;
const MULT: u32 = 0x9E37_79B9;

pub(crate) struct FastQueryMap {
    pub bloom: [u64; BLOOM_WORDS],
    pub qidx: Vec<u16>,
    pub caps: Vec<u32>,
    active_fids: Vec<u32>,
}

impl Default for FastQueryMap {
    fn default() -> Self {
        Self {
            bloom: [0u64; BLOOM_WORDS],
            qidx: Vec::new(),
            caps: Vec::new(),
            active_fids: Vec::new(),
        }
    }
}

impl FastQueryMap {
    pub fn load(&mut self, feature_count: usize, pairs: &[(u32, u32)]) -> bool {
        if pairs.len() >= (u16::MAX as usize - 1) {
            return false;
        }
        if self.qidx.len() < feature_count {
            self.qidx.resize(feature_count, u16::MAX);
        }
        self.clear();
        self.caps.reserve(pairs.len());
        self.active_fids.reserve(pairs.len());
        for &(fid, c) in pairs {
            if c > 0 && (fid as usize) < feature_count {
                let idx = self.caps.len() as u16;
                self.caps.push(c);
                unsafe { *self.qidx.get_unchecked_mut(fid as usize) = idx };
                self.bloom[((fid as usize) >> 6) & BLOOM_MASK] |= 1u64 << (fid & 63);
                self.active_fids.push(fid);
            }
        }
        true
    }

    pub fn clear(&mut self) {
        if !self.active_fids.is_empty() {
            for &fid in &self.active_fids {
                unsafe { *self.qidx.get_unchecked_mut(fid as usize) = u16::MAX };
                self.bloom[((fid as usize) >> 6) & BLOOM_MASK] = 0;
            }
            self.active_fids.clear();
        }
        self.caps.clear();
    }
}

// ============================================================================
// Open-addressed target table (public `prune`, `recommend::intersect`, fallback)
// ============================================================================

#[derive(Clone, Copy)]
struct Slot {
    key: u32,
    idx: u32,
}

#[derive(Default)]
pub(crate) struct PruneTarget {
    slots: Vec<Slot>,
    mask: usize,
    has_max_key: bool,
    max_key_idx: u32,
    pub caps: Vec<u32>,
}

impl PruneTarget {
    pub fn build_from_pairs(&mut self, pairs: &[(u32, u32)]) {
        self.reset(pairs.len());
        for &(k, cap) in pairs {
            if cap > 0 {
                self.insert(k, cap);
            }
        }
    }

    pub fn build_from_i32_map(&mut self, map: &HashMap<i32, usize>) {
        self.reset(map.len());
        for (&k, &cap) in map {
            if cap > 0 {
                self.insert(k as u32, cap as u32);
            }
        }
    }

    fn reset(&mut self, count: usize) {
        self.caps.clear();
        self.has_max_key = false;
        self.max_key_idx = 0;
        let cap = (count.saturating_mul(4).max(32)).next_power_of_two();
        self.slots.clear();
        self.slots.resize(cap, Slot { key: EMPTY_KEY, idx: 0 });
        self.mask = cap - 1;
    }

    #[inline]
    fn insert(&mut self, key: u32, cap: u32) {
        if key == EMPTY_KEY {
            if !self.has_max_key {
                self.has_max_key = true;
                self.max_key_idx = self.caps.len() as u32;
                self.caps.push(cap);
            } else {
                self.caps[self.max_key_idx as usize] += cap;
            }
            return;
        }
        let mut h = (key.wrapping_mul(MULT) as usize) & self.mask;
        loop {
            let s = self.slots[h];
            if s.key == EMPTY_KEY {
                let idx = self.caps.len() as u32;
                self.slots[h] = Slot { key, idx };
                self.caps.push(cap);
                return;
            }
            if s.key == key {
                self.caps[s.idx as usize] += cap;
                return;
            }
            h = (h + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn get(&self, key: u32) -> u32 {
        if key == EMPTY_KEY {
            return if self.has_max_key { self.max_key_idx } else { EMPTY_KEY };
        }
        let mut h = (key.wrapping_mul(MULT) as usize) & self.mask;
        loop {
            let s = unsafe { *self.slots.get_unchecked(h) };
            if s.key == key {
                return s.idx;
            }
            if s.key == EMPTY_KEY {
                return EMPTY_KEY;
            }
            h = (h + 1) & self.mask;
        }
    }
}

// ============================================================================
// Reusable Algorithm 1 scratch: direct bucket descent + signature-class greedy
// ============================================================================

#[derive(Clone, Copy)]
struct TokHits {
    token: u32,
    start: u32,
    end: u32,
    has_dup: bool,
}

#[derive(Clone, Copy)]
struct SigSlot {
    /// Top 2 bits encode inline tag kind:
    /// - `0b00`: `k >= 5` (or `k == 4` with `q >= 32768`), 62-bit hash of sorted `hits`
    /// - `0b01`: `k == 2` inline `(q0 << 16) | q1`
    /// - `0b10`: `k == 3` inline `(q0 << 32) | (q1 << 16) | q2`
    /// - `0b11`: `k == 4` inline `(q0 << 45) | (q1 << 30) | (q2 << 15) | q3` (when `q_len <= 32768`)
    tag: u64,
    cls_id: u32,
    epoch: u32,
}

#[derive(Clone, Copy)]
struct SigClass {
    sig_start: u32,
    sig_end: u32,
    has_dup: bool,
    max_needed: u32,
    count: u32,
    tok_head: u32,
    tok_tail: u32,
}

#[derive(Default)]
pub(crate) struct PruneScratch {
    rem: Vec<u32>,
    single_seen: Vec<u32>,
    hit_qfids: Vec<u32>,
    tok_hits: Vec<TokHits>,
    buckets: Vec<Vec<u32>>,
    cur_bucket: Vec<u32>,
    sig_single_cls: Vec<u32>,
    sig_slots: Vec<SigSlot>,
    sig_epoch: u32,
    sig_classes: Vec<SigClass>,
    sig_qfids: Vec<u32>,
    tok_nodes: Vec<(u32, u32)>,
    bucket_heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl PruneScratch {
    #[inline]
    fn reset_sig_table(&mut self, q_len: usize) {
        self.sig_single_cls.clear();
        self.sig_single_cls.resize(q_len, u32::MAX);
        self.sig_classes.clear();
        self.sig_qfids.clear();
        self.tok_nodes.clear();
        if self.sig_slots.len() < 256 {
            self.sig_slots.resize(256, SigSlot { tag: 0, cls_id: 0, epoch: 0 });
        }
        self.sig_epoch = self.sig_epoch.wrapping_add(1);
        if self.sig_epoch == 0 {
            for s in &mut self.sig_slots {
                s.epoch = 0;
            }
            self.sig_epoch = 1;
        }
    }

    #[cold]
    fn grow_sig_table(&mut self) {
        let new_cap = (self.sig_slots.len() * 2).max(256);
        let mask = new_cap - 1;
        let epoch = self.sig_epoch;
        let mut new_slots = vec![SigSlot { tag: 0, cls_id: 0, epoch: 0 }; new_cap];
        for s in &self.sig_slots {
            if s.epoch == epoch {
                let h = if (s.tag >> 62) != 0 {
                    mix_u64(s.tag)
                } else {
                    s.tag as usize
                };
                let mut pos = h & mask;
                while new_slots[pos].epoch == epoch {
                    pos = (pos + 1) & mask;
                }
                new_slots[pos] = *s;
            }
        }
        self.sig_slots = new_slots;
    }
}

#[inline(always)]
fn mix_u64(x: u64) -> usize {
    let mut z = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (z ^ (z >> 27)) as usize
}

#[inline(always)]
fn sort_small_u32(slice: &mut [u32]) {
    let n = slice.len();
    for i in 1..n {
        let key = unsafe { *slice.get_unchecked(i) };
        let mut j = i;
        while j > 0 && unsafe { *slice.get_unchecked(j - 1) } > key {
            unsafe { *slice.get_unchecked_mut(j) = *slice.get_unchecked(j - 1) };
            j -= 1;
        }
        unsafe { *slice.get_unchecked_mut(j) = key };
    }
}

#[inline(always)]
fn hash_sig(hits: &[u32]) -> u64 {
    let mut h = (hits.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for &q in hits {
        h = (h ^ (q as u64)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
    (h ^ (h >> 31)) & ((1u64 << 62) - 1)
}

#[inline(always)]
fn compute_gain_sorted(hits: &[u32], has_dup: bool, rem: &[u32]) -> usize {
    if !has_dup {
        let mut g = 0usize;
        for &q in hits {
            g += (unsafe { *rem.get_unchecked(q as usize) } > 0) as usize;
        }
        g
    } else {
        let mut g = 0usize;
        let mut i = 0usize;
        let len = hits.len();
        while i < len {
            let q = unsafe { *hits.get_unchecked(i) };
            let mut j = i + 1;
            while j < len && unsafe { *hits.get_unchecked(j) } == q {
                j += 1;
            }
            let mult = (j - i) as u32;
            let r = unsafe { *rem.get_unchecked(q as usize) };
            g += mult.min(r) as usize;
            i = j;
        }
        g
    }
}

#[inline]
fn run_greedy_buckets_into(
    max_gain: usize,
    expected_score: Option<usize>,
    initial_sorted: bool,
    scratch: &mut PruneScratch,
    kept: &mut Vec<usize>,
) -> usize {
    kept.clear();
    if max_gain == 0 {
        return 0;
    }
    let mut score = 0usize;
    let mut dirty_mask: u64 = if initial_sorted { 0 } else { u64::MAX };

    'outer: for val in (1..=max_gain).rev() {
        if scratch.buckets[val].is_empty() {
            continue;
        }
        std::mem::swap(&mut scratch.cur_bucket, &mut scratch.buckets[val]);
        if val >= 64 || (dirty_mask & (1u64 << val)) != 0 {
            if initial_sorted {
                scratch.cur_bucket.sort_unstable();
            } else {
                let tok_hits = &scratch.tok_hits;
                scratch.cur_bucket.sort_unstable_by_key(|&ei| unsafe {
                    tok_hits.get_unchecked(ei as usize).token
                });
            }
        }

        let n = scratch.cur_bucket.len();
        for idx in 0..n {
            let ei = unsafe { *scratch.cur_bucket.get_unchecked(idx) };
            let th = unsafe { *scratch.tok_hits.get_unchecked(ei as usize) };
            let hits = &scratch.hit_qfids[th.start as usize..th.end as usize];
            let cur_gain = if !th.has_dup {
                let mut g = 0usize;
                for &q in hits {
                    g += (unsafe { *scratch.rem.get_unchecked(q as usize) } > 0) as usize;
                }
                g
            } else {
                let mut g = 0usize;
                for (k, &q) in hits.iter().enumerate() {
                    let rem_q = unsafe { *scratch.rem.get_unchecked(q as usize) };
                    if rem_q > 0 {
                        let mut earlier = 0u32;
                        for &p in &hits[..k] {
                            if p == q {
                                earlier += 1;
                            }
                        }
                        if earlier < rem_q {
                            g += 1;
                        }
                    }
                }
                g
            };

            if cur_gain < val {
                if cur_gain > 0 {
                    scratch.buckets[cur_gain].push(ei);
                    if cur_gain < 64 {
                        dirty_mask |= 1u64 << cur_gain;
                    }
                }
                continue;
            }

            kept.push(th.token as usize);
            score += val;
            for &q in hits {
                let r = unsafe { scratch.rem.get_unchecked_mut(q as usize) };
                if *r > 0 {
                    *r -= 1;
                }
            }
            if expected_score == Some(score) {
                scratch.cur_bucket.clear();
                for b in &mut scratch.buckets[1..val] {
                    b.clear();
                }
                break 'outer;
            }
        }
        scratch.cur_bucket.clear();
    }

    kept.sort_unstable();
    score
}

#[inline]
fn run_greedy_sig_into(
    max_gain: usize,
    expected_score: Option<usize>,
    scratch: &mut PruneScratch,
    kept: &mut Vec<usize>,
) -> usize {
    kept.clear();
    if max_gain == 0 {
        return 0;
    }
    let mut score = 0usize;

    'outer: for val in (1..=max_gain).rev() {
        let blen = scratch.buckets[val].len();
        if blen == 0 {
            continue;
        }
        // Single-class bucket fast path: avoid BinaryHeap completely.
        if blen == 1 {
            let cid = scratch.buckets[val][0];
            scratch.buckets[val].clear();
            let c = unsafe { *scratch.sig_classes.get_unchecked(cid as usize) };
            let hits = &scratch.sig_qfids[c.sig_start as usize..c.sig_end as usize];
            let cur_gain = compute_gain_sorted(hits, c.has_dup, &scratch.rem);
            if cur_gain < val {
                if cur_gain > 0 {
                    scratch.buckets[cur_gain].push(cid);
                }
                continue;
            }
            let mut node = c.tok_head;
            loop {
                let (tok, next_node) = unsafe { *scratch.tok_nodes.get_unchecked(node as usize) };
                kept.push(tok as usize);
                score += val;
                let mut next_gain = 0usize;
                if !c.has_dup {
                    for &q in hits {
                        let r = unsafe { scratch.rem.get_unchecked_mut(q as usize) };
                        if *r > 0 {
                            *r -= 1;
                            next_gain += (*r > 0) as usize;
                        }
                    }
                } else {
                    for &q in hits {
                        let r = unsafe { scratch.rem.get_unchecked_mut(q as usize) };
                        if *r > 0 {
                            *r -= 1;
                        }
                    }
                    if next_node != u32::MAX {
                        next_gain = compute_gain_sorted(hits, true, &scratch.rem);
                    }
                }
                if expected_score == Some(score) {
                    for b in &mut scratch.buckets[1..val] {
                        b.clear();
                    }
                    break 'outer;
                }
                if next_node == u32::MAX {
                    break;
                }
                if next_gain < val {
                    unsafe { scratch.sig_classes.get_unchecked_mut(cid as usize).tok_head = next_node };
                    if next_gain > 0 {
                        scratch.buckets[next_gain].push(cid);
                    }
                    break;
                }
                node = next_node;
            }
            continue;
        }

        scratch.bucket_heap.clear();
        for &cid in &scratch.buckets[val] {
            let head_node = unsafe { scratch.sig_classes.get_unchecked(cid as usize).tok_head };
            let head_tok = unsafe { scratch.tok_nodes.get_unchecked(head_node as usize).0 };
            scratch.bucket_heap.push(Reverse((head_tok, cid)));
        }
        scratch.buckets[val].clear();

        while let Some(Reverse((mut head_tok, cid))) = scratch.bucket_heap.pop() {
            let c = unsafe { *scratch.sig_classes.get_unchecked(cid as usize) };
            let hits = &scratch.sig_qfids[c.sig_start as usize..c.sig_end as usize];
            let cur_gain = compute_gain_sorted(hits, c.has_dup, &scratch.rem);

            if cur_gain < val {
                if cur_gain > 0 {
                    scratch.buckets[cur_gain].push(cid);
                }
                continue;
            }

            let peek_tok = scratch.bucket_heap.peek().map_or(u32::MAX, |&Reverse((t, _))| t);
            let mut node = c.tok_head;
            loop {
                kept.push(head_tok as usize);
                score += val;
                let mut next_gain = 0usize;
                if !c.has_dup {
                    for &q in hits {
                        let r = unsafe { scratch.rem.get_unchecked_mut(q as usize) };
                        if *r > 0 {
                            *r -= 1;
                            next_gain += (*r > 0) as usize;
                        }
                    }
                } else {
                    for &q in hits {
                        let r = unsafe { scratch.rem.get_unchecked_mut(q as usize) };
                        if *r > 0 {
                            *r -= 1;
                        }
                    }
                    next_gain = compute_gain_sorted(hits, true, &scratch.rem);
                }
                if expected_score == Some(score) {
                    scratch.bucket_heap.clear();
                    for b in &mut scratch.buckets[1..val] {
                        b.clear();
                    }
                    break 'outer;
                }

                let next_node = unsafe { scratch.tok_nodes.get_unchecked(node as usize).1 };
                if next_node == u32::MAX {
                    break;
                }
                if next_gain < val {
                    unsafe { scratch.sig_classes.get_unchecked_mut(cid as usize).tok_head = next_node };
                    if next_gain > 0 {
                        scratch.buckets[next_gain].push(cid);
                    }
                    break;
                }
                let next_tok = unsafe { scratch.tok_nodes.get_unchecked(next_node as usize).0 };
                node = next_node;
                head_tok = next_tok;
                if next_tok >= peek_tok {
                    unsafe { scratch.sig_classes.get_unchecked_mut(cid as usize).tok_head = next_node };
                    scratch.bucket_heap.push(Reverse((next_tok, cid)));
                    break;
                }
            }
        }
    }

    kept.sort_unstable();
    score
}

const SIG_THRESHOLD: usize = 512;

/// Fast Algorithm 1 using 4 KB L1 Bloom + direct L1-hot `qidx[fid]`.
pub(crate) fn prune_with_fast_map(
    qmap: &FastQueryMap,
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    expected_score: Option<usize>,
    candidates: Option<&[usize]>,
    scratch: &mut PruneScratch,
) -> Pruned {
    let mut tokens = Vec::new();
    let score = prune_with_fast_map_into(
        qmap,
        all_features,
        offsets,
        feat_base,
        expected_score,
        candidates,
        scratch,
        &mut tokens,
    );
    Pruned { tokens, score }
}

/// Reusable-buffer version of `prune_with_fast_map` eliminating `kept` allocations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prune_with_fast_map_into(
    qmap: &FastQueryMap,
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    expected_score: Option<usize>,
    candidates: Option<&[usize]>,
    scratch: &mut PruneScratch,
    kept: &mut Vec<usize>,
) -> usize {
    let q_len = qmap.caps.len();
    if expected_score == Some(0) || q_len == 0 || offsets.len() < 2 {
        kept.clear();
        return 0;
    }
    let tok_count = offsets.len() - 1;
    if candidates.is_none() && tok_count >= SIG_THRESHOLD {
        return prune_with_fast_map_sig_into(
            qmap,
            all_features,
            offsets,
            feat_base,
            expected_score,
            scratch,
            kept,
        );
    }

    scratch.rem.clear();
    scratch.rem.extend_from_slice(&qmap.caps);
    scratch.single_seen.clear();
    scratch.single_seen.resize(q_len, 0u32);
    scratch.hit_qfids.clear();
    scratch.tok_hits.clear();
    scratch.cur_bucket.clear();

    let bloom = &qmap.bloom;
    let qidx = &qmap.qidx;
    let caps = &qmap.caps;
    let mut max_gain = 0usize;
    let mut total_gain = 0usize;
    let ascending = candidates.is_none_or(|c| c.windows(2).all(|w| w[0] < w[1]));

    let mut process_token = |tok: usize| {
        let s = feat_base + unsafe { *offsets.get_unchecked(tok) } as usize;
        let e = feat_base + unsafe { *offsets.get_unchecked(tok + 1) } as usize;
        let slice = &all_features[s..e];
        let slen = slice.len();

        let hit_start = scratch.hit_qfids.len() as u32;
        let mut gain = 0usize;
        let mut has_dup = false;

        if slen <= 32 {
            let mut hit_bits = 0u32;
            for (i, &f) in slice.iter().enumerate() {
                let b = (unsafe { *bloom.get_unchecked(((f as usize) >> 6) & BLOOM_MASK) } >> (f & 63)) & 1;
                hit_bits |= (b as u32) << i;
            }
            if hit_bits == 0 {
                return;
            }
            if hit_bits & (hit_bits - 1) == 0 {
                let i = hit_bits.trailing_zeros() as usize;
                let f = unsafe { *slice.get_unchecked(i) };
                let q16 = unsafe { *qidx.get_unchecked(f as usize) };
                if q16 != u16::MAX {
                    let q = q16 as usize;
                    total_gain += 1;
                    if ascending {
                        let seen = unsafe { scratch.single_seen.get_unchecked_mut(q) };
                        if *seen >= unsafe { *caps.get_unchecked(q) } {
                            return;
                        }
                        *seen += 1;
                    }
                    scratch.hit_qfids.push(q16 as u32);
                    let ei = scratch.tok_hits.len() as u32;
                    scratch.tok_hits.push(TokHits {
                        token: tok as u32,
                        start: hit_start,
                        end: hit_start + 1,
                        has_dup: false,
                    });
                    if scratch.buckets.len() <= 1 {
                        scratch.buckets.resize_with(2, Vec::new);
                    }
                    scratch.buckets[1].push(ei);
                    if max_gain < 1 {
                        max_gain = 1;
                    }
                }
                return;
            }
            let mut hb = hit_bits;
            while hb != 0 {
                let i = hb.trailing_zeros() as usize;
                hb &= hb - 1;
                let f = unsafe { *slice.get_unchecked(i) };
                let q16 = unsafe { *qidx.get_unchecked(f as usize) };
                if q16 != u16::MAX {
                    let q = q16 as u32;
                    let cap = unsafe { *caps.get_unchecked(q16 as usize) };
                    let prev = &scratch.hit_qfids[hit_start as usize..];
                    let mut dup_cnt = 0u32;
                    for &p in prev {
                        if p == q {
                            dup_cnt += 1;
                        }
                    }
                    if dup_cnt > 0 {
                        has_dup = true;
                    }
                    if dup_cnt < cap {
                        gain += 1;
                    }
                    scratch.hit_qfids.push(q);
                }
            }
        } else {
            for &f in slice {
                if ((unsafe { *bloom.get_unchecked(((f as usize) >> 6) & BLOOM_MASK) } >> (f & 63)) & 1) == 0 {
                    continue;
                }
                let q16 = unsafe { *qidx.get_unchecked(f as usize) };
                if q16 != u16::MAX {
                    let q = q16 as u32;
                    let cap = unsafe { *caps.get_unchecked(q16 as usize) };
                    let prev = &scratch.hit_qfids[hit_start as usize..];
                    let mut dup_cnt = 0u32;
                    for &p in prev {
                        if p == q {
                            dup_cnt += 1;
                        }
                    }
                    if dup_cnt > 0 {
                        has_dup = true;
                    }
                    if dup_cnt < cap {
                        gain += 1;
                    }
                    scratch.hit_qfids.push(q);
                }
            }
        }

        if gain == 0 {
            return;
        }
        total_gain += gain;
        let hit_end = scratch.hit_qfids.len() as u32;
        if ascending && hit_end - hit_start == 1 {
            let q = scratch.hit_qfids[hit_start as usize] as usize;
            let seen = unsafe { scratch.single_seen.get_unchecked_mut(q) };
            if *seen >= unsafe { *caps.get_unchecked(q) } {
                scratch.hit_qfids.truncate(hit_start as usize);
                return;
            }
            *seen += 1;
        }
        let ei = scratch.tok_hits.len() as u32;
        scratch.tok_hits.push(TokHits { token: tok as u32, start: hit_start, end: hit_end, has_dup });
        if gain >= scratch.buckets.len() {
            scratch.buckets.resize_with(gain + 1, Vec::new);
        }
        scratch.buckets[gain].push(ei);
        if gain > max_gain {
            max_gain = gain;
        }
    };

    match candidates {
        Some(cands) => {
            for &tok in cands {
                process_token(tok);
            }
        }
        None => {
            for tok in 0..tok_count {
                process_token(tok);
            }
        }
    }

    if expected_score == Some(total_gain) && total_gain > 0 {
        for b in &mut scratch.buckets[1..=max_gain] {
            b.clear();
        }
        kept.clear();
        kept.extend(scratch.tok_hits.iter().map(|th| th.token as usize));
        if !ascending {
            kept.sort_unstable();
        }
        return total_gain;
    }

    run_greedy_buckets_into(max_gain, expected_score, ascending, scratch, kept)
}

fn prune_with_fast_map_sig_into(
    qmap: &FastQueryMap,
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    expected_score: Option<usize>,
    scratch: &mut PruneScratch,
    kept: &mut Vec<usize>,
) -> usize {
    let q_len = qmap.caps.len();
    scratch.rem.clear();
    scratch.rem.extend_from_slice(&qmap.caps);
    scratch.reset_sig_table(q_len);

    let bloom = &qmap.bloom;
    let qidx = &qmap.qidx;
    let caps = &qmap.caps;
    let tok_count = offsets.len() - 1;
    let mut max_gain = 0usize;
    let mut local_q = [0u32; 32];
    let mut overflow_q: Vec<u32> = Vec::new();

    for tok in 0..tok_count {
        let s = feat_base + unsafe { *offsets.get_unchecked(tok) } as usize;
        let e = feat_base + unsafe { *offsets.get_unchecked(tok + 1) } as usize;
        let slice = &all_features[s..e];
        let slen = slice.len();

        let mut k = 0usize;
        let mut use_overflow = false;

        if slen <= 32 {
            let mut hit_bits = 0u32;
            for (i, &f) in slice.iter().enumerate() {
                let b = (unsafe { *bloom.get_unchecked(((f as usize) >> 6) & BLOOM_MASK) } >> (f & 63)) & 1;
                hit_bits |= (b as u32) << i;
            }
            if hit_bits == 0 {
                continue;
            }
            let mut hb = hit_bits;
            while hb != 0 {
                let i = hb.trailing_zeros() as usize;
                hb &= hb - 1;
                let f = unsafe { *slice.get_unchecked(i) };
                let q16 = unsafe { *qidx.get_unchecked(f as usize) };
                if q16 != u16::MAX {
                    unsafe { *local_q.get_unchecked_mut(k) = q16 as u32 };
                    k += 1;
                }
            }
        } else {
            overflow_q.clear();
            for &f in slice {
                if ((unsafe { *bloom.get_unchecked(((f as usize) >> 6) & BLOOM_MASK) } >> (f & 63)) & 1) == 0 {
                    continue;
                }
                let q16 = unsafe { *qidx.get_unchecked(f as usize) };
                if q16 != u16::MAX {
                    let q = q16 as u32;
                    if !use_overflow {
                        if k < 32 {
                            local_q[k] = q;
                            k += 1;
                        } else {
                            overflow_q.extend_from_slice(&local_q);
                            overflow_q.push(q);
                            use_overflow = true;
                        }
                    } else {
                        overflow_q.push(q);
                    }
                }
            }
        }

        if !use_overflow && k == 0 {
            continue;
        }

        if !use_overflow && k == 1 {
            let q = local_q[0];
            let existing = unsafe { *scratch.sig_single_cls.get_unchecked(q as usize) };
            if existing != u32::MAX {
                let c = unsafe { scratch.sig_classes.get_unchecked_mut(existing as usize) };
                if c.count < c.max_needed {
                    let nid = scratch.tok_nodes.len() as u32;
                    scratch.tok_nodes.push((tok as u32, u32::MAX));
                    unsafe { scratch.tok_nodes.get_unchecked_mut(c.tok_tail as usize).1 = nid };
                    c.tok_tail = nid;
                    c.count += 1;
                }
                continue;
            }
            let sig_start = scratch.sig_qfids.len() as u32;
            scratch.sig_qfids.push(q);
            let nid = scratch.tok_nodes.len() as u32;
            scratch.tok_nodes.push((tok as u32, u32::MAX));
            let cid = scratch.sig_classes.len() as u32;
            let cap = unsafe { *caps.get_unchecked(q as usize) };
            scratch.sig_classes.push(SigClass {
                sig_start,
                sig_end: sig_start + 1,
                has_dup: false,
                max_needed: cap,
                count: 1,
                tok_head: nid,
                tok_tail: nid,
            });
            unsafe { *scratch.sig_single_cls.get_unchecked_mut(q as usize) = cid };
            if scratch.buckets.len() <= 1 {
                scratch.buckets.resize_with(2, Vec::new);
            }
            scratch.buckets[1].push(cid);
            if max_gain < 1 {
                max_gain = 1;
            }
            continue;
        }

        let epoch = scratch.sig_epoch;
        let mask = scratch.sig_slots.len() - 1;

        if !use_overflow && k == 2 {
            let (q0, q1) = if local_q[0] <= local_q[1] {
                (local_q[0], local_q[1])
            } else {
                (local_q[1], local_q[0])
            };
            let tag = (1u64 << 62) | ((q0 as u64) << 16) | (q1 as u64);
            let mut pos = mix_u64(tag) & mask;
            let found_cid = loop {
                let slot = unsafe { *scratch.sig_slots.get_unchecked(pos) };
                if slot.epoch != epoch {
                    break u32::MAX;
                }
                if slot.tag == tag {
                    break slot.cls_id;
                }
                pos = (pos + 1) & mask;
            };
            if found_cid != u32::MAX {
                let c = unsafe { scratch.sig_classes.get_unchecked_mut(found_cid as usize) };
                if c.count < c.max_needed {
                    let nid = scratch.tok_nodes.len() as u32;
                    scratch.tok_nodes.push((tok as u32, u32::MAX));
                    unsafe { scratch.tok_nodes.get_unchecked_mut(c.tok_tail as usize).1 = nid };
                    c.tok_tail = nid;
                    c.count += 1;
                }
                continue;
            }
            let (gain, has_dup, max_needed) = if q0 == q1 {
                let c0 = unsafe { *caps.get_unchecked(q0 as usize) };
                (2u32.min(c0) as usize, true, c0.div_ceil(2))
            } else {
                let c0 = unsafe { *caps.get_unchecked(q0 as usize) };
                let c1 = unsafe { *caps.get_unchecked(q1 as usize) };
                (2usize, false, c0.max(c1))
            };
            let sig_start = scratch.sig_qfids.len() as u32;
            scratch.sig_qfids.push(q0);
            scratch.sig_qfids.push(q1);
            let nid = scratch.tok_nodes.len() as u32;
            scratch.tok_nodes.push((tok as u32, u32::MAX));
            let cid = scratch.sig_classes.len() as u32;
            scratch.sig_classes.push(SigClass {
                sig_start,
                sig_end: sig_start + 2,
                has_dup,
                max_needed,
                count: 1,
                tok_head: nid,
                tok_tail: nid,
            });
            unsafe {
                *scratch.sig_slots.get_unchecked_mut(pos) = SigSlot { tag, cls_id: cid, epoch };
            }
            if scratch.sig_classes.len() * 4 >= scratch.sig_slots.len() {
                scratch.grow_sig_table();
            }
            if gain >= scratch.buckets.len() {
                scratch.buckets.resize_with(gain + 1, Vec::new);
            }
            scratch.buckets[gain].push(cid);
            if gain > max_gain {
                max_gain = gain;
            }
            continue;
        }

        if !use_overflow && k == 3 {
            let mut q0 = local_q[0];
            let mut q1 = local_q[1];
            let mut q2 = local_q[2];
            if q0 > q1 { std::mem::swap(&mut q0, &mut q1); }
            if q1 > q2 { std::mem::swap(&mut q1, &mut q2); }
            if q0 > q1 { std::mem::swap(&mut q0, &mut q1); }
            let tag = (2u64 << 62) | ((q0 as u64) << 32) | ((q1 as u64) << 16) | (q2 as u64);
            let mut pos = mix_u64(tag) & mask;
            let found_cid = loop {
                let slot = unsafe { *scratch.sig_slots.get_unchecked(pos) };
                if slot.epoch != epoch {
                    break u32::MAX;
                }
                if slot.tag == tag {
                    break slot.cls_id;
                }
                pos = (pos + 1) & mask;
            };
            if found_cid != u32::MAX {
                let c = unsafe { scratch.sig_classes.get_unchecked_mut(found_cid as usize) };
                if c.count < c.max_needed {
                    let nid = scratch.tok_nodes.len() as u32;
                    scratch.tok_nodes.push((tok as u32, u32::MAX));
                    unsafe { scratch.tok_nodes.get_unchecked_mut(c.tok_tail as usize).1 = nid };
                    c.tok_tail = nid;
                    c.count += 1;
                }
                continue;
            }
            let c0 = unsafe { *caps.get_unchecked(q0 as usize) };
            let c1 = unsafe { *caps.get_unchecked(q1 as usize) };
            let c2 = unsafe { *caps.get_unchecked(q2 as usize) };
            let (gain, has_dup, max_needed) = if q0 != q1 && q1 != q2 {
                (3usize, false, c0.max(c1).max(c2))
            } else if q0 == q1 && q1 == q2 {
                (3u32.min(c0) as usize, true, c0.div_ceil(3))
            } else if q0 == q1 {
                (2u32.min(c0) as usize + 1, true, c0.div_ceil(2).max(c2))
            } else {
                (1 + 2u32.min(c1) as usize, true, c0.max(c1.div_ceil(2)))
            };
            let sig_start = scratch.sig_qfids.len() as u32;
            scratch.sig_qfids.push(q0);
            scratch.sig_qfids.push(q1);
            scratch.sig_qfids.push(q2);
            let nid = scratch.tok_nodes.len() as u32;
            scratch.tok_nodes.push((tok as u32, u32::MAX));
            let cid = scratch.sig_classes.len() as u32;
            scratch.sig_classes.push(SigClass {
                sig_start,
                sig_end: sig_start + 3,
                has_dup,
                max_needed,
                count: 1,
                tok_head: nid,
                tok_tail: nid,
            });
            unsafe {
                *scratch.sig_slots.get_unchecked_mut(pos) = SigSlot { tag, cls_id: cid, epoch };
            }
            if scratch.sig_classes.len() * 4 >= scratch.sig_slots.len() {
                scratch.grow_sig_table();
            }
            if gain >= scratch.buckets.len() {
                scratch.buckets.resize_with(gain + 1, Vec::new);
            }
            scratch.buckets[gain].push(cid);
            if gain > max_gain {
                max_gain = gain;
            }
            continue;
        }

        if !use_overflow && k == 4 && q_len <= 32768 {
            let mut q0 = local_q[0];
            let mut q1 = local_q[1];
            let mut q2 = local_q[2];
            let mut q3 = local_q[3];
            if q0 > q1 { std::mem::swap(&mut q0, &mut q1); }
            if q2 > q3 { std::mem::swap(&mut q2, &mut q3); }
            if q0 > q2 { std::mem::swap(&mut q0, &mut q2); }
            if q1 > q3 { std::mem::swap(&mut q1, &mut q3); }
            if q1 > q2 { std::mem::swap(&mut q1, &mut q2); }
            let tag = (3u64 << 62) | ((q0 as u64) << 45) | ((q1 as u64) << 30) | ((q2 as u64) << 15) | (q3 as u64);
            let mut pos = mix_u64(tag) & mask;
            let found_cid = loop {
                let slot = unsafe { *scratch.sig_slots.get_unchecked(pos) };
                if slot.epoch != epoch {
                    break u32::MAX;
                }
                if slot.tag == tag {
                    break slot.cls_id;
                }
                pos = (pos + 1) & mask;
            };
            if found_cid != u32::MAX {
                let c = unsafe { scratch.sig_classes.get_unchecked_mut(found_cid as usize) };
                if c.count < c.max_needed {
                    let nid = scratch.tok_nodes.len() as u32;
                    scratch.tok_nodes.push((tok as u32, u32::MAX));
                    unsafe { scratch.tok_nodes.get_unchecked_mut(c.tok_tail as usize).1 = nid };
                    c.tok_tail = nid;
                    c.count += 1;
                }
                continue;
            }
            let mut gain = 0usize;
            let mut has_dup = false;
            let mut max_needed = 0u32;
            let arr = [q0, q1, q2, q3];
            let mut i = 0usize;
            while i < 4 {
                let q = arr[i];
                let mut j = i + 1;
                while j < 4 && arr[j] == q {
                    j += 1;
                }
                let mult = (j - i) as u32;
                if mult > 1 {
                    has_dup = true;
                }
                let cap = unsafe { *caps.get_unchecked(q as usize) };
                gain += mult.min(cap) as usize;
                let need = cap.div_ceil(mult);
                if need > max_needed {
                    max_needed = need;
                }
                i = j;
            }
            let sig_start = scratch.sig_qfids.len() as u32;
            scratch.sig_qfids.extend_from_slice(&arr);
            let nid = scratch.tok_nodes.len() as u32;
            scratch.tok_nodes.push((tok as u32, u32::MAX));
            let cid = scratch.sig_classes.len() as u32;
            scratch.sig_classes.push(SigClass {
                sig_start,
                sig_end: sig_start + 4,
                has_dup,
                max_needed,
                count: 1,
                tok_head: nid,
                tok_tail: nid,
            });
            unsafe {
                *scratch.sig_slots.get_unchecked_mut(pos) = SigSlot { tag, cls_id: cid, epoch };
            }
            if scratch.sig_classes.len() * 4 >= scratch.sig_slots.len() {
                scratch.grow_sig_table();
            }
            if gain >= scratch.buckets.len() {
                scratch.buckets.resize_with(gain + 1, Vec::new);
            }
            scratch.buckets[gain].push(cid);
            if gain > max_gain {
                max_gain = gain;
            }
            continue;
        }

        // General path: `k >= 5` (or `k == 4` with `q_len > 32768`)
        let hits: &[u32] = if !use_overflow {
            let s = &mut local_q[..k];
            let mut already_sorted = true;
            for w in s.windows(2) {
                if w[0] > w[1] {
                    already_sorted = false;
                    break;
                }
            }
            if !already_sorted {
                sort_small_u32(s);
            }
            s
        } else {
            overflow_q.sort_unstable();
            &overflow_q
        };

        let tag = hash_sig(hits);
        let mut pos = (tag as usize) & mask;
        let found_cid = loop {
            let slot = unsafe { *scratch.sig_slots.get_unchecked(pos) };
            if slot.epoch != epoch {
                break u32::MAX;
            }
            if slot.tag == tag {
                let c = unsafe { *scratch.sig_classes.get_unchecked(slot.cls_id as usize) };
                if &scratch.sig_qfids[c.sig_start as usize..c.sig_end as usize] == hits {
                    break slot.cls_id;
                }
            }
            pos = (pos + 1) & mask;
        };

        if found_cid != u32::MAX {
            let c = unsafe { scratch.sig_classes.get_unchecked_mut(found_cid as usize) };
            if c.count < c.max_needed {
                let nid = scratch.tok_nodes.len() as u32;
                scratch.tok_nodes.push((tok as u32, u32::MAX));
                unsafe { scratch.tok_nodes.get_unchecked_mut(c.tok_tail as usize).1 = nid };
                c.tok_tail = nid;
                c.count += 1;
            }
            continue;
        }

        let mut gain = 0usize;
        let mut has_dup = false;
        let mut max_needed = 0u32;
        let mut i = 0usize;
        while i < hits.len() {
            let q = unsafe { *hits.get_unchecked(i) };
            let mut j = i + 1;
            while j < hits.len() && unsafe { *hits.get_unchecked(j) } == q {
                j += 1;
            }
            let mult = (j - i) as u32;
            if mult > 1 {
                has_dup = true;
            }
            let cap = unsafe { *caps.get_unchecked(q as usize) };
            gain += mult.min(cap) as usize;
            let need = cap.div_ceil(mult);
            if need > max_needed {
                max_needed = need;
            }
            i = j;
        }

        let sig_start = scratch.sig_qfids.len() as u32;
        scratch.sig_qfids.extend_from_slice(hits);
        let sig_end = scratch.sig_qfids.len() as u32;
        let nid = scratch.tok_nodes.len() as u32;
        scratch.tok_nodes.push((tok as u32, u32::MAX));
        let cid = scratch.sig_classes.len() as u32;
        scratch.sig_classes.push(SigClass {
            sig_start,
            sig_end,
            has_dup,
            max_needed,
            count: 1,
            tok_head: nid,
            tok_tail: nid,
        });
        unsafe {
            *scratch.sig_slots.get_unchecked_mut(pos) = SigSlot { tag, cls_id: cid, epoch };
        }
        if scratch.sig_classes.len() * 4 >= scratch.sig_slots.len() {
            scratch.grow_sig_table();
        }
        if gain >= scratch.buckets.len() {
            scratch.buckets.resize_with(gain + 1, Vec::new);
        }
        scratch.buckets[gain].push(cid);
        if gain > max_gain {
            max_gain = gain;
        }
    }

    run_greedy_sig_into(max_gain, expected_score, scratch, kept)
}

/// Algorithm 1 using `PruneTarget` (public `prune`, `recommend::intersect`, fallback).
pub(crate) fn prune_fast(
    target: &PruneTarget,
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    expected_score: Option<usize>,
    candidates: Option<&[usize]>,
    scratch: &mut PruneScratch,
) -> Pruned {
    let mut tokens = Vec::new();
    let score = prune_fast_into(
        target,
        all_features,
        offsets,
        feat_base,
        expected_score,
        candidates,
        scratch,
        &mut tokens,
    );
    Pruned { tokens, score }
}

/// Reusable-buffer version of `prune_fast` eliminating `kept` allocations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prune_fast_into(
    target: &PruneTarget,
    all_features: &[u32],
    offsets: &[u32],
    feat_base: usize,
    expected_score: Option<usize>,
    candidates: Option<&[usize]>,
    scratch: &mut PruneScratch,
    kept: &mut Vec<usize>,
) -> usize {
    let q_len = target.caps.len();
    if expected_score == Some(0) || q_len == 0 || offsets.len() < 2 {
        kept.clear();
        return 0;
    }

    scratch.rem.clear();
    scratch.rem.extend_from_slice(&target.caps);
    scratch.single_seen.clear();
    scratch.single_seen.resize(q_len, 0u32);
    scratch.hit_qfids.clear();
    scratch.tok_hits.clear();
    scratch.cur_bucket.clear();

    let mut max_gain = 0usize;
    let tok_count = offsets.len() - 1;
    let ascending = candidates.is_none_or(|c| c.windows(2).all(|w| w[0] < w[1]));

    let mut process_token = |tok: usize| {
        let s = feat_base + offsets[tok] as usize;
        let e = feat_base + offsets[tok + 1] as usize;
        let hit_start = scratch.hit_qfids.len() as u32;
        let mut gain = 0usize;
        let mut has_dup = false;
        for &f in &all_features[s..e] {
            let q = target.get(f);
            if q != EMPTY_KEY {
                let cap = target.caps[q as usize];
                let prev = &scratch.hit_qfids[hit_start as usize..];
                let mut dup_cnt = 0u32;
                for &p in prev {
                    if p == q {
                        dup_cnt += 1;
                    }
                }
                if dup_cnt > 0 {
                    has_dup = true;
                }
                if dup_cnt < cap {
                    gain += 1;
                }
                scratch.hit_qfids.push(q);
            }
        }
        if gain == 0 {
            return;
        }
        let hit_end = scratch.hit_qfids.len() as u32;
        if ascending && hit_end - hit_start == 1 {
            let q = scratch.hit_qfids[hit_start as usize] as usize;
            if scratch.single_seen[q] >= target.caps[q] {
                scratch.hit_qfids.truncate(hit_start as usize);
                return;
            }
            scratch.single_seen[q] += 1;
        }
        let ei = scratch.tok_hits.len() as u32;
        scratch.tok_hits.push(TokHits { token: tok as u32, start: hit_start, end: hit_end, has_dup });
        if gain >= scratch.buckets.len() {
            scratch.buckets.resize_with(gain + 1, Vec::new);
        }
        scratch.buckets[gain].push(ei);
        if gain > max_gain {
            max_gain = gain;
        }
    };

    match candidates {
        Some(cands) => {
            for &tok in cands {
                process_token(tok);
            }
        }
        None => {
            for tok in 0..tok_count {
                process_token(tok);
            }
        }
    }

    run_greedy_buckets_into(max_gain, expected_score, ascending, scratch, kept)
}
