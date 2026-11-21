//! Structural featurization (Section 3.2 of the Aroma paper):
//! - Public string-based `featurize()` + feature constructors for conformance tests.
//! - Zero-allocation algebraic 64-bit hashed `featurize_body_hashed()` operating
//!   directly on `Body` arenas for indexing and querying, plus precomputed
//!   `StructSpan` extraction for zero-reparse recommendation rendering.

use xxhash_rust::xxh3::xxh3_64;

use crate::simple::{Body, KW, TREE};
use crate::{Node, Token};

pub const VAR: &str = "#VAR";
const SEP: char = '\x1f';
const MAX_NAME_LENGTH: usize = 48;

pub fn label(children: &[Node]) -> String {
    let mut out = String::new();
    for child in children {
        match child {
            Node::Token(t) if t.keyword => out.push_str(&t.text),
            _ => out.push('#'),
        }
    }
    out
}

pub fn token_feature(name: &str) -> String {
    format!("T{SEP}{name}")
}

pub fn parent_feature(name: &str, index: usize, ancestor_label: &str) -> String {
    format!("P{SEP}{name}{SEP}{index}{SEP}{ancestor_label}")
}

pub fn sibling_feature(previous: &str, next: &str) -> String {
    format!("S{SEP}{previous}{SEP}{next}")
}

pub fn usage_feature(previous_context: &str, next_context: &str) -> String {
    format!("U{SEP}{previous_context}{SEP}{next_context}")
}

pub fn position_context(index: usize, parent_label: &str) -> String {
    format!("{index}:{parent_label}")
}

pub fn token_context(name: &str) -> String {
    format!(".{name}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFeatures {
    pub token: Token,
    pub features: Vec<String>,
}

/// Clips `text` to `MAX_NAME_LENGTH` UTF-16 code units, matching JavaScript's
/// `text.length <= 48 ? text : `${text.slice(0, 48)}\x1f${text.length}``.
fn clip_str(text: &str) -> std::borrow::Cow<'_, str> {
    if text.is_ascii() {
        if text.len() <= MAX_NAME_LENGTH {
            return std::borrow::Cow::Borrowed(text);
        }
        return std::borrow::Cow::Owned(format!("{}{SEP}{}", &text[..MAX_NAME_LENGTH], text.len()));
    }
    let mut u16_len = 0usize;
    let mut cut_byte = None;
    for (bi, ch) in text.char_indices() {
        let w = ch.len_utf16();
        if u16_len + w > MAX_NAME_LENGTH && cut_byte.is_none() {
            cut_byte = Some(bi);
        }
        u16_len += w;
    }
    if u16_len <= MAX_NAME_LENGTH {
        return std::borrow::Cow::Borrowed(text);
    }
    let cut = cut_byte.unwrap_or(text.len());
    let prefix = if text[..cut].chars().map(char::len_utf16).sum::<usize>() == MAX_NAME_LENGTH {
        text[..cut].to_string()
    } else {
        let u16s: Vec<u16> = text.encode_utf16().take(MAX_NAME_LENGTH).collect();
        String::from_utf16_lossy(&u16s)
    };
    std::borrow::Cow::Owned(format!("{prefix}{SEP}{u16_len}"))
}

// ============================================================================
// Algebraic 64-bit hashing primitives (zero string allocation on hot path)
// ============================================================================

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let t = (a as u128).wrapping_mul(b as u128);
    (t as u64) ^ ((t >> 64) as u64)
}

#[inline(always)]
fn mix(h: u64, x: u64) -> u64 {
    mum(h ^ 0xa076_1d64_78bd_642f, x ^ 0xe703_7ed1_a0b4_28db)
}

#[inline(always)]
fn hash_bytes(b: &[u8]) -> u64 {
    let len = b.len();
    if len <= 8 {
        let mut buf = [0u8; 8];
        match len {
            0 => return 0x9e37_79b9_7f4a_7c15,
            1 => buf[0] = b[0],
            2 => {
                buf[0] = b[0];
                buf[1] = b[1];
            }
            3 => {
                buf[0] = b[0];
                buf[1] = b[1];
                buf[2] = b[2];
            }
            4 => buf[..4].copy_from_slice(b),
            5..=7 => buf[..len].copy_from_slice(b),
            8 => buf.copy_from_slice(b),
            _ => unreachable!(),
        }
        return mix(u64::from_le_bytes(buf), len as u64);
    }
    xxh3_64(b)
}

const TAG_T: u64 = 0x5400_0000_0000_0001;
const TAG_P: u64 = 0x5000_0000_0000_0002;
const TAG_S: u64 = 0x5300_0000_0000_0003;
const TAG_U: u64 = 0x5500_0000_0000_0004;
const TAG_POS: u64 = 0x5059_0000_0000_0005;
const TAG_TOKCTX: u64 = 0x544b_0000_0000_0006;
const VAR_HASH: u64 = 0x5641_5200_0000_0007;
const LABEL_SEED: u64 = 0x6c61_6265_6c00_0008;
const LABEL_NONKW: u64 = 0x2323_2323_2323_2323;
const EMPTY_LABEL: u64 = 0x6c61_6265_6c00_0009;

#[inline(always)]
fn label_step(h: u64, x: u64) -> u64 {
    (h.rotate_left(7) ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[inline(always)]
fn clipped_name_hash(bytes: &[u8], raw: u64) -> u64 {
    if bytes.len() <= MAX_NAME_LENGTH {
        return raw;
    }
    clip_hash_slow(bytes, raw)
}

#[cold]
fn clip_hash_slow(bytes: &[u8], raw: u64) -> u64 {
    if bytes.is_ascii() {
        let h = hash_bytes(&bytes[..MAX_NAME_LENGTH]);
        return mix(mix(h, 0x434c_4950), bytes.len() as u64);
    }
    let s = std::str::from_utf8(bytes).unwrap_or("");
    let units: usize = s.chars().map(char::len_utf16).sum();
    if units <= MAX_NAME_LENGTH {
        return raw;
    }
    let c = clip_str(s);
    let prefix_bytes = c.as_bytes();
    let sep_pos = prefix_bytes.iter().rposition(|&b| b == 0x1f).unwrap_or(prefix_bytes.len());
    let h = hash_bytes(&prefix_bytes[..sep_pos]);
    mix(mix(h, 0x434c_4950), units as u64)
}

// ============================================================================
// Public string-based `featurize`
// ============================================================================

struct PubFrame<'a> {
    children: &'a [Node],
    label: String,
    index: usize,
    parent: Option<usize>,
    member_name: Option<Option<String>>,
}

struct PubLeaf<'a> {
    token: &'a Token,
    name: String,
    index: usize,
    parent: Option<usize>,
}

fn first_other_name_node(children: &[Node]) -> Option<String> {
    for child in children {
        match child {
            Node::Tree(cs) => {
                if let Some(found) = first_other_name_node(cs) {
                    return Some(found);
                }
            }
            Node::Token(t) if !t.keyword && !t.local => {
                return Some(clip_str(&t.text).into_owned());
            }
            _ => {}
        }
    }
    None
}

pub fn featurize(
    root: &Node,
    member_access_labels: &[&str],
    select: Option<&dyn Fn(&Token) -> bool>,
) -> Vec<TokenFeatures> {
    use rustc_hash::FxHashMap;
    let mut frames: Vec<PubFrame<'_>> = Vec::new();
    let mut leaves: Vec<PubLeaf<'_>> = Vec::new();

    fn visit<'a>(
        node: &'a Node,
        index: usize,
        parent: Option<usize>,
        frames: &mut Vec<PubFrame<'a>>,
        leaves: &mut Vec<PubLeaf<'a>>,
    ) {
        match node {
            Node::Token(t) => {
                if !t.keyword {
                    let name = if t.local {
                        VAR.to_string()
                    } else {
                        clip_str(&t.text).into_owned()
                    };
                    leaves.push(PubLeaf { token: t, name, index, parent });
                }
            }
            Node::Tree(children) => {
                let fid = frames.len();
                frames.push(PubFrame {
                    children,
                    label: label(children),
                    index,
                    parent,
                    member_name: None,
                });
                let mut pos = 0usize;
                for child in children {
                    let is_nk = match child {
                        Node::Tree(_) => true,
                        Node::Token(t) => !t.keyword,
                    };
                    if is_nk {
                        pos += 1;
                        visit(child, pos, Some(fid), frames, leaves);
                    }
                }
            }
        }
    }

    visit(root, 1, None, &mut frames, &mut leaves);

    let chosen: Vec<PubLeaf<'_>> = match select {
        Some(pred) => leaves.into_iter().filter(|l| pred(l.token)).collect(),
        None => leaves,
    };

    let mut result: Vec<TokenFeatures> = Vec::with_capacity(chosen.len());
    for leaf in &chosen {
        let mut feats = Vec::with_capacity(8);
        feats.push(token_feature(&leaf.name));
        let mut idx = leaf.index;
        let mut anc = leaf.parent;
        for _ in 0..3 {
            let Some(aid) = anc else { break };
            let f = &frames[aid];
            feats.push(parent_feature(&leaf.name, idx, &f.label));
            idx = f.index;
            anc = f.parent;
        }
        result.push(TokenFeatures { token: leaf.token.clone(), features: feats });
    }

    let mut context_of = |leaf: &PubLeaf<'_>| -> String {
        let Some(pid) = leaf.parent else {
            return position_context(leaf.index, "");
        };
        let is_ma = member_access_labels.iter().any(|&m| m == frames[pid].label);
        if is_ma {
            if frames[pid].member_name.is_none() {
                let mn = first_other_name_node(frames[pid].children);
                frames[pid].member_name = Some(mn);
            }
            if let Some(Some(member)) = &frames[pid].member_name {
                return token_context(member);
            }
        }
        position_context(leaf.index, &frames[pid].label)
    };

    let mut last_use: FxHashMap<&str, usize> = FxHashMap::default();
    for pos in 0..chosen.len() {
        if pos + 1 < chosen.len() {
            let pair = sibling_feature(&chosen[pos].name, &chosen[pos + 1].name);
            result[pos].features.push(pair.clone());
            result[pos + 1].features.push(pair);
        }
        if !chosen[pos].token.local {
            continue;
        }
        let raw = chosen[pos].token.text.as_str();
        if let Some(&prev) = last_use.get(raw) {
            let prev_ctx = context_of(&chosen[prev]);
            let cur_ctx = context_of(&chosen[pos]);
            let pair = usage_feature(&prev_ctx, &cur_ctx);
            result[prev].features.push(pair.clone());
            result[pos].features.push(pair);
        }
        last_use.insert(raw, pos);
    }

    result
}

// ============================================================================
// Fast zero-allocation 64-bit hashed featurizer + StructSpan extractor on `Body`
// ============================================================================

/// A precomputed structural-line witness: if any kept token index lies in
/// `[tok_start, tok_end)`, then `line` is marked `LineRole::Structure` (unless
/// already `Query` or `Added`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StructSpan {
    pub tok_start: u32,
    pub tok_end: u32,
    pub line: u32,
}

#[derive(Clone, Copy)]
struct FastFrame {
    body_item: u32,
    label_hash: u64,
    index: u32,
    parent: u32, // u32::MAX if none
    is_member_access: bool,
    /// Member context hash cached lazily: 0 = uncomputed, 1 = none, 2 = cached in `member_ctx_hash`
    member_state: u8,
    member_ctx_hash: u64,
}

#[derive(Clone, Copy)]
struct FastLeaf {
    body_item: u32,
    name_hash: u64,
    raw_hash: u64,
    index: u32,
    parent: u32,
    local: bool,
    line: u32,
    end_line: u32,
}

#[derive(Clone, Copy)]
struct LastUseSlot {
    hash: u64,
    leaf_idx: u32,
    epoch: u32,
}

#[derive(Clone, Copy)]
struct MemberLabel {
    len: u8,
    bytes: [u8; 16],
}

#[derive(Default)]
pub(crate) struct FeatScratch {
    frames: Vec<FastFrame>,
    leaves: Vec<FastLeaf>,
    counts: Vec<u32>,
    extra: Vec<(u32, u64)>,
    own_hashes: Vec<u64>,
    own_lens: Vec<u8>,
    anc_lines: Vec<u32>,
    lu_slots: Vec<LastUseSlot>,
    lu_epoch: u32,
    lu_used: usize,
    member_labels: Vec<MemberLabel>,
    member_labels_ptr: usize,
    max_member_len: usize,
}

impl FeatScratch {
    fn set_member_labels(&mut self, labels: &[&str]) {
        let ptr = labels.as_ptr() as usize;
        if self.member_labels_ptr == ptr && !self.member_labels.is_empty() {
            return;
        }
        self.member_labels_ptr = ptr;
        self.member_labels.clear();
        let mut max_len = 0usize;
        for &s in labels {
            let b = s.as_bytes();
            if b.len() <= 16 {
                let mut arr = [0u8; 16];
                arr[..b.len()].copy_from_slice(b);
                self.member_labels.push(MemberLabel { len: b.len() as u8, bytes: arr });
                max_len = max_len.max(b.len());
            }
        }
        self.max_member_len = max_len;
    }

    #[inline]
    fn reset_last_use(&mut self, max_locals_hint: usize) {
        let min_cap = (max_locals_hint.min(256) * 2).max(64).next_power_of_two();
        if self.lu_slots.len() < min_cap {
            self.lu_slots.resize(min_cap, LastUseSlot { hash: 0, leaf_idx: 0, epoch: 0 });
        }
        self.lu_epoch = self.lu_epoch.wrapping_add(1);
        if self.lu_epoch == 0 {
            for s in &mut self.lu_slots {
                s.epoch = 0;
            }
            self.lu_epoch = 1;
        }
        self.lu_used = 0;
    }

    #[inline]
    fn lu_replace(
        &mut self,
        body: &Body,
        base: &[u8],
        raw_hash: u64,
        raw_bytes: &[u8],
        cur_idx: u32,
    ) -> Option<usize> {
        if self.lu_used * 2 >= self.lu_slots.len() {
            self.grow_last_use();
        }
        let mask = self.lu_slots.len() - 1;
        let epoch = self.lu_epoch;
        let mut pos = (raw_hash as usize) & mask;
        loop {
            let s = unsafe { *self.lu_slots.get_unchecked(pos) };
            if s.epoch != epoch {
                unsafe {
                    *self.lu_slots.get_unchecked_mut(pos) =
                        LastUseSlot { hash: raw_hash, leaf_idx: cur_idx, epoch };
                }
                self.lu_used += 1;
                return None;
            }
            if s.hash == raw_hash {
                let prev_item = unsafe { self.leaves.get_unchecked(s.leaf_idx as usize).body_item };
                let prev_bytes = body.text(base, unsafe { body.items.get_unchecked(prev_item as usize) });
                if prev_bytes == raw_bytes {
                    let prev = s.leaf_idx as usize;
                    unsafe { self.lu_slots.get_unchecked_mut(pos).leaf_idx = cur_idx };
                    return Some(prev);
                }
            }
            pos = (pos + 1) & mask;
        }
    }

    #[cold]
    fn grow_last_use(&mut self) {
        let new_cap = (self.lu_slots.len() * 2).max(64);
        let mask = new_cap - 1;
        let epoch = self.lu_epoch;
        let mut new_slots = vec![LastUseSlot { hash: 0, leaf_idx: 0, epoch: 0 }; new_cap];
        for s in &self.lu_slots {
            if s.epoch == epoch {
                let mut pos = (s.hash as usize) & mask;
                while new_slots[pos].epoch == epoch {
                    pos = (pos + 1) & mask;
                }
                new_slots[pos] = *s;
            }
        }
        self.lu_slots = new_slots;
    }
}

#[inline(always)]
fn push_tok_line(
    line: u32,
    end_line: u32,
    out_token_lines: &mut Vec<u32>,
    out_token_end_delta: &mut Vec<u8>,
    out_end_overflow: &mut Vec<(u32, u32)>,
    tok_idx: u32,
) {
    out_token_lines.push(line);
    let d = end_line.saturating_sub(line);
    if d < 255 {
        out_token_end_delta.push(d as u8);
    } else {
        out_token_end_delta.push(255);
        out_end_overflow.push((tok_idx, d));
    }
}

/// Output of `extract_location_and_spans` (for duplicate methods where we skip featurization!).
pub(crate) fn extract_location_and_spans(
    body: &Body,
    root: u32,
    token_lines: &mut Vec<u32>,
    token_end_delta: &mut Vec<u8>,
    end_overflow: &mut Vec<(u32, u32)>,
    struct_spans: &mut Vec<StructSpan>,
    scratch: &mut FeatScratch,
) -> usize {
    let tok_base = token_lines.len();
    scratch.anc_lines.clear();

    #[allow(clippy::too_many_arguments)]
    fn walk(
        body: &Body,
        node: u32,
        tok_base: usize,
        token_lines: &mut Vec<u32>,
        token_end_delta: &mut Vec<u8>,
        end_overflow: &mut Vec<(u32, u32)>,
        struct_spans: &mut Vec<StructSpan>,
        anc_lines: &mut Vec<u32>,
    ) {
        let it = unsafe { body.items.get_unchecked(node as usize) };
        if it.kind != TREE {
            if it.kind != KW {
                let tok_idx = (token_lines.len() - tok_base) as u32;
                push_tok_line(it.line, it.end_line, token_lines, token_end_delta, end_overflow, tok_idx);
            }
            return;
        }
        let tok_start = (token_lines.len() - tok_base) as u32;
        let anc_mark = anc_lines.len();
        let children = body.children(it);
        for &cid in children {
            let cit = unsafe { body.items.get_unchecked(cid as usize) };
            if cit.kind == KW {
                let ln = cit.line;
                if !anc_lines.contains(&ln) {
                    anc_lines.push(ln);
                }
            }
        }
        let added_end = anc_lines.len();
        let span_mark = struct_spans.len();
        for &ln in &anc_lines[anc_mark..added_end] {
            struct_spans.push(StructSpan { tok_start, tok_end: 0, line: ln });
        }
        for &cid in children {
            if unsafe { body.items.get_unchecked(cid as usize).kind } != KW {
                walk(
                    body,
                    cid,
                    tok_base,
                    token_lines,
                    token_end_delta,
                    end_overflow,
                    struct_spans,
                    anc_lines,
                );
            }
        }
        let tok_end = (token_lines.len() - tok_base) as u32;
        if tok_end == tok_start {
            struct_spans.truncate(span_mark);
        } else {
            let own_count = added_end - anc_mark;
            for sp in &mut struct_spans[span_mark..span_mark + own_count] {
                sp.tok_end = tok_end;
            }
        }
        anc_lines.truncate(anc_mark);
    }

    walk(
        body,
        root,
        tok_base,
        token_lines,
        token_end_delta,
        end_overflow,
        struct_spans,
        &mut scratch.anc_lines,
    );
    token_lines.len() - tok_base
}

fn first_other_name_hash(body: &Body, base: &[u8], tree_item: u32) -> Option<u64> {
    let children = body.children(&body.items[tree_item as usize]);
    for &cid in children {
        let cit = &body.items[cid as usize];
        if cit.kind == TREE {
            if let Some(h) = first_other_name_hash(body, base, cid) {
                return Some(h);
            }
        } else if cit.kind != KW && !cit.local {
            let bytes = body.text(base, cit);
            return Some(clipped_name_hash(bytes, hash_bytes(bytes)));
        }
    }
    None
}

#[inline]
fn context_hash_of(
    body: &Body,
    base: &[u8],
    leaf: &FastLeaf,
    frames: &mut [FastFrame],
) -> u64 {
    if leaf.parent == u32::MAX {
        return mix(mix(TAG_POS, leaf.index as u64), EMPTY_LABEL);
    }
    let pid = leaf.parent as usize;
    if frames[pid].is_member_access {
        if frames[pid].member_state == 0 {
            if let Some(nh) = first_other_name_hash(body, base, frames[pid].body_item) {
                frames[pid].member_state = 2;
                frames[pid].member_ctx_hash = mix(TAG_TOKCTX, nh);
            } else {
                frames[pid].member_state = 1;
            }
        }
        if frames[pid].member_state == 2 {
            return frames[pid].member_ctx_hash;
        }
    }
    mix(mix(TAG_POS, leaf.index as u64), frames[pid].label_hash)
}

/// Featurizes a method's `Body` directly into 64-bit feature hashes (`hashes` +
/// CSR `offsets` relative to the start of this method), plus `token_lines`,
/// `token_end_delta`, `end_overflow` (relative token indices `0..n`), and `struct_spans`.
///
/// Returns the number of non-keyword tokens (0 if empty).
#[allow(clippy::too_many_arguments)]
pub(crate) fn featurize_body_hashed(
    body: &Body,
    base: &[u8],
    root: u32,
    member_access_labels: &[&str],
    select: Option<&mut dyn FnMut(u32, u32) -> bool>,
    scratch: &mut FeatScratch,
    out_hashes: &mut Vec<u64>,
    out_offsets: &mut Vec<u32>,
    out_token_lines: &mut Vec<u32>,
    out_token_end_delta: &mut Vec<u8>,
    out_end_overflow: &mut Vec<(u32, u32)>,
    out_struct_spans: Option<&mut Vec<StructSpan>>,
) -> usize {
    scratch.set_member_labels(member_access_labels);
    scratch.frames.clear();
    scratch.leaves.clear();
    scratch.anc_lines.clear();

    let want_spans = out_struct_spans.is_some();
    let mut spans_opt = out_struct_spans;

    #[allow(clippy::too_many_arguments)]
    fn visit(
        body: &Body,
        base: &[u8],
        node: u32,
        index: u32,
        parent: u32,
        want_spans: bool,
        scratch: &mut FeatScratch,
        spans_opt: &mut Option<&mut Vec<StructSpan>>,
    ) {
        let it = unsafe { body.items.get_unchecked(node as usize) };
        if it.kind != TREE {
            if it.kind != KW {
                let bytes = body.text(base, it);
                let raw_hash = hash_bytes(bytes);
                let name_hash = if it.local {
                    VAR_HASH
                } else {
                    clipped_name_hash(bytes, raw_hash)
                };
                scratch.leaves.push(FastLeaf {
                    body_item: node,
                    name_hash,
                    raw_hash,
                    index,
                    parent,
                    local: it.local,
                    line: it.line,
                    end_line: it.end_line,
                });
            }
            return;
        }
        let fid = scratch.frames.len() as u32;
        let children = body.children(it);
        let anc_mark = scratch.anc_lines.len();
        let tok_start = scratch.leaves.len() as u32;

        let mut lbl = LABEL_SEED;
        let mut m_buf = [0u8; 16];
        let mut m_len = 0usize;
        let mut m_ok = scratch.max_member_len > 0 && children.len() <= scratch.max_member_len;

        for &cid in children {
            let cit = unsafe { body.items.get_unchecked(cid as usize) };
            if cit.kind == KW {
                let kb = body.text(base, cit);
                lbl = label_step(lbl, hash_bytes(kb));
                if want_spans {
                    let ln = cit.line;
                    if !scratch.anc_lines.contains(&ln) {
                        scratch.anc_lines.push(ln);
                    }
                }
                if m_ok {
                    if m_len + kb.len() <= 16 {
                        m_buf[m_len..m_len + kb.len()].copy_from_slice(kb);
                        m_len += kb.len();
                    } else {
                        m_ok = false;
                    }
                }
            } else {
                lbl = label_step(lbl, LABEL_NONKW);
                if m_ok {
                    if m_len < 16 {
                        m_buf[m_len] = b'#';
                        m_len += 1;
                    } else {
                        m_ok = false;
                    }
                }
            }
        }
        let label_hash = mix(lbl, children.len() as u64);
        let is_member_access = m_ok && {
            let cand = &m_buf[..m_len];
            scratch
                .member_labels
                .iter()
                .any(|ml| &ml.bytes[..ml.len as usize] == cand)
        };

        scratch.frames.push(FastFrame {
            body_item: node,
            label_hash,
            index,
            parent,
            is_member_access,
            member_state: 0,
            member_ctx_hash: 0,
        });

        let added_end = scratch.anc_lines.len();
        let span_mark = spans_opt.as_ref().map_or(0, |v| v.len());
        if let Some(spans) = spans_opt.as_deref_mut() {
            for &ln in &scratch.anc_lines[anc_mark..added_end] {
                spans.push(StructSpan { tok_start, tok_end: 0, line: ln });
            }
        }

        let mut pos = 0u32;
        for &cid in children {
            if unsafe { body.items.get_unchecked(cid as usize).kind } != KW {
                pos += 1;
                visit(body, base, cid, pos, fid, want_spans, scratch, spans_opt);
            }
        }

        if let Some(spans) = spans_opt.as_deref_mut() {
            let tok_end = scratch.leaves.len() as u32;
            if tok_end == tok_start {
                spans.truncate(span_mark);
            } else {
                let own_count = added_end - anc_mark;
                for sp in &mut spans[span_mark..span_mark + own_count] {
                    sp.tok_end = tok_end;
                }
            }
        }
        scratch.anc_lines.truncate(anc_mark);
    }

    visit(body, base, root, 1, u32::MAX, want_spans, scratch, &mut spans_opt);

    if let Some(pred) = select {
        scratch.leaves.retain(|l| pred(l.line, l.end_line));
    }
    let n = scratch.leaves.len();
    if n == 0 {
        return 0;
    }

    let hash_t_var = mix(TAG_T, VAR_HASH);
    let hash_s_var_var = mix(mix(TAG_S, VAR_HASH), VAR_HASH);

    out_token_lines.reserve(n);
    out_token_end_delta.reserve(n);
    for (ti, l) in scratch.leaves.iter().enumerate() {
        push_tok_line(
            l.line,
            l.end_line,
            out_token_lines,
            out_token_end_delta,
            out_end_overflow,
            ti as u32,
        );
    }

    scratch.counts.clear();
    scratch.counts.resize(n, 0);
    scratch.own_hashes.clear();
    scratch.own_hashes.reserve(n * 4);
    scratch.own_lens.clear();
    scratch.own_lens.reserve(n);
    scratch.extra.clear();
    scratch.extra.reserve(n * 3);

    // Own features: `T` + up to 3 `P` ancestors
    let mut local_count = 0usize;
    for i in 0..n {
        let leaf = unsafe { *scratch.leaves.get_unchecked(i) };
        let nh = leaf.name_hash;
        if leaf.local {
            local_count += 1;
            scratch.own_hashes.push(hash_t_var);
        } else {
            scratch.own_hashes.push(mix(TAG_T, nh));
        }
        let mut own_c = 1u8;
        let p_base = mix(TAG_P, nh);
        let mut idx = leaf.index;
        let mut anc = leaf.parent;
        for _ in 0..3 {
            if anc == u32::MAX {
                break;
            }
            let f = unsafe { *scratch.frames.get_unchecked(anc as usize) };
            scratch.own_hashes.push(mix(mix(p_base, idx as u64), f.label_hash));
            own_c += 1;
            idx = f.index;
            anc = f.parent;
        }
        scratch.own_lens.push(own_c);
        scratch.counts[i] = own_c as u32;
    }

    // Sibling features `S`
    for i in 0..n.saturating_sub(1) {
        let l0 = unsafe { *scratch.leaves.get_unchecked(i) };
        let l1 = unsafe { *scratch.leaves.get_unchecked(i + 1) };
        let h = if l0.local && l1.local {
            hash_s_var_var
        } else {
            mix(mix(TAG_S, l0.name_hash), l1.name_hash)
        };
        scratch.extra.push((i as u32, h));
        scratch.extra.push(((i + 1) as u32, h));
        scratch.counts[i] += 1;
        scratch.counts[i + 1] += 1;
    }

    // Usage features `U` (zero-alloc epoch table)
    if local_count >= 2 {
        scratch.reset_last_use(local_count);
        for i in 0..n {
            let leaf = unsafe { *scratch.leaves.get_unchecked(i) };
            if !leaf.local {
                continue;
            }
            let raw_bytes = body.text(base, unsafe { body.items.get_unchecked(leaf.body_item as usize) });
            if let Some(prev) = scratch.lu_replace(body, base, leaf.raw_hash, raw_bytes, i as u32) {
                let prev_leaf = unsafe { *scratch.leaves.get_unchecked(prev) };
                let ca = context_hash_of(body, base, &prev_leaf, &mut scratch.frames);
                let cb = context_hash_of(body, base, &leaf, &mut scratch.frames);
                let h = mix(mix(TAG_U, ca), cb);
                scratch.extra.push((prev as u32, h));
                scratch.extra.push((i as u32, h));
                scratch.counts[prev] += 1;
                scratch.counts[i] += 1;
            }
        }
    }

    // Pack into CSR (`out_offsets`, `out_hashes`)
    let hash_base = out_hashes.len();
    let mut total = 0usize;
    let off_base = out_offsets.len();
    out_offsets.reserve(n + 1);
    for &c in &scratch.counts[..n] {
        out_offsets.push(total as u32);
        total += c as usize;
    }
    out_offsets.push(total as u32);
    out_hashes.resize(hash_base + total, 0u64);

    let mut own_pos = 0usize;
    for i in 0..n {
        let dst = hash_base + out_offsets[off_base + i] as usize;
        let len = scratch.own_lens[i] as usize;
        out_hashes[dst..dst + len].copy_from_slice(&scratch.own_hashes[own_pos..own_pos + len]);
        own_pos += len;
        scratch.counts[i] = (dst + len) as u32;
    }
    for &(tok, h) in &scratch.extra {
        let p = scratch.counts[tok as usize] as usize;
        unsafe { *out_hashes.get_unchecked_mut(p) = h };
        scratch.counts[tok as usize] = (p + 1) as u32;
    }

    n
}
