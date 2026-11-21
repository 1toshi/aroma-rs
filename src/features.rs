//! Structural featurization (Section 3.2 of the Aroma paper):
//! - Public string-based `featurize()` + feature constructors for conformance tests.
//! - Zero-allocation 64-bit hashed `featurize_body_hashed()` operating directly on
//!   `Body` arenas for indexing and querying, plus precomputed `StructSpan`
//!   extraction for zero-reparse recommendation rendering.

use rustc_hash::FxHashMap;
use xxhash_rust::xxh3::xxh3_64;

use crate::simple::{Body, KW, TREE};
use crate::{Node, Token};

pub const VAR: &str = "#VAR";
const SEP: char = '\x1f';
const SEP_BYTE: u8 = 0x1f;
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

#[inline]
fn write_clipped(bytes: &[u8], out: &mut Vec<u8>) {
    if bytes.is_ascii() {
        if bytes.len() <= MAX_NAME_LENGTH {
            out.extend_from_slice(bytes);
        } else {
            out.extend_from_slice(&bytes[..MAX_NAME_LENGTH]);
            out.push(SEP_BYTE);
            write_usize(bytes.len(), out);
        }
        return;
    }
    let s = String::from_utf8_lossy(bytes);
    let c = clip_str(&s);
    out.extend_from_slice(c.as_bytes());
}

#[inline(always)]
fn write_usize(mut v: usize, out: &mut Vec<u8>) {
    if v < 10 {
        out.push(b'0' + v as u8);
        return;
    }
    if v < 100 {
        out.push(b'0' + (v / 10) as u8);
        out.push(b'0' + (v % 10) as u8);
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = 20;
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    out.extend_from_slice(&buf[i..]);
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
// Fast zero-String 64-bit hashed featurizer + StructSpan extractor on `Body`
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

struct FastFrame {
    body_item: u32,
    label_start: u32,
    label_end: u32,
    index: u32,
    parent: u32, // u32::MAX if none
    /// Member name cached lazily: 0 = uncomputed, 1 = none, 2 = range in scratch.str_buf
    member_state: u8,
    member_start: u32,
    member_end: u32,
}

#[derive(Clone, Copy)]
struct FastLeaf {
    body_item: u32,
    /// Range in `scratch.str_buf` (already clipped or `#VAR`)
    name_start: u32,
    name_end: u32,
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

#[derive(Default)]
pub(crate) struct FeatScratch {
    frames: Vec<FastFrame>,
    leaves: Vec<FastLeaf>,
    str_buf: Vec<u8>,
    fmt_buf: Vec<u8>,
    ctx_a: Vec<u8>,
    ctx_b: Vec<u8>,
    counts: Vec<u32>,
    extra: Vec<(u32, u64)>,
    own_hashes: Vec<u64>,
    own_lens: Vec<u8>,
    anc_lines: Vec<u32>,
    // Zero-alloc epoch-stamped `last_use` table (replaces per-method FxHashMap!)
    lu_slots: Vec<LastUseSlot>,
    lu_epoch: u32,
    lu_used: usize,
}

impl FeatScratch {
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
        raw_bytes: &[u8],
        cur_idx: u32,
    ) -> Option<usize> {
        if self.lu_used * 2 >= self.lu_slots.len() {
            self.grow_last_use();
        }
        let h = xxh3_64(raw_bytes);
        let mask = self.lu_slots.len() - 1;
        let epoch = self.lu_epoch;
        let mut pos = (h as usize) & mask;
        loop {
            let s = unsafe { *self.lu_slots.get_unchecked(pos) };
            if s.epoch != epoch {
                unsafe {
                    *self.lu_slots.get_unchecked_mut(pos) =
                        LastUseSlot { hash: h, leaf_idx: cur_idx, epoch };
                }
                self.lu_used += 1;
                return None;
            }
            if s.hash == h {
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

/// Output of `extract_location_and_spans` (for duplicate methods where we skip featurization!).
pub(crate) fn extract_location_and_spans(
    body: &Body,
    root: u32,
    token_lines: &mut Vec<u32>,
    token_end_lines: &mut Vec<u32>,
    struct_spans: &mut Vec<StructSpan>,
    scratch: &mut FeatScratch,
) -> usize {
    let tok_base = token_lines.len();
    scratch.anc_lines.clear();

    fn walk(
        body: &Body,
        node: u32,
        tok_base: usize,
        token_lines: &mut Vec<u32>,
        token_end_lines: &mut Vec<u32>,
        struct_spans: &mut Vec<StructSpan>,
        anc_lines: &mut Vec<u32>,
    ) {
        let it = &body.items[node as usize];
        if it.kind != TREE {
            if it.kind != KW {
                token_lines.push(it.line);
                token_end_lines.push(it.end_line);
            }
            return;
        }
        let tok_start = (token_lines.len() - tok_base) as u32;
        let anc_mark = anc_lines.len();
        let children = body.children(it);
        for &cid in children {
            let cit = &body.items[cid as usize];
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
            if body.items[cid as usize].kind != KW {
                walk(body, cid, tok_base, token_lines, token_end_lines, struct_spans, anc_lines);
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

    walk(body, root, tok_base, token_lines, token_end_lines, struct_spans, &mut scratch.anc_lines);
    token_lines.len() - tok_base
}

fn first_other_name_body(body: &Body, base: &[u8], tree_item: u32, out: &mut Vec<u8>) -> bool {
    let children = body.children(&body.items[tree_item as usize]);
    for &cid in children {
        let cit = &body.items[cid as usize];
        if cit.kind == TREE {
            if first_other_name_body(body, base, cid, out) {
                return true;
            }
        } else if cit.kind != KW && !cit.local {
            write_clipped(body.text(base, cit), out);
            return true;
        }
    }
    false
}

/// Featurizes a method's `Body` directly into 64-bit feature hashes (`hashes` +
/// CSR `offsets` relative to the start of this method), plus `token_lines`,
/// `token_end_lines`, and `struct_spans`.
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
    out_token_end_lines: &mut Vec<u32>,
    out_struct_spans: Option<&mut Vec<StructSpan>>,
) -> usize {
    scratch.frames.clear();
    scratch.leaves.clear();
    scratch.str_buf.clear();
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
                let name_start = scratch.str_buf.len() as u32;
                if it.local {
                    scratch.str_buf.extend_from_slice(VAR.as_bytes());
                } else {
                    write_clipped(body.text(base, it), &mut scratch.str_buf);
                }
                let name_end = scratch.str_buf.len() as u32;
                scratch.leaves.push(FastLeaf {
                    body_item: node,
                    name_start,
                    name_end,
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
        let label_start = scratch.str_buf.len() as u32;
        let children = body.children(it);
        let anc_mark = scratch.anc_lines.len();
        let tok_start = scratch.leaves.len() as u32;

        for &cid in children {
            let cit = unsafe { body.items.get_unchecked(cid as usize) };
            if cit.kind == KW {
                scratch.str_buf.extend_from_slice(body.text(base, cit));
                if want_spans {
                    let ln = cit.line;
                    if !scratch.anc_lines.contains(&ln) {
                        scratch.anc_lines.push(ln);
                    }
                }
            } else {
                scratch.str_buf.push(b'#');
            }
        }
        let label_end = scratch.str_buf.len() as u32;
        scratch.frames.push(FastFrame {
            body_item: node,
            label_start,
            label_end,
            index,
            parent,
            member_state: 0,
            member_start: 0,
            member_end: 0,
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

    let hash_t_var = xxh3_64(b"T\x1f#VAR");
    let hash_s_var_var = xxh3_64(b"S\x1f#VAR\x1f#VAR");

    out_token_lines.reserve(n);
    out_token_end_lines.reserve(n);
    for l in &scratch.leaves {
        out_token_lines.push(l.line);
        out_token_end_lines.push(l.end_line);
    }

    scratch.counts.clear();
    scratch.counts.resize(n, 0);
    scratch.own_hashes.clear();
    scratch.own_hashes.reserve(n * 4);
    scratch.own_lens.clear();
    scratch.own_lens.reserve(n);
    scratch.extra.clear();
    scratch.extra.reserve(n * 3);

    // Own features: `T\x1f{name}` + up to 3 `P\x1f{name}\x1f{idx}\x1f{anc_label}`
    let mut local_count = 0usize;
    for i in 0..n {
        let leaf = unsafe { *scratch.leaves.get_unchecked(i) };
        let ns = leaf.name_start as usize;
        let ne = leaf.name_end as usize;

        if leaf.local {
            local_count += 1;
            scratch.own_hashes.push(hash_t_var);
            let mut own_c = 1u8;
            let mut idx = leaf.index;
            let mut anc = leaf.parent;
            if anc != u32::MAX {
                scratch.fmt_buf.clear();
                scratch.fmt_buf.extend_from_slice(b"P\x1f#VAR\x1f");
                let prefix_len = scratch.fmt_buf.len();
                for _ in 0..3 {
                    let f = unsafe { scratch.frames.get_unchecked(anc as usize) };
                    scratch.fmt_buf.truncate(prefix_len);
                    write_usize(idx as usize, &mut scratch.fmt_buf);
                    scratch.fmt_buf.push(SEP_BYTE);
                    scratch.fmt_buf.extend_from_slice(
                        &scratch.str_buf[f.label_start as usize..f.label_end as usize],
                    );
                    scratch.own_hashes.push(xxh3_64(&scratch.fmt_buf));
                    own_c += 1;
                    idx = f.index;
                    anc = f.parent;
                    if anc == u32::MAX {
                        break;
                    }
                }
            }
            scratch.own_lens.push(own_c);
            scratch.counts[i] = own_c as u32;
        } else {
            scratch.fmt_buf.clear();
            scratch.fmt_buf.push(b'T');
            scratch.fmt_buf.push(SEP_BYTE);
            scratch.fmt_buf.extend_from_slice(&scratch.str_buf[ns..ne]);
            scratch.own_hashes.push(xxh3_64(&scratch.fmt_buf));
            let mut own_c = 1u8;

            let mut idx = leaf.index;
            let mut anc = leaf.parent;
            if anc != u32::MAX {
                scratch.fmt_buf[0] = b'P';
                scratch.fmt_buf.push(SEP_BYTE);
                let prefix_len = scratch.fmt_buf.len();
                for _ in 0..3 {
                    let f = unsafe { scratch.frames.get_unchecked(anc as usize) };
                    scratch.fmt_buf.truncate(prefix_len);
                    write_usize(idx as usize, &mut scratch.fmt_buf);
                    scratch.fmt_buf.push(SEP_BYTE);
                    scratch.fmt_buf.extend_from_slice(
                        &scratch.str_buf[f.label_start as usize..f.label_end as usize],
                    );
                    scratch.own_hashes.push(xxh3_64(&scratch.fmt_buf));
                    own_c += 1;
                    idx = f.index;
                    anc = f.parent;
                    if anc == u32::MAX {
                        break;
                    }
                }
            }
            scratch.own_lens.push(own_c);
            scratch.counts[i] = own_c as u32;
        }
    }

    // Sibling features `S\x1f{prev}\x1f{next}`
    for i in 0..n.saturating_sub(1) {
        let l0 = unsafe { *scratch.leaves.get_unchecked(i) };
        let l1 = unsafe { *scratch.leaves.get_unchecked(i + 1) };
        let h = if l0.local && l1.local {
            hash_s_var_var
        } else {
            scratch.fmt_buf.clear();
            scratch.fmt_buf.push(b'S');
            scratch.fmt_buf.push(SEP_BYTE);
            scratch.fmt_buf.extend_from_slice(
                &scratch.str_buf[l0.name_start as usize..l0.name_end as usize],
            );
            scratch.fmt_buf.push(SEP_BYTE);
            scratch.fmt_buf.extend_from_slice(
                &scratch.str_buf[l1.name_start as usize..l1.name_end as usize],
            );
            xxh3_64(&scratch.fmt_buf)
        };
        scratch.extra.push((i as u32, h));
        scratch.extra.push(((i + 1) as u32, h));
        scratch.counts[i] += 1;
        scratch.counts[i + 1] += 1;
    }

    // Usage features `U\x1f{prev_ctx}\x1f{cur_ctx}` (zero-alloc epoch table!)
    if local_count >= 2 {
        scratch.reset_last_use(local_count);
        for i in 0..n {
            let leaf = unsafe { *scratch.leaves.get_unchecked(i) };
            if !leaf.local {
                continue;
            }
            let raw_bytes = body.text(base, unsafe { body.items.get_unchecked(leaf.body_item as usize) });
            if let Some(prev) = scratch.lu_replace(body, base, raw_bytes, i as u32) {
                let prev_leaf = unsafe { *scratch.leaves.get_unchecked(prev) };
                write_context(
                    body,
                    base,
                    member_access_labels,
                    &prev_leaf,
                    &mut scratch.frames,
                    &mut scratch.str_buf,
                    &mut scratch.ctx_a,
                );
                write_context(
                    body,
                    base,
                    member_access_labels,
                    &leaf,
                    &mut scratch.frames,
                    &mut scratch.str_buf,
                    &mut scratch.ctx_b,
                );
                scratch.fmt_buf.clear();
                scratch.fmt_buf.push(b'U');
                scratch.fmt_buf.push(SEP_BYTE);
                scratch.fmt_buf.extend_from_slice(&scratch.ctx_a);
                scratch.fmt_buf.push(SEP_BYTE);
                scratch.fmt_buf.extend_from_slice(&scratch.ctx_b);
                let h = xxh3_64(&scratch.fmt_buf);
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

fn write_context(
    body: &Body,
    base: &[u8],
    member_access_labels: &[&str],
    leaf: &FastLeaf,
    frames: &mut [FastFrame],
    str_buf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) {
    out.clear();
    if leaf.parent == u32::MAX {
        write_usize(leaf.index as usize, out);
        out.push(b':');
        return;
    }
    let pid = leaf.parent as usize;
    let ls = frames[pid].label_start as usize;
    let le = frames[pid].label_end as usize;
    let is_ma = member_access_labels.iter().any(|m| m.as_bytes() == &str_buf[ls..le]);
    if is_ma {
        if frames[pid].member_state == 0 {
            let ms = str_buf.len() as u32;
            if first_other_name_body(body, base, frames[pid].body_item, str_buf) {
                frames[pid].member_state = 2;
                frames[pid].member_start = ms;
                frames[pid].member_end = str_buf.len() as u32;
            } else {
                frames[pid].member_state = 1;
            }
        }
        if frames[pid].member_state == 2 {
            out.push(b'.');
            out.extend_from_slice(
                &str_buf[frames[pid].member_start as usize..frames[pid].member_end as usize],
            );
            return;
        }
    }
    write_usize(leaf.index as usize, out);
    out.push(b':');
    out.extend_from_slice(&str_buf[ls..le]);
}
