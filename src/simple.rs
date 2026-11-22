//! Simplified parse trees (Section 3.1) in a compact arena form used by the
//! indexer, plus conversions to and from the public `Node` type.

use crate::lezer::{SyntaxTree, NONE};
use crate::text::{Src, SIDE};
use crate::{Node, Token};

pub(crate) const KW: u8 = 0;
pub(crate) const NAME: u8 = 1;
pub(crate) const TREE: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Item {
    pub kind: u8,
    pub local: bool,
    pub line: u32,
    pub end_line: u32,
    /// Token: text reference (byte range of the source, or of `side` when tagged with `SIDE`).
    /// Tree: range of `kids`.
    pub a: u32,
    pub b: u32,
}

/// An arena of simplified trees. Several trees (e.g. all methods of a file)
/// can share one arena.
#[derive(Default)]
pub(crate) struct Body {
    pub items: Vec<Item>,
    pub kids: Vec<u32>,
    pub side: Vec<u8>,
    scratch: Vec<u32>,
}

const DK1: u64 = 0x9e37_79b9_7f4a_7c15;
const DK2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const DK3: u64 = 0x1656_67b1_9e37_79f9;

#[inline(always)]
fn digest_hash_bytes(b: &[u8]) -> u64 {
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
        let t = ((u64::from_le_bytes(buf) ^ 0xa076_1d64_78bd_642f) as u128)
            .wrapping_mul(((len as u64) ^ 0xe703_7ed1_a0b4_28db) as u128);
        return (t as u64) ^ ((t >> 64) as u64);
    }
    xxhash_rust::xxh3::xxh3_64(b)
}

#[inline(always)]
fn dg_step(d1: &mut u64, d2: &mut u64, x: u64) {
    *d1 = (d1.rotate_left(23) ^ x).wrapping_mul(DK1);
    *d2 = (d2.rotate_left(31) ^ x.wrapping_mul(DK2)).wrapping_mul(DK3);
}

impl Body {
    pub fn clear(&mut self) {
        self.items.clear();
        self.kids.clear();
        self.side.clear();
        self.scratch.clear();
    }

    #[inline(always)]
    pub fn text<'s>(&'s self, base: &'s [u8], it: &Item) -> &'s [u8] {
        if it.a & SIDE != 0 {
            unsafe { self.side.get_unchecked((it.a & !SIDE) as usize..(it.b & !SIDE) as usize) }
        } else {
            unsafe { base.get_unchecked(it.a as usize..it.b as usize) }
        }
    }

    #[inline(always)]
    pub fn children(&self, it: &Item) -> &[u32] {
        unsafe { self.kids.get_unchecked(it.a as usize..it.b as usize) }
    }

    #[inline(always)]
    pub(crate) fn mark(&self) -> usize {
        self.scratch.len()
    }

    #[inline(always)]
    pub(crate) fn push_child(&mut self, id: u32) {
        self.scratch.push(id);
    }

    #[inline(always)]
    pub(crate) fn push_tok(
        &mut self,
        kind: u8,
        local: bool,
        line: u32,
        end_line: u32,
        a: u32,
        b: u32,
    ) -> u32 {
        self.push(Item { kind, local, line, end_line, a, b })
    }

    #[inline]
    pub(crate) fn close_node(&mut self, mark: usize) -> Option<u32> {
        self.close(mark)
    }

    #[inline(always)]
    fn push(&mut self, it: Item) -> u32 {
        let id = self.items.len() as u32;
        self.items.push(it);
        id
    }

    /// Collapses the children above `mark` on the scratch stack into a node
    /// (Section 3.1: no empty lists, no list of a single subtree or name).
    #[inline]
    fn close(&mut self, mark: usize) -> Option<u32> {
        let n = self.scratch.len() - mark;
        if n == 0 {
            return None;
        }
        if n == 1 {
            let only = unsafe { *self.scratch.get_unchecked(mark) };
            let it = unsafe { self.items.get_unchecked(only as usize) };
            if it.kind != KW {
                self.scratch.truncate(mark);
                return Some(only);
            }
        }
        let a = self.kids.len() as u32;
        self.kids.extend_from_slice(&self.scratch[mark..]);
        let b = self.kids.len() as u32;
        self.scratch.truncate(mark);
        Some(self.push(Item { kind: TREE, local: false, line: 0, end_line: 0, a, b }))
    }

    /// The public form of a tree.
    pub fn to_node(&self, base: &[u8], root: u32) -> Node {
        let it = &self.items[root as usize];
        if it.kind == TREE {
            Node::Tree(self.children(it).iter().map(|&c| self.to_node(base, c)).collect())
        } else {
            Node::Token(Token {
                text: String::from_utf8_lossy(self.text(base, it)).into_owned(),
                keyword: it.kind == KW,
                local: it.local,
                line: it.line,
                end_line: it.end_line,
            })
        }
    }

    /// Adds a public tree to the arena (texts go to `side`). Returns its root.
    pub fn push_node(&mut self, node: &Node) -> u32 {
        match node {
            Node::Token(t) => {
                let start = self.side.len() as u32;
                self.side.extend_from_slice(t.text.as_bytes());
                let end = self.side.len() as u32;
                self.push(Item {
                    kind: if t.keyword { KW } else { NAME },
                    local: t.local,
                    line: t.line,
                    end_line: t.end_line,
                    a: SIDE | start,
                    b: SIDE | end,
                })
            }
            Node::Tree(children) => {
                let ids: Vec<u32> = children.iter().map(|c| self.push_node(c)).collect();
                let a = self.kids.len() as u32;
                self.kids.extend_from_slice(&ids);
                let b = self.kids.len() as u32;
                self.push(Item { kind: TREE, local: false, line: 0, end_line: 0, a, b })
            }
        }
    }

    /// The deduplication key of a method: a fast register-only 128-bit hash of
    /// its tokens, which are keywords and locals, and how they nest.
    pub fn digest(&self, base: &[u8], root: u32) -> u128 {
        let mut d1 = 0x1234_5678_9abc_def0u64;
        let mut d2 = 0x0fed_cba9_8765_4321u64;
        self.digest_rec(base, root, &mut d1, &mut d2);
        ((d1 as u128) << 64) | (d2 as u128)
    }

    fn digest_rec(&self, base: &[u8], n: u32, d1: &mut u64, d2: &mut u64) {
        let it = unsafe { self.items.get_unchecked(n as usize) };
        if it.kind == TREE {
            dg_step(d1, d2, 0x28);
            for &c in self.children(it) {
                self.digest_rec(base, c, d1, d2);
            }
            dg_step(d1, d2, 0x29);
        } else {
            let tag = if it.kind == KW {
                0x6bu64 << 56
            } else if it.local {
                0x6cu64 << 56
            } else {
                0x6eu64 << 56
            };
            let th = digest_hash_bytes(self.text(base, it));
            dg_step(d1, d2, th ^ tag);
        }
    }
}

const F_NAME: u8 = 1;
const F_TEXT: u8 = 2;
const F_COMMENT: u8 = 4;
const F_EXPRSTMT: u8 = 8;
const F_DOCSTR: u8 = 16;

/// What a Lezer grammar's node types mean for a simplified tree (`Grammar` in lezer.ts).
/// Packed into a single `u8` flags byte per node type so `Simplifier::convert`
/// performs ONE table load per syntax node.
pub(crate) struct Cfg {
    pub flags: Vec<u8>,
}

impl Cfg {
    pub fn new(node_names: &[&str], names: &[&str], text: &[&str], comments: &[&str], docstring: bool) -> Cfg {
        let mut flags = vec![0u8; node_names.len()];
        for (i, &n) in node_names.iter().enumerate() {
            let mut f = 0u8;
            if names.contains(&n) {
                f |= F_NAME;
            }
            if text.contains(&n) {
                f |= F_TEXT;
            }
            if comments.contains(&n) {
                f |= F_COMMENT;
            }
            if docstring {
                if n == "ExpressionStatement" {
                    f |= F_EXPRSTMT;
                } else if n == "String" || n == "ContinuedString" {
                    f |= F_DOCSTR;
                }
            }
            flags[i] = f;
        }
        Cfg { flags }
    }
}

/// `simplify` of lezer.ts: the simplified tree of a syntax node, added to `body`.
pub(crate) struct Simplifier<'a, 'b, F: Fn(u32) -> bool> {
    pub src: &'a Src<'b>,
    pub tree: &'a SyntaxTree,
    pub cfg: &'a Cfg,
    pub is_local: F,
}

impl<F: Fn(u32) -> bool> Simplifier<'_, '_, F> {
    pub fn run(&self, body: &mut Body, root: u32) -> Option<u32> {
        let mark = body.scratch.len();
        let mut line_cur = 0usize;
        let r = self.convert(body, root, &mut line_cur);
        debug_assert_eq!(body.scratch.len(), mark);
        r
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn push_token(
        &self,
        body: &mut Body,
        from: u32,
        to: u32,
        ta_u: u32,
        tb_u: u32,
        keyword: bool,
        local: bool,
        line_cur: &mut usize,
    ) -> u32 {
        let (ta, tb) = self.src.text_ref(ta_u, tb_u, &mut body.side);
        let line = self.src.line_of_cursor(from, line_cur);
        let end_off = from.max(to.saturating_sub(1));
        let end_line = if end_off == from {
            line
        } else {
            self.src.line_of_cursor(end_off, line_cur)
        };
        body.push(Item { kind: if keyword { KW } else { NAME }, local, line, end_line, a: ta, b: tb })
    }

    fn convert(&self, body: &mut Body, n: u32, line_cur: &mut usize) -> Option<u32> {
        let tree = self.tree;
        let ty = tree.ty(n) as usize;
        let fl = unsafe { *self.cfg.flags.get_unchecked(ty) };
        if (fl & F_COMMENT) != 0 {
            return None;
        }
        if (fl & F_EXPRSTMT) != 0 {
            let first = tree.first_child(n);
            if first != NONE
                && tree.next_sibling(first) == NONE
                && (unsafe { *self.cfg.flags.get_unchecked(tree.ty(first) as usize) } & F_DOCSTR) != 0
            {
                return None;
            }
        }
        let (from, to) = (tree.from(n), tree.to(n));
        if (fl & F_NAME) != 0 {
            let (a, b) = self.src.trim(from, to);
            if a >= b {
                return None;
            }
            let local = (self.is_local)(n);
            return Some(self.push_token(body, from, to, a, b, false, local, line_cur));
        }
        let first = tree.first_child(n);
        if first == NONE {
            if from == to {
                return None;
            }
            let is_kw = !tree.is_error(n);
            let (a, b) = if is_kw { (from, to) } else { self.src.trim(from, to) };
            return Some(self.push_token(body, from, to, a, b, is_kw, false, line_cur));
        }
        let mark = body.scratch.len();
        let literal = (fl & F_TEXT) != 0;
        let mut at = from;
        let mut c = first;
        while c != NONE {
            let cf = tree.from(c);
            if at < cf {
                self.gap(body, literal, at, cf, line_cur);
            }
            at = tree.to(c);
            if let Some(x) = self.convert(body, c, line_cur) {
                body.scratch.push(x);
            }
            c = tree.next_sibling(c);
        }
        if at < to {
            self.gap(body, literal, at, to, line_cur);
        }
        body.close(mark)
    }

    #[inline]
    fn gap(&self, body: &mut Body, literal: bool, start: u32, end: u32, line_cur: &mut usize) {
        let src = self.src;
        if literal {
            let (a, b) = src.trim(start, end);
            if a < b {
                let t = self.push_token(body, start, end, a, b, false, false, line_cur);
                body.scratch.push(t);
            }
            return;
        }
        if src.is_ascii() {
            let bytes = src.text.as_bytes();
            let mut i = start as usize;
            let e = end as usize;
            if i + 1 == e {
                let b = unsafe { *bytes.get_unchecked(i) };
                if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                    return;
                }
            }
            while i < e {
                while i < e {
                    let b = unsafe { *bytes.get_unchecked(i) };
                    if b > 32 || (b != 32 && !(9..=13).contains(&b)) {
                        break;
                    }
                    i += 1;
                }
                if i >= e {
                    break;
                }
                let s = i;
                while i < e {
                    let b = unsafe { *bytes.get_unchecked(i) };
                    if b <= 32 && (b == 32 || (9..=13).contains(&b)) {
                        break;
                    }
                    i += 1;
                }
                if !(i - s == 1 && unsafe { *bytes.get_unchecked(s) } == b'\\') {
                    let t = self.push_token(body, s as u32, i as u32, s as u32, i as u32, true, false, line_cur);
                    body.scratch.push(t);
                }
            }
            return;
        }
        let mut i = start;
        while i < end {
            while i < end && crate::text::js_ws(src.unit(i)) {
                i += 1;
            }
            if i >= end {
                break;
            }
            let s = i;
            while i < end && !crate::text::js_ws(src.unit(i)) {
                i += 1;
            }
            if !(i - s == 1 && src.unit(s) == b'\\' as u32) {
                let t = self.push_token(body, s, i, s, i, true, false, line_cur);
                body.scratch.push(t);
            }
        }
    }
}
