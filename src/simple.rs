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

impl Body {
    pub fn clear(&mut self) {
        self.items.clear();
        self.kids.clear();
        self.side.clear();
        self.scratch.clear();
    }

    #[inline]
    pub fn text<'s>(&'s self, base: &'s [u8], it: &Item) -> &'s [u8] {
        if it.a & SIDE != 0 {
            &self.side[(it.a & !SIDE) as usize..(it.b & !SIDE) as usize]
        } else {
            &base[it.a as usize..it.b as usize]
        }
    }

    #[inline]
    pub fn children(&self, it: &Item) -> &[u32] {
        &self.kids[it.a as usize..it.b as usize]
    }

    fn push(&mut self, it: Item) -> u32 {
        self.items.push(it);
        (self.items.len() - 1) as u32
    }

    /// Collapses the children above `mark` on the scratch stack into a node
    /// (Section 3.1: no empty lists, no list of a single subtree or name).
    fn close(&mut self, mark: usize) -> Option<u32> {
        let n = self.scratch.len() - mark;
        if n == 0 {
            return None;
        }
        if n == 1 {
            let only = self.scratch[mark];
            let it = &self.items[only as usize];
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

    /// The deduplication key of a method: a 128-bit hash of its tokens, which
    /// are keywords and locals, and how they nest.
    pub fn digest(&self, base: &[u8], root: u32) -> u128 {
        let mut h = xxhash_rust::xxh3::Xxh3::new();
        self.digest_rec(base, root, &mut h);
        h.digest128()
    }

    fn digest_rec(&self, base: &[u8], n: u32, h: &mut xxhash_rust::xxh3::Xxh3) {
        let it = &self.items[n as usize];
        match it.kind {
            TREE => {
                h.update(b"(");
                for &c in self.children(it) {
                    self.digest_rec(base, c, h);
                }
                h.update(b")");
            }
            k => {
                h.update(if k == KW {
                    b"k"
                } else if it.local {
                    b"l"
                } else {
                    b"n"
                });
                h.update(self.text(base, it));
                h.update(b"\0");
            }
        }
    }
}

/// What a Lezer grammar's node types mean for a simplified tree (`Grammar` in lezer.ts).
pub(crate) struct Cfg {
    pub names: Vec<bool>,
    pub text: Vec<bool>,
    pub comments: Vec<bool>,
    /// Python: ExpressionStatement / String / ContinuedString type ids for docstring skipping.
    pub docstring: Option<(u16, u16, u16)>,
}

impl Cfg {
    pub fn new(node_names: &[&str], names: &[&str], text: &[&str], comments: &[&str], docstring: bool) -> Cfg {
        let flag = |set: &[&str]| node_names.iter().map(|n| set.contains(n)).collect::<Vec<bool>>();
        let id = |name: &str| node_names.iter().position(|n| *n == name).unwrap() as u16;
        Cfg {
            names: flag(names),
            text: flag(text),
            comments: flag(comments),
            docstring: if docstring { Some((id("ExpressionStatement"), id("String"), id("ContinuedString"))) } else { None },
        }
    }

    #[inline]
    fn skip(&self, tree: &SyntaxTree, n: u32) -> bool {
        if let Some((stmt, s1, s2)) = self.docstring
            && tree.ty(n) == stmt
        {
            let first = tree.first_child(n);
            if first != NONE && tree.next_sibling(first) == NONE {
                let t = tree.ty(first);
                return t == s1 || t == s2;
            }
        }
        false
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

    #[inline]
    fn token(&self, body: &mut Body, from: u32, to: u32, keyword: bool, local: bool, line_cur: &mut usize) -> u32 {
        let (a, b) = if keyword { (from, to) } else { self.src.trim(from, to) };
        let (ta, tb) = self.src.text_ref(a, b, &mut body.side);
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
        if self.cfg.comments[ty] || self.cfg.skip(tree, n) {
            return None;
        }
        let (from, to) = (tree.from(n), tree.to(n));
        if self.cfg.names[ty] {
            let (a, b) = self.src.trim(from, to);
            if a >= b {
                return None;
            }
            let local = (self.is_local)(n);
            return Some(self.token(body, from, to, false, local, line_cur));
        }
        let first = tree.first_child(n);
        if first == NONE {
            return if from == to {
                None
            } else {
                Some(self.token(body, from, to, !tree.is_error(n), false, line_cur))
            };
        }
        let mark = body.scratch.len();
        let literal = self.cfg.text[ty];
        let mut at = from;
        let mut c = first;
        while c != NONE {
            self.gap(body, literal, at, tree.from(c), line_cur);
            at = tree.to(c);
            if let Some(x) = self.convert(body, c, line_cur) {
                body.scratch.push(x);
            }
            c = tree.next_sibling(c);
        }
        self.gap(body, literal, at, to, line_cur);
        body.close(mark)
    }

    #[inline]
    fn gap(&self, body: &mut Body, literal: bool, start: u32, end: u32, line_cur: &mut usize) {
        if start >= end {
            return;
        }
        let src = self.src;
        if literal {
            let (a, b) = src.trim(start, end);
            if a < b {
                let t = self.token(body, start, end, false, false, line_cur);
                body.scratch.push(t);
            }
            return;
        }
        if src.is_ascii() {
            let bytes = src.text.as_bytes();
            let mut i = start as usize;
            let e = end as usize;
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
                    let t = self.token(body, s as u32, i as u32, true, false, line_cur);
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
                let t = self.token(body, s, i, true, false, line_cur);
                body.scratch.push(t);
            }
        }
    }
}
