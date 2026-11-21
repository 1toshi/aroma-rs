//! Single-pass syntax tree builder from a parse's postfix record buffer.
//! Dissolves repeat nodes (`id >= min_repeat_term`), skips context/look-ahead
//! metadata records (`size < 0`), and handles deep nesting (`depth > 2500`).

use super::stack::Rec;
use super::Grammar;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub ty: u16,
    pub from: u32,
    pub to: u32,
    pub parent: u32,
    pub first: u32,
    pub last: u32,
    pub next: u32,
}

#[derive(Default)]
pub struct SyntaxTree {
    pub nodes: Vec<Node>,
}

impl SyntaxTree {
    #[inline(always)]
    pub fn ty(&self, n: u32) -> u16 {
        unsafe { self.nodes.get_unchecked(n as usize).ty }
    }
    #[inline(always)]
    pub fn from(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).from }
    }
    #[inline(always)]
    pub fn to(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).to }
    }
    #[inline(always)]
    pub fn parent(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).parent }
    }
    #[inline(always)]
    pub fn first_child(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).first }
    }
    #[inline(always)]
    pub fn last_child(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).last }
    }
    #[inline(always)]
    pub fn next_sibling(&self, n: u32) -> u32 {
        unsafe { self.nodes.get_unchecked(n as usize).next }
    }
    #[inline(always)]
    pub fn is_error(&self, n: u32) -> bool {
        unsafe { self.nodes.get_unchecked(n as usize).ty == 0 }
    }
}

struct Builder<'a> {
    buf: &'a [Rec],
    pos: usize,
    nodes: &'a mut Vec<Node>,
    min_repeat: i32,
}

impl Builder<'_> {
    #[inline(always)]
    fn add(&mut self, parent: u32, ty: i32, from: i32, to: i32) -> u32 {
        let id = self.nodes.len() as u32;
        let first = unsafe { self.nodes.get_unchecked(parent as usize).first };
        self.nodes.push(Node {
            ty: ty as u16,
            from: from as u32,
            to: to as u32,
            parent,
            first: NONE,
            last: NONE,
            next: first,
        });
        if first == NONE {
            unsafe { self.nodes.get_unchecked_mut(parent as usize).last = id };
        }
        unsafe { self.nodes.get_unchecked_mut(parent as usize).first = id };
        id
    }

    fn take_node(&mut self, parent: u32, depth: u32) {
        let p = self.pos - 1;
        let r = unsafe { *self.buf.get_unchecked(p) };
        let id = r[0];
        let start = r[1];
        let end = r[2];
        let size = r[3];
        if size < 0 {
            self.pos = p;
            return;
        }
        let end_pos = self.pos - (size as usize >> 2);
        self.pos = p;
        if size == 4 {
            if id < self.min_repeat {
                self.add(parent, id, start, end);
            }
            return;
        }
        let node = if id >= self.min_repeat {
            parent
        } else {
            self.add(parent, id, start, end)
        };
        let local_in_repeat = if id >= self.min_repeat { id } else { -1 };
        while self.pos > end_pos {
            let q = self.pos - 1;
            let qr = unsafe { *self.buf.get_unchecked(q) };
            if local_in_repeat >= 0 && qr[0] == local_in_repeat && qr[3] >= 0 {
                self.pos = q;
            } else if depth > 2500 {
                self.take_flat_node(node, end_pos);
            } else {
                self.take_node(node, depth + 1);
            }
        }
    }

    fn take_flat_node(&mut self, parent: u32, min_pos: usize) {
        while self.pos > min_pos {
            let p = self.pos - 1;
            let r = self.buf[p];
            if r[3] == 4 && r[0] < self.min_repeat {
                self.add(parent, r[0], r[1], r[2]);
            }
            self.pos = p;
        }
    }
}

pub(crate) fn build_tree_into(g: &Grammar, buffer: &[Rec], length: u32, out: &mut SyntaxTree) {
    out.nodes.clear();
    let est = (buffer.len() / 2).max(64);
    if out.nodes.capacity() < est {
        out.nodes.reserve(est - out.nodes.capacity());
    }
    out.nodes.push(Node {
        ty: g.top_term,
        from: 0,
        to: length,
        parent: NONE,
        first: NONE,
        last: NONE,
        next: NONE,
    });
    let mut b = Builder {
        buf: buffer,
        pos: buffer.len(),
        nodes: &mut out.nodes,
        min_repeat: g.min_repeat_term as i32,
    };
    while b.pos > 0 {
        b.take_node(0, 0);
    }
}
