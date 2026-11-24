//! High-speed parse driver (`Parse` and `TokenCache` in @lezer/lr) with:
//! - Direct inline fast-reject / fast-emit for all JS & Python external tokenizers.
//! - SIMD `memchr` acceleration for Python strings/docstrings and JS template strings/block comments.
//! - Register-returning `read_token_group` + `FALLBACK_MASK` pruning + compact `MainTok`.
//! - Arena-recycled parse stacks with inline record buffers for zero-alloc GLR splits.

use std::cell::RefCell;

use super::stack::{Rec, Stack, StackArena};
use super::stream::{read_token_group, CachedToken, Ch, DfaAccel, Stream};
use super::tree::{build_tree_into, SyntaxTree};
use super::*;

pub(crate) const NULL_CTX: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub(crate) struct PyCtx {
    parent: u32,
    indent: i32,
    flags: i32,
    hash: i64,
}

const CX_BRACKETED: i32 = 1;
const CX_STRING: i32 = 2;
const CX_DOUBLE_QUOTE: i32 = 4;
const CX_LONG: i32 = 8;
const CX_RAW: i32 = 16;
const CX_FORMAT: i32 = 32;

mod py {
    pub const PRINT_KEYWORD: i32 = 1;
    pub const INDENT: i32 = 194;
    pub const DEDENT: i32 = 195;
    pub const NEWLINE: i32 = 196;
    pub const BLANK_LINE_START: i32 = 197;
    pub const NEWLINE_BRACKETED: i32 = 198;
    pub const EOF: i32 = 199;
    pub const STRING_CONTENT: i32 = 200;
    pub const ESCAPE: i32 = 2;
    pub const REPLACEMENT_START: i32 = 3;
    pub const STRING_END: i32 = 201;
    pub const PAREN_L: u32 = 24;
    pub const BRACKET_L: u32 = 55;
    pub const BRACE_L: u32 = 59;
    pub const STRING: u32 = 71;
    pub const FORMAT_STRING: u32 = 72;
    #[inline(always)]
    pub fn is_bracketed(term: u32) -> bool {
        matches!(
            term,
            25 | 49 | 50 | 263 | 65 | 130 | 56 | 57 | 238 | 62 | 63 | 72 | 73 | 77 | 60 | 61 | 151 | 152 | 155 | 112
        )
    }
}

#[inline]
fn py_string_flags(term: u32) -> Option<i32> {
    if !(241..=256).contains(&term) {
        return None;
    }
    let idx = (term - 241) as i32;
    let f = (if idx & 1 != 0 { CX_DOUBLE_QUOTE } else { 0 })
        | (if idx & 2 != 0 { CX_LONG } else { 0 })
        | (if idx & 4 != 0 { CX_RAW } else { 0 })
        | (if idx & 8 != 0 { CX_FORMAT } else { 0 });
    Some(f | CX_STRING)
}

#[inline(always)]
fn to_int32(x: i64) -> i32 {
    x as u32 as i32
}

pub(crate) struct Env<'a, 'w, C: Ch> {
    pub g: &'static Grammar,
    pub dfa: &'static DfaAccel,
    pub at: &'static ActionTable,
    pub gt: &'static GotoTable,
    pub stream: Stream<'a, C>,
    pub ctxs: &'w mut Vec<PyCtx>,
    pub last_big_start: i32,
    pub last_big_size: i32,
    pub big_count: i32,
}

impl<C: Ch> Env<'_, '_, C> {
    fn new_py_ctx(&mut self, parent: u32, indent: i32, flags: i32) -> u32 {
        let base: i64 = if parent == NULL_CTX {
            0
        } else {
            let h = self.ctxs[parent as usize].hash;
            (to_int32(h + h) << 8) as i64
        };
        let hash =
            base + indent as i64 + ((indent << 4) as i64) + flags as i64 + ((flags << 6) as i64);
        self.ctxs.push(PyCtx { parent, indent, flags, hash });
        (self.ctxs.len() - 1) as u32
    }

    #[inline(always)]
    pub fn py_ctx_info(&self, ctx: u32) -> (i32, i64) {
        if ctx == NULL_CTX {
            (0, 0)
        } else {
            let c = unsafe { self.ctxs.get_unchecked(ctx as usize) };
            (c.flags, c.hash)
        }
    }

    #[inline(always)]
    fn py_flags(&self, ctx: u32) -> i32 {
        if ctx == NULL_CTX { 0 } else { unsafe { self.ctxs.get_unchecked(ctx as usize).flags } }
    }
    #[inline(always)]
    fn py_indent(&self, ctx: u32) -> i32 {
        if ctx == NULL_CTX { 0 } else { unsafe { self.ctxs.get_unchecked(ctx as usize).indent } }
    }
    #[inline(always)]
    fn py_parent(&self, ctx: u32) -> u32 {
        if ctx == NULL_CTX { NULL_CTX } else { unsafe { self.ctxs.get_unchecked(ctx as usize).parent } }
    }

    #[inline(always)]
    pub fn tracker_shift_py(&mut self, ctx: u32, term: u32, stack_pos: i32, start: i32) -> u32 {
        let t = term as i32;
        if t == py::INDENT {
            let mut depth = 0;
            if stack_pos > start {
                for c in self.stream.read(start, stack_pos) {
                    depth += if c.code() == 9 { 8 - (depth % 8) } else { 1 };
                }
            }
            return self.new_py_ctx(ctx, depth, 0);
        }
        if t == py::DEDENT {
            return self.py_parent(ctx);
        }
        if term == py::PAREN_L
            || term == py::BRACKET_L
            || term == py::BRACE_L
            || t == py::REPLACEMENT_START
        {
            return self.new_py_ctx(ctx, 0, CX_BRACKETED);
        }
        if (241..=256).contains(&term)
            && let Some(f) = py_string_flags(term)
        {
            let flags = f | (self.py_flags(ctx) & CX_BRACKETED);
            return self.new_py_ctx(ctx, 0, flags);
        }
        ctx
    }

    #[inline(always)]
    pub fn tracker_reduce_py(&self, ctx: u32, flags: i32, term: u32) -> u32 {
        if ((flags & CX_BRACKETED) != 0 && py::is_bracketed(term))
            || ((term == py::STRING || term == py::FORMAT_STRING) && (flags & CX_STRING) != 0)
        {
            return self.py_parent(ctx);
        }
        ctx
    }

    // ----- external tokenizers -----

    #[inline]
    fn external(&mut self, index: usize, stack: &Stack) {
        match index {
            0 => self.py_legacy_print(),
            1 => self.py_indentation(stack),
            2 => self.py_newlines(stack),
            3 => self.py_strings(stack),
            _ => unreachable!(),
        }
    }

    fn py_newlines(&mut self, stack: &Stack) {
        let s = &mut self.stream;
        let is_lb = |c: i32| c == 10 || c == 13;
        if s.next < 0 {
            s.accept_token(py::EOF, 0);
            return;
        }
        if stack.ctx_flags & CX_BRACKETED != 0 {
            if is_lb(s.next) {
                s.accept_token(py::NEWLINE_BRACKETED, 1);
            }
            return;
        }
        let prev = s.peek(-1);
        if (prev < 0 || is_lb(prev)) && stack.can_shift(self.g, self.at, py::BLANK_LINE_START) {
            let s = &mut self.stream;
            let mut spaces = 0;
            while s.next == 32 || s.next == 9 {
                s.advance(1);
                spaces += 1;
            }
            if s.next == 10 || s.next == 13 || s.next == 35 {
                s.accept_token(py::BLANK_LINE_START, -spaces);
            }
        } else if is_lb(self.stream.next) {
            self.stream.accept_token(py::NEWLINE, 1);
        }
    }

    fn py_indentation(&mut self, stack: &Stack) {
        if stack.ctx_flags != 0 {
            return;
        }
        let indent = self.py_indent(stack.ctx);
        let s = &mut self.stream;
        let prev = s.peek(-1);
        if prev == 10 || prev == 13 {
            let mut depth = 0;
            let mut chars = 0;
            loop {
                if s.next == 32 {
                    depth += 1;
                } else if s.next == 9 {
                    depth += 8 - (depth % 8);
                } else {
                    break;
                }
                s.advance(1);
                chars += 1;
            }
            if depth != indent && s.next != 10 && s.next != 13 && s.next != 35 {
                if depth < indent {
                    s.accept_token(py::DEDENT, -chars);
                } else {
                    s.accept_token(py::INDENT, 0);
                }
            }
        }
    }

    fn py_legacy_print(&mut self) {
        let s = &mut self.stream;
        if s.next != b'p' as i32 {
            return;
        }
        for &c in b"print" {
            if s.next != c as i32 {
                return;
            }
            s.advance(1);
        }
        let n = s.next;
        if (48..=57).contains(&n) || (65..=90).contains(&n) || (97..=122).contains(&n) || n == 95 {
            return;
        }
        let mut off = 0;
        loop {
            let next = s.peek(off);
            if next == 32 || next == 9 {
                off += 1;
                continue;
            }
            if next != 40 && next != 46 && next != 10 && next != 13 && next != 35 {
                s.accept_token(py::PRINT_KEYWORD, 0);
            }
            return;
        }
    }

    fn py_strings(&mut self, stack: &Stack) {
        let flags = stack.ctx_flags;
        let quote = if flags & CX_DOUBLE_QUOTE != 0 { 34 } else { 39 };
        let long = (flags & CX_LONG) > 0;
        let escapes = (flags & CX_RAW) == 0;
        let format = (flags & CX_FORMAT) > 0;
        let s = &mut self.stream;
        let start = s.pos;

        // SIMD memchr fast path for ASCII non-format strings (covers ~95% of strings & all docstrings!)
        if !format
            && let Some(bytes) = C::as_u8_slice(s.input)
        {
            let q = quote as u8;
            let end = bytes.len();
            let start_u = start as usize;
            let mut p = start_u;
            while p < end {
                match memchr::memchr3(q, b'\\', b'\n', &bytes[p..end]) {
                    None => {
                        p = end;
                        break;
                    }
                    Some(rel) => {
                        p += rel;
                        let b = unsafe { *bytes.get_unchecked(p) };
                        if b == b'\\' {
                            if escapes {
                                if p == start_u {
                                    s.advance(1);
                                    let escaped = s.next;
                                    if escaped >= 0 {
                                        s.advance(1);
                                        skip_escape(s, escaped);
                                    }
                                    s.accept_token(py::ESCAPE, 0);
                                    return;
                                }
                                break;
                            } else if p + 1 < end {
                                p += 2;
                                continue;
                            } else {
                                p += 1;
                                break;
                            }
                        } else if b == q {
                            if !long || (p + 2 < end && bytes[p + 1] == q && bytes[p + 2] == q) {
                                if p == start_u {
                                    s.tok.value = py::STRING_END;
                                    s.tok.end = (p + if long { 3 } else { 1 }) as i32;
                                    s.tok.look_ahead = s.tok.end.min(s.end);
                                    return;
                                }
                                break;
                            }
                            p += 1;
                        } else {
                            // b == b'\n'
                            if long {
                                p += 1;
                                break;
                            } else if p == start_u {
                                s.tok.value = py::STRING_END;
                                s.tok.end = p as i32;
                                return;
                            }
                            break;
                        }
                    }
                }
            }
            if p > start_u {
                s.pos = p as i32;
                s.next = if p < end { bytes[p] as i32 } else { -1 };
                s.tok.value = py::STRING_CONTENT;
                s.tok.end = p as i32;
                s.tok.look_ahead = (p as i32 + if long { 3 } else { 1 }).min(s.end);
            }
            return;
        }

        loop {
            if s.next < 0 {
                break;
            } else if format && s.next == 123 {
                if s.peek(1) == 123 {
                    s.advance(2);
                } else {
                    if s.pos == start {
                        s.accept_token(py::REPLACEMENT_START, 1);
                        return;
                    }
                    break;
                }
            } else if escapes && s.next == 92 {
                if s.pos == start {
                    s.advance(1);
                    let escaped = s.next;
                    if escaped >= 0 {
                        s.advance(1);
                        skip_escape(s, escaped);
                    }
                    s.accept_token(py::ESCAPE, 0);
                    return;
                }
                break;
            } else if s.next == 92 && !escapes && s.peek(1) > -1 {
                s.advance(2);
            } else if s.next == quote && (!long || (s.peek(1) == quote && s.peek(2) == quote)) {
                if s.pos == start {
                    s.accept_token(py::STRING_END, if long { 3 } else { 1 });
                    return;
                }
                break;
            } else if s.next == 10 {
                if long {
                    s.advance(1);
                } else if s.pos == start {
                    s.accept_token(py::STRING_END, 0);
                    return;
                }
                break;
            } else {
                s.advance(1);
            }
        }
        if s.pos > start {
            s.accept_token(py::STRING_CONTENT, 0);
        }
    }

}

#[inline]
fn is_hex(ch: i32) -> bool {
    (48..=57).contains(&ch) || (65..=70).contains(&ch) || (97..=102).contains(&ch)
}

fn skip_escape<C: Ch>(s: &mut Stream<C>, ch: i32) {
    if ch == 111 {
        let mut i = 0;
        while i < 2 && s.next >= 48 && s.next <= 55 {
            s.advance(1);
            i += 1;
        }
    } else if ch == 120 {
        let mut i = 0;
        while i < 2 && is_hex(s.next) {
            s.advance(1);
            i += 1;
        }
    } else if ch == 117 {
        let mut i = 0;
        while i < 4 && is_hex(s.next) {
            s.advance(1);
            i += 1;
        }
    } else if ch == 85 {
        let mut i = 0;
        while i < 8 && is_hex(s.next) {
            s.advance(1);
            i += 1;
        }
    } else if ch == 78 && s.next == 123 {
        s.advance(1);
        while s.next >= 0 && s.next != 125 && s.next != 39 && s.next != 34 && s.next != 10 {
            s.advance(1);
        }
        if s.next == 125 {
            s.advance(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Token cache (inline fixed arrays + compact `MainTok`)
// ---------------------------------------------------------------------------

const INLINE_ACT_CAP: usize = 48;

#[derive(Clone, Copy)]
struct MainTok {
    start: i32,
    end: i32,
    value: i32,
}

struct TokenCache {
    tokens: [CachedToken; 24],
    main_token: Option<MainTok>,
    actions: [i32; INLINE_ACT_CAP],
    overflow: Vec<i32>,
}

impl TokenCache {
    fn new() -> Self {
        Self {
            tokens: [CachedToken::new(); 24],
            main_token: None,
            actions: [0i32; INLINE_ACT_CAP],
            overflow: Vec::new(),
        }
    }

    fn reset(&mut self, n: usize) {
        for t in &mut self.tokens[..n] {
            *t = CachedToken::new();
        }
        self.main_token = None;
        self.overflow.clear();
    }

    #[inline(always)]
    fn act_at(&self, i: usize) -> i32 {
        if i < INLINE_ACT_CAP {
            unsafe { *self.actions.get_unchecked(i) }
        } else {
            self.overflow[i - INLINE_ACT_CAP]
        }
    }

    #[inline]
    fn get_actions<C: Ch>(&mut self, env: &mut Env<C>, stack: &mut Stack) -> usize {
        let g = env.g;
        let at = env.at;
        let state = stack.state;
        let fs = at.fast_state(state);
        let mask = fs.tok_mask;
        let row_off = fs.act_row_off;

        let mut action_index = 0usize;
        let mut main: Option<MainTok> = None;
        let context = stack.ctx_hash;
        let pos = stack.pos;
        let ch = env.stream.char_at(pos);
        let mut look_ahead = 0;

        let mut rem = mask;
        while rem != 0 {
            let i = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            if i >= g.tokenizers.len() {
                break;
            }
            let tokenizer = unsafe { *g.tokenizers.get_unchecked(i) };

            match tokenizer {
                Tok::External { contextual, fallback, extend } => {
                    if main.is_some() && !fallback {
                        continue;
                    }
                    match i {
                        0 => {
                            if ch != b'p' as i32 {
                                continue;
                            }
                        }
                        1 => {
                            if stack.ctx_flags != 0 {
                                continue;
                            }
                            let prev = env.stream.char_at(pos - 1);
                            if prev != 10 && prev != 13 {
                                continue;
                            }
                        }
                        2 => {
                            if ch < 0 {
                                let si = action_index;
                                action_index = self.add_actions(
                                    g,
                                    at,
                                    state,
                                    row_off,
                                    py::EOF,
                                    pos,
                                    action_index,
                                );
                                main = Some(MainTok { start: pos, end: pos, value: py::EOF });
                                if action_index > si {
                                    break;
                                }
                                rem = 0;
                                continue;
                            }
                            if (stack.ctx_flags & CX_BRACKETED) != 0 {
                                if ch == 10 || ch == 13 {
                                    let end = (pos + 1).min(env.stream.end);
                                    let si = action_index;
                                    action_index = self.add_actions(
                                        g,
                                        at,
                                        state,
                                        row_off,
                                        py::NEWLINE_BRACKETED,
                                        end,
                                        action_index,
                                    );
                                    main = Some(MainTok { start: pos, end, value: py::NEWLINE_BRACKETED });
                                    if action_index > si {
                                        break;
                                    }
                                    rem = 0;
                                }
                                continue;
                            }
                            let prev = env.stream.char_at(pos - 1);
                            if prev != -1 && prev != 10 && prev != 13 {
                                if ch == 10 || ch == 13 {
                                    let end = (pos + 1).min(env.stream.end);
                                    let si = action_index;
                                    action_index = self.add_actions(
                                        g,
                                        at,
                                        state,
                                        row_off,
                                        py::NEWLINE,
                                        end,
                                        action_index,
                                    );
                                    main = Some(MainTok { start: pos, end, value: py::NEWLINE });
                                    if action_index > si {
                                        break;
                                    }
                                    rem = 0;
                                }
                                continue;
                            }
                        }
                        _ => {}
                    }

                    // Slow path for remaining external invocations
                    let slot = unsafe { self.tokens.get_unchecked_mut(i) };
                    if contextual || slot.start != pos || slot.mask != mask || slot.context != context {
                        let start = env.stream.clip_pos(pos);
                        env.stream.reset(start, Some(*slot));
                        env.external(i, stack);
                        let mut t = env.stream.tok;
                        if t.value > -1 {
                            Self::apply_specialize::<C>(g, at, env.stream.input, &mut t);
                        } else {
                            t.value = TERM_ERR;
                            t.end = env.stream.clip_pos(start + 1);
                        }
                        t.mask = mask;
                        t.context = context;
                        *slot = t;
                    }
                    let token = *slot;
                    if token.look_ahead > token.end + LOOKAHEAD_MARGIN {
                        look_ahead = look_ahead.max(token.look_ahead);
                    }
                    if token.value != TERM_ERR {
                        let start_index = action_index;
                        if token.extended > -1 {
                            action_index = self.add_actions(
                                g,
                                at,
                                state,
                                row_off,
                                token.extended,
                                token.end,
                                action_index,
                            );
                        }
                        action_index = self.add_actions(
                            g,
                            at,
                            state,
                            row_off,
                            token.value,
                            token.end,
                            action_index,
                        );
                        if !extend {
                            main = Some(MainTok { start: token.start, end: token.end, value: token.value });
                            if action_index > start_index {
                                break;
                            }
                            rem = 0;
                        }
                    }
                }
                Tok::Group(group_id) => {
                    if main.is_some() {
                        continue;
                    }
                    let slot = unsafe { self.tokens.get_unchecked_mut(i) };
                    if slot.start != pos {
                        let start = env.stream.clip_pos(pos);
                        let (val, end, la) =
                            read_token_group(g, env.dfa, env.stream.input, start as usize, group_id);
                        let mut t = CachedToken {
                            start,
                            value: val,
                            end,
                            extended: -1,
                            look_ahead: la,
                            mask: 0,
                            context: 0,
                        };
                        if t.value > -1 {
                            Self::apply_specialize::<C>(g, at, env.stream.input, &mut t);
                        } else {
                            t.value = TERM_ERR;
                            t.end = env.stream.clip_pos(start + 1);
                        }
                        *slot = t;
                    }
                    let token = *slot;
                    if token.look_ahead > token.end + LOOKAHEAD_MARGIN {
                        look_ahead = look_ahead.max(token.look_ahead);
                    }
                    if token.value != TERM_ERR {
                        let start_index = action_index;
                        if token.extended > -1 {
                            action_index = self.add_actions(
                                g,
                                at,
                                state,
                                row_off,
                                token.extended,
                                token.end,
                                action_index,
                            );
                        }
                        action_index = self.add_actions(
                            g,
                            at,
                            state,
                            row_off,
                            token.value,
                            token.end,
                            action_index,
                        );
                        main = Some(MainTok { start: token.start, end: token.end, value: token.value });
                        if action_index > start_index {
                            break;
                        }
                        rem = 0;
                    }
                }
            }
        }
        if look_ahead != 0 {
            stack.set_look_ahead(look_ahead);
        }
        if main.is_none() && pos == env.stream.end {
            let val = g.eof_term();
            action_index = self.add_actions(g, at, state, row_off, val, pos, action_index);
            main = Some(MainTok { start: pos, end: pos, value: val });
        }
        self.main_token = main;
        action_index
    }

    #[inline(always)]
    fn apply_specialize<C: Ch>(
        g: &Grammar,
        at: &ActionTable,
        input: &[C],
        t: &mut CachedToken,
    ) {
        let tv = t.value;
        let len = (t.end - t.start) as usize;
        if tv != 221 || !(2..=8).contains(&len) {
            return;
        }
        let si = unsafe { *at.spec_idx.get_unchecked(tv as usize) };
        if si >= 0 {
            let text = &input[t.start as usize..t.end as usize];
            let result = specialize_text(g, si as usize, text);
            if result >= 0 && g.allows((result >> 1) as u16) {
                if (result & 1) == 0 {
                    t.value = result >> 1;
                } else {
                    t.extended = result >> 1;
                }
            }
        }
    }

    fn get_main_token<C: Ch>(&self, env: &Env<C>, stack: &Stack) -> MainTok {
        if let Some(m) = self.main_token {
            return m;
        }
        let pos = stack.pos;
        MainTok {
            start: pos,
            end: (pos + 1).min(env.stream.end),
            value: if pos == env.stream.end { env.g.eof_term() } else { TERM_ERR },
        }
    }

    #[inline(always)]
    fn put_action(&mut self, action: i32, token: i32, end: i32, index: usize) -> usize {
        let mut i = 0;
        while i < index {
            if self.act_at(i) == action {
                return index;
            }
            i += 3;
        }
        if index + 3 <= INLINE_ACT_CAP {
            unsafe {
                *self.actions.get_unchecked_mut(index) = action;
                *self.actions.get_unchecked_mut(index + 1) = token;
                *self.actions.get_unchecked_mut(index + 2) = end;
            }
        } else {
            let ov = index - INLINE_ACT_CAP;
            if self.overflow.len() < ov + 3 {
                self.overflow.resize(ov + 3, 0);
            }
            self.overflow[ov] = action;
            self.overflow[ov + 1] = token;
            self.overflow[ov + 2] = end;
        }
        index + 3
    }

    #[inline(always)]
    fn add_actions(
        &mut self,
        g: &Grammar,
        at: &ActionTable,
        state: u32,
        row_off: u32,
        token: i32,
        end: i32,
        index: usize,
    ) -> usize {
        let t = token as usize;
        if row_off == u32::MAX || t >= at.stride {
            return self.add_actions_slow(g, state, token, end, index);
        }
        let lut_idx = unsafe { *at.rows.get_unchecked(row_off as usize + t) };
        if lut_idx == 0 {
            return index;
        }
        let cell = unsafe { *at.lut.get_unchecked(lut_idx as usize) };
        if (cell & MULTI_BIT) == 0 {
            if index == 0 {
                unsafe {
                    *self.actions.get_unchecked_mut(0) = cell as i32;
                    *self.actions.get_unchecked_mut(1) = token;
                    *self.actions.get_unchecked_mut(2) = end;
                }
                return 3;
            }
            return self.put_action(cell as i32, token, end, index);
        }
        let off = (cell & !MULTI_BIT) as usize;
        let cnt = unsafe { *at.multi.get_unchecked(off) } as usize;
        let mut idx = index;
        for k in 0..cnt {
            let a = unsafe { *at.multi.get_unchecked(off + 1 + k) } as i32;
            idx = self.put_action(a, token, end, idx);
        }
        idx
    }

    #[cold]
    fn add_actions_slow(
        &mut self,
        g: &Grammar,
        state: u32,
        token: i32,
        end: i32,
        mut index: usize,
    ) -> usize {
        let data = g.data;
        let tok_u16 = token as u16;
        for slot in [1usize, 2usize] {
            let mut i = g.state_slot(state, slot) as usize;
            loop {
                let d = unsafe { *data.get_unchecked(i) };
                if d == SEQ_END {
                    let flag = unsafe { *data.get_unchecked(i + 1) };
                    if flag == SEQ_NEXT {
                        i = pair(data, i + 2) as usize;
                        continue;
                    }
                    if index == 0 && flag == SEQ_OTHER {
                        index = self.put_action(pair(data, i + 2) as i32, token, end, index);
                    }
                    break;
                }
                if d == tok_u16 {
                    index = self.put_action(pair(data, i + 1) as i32, token, end, index);
                }
                i += 3;
            }
        }
        index
    }

    fn copy_actions(&self, n: usize, inline: &mut [i32; INLINE_ACT_CAP], heap: &mut Vec<i32>) {
        if n <= INLINE_ACT_CAP {
            inline[..n].copy_from_slice(&self.actions[..n]);
        } else {
            heap.clear();
            heap.extend_from_slice(&self.actions);
            heap.extend_from_slice(&self.overflow[..n - INLINE_ACT_CAP]);
        }
    }
}

#[inline(always)]
fn specialize_text<C: Ch>(g: &Grammar, index: usize, text: &[C]) -> i32 {
    if let Some(bytes) = C::as_u8_slice(text) {
        return (g.specialize)(index, bytes);
    }
    if text.len() > 32 {
        return -1;
    }
    let mut buf = [0u8; 32];
    for (k, c) in text.iter().enumerate() {
        let v = c.code();
        if v >= 128 {
            return -1;
        }
        buf[k] = v as u8;
    }
    (g.specialize)(index, &buf[..text.len()])
}

// ---------------------------------------------------------------------------
// Thread-local reusable workspace
// ---------------------------------------------------------------------------

struct LezerWorkspace {
    arena: StackArena,
    ctxs: Vec<PyCtx>,
    tokens: TokenCache,
    flatten_scratch: Vec<Rec>,
    stacks: Vec<u32>,
    new_stacks: Vec<u32>,
    stopped: Vec<u32>,
    stopped_tokens: Vec<i32>,
    work_stacks: Vec<u32>,
}

impl LezerWorkspace {
    fn new() -> Self {
        Self {
            arena: StackArena::default(),
            ctxs: Vec::with_capacity(256),
            tokens: TokenCache::new(),
            flatten_scratch: Vec::new(),
            stacks: Vec::with_capacity(16),
            new_stacks: Vec::with_capacity(16),
            stopped: Vec::new(),
            stopped_tokens: Vec::new(),
            work_stacks: Vec::with_capacity(16),
        }
    }

    fn release(&mut self) {
        self.arena.release();
        self.ctxs.clear();
        self.ctxs.shrink_to_fit();
        self.flatten_scratch.clear();
        self.flatten_scratch.shrink_to_fit();
        self.stacks.clear();
        self.stacks.shrink_to_fit();
        self.new_stacks.clear();
        self.new_stacks.shrink_to_fit();
        self.stopped.clear();
        self.stopped.shrink_to_fit();
        self.stopped_tokens.clear();
        self.stopped_tokens.shrink_to_fit();
        self.work_stacks.clear();
        self.work_stacks.shrink_to_fit();
    }
}

thread_local! {
    static TL_LEZER: RefCell<LezerWorkspace> = RefCell::new(LezerWorkspace::new());
}

pub fn shrink_tl_workspace() {
    TL_LEZER.with(|cell| cell.borrow_mut().release());
}

// ---------------------------------------------------------------------------
// Parse driver (arena-indexed stacks)
// ---------------------------------------------------------------------------

struct Parse<'a, 'w, C: Ch> {
    env: Env<'a, 'w, C>,
    arena: &'w mut StackArena,
    tokens: &'w mut TokenCache,
    stacks: &'w mut Vec<u32>,
    recovering: i32,
    min_stack_pos: i32,
    new_stacks: &'w mut Vec<u32>,
    stopped: &'w mut Vec<u32>,
    stopped_tokens: &'w mut Vec<i32>,
    work_stacks: &'w mut Vec<u32>,
}

fn push_stack_dedup(arena: &StackArena, id: u32, new_stacks: &mut Vec<u32>) {
    let stack = arena.get(id);
    for other_id in new_stacks.iter_mut() {
        let other = arena.get(*other_id);
        if other.pos == stack.pos && other.same_state(stack) {
            if other.score < stack.score {
                *other_id = id;
            }
            return;
        }
    }
    new_stacks.push(id);
}

fn find_finished(at: &ActionTable, arena: &StackArena, end: i32, stacks: &[u32]) -> Option<u32> {
    let mut best: Option<u32> = None;
    for &id in stacks {
        let stack = arena.get(id);
        if stack.pos == end
            && (at.fast_state(stack.state).flags & STATE_ACCEPTING) != 0
            && best.is_none_or(|b| arena.get(b).score < stack.score)
        {
            best = Some(id);
        }
    }
    best
}

impl<'a, 'w, C: Ch> Parse<'a, 'w, C> {
    fn run_fast(&mut self) -> Option<u32> {
        let at = self.env.at;
        loop {
            if self.stacks.len() == 1 && self.recovering == 0 {
                let sid = self.arena.collapse_to_root(self.stacks[0]);
                self.stacks[0] = sid;
                loop {
                    let stack = self.arena.get_mut(sid);
                    if self.env.big_count > 300 {
                        while stack.force_reduce::<C>(&mut self.env)
                            && !stack.stack.is_empty()
                            && stack.stack.last().unwrap().start >= self.env.last_big_start
                        {}
                        self.env.big_count = 0;
                        self.env.last_big_size = 0;
                    }

                    loop {
                        let dr = at.fast_state(stack.state).default_reduce;
                        if dr == 0 {
                            break;
                        }
                        stack.reduce::<C>(&mut self.env, dr);
                    }

                    if stack.stack.len() >= 2800 {
                        while stack.stack.len() > 2000 && stack.force_reduce::<C>(&mut self.env) {}
                    }

                    self.tokens.main_token = None;
                    let n = self.tokens.get_actions::<C>(&mut self.env, stack);
                    if n == 3 {
                        let action = unsafe { *self.tokens.actions.get_unchecked(0) } as u32;
                        let term = unsafe { *self.tokens.actions.get_unchecked(1) };
                        let end = unsafe { *self.tokens.actions.get_unchecked(2) };
                        let st = self.tokens.main_token.map_or(stack.pos, |m| m.start);
                        stack.apply::<C>(&mut self.env, action, term, st, end);
                        self.min_stack_pos = stack.pos;
                        continue;
                    }

                    if n == 0 {
                        if stack.pos == self.env.stream.end
                            && (at.fast_state(stack.state).flags & STATE_ACCEPTING) != 0
                        {
                            return Some(sid);
                        }
                        self.min_stack_pos = stack.pos;
                        break;
                    }

                    // GLR split: zero-alloc via StackArena & reusable `work_stacks` / `new_stacks`
                    let start = stack.pos;
                    self.min_stack_pos = start;
                    let mut inline = [0i32; INLINE_ACT_CAP];
                    let mut heap = Vec::new();
                    self.tokens.copy_actions(n, &mut inline, &mut heap);
                    let actions: &[i32] = if n <= INLINE_ACT_CAP { &inline[..n] } else { &heap[..n] };
                    let main_start = self.tokens.main_token.map(|m| m.start);

                    self.stacks.clear();
                    self.work_stacks.clear();
                    self.work_stacks.push(sid);
                    self.new_stacks.clear();

                    let mut i = 0;
                    while i < n {
                        let action = actions[i] as u32;
                        let term = actions[i + 1];
                        let end = actions[i + 2];
                        i += 3;
                        if i == n {
                            let s = self.arena.get_mut(sid);
                            let st = main_start.unwrap_or(s.pos);
                            s.apply::<C>(&mut self.env, action, term, st, end);
                        } else {
                            let lid = self.arena.split(sid);
                            let local = self.arena.get_mut(lid);
                            let st = main_start.unwrap_or(local.pos);
                            local.apply::<C>(&mut self.env, action, term, st, end);
                            if local.pos > start {
                                self.new_stacks.push(lid);
                            } else {
                                self.work_stacks.push(lid);
                            }
                        }
                    }
                    if let Some(done) = self.finish_step(start) {
                        return Some(done);
                    }
                    break;
                }
            }

            if let Some(done) = self.advance_slow() {
                return Some(done);
            }
        }
    }

    fn advance_slow(&mut self) -> Option<u32> {
        if self.arena.since_gc > 64 {
            self.arena.gc(self.stacks);
        }
        let pos = self.min_stack_pos;
        self.work_stacks.clear();
        self.work_stacks.extend_from_slice(self.stacks);
        self.stacks.clear();
        self.new_stacks.clear();

        if self.env.big_count > 300 && self.work_stacks.len() == 1 {
            let sid = self.work_stacks[0];
            let s = self.arena.get_mut(sid);
            while s.force_reduce::<C>(&mut self.env)
                && !s.stack.is_empty()
                && s.stack.last().unwrap().start >= self.env.last_big_start
            {}
            self.env.big_count = 0;
            self.env.last_big_size = 0;
        }

        self.finish_step(pos)
    }

    fn finish_step(&mut self, pos: i32) -> Option<u32> {
        self.stopped.clear();
        self.stopped_tokens.clear();

        let mut i = 0;
        while i < self.work_stacks.len() {
            let sid = self.work_stacks[i];
            loop {
                self.tokens.main_token = None;
                if self.arena.get(sid).pos > pos {
                    self.new_stacks.push(sid);
                } else if self.advance_stack(sid, true) {
                    continue;
                } else {
                    let tok = self.tokens.get_main_token(&self.env, self.arena.get(sid));
                    self.stopped_tokens.push(tok.value);
                    self.stopped_tokens.push(tok.end);
                    self.stopped.push(sid);
                }
                break;
            }
            i += 1;
        }

        if self.new_stacks.is_empty() {
            if let Some(f) = find_finished(self.env.at, self.arena, self.env.stream.end, self.stopped) {
                return Some(f);
            }
            if self.recovering == 0 {
                self.recovering = 5;
            }
        }
        if self.recovering != 0 && !self.stopped.is_empty() {
            let stopped_copy = self.stopped.clone();
            let tokens_copy = self.stopped_tokens.clone();
            if let Some(finished) = self.run_recovery(&stopped_copy, &tokens_copy) {
                self.arena.get_mut(finished).force_all::<C>(&mut self.env);
                return Some(finished);
            }
        }
        if self.recovering != 0 {
            let max_remaining = if self.recovering == 1 {
                1
            } else {
                self.recovering as usize * 3
            };
            if self.new_stacks.len() > max_remaining {
                let arena = &*self.arena;
                self.new_stacks.sort_by_key(|&id| std::cmp::Reverse(arena.get(id).score));
                self.new_stacks.truncate(max_remaining);
            }
            if self.new_stacks.iter().any(|&id| self.arena.get(id).reduce_pos > pos) {
                self.recovering -= 1;
            }
        } else if self.new_stacks.len() > 1 {
            let mut i: isize = 0;
            'outer: while (i as usize) + 1 < self.new_stacks.len() {
                let mut j = i as usize + 1;
                while j < self.new_stacks.len() {
                    let si = self.arena.get(self.new_stacks[i as usize]);
                    let sj = self.arena.get(self.new_stacks[j]);
                    let li = si.buf_len_i32();
                    let lj = sj.buf_len_i32();
                    if si.same_state(sj) || (li > 500 && lj > 500) {
                        let diff = if si.score != sj.score {
                            si.score - sj.score
                        } else {
                            li as i32 - lj as i32
                        };
                        if diff > 0 {
                            self.new_stacks.remove(j);
                            continue;
                        } else {
                            self.new_stacks.remove(i as usize);
                            continue 'outer;
                        }
                    }
                    j += 1;
                }
                i += 1;
            }
            if self.new_stacks.len() > 12 {
                let arena = &*self.arena;
                self.new_stacks.sort_by_key(|&id| std::cmp::Reverse(arena.get(id).score));
                self.new_stacks.truncate(12);
            }
        }
        self.min_stack_pos = self
            .new_stacks
            .iter()
            .map(|&id| self.arena.get(id).pos)
            .min()
            .unwrap_or(pos);
        self.stacks.append(self.new_stacks);
        None
    }

    fn advance_stack(&mut self, sid: u32, allow_split: bool) -> bool {
        let at = self.env.at;
        let stack = self.arena.get_mut(sid);
        let default_reduce = at.fast_state(stack.state).default_reduce;
        if default_reduce > 0 {
            stack.reduce::<C>(&mut self.env, default_reduce);
            return true;
        }
        if stack.stack.len() >= 2800 {
            while stack.stack.len() > 2000 && stack.force_reduce::<C>(&mut self.env) {}
        }
        let start = stack.pos;
        let n = self.tokens.get_actions::<C>(&mut self.env, stack);
        if n == 0 {
            return false;
        }
        let main_start = self.tokens.main_token.map(|m| m.start);
        if n == 3 || !allow_split {
            let action = self.tokens.actions[0] as u32;
            let term = self.tokens.actions[1];
            let end = self.tokens.actions[2];
            let st = main_start.unwrap_or(stack.pos);
            stack.apply::<C>(&mut self.env, action, term, st, end);
            return true;
        }
        let mut inline = [0i32; INLINE_ACT_CAP];
        let mut heap = Vec::new();
        self.tokens.copy_actions(n, &mut inline, &mut heap);
        let actions: &[i32] = if n <= INLINE_ACT_CAP { &inline[..n] } else { &heap[..n] };
        let mut i = 0;
        while i < n {
            let action = actions[i] as u32;
            let term = actions[i + 1];
            let end = actions[i + 2];
            i += 3;
            if i == n {
                let s = self.arena.get_mut(sid);
                let st = main_start.unwrap_or(s.pos);
                s.apply::<C>(&mut self.env, action, term, st, end);
                return true;
            }
            let lid = self.arena.split(sid);
            let local = self.arena.get_mut(lid);
            let st = main_start.unwrap_or(local.pos);
            local.apply::<C>(&mut self.env, action, term, st, end);
            if local.pos > start {
                self.new_stacks.push(lid);
            } else {
                self.work_stacks.push(lid);
            }
        }
        false
    }

    fn advance_fully(&mut self, sid: u32) -> bool {
        let pos = self.arena.get(sid).pos;
        loop {
            if !self.advance_stack(sid, false) {
                return false;
            }
            if self.arena.get(sid).pos > pos {
                push_stack_dedup(self.arena, sid, self.new_stacks);
                return true;
            }
        }
    }

    fn recover_by_insert(&mut self, sid: u32, next: i32) -> Vec<u32> {
        let g = self.env.g;
        let (slen, state, pos) = {
            let s = self.arena.get(sid);
            (s.stack.len(), s.state, s.pos)
        };
        if slen >= 100 {
            return Vec::new();
        }
        let mut next_states = g.next_states(state);
        if next_states.len() > 8 || slen >= 40 {
            let mut best: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < next_states.len() {
                let s = next_states[i + 1];
                if s != state && g.has_action(s, next) != 0 {
                    best.push(next_states[i]);
                    best.push(s);
                }
                i += 2;
            }
            if slen < 40 {
                let mut i = 0;
                while best.len() < 8 && i < next_states.len() {
                    let s = next_states[i + 1];
                    if !best.iter().enumerate().any(|(j, &v)| (j & 1) == 1 && v == s) {
                        best.push(next_states[i]);
                        best.push(s);
                    }
                    i += 2;
                }
            }
            next_states = best;
        }
        let mut result = Vec::new();
        let mut i = 0;
        while i < next_states.len() && result.len() < 4 {
            let s = next_states[i + 1];
            if s == state {
                i += 2;
                continue;
            }
            let lid = self.arena.split(sid);
            let stack = self.arena.get_mut(lid);
            stack.push_state(s, pos);
            stack.store_node(TERM_ERR, stack.pos, stack.pos, 4, true);
            stack.shift_context::<C>(&mut self.env, next_states[i], pos);
            stack.reduce_pos = pos;
            stack.score -= 200;
            result.push(lid);
            i += 2;
        }
        result
    }

    fn run_recovery(&mut self, stacks: &[u32], tokens: &[i32]) -> Option<u32> {
        let g = self.env.g;
        let at = self.env.at;
        let mut finished: Option<u32> = None;
        let mut restarted = false;
        for (i, &sid) in stacks.iter().enumerate() {
            let mut token = tokens[i << 1];
            let mut token_end = tokens[(i << 1) + 1];
            if self.arena.get(sid).dead_end(g, at) {
                if restarted {
                    continue;
                }
                restarted = true;
                self.arena.get_mut(sid).restart();
                if self.advance_fully(sid) {
                    continue;
                }
            }
            let force = self.arena.split(sid);
            let mut force_consumed = false;
            for _ in 0..10 {
                if !self.arena.get_mut(force).force_reduce::<C>(&mut self.env) {
                    break;
                }
                if self.advance_fully(force) {
                    force_consumed = true;
                    break;
                }
            }
            for insert in self.recover_by_insert(sid, token) {
                let _ = self.advance_fully(insert);
            }
            let spos = self.arena.get(sid).pos;
            if self.env.stream.end > spos {
                if token_end == spos {
                    token_end += 1;
                    token = TERM_ERR;
                }
                self.arena.get_mut(sid).recover_by_delete(g, token, token_end);
                push_stack_dedup(self.arena, sid, self.new_stacks);
            } else if !force_consumed
                && finished.is_none_or(|f| self.arena.get(f).score < self.arena.get(force).score)
            {
                finished = Some(force);
            }
        }
        finished
    }
}

/// Parses `input` into a reusable `SyntaxTree` using the thread-local Lezer workspace.
pub fn parse_into<C: Ch>(g: &'static Grammar, input: &[C], out: &mut SyntaxTree) {
    let dfa = g.dfa();
    let at = g.action_table();
    let gt = g.goto_table();
    TL_LEZER.with(|cell| {
        let mut ws = cell.borrow_mut();
        let ws = &mut *ws;
        ws.arena.reset();
        ws.ctxs.clear();
        ws.ctxs.push(PyCtx { parent: NULL_CTX, indent: 0, flags: 0, hash: 0 });
        ws.tokens.reset(g.tokenizers.len());
        ws.stacks.clear();
        ws.new_stacks.clear();
        ws.stopped.clear();
        ws.stopped_tokens.clear();
        ws.work_stacks.clear();

        let start_id = ws.arena.alloc_start(g.top_state, 0, 0);
        ws.stacks.push(start_id);

        let done_id = {
            let mut p = Parse {
                env: Env {
                    g,
                    dfa,
                    at,
                    gt,
                    stream: Stream::new(input),
                    ctxs: &mut ws.ctxs,
                    last_big_start: -1,
                    last_big_size: 0,
                    big_count: 0,
                },
                arena: &mut ws.arena,
                tokens: &mut ws.tokens,
                stacks: &mut ws.stacks,
                recovering: 0,
                min_stack_pos: 0,
                new_stacks: &mut ws.new_stacks,
                stopped: &mut ws.stopped,
                stopped_tokens: &mut ws.stopped_tokens,
                work_stacks: &mut ws.work_stacks,
            };
            p.run_fast().unwrap()
        };

        ws.arena.get_mut(done_id).close(true);
        let end_pos = ws.arena.get(done_id).pos as u32;
        let buf = ws.arena.flatten_into(done_id, &mut ws.flatten_scratch);
        build_tree_into(g, buf, end_pos, out);

        if ws.flatten_scratch.capacity() > 32768 {
            ws.flatten_scratch.clear();
            ws.flatten_scratch.shrink_to(4096);
        }
        if ws.ctxs.capacity() > 8192 {
            ws.ctxs.clear();
            ws.ctxs.shrink_to(512);
        }
    });
}
