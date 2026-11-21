//! High-speed parse driver (`Parse` and `TokenCache` in @lezer/lr) with:
//! - Const-generic `IS_PY` specialization eliminating runtime grammar-kind branches.
//! - Direct inline fast-reject / fast-emit for all JS & Python external tokenizers.
//! - Register-returning `read_token_group` (zero `Stream` mutation on DFA tokens).
//! - Thread-local reusable `LezerWorkspace` + tight deterministic `run_fast`.

use std::cell::RefCell;

use super::stack::{BufPool, Frame, Rec, Stack};
use super::stream::{read_token_group, read_token_local, CachedToken, Ch, DfaAccel, Stream};
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

mod jst {
    pub const NO_SEMI: i32 = 317;
    pub const NO_SEMI_TYPE: i32 = 318;
    pub const INCDEC: i32 = 1;
    pub const INCDEC_PREFIX: i32 = 2;
    pub const QUESTION_DOT: i32 = 3;
    pub const JSX_START_TAG: i32 = 4;
    pub const INSERT_SEMI: i32 = 319;
    pub const DIALECT_JSX: usize = 0;
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
    pub bufs: &'w mut BufPool,
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
    fn external<const IS_PY: bool>(&mut self, index: usize, stack: &Stack) {
        if IS_PY {
            match index {
                0 => self.py_legacy_print(),
                1 => self.py_indentation(stack),
                2 => self.py_newlines(stack),
                3 => self.py_strings(stack),
                _ => unreachable!(),
            }
        } else {
            match index {
                2 => self.js_operator_token(stack),
                3 => self.js_jsx(),
                _ => unreachable!(),
            }
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
        if (prev < 0 || is_lb(prev)) && stack.can_shift(self.g, py::BLANK_LINE_START) {
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

    #[inline]
    fn js_operator_token(&mut self, stack: &Stack) {
        let next = self.stream.next;
        if next == 43 || next == 45 {
            self.stream.advance(1);
            if next == self.stream.next {
                self.stream.advance(1);
                let may_postfix = stack.ctx == 0 && stack.can_shift(self.g, jst::INCDEC);
                self.stream.accept_token(if may_postfix { jst::INCDEC } else { jst::INCDEC_PREFIX }, 0);
            }
        } else if next == 63 && self.stream.peek(1) == 46 {
            self.stream.advance(1);
            self.stream.advance(1);
            let n = self.stream.next;
            if !(48..=57).contains(&n) {
                self.stream.accept_token(jst::QUESTION_DOT, 0);
            }
        }
    }

    #[inline]
    fn js_jsx(&mut self) {
        if self.stream.next != 60 || !self.g.dialect_flags[jst::DIALECT_JSX] {
            return;
        }
        let s = &mut self.stream;
        s.advance(1);
        if s.next == 47 {
            return;
        }
        let mut back = 0;
        while is_js_space(s.next) {
            s.advance(1);
            back += 1;
        }
        if ident_char(s.next, true) {
            s.advance(1);
            back += 1;
            while ident_char(s.next, false) {
                s.advance(1);
                back += 1;
            }
            while is_js_space(s.next) {
                s.advance(1);
                back += 1;
            }
            if s.next == 44 {
                return;
            }
            let ext = b"extends";
            let mut i = 0;
            loop {
                if i == 7 {
                    if !ident_char(s.next, true) {
                        return;
                    }
                    break;
                }
                if s.next != ext[i] as i32 {
                    break;
                }
                s.advance(1);
                back += 1;
                i += 1;
            }
        }
        s.accept_token(jst::JSX_START_TAG, -back);
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

#[inline(always)]
fn is_js_space(c: i32) -> bool {
    if c < 128 {
        matches!(c, 9 | 10 | 11 | 12 | 13 | 32)
    } else {
        matches!(c, 133 | 160 | 5760 | 8192..=8202 | 8232 | 8233 | 8239 | 8287 | 12288)
    }
}

#[inline(always)]
fn ident_char(ch: i32, start: bool) -> bool {
    (65..=90).contains(&ch)
        || (97..=122).contains(&ch)
        || ch == 95
        || ch >= 192
        || (!start && (48..=57).contains(&ch))
}

// ---------------------------------------------------------------------------
// Token cache (inline fixed arrays, zero per-token heap/resize overhead!)
// ---------------------------------------------------------------------------

const INLINE_ACT_CAP: usize = 48;

struct TokenCache {
    tokens: [CachedToken; 24],
    main_token: Option<CachedToken>,
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
    fn get_actions<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>, stack: &mut Stack) -> usize {
        let g = env.g;
        let mut action_index = 0usize;
        let mut main: Option<CachedToken> = None;
        let mask = g.state_slot(stack.state, 3);
        let context = if IS_PY { stack.ctx_hash } else { 0 };
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

            // Inline fast-reject / fast-emit for external tokenizers (eliminates millions of Stream::resets!)
            match tokenizer {
                Tok::External { contextual, fallback, extend } => {
                    if main.is_some() && !fallback {
                        continue;
                    }
                    if !IS_PY {
                        match i {
                            0 => {
                                // js_no_semicolon
                                if stack.ctx == 0
                                    && ch != -1
                                    && ch != 125
                                    && ch != 59
                                    && !is_js_space(ch)
                                {
                                    let is_comment = ch == 47 && {
                                        let a = env.stream.char_at(pos + 1);
                                        a == 47 || a == 42
                                    };
                                    if !is_comment {
                                        let si = action_index;
                                        action_index = self.add_actions(
                                            g,
                                            env.at,
                                            stack.state,
                                            jst::NO_SEMI,
                                            pos,
                                            action_index,
                                        );
                                        main = Some(CachedToken {
                                            start: pos,
                                            value: jst::NO_SEMI,
                                            end: pos,
                                            extended: -1,
                                            look_ahead: 0,
                                            mask: 0,
                                            context: 0,
                                        });
                                        if action_index > si {
                                            break;
                                        }
                                    }
                                }
                                continue;
                            }
                            1 => {
                                // js_no_semicolon_type
                                if ch == 91 && stack.ctx == 0 {
                                    let si = action_index;
                                    action_index = self.add_actions(
                                        g,
                                        env.at,
                                        stack.state,
                                        jst::NO_SEMI_TYPE,
                                        pos,
                                        action_index,
                                    );
                                    main = Some(CachedToken {
                                        start: pos,
                                        value: jst::NO_SEMI_TYPE,
                                        end: pos,
                                        extended: -1,
                                        look_ahead: 0,
                                        mask: 0,
                                        context: 0,
                                    });
                                    if action_index > si {
                                        break;
                                    }
                                }
                                continue;
                            }
                            2 => {
                                // js_operator_token: only '+', '-', '?'
                                if ch != 43 && ch != 45 && ch != 63 {
                                    continue;
                                }
                            }
                            3 => {
                                // js_jsx: only '<' when JSX dialect enabled
                                if ch != 60 || !g.dialect_flags[jst::DIALECT_JSX] {
                                    continue;
                                }
                            }
                            17 => {
                                // js_insert_semicolon (fallback: true)
                                if ch == 125 || ch == -1 || stack.ctx != 0 {
                                    let si = action_index;
                                    action_index = self.add_actions(
                                        g,
                                        env.at,
                                        stack.state,
                                        jst::INSERT_SEMI,
                                        pos,
                                        action_index,
                                    );
                                    main = Some(CachedToken {
                                        start: pos,
                                        value: jst::INSERT_SEMI,
                                        end: pos,
                                        extended: -1,
                                        look_ahead: 0,
                                        mask: 0,
                                        context: 0,
                                    });
                                    if action_index > si {
                                        break;
                                    }
                                }
                                continue;
                            }
                            _ => {}
                        }
                    } else {
                        match i {
                            0 => {
                                // py_legacy_print: only starts with 'p'
                                if ch != b'p' as i32 {
                                    continue;
                                }
                            }
                            1 => {
                                // py_indentation: only when flags == 0 and right after '\n' or '\r'
                                if stack.ctx_flags != 0 {
                                    continue;
                                }
                                let prev = env.stream.char_at(pos - 1);
                                if prev != 10 && prev != 13 {
                                    continue;
                                }
                            }
                            2 => {
                                // py_newlines
                                if ch < 0 {
                                    let si = action_index;
                                    action_index = self.add_actions(
                                        g,
                                        env.at,
                                        stack.state,
                                        py::EOF,
                                        pos,
                                        action_index,
                                    );
                                    main = Some(CachedToken {
                                        start: pos,
                                        value: py::EOF,
                                        end: pos,
                                        extended: -1,
                                        look_ahead: 0,
                                        mask: 0,
                                        context: 0,
                                    });
                                    if action_index > si {
                                        break;
                                    }
                                    continue;
                                }
                                if (stack.ctx_flags & CX_BRACKETED) != 0 {
                                    if ch == 10 || ch == 13 {
                                        let end = (pos + 1).min(env.stream.end);
                                        let si = action_index;
                                        action_index = self.add_actions(
                                            g,
                                            env.at,
                                            stack.state,
                                            py::NEWLINE_BRACKETED,
                                            end,
                                            action_index,
                                        );
                                        main = Some(CachedToken {
                                            start: pos,
                                            value: py::NEWLINE_BRACKETED,
                                            end,
                                            extended: -1,
                                            look_ahead: 0,
                                            mask: 0,
                                            context: 0,
                                        });
                                        if action_index > si {
                                            break;
                                        }
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
                                            env.at,
                                            stack.state,
                                            py::NEWLINE,
                                            end,
                                            action_index,
                                        );
                                        main = Some(CachedToken {
                                            start: pos,
                                            value: py::NEWLINE,
                                            end,
                                            extended: -1,
                                            look_ahead: 0,
                                            mask: 0,
                                            context: 0,
                                        });
                                        if action_index > si {
                                            break;
                                        }
                                    }
                                    continue;
                                }
                                // Line start (not bracketed): fall through to full `py_newlines`
                            }
                            _ => {}
                        }
                    }

                    // Slow path for remaining external invocations
                    let slot = unsafe { self.tokens.get_unchecked_mut(i) };
                    if contextual || slot.start != pos || slot.mask != mask || slot.context != context {
                        let start = env.stream.clip_pos(pos);
                        env.stream.reset(start, Some(*slot));
                        env.external::<IS_PY>(i, stack);
                        let mut t = env.stream.tok;
                        if t.value > -1 {
                            Self::apply_specialize(g, env.at, env.stream.input, &mut t);
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
                                env.at,
                                stack.state,
                                token.extended,
                                token.end,
                                action_index,
                            );
                        }
                        action_index = self.add_actions(
                            g,
                            env.at,
                            stack.state,
                            token.value,
                            token.end,
                            action_index,
                        );
                        if !extend {
                            main = Some(token);
                            if action_index > start_index {
                                break;
                            }
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
                            Self::apply_specialize(g, env.at, env.stream.input, &mut t);
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
                                env.at,
                                stack.state,
                                token.extended,
                                token.end,
                                action_index,
                            );
                        }
                        action_index = self.add_actions(
                            g,
                            env.at,
                            stack.state,
                            token.value,
                            token.end,
                            action_index,
                        );
                        main = Some(token);
                        if action_index > start_index {
                            break;
                        }
                    }
                }
                Tok::Local(k) => {
                    if main.is_some() {
                        continue;
                    }
                    let slot = unsafe { self.tokens.get_unchecked_mut(i) };
                    if slot.start != pos {
                        let start = env.stream.clip_pos(pos);
                        env.stream.reset(start, Some(*slot));
                        let lg = &g.locals[k as usize];
                        local_token(g, lg, &mut env.stream);
                        let mut t = env.stream.tok;
                        if t.value > -1 {
                            Self::apply_specialize(g, env.at, env.stream.input, &mut t);
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
                                env.at,
                                stack.state,
                                token.extended,
                                token.end,
                                action_index,
                            );
                        }
                        action_index = self.add_actions(
                            g,
                            env.at,
                            stack.state,
                            token.value,
                            token.end,
                            action_index,
                        );
                        main = Some(token);
                        if action_index > start_index {
                            break;
                        }
                    }
                }
            }
        }
        if look_ahead != 0 {
            stack.set_look_ahead(env.bufs, look_ahead);
        }
        if main.is_none() && pos == env.stream.end {
            let mut m = CachedToken::new();
            m.value = g.eof_term();
            m.start = pos;
            m.end = pos;
            action_index = self.add_actions(g, env.at, stack.state, m.value, m.end, action_index);
            main = Some(m);
        }
        self.main_token = main;
        action_index
    }

    #[inline(always)]
    fn apply_specialize<C: Ch>(g: &Grammar, at: &ActionTable, input: &[C], t: &mut CachedToken) {
        let tv = t.value as usize;
        if tv < at.spec_idx.len() {
            let si = unsafe { *at.spec_idx.get_unchecked(tv) };
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
    }

    fn get_main_token<C: Ch>(&self, env: &Env<C>, stack: &Stack) -> CachedToken {
        if let Some(m) = self.main_token {
            return m;
        }
        let mut main = CachedToken::new();
        let pos = stack.pos;
        main.start = pos;
        main.end = (pos + 1).min(env.stream.end);
        main.value = if pos == env.stream.end { env.g.eof_term() } else { TERM_ERR };
        main
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
        token: i32,
        end: i32,
        index: usize,
    ) -> usize {
        let s = state as usize;
        let t = token as usize;
        let row = unsafe { *at.row_of_state.get_unchecked(s) };
        if row == u16::MAX || t >= at.stride {
            return self.add_actions_slow(g, state, token, end, index);
        }
        let lut_idx = unsafe { *at.rows.get_unchecked(row as usize * at.stride + t) };
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

fn local_token<C: Ch>(g: &Grammar, lg: &LocalGroup, s: &mut Stream<C>) {
    let start = s.pos;
    let mut skipped = 0;
    loop {
        let at_eof = s.next < 0;
        let next_pos = if s.pos + 1 < s.end { Some(s.pos + 1) } else { None };
        read_token_local(g, lg.data, s, lg.prec);
        if s.tok.value > -1 {
            break;
        }
        if lg.else_token < 0 {
            return;
        }
        if !at_eof {
            skipped += 1;
        }
        match next_pos {
            None => break,
            Some(p) => s.reset_same_token(p),
        }
    }
    if skipped > 0 {
        s.reset_same_token(start);
        s.accept_token(lg.else_token, skipped);
    }
}

// ---------------------------------------------------------------------------
// Thread-local reusable workspace
// ---------------------------------------------------------------------------

struct LezerWorkspace {
    bufs: BufPool,
    ctxs: Vec<PyCtx>,
    tokens: TokenCache,
    reusable_stack: Vec<Frame>,
    flatten_scratch: Vec<Rec>,
    stacks: Vec<Stack>,
    new_stacks: Vec<Stack>,
    stopped: Vec<Stack>,
    stopped_tokens: Vec<i32>,
    work_stacks: Vec<Option<Stack>>,
}

impl LezerWorkspace {
    fn new() -> Self {
        Self {
            bufs: BufPool::default(),
            ctxs: Vec::with_capacity(256),
            tokens: TokenCache::new(),
            reusable_stack: Vec::with_capacity(64),
            flatten_scratch: Vec::new(),
            stacks: Vec::with_capacity(4),
            new_stacks: Vec::new(),
            stopped: Vec::new(),
            stopped_tokens: Vec::new(),
            work_stacks: Vec::new(),
        }
    }

    fn release(&mut self) {
        self.bufs.release();
        self.ctxs.clear();
        self.ctxs.shrink_to_fit();
        self.reusable_stack.clear();
        self.reusable_stack.shrink_to_fit();
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
// Parse driver (const-generic over `IS_PY`)
// ---------------------------------------------------------------------------

struct Parse<'a, 'w, C: Ch> {
    env: Env<'a, 'w, C>,
    tokens: &'w mut TokenCache,
    stacks: &'w mut Vec<Stack>,
    recovering: i32,
    min_stack_pos: i32,
    new_stacks: &'w mut Vec<Stack>,
    stopped: &'w mut Vec<Stack>,
    stopped_tokens: &'w mut Vec<i32>,
    work_stacks: &'w mut Vec<Option<Stack>>,
}

fn push_stack_dedup(stack: Stack, new_stacks: &mut Vec<Stack>) {
    for other in new_stacks.iter_mut() {
        if other.pos == stack.pos && other.same_state(&stack) {
            if other.score < stack.score {
                *other = stack;
            }
            return;
        }
    }
    new_stacks.push(stack);
}

fn find_finished(g: &Grammar, end: i32, stacks: &[Stack]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, stack) in stacks.iter().enumerate() {
        if stack.pos == end
            && g.state_flag(stack.state, STATE_ACCEPTING)
            && best.is_none_or(|b| stacks[b].score < stack.score)
        {
            best = Some(i);
        }
    }
    best
}

impl<'a, 'w, C: Ch> Parse<'a, 'w, C> {
    fn run_fast<const IS_PY: bool>(&mut self) -> Option<Stack> {
        let g = self.env.g;
        loop {
            if self.stacks.len() == 1 && self.recovering == 0 {
                let stack = &mut self.stacks[0];
                loop {
                    if self.env.big_count > 300 {
                        while stack.force_reduce::<IS_PY, C>(&mut self.env)
                            && !stack.stack.is_empty()
                            && stack.stack.last().unwrap().start >= self.env.last_big_start
                        {}
                        self.env.big_count = 0;
                        self.env.last_big_size = 0;
                    }

                    loop {
                        let dr = g.state_slot(stack.state, 4);
                        if dr == 0 {
                            break;
                        }
                        stack.reduce::<IS_PY, C>(&mut self.env, dr);
                    }

                    if stack.stack.len() >= 2800 {
                        while stack.stack.len() > 2000 && stack.force_reduce::<IS_PY, C>(&mut self.env) {}
                    }

                    self.tokens.main_token = None;
                    let n = self.tokens.get_actions::<IS_PY, C>(&mut self.env, stack);
                    if n == 3 {
                        let action = unsafe { *self.tokens.actions.get_unchecked(0) } as u32;
                        let term = unsafe { *self.tokens.actions.get_unchecked(1) };
                        let end = unsafe { *self.tokens.actions.get_unchecked(2) };
                        let st = self.tokens.main_token.map_or(stack.pos, |m| m.start);
                        stack.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
                        self.min_stack_pos = stack.pos;
                        continue;
                    }

                    if n == 0 {
                        if stack.pos == self.env.stream.end
                            && g.state_flag(stack.state, STATE_ACCEPTING)
                        {
                            return Some(self.stacks.pop().unwrap());
                        }
                        self.min_stack_pos = stack.pos;
                        break;
                    }

                    let start = stack.pos;
                    self.min_stack_pos = start;
                    let mut inline = [0i32; INLINE_ACT_CAP];
                    let mut heap = Vec::new();
                    self.tokens.copy_actions(n, &mut inline, &mut heap);
                    let actions: &[i32] = if n <= INLINE_ACT_CAP { &inline[..n] } else { &heap[..n] };
                    let main_start = self.tokens.main_token.map(|m| m.start);
                    let mut new_stacks_tmp = Vec::new();
                    let mut split_tmp = Vec::new();
                    let mut i = 0;
                    while i < n {
                        let action = actions[i] as u32;
                        let term = actions[i + 1];
                        let end = actions[i + 2];
                        i += 3;
                        if i == n {
                            let st = main_start.unwrap_or(stack.pos);
                            stack.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
                        } else {
                            let mut local = stack.split(self.env.bufs);
                            let st = main_start.unwrap_or(local.pos);
                            local.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
                            if local.pos > start {
                                new_stacks_tmp.push(local);
                            } else {
                                split_tmp.push(local);
                            }
                        }
                    }
                    let primary = self.stacks.pop().unwrap();
                    self.work_stacks.clear();
                    self.work_stacks.push(Some(primary));
                    for s in split_tmp {
                        self.work_stacks.push(Some(s));
                    }
                    self.new_stacks.clear();
                    self.new_stacks.extend(new_stacks_tmp);
                    if let Some(done) = self.finish_step::<IS_PY>(start) {
                        return Some(done);
                    }
                    break;
                }
            }

            if let Some(done) = self.advance_slow::<IS_PY>() {
                return Some(done);
            }
        }
    }

    fn advance_slow<const IS_PY: bool>(&mut self) -> Option<Stack> {
        let pos = self.min_stack_pos;
        self.work_stacks.clear();
        for s in self.stacks.drain(..) {
            self.work_stacks.push(Some(s));
        }
        self.new_stacks.clear();

        if self.env.big_count > 300 && self.work_stacks.len() == 1 {
            let s = self.work_stacks[0].as_mut().unwrap();
            while s.force_reduce::<IS_PY, C>(&mut self.env)
                && !s.stack.is_empty()
                && s.stack.last().unwrap().start >= self.env.last_big_start
            {}
            self.env.big_count = 0;
            self.env.last_big_size = 0;
        }

        self.finish_step::<IS_PY>(pos)
    }

    fn finish_step<const IS_PY: bool>(&mut self, pos: i32) -> Option<Stack> {
        self.stopped.clear();
        self.stopped_tokens.clear();

        let mut i = 0;
        while i < self.work_stacks.len() {
            let mut stack = self.work_stacks[i].take().unwrap();
            loop {
                self.tokens.main_token = None;
                if stack.pos > pos {
                    self.new_stacks.push(stack);
                } else if self.advance_stack::<IS_PY>(&mut stack, true) {
                    continue;
                } else {
                    let tok = self.tokens.get_main_token(&self.env, &stack);
                    self.stopped_tokens.push(tok.value);
                    self.stopped_tokens.push(tok.end);
                    self.stopped.push(stack);
                }
                break;
            }
            i += 1;
        }

        if self.new_stacks.is_empty() {
            if let Some(f) = find_finished(self.env.g, self.env.stream.end, self.stopped) {
                return Some(self.stopped.swap_remove(f));
            }
            if self.recovering == 0 {
                self.recovering = 5;
            }
        }
        if self.recovering != 0 && !self.stopped.is_empty() {
            let stopped_owned = std::mem::take(self.stopped);
            let tokens_copy = self.stopped_tokens.clone();
            if let Some(mut finished) = self.run_recovery::<IS_PY>(stopped_owned, &tokens_copy) {
                finished.force_all::<IS_PY, C>(&mut self.env);
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
                self.new_stacks.sort_by_key(|a| std::cmp::Reverse(a.score));
                self.new_stacks.truncate(max_remaining);
            }
            if self.new_stacks.iter().any(|s| s.reduce_pos > pos) {
                self.recovering -= 1;
            }
        } else if self.new_stacks.len() > 1 {
            let mut i: isize = 0;
            'outer: while (i as usize) + 1 < self.new_stacks.len() {
                let mut j = i as usize + 1;
                while j < self.new_stacks.len() {
                    let (si, sj) = (&self.new_stacks[i as usize], &self.new_stacks[j]);
                    let li = si.buf_len_i32(self.env.bufs);
                    let lj = sj.buf_len_i32(self.env.bufs);
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
                self.new_stacks.sort_by_key(|a| std::cmp::Reverse(a.score));
                self.new_stacks.truncate(12);
            }
        }
        self.min_stack_pos = self.new_stacks.iter().map(|s| s.pos).min().unwrap_or(pos);
        self.stacks.append(self.new_stacks);
        None
    }

    fn advance_stack<const IS_PY: bool>(&mut self, stack: &mut Stack, allow_split: bool) -> bool {
        let g = self.env.g;
        let default_reduce = g.state_slot(stack.state, 4);
        if default_reduce > 0 {
            stack.reduce::<IS_PY, C>(&mut self.env, default_reduce);
            return true;
        }
        if stack.stack.len() >= 2800 {
            while stack.stack.len() > 2000 && stack.force_reduce::<IS_PY, C>(&mut self.env) {}
        }
        let start = stack.pos;
        let n = self.tokens.get_actions::<IS_PY, C>(&mut self.env, stack);
        if n == 0 {
            return false;
        }
        let main_start = self.tokens.main_token.map(|m| m.start);
        if n == 3 || !allow_split {
            let action = self.tokens.actions[0] as u32;
            let term = self.tokens.actions[1];
            let end = self.tokens.actions[2];
            let st = main_start.unwrap_or(stack.pos);
            stack.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
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
                let st = main_start.unwrap_or(stack.pos);
                stack.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
                return true;
            }
            let mut local = stack.split(self.env.bufs);
            let st = main_start.unwrap_or(local.pos);
            local.apply::<IS_PY, C>(&mut self.env, action, term, st, end);
            if local.pos > start {
                self.new_stacks.push(local);
            } else {
                self.work_stacks.push(Some(local));
            }
        }
        false
    }

    fn advance_fully<const IS_PY: bool>(&mut self, mut stack: Stack) -> Result<(), Stack> {
        let pos = stack.pos;
        loop {
            if !self.advance_stack::<IS_PY>(&mut stack, false) {
                return Err(stack);
            }
            if stack.pos > pos {
                push_stack_dedup(stack, self.new_stacks);
                return Ok(());
            }
        }
    }

    fn run_recovery<const IS_PY: bool>(&mut self, stacks: Vec<Stack>, tokens: &[i32]) -> Option<Stack> {
        let g = self.env.g;
        let mut finished: Option<Stack> = None;
        let mut restarted = false;
        for (i, stack) in stacks.into_iter().enumerate() {
            let mut stack = stack;
            let mut token = tokens[i << 1];
            let mut token_end = tokens[(i << 1) + 1];
            if stack.dead_end(g) {
                if restarted {
                    continue;
                }
                restarted = true;
                stack.restart(self.env.bufs);
                match self.advance_fully::<IS_PY>(stack) {
                    Ok(()) => continue,
                    Err(s) => stack = s,
                }
            }
            let mut force = stack.split(self.env.bufs);
            for _ in 0..10 {
                if !force.force_reduce::<IS_PY, C>(&mut self.env) {
                    break;
                }
                match self.advance_fully::<IS_PY>(force) {
                    Ok(()) => {
                        force = stack.split(self.env.bufs);
                        force.pos = i32::MIN;
                        break;
                    }
                    Err(s) => force = s,
                }
            }
            for insert in stack.recover_by_insert::<IS_PY, C>(&mut self.env, token) {
                let _ = self.advance_fully::<IS_PY>(insert);
            }
            if self.env.stream.end > stack.pos {
                if token_end == stack.pos {
                    token_end += 1;
                    token = TERM_ERR;
                }
                stack.recover_by_delete(self.env.bufs, g, token, token_end);
                push_stack_dedup(stack, self.new_stacks);
            } else if force.pos != i32::MIN
                && finished.as_ref().is_none_or(|f| f.score < force.score)
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
        ws.bufs.reset();
        ws.ctxs.clear();
        ws.ctxs.push(PyCtx { parent: NULL_CTX, indent: 0, flags: 0, hash: 0 });
        ws.tokens.reset(g.tokenizers.len());
        ws.stacks.clear();
        ws.new_stacks.clear();
        ws.stopped.clear();
        ws.stopped_tokens.clear();
        ws.work_stacks.clear();

        let reusable = std::mem::take(&mut ws.reusable_stack);
        ws.stacks.push(Stack::start(g.top_state, 0, 0, reusable));

        let mut finished = {
            let mut p = Parse {
                env: Env {
                    g,
                    dfa,
                    at,
                    gt,
                    stream: Stream::new(input),
                    bufs: &mut ws.bufs,
                    ctxs: &mut ws.ctxs,
                    last_big_start: -1,
                    last_big_size: 0,
                    big_count: 0,
                },
                tokens: &mut ws.tokens,
                stacks: &mut ws.stacks,
                recovering: 0,
                min_stack_pos: 0,
                new_stacks: &mut ws.new_stacks,
                stopped: &mut ws.stopped,
                stopped_tokens: &mut ws.stopped_tokens,
                work_stacks: &mut ws.work_stacks,
            };
            if g.kind == Kind::Python {
                p.run_fast::<true>().unwrap()
            } else {
                p.run_fast::<false>().unwrap()
            }
        };

        finished.close(&mut ws.bufs, g.kind == Kind::Python);
        let buf_id = finished.buf_id;
        let end_pos = finished.pos as u32;
        ws.reusable_stack = finished.stack;
        if ws.reusable_stack.capacity() > 1024 {
            ws.reusable_stack.shrink_to(128);
        }

        let buf = ws.bufs.flatten_into(buf_id, &mut ws.flatten_scratch);
        build_tree_into(g, buf, end_pos, out);

        if ws.flatten_scratch.capacity() > 16384 {
            ws.flatten_scratch.clear();
            ws.flatten_scratch.shrink_to(4096);
        }
        if ws.ctxs.capacity() > 4096 {
            ws.ctxs.clear();
            ws.ctxs.shrink_to(256);
        }
    });
}
