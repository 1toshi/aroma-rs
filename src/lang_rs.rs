//! Rust language front-end: zero-dependency byte-slice lexer, Pass-1 local
//! scope binder, and error-recovering recursive-descent / Pratt parser directly
//! into `Body` arenas.

use std::cell::RefCell;
use std::collections::HashSet;

use crate::lang_py::FastMethod;
use crate::simple::{Body, KW, NAME};
use crate::text::NameSet;
use crate::{Language, Node, ParsedMethod, SnippetOptions};

const MAX_DEPTH: usize = 128;

/// Thread-local reusable buffers for the Rust front-end.
#[derive(Default)]
pub(crate) struct RsScratch {
    line_starts: Vec<u32>,
    toks: Vec<RsTok>,
    bound: NameSet,
}

impl RsScratch {
    fn release(&mut self) {
        self.line_starts.clear();
        self.line_starts.shrink_to_fit();
        self.toks.clear();
        self.toks.shrink_to_fit();
        self.bound.release();
    }
}

thread_local! {
    static TL_RS: RefCell<RsScratch> = RefCell::new(RsScratch::default());
}

pub(crate) fn shrink_tl_scratch() {
    TL_RS.with(|cell| cell.borrow_mut().release());
}

// ============================================================================
// Token kinds & byte-slice lexer
// ============================================================================

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum TK {
    Ident = 1,
    Literal,
    Lifetime,
    // Control / syntactic keywords (emitted as `KW`)
    KwFn,
    KwLet,
    KwFor,
    KwIn,
    KwIf,
    KwElse,
    KwWhile,
    KwLoop,
    KwMatch,
    KwImpl,
    KwTrait,
    KwMod,
    KwPub,
    KwStruct,
    KwEnum,
    KwType,
    KwConst,
    KwStatic,
    KwUnsafe,
    KwAsync,
    KwAwait,
    KwExtern,
    KwUse,
    KwWhere,
    KwAs,
    KwReturn,
    KwBreak,
    KwContinue,
    KwMut,
    KwRef,
    KwMove,
    // Special identifiers (emitted as `NAME`)
    KwSelfLower,
    KwSelfUpper,
    KwCrate,
    KwSuper,
    KwTrue,
    KwFalse,
    // Punctuation / Operators
    Eq,
    AssignOp,
    EqEq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
    AndAnd,
    OrOr,
    Pipe,
    Amp,
    Caret,
    Shl,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Question,
    At,
    Hash,
    Dollar,
    Tilde,
    Dot,
    DotDot,
    DotDotEq,
    DotDotDot,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Arrow,
    FatArrow,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
}

impl TK {
    #[inline(always)]
    fn is_ident_like(self) -> bool {
        matches!(self, TK::Ident | TK::KwSelfLower)
    }

    #[inline(always)]
    fn is_name_like(self) -> bool {
        matches!(
            self,
            TK::Ident
                | TK::KwSelfLower
                | TK::KwSelfUpper
                | TK::KwCrate
                | TK::KwSuper
                | TK::KwTrue
                | TK::KwFalse
        )
    }

    #[inline(always)]
    fn is_modifier_kw(self) -> bool {
        matches!(
            self,
            TK::KwPub | TK::KwConst | TK::KwAsync | TK::KwUnsafe | TK::KwExtern
        )
    }
}

#[derive(Clone, Copy)]
struct RsTok {
    kind: TK,
    start: u32,
    end: u32,
}

#[inline(always)]
fn is_ascii_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline(always)]
fn is_ascii_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn decode_utf8_char(slice: &[u8]) -> Option<(char, usize)> {
    let s = std::str::from_utf8(slice).ok()?;
    let ch = s.chars().next()?;
    Some((ch, ch.len_utf8()))
}

#[inline]
fn next_utf8_len(bytes: &[u8], i: usize) -> usize {
    if bytes[i] < 0x80 {
        return 1;
    }
    let end = (i + 4).min(bytes.len());
    for l in 1..=(end - i) {
        if let Ok(s) = std::str::from_utf8(&bytes[i..i + l])
            && let Some(ch) = s.chars().next()
        {
            return ch.len_utf8();
        }
    }
    1
}

#[inline]
fn is_ident_start_at(bytes: &[u8], i: usize) -> bool {
    if i >= bytes.len() {
        return false;
    }
    let b = bytes[i];
    if b < 0x80 {
        return is_ascii_ident_start(b);
    }
    let end = (i + 4).min(bytes.len());
    for l in 2..=(end - i) {
        if let Ok(s) = std::str::from_utf8(&bytes[i..i + l])
            && let Some(ch) = s.chars().next()
        {
            return ch.is_alphabetic() || ch == '_';
        }
    }
    false
}

#[inline]
fn scan_ident(bytes: &[u8], start: usize) -> usize {
    let n = bytes.len();
    let mut i = start;
    while i < n {
        let b = bytes[i];
        if b < 0x80 {
            if is_ascii_ident_continue(b) {
                i += 1;
            } else {
                break;
            }
        } else {
            let end = (i + 4).min(n);
            let mut adv = 0usize;
            for l in 2..=(end - i) {
                if let Ok(s) = std::str::from_utf8(&bytes[i..i + l])
                    && let Some(ch) = s.chars().next()
                {
                    if ch.is_alphanumeric() || ch == '_' {
                        adv = ch.len_utf8();
                    }
                    break;
                }
            }
            if adv == 0 {
                break;
            }
            i += adv;
        }
    }
    i
}

#[inline]
fn keyword_or_ident(word: &[u8]) -> TK {
    match word {
        b"fn" => TK::KwFn,
        b"let" => TK::KwLet,
        b"for" => TK::KwFor,
        b"in" => TK::KwIn,
        b"if" => TK::KwIf,
        b"else" => TK::KwElse,
        b"while" => TK::KwWhile,
        b"loop" => TK::KwLoop,
        b"match" => TK::KwMatch,
        b"impl" => TK::KwImpl,
        b"trait" => TK::KwTrait,
        b"mod" => TK::KwMod,
        b"pub" => TK::KwPub,
        b"struct" => TK::KwStruct,
        b"enum" => TK::KwEnum,
        b"type" => TK::KwType,
        b"const" => TK::KwConst,
        b"static" => TK::KwStatic,
        b"unsafe" => TK::KwUnsafe,
        b"async" => TK::KwAsync,
        b"await" => TK::KwAwait,
        b"extern" => TK::KwExtern,
        b"use" => TK::KwUse,
        b"where" => TK::KwWhere,
        b"as" => TK::KwAs,
        b"return" => TK::KwReturn,
        b"break" => TK::KwBreak,
        b"continue" => TK::KwContinue,
        b"mut" => TK::KwMut,
        b"ref" => TK::KwRef,
        b"move" => TK::KwMove,
        b"self" => TK::KwSelfLower,
        b"Self" => TK::KwSelfUpper,
        b"crate" => TK::KwCrate,
        b"super" => TK::KwSuper,
        b"true" => TK::KwTrue,
        b"false" => TK::KwFalse,
        _ => TK::Ident,
    }
}

#[inline]
fn skip_line_comment(bytes: &[u8], i: usize) -> usize {
    match memchr::memchr(b'\n', &bytes[i..]) {
        Some(rel) => i + rel + 1,
        None => bytes.len(),
    }
}

#[inline]
fn skip_block_comment(bytes: &[u8], mut j: usize) -> usize {
    let n = bytes.len();
    let mut depth = 1usize;
    while j < n && depth > 0 {
        let Some(rel) = memchr::memchr2(b'/', b'*', &bytes[j..]) else {
            return n;
        };
        let p = j + rel;
        if bytes[p] == b'/' && p + 1 < n && bytes[p + 1] == b'*' {
            depth += 1;
            j = p + 2;
        } else if bytes[p] == b'*' && p + 1 < n && bytes[p + 1] == b'/' {
            depth -= 1;
            j = p + 2;
        } else {
            j = p + 1;
        }
    }
    j
}

#[inline]
fn scan_string(bytes: &[u8], mut j: usize) -> usize {
    let n = bytes.len();
    while j < n {
        let Some(rel) = memchr::memchr2(b'"', b'\\', &bytes[j..]) else {
            return n;
        };
        let p = j + rel;
        if bytes[p] == b'"' {
            return p + 1;
        }
        j = (p + 2).min(n);
    }
    n
}

#[inline]
fn scan_raw_string(bytes: &[u8], mut j: usize, hashes: usize) -> usize {
    let n = bytes.len();
    while j < n {
        let Some(rel) = memchr::memchr(b'"', &bytes[j..]) else {
            return n;
        };
        let p = j + rel;
        if p + 1 + hashes <= n && bytes[p + 1..p + 1 + hashes].iter().all(|&b| b == b'#') {
            return p + 1 + hashes;
        }
        j = p + 1;
    }
    n
}

#[inline]
fn scan_escaped_char_or_byte(bytes: &[u8], start: usize) -> usize {
    let n = bytes.len();
    let mut j = start + 1;
    if j < n && bytes[j] == b'\\' {
        j += 1;
        if j < n {
            j += next_utf8_len(bytes, j);
        }
    }
    let limit = (start + 24).min(n);
    while j < limit {
        let b = bytes[j];
        if b == b'\'' {
            return j + 1;
        }
        if b == b'\n' || b == b'\r' {
            return j;
        }
        j += 1;
    }
    j
}

fn lex_rust(bytes: &[u8], line_starts: &mut Vec<u32>, toks: &mut Vec<RsTok>) {
    line_starts.clear();
    line_starts.push(0);
    for pos in memchr::memchr_iter(b'\n', bytes) {
        line_starts.push(pos as u32 + 1);
    }

    toks.clear();
    let n = bytes.len();
    let mut i = 0usize;

    // Optional Unix shebang `#!` at byte 0 (excluding `#![attr]`)
    if n >= 2 && bytes[0] == b'#' && bytes[1] == b'!' && (n == 2 || bytes[2] != b'[') {
        i = skip_line_comment(bytes, 2);
    }

    while i < n {
        let b = bytes[i];
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            i += 1;
            continue;
        }

        if b == b'/' && i + 1 < n {
            if bytes[i + 1] == b'/' {
                i = skip_line_comment(bytes, i + 2);
                continue;
            }
            if bytes[i + 1] == b'*' {
                i = skip_block_comment(bytes, i + 2);
                continue;
            }
        }

        // Raw strings `r"..."` / `r#"..."#` and raw identifiers `r#ident`
        if b == b'r' {
            let mut h = 0usize;
            while i + 1 + h < n && bytes[i + 1 + h] == b'#' {
                h += 1;
            }
            if i + 1 + h < n && bytes[i + 1 + h] == b'"' {
                let end = scan_raw_string(bytes, i + 2 + h, h);
                toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
            if h == 1 && is_ident_start_at(bytes, i + 2) {
                let end = scan_ident(bytes, i + 2);
                toks.push(RsTok { kind: TK::Ident, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
        }

        // Byte / C strings, raw byte / C strings, byte char literals
        if b == b'b' || b == b'c' {
            if i + 1 < n && bytes[i + 1] == b'"' {
                let end = scan_string(bytes, i + 2);
                toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
            if i + 1 < n && bytes[i + 1] == b'r' {
                let mut h = 0usize;
                while i + 2 + h < n && bytes[i + 2 + h] == b'#' {
                    h += 1;
                }
                if i + 2 + h < n && bytes[i + 2 + h] == b'"' {
                    let end = scan_raw_string(bytes, i + 3 + h, h);
                    toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                    i = end;
                    continue;
                }
            }
            if b == b'b' && i + 1 < n && bytes[i + 1] == b'\'' {
                let end = if i + 2 < n && bytes[i + 2] == b'\\' {
                    scan_escaped_char_or_byte(bytes, i + 1)
                } else if i + 3 < n && bytes[i + 3] == b'\'' {
                    i + 4
                } else {
                    scan_escaped_char_or_byte(bytes, i + 1)
                };
                toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
        }

        if b == b'"' {
            let end = scan_string(bytes, i + 1);
            toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
            i = end;
            continue;
        }

        // Char literals vs lifetimes
        if b == b'\'' {
            if i + 1 < n && bytes[i + 1] == b'\\' {
                let end = scan_escaped_char_or_byte(bytes, i);
                toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
            let clen = if i + 1 < n { next_utf8_len(bytes, i + 1) } else { 0 };
            if i + 1 < n
                && clen > 0
                && !matches!(bytes[i + 1], b'\'' | b'\n' | b'\r')
                && i + 1 + clen < n
                && bytes[i + 1 + clen] == b'\''
            {
                let end = i + 2 + clen;
                toks.push(RsTok { kind: TK::Literal, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
            if is_ident_start_at(bytes, i + 1) {
                let end = scan_ident(bytes, i + 1);
                toks.push(RsTok { kind: TK::Lifetime, start: i as u32, end: end as u32 });
                i = end;
                continue;
            }
            i += 1;
            continue;
        }

        // Numbers (preserving `t.0.1`, `0..10`, `0..=n`, `0.abs()`)
        if b.is_ascii_digit() {
            let prev_is_dot = toks.last().is_some_and(|t| t.kind == TK::Dot);
            let mut j = i;
            if prev_is_dot {
                while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                    j += 1;
                }
            } else {
                if b == b'0' && j + 1 < n && matches!(bytes[j + 1], b'x' | b'X' | b'b' | b'B' | b'o' | b'O') {
                    let radix = bytes[j + 1];
                    j += 2;
                    while j < n {
                        let d = bytes[j];
                        let ok = match radix {
                            b'x' | b'X' => d.is_ascii_hexdigit() || d == b'_',
                            b'b' | b'B' => matches!(d, b'0' | b'1' | b'_'),
                            _ => matches!(d, b'0'..=b'7' | b'_'),
                        };
                        if !ok {
                            break;
                        }
                        j += 1;
                    }
                } else {
                    while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                        j += 1;
                    }
                    if j + 1 < n && bytes[j] == b'.' && bytes[j + 1].is_ascii_digit() {
                        j += 1;
                        while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                            j += 1;
                        }
                    }
                    if j < n && matches!(bytes[j], b'e' | b'E') {
                        let mut e = j + 1;
                        if e < n && matches!(bytes[e], b'+' | b'-') {
                            e += 1;
                        }
                        if e < n && bytes[e].is_ascii_digit() {
                            j = e + 1;
                            while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                                j += 1;
                            }
                        }
                    }
                }
                if is_ident_start_at(bytes, j) {
                    j = scan_ident(bytes, j);
                }
            }
            toks.push(RsTok { kind: TK::Literal, start: i as u32, end: j as u32 });
            i = j;
            continue;
        }

        if is_ident_start_at(bytes, i) {
            let end = scan_ident(bytes, i);
            let kind = keyword_or_ident(&bytes[i..end]);
            toks.push(RsTok { kind, start: i as u32, end: end as u32 });
            i = end;
            continue;
        }

        let b1 = if i + 1 < n { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < n { bytes[i + 2] } else { 0 };
        let (kind, adv) = match (b, b1, b2) {
            (b'.', b'.', b'=') => (TK::DotDotEq, 3),
            (b'.', b'.', b'.') => (TK::DotDotDot, 3),
            (b'.', b'.', _) => (TK::DotDot, 2),
            (b'.', _, _) => (TK::Dot, 1),
            (b'<', b'<', b'=') => (TK::AssignOp, 3),
            (b'<', b'<', _) => (TK::Shl, 2),
            (b'<', b'=', _) => (TK::Le, 2),
            (b'<', _, _) => (TK::Lt, 1),
            (b'>', b'=', _) => (TK::Ge, 2),
            (b'>', _, _) => (TK::Gt, 1),
            (b'=', b'=', _) => (TK::EqEq, 2),
            (b'=', b'>', _) => (TK::FatArrow, 2),
            (b'=', _, _) => (TK::Eq, 1),
            (b'!', b'=', _) => (TK::Ne, 2),
            (b'!', _, _) => (TK::Bang, 1),
            (b'&', b'&', _) => (TK::AndAnd, 2),
            (b'&', b'=', _) => (TK::AssignOp, 2),
            (b'&', _, _) => (TK::Amp, 1),
            (b'|', b'|', _) => (TK::OrOr, 2),
            (b'|', b'=', _) => (TK::AssignOp, 2),
            (b'|', _, _) => (TK::Pipe, 1),
            (b'^', b'=', _) => (TK::AssignOp, 2),
            (b'^', _, _) => (TK::Caret, 1),
            (b'+', b'=', _) => (TK::AssignOp, 2),
            (b'+', _, _) => (TK::Plus, 1),
            (b'-', b'>', _) => (TK::Arrow, 2),
            (b'-', b'=', _) => (TK::AssignOp, 2),
            (b'-', _, _) => (TK::Minus, 1),
            (b'*', b'=', _) => (TK::AssignOp, 2),
            (b'*', _, _) => (TK::Star, 1),
            (b'/', b'=', _) => (TK::AssignOp, 2),
            (b'/', _, _) => (TK::Slash, 1),
            (b'%', b'=', _) => (TK::AssignOp, 2),
            (b'%', _, _) => (TK::Percent, 1),
            (b':', b':', _) => (TK::ColonColon, 2),
            (b':', _, _) => (TK::Colon, 1),
            (b';', _, _) => (TK::Semi, 1),
            (b',', _, _) => (TK::Comma, 1),
            (b'?', _, _) => (TK::Question, 1),
            (b'@', _, _) => (TK::At, 1),
            (b'#', _, _) => (TK::Hash, 1),
            (b'$', _, _) => (TK::Dollar, 1),
            (b'~', _, _) => (TK::Tilde, 1),
            (b'(', _, _) => (TK::LParen, 1),
            (b')', _, _) => (TK::RParen, 1),
            (b'[', _, _) => (TK::LBracket, 1),
            (b']', _, _) => (TK::RBracket, 1),
            (b'{', _, _) => (TK::LBrace, 1),
            (b'}', _, _) => (TK::RBrace, 1),
            _ => {
                i += next_utf8_len(bytes, i);
                continue;
            }
        };
        toks.push(RsTok { kind, start: i as u32, end: (i + adv) as u32 });
        i += adv;
    }
}

// ============================================================================
// Helpers
// ============================================================================

#[inline(always)]
fn line_of(ls: &[u32], offset: u32) -> u32 {
    ls.partition_point(|&s| s <= offset) as u32
}

#[inline(always)]
fn line_of_cursor(ls: &[u32], offset: u32, cur: &mut usize) -> u32 {
    let n = ls.len();
    let mut i = *cur;
    if i < n && unsafe { *ls.get_unchecked(i) } <= offset {
        if i + 1 == n || offset < unsafe { *ls.get_unchecked(i + 1) } {
            return (i + 1) as u32;
        }
        i += 1;
        if i + 1 == n || offset < unsafe { *ls.get_unchecked(i + 1) } {
            *cur = i;
            return (i + 1) as u32;
        }
        let limit = (i + 4).min(n);
        while i + 1 < limit && unsafe { *ls.get_unchecked(i + 1) } <= offset {
            i += 1;
        }
        if i + 1 == n || offset < unsafe { *ls.get_unchecked(i + 1) } {
            *cur = i;
            return (i + 1) as u32;
        }
    }
    let idx = ls.partition_point(|&s| s <= offset).saturating_sub(1);
    *cur = idx;
    (idx + 1) as u32
}

#[inline]
fn tok_bytes(bytes: &[u8], t: RsTok) -> &[u8] {
    &bytes[t.start as usize..t.end as usize]
}

#[inline]
fn tok_str(source: &str, t: RsTok) -> &str {
    &source[t.start as usize..t.end as usize]
}

#[inline]
fn ident_is_uppercase(bytes: &[u8], start: u32, end: u32) -> bool {
    let s = &bytes[start as usize..end as usize];
    let head = if s.starts_with(b"r#") { &s[2..] } else { s };
    match head.first().copied() {
        Some(b) if b < 0x80 => b.is_ascii_uppercase(),
        Some(_) => decode_utf8_char(head).is_some_and(|(ch, _)| ch.is_uppercase()),
        None => false,
    }
}

#[inline]
fn looks_global_rs(bytes: &[u8], source: &str, t: RsTok, globals: Option<&HashSet<String>>) -> bool {
    if t.kind != TK::Ident && t.kind != TK::KwSelfLower {
        return true;
    }
    if tok_bytes(bytes, t) == b"_" || ident_is_uppercase(bytes, t.start, t.end) {
        return true;
    }
    globals.is_some_and(|g| g.contains(tok_str(source, t)))
}

/// Returns `(matching_idx, closed)`.
fn find_matching_delim_checked(toks: &[RsTok], open_idx: usize, limit: usize) -> (usize, bool) {
    let open = toks[open_idx].kind;
    let close = match open {
        TK::LParen => TK::RParen,
        TK::LBracket => TK::RBracket,
        TK::LBrace => TK::RBrace,
        TK::Lt => TK::Gt,
        _ => return (open_idx, true),
    };
    let mut depth = 1usize;
    let mut i = open_idx + 1;
    while i < limit {
        let k = toks[i].kind;
        if k == open {
            depth += 1;
        } else if k == close {
            depth -= 1;
            if depth == 0 {
                return (i, true);
            }
        }
        i += 1;
    }
    (limit.saturating_sub(1), false)
}

#[inline]
fn find_matching_delim(toks: &[RsTok], open_idx: usize, limit: usize) -> usize {
    find_matching_delim_checked(toks, open_idx, limit).0
}

fn find_matching_fn_paren(toks: &[RsTok], open_idx: usize, limit: usize) -> Option<usize> {
    let mut paren = 1usize;
    let mut bracket = 0usize;
    let mut i = open_idx + 1;
    while i < limit {
        match toks[i].kind {
            TK::LParen => paren += 1,
            TK::RParen => {
                paren -= 1;
                if paren == 0 {
                    return Some(i);
                }
            }
            TK::LBracket => bracket += 1,
            TK::RBracket => bracket = bracket.saturating_sub(1),
            TK::LBrace | TK::Semi if paren == 1 && bracket == 0 => return None,
            _ => {}
        }
        i += 1;
    }
    None
}

fn skip_angles(toks: &[RsTok], lt_idx: usize, limit: usize) -> usize {
    if lt_idx >= limit || toks[lt_idx].kind != TK::Lt {
        return lt_idx;
    }
    let mut angle = 1usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut i = lt_idx + 1;
    while i < limit {
        match toks[i].kind {
            TK::LParen => paren += 1,
            TK::RParen => paren = paren.saturating_sub(1),
            TK::LBracket => bracket += 1,
            TK::RBracket => bracket = bracket.saturating_sub(1),
            TK::LBrace => {
                if paren == 0 && bracket == 0 && brace == 0 {
                    return i;
                }
                brace += 1;
            }
            TK::RBrace => {
                if paren == 0 && bracket == 0 && brace == 0 {
                    return i;
                }
                brace = brace.saturating_sub(1);
            }
            TK::Semi if paren == 0 && bracket == 0 && brace == 0 => return i,
            TK::Lt if paren == 0 && bracket == 0 && brace == 0 => angle += 1,
            TK::Gt if paren == 0 && bracket == 0 && brace == 0 => {
                angle -= 1;
                if angle == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    i
}

#[inline]
fn is_contiguous_shr(toks: &[RsTok], i: usize, limit: usize) -> bool {
    i + 1 < limit
        && toks[i].kind == TK::Gt
        && toks[i + 1].kind == TK::Gt
        && toks[i].end == toks[i + 1].start
}

#[inline]
fn is_contiguous_shr_eq(toks: &[RsTok], i: usize, limit: usize) -> bool {
    i + 1 < limit
        && toks[i].kind == TK::Gt
        && toks[i + 1].kind == TK::Ge
        && toks[i].end == toks[i + 1].start
}

// ============================================================================
// Item walker (`mod`, `impl`, `trait`, `fn`)
// ============================================================================

struct RawFnItem {
    name: String,
    start_tok: usize,
    end_tok: usize,
    params_start: usize,
    params_end: usize,
    body_start: usize,
    body_end: usize,
}

fn skip_attributes(toks: &[RsTok], mut i: usize, limit: usize) -> usize {
    while i < limit && toks[i].kind == TK::Hash {
        if i + 1 < limit && toks[i + 1].kind == TK::Bang && i + 2 < limit && toks[i + 2].kind == TK::LBracket {
            i = (find_matching_delim(toks, i + 2, limit) + 1).min(limit);
        } else if i + 1 < limit && toks[i + 1].kind == TK::LBracket {
            i = (find_matching_delim(toks, i + 1, limit) + 1).min(limit);
        } else {
            break;
        }
    }
    i
}

fn skip_visibility_and_modifiers(bytes: &[u8], toks: &[RsTok], mut i: usize, limit: usize) -> usize {
    while i < limit {
        let k = toks[i].kind;
        if k == TK::KwPub {
            i += 1;
            if i < limit && toks[i].kind == TK::LParen {
                i = (find_matching_delim(toks, i, limit) + 1).min(limit);
            }
            continue;
        }
        if k == TK::KwExtern {
            i += 1;
            if i < limit && toks[i].kind == TK::Literal {
                i += 1;
            }
            continue;
        }
        if k.is_modifier_kw() || (k == TK::Ident && tok_bytes(bytes, toks[i]) == b"default") {
            if k == TK::KwConst && !(i + 1 < limit && matches!(toks[i + 1].kind, TK::KwFn | TK::KwUnsafe | TK::KwExtern)) {
                break;
            }
            i += 1;
            continue;
        }
        break;
    }
    i
}

fn skip_to_semi_or_block(toks: &[RsTok], mut i: usize, limit: usize) -> usize {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    while i < limit {
        match toks[i].kind {
            TK::LParen => paren += 1,
            TK::RParen => paren = paren.saturating_sub(1),
            TK::LBracket => bracket += 1,
            TK::RBracket => bracket = bracket.saturating_sub(1),
            TK::Semi if paren == 0 && bracket == 0 => return i + 1,
            TK::LBrace if paren == 0 && bracket == 0 => {
                let (end, closed) = find_matching_delim_checked(toks, i, limit);
                return if closed { (end + 1).min(limit) } else { i + 1 };
            }
            TK::KwFn | TK::KwPub | TK::KwImpl | TK::KwTrait | TK::KwMod | TK::KwStruct | TK::KwEnum
                if paren == 0 && bracket == 0 =>
            {
                return i;
            }
            _ => {}
        }
        i += 1;
    }
    i
}

fn extract_concise_type(source: &str, toks: &[RsTok], start: usize, end: usize) -> String {
    if start >= end {
        return "impl".to_string();
    }
    let mut last_seg: Option<(u32, u32)> = None;
    let mut saw_as = false;
    let mut i = start;
    while i < end {
        match toks[i].kind {
            TK::Lt => {
                i = skip_angles(toks, i, end);
                continue;
            }
            TK::LParen | TK::LBracket | TK::LBrace => {
                let m = find_matching_delim(toks, i, end);
                if last_seg.is_none() && toks[i].kind != TK::LBrace {
                    let s = &source[toks[i].start as usize..toks[m].end as usize];
                    if s.len() <= 32 {
                        return s.split_whitespace().collect::<Vec<_>>().join(" ");
                    }
                }
                i = (m + 1).min(end);
                continue;
            }
            TK::KwAs => saw_as = true,
            k if k.is_name_like() && !saw_as => {
                last_seg = Some((toks[i].start, toks[i].end));
            }
            _ => {}
        }
        i += 1;
    }
    if let Some((a, b)) = last_seg {
        return source[a as usize..b as usize].to_string();
    }
    "impl".to_string()
}

fn find_impl_context(source: &str, toks: &[RsTok], header_start: usize, lbrace: usize) -> String {
    let mut i = header_start;
    if i < lbrace && toks[i].kind == TK::Lt {
        i = skip_angles(toks, i, lbrace);
    }
    let ty_start = i;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut top_for: Option<usize> = None;
    let mut top_where: Option<usize> = None;

    while i < lbrace {
        match toks[i].kind {
            TK::LParen => paren += 1,
            TK::RParen => paren = paren.saturating_sub(1),
            TK::LBracket => bracket += 1,
            TK::RBracket => bracket = bracket.saturating_sub(1),
            TK::Lt if paren == 0 && bracket == 0 => {
                i = skip_angles(toks, i, lbrace);
                continue;
            }
            TK::KwWhere if paren == 0 && bracket == 0 => {
                top_where = Some(i);
                break;
            }
            TK::KwFor
                if paren == 0
                    && bracket == 0
                    && i + 1 < lbrace
                    && toks[i + 1].kind != TK::Lt
                    && top_for.is_none() =>
            {
                top_for = Some(i);
            }
            _ => {}
        }
        i += 1;
    }

    let hdr_end = top_where.unwrap_or(lbrace);
    if let Some(for_idx) = top_for {
        let trait_name = extract_concise_type(source, toks, ty_start, for_idx);
        let self_name = extract_concise_type(source, toks, for_idx + 1, hdr_end);
        format!("<{self_name} as {trait_name}>")
    } else {
        extract_concise_type(source, toks, ty_start, hdr_end)
    }
}

fn push_qualified_name(ctx_stack: &[String], leaf: &str) -> String {
    if ctx_stack.is_empty() {
        return leaf.to_string();
    }
    let mut out = String::new();
    for (idx, c) in ctx_stack.iter().enumerate() {
        if idx > 0 {
            out.push_str("::");
        }
        out.push_str(c);
    }
    out.push_str("::");
    out.push_str(leaf);
    out
}

fn walk_items(
    source: &str,
    bytes: &[u8],
    toks: &[RsTok],
    range_start: usize,
    range_end: usize,
    ctx_stack: &mut Vec<String>,
    out: &mut Vec<RawFnItem>,
) {
    let mut i = range_start;
    while i < range_end {
        let after_attr = skip_attributes(toks, i, range_end);
        if after_attr >= range_end {
            break;
        }
        let item_start = after_attr;
        let kw_idx = skip_visibility_and_modifiers(bytes, toks, after_attr, range_end);
        if kw_idx >= range_end {
            break;
        }

        match toks[kw_idx].kind {
            TK::KwFn => {
                let mut p = kw_idx + 1;
                if p >= range_end || !toks[p].kind.is_name_like() {
                    i = kw_idx + 1;
                    continue;
                }
                let fn_name = tok_str(source, toks[p]);
                p += 1;
                if p < range_end && toks[p].kind == TK::Lt {
                    p = skip_angles(toks, p, range_end);
                }
                if p >= range_end || toks[p].kind != TK::LParen {
                    i = kw_idx + 1;
                    continue;
                }
                let Some(rparen) = find_matching_fn_paren(toks, p, range_end) else {
                    // Malformed missing `)` in fn header: skip header conservatively to recover siblings
                    i = p + 1;
                    continue;
                };
                let params_start = p + 1;
                let params_end = rparen;
                p = (rparen + 1).min(range_end);

                let mut paren = 0usize;
                let mut bracket = 0usize;
                let mut found_brace: Option<usize> = None;
                let mut found_semi: Option<usize> = None;
                while p < range_end {
                    match toks[p].kind {
                        TK::LParen => paren += 1,
                        TK::RParen => paren = paren.saturating_sub(1),
                        TK::LBracket => bracket += 1,
                        TK::RBracket => bracket = bracket.saturating_sub(1),
                        TK::Semi if paren == 0 && bracket == 0 => {
                            found_semi = Some(p);
                            break;
                        }
                        TK::LBrace if paren == 0 && bracket == 0 => {
                            found_brace = Some(p);
                            break;
                        }
                        TK::KwFn | TK::KwStruct | TK::KwEnum | TK::KwImpl | TK::KwTrait | TK::KwMod
                            if paren == 0 && bracket == 0 =>
                        {
                            break;
                        }
                        _ => {}
                    }
                    p += 1;
                }

                if let Some(semi_idx) = found_semi {
                    i = semi_idx + 1;
                    continue;
                }
                if let Some(lbrace) = found_brace {
                    let (rbrace, closed) = find_matching_delim_checked(toks, lbrace, range_end);
                    out.push(RawFnItem {
                        name: push_qualified_name(ctx_stack, fn_name),
                        start_tok: item_start,
                        end_tok: rbrace,
                        params_start,
                        params_end,
                        body_start: lbrace,
                        body_end: (rbrace + 1).min(range_end),
                    });
                    i = if closed { (rbrace + 1).min(range_end) } else { lbrace + 1 };
                    continue;
                }
                i = p.max(kw_idx + 1);
            }
            TK::KwMod => {
                let mut p = kw_idx + 1;
                if p < range_end && toks[p].kind.is_name_like() {
                    let mod_name = tok_str(source, toks[p]).to_string();
                    p += 1;
                    if p < range_end && toks[p].kind == TK::LBrace {
                        let (rbrace, closed) = find_matching_delim_checked(toks, p, range_end);
                        ctx_stack.push(mod_name);
                        walk_items(
                            source,
                            bytes,
                            toks,
                            p + 1,
                            if closed { rbrace } else { range_end },
                            ctx_stack,
                            out,
                        );
                        ctx_stack.pop();
                        i = if closed { (rbrace + 1).min(range_end) } else { range_end };
                        continue;
                    }
                    i = skip_to_semi_or_block(toks, p, range_end);
                } else {
                    i = kw_idx + 1;
                }
            }
            TK::KwTrait => {
                let mut p = kw_idx + 1;
                if p < range_end && toks[p].kind == TK::KwUnsafe {
                    p += 1;
                }
                if p < range_end && toks[p].kind.is_name_like() {
                    let trait_name = tok_str(source, toks[p]).to_string();
                    p += 1;
                    let mut paren = 0usize;
                    let mut bracket = 0usize;
                    let mut lbrace_opt = None;
                    while p < range_end {
                        match toks[p].kind {
                            TK::LParen => paren += 1,
                            TK::RParen => paren = paren.saturating_sub(1),
                            TK::LBracket => bracket += 1,
                            TK::RBracket => bracket = bracket.saturating_sub(1),
                            TK::Semi if paren == 0 && bracket == 0 => {
                                p += 1;
                                break;
                            }
                            TK::LBrace if paren == 0 && bracket == 0 => {
                                lbrace_opt = Some(p);
                                break;
                            }
                            _ => {}
                        }
                        p += 1;
                    }
                    if let Some(lbrace) = lbrace_opt {
                        let (rbrace, closed) = find_matching_delim_checked(toks, lbrace, range_end);
                        ctx_stack.push(trait_name);
                        walk_items(
                            source,
                            bytes,
                            toks,
                            lbrace + 1,
                            if closed { rbrace } else { range_end },
                            ctx_stack,
                            out,
                        );
                        ctx_stack.pop();
                        i = if closed { (rbrace + 1).min(range_end) } else { range_end };
                    } else {
                        i = p.max(kw_idx + 1);
                    }
                } else {
                    i = kw_idx + 1;
                }
            }
            TK::KwImpl => {
                let hdr_start = kw_idx + 1;
                let mut p = hdr_start;
                let mut paren = 0usize;
                let mut bracket = 0usize;
                let mut lbrace_opt = None;
                while p < range_end {
                    match toks[p].kind {
                        TK::LParen => paren += 1,
                        TK::RParen => paren = paren.saturating_sub(1),
                        TK::LBracket => bracket += 1,
                        TK::RBracket => bracket = bracket.saturating_sub(1),
                        TK::LBrace if paren == 0 && bracket == 0 => {
                            lbrace_opt = Some(p);
                            break;
                        }
                        TK::Semi if paren == 0 && bracket == 0 => {
                            p += 1;
                            break;
                        }
                        _ => {}
                    }
                    p += 1;
                }
                if let Some(lbrace) = lbrace_opt {
                    let ctx_name = find_impl_context(source, toks, hdr_start, lbrace);
                    let (rbrace, closed) = find_matching_delim_checked(toks, lbrace, range_end);
                    ctx_stack.push(ctx_name);
                    walk_items(
                        source,
                        bytes,
                        toks,
                        lbrace + 1,
                        if closed { rbrace } else { range_end },
                        ctx_stack,
                        out,
                    );
                    ctx_stack.pop();
                    i = if closed { (rbrace + 1).min(range_end) } else { range_end };
                } else {
                    i = p.max(kw_idx + 1);
                }
            }
            TK::LBrace => {
                let (rbrace, closed) = find_matching_delim_checked(toks, kw_idx, range_end);
                i = if closed { (rbrace + 1).min(range_end) } else { kw_idx + 1 };
            }
            TK::KwUse | TK::KwType | TK::KwStatic | TK::KwStruct | TK::KwEnum | TK::KwConst => {
                i = skip_to_semi_or_block(toks, kw_idx + 1, range_end).max(kw_idx + 1);
            }
            TK::Ident => {
                if tok_bytes(bytes, toks[kw_idx]) == b"union"
                    && kw_idx + 1 < range_end
                    && toks[kw_idx + 1].kind.is_name_like()
                {
                    i = skip_to_semi_or_block(toks, kw_idx + 2, range_end).max(kw_idx + 1);
                } else if kw_idx + 1 < range_end && toks[kw_idx + 1].kind == TK::Bang {
                    let mut p = kw_idx + 2;
                    if tok_bytes(bytes, toks[kw_idx]) == b"macro_rules"
                        && p < range_end
                        && toks[p].kind.is_name_like()
                    {
                        p += 1;
                    }
                    if p < range_end && matches!(toks[p].kind, TK::LParen | TK::LBracket | TK::LBrace) {
                        let (close, closed) = find_matching_delim_checked(toks, p, range_end);
                        p = if closed { (close + 1).min(range_end) } else { p + 1 };
                        if p < range_end && toks[p].kind == TK::Semi {
                            p += 1;
                        }
                        i = p;
                    } else {
                        i = kw_idx + 2;
                    }
                } else {
                    i = skip_to_semi_or_block(toks, kw_idx + 1, range_end).max(kw_idx + 1);
                }
            }
            _ => {
                if kw_idx > after_attr && toks[kw_idx].kind == TK::LBrace {
                    let (rbrace, closed) = find_matching_delim_checked(toks, kw_idx, range_end);
                    i = if closed { (rbrace + 1).min(range_end) } else { kw_idx + 1 };
                } else {
                    i = (kw_idx + 1).max(i + 1);
                }
            }
        }
    }
}

// ============================================================================
// Pass 1: Variable binding collection
// ============================================================================

#[inline]
fn can_end_expr(k: TK) -> bool {
    matches!(
        k,
        TK::Ident
            | TK::Literal
            | TK::KwSelfLower
            | TK::KwSelfUpper
            | TK::KwCrate
            | TK::KwSuper
            | TK::KwTrue
            | TK::KwFalse
            | TK::RParen
            | TK::RBracket
            | TK::Question
    )
}

fn bind_pattern(bytes: &[u8], toks: &[RsTok], pat_start: usize, pat_end: usize, bound: &mut NameSet) {
    let mut delim_is_struct: Vec<bool> = Vec::with_capacity(8);
    let mut k = pat_start;
    while k < pat_end {
        let t = toks[k];
        match t.kind {
            TK::LBrace => {
                delim_is_struct.push(true);
                k += 1;
                continue;
            }
            TK::LParen | TK::LBracket => {
                delim_is_struct.push(false);
                k += 1;
                continue;
            }
            TK::RBrace | TK::RParen | TK::RBracket => {
                delim_is_struct.pop();
                k += 1;
                continue;
            }
            TK::Lt => {
                k = skip_angles(toks, k, pat_end);
                continue;
            }
            TK::Ident | TK::KwSelfLower => {
                let b = tok_bytes(bytes, t);
                if b == b"_" || ident_is_uppercase(bytes, t.start, t.end) {
                    k += 1;
                    continue;
                }
                let prev_k = if k > pat_start { Some(toks[k - 1].kind) } else { None };
                let next_k = if k + 1 < pat_end { Some(toks[k + 1].kind) } else { None };
                if prev_k == Some(TK::ColonColon)
                    || matches!(next_k, Some(TK::ColonColon | TK::Bang | TK::LParen | TK::LBrace))
                {
                    k += 1;
                    continue;
                }
                if delim_is_struct.last() == Some(&true) && next_k == Some(TK::Colon) {
                    k += 1;
                    continue;
                }
                bound.insert_bytes(bytes, t.start, t.end);
            }
            _ => {}
        }
        k += 1;
    }
}

fn collect_params_bindings(bytes: &[u8], toks: &[RsTok], p_start: usize, p_end: usize, bound: &mut NameSet) {
    let mut i = p_start;
    while i < p_end {
        let mut depth = 0usize;
        let mut angle = 0usize;
        let mut top_colon: Option<usize> = None;
        let chunk_start = i;
        while i < p_end {
            match toks[i].kind {
                TK::LParen | TK::LBracket | TK::LBrace => depth += 1,
                TK::RParen | TK::RBracket | TK::RBrace => depth = depth.saturating_sub(1),
                TK::Lt if depth == 0 => angle += 1,
                TK::Gt if depth == 0 => angle = angle.saturating_sub(1),
                TK::Colon if depth == 0 && angle == 0 && top_colon.is_none() => top_colon = Some(i),
                TK::Comma if depth == 0 && angle == 0 => break,
                _ => {}
            }
            i += 1;
        }
        let pat_end = top_colon.unwrap_or(i);
        bind_pattern(bytes, toks, chunk_start, pat_end, bound);
        if i < p_end && toks[i].kind == TK::Comma {
            i += 1;
        }
    }
}

fn find_closure_close_pipe(toks: &[RsTok], pipe_idx: usize, limit: usize) -> Option<usize> {
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut j = pipe_idx + 1;
    let max_look = (pipe_idx + 128).min(limit);
    while j < max_look {
        match toks[j].kind {
            TK::LParen => paren += 1,
            TK::RParen => {
                if paren == 0 && bracket == 0 && brace == 0 {
                    return None;
                }
                paren -= 1;
            }
            TK::LBracket => bracket += 1,
            TK::RBracket => {
                if paren == 0 && bracket == 0 && brace == 0 {
                    return None;
                }
                bracket -= 1;
            }
            TK::LBrace => {
                if paren == 0 && bracket == 0 && brace == 0 {
                    return None;
                }
                brace += 1;
            }
            TK::RBrace => {
                if brace == 0 {
                    return None;
                }
                brace -= 1;
            }
            TK::Semi | TK::Eq | TK::KwLet | TK::KwReturn | TK::KwFor | TK::KwWhile
                if paren == 0 && bracket == 0 && brace == 0 =>
            {
                return None;
            }
            TK::Lt if paren == 0 && bracket == 0 && brace == 0 => angle += 1,
            TK::Gt if paren == 0 && bracket == 0 && brace == 0 => angle = angle.saturating_sub(1),
            TK::Pipe if paren == 0 && bracket == 0 && brace == 0 && angle == 0 => return Some(j),
            _ => {}
        }
        j += 1;
    }
    None
}

fn collect_range_bindings(
    bytes: &[u8],
    toks: &[RsTok],
    start: usize,
    end: usize,
    bound: &mut NameSet,
    depth: usize,
) {
    if depth >= MAX_DEPTH {
        return;
    }
    let mut i = start;
    while i < end {
        match toks[i].kind {
            TK::KwFn => {
                let mut p = i + 1;
                if p < end && toks[p].kind.is_name_like() {
                    p += 1;
                }
                if p < end && toks[p].kind == TK::Lt {
                    p = skip_angles(toks, p, end);
                }
                if p < end
                    && toks[p].kind == TK::LParen
                    && let Some(rparen) = find_matching_fn_paren(toks, p, end)
                {
                    collect_params_bindings(bytes, toks, p + 1, rparen, bound);
                    i = (rparen + 1).min(end);
                } else {
                    i += 1;
                }
            }
            TK::KwLet => {
                let pat_s = i + 1;
                let mut j = pat_s;
                let (mut p, mut bk, mut br) = (0usize, 0usize, 0usize);
                while j < end {
                    match toks[j].kind {
                        TK::LParen => p += 1,
                        TK::RParen => p = p.saturating_sub(1),
                        TK::LBracket => bk += 1,
                        TK::RBracket => bk = bk.saturating_sub(1),
                        TK::LBrace => br += 1,
                        TK::RBrace => br = br.saturating_sub(1),
                        TK::Colon | TK::Eq | TK::Semi if p == 0 && bk == 0 && br == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                bind_pattern(bytes, toks, pat_s, j, bound);
                i = j.max(i + 1);
            }
            TK::KwFor => {
                let pat_s = i + 1;
                let mut j = pat_s;
                let (mut p, mut bk, mut br) = (0usize, 0usize, 0usize);
                while j < end {
                    match toks[j].kind {
                        TK::LParen => p += 1,
                        TK::RParen => p = p.saturating_sub(1),
                        TK::LBracket => bk += 1,
                        TK::RBracket => bk = bk.saturating_sub(1),
                        TK::LBrace => br += 1,
                        TK::RBrace => {
                            if p == 0 && bk == 0 && br == 0 {
                                break;
                            }
                            br -= 1;
                        }
                        TK::KwIn | TK::Semi if p == 0 && bk == 0 && br == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                bind_pattern(bytes, toks, pat_s, j, bound);
                i = j.max(i + 1);
            }
            TK::KwMatch => {
                let mut j = i + 1;
                let (mut p, mut bk) = (0usize, 0usize);
                while j < end {
                    match toks[j].kind {
                        TK::LParen => p += 1,
                        TK::RParen => p = p.saturating_sub(1),
                        TK::LBracket => bk += 1,
                        TK::RBracket => bk = bk.saturating_sub(1),
                        TK::LBrace | TK::Semi | TK::RBrace if p == 0 && bk == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                if j >= end || toks[j].kind != TK::LBrace {
                    i = (i + 1).min(end);
                    continue;
                }
                let rbrace = find_matching_delim(toks, j, end);
                collect_match_arms_bindings(bytes, toks, j + 1, rbrace, bound, depth + 1);
                i = (rbrace + 1).min(end);
            }
            TK::Pipe => {
                let is_cand = i == start || !can_end_expr(toks[i - 1].kind);
                if is_cand && let Some(close) = find_closure_close_pipe(toks, i, end) {
                    collect_params_bindings(bytes, toks, i + 1, close, bound);
                    i = close + 1;
                    continue;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
}

fn collect_match_arms_bindings(
    bytes: &[u8],
    toks: &[RsTok],
    start: usize,
    end: usize,
    bound: &mut NameSet,
    depth: usize,
) {
    if depth >= MAX_DEPTH {
        return;
    }
    let mut i = start;
    while i < end {
        while i < end && toks[i].kind == TK::Comma {
            i += 1;
        }
        if i >= end {
            break;
        }
        let arm_start = i;
        let (mut p, mut bk, mut br) = (0usize, 0usize, 0usize);
        let mut pat_end = None;
        let mut arrow_idx = None;
        while i < end {
            match toks[i].kind {
                TK::LParen => p += 1,
                TK::RParen => p = p.saturating_sub(1),
                TK::LBracket => bk += 1,
                TK::RBracket => bk = bk.saturating_sub(1),
                TK::LBrace => br += 1,
                TK::RBrace => br = br.saturating_sub(1),
                TK::KwIf if p == 0 && bk == 0 && br == 0 && pat_end.is_none() => pat_end = Some(i),
                TK::FatArrow if p == 0 && bk == 0 && br == 0 => {
                    if pat_end.is_none() {
                        pat_end = Some(i);
                    }
                    arrow_idx = Some(i);
                    break;
                }
                TK::Comma | TK::Semi if p == 0 && bk == 0 && br == 0 => break,
                _ => {}
            }
            i += 1;
        }
        let pe = pat_end.unwrap_or(i);
        bind_pattern(bytes, toks, arm_start, pe, bound);
        let Some(arr) = arrow_idx else {
            i = (i + 1).min(end);
            continue;
        };
        if pe < arr {
            collect_range_bindings(bytes, toks, pe + 1, arr, bound, depth + 1);
        }
        let rhs_start = arr + 1;
        let mut j = rhs_start;
        if j < end && toks[j].kind == TK::LBrace {
            let close = find_matching_delim(toks, j, end);
            collect_range_bindings(bytes, toks, rhs_start, (close + 1).min(end), bound, depth + 1);
            j = (close + 1).min(end);
            if j < end && toks[j].kind == TK::Comma {
                j += 1;
            }
        } else {
            let (mut rp, mut rbk, mut rbr) = (0usize, 0usize, 0usize);
            while j < end {
                match toks[j].kind {
                    TK::LParen => rp += 1,
                    TK::RParen => rp = rp.saturating_sub(1),
                    TK::LBracket => rbk += 1,
                    TK::RBracket => rbk = rbk.saturating_sub(1),
                    TK::LBrace => rbr += 1,
                    TK::RBrace => rbr = rbr.saturating_sub(1),
                    TK::Comma if rp == 0 && rbk == 0 && rbr == 0 => {
                        j += 1;
                        break;
                    }
                    _ => {}
                }
                j += 1;
            }
            collect_range_bindings(bytes, toks, rhs_start, j, bound, depth + 1);
        }
        i = j.max(arr + 1);
    }
}

fn collect_fn_bindings(bytes: &[u8], toks: &[RsTok], item: &RawFnItem, bound: &mut NameSet) {
    bound.clear();
    collect_params_bindings(bytes, toks, item.params_start, item.params_end, bound);
    collect_range_bindings(bytes, toks, item.body_start, item.body_end, bound, 0);
}

// ============================================================================
// Pass 2: Recursive-descent + Pratt parser directly into `Body`
//
// Invariant: `body.scratch` is ONLY pushed inside an active `mark = body.mark()`
// scope; every `parse_* -> Option<u32>` preserves `body.scratch.len()`!
// ============================================================================

struct RsParser<'a> {
    source: &'a str,
    bytes: &'a [u8],
    line_starts: &'a [u32],
    toks: &'a [RsTok],
    bound: &'a NameSet,
    globals: Option<&'a HashSet<String>>,
    snippet_mode: bool,
    pos: usize,
    end: usize,
    line_cur: usize,
    body: &'a mut Body,
}

impl<'a> RsParser<'a> {
    #[inline(always)]
    fn cur_kind(&self) -> Option<TK> {
        if self.pos < self.end { Some(self.toks[self.pos].kind) } else { None }
    }

    #[inline(always)]
    fn peek_kind(&self, offset: usize) -> Option<TK> {
        let idx = self.pos + offset;
        if idx < self.end { Some(self.toks[idx].kind) } else { None }
    }

    #[inline(always)]
    fn emit_tok_range(&mut self, kind: u8, local: bool, start: u32, end: u32) -> u32 {
        let line = line_of_cursor(self.line_starts, start, &mut self.line_cur);
        let end_off = start.max(end.saturating_sub(1));
        let end_line = if end_off == start {
            line
        } else {
            line_of_cursor(self.line_starts, end_off, &mut self.line_cur)
        };
        self.body.push_tok(kind, local, line, end_line, start, end)
    }

    /// Emits current token WITHOUT pushing onto `body.scratch`.
    #[inline(always)]
    fn emit_tok(&mut self, kind: u8, local: bool) -> u32 {
        let t = self.toks[self.pos];
        self.pos += 1;
        self.emit_tok_range(kind, local, t.start, t.end)
    }

    /// Emits current token AND pushes onto the active `body.scratch` scope.
    #[inline(always)]
    fn push_tok(&mut self, kind: u8, local: bool) {
        let id = self.emit_tok(kind, local);
        self.body.push_child(id);
    }

    #[inline(always)]
    fn emit_ident(&mut self, force_non_local: bool, is_callee: bool) -> u32 {
        let t = self.toks[self.pos];
        let local = if force_non_local {
            false
        } else if self.bound.contains_bytes(self.bytes, t.start, t.end) {
            true
        } else if self.snippet_mode {
            !is_callee && !looks_global_rs(self.bytes, self.source, t, self.globals)
        } else {
            false
        };
        self.pos += 1;
        self.emit_tok_range(NAME, local, t.start, t.end)
    }

    #[inline(always)]
    fn push_ident(&mut self, force_non_local: bool, is_callee: bool) {
        let id = self.emit_ident(force_non_local, is_callee);
        self.body.push_child(id);
    }

    #[inline(always)]
    fn push_opt(&mut self, node: Option<u32>) {
        if let Some(id) = node {
            self.body.push_child(id);
        }
    }

    #[inline]
    fn maybe_skip_block_attrs(&mut self) {
        while self.pos < self.end && self.toks[self.pos].kind == TK::Hash {
            let p = self.pos;
            if p + 1 < self.end && self.toks[p + 1].kind == TK::Bang && p + 2 < self.end && self.toks[p + 2].kind == TK::LBracket {
                self.pos = (find_matching_delim(self.toks, p + 2, self.end) + 1).min(self.end);
            } else if p + 1 < self.end && self.toks[p + 1].kind == TK::LBracket {
                self.pos = (find_matching_delim(self.toks, p + 1, self.end) + 1).min(self.end);
            } else {
                break;
            }
        }
    }

    /// Emits `self.pos..limit` tokens directly onto the active `body.scratch` scope.
    fn push_raw_span(&mut self, limit: usize) {
        while self.pos < limit {
            let k = self.toks[self.pos].kind;
            if k.is_name_like() || k == TK::Literal || k == TK::Lifetime {
                self.push_tok(NAME, false);
            } else {
                self.push_tok(KW, false);
            }
        }
    }

    fn parse_generics(&mut self) -> Option<u32> {
        if self.cur_kind() != Some(TK::Lt) {
            return None;
        }
        let end_idx = skip_angles(self.toks, self.pos, self.end);
        let mark = self.body.mark();
        self.push_raw_span(end_idx);
        self.body.close_node(mark)
    }

    fn parse_type(&mut self, stop_at_brace: bool, depth: usize) -> Option<u32> {
        if depth >= MAX_DEPTH || self.pos >= self.end {
            return None;
        }
        let (mut paren, mut bracket, mut angle, mut brace) = (0usize, 0usize, 0usize, 0usize);
        let mark = self.body.mark();
        while self.pos < self.end {
            let t = self.toks[self.pos];
            match t.kind {
                TK::LParen => paren += 1,
                TK::RParen => {
                    if paren == 0 && bracket == 0 && angle == 0 && brace == 0 {
                        break;
                    }
                    paren -= 1;
                }
                TK::LBracket => bracket += 1,
                TK::RBracket => {
                    if paren == 0 && bracket == 0 && angle == 0 && brace == 0 {
                        break;
                    }
                    bracket -= 1;
                }
                TK::Lt => angle += 1,
                TK::Gt => {
                    if angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
                        break;
                    }
                    angle = angle.saturating_sub(1);
                }
                TK::LBrace => {
                    if stop_at_brace && paren == 0 && bracket == 0 && angle == 0 && brace == 0 {
                        break;
                    }
                    brace += 1;
                }
                TK::RBrace => {
                    if paren == 0 && bracket == 0 && angle == 0 && brace == 0 {
                        break;
                    }
                    brace = brace.saturating_sub(1);
                }
                TK::Semi | TK::Eq | TK::Comma | TK::Pipe | TK::KwWhere | TK::FatArrow | TK::KwAs
                    if paren == 0 && bracket == 0 && angle == 0 && brace == 0 =>
                {
                    break;
                }
                _ => {}
            }
            if t.kind.is_name_like() || t.kind == TK::Literal || t.kind == TK::Lifetime {
                self.push_tok(NAME, false);
            } else {
                self.push_tok(KW, false);
            }
        }
        self.body.close_node(mark)
    }

    fn parse_pattern(&mut self, depth: usize) -> Option<u32> {
        if depth >= MAX_DEPTH || self.pos >= self.end {
            return None;
        }
        let mark = self.body.mark();
        let (mut paren, mut bracket) = (0usize, 0usize);
        let mut brace_stack: Vec<bool> = Vec::new();

        while self.pos < self.end {
            let t = self.toks[self.pos];
            match t.kind {
                TK::Colon | TK::Eq | TK::Semi | TK::KwIn | TK::KwIf | TK::FatArrow | TK::Pipe
                    if paren == 0 && bracket == 0 && brace_stack.is_empty() =>
                {
                    if t.kind == TK::Pipe {
                        self.push_tok(KW, false);
                        continue;
                    }
                    break;
                }
                TK::Comma | TK::RParen | TK::RBracket | TK::RBrace
                    if paren == 0 && bracket == 0 && brace_stack.is_empty() =>
                {
                    break;
                }
                TK::LBrace => {
                    brace_stack.push(true);
                    self.push_tok(KW, false);
                }
                TK::RBrace => {
                    brace_stack.pop();
                    self.push_tok(KW, false);
                }
                TK::LParen => {
                    paren += 1;
                    brace_stack.push(false);
                    self.push_tok(KW, false);
                }
                TK::RParen => {
                    paren = paren.saturating_sub(1);
                    brace_stack.pop();
                    self.push_tok(KW, false);
                }
                TK::LBracket => {
                    bracket += 1;
                    brace_stack.push(false);
                    self.push_tok(KW, false);
                }
                TK::RBracket => {
                    bracket = bracket.saturating_sub(1);
                    brace_stack.pop();
                    self.push_tok(KW, false);
                }
                TK::Lt => {
                    let end_idx = skip_angles(self.toks, self.pos, self.end);
                    self.push_raw_span(end_idx);
                }
                TK::Ident | TK::KwSelfLower => {
                    let prev_k = if self.pos > 0 { Some(self.toks[self.pos - 1].kind) } else { None };
                    let next_k = self.peek_kind(1);
                    let cur_struct = brace_stack.last() == Some(&true);
                    let disqualify = prev_k == Some(TK::ColonColon)
                        || matches!(next_k, Some(TK::ColonColon | TK::Bang | TK::LParen | TK::LBrace))
                        || (cur_struct && next_k == Some(TK::Colon))
                        || tok_bytes(self.bytes, t) == b"_"
                        || ident_is_uppercase(self.bytes, t.start, t.end);
                    let local = !disqualify && self.bound.contains_bytes(self.bytes, t.start, t.end);
                    self.push_tok(NAME, local);
                }
                TK::Literal
                | TK::Lifetime
                | TK::KwSelfUpper
                | TK::KwCrate
                | TK::KwSuper
                | TK::KwTrue
                | TK::KwFalse => self.push_tok(NAME, false),
                _ => self.push_tok(KW, false),
            }
        }
        self.body.close_node(mark)
    }

    fn parse_closure(&mut self, start_pipe: TK, is_move: bool, allow_struct: bool, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        if is_move {
            self.push_tok(KW, false);
        }
        if start_pipe == TK::OrOr {
            self.push_tok(KW, false);
        } else {
            self.push_tok(KW, false); // opening `|`
            let close_opt = find_closure_close_pipe(self.toks, self.pos - 1, self.end);
            let limit = close_opt.unwrap_or(self.pos);
            while self.pos < limit {
                if self.cur_kind() == Some(TK::Comma) {
                    self.push_tok(KW, false);
                    continue;
                }
                let mut d = 0usize;
                let mut colon_pos = None;
                let mut comma_pos = limit;
                let mut scan = self.pos;
                while scan < limit {
                    match self.toks[scan].kind {
                        TK::LParen | TK::LBracket | TK::LBrace => d += 1,
                        TK::RParen | TK::RBracket | TK::RBrace => d = d.saturating_sub(1),
                        TK::Colon if d == 0 && colon_pos.is_none() => colon_pos = Some(scan),
                        TK::Comma if d == 0 => {
                            comma_pos = scan;
                            break;
                        }
                        _ => {}
                    }
                    scan += 1;
                }
                let pat_stop = colon_pos.unwrap_or(comma_pos);
                let saved_end = self.end;
                self.end = pat_stop;
                let pnode = self.parse_pattern(depth + 1);
                self.push_opt(pnode);
                if self.pos < pat_stop {
                    self.push_raw_span(pat_stop);
                }
                self.end = saved_end;

                if self.pos < comma_pos && self.cur_kind() == Some(TK::Colon) {
                    self.push_tok(KW, false);
                    let saved = self.end;
                    self.end = comma_pos;
                    let ty = self.parse_type(false, depth + 1);
                    self.push_opt(ty);
                    self.end = saved;
                }
                if self.pos == comma_pos && self.cur_kind() == Some(TK::Comma) {
                    self.push_tok(KW, false);
                } else if self.pos < comma_pos {
                    self.push_raw_span(comma_pos);
                }
            }
            if close_opt.is_some() && self.cur_kind() == Some(TK::Pipe) {
                self.push_tok(KW, false);
            }
        }
        if self.cur_kind() == Some(TK::Arrow) {
            self.push_tok(KW, false);
            let ty = self.parse_type(true, depth + 1);
            self.push_opt(ty);
        }
        let body_e = self.parse_expr_bp(0, allow_struct, depth + 1);
        self.push_opt(body_e);
        self.body.close_node(mark)
    }

    fn parse_block(&mut self, depth: usize) -> Option<u32> {
        if self.cur_kind() != Some(TK::LBrace) || depth >= MAX_DEPTH {
            return None;
        }
        let mark = self.body.mark();
        self.push_tok(KW, false); // `{`
        self.parse_stmt_sequence(depth + 1, true);
        if self.cur_kind() == Some(TK::RBrace) {
            self.push_tok(KW, false); // `}`
        }
        self.body.close_node(mark)
    }

    fn parse_stmt_sequence(&mut self, depth: usize, stop_at_rbrace: bool) {
        if depth >= MAX_DEPTH {
            while self.pos < self.end {
                if stop_at_rbrace && self.cur_kind() == Some(TK::RBrace) {
                    break;
                }
                self.pos += 1;
            }
            return;
        }
        while self.pos < self.end {
            self.maybe_skip_block_attrs();
            if self.pos >= self.end {
                break;
            }
            let k = self.toks[self.pos].kind;
            if stop_at_rbrace && k == TK::RBrace {
                break;
            }
            if k == TK::Semi {
                self.push_tok(KW, false);
                continue;
            }
            let before = self.pos;
            let stmt = self.parse_stmt(depth);
            self.push_opt(stmt);
            if self.pos == before {
                self.push_tok(KW, false);
            }
        }
    }

    fn parse_stmt(&mut self, depth: usize) -> Option<u32> {
        let k = self.cur_kind()?;
        if k == TK::KwLet {
            let mark = self.body.mark();
            self.push_tok(KW, false); // `let`
            let pnode = self.parse_pattern(depth + 1);
            self.push_opt(pnode);
            if self.cur_kind() == Some(TK::Colon) {
                self.push_tok(KW, false);
                let ty = self.parse_type(false, depth + 1);
                self.push_opt(ty);
            }
            if self.cur_kind() == Some(TK::Eq) {
                self.push_tok(KW, false);
                let rhs = self.parse_expr_bp(0, true, depth + 1);
                self.push_opt(rhs);
                if self.cur_kind() == Some(TK::KwElse) {
                    self.push_tok(KW, false);
                    let blk = self.parse_block(depth + 1);
                    self.push_opt(blk);
                }
            }
            if self.cur_kind() == Some(TK::Semi) {
                self.push_tok(KW, false);
            }
            return self.body.close_node(mark);
        }

        if self.looks_like_local_item() {
            return self.parse_local_item(depth);
        }

        let mark = self.body.mark();
        let e = self.parse_expr_bp(0, true, depth + 1);
        self.push_opt(e);
        if self.cur_kind() == Some(TK::Semi) {
            self.push_tok(KW, false);
        }
        self.body.close_node(mark)
    }

    fn looks_like_local_item(&self) -> bool {
        let Some(k) = self.cur_kind() else {
            return false;
        };
        if matches!(
            k,
            TK::KwUse | TK::KwType | TK::KwStatic | TK::KwFn | TK::KwStruct | TK::KwEnum | TK::KwImpl | TK::KwTrait | TK::KwMod | TK::KwPub
        ) {
            return true;
        }
        k == TK::KwConst && self.peek_kind(1) != Some(TK::LBrace)
    }

    fn parse_local_item(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        let kw_idx = skip_visibility_and_modifiers(self.bytes, self.toks, self.pos, self.end);
        if kw_idx > self.pos {
            self.push_raw_span(kw_idx);
        }
        if self.pos >= self.end {
            return self.body.close_node(mark);
        }
        if self.cur_kind() == Some(TK::KwFn) {
            self.push_tok(KW, false); // `fn`
            if self.cur_kind().is_some_and(TK::is_name_like) {
                self.push_tok(NAME, false);
            }
            let gens = self.parse_generics();
            self.push_opt(gens);
            if self.cur_kind() == Some(TK::LParen) {
                let rparen = find_matching_delim(self.toks, self.pos, self.end);
                self.push_raw_span((rparen + 1).min(self.end));
            }
            if self.cur_kind() == Some(TK::Arrow) {
                self.push_tok(KW, false);
                let ty = self.parse_type(true, depth + 1);
                self.push_opt(ty);
            }
            if self.cur_kind() == Some(TK::KwWhere) {
                while self.pos < self.end && !matches!(self.toks[self.pos].kind, TK::LBrace | TK::Semi) {
                    self.push_tok(KW, false);
                }
            }
            if self.cur_kind() == Some(TK::LBrace) {
                let blk = self.parse_block(depth + 1);
                self.push_opt(blk);
            } else if self.cur_kind() == Some(TK::Semi) {
                self.push_tok(KW, false);
            }
            return self.body.close_node(mark);
        }

        let item_end = skip_to_semi_or_block(self.toks, self.pos, self.end);
        self.push_raw_span(item_end);
        self.body.close_node(mark)
    }

    fn parse_cond(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        if self.cur_kind() == Some(TK::KwLet) {
            self.push_tok(KW, false);
            let p = self.parse_pattern(depth + 1);
            self.push_opt(p);
            if self.cur_kind() == Some(TK::Eq) {
                self.push_tok(KW, false);
            }
            let e = self.parse_expr_bp(8, false, depth + 1);
            self.push_opt(e);
            while self.cur_kind() == Some(TK::AndAnd) {
                self.push_tok(KW, false);
                if self.cur_kind() == Some(TK::KwLet) {
                    self.push_tok(KW, false);
                    let cp = self.parse_pattern(depth + 1);
                    self.push_opt(cp);
                    if self.cur_kind() == Some(TK::Eq) {
                        self.push_tok(KW, false);
                    }
                    let ce = self.parse_expr_bp(8, false, depth + 1);
                    self.push_opt(ce);
                } else {
                    let ce = self.parse_expr_bp(8, false, depth + 1);
                    self.push_opt(ce);
                }
            }
            return self.body.close_node(mark);
        }
        let e = self.parse_expr_bp(0, false, depth + 1);
        self.push_opt(e);
        self.body.close_node(mark)
    }

    fn parse_if(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        self.push_tok(KW, false); // `if`
        let cond = self.parse_cond(depth + 1);
        self.push_opt(cond);
        let then_blk = self.parse_block(depth + 1);
        self.push_opt(then_blk);
        if self.cur_kind() == Some(TK::KwElse) {
            self.push_tok(KW, false);
            if self.cur_kind() == Some(TK::KwIf) {
                let nested = self.parse_if(depth + 1);
                self.push_opt(nested);
            } else {
                let else_blk = self.parse_block(depth + 1);
                self.push_opt(else_blk);
            }
        }
        self.body.close_node(mark)
    }

    fn parse_match(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        self.push_tok(KW, false); // `match`
        let scrut = self.parse_expr_bp(0, false, depth + 1);
        self.push_opt(scrut);
        if self.cur_kind() == Some(TK::LBrace) {
            let bmark = self.body.mark();
            let rbrace = find_matching_delim(self.toks, self.pos, self.end);
            self.push_tok(KW, false); // `{`
            let saved_end = self.end;
            self.end = rbrace;
            while self.pos < self.end {
                self.maybe_skip_block_attrs();
                if self.pos >= self.end {
                    break;
                }
                if self.cur_kind() == Some(TK::Comma) {
                    self.push_tok(KW, false);
                    continue;
                }
                let amark = self.body.mark();
                let p = self.parse_pattern(depth + 1);
                self.push_opt(p);
                if self.cur_kind() == Some(TK::KwIf) {
                    self.push_tok(KW, false);
                    let g = self.parse_expr_bp(0, false, depth + 1);
                    self.push_opt(g);
                }
                if self.cur_kind() == Some(TK::FatArrow) {
                    self.push_tok(KW, false);
                    let rhs = if self.cur_kind() == Some(TK::LBrace) {
                        self.parse_block(depth + 1)
                    } else {
                        self.parse_expr_bp(0, true, depth + 1)
                    };
                    self.push_opt(rhs);
                } else {
                    self.push_tok(KW, false);
                }
                if self.cur_kind() == Some(TK::Comma) {
                    self.push_tok(KW, false);
                }
                let arm = self.body.close_node(amark);
                self.push_opt(arm);
            }
            self.end = saved_end;
            if self.cur_kind() == Some(TK::RBrace) {
                self.push_tok(KW, false);
            }
            let arms_blk = self.body.close_node(bmark);
            self.push_opt(arms_blk);
        }
        self.body.close_node(mark)
    }

    fn parse_macro_args(&mut self, close_idx: usize, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        let saved_end = self.end;
        self.push_tok(KW, false); // opening delim
        self.end = close_idx;
        if depth < MAX_DEPTH {
            while self.pos < self.end {
                let k = self.toks[self.pos].kind;
                if matches!(k, TK::Comma | TK::Semi | TK::Colon | TK::FatArrow | TK::Eq) {
                    self.push_tok(KW, false);
                    continue;
                }
                let before = self.pos;
                let e = self.parse_expr_bp(0, true, depth + 1);
                self.push_opt(e);
                if self.pos == before {
                    let t = self.toks[self.pos];
                    if t.kind.is_ident_like() {
                        self.push_ident(false, false);
                    } else if t.kind.is_name_like() || t.kind == TK::Literal || t.kind == TK::Lifetime {
                        self.push_tok(NAME, false);
                    } else {
                        self.push_tok(KW, false);
                    }
                }
            }
        } else {
            self.push_raw_span(close_idx);
        }
        self.end = saved_end;
        if self.pos == close_idx && self.pos < self.end {
            self.push_tok(KW, false); // closing delim
        }
        self.body.close_node(mark)
    }

    fn parse_arg_list(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        let rparen = find_matching_delim(self.toks, self.pos, self.end);
        self.push_tok(KW, false); // `(`
        let saved_end = self.end;
        self.end = rparen;
        while self.pos < self.end {
            if self.cur_kind() == Some(TK::Comma) {
                self.push_tok(KW, false);
                continue;
            }
            let before = self.pos;
            let arg = self.parse_expr_bp(0, true, depth + 1);
            self.push_opt(arg);
            if self.cur_kind() == Some(TK::Comma) || self.pos == before {
                self.push_tok(KW, false);
            }
        }
        self.end = saved_end;
        if self.cur_kind() == Some(TK::RParen) {
            self.push_tok(KW, false);
        }
        self.body.close_node(mark)
    }

    fn parse_struct_literal(&mut self, depth: usize) -> Option<u32> {
        let mark = self.body.mark();
        let rbrace = find_matching_delim(self.toks, self.pos, self.end);
        self.push_tok(KW, false); // `{`
        let saved_end = self.end;
        self.end = rbrace;
        while self.pos < self.end {
            let k = self.toks[self.pos].kind;
            if k == TK::Comma {
                self.push_tok(KW, false);
                continue;
            }
            if k == TK::DotDot {
                self.push_tok(KW, false);
                let base = self.parse_expr_bp(0, true, depth + 1);
                self.push_opt(base);
                continue;
            }
            if k.is_ident_like() && self.peek_kind(1) == Some(TK::Colon) {
                let fmark = self.body.mark();
                self.push_ident(true, false);
                self.push_tok(KW, false); // `:`
                let val = self.parse_expr_bp(0, true, depth + 1);
                self.push_opt(val);
                let f = self.body.close_node(fmark);
                self.push_opt(f);
                if self.cur_kind() == Some(TK::Comma) {
                    self.push_tok(KW, false);
                }
                continue;
            }
            let before = self.pos;
            let val = self.parse_expr_bp(0, true, depth + 1);
            self.push_opt(val);
            if self.cur_kind() == Some(TK::Comma) || self.pos == before {
                self.push_tok(KW, false);
            }
        }
        self.end = saved_end;
        if self.cur_kind() == Some(TK::RBrace) {
            self.push_tok(KW, false);
        }
        self.body.close_node(mark)
    }

    fn looks_like_struct_lit(&self) -> bool {
        if self.cur_kind() != Some(TK::LBrace) {
            return false;
        }
        match (self.peek_kind(1), self.peek_kind(2)) {
            (Some(TK::RBrace), _) | (Some(TK::DotDot), _) => true,
            (Some(k1), Some(TK::Colon | TK::Comma | TK::RBrace)) if k1.is_ident_like() => true,
            _ => false,
        }
    }

    fn parse_prefix(&mut self, allow_struct: bool, depth: usize) -> Option<(u32, bool)> {
        if depth >= MAX_DEPTH || self.pos >= self.end {
            return None;
        }
        let t = self.toks[self.pos];
        match t.kind {
            TK::KwReturn => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                if !matches!(
                    self.cur_kind(),
                    None | Some(TK::Semi | TK::RBrace | TK::Comma | TK::RParen | TK::RBracket)
                ) {
                    let e = self.parse_expr_bp(0, allow_struct, depth + 1);
                    self.push_opt(e);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwBreak => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                if self.cur_kind() == Some(TK::Lifetime) {
                    self.push_tok(NAME, false);
                }
                if !matches!(
                    self.cur_kind(),
                    None | Some(TK::Semi | TK::RBrace | TK::Comma | TK::RParen | TK::RBracket)
                ) {
                    let e = self.parse_expr_bp(0, allow_struct, depth + 1);
                    self.push_opt(e);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwContinue => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                if self.cur_kind() == Some(TK::Lifetime) {
                    self.push_tok(NAME, false);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwIf => self.parse_if(depth + 1).map(|n| (n, false)),
            TK::KwMatch => self.parse_match(depth + 1).map(|n| (n, false)),
            TK::KwFor => {
                let mark = self.body.mark();
                self.push_tok(KW, false); // `for`
                let p = self.parse_pattern(depth + 1);
                self.push_opt(p);
                if self.cur_kind() == Some(TK::KwIn) {
                    self.push_tok(KW, false);
                }
                let iter = self.parse_expr_bp(0, false, depth + 1);
                self.push_opt(iter);
                let blk = self.parse_block(depth + 1);
                self.push_opt(blk);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwWhile => {
                let mark = self.body.mark();
                self.push_tok(KW, false); // `while`
                let cond = self.parse_cond(depth + 1);
                self.push_opt(cond);
                let blk = self.parse_block(depth + 1);
                self.push_opt(blk);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwLoop => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                let blk = self.parse_block(depth + 1);
                self.push_opt(blk);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::Lifetime if self.peek_kind(1) == Some(TK::Colon) => {
                let mark = self.body.mark();
                self.push_tok(NAME, false);
                self.push_tok(KW, false);
                let e = self.parse_expr_bp(0, allow_struct, depth + 1);
                self.push_opt(e);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwUnsafe | TK::KwConst if self.peek_kind(1) == Some(TK::LBrace) => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                let blk = self.parse_block(depth + 1);
                self.push_opt(blk);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::KwAsync => {
                if self.peek_kind(1) == Some(TK::LBrace)
                    || (self.peek_kind(1) == Some(TK::KwMove) && self.peek_kind(2) == Some(TK::LBrace))
                {
                    let mark = self.body.mark();
                    self.push_tok(KW, false);
                    if self.cur_kind() == Some(TK::KwMove) {
                        self.push_tok(KW, false);
                    }
                    let blk = self.parse_block(depth + 1);
                    self.push_opt(blk);
                    self.body.close_node(mark).map(|n| (n, false))
                } else if self.peek_kind(1) == Some(TK::KwMove)
                    && matches!(self.peek_kind(2), Some(TK::Pipe | TK::OrOr))
                {
                    let mark = self.body.mark();
                    self.push_tok(KW, false); // `async`
                    if let Some(pipe_k) = self.peek_kind(1) {
                        let cl = self.parse_closure(pipe_k, true, allow_struct, depth + 1);
                        self.push_opt(cl);
                    }
                    self.body.close_node(mark).map(|n| (n, false))
                } else {
                    Some((self.emit_tok(KW, false), false))
                }
            }
            TK::KwMove => match self.peek_kind(1) {
                Some(pipe_k @ (TK::Pipe | TK::OrOr)) => {
                    self.parse_closure(pipe_k, true, allow_struct, depth + 1).map(|n| (n, false))
                }
                _ => Some((self.emit_tok(KW, false), false)),
            },
            TK::Pipe | TK::OrOr => {
                self.parse_closure(t.kind, false, allow_struct, depth + 1).map(|n| (n, false))
            }
            TK::LBrace => self.parse_block(depth + 1).map(|n| (n, false)),
            TK::LParen => {
                let mark = self.body.mark();
                let rparen = find_matching_delim(self.toks, self.pos, self.end);
                self.push_tok(KW, false); // `(`
                let saved_end = self.end;
                self.end = rparen;
                while self.pos < self.end {
                    if self.cur_kind() == Some(TK::Comma) {
                        self.push_tok(KW, false);
                        continue;
                    }
                    let before = self.pos;
                    let e = self.parse_expr_bp(0, true, depth + 1);
                    self.push_opt(e);
                    if self.cur_kind() == Some(TK::Comma) || self.pos == before {
                        self.push_tok(KW, false);
                    }
                }
                self.end = saved_end;
                if self.cur_kind() == Some(TK::RParen) {
                    self.push_tok(KW, false);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::LBracket => {
                let mark = self.body.mark();
                let rbracket = find_matching_delim(self.toks, self.pos, self.end);
                self.push_tok(KW, false); // `[`
                let saved_end = self.end;
                self.end = rbracket;
                while self.pos < self.end {
                    if matches!(self.cur_kind(), Some(TK::Comma | TK::Semi)) {
                        self.push_tok(KW, false);
                        continue;
                    }
                    let before = self.pos;
                    let e = self.parse_expr_bp(0, true, depth + 1);
                    self.push_opt(e);
                    if matches!(self.cur_kind(), Some(TK::Comma | TK::Semi)) || self.pos == before {
                        self.push_tok(KW, false);
                    }
                }
                self.end = saved_end;
                if self.cur_kind() == Some(TK::RBracket) {
                    self.push_tok(KW, false);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::DotDot | TK::DotDotEq => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                if !matches!(
                    self.cur_kind(),
                    None | Some(TK::Semi | TK::RBrace | TK::RParen | TK::RBracket | TK::Comma)
                ) {
                    let e = self.parse_expr_bp(4, allow_struct, depth + 1);
                    self.push_opt(e);
                }
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::Minus | TK::Bang | TK::Star | TK::Amp => {
                let mark = self.body.mark();
                self.push_tok(KW, false);
                if t.kind == TK::Amp && self.cur_kind() == Some(TK::KwMut) {
                    self.push_tok(KW, false);
                }
                let rhs = self.parse_expr_bp(25, allow_struct, depth + 1);
                self.push_opt(rhs);
                self.body.close_node(mark).map(|n| (n, false))
            }
            TK::Literal | TK::KwTrue | TK::KwFalse => Some((self.emit_tok(NAME, false), false)),
            TK::Lt => {
                let mark = self.body.mark();
                let end_idx = skip_angles(self.toks, self.pos, self.end);
                self.push_raw_span(end_idx);
                while self.cur_kind() == Some(TK::ColonColon) {
                    self.push_tok(KW, false);
                    if self.cur_kind().is_some_and(TK::is_name_like) {
                        self.push_ident(true, false);
                    } else if self.cur_kind() == Some(TK::Lt) {
                        let gens = self.parse_generics();
                        self.push_opt(gens);
                    } else {
                        break;
                    }
                }
                self.body.close_node(mark).map(|n| (n, true))
            }
            TK::ColonColon => {
                let mark = self.body.mark();
                while self.cur_kind() == Some(TK::ColonColon) {
                    self.push_tok(KW, false);
                    if self.cur_kind().is_some_and(TK::is_name_like) {
                        self.push_ident(true, false);
                    } else if self.cur_kind() == Some(TK::Lt) {
                        let gens = self.parse_generics();
                        self.push_opt(gens);
                    } else {
                        break;
                    }
                }
                self.body.close_node(mark).map(|n| (n, true))
            }
            k if k.is_name_like() => {
                let has_path = self.peek_kind(1) == Some(TK::ColonColon);
                let is_macro = self.peek_kind(1) == Some(TK::Bang)
                    && matches!(self.peek_kind(2), Some(TK::LParen | TK::LBracket | TK::LBrace));
                let is_call = self.peek_kind(1) == Some(TK::LParen);
                let is_struct = allow_struct && self.peek_kind(1) == Some(TK::LBrace);
                if has_path {
                    let mark = self.body.mark();
                    self.push_ident(true, false);
                    while self.cur_kind() == Some(TK::ColonColon) {
                        self.push_tok(KW, false);
                        if self.cur_kind().is_some_and(TK::is_name_like) {
                            self.push_ident(true, false);
                        } else if self.cur_kind() == Some(TK::Lt) {
                            let gens = self.parse_generics();
                            self.push_opt(gens);
                        } else {
                            break;
                        }
                    }
                    self.body.close_node(mark).map(|n| (n, true))
                } else {
                    Some((self.emit_ident(is_macro || is_struct, is_call), true))
                }
            }
            _ => None,
        }
    }

    fn parse_expr_bp(&mut self, min_bp: u8, allow_struct: bool, depth: usize) -> Option<u32> {
        if depth >= MAX_DEPTH {
            return None;
        }
        let (mut lhs, mut is_path) = self.parse_prefix(allow_struct, depth + 1)?;

        loop {
            if self.pos >= self.end {
                break;
            }
            let k = self.toks[self.pos].kind;

            match k {
                TK::Bang if is_path && matches!(self.peek_kind(1), Some(TK::LParen | TK::LBracket | TK::LBrace)) => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    self.push_tok(KW, false); // `!`
                    let close_idx = find_matching_delim(self.toks, self.pos, self.end);
                    let args = self.parse_macro_args(close_idx, depth + 1);
                    self.push_opt(args);
                    if let Some(n) = self.body.close_node(mark) {
                        lhs = n;
                    }
                    is_path = false;
                    continue;
                }
                TK::LBrace if allow_struct && is_path && self.looks_like_struct_lit() => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    let sl = self.parse_struct_literal(depth + 1);
                    self.push_opt(sl);
                    if let Some(n) = self.body.close_node(mark) {
                        lhs = n;
                    }
                    is_path = false;
                    continue;
                }
                TK::LParen => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    let args = self.parse_arg_list(depth + 1);
                    self.push_opt(args);
                    if let Some(n) = self.body.close_node(mark) {
                        lhs = n;
                    }
                    is_path = false;
                    continue;
                }
                TK::LBracket => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    let rbracket = find_matching_delim(self.toks, self.pos, self.end);
                    self.push_tok(KW, false); // `[`
                    let saved_end = self.end;
                    self.end = rbracket;
                    let idx_e = self.parse_expr_bp(0, true, depth + 1);
                    self.push_opt(idx_e);
                    self.end = saved_end;
                    if self.cur_kind() == Some(TK::RBracket) {
                        self.push_tok(KW, false);
                    }
                    if let Some(n) = self.body.close_node(mark) {
                        lhs = n;
                    }
                    is_path = false;
                    continue;
                }
                TK::Dot => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    self.push_tok(KW, false); // `.`
                    let mut got_method = false;
                    match self.cur_kind() {
                        Some(TK::KwAwait) => self.push_tok(KW, false),
                        Some(nk) if nk.is_name_like() || nk == TK::Literal => {
                            self.push_tok(NAME, false);
                            got_method = true;
                        }
                        _ => {}
                    }
                    let member_node = self.body.close_node(mark).unwrap_or(lhs);
                    if got_method
                        && (self.cur_kind() == Some(TK::LParen)
                            || (self.cur_kind() == Some(TK::ColonColon) && self.peek_kind(1) == Some(TK::Lt)))
                    {
                        let call_mark = self.body.mark();
                        self.body.push_child(member_node);
                        if self.cur_kind() == Some(TK::ColonColon) && self.peek_kind(1) == Some(TK::Lt) {
                            self.push_tok(KW, false); // `::`
                            let gens = self.parse_generics();
                            self.push_opt(gens);
                        }
                        if self.cur_kind() == Some(TK::LParen) {
                            let args = self.parse_arg_list(depth + 1);
                            self.push_opt(args);
                        }
                        lhs = self.body.close_node(call_mark).unwrap_or(member_node);
                    } else {
                        lhs = member_node;
                    }
                    is_path = false;
                    continue;
                }
                TK::Question => {
                    let mark = self.body.mark();
                    self.body.push_child(lhs);
                    self.push_tok(KW, false);
                    if let Some(n) = self.body.close_node(mark) {
                        lhs = n;
                    }
                    is_path = false;
                    continue;
                }
                _ => {}
            }

            let is_shr_eq = is_contiguous_shr_eq(self.toks, self.pos, self.end);
            let is_shr = !is_shr_eq && is_contiguous_shr(self.toks, self.pos, self.end);
            let (l_bp, r_bp): (u8, u8) = if is_shr_eq {
                (2, 1)
            } else if is_shr {
                (17, 18)
            } else {
                match k {
                    TK::Eq | TK::AssignOp => (2, 1),
                    TK::DotDot | TK::DotDotEq => (3, 4),
                    TK::OrOr => (5, 6),
                    TK::AndAnd => (7, 8),
                    TK::EqEq | TK::Ne | TK::Lt | TK::Le | TK::Gt | TK::Ge => (9, 10),
                    TK::Pipe => (11, 12),
                    TK::Caret => (13, 14),
                    TK::Amp => (15, 16),
                    TK::Shl => (17, 18),
                    TK::Plus | TK::Minus => (19, 20),
                    TK::Star | TK::Slash | TK::Percent => (21, 22),
                    TK::KwAs => (23, 24),
                    _ => break,
                }
            };

            if l_bp < min_bp {
                break;
            }

            if k == TK::KwAs {
                let mark = self.body.mark();
                self.body.push_child(lhs);
                self.push_tok(KW, false); // `as`
                let ty = self.parse_type(true, depth + 1);
                self.push_opt(ty);
                if let Some(n) = self.body.close_node(mark) {
                    lhs = n;
                }
                is_path = false;
                continue;
            }

            let mark = self.body.mark();
            self.body.push_child(lhs);
            if is_shr || is_shr_eq {
                let t0 = self.toks[self.pos];
                let t1 = self.toks[self.pos + 1];
                self.pos += 2;
                let op_id = self.emit_tok_range(KW, false, t0.start, t1.end);
                self.body.push_child(op_id);
            } else {
                self.push_tok(KW, false);
            }

            if matches!(k, TK::DotDot | TK::DotDotEq)
                && matches!(
                    self.cur_kind(),
                    None | Some(TK::Semi | TK::RBrace | TK::RParen | TK::RBracket | TK::Comma | TK::LBrace)
                )
            {
                if let Some(n) = self.body.close_node(mark) {
                    lhs = n;
                }
                is_path = false;
                continue;
            }

            if k == TK::AndAnd && !allow_struct && self.cur_kind() == Some(TK::KwLet) {
                self.push_tok(KW, false); // `let`
                let p = self.parse_pattern(depth + 1);
                self.push_opt(p);
                if self.cur_kind() == Some(TK::Eq) {
                    self.push_tok(KW, false);
                }
                let rhs = self.parse_expr_bp(8, false, depth + 1);
                self.push_opt(rhs);
                if let Some(n) = self.body.close_node(mark) {
                    lhs = n;
                }
                is_path = false;
                continue;
            }

            let rhs = self.parse_expr_bp(r_bp, allow_struct, depth + 1);
            self.push_opt(rhs);
            if let Some(n) = self.body.close_node(mark) {
                lhs = n;
            }
            is_path = false;
        }

        Some(lhs)
    }
}

// ============================================================================
// Front-end Entrypoints & `Language` impl
// ============================================================================

pub(crate) fn parse_methods_into(
    source: &str,
    line_range: Option<(u32, u32)>,
    body: &mut Body,
    out: &mut Vec<FastMethod>,
) {
    TL_RS.with(|cell| {
        let mut sc = cell.borrow_mut();
        let RsScratch { line_starts, toks, bound } = &mut *sc;
        let bytes = source.as_bytes();
        lex_rust(bytes, line_starts, toks);

        let mut items: Vec<RawFnItem> = Vec::new();
        let mut ctx_stack: Vec<String> = Vec::new();
        walk_items(source, bytes, toks, 0, toks.len(), &mut ctx_stack, &mut items);

        for item in items {
            let sl = line_of(line_starts, toks[item.start_tok].start);
            let el = line_of(line_starts, toks[item.end_tok].end.saturating_sub(1));
            if let Some((min_ln, max_ln)) = line_range
                && (el < min_ln || sl > max_ln)
            {
                continue;
            }
            collect_fn_bindings(bytes, toks, &item, bound);
            let scratch_base = body.mark();
            let mut parser = RsParser {
                source,
                bytes,
                line_starts,
                toks,
                bound,
                globals: None,
                snippet_mode: false,
                pos: item.body_start,
                end: item.body_end,
                line_cur: 0,
                body,
            };
            if let Some(root) = parser.parse_block(0) {
                debug_assert_eq!(parser.body.mark(), scratch_base);
                out.push(FastMethod {
                    name: item.name,
                    start_line: sl,
                    end_line: el,
                    body_root: root,
                });
            }
        }
    });
}

pub(crate) fn parse_snippet_into(
    source: &str,
    options: &SnippetOptions,
    body: &mut Body,
) -> Option<u32> {
    TL_RS.with(|cell| {
        let mut sc = cell.borrow_mut();
        let RsScratch { line_starts, toks, bound } = &mut *sc;
        let bytes = source.as_bytes();
        lex_rust(bytes, line_starts, toks);
        if toks.is_empty() {
            return None;
        }
        bound.clear();

        let mut items: Vec<RawFnItem> = Vec::new();
        let mut ctx_stack: Vec<String> = Vec::new();
        walk_items(source, bytes, toks, 0, toks.len(), &mut ctx_stack, &mut items);
        for item in &items {
            collect_params_bindings(bytes, toks, item.params_start, item.params_end, bound);
        }
        collect_range_bindings(bytes, toks, 0, toks.len(), bound, 0);

        let mark = body.mark();
        let mut parser = RsParser {
            source,
            bytes,
            line_starts,
            toks,
            bound,
            globals: options.globals.as_ref(),
            snippet_mode: true,
            pos: 0,
            end: toks.len(),
            line_cur: 0,
            body,
        };
        parser.parse_stmt_sequence(0, false);
        let root = parser.body.close_node(mark);
        debug_assert_eq!(parser.body.mark(), mark);
        root
    })
}

pub struct RustLanguage;

impl Language for RustLanguage {
    fn id(&self) -> &str {
        "rust"
    }

    fn extensions(&self) -> &[&str] {
        &[".rs"]
    }

    fn member_access_labels(&self) -> &[&str] {
        &["#.#"]
    }

    fn parse_methods(&self, source: &str, _path: &str) -> Vec<ParsedMethod> {
        let mut body = Body::default();
        let mut raw = Vec::new();
        parse_methods_into(source, None, &mut body, &mut raw);
        let base = source.as_bytes();
        raw.into_iter()
            .map(|m| ParsedMethod {
                name: m.name,
                start_line: m.start_line,
                end_line: m.end_line,
                body: body.to_node(base, m.body_root),
            })
            .collect()
    }

    fn parse_snippet(&self, source: &str, options: &SnippetOptions) -> Option<Node> {
        let mut body = Body::default();
        let root = parse_snippet_into(source, options, &mut body)?;
        Some(body.to_node(source.as_bytes(), root))
    }

    fn builtin_kind(&self) -> u8 {
        3
    }
}
