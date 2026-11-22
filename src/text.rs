//! Source text in the shape the reference sees it: a JavaScript string of
//! UTF-16 code units. ASCII sources (the vast majority) are used as bytes
//! directly; others carry their UTF-16 units and a unit -> byte offset map.
//! Buffers are reusable via `SrcBuf` to eliminate per-file allocations.

use std::collections::HashSet;

use crate::lezer::{self, Grammar, SyntaxTree};

/// JavaScript's `\s` (WhiteSpace and LineTerminator), on a UTF-16 code unit.
#[inline(always)]
pub(crate) fn js_ws(c: u32) -> bool {
    if c < 128 {
        matches!(c, 9..=13 | 32)
    } else {
        matches!(c, 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000 | 0xFEFF)
    }
}

#[inline(always)]
pub(crate) fn is_ascii_ws(b: u8) -> bool {
    b <= 32 && (b == 32 || (9..=13).contains(&b))
}

#[inline]
pub(crate) fn js_ws_char(c: char) -> bool {
    (c as u32) < 0x10000 && js_ws(c as u32)
}

/// `String.prototype.trimEnd`.
pub(crate) fn js_trim_end(s: &str) -> &str {
    s.trim_end_matches(js_ws_char)
}

/// `String.prototype.trim`.
pub(crate) fn js_trim(s: &str) -> &str {
    s.trim_matches(js_ws_char)
}

/// Number of leading whitespace characters (all such characters are one UTF-16 unit).
pub(crate) fn js_indent(s: &str) -> usize {
    s.chars().take_while(|&c| js_ws_char(c)).count()
}

/// Drops the first `n` characters of a line known to start with at least that much whitespace,
/// or all of a shorter line.
pub(crate) fn drop_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

/// UTF-16 length of a string.
#[inline]
pub(crate) fn utf16_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().map(char::len_utf16).sum() }
}

/// Fast check whether UTF-16 length exceeds `max`. Since UTF-16 code units are
/// always `<=` UTF-8 byte length, any string with `s.len() <= max` immediately
/// returns `false` without scanning a single byte.
#[inline(always)]
pub(crate) fn utf16_len_exceeds(s: &str, max: usize) -> bool {
    s.len() > max && utf16_len(s) > max
}

pub(crate) const SIDE: u32 = 1 << 31;

/// Reusable scratch backing `Src<'a>` (stored in thread-local front-end scratch).
#[derive(Default)]
pub(crate) struct SrcBuf {
    pub units: Vec<u16>,
    pub byte_of: Vec<u32>,
    pub line_starts: Vec<u32>,
}

impl SrcBuf {
    pub fn release(&mut self) {
        self.units.clear();
        self.units.shrink_to_fit();
        self.byte_of.clear();
        self.byte_of.shrink_to_fit();
        self.line_starts.clear();
        self.line_starts.shrink_to_fit();
    }
}

pub(crate) struct Src<'a> {
    pub text: &'a str,
    is_ascii: bool,
    units: &'a [u16],
    byte_of: &'a [u32],
    line_starts: &'a [u32],
}

/// Monotonic O(1) cursor-assisted 1-based line lookup over byte/unit `line_starts`.
#[inline(always)]
pub(crate) fn line_of_cursor(ls: &[u32], offset: u32, cur: &mut usize) -> u32 {
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

impl<'a> Src<'a> {
    pub fn build(text: &'a str, buf: &'a mut SrcBuf) -> Self {
        buf.line_starts.clear();
        buf.line_starts.push(0);
        if text.is_ascii() {
            for pos in memchr::memchr_iter(b'\n', text.as_bytes()) {
                buf.line_starts.push(pos as u32 + 1);
            }
            Src {
                text,
                is_ascii: true,
                units: &[],
                byte_of: &[],
                line_starts: &buf.line_starts,
            }
        } else {
            buf.units.clear();
            buf.byte_of.clear();
            buf.units.reserve(text.len());
            buf.byte_of.reserve(text.len() + 1);
            for (bi, c) in text.char_indices() {
                let mut enc_buf = [0u16; 2];
                let enc = c.encode_utf16(&mut enc_buf);
                buf.byte_of.push(bi as u32);
                if enc.len() == 2 {
                    buf.byte_of.push(u32::MAX);
                }
                for &u in enc.iter() {
                    if u == b'\n' as u16 {
                        buf.line_starts.push(buf.units.len() as u32 + 1);
                    }
                    buf.units.push(u);
                }
            }
            buf.byte_of.push(text.len() as u32);
            Src {
                text,
                is_ascii: false,
                units: &buf.units,
                byte_of: &buf.byte_of,
                line_starts: &buf.line_starts,
            }
        }
    }

    #[inline(always)]
    pub fn unit(&self, i: u32) -> u32 {
        if self.is_ascii {
            unsafe { *self.text.as_bytes().get_unchecked(i as usize) as u32 }
        } else {
            unsafe { *self.units.get_unchecked(i as usize) as u32 }
        }
    }

    #[inline(always)]
    pub fn is_ascii(&self) -> bool {
        self.is_ascii
    }

    #[inline(always)]
    pub fn units_slice(&self) -> &'a [u16] {
        self.units
    }

    pub fn parse_into(&self, g: &'static Grammar, tree: &mut SyntaxTree) {
        if self.is_ascii {
            lezer::parse_into(g, lezer::AsciiByte::slice_from_ascii(self.text.as_bytes()), tree);
        } else {
            lezer::parse_into(g, self.units, tree);
        }
    }

    /// 1-based line of a unit offset.
    #[inline]
    pub fn line_of(&self, offset: u32) -> u32 {
        self.line_starts.partition_point(|&s| s <= offset) as u32
    }

    /// Monotonic O(1) cursor-assisted 1-based line lookup (for DFS token emission).
    #[inline(always)]
    pub fn line_of_cursor(&self, offset: u32, cur: &mut usize) -> u32 {
        line_of_cursor(self.line_starts, offset, cur)
    }

    /// 1-based `(start_line, end_line)` span trimming trailing JavaScript whitespace.
    #[inline]
    pub fn line_span(&self, from: u32, to: u32) -> (u32, u32) {
        let mut end = to;
        while end > from + 1 && js_ws(self.unit(end - 1)) {
            end -= 1;
        }
        (self.line_of(from), self.line_of(end.saturating_sub(1)))
    }

    /// True if identifier range `from..to` looks like an external/global symbol.
    #[inline]
    pub fn looks_global(&self, from: u32, to: u32, globals: Option<&HashSet<String>>) -> bool {
        if from < to {
            let first = self.unit(from);
            if (b'A' as u32..=b'Z' as u32).contains(&first) {
                return true;
            }
        }
        if let Some(g) = globals {
            match self.str_range(from, to) {
                Some(s) => g.contains(s),
                None => g.contains(&self.string(from, to)),
            }
        } else {
            false
        }
    }

    /// Appends source text for unit span `(from, to)` onto `out`.
    #[inline]
    pub fn push_span(&self, span: (u32, u32), out: &mut String) {
        match self.str_range(span.0, span.1) {
            Some(s) => out.push_str(s),
            None => out.push_str(&self.string(span.0, span.1)),
        }
    }

    /// The source text between two unit offsets, if both fall on character boundaries.
    #[inline]
    pub fn str_range(&self, a: u32, b: u32) -> Option<&'a str> {
        if a >= b {
            return Some("");
        }
        if self.is_ascii {
            Some(&self.text[a as usize..b as usize])
        } else {
            let (x, y) = (self.byte_of[a as usize], self.byte_of[b as usize]);
            if x == u32::MAX || y == u32::MAX { None } else { Some(&self.text[x as usize..y as usize]) }
        }
    }

    /// A text reference for the units `a..b`: a byte range of the source, or,
    /// when the range splits a surrogate pair, a range of `side`.
    #[inline(always)]
    pub fn text_ref(&self, a: u32, b: u32, side: &mut Vec<u8>) -> (u32, u32) {
        if a >= b {
            return (0, 0);
        }
        if self.is_ascii {
            (a, b)
        } else {
            let (x, y) = (self.byte_of[a as usize], self.byte_of[b as usize]);
            if x != u32::MAX && y != u32::MAX {
                (x, y)
            } else {
                let s = String::from_utf16_lossy(&self.units[a as usize..b as usize]);
                let start = side.len() as u32;
                side.extend_from_slice(s.as_bytes());
                (SIDE | start, SIDE | side.len() as u32)
            }
        }
    }

    /// `source.slice(a, b)` as a String (lossy for split surrogate pairs).
    pub fn string(&self, a: u32, b: u32) -> String {
        match self.str_range(a, b) {
            Some(s) => s.to_string(),
            None => String::from_utf16_lossy(&self.units[a as usize..b as usize]),
        }
    }

    /// The range `a..b` without leading and trailing JavaScript whitespace.
    #[inline(always)]
    pub fn trim(&self, mut a: u32, mut b: u32) -> (u32, u32) {
        if a >= b {
            return (a, b);
        }
        if self.is_ascii {
            let bytes = self.text.as_bytes();
            let first = unsafe { *bytes.get_unchecked(a as usize) };
            let last = unsafe { *bytes.get_unchecked(b as usize - 1) };
            if first > 32 && last > 32 {
                return (a, b);
            }
            while a < b && is_ascii_ws(unsafe { *bytes.get_unchecked(a as usize) }) {
                a += 1;
            }
            while b > a && is_ascii_ws(unsafe { *bytes.get_unchecked(b as usize - 1) }) {
                b -= 1;
            }
            return (a, b);
        }
        while a < b && js_ws(self.unit(a)) {
            a += 1;
        }
        while b > a && js_ws(self.unit(b - 1)) {
            b -= 1;
        }
        (a, b)
    }
}

/// `dedent` of the Python front end: lines without their common indentation.
pub(crate) fn dedent_source(source: &str) -> String {
    let lines: Vec<&str> = source.split('\n').collect();
    let common = lines.iter().filter(|l| !js_trim(l).is_empty()).map(|l| js_indent(l)).min().unwrap_or(0);
    let parts: Vec<&str> = lines.iter().map(|l| drop_chars(l, common)).collect();
    parts.join("\n")
}

// ============================================================================
// Zero-allocation open-addressing NameSet over `(start, len)` spans in `Src`
// ============================================================================

#[inline(always)]
fn hash_name_u8(b: &[u8]) -> u32 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let len = b.len();
    let mut h: u64 = (len as u64).wrapping_mul(K);
    if len <= 8 {
        let mut buf = [0u8; 8];
        if len > 0 {
            buf[..len].copy_from_slice(b);
        }
        h = (h.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(K);
        return (h ^ (h >> 29)) as u32;
    }
    let mut i = 0usize;
    while i + 8 <= len {
        let w = u64::from_le_bytes(unsafe { *(b.as_ptr().add(i) as *const [u8; 8]) });
        h = (h.rotate_left(5) ^ w).wrapping_mul(K);
        i += 8;
    }
    if i < len {
        let mut buf = [0u8; 8];
        buf[..len - i].copy_from_slice(&b[i..]);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(K);
    }
    (h ^ (h >> 29)) as u32
}

#[inline(always)]
fn hash_name_u16(u: &[u16]) -> u32 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut h: u64 = (u.len() as u64).wrapping_mul(K);
    for &c in u {
        h = (h.rotate_left(5) ^ (c as u64)).wrapping_mul(K);
    }
    (h ^ (h >> 29)) as u32
}

#[derive(Clone, Copy)]
struct NameSlot {
    hash: u32,
    start: u32,
    len: u32,
}

const EMPTY_SLOT: u32 = u32::MAX;

#[inline(always)]
fn probe_start(h32: u32, mask: usize) -> usize {
    ((h32 ^ (h32 >> 15)) as usize) & mask
}

pub(crate) struct NameSet {
    slots: Vec<NameSlot>,
    used: Vec<u32>,
    mask: usize,
}

impl Default for NameSet {
    fn default() -> Self {
        Self::new()
    }
}

trait NameBuf: Copy {
    fn hash_slice(self, from: u32, to: u32) -> u32;
    fn eq_slot(self, s: NameSlot, from: u32, to: u32) -> bool;
}

impl NameBuf for &[u8] {
    #[inline(always)]
    fn hash_slice(self, from: u32, to: u32) -> u32 {
        hash_name_u8(unsafe { self.get_unchecked(from as usize..to as usize) })
    }

    #[inline(always)]
    fn eq_slot(self, s: NameSlot, from: u32, to: u32) -> bool {
        unsafe {
            self.get_unchecked(s.start as usize..(s.start + s.len) as usize)
                == self.get_unchecked(from as usize..to as usize)
        }
    }
}

impl NameBuf for &[u16] {
    #[inline(always)]
    fn hash_slice(self, from: u32, to: u32) -> u32 {
        hash_name_u16(&self[from as usize..to as usize])
    }

    #[inline(always)]
    fn eq_slot(self, s: NameSlot, from: u32, to: u32) -> bool {
        self[s.start as usize..(s.start + s.len) as usize] == self[from as usize..to as usize]
    }
}

impl NameSet {
    pub fn new() -> Self {
        Self {
            slots: vec![NameSlot { hash: 0, start: EMPTY_SLOT, len: 0 }; 64],
            used: Vec::with_capacity(32),
            mask: 63,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        for &i in &self.used {
            unsafe { self.slots.get_unchecked_mut(i as usize).start = EMPTY_SLOT };
        }
        self.used.clear();
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.used.is_empty()
    }

    pub fn release(&mut self) {
        self.clear();
        if self.slots.len() > 64 {
            self.slots.truncate(64);
            self.slots.shrink_to_fit();
            self.mask = 63;
        }
        self.used.shrink_to_fit();
    }

    #[inline(always)]
    fn contains_in<B: NameBuf>(&self, buf: B, from: u32, to: u32) -> bool {
        let len = to - from;
        let h32 = buf.hash_slice(from, to);
        let mut i = probe_start(h32, self.mask);
        loop {
            let s = unsafe { *self.slots.get_unchecked(i) };
            if s.start == EMPTY_SLOT {
                return false;
            }
            if s.hash == h32 && s.len == len && buf.eq_slot(s, from, to) {
                return true;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert_in<B: NameBuf>(&mut self, buf: B, from: u32, to: u32) {
        let len = to - from;
        let h32 = buf.hash_slice(from, to);
        let mut i = probe_start(h32, self.mask);
        loop {
            let s = unsafe { *self.slots.get_unchecked(i) };
            if s.start == EMPTY_SLOT {
                unsafe {
                    *self.slots.get_unchecked_mut(i) = NameSlot { hash: h32, start: from, len };
                }
                self.used.push(i as u32);
                if self.used.len() * 2 > self.slots.len() {
                    self.grow();
                }
                return;
            }
            if s.hash == h32 && s.len == len && buf.eq_slot(s, from, to) {
                return;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn contains_bytes(&self, bytes: &[u8], from: u32, to: u32) -> bool {
        !self.used.is_empty() && self.contains_in(bytes, from, to)
    }

    #[inline(always)]
    pub fn insert_bytes(&mut self, bytes: &[u8], from: u32, to: u32) {
        self.insert_in(bytes, from, to);
    }

    #[inline(always)]
    pub fn contains(&self, src: &Src<'_>, from: u32, to: u32) -> bool {
        if self.used.is_empty() {
            return false;
        }
        if src.is_ascii() {
            self.contains_in(src.text.as_bytes(), from, to)
        } else {
            self.contains_in(src.units_slice(), from, to)
        }
    }

    #[inline]
    pub fn insert(&mut self, src: &Src<'_>, from: u32, to: u32) {
        if src.is_ascii() {
            self.insert_in(src.text.as_bytes(), from, to);
        } else {
            self.insert_in(src.units_slice(), from, to);
        }
    }

    #[cold]
    fn grow(&mut self) {
        let newcap = self.slots.len() * 2;
        let old: Vec<NameSlot> = self.used.iter().map(|&i| self.slots[i as usize]).collect();
        self.slots.clear();
        self.slots.resize(newcap, NameSlot { hash: 0, start: EMPTY_SLOT, len: 0 });
        self.mask = newcap - 1;
        self.used.clear();
        for s in old {
            let mut i = probe_start(s.hash, self.mask);
            while self.slots[i].start != EMPTY_SLOT {
                i = (i + 1) & self.mask;
            }
            self.slots[i] = s;
            self.used.push(i as u32);
        }
    }
}
