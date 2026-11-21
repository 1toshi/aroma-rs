//! Source text in the shape the reference sees it: a JavaScript string of
//! UTF-16 code units. ASCII sources (the vast majority) are used as bytes
//! directly; others carry their UTF-16 units and a unit -> byte offset map.

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

pub(crate) const SIDE: u32 = 1 << 31;

pub(crate) struct Src<'a> {
    pub text: &'a str,
    units: Option<Vec<u16>>,
    byte_of: Vec<u32>,
    line_starts: Vec<u32>,
}

impl<'a> Src<'a> {
    pub fn new(text: &'a str) -> Self {
        let mut line_starts = vec![0u32];
        if text.is_ascii() {
            for pos in memchr::memchr_iter(b'\n', text.as_bytes()) {
                line_starts.push(pos as u32 + 1);
            }
            Src { text, units: None, byte_of: Vec::new(), line_starts }
        } else {
            let mut units = Vec::with_capacity(text.len());
            let mut byte_of = Vec::with_capacity(text.len() + 1);
            for (bi, c) in text.char_indices() {
                let mut buf = [0u16; 2];
                let enc = c.encode_utf16(&mut buf);
                byte_of.push(bi as u32);
                if enc.len() == 2 {
                    byte_of.push(u32::MAX);
                }
                for &u in enc.iter() {
                    if u == b'\n' as u16 {
                        line_starts.push(units.len() as u32 + 1);
                    }
                    units.push(u);
                }
            }
            byte_of.push(text.len() as u32);
            Src { text, units: Some(units), byte_of, line_starts }
        }
    }

    #[inline(always)]
    pub fn unit(&self, i: u32) -> u32 {
        match &self.units {
            None => self.text.as_bytes()[i as usize] as u32,
            Some(u) => u[i as usize] as u32,
        }
    }

    #[inline(always)]
    pub fn is_ascii(&self) -> bool {
        self.units.is_none()
    }

    pub fn parse_into(&self, g: &'static Grammar, tree: &mut SyntaxTree) {
        match &self.units {
            None => lezer::parse_into(g, lezer::AsciiByte::slice_from_ascii(self.text.as_bytes()), tree),
            Some(u) => lezer::parse_into(g, u, tree),
        }
    }

    /// 1-based line of a unit offset.
    #[inline]
    pub fn line_of(&self, offset: u32) -> u32 {
        let ls = &self.line_starts;
        ls.partition_point(|&s| s <= offset) as u32
    }

    /// Monotonic O(1) cursor-assisted 1-based line lookup (for DFS token emission).
    #[inline(always)]
    pub fn line_of_cursor(&self, offset: u32, cur: &mut usize) -> u32 {
        let ls = &self.line_starts;
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
            // Linear peek up to 4 lines before falling back to binary search
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

    /// The source text between two unit offsets, if both fall on character boundaries.
    #[inline]
    pub fn str_range(&self, a: u32, b: u32) -> Option<&'a str> {
        if a >= b {
            return Some("");
        }
        match &self.units {
            None => Some(&self.text[a as usize..b as usize]),
            Some(_) => {
                let (x, y) = (self.byte_of[a as usize], self.byte_of[b as usize]);
                if x == u32::MAX || y == u32::MAX { None } else { Some(&self.text[x as usize..y as usize]) }
            }
        }
    }

    /// A text reference for the units `a..b`: a byte range of the source, or,
    /// when the range splits a surrogate pair, a range of `side`.
    #[inline]
    pub fn text_ref(&self, a: u32, b: u32, side: &mut Vec<u8>) -> (u32, u32) {
        if a >= b {
            return (0, 0);
        }
        match &self.units {
            None => (a, b),
            Some(u) => {
                let (x, y) = (self.byte_of[a as usize], self.byte_of[b as usize]);
                if x != u32::MAX && y != u32::MAX {
                    (x, y)
                } else {
                    let s = String::from_utf16_lossy(&u[a as usize..b as usize]);
                    let start = side.len() as u32;
                    side.extend_from_slice(s.as_bytes());
                    (SIDE | start, SIDE | side.len() as u32)
                }
            }
        }
    }

    /// `source.slice(a, b)` as a String (lossy for split surrogate pairs).
    pub fn string(&self, a: u32, b: u32) -> String {
        match self.str_range(a, b) {
            Some(s) => s.to_string(),
            None => String::from_utf16_lossy(&self.units.as_ref().unwrap()[a as usize..b as usize]),
        }
    }

    /// The range `a..b` without leading and trailing JavaScript whitespace.
    #[inline]
    pub fn trim(&self, mut a: u32, mut b: u32) -> (u32, u32) {
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
