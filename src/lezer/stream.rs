//! The input stream tokenizers read from (`InputStream` in @lezer/lr) and the
//! precomputed L1-resident `u8`-indexed ASCII DFA tokenizer (`readToken`) with
//! per-dialect `(group, row)` O(1) accept + sink-state bit, state-0 peeling,
//! O(1) precedence ranking, and tight self-loop unrolling.

use super::{Grammar, Tok};

/// Transparent wrapper around an ASCII byte (`< 128`) so `read_token_group`
/// compiles away all non-ASCII branches and binary-search fallbacks.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AsciiByte(pub u8);

impl AsciiByte {
    #[inline(always)]
    pub fn slice_from_ascii(bytes: &[u8]) -> &[Self] {
        debug_assert!(bytes.is_ascii());
        // SAFETY: `AsciiByte` is `#[repr(transparent)]` over `u8`.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const AsciiByte, bytes.len()) }
    }
}

pub trait Ch: Copy + Send + Sync + 'static {
    const ALWAYS_ASCII: bool = false;
    fn code(self) -> i32;
    fn as_u8_slice(slice: &[Self]) -> Option<&[u8]>;
}
impl Ch for AsciiByte {
    const ALWAYS_ASCII: bool = true;
    #[inline(always)]
    fn code(self) -> i32 {
        self.0 as i32
    }
    #[inline(always)]
    fn as_u8_slice(slice: &[Self]) -> Option<&[u8]> {
        // SAFETY: `AsciiByte` is `#[repr(transparent)]` over `u8`.
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) })
    }
}
impl Ch for u8 {
    #[inline(always)]
    fn code(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn as_u8_slice(slice: &[Self]) -> Option<&[u8]> {
        Some(slice)
    }
}
impl Ch for u16 {
    #[inline(always)]
    fn code(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn as_u8_slice(_slice: &[Self]) -> Option<&[u8]> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CachedToken {
    pub start: i32,
    pub value: i32,
    pub end: i32,
    pub extended: i32,
    pub look_ahead: i32,
    pub mask: u32,
    pub context: i64,
}

impl CachedToken {
    pub const fn new() -> Self {
        CachedToken {
            start: -1,
            value: -1,
            end: -1,
            extended: -1,
            look_ahead: 0,
            mask: 0,
            context: 0,
        }
    }
}

pub(crate) struct Stream<'a, C: Ch> {
    pub input: &'a [C],
    pub end: i32,
    pub pos: i32,
    pub next: i32,
    pub tok: CachedToken,
    pub tok_active: bool,
}

impl<'a, C: Ch> Stream<'a, C> {
    pub fn new(input: &'a [C]) -> Self {
        let mut s = Stream {
            input,
            end: input.len() as i32,
            pos: 0,
            next: -1,
            tok: CachedToken::new(),
            tok_active: false,
        };
        s.next = s.char_at(0);
        s
    }
    #[inline(always)]
    pub fn char_at(&self, p: i32) -> i32 {
        if p >= 0 && p < self.end {
            unsafe { self.input.get_unchecked(p as usize).code() }
        } else {
            -1
        }
    }
    #[inline(always)]
    pub fn peek(&mut self, offset: i32) -> i32 {
        let p = self.pos + offset;
        if p < 0 || p >= self.end {
            return -1;
        }
        if self.tok_active && p >= self.tok.look_ahead {
            self.tok.look_ahead = p + 1;
        }
        unsafe { self.input.get_unchecked(p as usize).code() }
    }
    #[inline(always)]
    pub fn advance(&mut self, n: i32) -> i32 {
        if self.pos + n >= self.end {
            self.pos = self.end;
            self.next = -1;
            return -1;
        }
        self.pos += n;
        if self.tok_active && self.pos >= self.tok.look_ahead {
            self.tok.look_ahead = self.pos + 1;
        }
        self.next = unsafe { self.input.get_unchecked(self.pos as usize).code() };
        self.next
    }
    #[inline(always)]
    pub fn accept_token(&mut self, token: i32, end_offset: i32) {
        let end = if end_offset != 0 {
            let p = self.pos + end_offset;
            if p < 0 || p > self.end { self.pos } else { p }
        } else {
            self.pos
        };
        self.tok.value = token;
        self.tok.end = end;
    }
    #[inline(always)]
    pub fn reset(&mut self, pos: i32, token: Option<CachedToken>) {
        if let Some(mut t) = token {
            t.start = pos;
            t.look_ahead = pos + 1;
            t.value = -1;
            t.extended = -1;
            self.tok = t;
            self.tok_active = true;
        } else {
            self.tok_active = false;
        }
        if self.pos != pos {
            self.pos = pos;
            self.next = self.char_at(pos);
        }
    }
    #[inline(always)]
    pub fn clip_pos(&self, pos: i32) -> i32 {
        if pos >= self.end { self.end } else { pos.max(0) }
    }
    #[inline(always)]
    pub fn read(&self, from: i32, to: i32) -> &'a [C] {
        &self.input[from as usize..to as usize]
    }
}

// ============================================================================
// Precomputed L1-resident `u8`-indexed ASCII DFA + per-dialect accept/sink table
// ============================================================================

const NO_EDGE: u8 = u8::MAX;
const STOP_ACC: u16 = 0xFFFF;
const NO_ACC: u16 = 0xFFFE;
const MULTI_ACC: u16 = 0xFFFD;
/// Set on single-accept entries in `acc_by_group` when the DFA state has zero
/// outgoing edges (no ASCII edges, no hi_edges, no EOF edge), allowing immediate
/// loop exit without reading `input[pos]` or indexing `ascii[row * 128 + ch]`.
const SINK_BIT: u16 = 0x4000;
const TERM_MASK: u16 = 0x3FFF;

#[derive(Clone, Copy)]
pub(crate) struct DfaStateMeta {
    pub acc_start: u16,
    pub acc_end: u16,
    pub eof_row: u8, // NO_EDGE if no EOF transition
    pub edge_start: u16,
    pub edge_count: u16,
}

pub(crate) struct DfaAccel {
    pub n_rows: usize,
    pub meta: Vec<DfaStateMeta>,
    /// Flat `[row * 128 + ch]` -> next `row: u8` (or `NO_EDGE` if no edge).
    /// Python = 11.5 KB, JS/TS = 25 KB — fits completely in L1d cache!
    pub ascii: Vec<u8>,
    /// Precomputed per-dialect `[group * n_rows + row]`:
    /// - `STOP_ACC`: `(group_mask & state.mask) == 0` (stop scanning!)
    /// - `NO_ACC`: in group, no allowed accepting token at this row
    /// - `MULTI_ACC`: in group, > 1 allowed accepting tokens (fallback to scan)
    /// - `term | (SINK_BIT if no outgoing edges)`: unique allowed accepting token!
    pub acc_by_group: Vec<u16>,
    /// Non-ASCII edges (`>= 128`): `[from, to_u16_0_is_65536, target_row]`
    pub hi_edges: Vec<[u16; 3]>,
    /// O(1) precedence rank: `0` = not in `prec_table`, else `offset + 1`.
    pub prec_rank: Vec<u16>,
}

impl DfaAccel {
    pub fn build(g: &Grammar) -> Self {
        let data = g.token_data;
        let prec_table = g.data;
        let prec_offset = g.token_prec;
        let num_terms = g.max_term as usize + 1;

        let mut off_to_row = vec![NO_EDGE; data.len()];
        let mut offsets: Vec<usize> = Vec::new();
        let mut i = 0usize;
        while i < data.len() {
            let row = offsets.len();
            assert!(row < NO_EDGE as usize, "DFA state count exceeds u8");
            off_to_row[i] = row as u8;
            offsets.push(i);
            let acc_end = data[i + 1] as usize;
            let num_edges = data[i + 2] as usize;
            i = acc_end + num_edges * 3;
        }

        let n = offsets.len();
        let mut meta = Vec::with_capacity(n);
        let mut ascii = vec![NO_EDGE; n * 128];
        let mut hi_edges: Vec<[u16; 3]> = Vec::new();
        let mut masks = Vec::with_capacity(n);
        let mut is_sink = vec![false; n];

        for (row, &off) in offsets.iter().enumerate() {
            let mask = data[off];
            masks.push(mask);
            let acc_end = data[off + 1] as usize;
            let num_edges = data[off + 2] as usize;
            let mut eof_row = NO_EDGE;
            let edge_start = hi_edges.len() as u16;
            is_sink[row] = num_edges == 0;

            for e in 0..num_edges {
                let idx = acc_end + e * 3;
                let from = data[idx];
                let to = data[idx + 1];
                let target_off = data[idx + 2] as usize;
                let target_row = off_to_row[target_off];

                if from == 65535 && e == num_edges - 1 {
                    eof_row = target_row;
                    continue;
                }
                let to_u32: u32 = if to == 0 { 0x10000 } else { to as u32 };
                if (from as u32) < 128 {
                    let a_end = to_u32.min(128) as usize;
                    let base = row * 128;
                    for ch in (from as usize)..a_end {
                        ascii[base + ch] = target_row;
                    }
                }
                if to_u32 > 128 {
                    let hi_from = from.max(128);
                    hi_edges.push([hi_from, to, target_row as u16]);
                }
            }
            let edge_count = hi_edges.len() as u16 - edge_start;
            meta.push(DfaStateMeta {
                acc_start: (off + 3) as u16,
                acc_end: acc_end as u16,
                eof_row,
                edge_start,
                edge_count,
            });
        }

        let num_groups = g
            .tokenizers
            .iter()
            .filter_map(|t| match *t {
                Tok::Group(id) => Some(id as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(1);

        let mut acc_by_group = vec![STOP_ACC; num_groups * n];
        for grp in 0..num_groups {
            let group_mask: u16 = 1 << grp;
            let base = grp * n;
            for row in 0..n {
                if (group_mask & masks[row]) == 0 {
                    continue;
                }
                let m = meta[row];
                let mut found: u16 = NO_ACC;
                let mut j = m.acc_start as usize;
                let ae = m.acc_end as usize;
                while j < ae {
                    if (data[j + 1] & group_mask) != 0 {
                        let term = data[j];
                        if g.allows(term) {
                            if found == NO_ACC {
                                found = term;
                            } else {
                                found = MULTI_ACC;
                                break;
                            }
                        }
                    }
                    j += 2;
                }
                if found < MULTI_ACC && is_sink[row] {
                    found |= SINK_BIT;
                }
                acc_by_group[base + row] = found;
            }
        }

        let mut prec_rank = vec![0u16; num_terms.max(512)];
        let mut p = prec_offset;
        let mut r = 1u16;
        while prec_table[p] != 65535 {
            let term = prec_table[p] as usize;
            if term >= prec_rank.len() {
                prec_rank.resize(term + 1, 0);
            }
            if prec_rank[term] == 0 {
                prec_rank[term] = r;
            }
            r += 1;
            p += 1;
        }

        Self { n_rows: n, meta, ascii, acc_by_group, hi_edges, prec_rank }
    }
}

/// Register-resident, L1 `u8`-DFA accelerated `readToken` returning
/// `(value, end, look_ahead)` directly in registers without mutating `Stream`.
/// Peels state 0 on ASCII input and exits immediately on sink states (`SINK_BIT`).
#[inline]
pub(crate) fn read_token_group<C: Ch>(
    g: &Grammar,
    dfa: &DfaAccel,
    input: &[C],
    start: usize,
    group: u16,
) -> (i32, i32, i32) {
    let prec_rank = &dfa.prec_rank;
    let ascii = &dfa.ascii;
    let acc_base = group as usize * dfa.n_rows;
    let acc_col = unsafe { dfa.acc_by_group.get_unchecked(acc_base..acc_base + dfa.n_rows) };

    let end = input.len();
    let mut pos = start;
    let mut row = 0usize;
    let mut cur = -1i32;
    let mut cur_end = -1i32;

    // Peel state 0 when it is non-accepting (the universal case in Lezer DFAs)
    if pos < end && unsafe { *acc_col.get_unchecked(0) } == NO_ACC {
        let ch0 = unsafe { input.get_unchecked(pos).code() };
        if C::ALWAYS_ASCII || ch0 < 128 {
            let r1 = unsafe { *ascii.get_unchecked(ch0 as usize) };
            if r1 == NO_EDGE {
                return (-1, -1, start as i32 + 1);
            }
            row = r1 as usize;
            pos += 1;
        }
    }

    loop {
        let acc = unsafe { *acc_col.get_unchecked(row) };
        if acc == STOP_ACC {
            break;
        }
        let mut accepted_here = false;
        if acc < MULTI_ACC {
            let term = (acc & TERM_MASK) as i32;
            if cur == term {
                cur_end = pos as i32;
                accepted_here = true;
            } else if cur == -1 {
                cur = term;
                cur_end = pos as i32;
                accepted_here = true;
            } else {
                let rp = unsafe { *prec_rank.get_unchecked(cur as usize) };
                if rp == 0 || unsafe { *prec_rank.get_unchecked(term as usize) } < rp {
                    cur = term;
                    cur_end = pos as i32;
                    accepted_here = true;
                }
            }
            if (acc & SINK_BIT) != 0 {
                break;
            }
        } else if acc == MULTI_ACC {
            let m = unsafe { *dfa.meta.get_unchecked(row) };
            let data = g.token_data;
            let group_mask: u16 = 1 << group;
            let mut i = m.acc_start as usize;
            let ae = m.acc_end as usize;
            while i < ae {
                if (unsafe { *data.get_unchecked(i + 1) } & group_mask) != 0 {
                    let term = unsafe { *data.get_unchecked(i) };
                    if cur == term as i32 {
                        cur_end = pos as i32;
                        accepted_here = true;
                        break;
                    }
                    if g.allows(term) {
                        let ok = if cur == -1 {
                            true
                        } else {
                            let rp = unsafe { *prec_rank.get_unchecked(cur as usize) };
                            rp == 0 || unsafe { *prec_rank.get_unchecked(term as usize) } < rp
                        };
                        if ok {
                            cur = term as i32;
                            cur_end = pos as i32;
                            accepted_here = true;
                            break;
                        }
                    }
                }
                i += 2;
            }
        }

        if pos < end {
            let ch = unsafe { input.get_unchecked(pos).code() };
            if C::ALWAYS_ASCII || ch < 128 {
                let next_row = unsafe { *ascii.get_unchecked((row << 7) | (ch as usize)) };
                if next_row == NO_EDGE {
                    break;
                }
                let nr = next_row as usize;
                pos += 1;
                if nr == row && acc != MULTI_ACC {
                    let row_base = row << 7;
                    let row_u8 = row as u8;
                    let mut exit_row = NO_EDGE;
                    let mut hit_non_ascii = false;
                    while pos < end {
                        let c2 = unsafe { input.get_unchecked(pos).code() };
                        if !C::ALWAYS_ASCII && c2 >= 128 {
                            hit_non_ascii = true;
                            break;
                        }
                        let r2 = unsafe { *ascii.get_unchecked(row_base | (c2 as usize)) };
                        if r2 != row_u8 {
                            exit_row = r2;
                            break;
                        }
                        pos += 1;
                    }
                    if accepted_here {
                        cur_end = pos as i32;
                    }
                    if hit_non_ascii {
                        continue;
                    }
                    if pos < end {
                        if exit_row == NO_EDGE {
                            break;
                        }
                        row = exit_row as usize;
                        pos += 1;
                        continue;
                    }
                    let eof_row = unsafe { dfa.meta.get_unchecked(row).eof_row };
                    if eof_row != NO_EDGE {
                        row = eof_row as usize;
                        continue;
                    }
                    break;
                }
                row = nr;
            } else {
                let m = unsafe { *dfa.meta.get_unchecked(row) };
                let mut low = 0usize;
                let mut high = m.edge_count as usize;
                let base = m.edge_start as usize;
                let mut matched = false;
                while low < high {
                    let mid = (low + high) >> 1;
                    let e = unsafe { *dfa.hi_edges.get_unchecked(base + mid) };
                    let from = e[0] as i32;
                    let to = if e[1] == 0 { 0x10000 } else { e[1] as i32 };
                    if ch < from {
                        high = mid;
                    } else if ch >= to {
                        low = mid + 1;
                    } else {
                        row = e[2] as usize;
                        pos += 1;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    break;
                }
            }
        } else {
            let eof_row = unsafe { dfa.meta.get_unchecked(row).eof_row };
            if eof_row != NO_EDGE {
                row = eof_row as usize;
            } else {
                break;
            }
        }
    }

    let la = if pos > start {
        let max_la = if pos < end { pos as i32 + 1 } else { end as i32 };
        max_la.max(start as i32 + 1)
    } else {
        start as i32 + 1
    };
    (cur, cur_end, la)
}

