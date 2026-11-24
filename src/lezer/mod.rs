//! Port of the `@lezer/lr` runtime (v1.4.12) with the parse tables of
//! `@lezer/python` 1.1.19, plus its external tokenizers and context tracker.
//! TypeScript and JavaScript are parsed by oxc (`src/lang_ts.rs`). Positions are UTF-16 code unit offsets,
//! as in JavaScript; the parser is generic over the code unit type so ASCII
//! sources are parsed straight from their bytes.

#![allow(clippy::too_many_arguments)]

use std::sync::OnceLock;

use rustc_hash::FxHashMap;

#[allow(clippy::all, clippy::unreadable_literal, dead_code)]
pub mod tables_py;

mod parse;
mod stack;
mod stream;
mod tree;

pub use parse::{parse_into, shrink_tl_workspace};
pub use stream::AsciiByte;
pub use tree::{SyntaxTree, NONE};

use stream::DfaAccel;

#[derive(Clone, Copy, Debug)]
pub enum Tok {
    Group(u16),
    External { contextual: bool, fallback: bool, extend: bool },
}

pub struct Grammar {
    pub(crate) states: &'static [u32],
    pub(crate) data: &'static [u16],
    pub(crate) goto: &'static [u16],
    pub(crate) token_data: &'static [u16],
    pub(crate) token_prec: usize,
    pub min_repeat_term: u16,
    pub max_term: u16,
    pub max_node: u16,
    pub(crate) top_state: u32,
    pub top_term: u16,
    pub node_names: &'static [&'static str],
    #[allow(dead_code)]
    pub(crate) dyn_prec: &'static [i32],
    pub(crate) tokenizers: &'static [Tok],
    pub(crate) specialized: &'static [u16],
    pub(crate) specialize: fn(usize, &[u8]) -> i32,
    pub(crate) disabled: &'static [u8],
    dfa_cell: &'static OnceLock<DfaAccel>,
    action_cell: &'static OnceLock<ActionTable>,
    goto_cell: &'static OnceLock<GotoTable>,
}

static PY_DFA: OnceLock<DfaAccel> = OnceLock::new();
static PY_ACTIONS: OnceLock<ActionTable> = OnceLock::new();
static PY_GOTO: OnceLock<GotoTable> = OnceLock::new();

/// Compact single-target-accelerated goto table.
/// Over 80% of LR non-terminals have a single unconditional goto target across
/// all states; those resolve in a single load from `single[term]` (300 bytes in L1).
/// Only multi-target non-terminals occupy columns in `multi`, cutting working set ~5x.
pub(crate) struct GotoTable {
    pub nstates: usize,
    /// `single[term]`: direct target state if unique for all states, else `u16::MAX`.
    pub single: Vec<u16>,
    /// For multi-target terms, column index in `multi`.
    pub multi_col: Vec<u16>,
    /// Column-major `[col * nstates + state]` -> target state.
    pub multi: Vec<u16>,
}

impl GotoTable {
    fn build(g: &Grammar) -> Self {
        let nstates = g.states.len() / 6;
        let stride = g.goto[0] as usize;
        let table = g.goto;

        let mut single = vec![u16::MAX; stride];
        let mut multi_col = vec![u16::MAX; stride];
        let mut multi_cols: Vec<Vec<u16>> = Vec::new();

        for term in 0..stride {
            let mut pos = table[term + 1] as usize;
            let first_tag = table[pos] as usize;
            let first_target = table[pos + 1];
            if (first_tag & 1) != 0 {
                single[term] = first_target;
                continue;
            }
            let mut scan = pos;
            let default_target;
            loop {
                let group_tag = table[scan] as usize;
                let target = table[scan + 1];
                scan += 2 + (group_tag >> 1);
                if (group_tag & 1) != 0 {
                    default_target = target;
                    break;
                }
            }
            let mut col_data = vec![default_target; nstates];
            loop {
                let group_tag = table[pos] as usize;
                let target = table[pos + 1];
                pos += 2;
                let last = group_tag & 1;
                if last != 0 {
                    break;
                }
                let end = pos + (group_tag >> 1);
                while pos < end {
                    let s = table[pos] as usize;
                    if s < nstates {
                        col_data[s] = target;
                    }
                    pos += 1;
                }
            }
            multi_col[term] = multi_cols.len() as u16;
            multi_cols.push(col_data);
        }

        let mut multi = Vec::with_capacity(multi_cols.len() * nstates);
        for col in multi_cols {
            multi.extend_from_slice(&col);
        }
        Self { nstates, single, multi_col, multi }
    }
}

pub(crate) const MULTI_BIT: u32 = 0x8000_0000;

/// 16-byte aligned packed LR state descriptor combining `default_reduce`,
/// `tok_mask`, pre-multiplied `act_row_off` (`u32::MAX` if `SEQ_OTHER`), and `flags`
/// into a single cache-line-friendly load per LR step.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct FastState {
    pub default_reduce: u32,
    pub tok_mask: u32,
    pub act_row_off: u32,
    pub flags: u32,
}

/// Row-deduplicated `u16`-indexed action table (~8-10x smaller than raw 2D `u32`
/// table, keeping working set comfortably inside L2 cache) plus `FastState` and
/// O(1) `spec_idx`.
pub(crate) struct ActionTable {
    pub stride: usize,
    pub fast_states: Vec<FastState>,
    /// Flat `[act_row_off + term]` -> `0` (empty) or `idx` into `lut`.
    pub rows: Vec<u16>,
    /// `lut[0] = u32::MAX` (unused); `lut[idx]` is single action or `MULTI_BIT | off`.
    pub lut: Vec<u32>,
    /// Multi-action sequences: `[count, a0, a1, ...]`.
    pub multi: Vec<u32>,
    /// `spec_idx[term]`: index in `g.specialized` (or `-1` if not specialized).
    pub spec_idx: Vec<i8>,
}

impl ActionTable {
    #[inline(always)]
    pub fn fast_state(&self, state: u32) -> FastState {
        unsafe { *self.fast_states.get_unchecked(state as usize) }
    }

    fn build(g: &Grammar) -> Self {
        let nstates = g.states.len() / 6;
        let stride = g.max_term as usize + 1;
        let data = g.data;

        let mut fast_states = Vec::with_capacity(nstates);
        let mut pair_to_row: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut rows: Vec<u16> = Vec::new();
        let mut lut: Vec<u32> = vec![u32::MAX]; // index 0 = empty
        let mut lut_map: FxHashMap<u32, u16> = FxHashMap::default();
        let mut multi: Vec<u32> = Vec::new();
        let mut multi_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();

        let mut row_entries: Vec<(u16, u32)> = Vec::with_capacity(64);
        let mut acts: Vec<u32> = Vec::with_capacity(8);

        for s in 0..nstates {
            let flags = g.state_slot(s as u32, 0);
            let s1 = g.state_slot(s as u32, 1);
            let s2 = g.state_slot(s as u32, 2);
            let tok_mask = g.state_slot(s as u32, 3);
            let default_reduce = g.state_slot(s as u32, 4);

            let act_row_off = if let Some(&existing_off) = pair_to_row.get(&(s1, s2)) {
                existing_off
            } else {
                row_entries.clear();
                let mut other = false;
                for start in [s1, s2] {
                    let mut i = start as usize;
                    loop {
                        let d = data[i];
                        if d == SEQ_END {
                            if data[i + 1] == SEQ_NEXT {
                                i = pair(data, i + 2) as usize;
                                continue;
                            }
                            if data[i + 1] == SEQ_OTHER {
                                other = true;
                            }
                            break;
                        }
                        let act = pair(data, i + 1);
                        row_entries.push((d, act));
                        i += 3;
                    }
                }
                if other {
                    pair_to_row.insert((s1, s2), u32::MAX);
                    u32::MAX
                } else {
                    let row_base = rows.len();
                    let off_u32 = row_base as u32;
                    pair_to_row.insert((s1, s2), off_u32);
                    rows.resize(row_base + stride, 0u16);

                    for idx in 0..row_entries.len() {
                        let (term, act) = row_entries[idx];
                        let t = term as usize;
                        if t >= stride || rows[row_base + t] != 0 {
                            continue;
                        }
                        acts.clear();
                        acts.push(act);
                        for &(t2, a2) in &row_entries[idx + 1..] {
                            if t2 == term && !acts.contains(&a2) {
                                acts.push(a2);
                            }
                        }
                        let cell = if acts.len() == 1 {
                            debug_assert_eq!(act & MULTI_BIT, 0);
                            act
                        } else if let Some(&off) = multi_map.get(&acts) {
                            MULTI_BIT | off
                        } else {
                            let off = multi.len() as u32;
                            multi.push(acts.len() as u32);
                            multi.extend_from_slice(&acts);
                            multi_map.insert(acts.clone(), off);
                            MULTI_BIT | off
                        };
                        let lut_idx = if let Some(&id) = lut_map.get(&cell) {
                            id
                        } else {
                            let id = lut.len() as u16;
                            lut.push(cell);
                            lut_map.insert(cell, id);
                            id
                        };
                        rows[row_base + t] = lut_idx;
                    }
                    off_u32
                }
            };

            fast_states.push(FastState { default_reduce, tok_mask, act_row_off, flags });
        }

        let mut spec_idx = vec![-1i8; stride];
        for (i, &spec_term) in g.specialized.iter().enumerate() {
            let t = spec_term as usize;
            if t < stride {
                spec_idx[t] = i as i8;
            }
        }

        Self { stride, fast_states, rows, lut, multi, spec_idx }
    }
}

pub static PYTHON: Grammar = Grammar {
    states: &tables_py::PY_STATES,
    data: &tables_py::PY_DATA,
    goto: &tables_py::PY_GOTO,
    token_data: &tables_py::PY_TOKEN_DATA,
    token_prec: tables_py::PY_TOKEN_PREC,
    min_repeat_term: tables_py::PY_MIN_REPEAT_TERM,
    max_term: tables_py::PY_MAX_TERM,
    max_node: tables_py::PY_MAX_NODE,
    top_state: tables_py::PY_TOP_STATE,
    top_term: tables_py::PY_TOP_TERM,
    node_names: &tables_py::PY_NODE_NAMES,
    dyn_prec: &tables_py::PY_DYN_PREC,
    tokenizers: &tables_py::PY_TOKENIZERS,
    specialized: &tables_py::PY_SPECIALIZED,
    specialize: tables_py::py_specialize,
    disabled: &tables_py::PY_DIALECT_DEFAULT_DISABLED,
    dfa_cell: &PY_DFA,
    action_cell: &PY_ACTIONS,
    goto_cell: &PY_GOTO,
};


// Action / state encoding constants.
pub(crate) const REDUCE_FLAG: u32 = 65536;
pub(crate) const VALUE_MASK: u32 = 65535;
pub(crate) const REDUCE_DEPTH_SHIFT: u32 = 19;
pub(crate) const REPEAT_FLAG: u32 = 131072;
pub(crate) const GOTO_FLAG: u32 = 131072;
pub(crate) const STAY_FLAG: u32 = 262144;
pub(crate) const SEQ_END: u16 = 65535;
pub(crate) const SEQ_NEXT: u16 = 1;
pub(crate) const SEQ_OTHER: u16 = 2;
pub(crate) const STATE_SKIPPED: u32 = 1;
pub(crate) const STATE_ACCEPTING: u32 = 2;
pub(crate) const LOOKAHEAD_MARGIN: i32 = 25;
pub(crate) const TERM_ERR: i32 = 0;

#[inline(always)]
pub(crate) fn pair(data: &[u16], off: usize) -> u32 {
    unsafe {
        *data.get_unchecked(off) as u32 | ((*data.get_unchecked(off + 1) as u32) << 16)
    }
}

impl Grammar {
    #[inline(always)]
    pub(crate) fn dfa(&self) -> &'static DfaAccel {
        self.dfa_cell.get_or_init(|| DfaAccel::build(self))
    }

    #[inline(always)]
    pub(crate) fn action_table(&self) -> &'static ActionTable {
        self.action_cell.get_or_init(|| ActionTable::build(self))
    }

    #[inline(always)]
    pub(crate) fn goto_table(&self) -> &'static GotoTable {
        self.goto_cell.get_or_init(|| GotoTable::build(self))
    }

    #[inline(always)]
    pub(crate) fn get_goto_loose(&self, gt: &GotoTable, state: u32, term: u32) -> u32 {
        let t = term as usize;
        let s = unsafe { *gt.single.get_unchecked(t) };
        if s != u16::MAX {
            return s as u32;
        }
        let col = unsafe { *gt.multi_col.get_unchecked(t) } as usize;
        unsafe { *gt.multi.get_unchecked(col * gt.nstates + state as usize) as u32 }
    }

    #[inline(always)]
    pub(crate) fn state_slot(&self, state: u32, slot: usize) -> u32 {
        unsafe { *self.states.get_unchecked(state as usize * 6 + slot) }
    }
    #[inline(always)]
    pub(crate) fn allows(&self, term: u16) -> bool {
        self.disabled.is_empty() || unsafe { *self.disabled.get_unchecked(term as usize) == 0 }
    }
    #[inline(always)]
    pub(crate) fn eof_term(&self) -> i32 {
        self.max_node as i32 + 1
    }

    #[inline]
    pub(crate) fn get_goto(&self, state: u32, term: u32, loose: bool) -> i32 {
        let table = self.goto;
        if term >= table[0] as u32 {
            return -1;
        }
        let mut pos = unsafe { *table.get_unchecked(term as usize + 1) } as usize;
        loop {
            let group_tag = unsafe { *table.get_unchecked(pos) } as usize;
            let target = unsafe { *table.get_unchecked(pos + 1) } as i32;
            pos += 2;
            let last = group_tag & 1;
            if last != 0 && loose {
                return target;
            }
            let end = pos + (group_tag >> 1);
            while pos < end {
                if unsafe { *table.get_unchecked(pos) } as u32 == state {
                    return target;
                }
                pos += 1;
            }
            if last != 0 {
                return -1;
            }
        }
    }

    pub(crate) fn has_action(&self, state: u32, terminal: i32) -> u32 {
        let data = self.data;
        for set in 0..2 {
            let mut i = self.state_slot(state, if set == 1 { 2 } else { 1 }) as usize;
            loop {
                let mut next = unsafe { *data.get_unchecked(i) };
                if next == SEQ_END {
                    let flag = unsafe { *data.get_unchecked(i + 1) };
                    if flag == SEQ_NEXT {
                        i = pair(data, i + 2) as usize;
                        next = unsafe { *data.get_unchecked(i) };
                    } else if flag == SEQ_OTHER {
                        return pair(data, i + 2);
                    } else {
                        break;
                    }
                }
                if next as i32 == terminal || next as i32 == TERM_ERR {
                    return pair(data, i + 1);
                }
                i += 3;
            }
        }
        0
    }

    pub(crate) fn all_actions<T>(&self, state: u32, mut f: impl FnMut(u32) -> Option<T>) -> Option<T> {
        let deflt = self.state_slot(state, 4);
        let mut result = if deflt != 0 { f(deflt) } else { None };
        let mut i = self.state_slot(state, 1) as usize;
        while result.is_none() {
            if self.data[i] == SEQ_END {
                if self.data[i + 1] == SEQ_NEXT {
                    i = pair(self.data, i + 2) as usize;
                } else {
                    break;
                }
            }
            result = f(pair(self.data, i + 1));
            i += 3;
        }
        result
    }

    pub(crate) fn valid_action(&self, state: u32, action: u32) -> bool {
        self.all_actions(state, |a| if a == action { Some(()) } else { None }).is_some()
    }

    pub(crate) fn next_states(&self, state: u32) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::new();
        let mut i = self.state_slot(state, 1) as usize;
        loop {
            if self.data[i] == SEQ_END {
                if self.data[i + 1] == SEQ_NEXT {
                    i = pair(self.data, i + 2) as usize;
                } else {
                    break;
                }
            }
            if (self.data[i + 2] as u32 & (REDUCE_FLAG >> 16)) == 0 {
                let value = self.data[i + 1] as u32;
                if !result.iter().enumerate().any(|(j, &v)| (j & 1) == 1 && v == value) {
                    result.push(self.data[i] as u32);
                    result.push(value);
                }
            }
            i += 3;
        }
        result
    }
}
