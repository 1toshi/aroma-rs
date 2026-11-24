//! Arena-backed parse stacks (`Stack` in @lezer/lr) with inline `buffer: Vec<Rec>`
//! (zero indirection on shift/reduce pushes) and slot recycling via `StackArena`
//! (zero heap allocation on GLR splits).

use super::parse::Env;
use super::stream::Ch;
use super::*;

pub(crate) const NO_PARENT: u32 = u32::MAX;
pub(crate) type Rec = [i32; 4];

#[derive(Clone, Copy)]
pub(crate) struct Frame {
    pub state: u32,
    pub start: i32,
    /// Cumulative record count (`buffer_base + buffer.len()`) when this frame was pushed.
    pub buf_pos: u32,
}

pub(crate) struct Stack {
    pub stack: Vec<Frame>,
    pub state: u32,
    pub reduce_pos: i32,
    pub pos: i32,
    pub score: i32,
    pub buffer: Vec<Rec>,
    pub buffer_base: usize,
    pub parent: u32,
    pub ctx: u32,
    pub ctx_flags: i32,
    pub ctx_hash: i64,
    pub look_ahead: i32,
}

impl Stack {
    fn empty() -> Self {
        Self {
            stack: Vec::new(),
            state: 0,
            reduce_pos: 0,
            pos: 0,
            score: 0,
            buffer: Vec::new(),
            buffer_base: 0,
            parent: NO_PARENT,
            ctx: 0,
            ctx_flags: 0,
            ctx_hash: 0,
            look_ahead: 0,
        }
    }

    pub fn init(&mut self, state: u32, pos: i32, ctx_hash: i64) {
        self.stack.clear();
        self.buffer.clear();
        self.state = state;
        self.reduce_pos = pos;
        self.pos = pos;
        self.score = 0;
        self.buffer_base = 0;
        self.parent = NO_PARENT;
        self.ctx = 0;
        self.ctx_flags = 0;
        self.ctx_hash = ctx_hash;
        self.look_ahead = 0;
    }

    /// Output buffer length in `i32` units (matching @lezer's `stack.buffer.length`).
    #[inline(always)]
    pub fn buf_len_i32(&self) -> usize {
        self.buffer.len() << 2
    }

    #[inline(always)]
    pub fn push_state(&mut self, state: u32, start: i32) {
        let bl = (self.buffer_base + self.buffer.len()) as u32;
        self.stack.push(Frame { state: self.state, start, buf_pos: bl });
        self.state = state;
    }

    #[inline]
    pub fn reduce<C: Ch>(&mut self, env: &mut Env<C>, action: u32) {
        let depth = action >> REDUCE_DEPTH_SHIFT;
        let ty = action & VALUE_MASK;
        let g = env.g;
        let lookahead_record =
            self.reduce_pos < self.pos - LOOKAHEAD_MARGIN && self.set_look_ahead(self.pos);
        let min_repeat = g.min_repeat_term as u32;
        if depth == 0 {
            if ty < min_repeat && self.reduce_pos < self.pos {
                self.reduce_pos = self.pos;
            }
            let goto = g.get_goto_loose(env.gt, self.state, ty);
            self.push_state(goto, self.reduce_pos);
            if ty < min_repeat {
                let sz = if lookahead_record { 8 } else { 4 };
                if self.pos == self.reduce_pos {
                    self.buffer.push([ty as i32, self.reduce_pos, self.reduce_pos, sz]);
                } else {
                    self.store_node(ty as i32, self.reduce_pos, self.reduce_pos, sz, true);
                }
            }
            self.reduce_context_py(env, ty);
            return;
        }
        let base = self.stack.len()
            - (depth as usize - 1)
            - if action & STAY_FLAG != 0 { 2 } else { 0 };
        let (start, buffer_base, base_state) = if base > 0 {
            let f = unsafe { *self.stack.get_unchecked(base - 1) };
            (f.start, f.buf_pos as usize, f.state)
        } else {
            (0, 0, 0)
        };
        if ty < min_repeat && start == self.reduce_pos && self.reduce_pos < self.pos {
            self.reduce_pos = self.pos;
        }
        let size = self.reduce_pos - start;
        if size >= 2000 && !(ty >= min_repeat && ty <= g.max_node as u32) {
            if start == env.last_big_start {
                env.big_count += 1;
                env.last_big_size = size;
            } else if env.last_big_size < size {
                env.big_count = 1;
                env.last_big_start = start;
                env.last_big_size = size;
            }
        }
        let count_recs = (self.buffer_base + self.buffer.len()) as i64 - buffer_base as i64;
        if ty < min_repeat || (action & REPEAT_FLAG) != 0 {
            let pos = if self.reduce_pos == self.pos
                || (env.at.fast_state(self.state).flags & STATE_SKIPPED) != 0
            {
                self.pos
            } else {
                self.reduce_pos
            };
            let byte_sz = ((count_recs + 1) << 2) as i32;
            if self.pos == pos {
                self.buffer.push([ty as i32, start, pos, byte_sz]);
            } else {
                self.store_node_sink(ty as i32, start, pos, byte_sz);
            }
        }
        if action & STAY_FLAG != 0 {
            self.state = unsafe { self.stack.get_unchecked(base).state };
        } else {
            self.state = g.get_goto_loose(env.gt, base_state, ty);
        }
        self.stack.truncate(base);
        self.reduce_context_py(env, ty);
    }

    #[inline]
    pub fn store_node(
        &mut self,
        term: i32,
        start: i32,
        end: i32,
        size: i32,
        must_sink: bool,
    ) {
        if term == TERM_ERR
            && (self.stack.is_empty()
                || (self.stack.last().unwrap().buf_pos as usize)
                    < self.buffer.len() + self.buffer_base)
            && let Some(last) = self.buffer.last_mut()
            && last[0] == TERM_ERR
            && last[3] > -1
        {
            if start == end {
                return;
            }
            if last[2] >= start {
                last[2] = end;
                return;
            }
        }
        if !must_sink || self.pos == end {
            self.buffer.push([term, start, end, size]);
        } else {
            self.store_node_sink(term, start, end, size);
        }
    }

    #[cold]
    fn store_node_sink(&mut self, term: i32, start: i32, end: i32, mut size: i32) {
        let buffer = &mut self.buffer;
        let mut index = buffer.len();
        if index > 0 && (buffer[index - 1][0] != TERM_ERR || buffer[index - 1][3] < 0) {
            let mut must_move = false;
            let mut scan = index;
            while scan > 0 && buffer[scan - 1][2] > end {
                if buffer[scan - 1][3] >= 0 {
                    must_move = true;
                    break;
                }
                scan -= 1;
            }
            if must_move {
                buffer.push([0; 4]);
                while index > 0 && buffer[index - 1][2] > end {
                    buffer[index] = buffer[index - 1];
                    index -= 1;
                    if size > 4 {
                        size -= 4;
                    }
                }
                buffer[index] = [term, start, end, size];
                return;
            }
        }
        buffer.push([term, start, end, size]);
    }

    #[inline]
    pub fn shift<C: Ch>(
        &mut self,
        env: &mut Env<C>,
        action: u32,
        ty: i32,
        start: i32,
        end: i32,
    ) {
        let g = env.g;
        if action & GOTO_FLAG != 0 {
            self.push_state(action & VALUE_MASK, self.pos);
        } else if action & STAY_FLAG == 0 {
            let next_state = action;
            self.pos = end;
            let skipped = (env.at.fast_state(next_state).flags & STATE_SKIPPED) != 0;
            if !skipped && (end > start || ty <= g.max_node as i32) {
                self.reduce_pos = end;
            }
            self.push_state(next_state, if skipped { start } else { start.min(self.reduce_pos) });
            self.shift_context::<C>(env, ty as u32, start);
            if ty <= g.max_node as i32 {
                self.buffer.push([ty, start, end, 4]);
            }
        } else {
            self.pos = end;
            self.shift_context::<C>(env, ty as u32, start);
            if ty <= g.max_node as i32 {
                self.buffer.push([ty, start, end, 4]);
            }
        }
    }

    #[inline(always)]
    pub fn apply<C: Ch>(
        &mut self,
        env: &mut Env<C>,
        action: u32,
        next: i32,
        next_start: i32,
        next_end: i32,
    ) {
        if action & REDUCE_FLAG != 0 {
            self.reduce::<C>(env, action);
        } else {
            self.shift::<C>(env, action, next, next_start, next_end);
        }
    }

    pub fn recover_by_delete(&mut self, g: &Grammar, next: i32, next_end: i32) {
        let is_node = next <= g.max_node as i32;
        if is_node {
            self.store_node(next, self.pos, next_end, 4, false);
        }
        self.store_node(TERM_ERR, self.pos, next_end, if is_node { 8 } else { 4 }, false);
        self.pos = next_end;
        self.reduce_pos = next_end;
        self.score -= 190;
    }

    pub fn can_shift(&self, g: &Grammar, at: &ActionTable, term: i32) -> bool {
        let mut state = self.state;
        let mut copy: Option<Vec<u32>> = None;
        let mut base = self.stack.len() as i64;
        loop {
            let dr = at.fast_state(state).default_reduce;
            let action = if dr != 0 { dr } else { g.has_action(state, term) };
            if action == 0 {
                return false;
            }
            if action & REDUCE_FLAG == 0 {
                return true;
            }
            let t = action & VALUE_MASK;
            let depth = action >> REDUCE_DEPTH_SHIFT;
            if depth == 0 {
                let v = copy.get_or_insert_with(|| self.stack.iter().map(|f| f.state).collect());
                v.push(state);
                base += 1;
            } else {
                base -= depth as i64 - 1;
            }
            let below = if let Some(ref v) = copy {
                if base >= 1 { v.get(base as usize - 1).copied().unwrap_or(u32::MAX) } else { u32::MAX }
            } else if base >= 1 {
                self.stack.get(base as usize - 1).map_or(u32::MAX, |f| f.state)
            } else {
                u32::MAX
            };
            state = g.get_goto(below, t, true) as u32;
        }
    }

    pub fn force_reduce<C: Ch>(&mut self, env: &mut Env<C>) -> bool {
        let g = env.g;
        let mut reduce = g.state_slot(self.state, 5);
        if reduce & REDUCE_FLAG == 0 {
            return false;
        }
        if !g.valid_action(self.state, reduce) {
            let depth = reduce >> REDUCE_DEPTH_SHIFT;
            let term = reduce & VALUE_MASK;
            let target = self.stack.len() as i64 - depth as i64;
            if target < 0 || g.get_goto(self.stack[target as usize].state, term, false) < 0 {
                match self.find_forced_reduction(g) {
                    Some(b) => reduce = b,
                    None => return false,
                }
            }
            self.store_node(TERM_ERR, self.pos, self.pos, 4, true);
            self.score -= 100;
        }
        self.reduce_pos = self.pos;
        self.reduce::<C>(env, reduce);
        true
    }

    fn find_forced_reduction(&self, g: &Grammar) -> Option<u32> {
        let mut seen: Vec<u32> = Vec::new();
        self.explore(g, self.state, 0, &mut seen)
    }

    fn explore(&self, g: &Grammar, state: u32, depth: i64, seen: &mut Vec<u32>) -> Option<u32> {
        if seen.contains(&state) {
            return None;
        }
        seen.push(state);
        g.all_actions(state, |action| {
            if action & (STAY_FLAG | GOTO_FLAG) != 0 {
                None
            } else if action & REDUCE_FLAG != 0 {
                let r_depth = (action >> REDUCE_DEPTH_SHIFT) as i64 - depth;
                if r_depth > 1 {
                    let term = action & VALUE_MASK;
                    let target = self.stack.len() as i64 - r_depth;
                    if target >= 0 && g.get_goto(self.stack[target as usize].state, term, false) >= 0 {
                        return Some(((r_depth as u32) << REDUCE_DEPTH_SHIFT) | REDUCE_FLAG | term);
                    }
                }
                None
            } else {
                self.explore(g, action, depth + 1, seen)
            }
        })
    }

    pub fn force_all<C: Ch>(&mut self, env: &mut Env<C>) {
        while (env.at.fast_state(self.state).flags & STATE_ACCEPTING) == 0 {
            if !self.force_reduce::<C>(env) {
                self.store_node(TERM_ERR, self.pos, self.pos, 4, true);
                break;
            }
        }
    }

    pub fn dead_end(&self, g: &Grammar, at: &ActionTable) -> bool {
        if self.stack.len() != 1 {
            return false;
        }
        g.data[g.state_slot(self.state, 1) as usize] == SEQ_END
            && at.fast_state(self.state).default_reduce == 0
    }

    pub fn restart(&mut self) {
        self.store_node(TERM_ERR, self.pos, self.pos, 4, true);
        self.state = self.stack[0].state;
        self.stack.clear();
    }

    pub fn same_state(&self, other: &Stack) -> bool {
        if self.state != other.state || self.stack.len() != other.stack.len() {
            return false;
        }
        for (a, b) in self.stack.iter().zip(other.stack.iter()) {
            if a.state != b.state {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    pub fn shift_context<C: Ch>(&mut self, env: &mut Env<C>, term: u32, start: i32) {
        let cx = env.tracker_shift_py(self.ctx, term, self.pos, start);
        if cx != self.ctx {
            self.update_context_changed_py(env, cx);
        }
    }

    #[inline(always)]
    pub fn reduce_context_py<C: Ch>(&mut self, env: &mut Env<C>, term: u32) {
        if self.ctx_flags != 0 {
            let cx = env.tracker_reduce_py(self.ctx, self.ctx_flags, term);
            if cx != self.ctx {
                self.update_context_changed_py(env, cx);
            }
        }
    }

    fn emit_context(&mut self) {
        if self.buffer.last().map(|r| r[3]) != Some(-3) {
            let h = (self.ctx_hash != 0) as i32;
            self.store_node(h, self.reduce_pos, self.reduce_pos, -3, true);
        }
    }

    #[cold]
    fn emit_look_ahead(&mut self) {
        if self.buffer.last().map(|r| r[3]) != Some(-4) {
            self.buffer.push([self.look_ahead, self.pos, self.pos, -4]);
        }
    }

    #[inline]
    fn update_context_changed_py<C: Ch>(&mut self, env: &mut Env<C>, cx: u32) {
        let (flags, h) = env.py_ctx_info(cx);
        if h != self.ctx_hash {
            self.emit_context();
        }
        self.ctx = cx;
        self.ctx_flags = flags;
        self.ctx_hash = h;
    }

    #[inline]
    pub fn set_look_ahead(&mut self, look_ahead: i32) -> bool {
        if look_ahead <= self.look_ahead {
            return false;
        }
        self.emit_look_ahead();
        self.look_ahead = look_ahead;
        true
    }

    pub fn close(&mut self, strict: bool) {
        if strict {
            self.emit_context();
        }
        if self.look_ahead > 0 {
            self.emit_look_ahead();
        }
    }
}

// ============================================================================
// Reusable `StackArena` with slot recycling & parent-chain GC
// ============================================================================

#[derive(Default)]
pub(crate) struct StackArena {
    pub v: Vec<Stack>,
    free: Vec<u32>,
    mark: Vec<bool>,
    chain_tmp: Vec<u32>,
    buf_tmp: Vec<Rec>,
    pub since_gc: usize,
}

impl StackArena {
    pub fn reset(&mut self) {
        if self.v.len() > 4 {
            self.v.truncate(4);
        }
        if let Some(first) = self.v.first_mut() {
            if first.buffer.capacity() > 16384 {
                first.buffer.clear();
                first.buffer.shrink_to(4096);
            }
            if first.stack.capacity() > 512 {
                first.stack.clear();
                first.stack.shrink_to(128);
            }
        }
        for s in self.v.iter_mut().skip(1) {
            if s.buffer.capacity() > 2048 {
                s.buffer.clear();
                s.buffer.shrink_to(512);
            }
            if s.stack.capacity() > 256 {
                s.stack.clear();
                s.stack.shrink_to(64);
            }
        }
        self.free.clear();
        for i in (0..self.v.len()).rev() {
            self.free.push(i as u32);
        }
        self.since_gc = 0;
    }

    pub fn release(&mut self) {
        self.v.clear();
        self.v.shrink_to_fit();
        self.free.clear();
        self.free.shrink_to_fit();
        self.mark.clear();
        self.mark.shrink_to_fit();
        self.chain_tmp.clear();
        self.chain_tmp.shrink_to_fit();
        self.buf_tmp.clear();
        self.buf_tmp.shrink_to_fit();
        self.since_gc = 0;
    }

    /// When GLR converges back to a single active stack `sid`, collapses its
    /// parent chain into slot 0 in-place and frees all other slots. This keeps
    /// `self.v.len()` bounded by max simultaneous GLR branches (~4-16) instead
    /// of total splits in the file (which can reach thousands on large TS files).
    #[inline]
    pub fn collapse_to_root(&mut self, sid: u32) -> u32 {
        if sid == 0 && self.v[0].parent == NO_PARENT {
            if self.since_gc > 0 {
                self.free.clear();
                for i in (1..self.v.len()).rev() {
                    self.free.push(i as u32);
                }
                self.since_gc = 0;
            }
            return 0;
        }
        self.collapse_to_root_slow(sid)
    }

    fn collapse_to_root_slow(&mut self, sid: u32) -> u32 {
        let sid_u = sid as usize;
        if sid != 0 && self.v[sid_u].parent == NO_PARENT {
            self.v.swap(0, sid_u);
        } else if sid != 0 && self.v[sid_u].parent == 0 && self.v[0].parent == NO_PARENT {
            let (l, r) = self.v.split_at_mut(sid_u);
            let root = &mut l[0];
            let child = &mut r[0];
            let upto = child.buffer_base - root.buffer_base;
            root.buffer.truncate(upto);
            root.buffer.extend_from_slice(&child.buffer);
            std::mem::swap(&mut root.stack, &mut child.stack);
            root.state = child.state;
            root.reduce_pos = child.reduce_pos;
            root.pos = child.pos;
            root.score = child.score;
            root.buffer_base = 0;
            root.parent = NO_PARENT;
            root.ctx = child.ctx;
            root.ctx_flags = child.ctx_flags;
            root.ctx_hash = child.ctx_hash;
            root.look_ahead = child.look_ahead;
        } else {
            self.chain_tmp.clear();
            let mut cur = sid;
            while cur != NO_PARENT {
                self.chain_tmp.push(cur);
                cur = self.v[cur as usize].parent;
            }
            let clen = self.chain_tmp.len();
            if clen >= 2 && self.chain_tmp[clen - 1] == 0 {
                let first_child = self.chain_tmp[clen - 2] as usize;
                let upto = self.v[first_child].buffer_base - self.v[0].buffer_base;
                self.v[0].buffer.truncate(upto);
                for k in (0..clen - 1).rev() {
                    let cid = self.chain_tmp[k] as usize;
                    let c_upto = if k == 0 {
                        self.v[cid].buffer.len()
                    } else {
                        self.v[self.chain_tmp[k - 1] as usize].buffer_base - self.v[cid].buffer_base
                    };
                    let (l, r) = self.v.split_at_mut(cid);
                    l[0].buffer.extend_from_slice(&r[0].buffer[..c_upto]);
                }
                let (l, r) = self.v.split_at_mut(sid_u);
                let root = &mut l[0];
                let leaf = &mut r[0];
                std::mem::swap(&mut root.stack, &mut leaf.stack);
                root.state = leaf.state;
                root.reduce_pos = leaf.reduce_pos;
                root.pos = leaf.pos;
                root.score = leaf.score;
                root.buffer_base = 0;
                root.parent = NO_PARENT;
                root.ctx = leaf.ctx;
                root.ctx_flags = leaf.ctx_flags;
                root.ctx_hash = leaf.ctx_hash;
                root.look_ahead = leaf.look_ahead;
            } else {
                self.buf_tmp.clear();
                for k in (0..clen).rev() {
                    let cid = self.chain_tmp[k] as usize;
                    let c_upto = if k == 0 {
                        self.v[cid].buffer.len()
                    } else {
                        self.v[self.chain_tmp[k - 1] as usize].buffer_base - self.v[cid].buffer_base
                    };
                    self.buf_tmp.extend_from_slice(&self.v[cid].buffer[..c_upto]);
                }
                let (state, reduce_pos, pos, score, ctx, ctx_flags, ctx_hash, look_ahead) = {
                    let leaf = &self.v[sid_u];
                    (
                        leaf.state,
                        leaf.reduce_pos,
                        leaf.pos,
                        leaf.score,
                        leaf.ctx,
                        leaf.ctx_flags,
                        leaf.ctx_hash,
                        leaf.look_ahead,
                    )
                };
                if sid != 0 {
                    let (l, r) = self.v.split_at_mut(sid_u);
                    std::mem::swap(&mut l[0].stack, &mut r[0].stack);
                }
                let root = &mut self.v[0];
                std::mem::swap(&mut root.buffer, &mut self.buf_tmp);
                self.buf_tmp.clear();
                root.state = state;
                root.reduce_pos = reduce_pos;
                root.pos = pos;
                root.score = score;
                root.buffer_base = 0;
                root.parent = NO_PARENT;
                root.ctx = ctx;
                root.ctx_flags = ctx_flags;
                root.ctx_hash = ctx_hash;
                root.look_ahead = look_ahead;
            }
        }
        self.free.clear();
        for i in (1..self.v.len()).rev() {
            self.free.push(i as u32);
        }
        self.since_gc = 0;
        0
    }

    #[inline(always)]
    pub fn get(&self, id: u32) -> &Stack {
        unsafe { self.v.get_unchecked(id as usize) }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, id: u32) -> &mut Stack {
        unsafe { self.v.get_unchecked_mut(id as usize) }
    }

    pub fn alloc_start(&mut self, state: u32, pos: i32, ctx_hash: i64) -> u32 {
        let id = self.alloc_slot();
        self.v[id as usize].init(state, pos, ctx_hash);
        id
    }

    #[inline]
    fn alloc_slot(&mut self) -> u32 {
        self.since_gc += 1;
        if let Some(i) = self.free.pop() {
            let s = unsafe { self.v.get_unchecked_mut(i as usize) };
            s.stack.clear();
            s.buffer.clear();
            return i;
        }
        let id = self.v.len() as u32;
        self.v.push(Stack::empty());
        id
    }

    #[inline]
    pub fn split(&mut self, idx: u32) -> u32 {
        let (off, base) = {
            let p = self.get(idx);
            let buffer = &p.buffer;
            let mut off = buffer.len();
            if off > 0 && buffer[off - 1][0] == TERM_ERR {
                off -= 1;
            }
            while off > 0 && buffer[off - 1][2] > p.reduce_pos {
                off -= 1;
            }
            (off, p.buffer_base + off)
        };
        let mut parent = idx;
        while parent != NO_PARENT && base == self.get(parent).buffer_base {
            parent = self.get(parent).parent;
        }
        let n = self.alloc_slot();
        let (a, b) = if (idx as usize) < (n as usize) {
            let (l, r) = self.v.split_at_mut(n as usize);
            (&l[idx as usize], &mut r[0])
        } else {
            let (l, r) = self.v.split_at_mut(idx as usize);
            (&r[0], &mut l[n as usize])
        };
        b.stack.extend_from_slice(&a.stack);
        b.buffer.extend_from_slice(&a.buffer[off..]);
        b.state = a.state;
        b.reduce_pos = a.reduce_pos;
        b.pos = a.pos;
        b.score = a.score;
        b.buffer_base = base;
        b.parent = parent;
        b.ctx = a.ctx;
        b.ctx_flags = a.ctx_flags;
        b.ctx_hash = a.ctx_hash;
        b.look_ahead = a.look_ahead;
        n
    }

    pub fn gc(&mut self, live: &[u32]) {
        let n = self.v.len();
        self.mark.clear();
        self.mark.resize(n, false);
        for &l in live {
            let mut c = l;
            while c != NO_PARENT && !self.mark[c as usize] {
                self.mark[c as usize] = true;
                c = self.v[c as usize].parent;
            }
        }
        self.free.clear();
        for i in (0..n).rev() {
            if !self.mark[i] {
                self.free.push(i as u32);
            }
        }
        self.since_gc = 0;
    }

    /// Reconstructs the linear record buffer for `idx` (zero-copy when unsplit!).
    pub fn flatten_into<'a>(&'a self, idx: u32, scratch: &'a mut Vec<Rec>) -> &'a [Rec] {
        let top = self.get(idx);
        if top.parent == NO_PARENT {
            return &top.buffer;
        }
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = idx;
        while cur != NO_PARENT {
            chain.push(cur);
            cur = self.get(cur).parent;
        }
        let total = top.buffer_base + top.buffer.len();
        scratch.clear();
        scratch.reserve(total);
        for k in (0..chain.len()).rev() {
            let l = self.get(chain[k]);
            let upto = if k == 0 {
                l.buffer.len()
            } else {
                self.get(chain[k - 1]).buffer_base - l.buffer_base
            };
            scratch.extend_from_slice(&l.buffer[..upto]);
        }
        scratch
    }
}
