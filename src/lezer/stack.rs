//! Arena-backed parse stacks (`Stack` in @lezer/lr) using compact `Frame`
//! structs (1 push per shift instead of 3) and `[i32; 4]` record buffers
//! (1 push per node instead of 4-element slice copy).

use super::parse::Env;
use super::stream::Ch;
use super::*;

pub(crate) const NO_PARENT: u32 = u32::MAX;
pub(crate) type Rec = [i32; 4];

#[derive(Clone, Copy)]
pub(crate) struct Frame {
    pub state: u32,
    pub start: i32,
    /// Cumulative record count (`base + buf.len()`) when this frame was pushed.
    pub buf_pos: u32,
}

pub(crate) struct BufLink {
    pub buf: Vec<Rec>,
    /// Cumulative record offset of `buf[0]`.
    pub base: usize,
    pub parent: u32,
}

#[derive(Default)]
pub(crate) struct BufPool {
    pub links: Vec<BufLink>,
}

impl BufPool {
    pub fn reset(&mut self) {
        if self.links.is_empty() {
            self.links.push(BufLink {
                buf: Vec::with_capacity(1024),
                base: 0,
                parent: NO_PARENT,
            });
        } else {
            self.links.truncate(1);
            let first = &mut self.links[0];
            first.buf.clear();
            if first.buf.capacity() > 16384 {
                first.buf.shrink_to(4096);
            }
            first.base = 0;
            first.parent = NO_PARENT;
        }
    }

    pub fn release(&mut self) {
        self.links.clear();
        self.links.shrink_to_fit();
    }

    #[inline(always)]
    pub fn link(&self, id: u32) -> &BufLink {
        unsafe { self.links.get_unchecked(id as usize) }
    }

    #[inline(always)]
    pub fn link_mut(&mut self, id: u32) -> &mut BufLink {
        unsafe { self.links.get_unchecked_mut(id as usize) }
    }

    pub fn alloc(&mut self, buf: Vec<Rec>, base: usize, parent: u32) -> u32 {
        let id = self.links.len() as u32;
        self.links.push(BufLink { buf, base, parent });
        id
    }

    /// Reconstructs the linear record buffer for `buf_id` (zero-copy when unsplit!).
    pub fn flatten_into<'a>(&'a self, buf_id: u32, scratch: &'a mut Vec<Rec>) -> &'a [Rec] {
        let top = self.link(buf_id);
        if top.parent == NO_PARENT {
            return &top.buf;
        }
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = buf_id;
        while cur != NO_PARENT {
            chain.push(cur);
            cur = self.link(cur).parent;
        }
        let total = top.base + top.buf.len();
        scratch.clear();
        scratch.reserve(total);
        for k in (0..chain.len()).rev() {
            let l = self.link(chain[k]);
            let upto = if k == 0 {
                l.buf.len()
            } else {
                self.link(chain[k - 1]).base - l.base
            };
            scratch.extend_from_slice(&l.buf[..upto]);
        }
        scratch
    }
}

pub(crate) struct Stack {
    pub stack: Vec<Frame>,
    pub state: u32,
    pub reduce_pos: i32,
    pub pos: i32,
    pub score: i32,
    pub buf_id: u32,
    pub ctx: u32,
    pub ctx_flags: i32,
    pub ctx_hash: i64,
    pub look_ahead: i32,
}

impl Stack {
    pub fn start(state: u32, pos: i32, ctx_hash: i64, reusable_stack: Vec<Frame>) -> Stack {
        let mut stack = reusable_stack;
        stack.clear();
        if stack.capacity() < 32 {
            stack.reserve(32);
        }
        Stack {
            stack,
            state,
            reduce_pos: pos,
            pos,
            score: 0,
            buf_id: 0,
            ctx: 0,
            ctx_flags: 0,
            ctx_hash,
            look_ahead: 0,
        }
    }

    /// Output buffer length in `i32` units (matching @lezer's `stack.buffer.length`).
    #[inline(always)]
    pub fn buf_len_i32(&self, pool: &BufPool) -> usize {
        pool.link(self.buf_id).buf.len() << 2
    }

    #[inline(always)]
    pub fn push_state(&mut self, pool: &BufPool, state: u32, start: i32) {
        let link = pool.link(self.buf_id);
        let bl = (link.base + link.buf.len()) as u32;
        self.stack.push(Frame { state: self.state, start, buf_pos: bl });
        self.state = state;
    }

    #[inline]
    pub fn reduce<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>, action: u32) {
        let depth = action >> REDUCE_DEPTH_SHIFT;
        let ty = action & VALUE_MASK;
        let g = env.g;
        let lookahead_record =
            self.reduce_pos < self.pos - LOOKAHEAD_MARGIN && self.set_look_ahead(env.bufs, self.pos);
        let dprec = g.dynamic_precedence(ty);
        if dprec != 0 {
            self.score += dprec;
        }
        let min_repeat = g.min_repeat_term as u32;
        if depth == 0 {
            if ty < min_repeat && self.reduce_pos < self.pos {
                self.reduce_pos = self.pos;
            }
            let goto = g.get_goto_loose(env.gt, self.state, ty);
            self.push_state(env.bufs, goto, self.reduce_pos);
            if ty < min_repeat {
                let sz = if lookahead_record { 8 } else { 4 };
                if self.pos == self.reduce_pos {
                    env.bufs.link_mut(self.buf_id).buf.push([ty as i32, self.reduce_pos, self.reduce_pos, sz]);
                } else {
                    self.store_node(env.bufs, ty as i32, self.reduce_pos, self.reduce_pos, sz, true);
                }
            }
            if IS_PY {
                self.reduce_context_py(env, ty, self.reduce_pos);
            }
            return;
        }
        let base = self.stack.len()
            - (depth as usize - 1)
            - if action & STAY_FLAG != 0 { 2 } else { 0 };
        let (start, buffer_base) = if base > 0 {
            let f = unsafe { *self.stack.get_unchecked(base - 1) };
            (f.start, f.buf_pos as usize)
        } else {
            (0, 0)
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
        let link = env.bufs.link(self.buf_id);
        let count_recs = (link.base + link.buf.len()) as i64 - buffer_base as i64;
        if ty < min_repeat || (action & REPEAT_FLAG) != 0 {
            let pos = if g.state_flag(self.state, STATE_SKIPPED) {
                self.pos
            } else {
                self.reduce_pos
            };
            let byte_sz = ((count_recs + 1) << 2) as i32;
            if self.pos == pos {
                env.bufs.link_mut(self.buf_id).buf.push([ty as i32, start, pos, byte_sz]);
            } else {
                self.store_node(env.bufs, ty as i32, start, pos, byte_sz, true);
            }
        }
        if action & STAY_FLAG != 0 {
            self.state = unsafe { self.stack.get_unchecked(base).state };
        } else {
            let base_state = unsafe { self.stack.get_unchecked(base - 1).state };
            self.state = g.get_goto_loose(env.gt, base_state, ty);
        }
        self.stack.truncate(base);
        if IS_PY {
            self.reduce_context_py(env, ty, start);
        }
    }

    #[inline]
    pub fn store_node(
        &mut self,
        pool: &mut BufPool,
        term: i32,
        start: i32,
        end: i32,
        mut size: i32,
        must_sink: bool,
    ) {
        let link = pool.link_mut(self.buf_id);
        let bb = link.base;
        let buffer = &mut link.buf;
        if term == TERM_ERR
            && (self.stack.is_empty()
                || (self.stack.last().unwrap().buf_pos as usize) < buffer.len() + bb)
            && let Some(last) = buffer.last_mut()
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
            buffer.push([term, start, end, size]);
            return;
        }
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
    pub fn shift<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>, action: u32, ty: i32, start: i32, end: i32) {
        let g = env.g;
        if action & GOTO_FLAG != 0 {
            self.push_state(env.bufs, action & VALUE_MASK, self.pos);
        } else if action & STAY_FLAG == 0 {
            let next_state = action;
            self.pos = end;
            let skipped = g.state_flag(next_state, STATE_SKIPPED);
            if !skipped && (end > start || ty <= g.max_node as i32) {
                self.reduce_pos = end;
            }
            self.push_state(env.bufs, next_state, if skipped { start } else { start.min(self.reduce_pos) });
            self.shift_context::<IS_PY, C>(env, ty as u32, start);
            if ty <= g.max_node as i32 {
                env.bufs.link_mut(self.buf_id).buf.push([ty, start, end, 4]);
            }
        } else {
            self.pos = end;
            self.shift_context::<IS_PY, C>(env, ty as u32, start);
            if ty <= g.max_node as i32 {
                env.bufs.link_mut(self.buf_id).buf.push([ty, start, end, 4]);
            }
        }
    }

    #[inline(always)]
    pub fn apply<const IS_PY: bool, C: Ch>(
        &mut self,
        env: &mut Env<C>,
        action: u32,
        next: i32,
        next_start: i32,
        next_end: i32,
    ) {
        if action & REDUCE_FLAG != 0 {
            self.reduce::<IS_PY, C>(env, action);
        } else {
            self.shift::<IS_PY, C>(env, action, next, next_start, next_end);
        }
    }

    pub fn split(&self, pool: &mut BufPool) -> Stack {
        let (new_buf, base) = {
            let link = pool.link(self.buf_id);
            let buffer = &link.buf;
            let mut off = buffer.len();
            if off > 0 && buffer[off - 1][0] == TERM_ERR {
                off -= 1;
            }
            while off > 0 && buffer[off - 1][2] > self.reduce_pos {
                off -= 1;
            }
            (buffer[off..].to_vec(), link.base + off)
        };
        let mut parent = self.buf_id;
        while parent != NO_PARENT && base == pool.link(parent).base {
            parent = pool.link(parent).parent;
        }
        let new_id = pool.alloc(new_buf, base, parent);
        Stack {
            stack: self.stack.clone(),
            state: self.state,
            reduce_pos: self.reduce_pos,
            pos: self.pos,
            score: self.score,
            buf_id: new_id,
            ctx: self.ctx,
            ctx_flags: self.ctx_flags,
            ctx_hash: self.ctx_hash,
            look_ahead: self.look_ahead,
        }
    }

    pub fn recover_by_delete(&mut self, pool: &mut BufPool, g: &Grammar, next: i32, next_end: i32) {
        let is_node = next <= g.max_node as i32;
        if is_node {
            self.store_node(pool, next, self.pos, next_end, 4, false);
        }
        self.store_node(pool, TERM_ERR, self.pos, next_end, if is_node { 8 } else { 4 }, false);
        self.pos = next_end;
        self.reduce_pos = next_end;
        self.score -= 190;
    }

    pub fn can_shift(&self, g: &Grammar, term: i32) -> bool {
        let mut state = self.state;
        let mut copy: Option<Vec<u32>> = None;
        let mut base = self.stack.len() as i64;
        loop {
            let dr = g.state_slot(state, 4);
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

    pub fn recover_by_insert<const IS_PY: bool, C: Ch>(&self, env: &mut Env<C>, next: i32) -> Vec<Stack> {
        let g = env.g;
        if self.stack.len() >= 100 {
            return Vec::new();
        }
        let mut next_states = g.next_states(self.state);
        if next_states.len() > 8 || self.stack.len() >= 40 {
            let mut best: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < next_states.len() {
                let s = next_states[i + 1];
                if s != self.state && g.has_action(s, next) != 0 {
                    best.push(next_states[i]);
                    best.push(s);
                }
                i += 2;
            }
            if self.stack.len() < 40 {
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
            if s == self.state {
                i += 2;
                continue;
            }
            let mut stack = self.split(env.bufs);
            stack.push_state(env.bufs, s, self.pos);
            stack.store_node(env.bufs, TERM_ERR, stack.pos, stack.pos, 4, true);
            stack.shift_context::<IS_PY, C>(env, next_states[i], self.pos);
            stack.reduce_pos = self.pos;
            stack.score -= 200;
            result.push(stack);
            i += 2;
        }
        result
    }

    pub fn force_reduce<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>) -> bool {
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
            self.store_node(env.bufs, TERM_ERR, self.pos, self.pos, 4, true);
            self.score -= 100;
        }
        self.reduce_pos = self.pos;
        self.reduce::<IS_PY, C>(env, reduce);
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

    pub fn force_all<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>) {
        while !env.g.state_flag(self.state, STATE_ACCEPTING) {
            if !self.force_reduce::<IS_PY, C>(env) {
                self.store_node(env.bufs, TERM_ERR, self.pos, self.pos, 4, true);
                break;
            }
        }
    }

    pub fn dead_end(&self, g: &Grammar) -> bool {
        if self.stack.len() != 1 {
            return false;
        }
        g.data[g.state_slot(self.state, 1) as usize] == SEQ_END && g.state_slot(self.state, 4) == 0
    }

    pub fn restart(&mut self, pool: &mut BufPool) {
        self.store_node(pool, TERM_ERR, self.pos, self.pos, 4, true);
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
    pub fn shift_context<const IS_PY: bool, C: Ch>(&mut self, env: &mut Env<C>, term: u32, start: i32) {
        if !IS_PY {
            // JS context tracker: `5 = LineComment`, `6 = BlockComment`, `321 = spaces`, `322 = newline`.
            // JS `ctx_hash` is always 0, so no context records are ever emitted.
            if term != 5 && term != 6 && term != 321 {
                self.ctx = (term == 322) as u32;
            }
            return;
        }
        let cx = env.tracker_shift_py(self.ctx, term, self.pos, start);
        if cx != self.ctx {
            self.update_context_changed_py(env, cx);
        }
    }

    #[inline(always)]
    pub fn reduce_context_py<C: Ch>(&mut self, env: &mut Env<C>, term: u32, _start: i32) {
        if self.ctx_flags != 0 {
            let cx = env.tracker_reduce_py(self.ctx, self.ctx_flags, term);
            if cx != self.ctx {
                self.update_context_changed_py(env, cx);
            }
        }
    }

    fn emit_context(&mut self, pool: &mut BufPool) {
        let last = pool.link(self.buf_id).buf.last().map(|r| r[3]);
        if last != Some(-3) {
            let h = (self.ctx_hash != 0) as i32;
            self.store_node(pool, h, self.reduce_pos, self.reduce_pos, -3, true);
        }
    }

    fn emit_look_ahead(&mut self, pool: &mut BufPool) {
        let b = &mut pool.link_mut(self.buf_id).buf;
        if b.last().map(|r| r[3]) != Some(-4) {
            b.push([self.look_ahead, self.pos, self.pos, -4]);
        }
    }

    #[inline]
    fn update_context_changed_py<C: Ch>(&mut self, env: &mut Env<C>, cx: u32) {
        let (flags, h) = env.py_ctx_info(cx);
        if h != self.ctx_hash {
            self.emit_context(env.bufs);
        }
        self.ctx = cx;
        self.ctx_flags = flags;
        self.ctx_hash = h;
    }

    pub fn set_look_ahead(&mut self, pool: &mut BufPool, look_ahead: i32) -> bool {
        if look_ahead <= self.look_ahead {
            return false;
        }
        self.emit_look_ahead(pool);
        self.look_ahead = look_ahead;
        true
    }

    pub fn close(&mut self, pool: &mut BufPool, strict: bool) {
        if strict {
            self.emit_context(pool);
        }
        if self.look_ahead > 0 {
            self.emit_look_ahead(pool);
        }
    }
}
