//! Group-uniform if-conversion for multi-wave wave64 compute shaders on a 32-lane host (#527,
//! design note #393).
//!
//! Single-wave lockstep (`emit/lockstep.rs`) runs a 64-lane console wave with every thread of the
//! group following the wave's one program counter. In a group of several console waves (e.g. 128
//! or 256 threads = 2 or 4 console waves), each console wave can compute a different wave vote
//! (`s_bcnt1 sN, vcc; s_cbranch_scc0`, `s_and vcc, exec, mask; s_cbranch_vccz`, `s_cbranch_execz`,
//! etc.), so branching on a wave's own vote would split console waves across group barriers and
//! hang the GPU.
//!
//! Here the reducible CFG (already verified by `structure::lockstep` and `lockstep::split_branch`)
//! is decomposed into a loop-respecting topological tree of blocks and natural loops, then lowered
//! with group-uniform control flow:
//! - Every thread in the group executes the same basic blocks in the same order.
//! - Each console wave carries a wave-uniform active predicate (`active[b]: i1`) and per-edge
//!   predicates (`edge_active[(q, b)]: i1`). Instructions receive `active[b] && exec_guard`, so
//!   inactive waves perform no memory/LDS/atomic side effects and keep their old VGPR values.
//! - Non-header PHIs become `select` chains on `edge_active` in a dedicated `phi_block[b]`.
//! - Natural loops run while **any** console wave in the group still wants to continue
//!   (`group_any(continue_h)` through `xchg` + barriers, bounded by `LOOP_LIMIT`), accumulating
//!   exit-edge flags and snapshotting live-out SSA values (LCSSA) when each wave exits.
//! - Because every thread in the group arrives at every block and loop iteration together, all
//!   `wide_ballot`, `wide_read_lane`, `halves_reduce`, `s_barrier`, and `lds_barriers` barriers
//!   are group-uniform.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::{Lower, Stop, Val, Why, unsupported};
use crate::emit::bitcode::{Bin, Term, V, pred};
use crate::gfx10::Fields;
use crate::ir::{EXEC, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Map, Set};

/// State held on [`Lower`] when lowering a multi-wave if-converted shader (`st.predicated`).
pub(super) struct PredState {
    /// Dedicated DXIL block immediately preceding IR block `b`'s body (or `header_dxil` for a
    /// loop header), where non-header select chains or 2-input header phis live.
    pub(super) phi_block: Vec<Option<usize>>,
    /// Whether IR block `b` is the header of a natural loop.
    pub(super) is_header: Vec<bool>,
    /// For each loop header `h`: `(pre_dxil, latch_dxil)` once lowered.
    pub(super) header_edges: Map<usize, (usize, usize)>,
    /// Wave-uniform predicate `i1` telling whether this console wave took IR edge `(from, to)`.
    pub(super) edge_active: Map<(usize, usize), V>,
    /// Current IR block's wave-uniform active predicate while lowering its instructions.
    pub(super) block_active: Option<V>,
    /// Non-header data phis whose args include a missing value or a lane mask, deferred until
    /// actually read so dead phis never emit dead ballots or fail the shader.
    pub(super) lazy_phis: Map<Value, (usize, usize)>,
    /// Cycle guard while resolving lazy non-header phis.
    pub(super) resolving_phis: Set<Value>,
    /// Mask phis created lazily at a loop header `h`, filled when loop `h` closes.
    pub(super) pending_header_masks: Map<usize, Vec<(V, usize)>>,
    /// Spill phis created lazily at a loop header `h`, filled when loop `h` closes.
    pub(super) pending_header_spills: Map<usize, Vec<(V, usize, u32)>>,
    /// Deferred phi retries (same mechanism as `Lower::fill_phis`).
    pub(super) retry: Vec<(Value, Why)>,
}

impl PredState {
    pub(super) fn new(blocks: usize) -> Self {
        Self {
            phi_block: vec![None; blocks],
            is_header: vec![false; blocks],
            header_edges: Map::default(),
            edge_active: Map::default(),
            block_active: None,
            lazy_phis: Map::default(),
            resolving_phis: Set::default(),
            pending_header_masks: Map::default(),
            pending_header_spills: Map::default(),
            retry: Vec::new(),
        }
    }
}

/// Loop-respecting region tree of a reducible CFG.
#[derive(Debug)]
enum Region {
    Block(usize),
    Loop {
        header: usize,
        body: Vec<Region>,
        members: HashSet<usize>,
        forward_preds: Vec<usize>,
        back_edges: Vec<usize>,
        exit_edges: Vec<(usize, usize)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ScopeNode {
    Block(usize),
    Loop(usize),
}

impl Lower<'_> {
    /// Current IR block's wave-uniform active predicate in predicated mode.
    pub(super) fn pred_active(&self) -> Option<V> {
        self.pred.as_ref().and_then(|p| p.block_active)
    }

    /// Resolves a deferred non-header phi when first read by `Lower::get`.
    pub(super) fn pred_resolve_lazy_phi(&mut self, v: Value) -> Option<Result<Val, Why>> {
        let (b, i) = self.pred.as_ref()?.lazy_phis.get(&v).copied()?;
        if !self.pred.as_mut().expect("pred").resolving_phis.insert(v) {
            return Some(Err("a cyclic non-header phi"));
        }
        let phi_blk = self.pred.as_ref().expect("pred").phi_block[b].expect("phi block");
        let res = self.eval_non_header_phi(b, i, phi_blk, true);
        let pred = self.pred.as_mut().expect("pred");
        pred.resolving_phis.remove(&v);
        pred.lazy_phis.remove(&v);
        match res {
            Ok(val) => {
                self.vals.insert(v, val);
                Some(Ok(val))
            }
            Err(why) => {
                self.vals.insert(v, Val::Missing(why));
                Some(Err(why))
            }
        }
    }

    /// Lowers `Atom::Phi(id)` in predicated mode: a select chain in `phi_block[b]` for a
    /// non-header block, or a 2-input DXIL phi at `header_dxil` for a loop header.
    pub(super) fn pred_mask_phi(&mut self, id: Value) -> Result<V, Why> {
        if let Some(&v) = self.mask_phis.get(&id) {
            return Ok(v);
        }
        let &(b, i) = self
            .phi_at
            .get(&id)
            .ok_or("a lane-mask join without a phi")?;
        let is_header = self.pred.as_ref().expect("pred").is_header[b];
        let phi_blk = self
            .pred
            .as_ref()
            .expect("pred")
            .phi_block[b]
            .ok_or("an unreachable join")?;
        if is_header {
            let v = self.s.m.phi(phi_blk, self.s.t.i1);
            self.mask_phis.insert(id, v);
            let edges = self.pred.as_ref().expect("pred").header_edges.get(&b).copied();
            if let Some((pre_dxil, latch_dxil)) = edges
                && latch_dxil != usize::MAX
            {
                // Loop already closed (unlikely, since live-outs are snapshotted, but handle
                // defensively by filling immediately).
                self.fill_header_mask_phi(v, b, i, pre_dxil, latch_dxil)?;
            } else {
                self.pred
                    .as_mut()
                    .expect("pred")
                    .pending_header_masks
                    .entry(b)
                    .or_default()
                    .push((v, i));
            }
            return Ok(v);
        }
        let v = self.eval_non_header_mask_phi(b, i, phi_blk)?;
        self.mask_phis.insert(id, v);
        Ok(v)
    }

    /// Lowers a `spill_slot` phi in predicated mode.
    pub(super) fn pred_spill_phi(
        &mut self,
        v: Value,
        b: usize,
        i: usize,
        lane: u32,
    ) -> Result<V, Why> {
        const NOT: Why = "a VGPR lane no SGPR spill fills";
        let is_header = self.pred.as_ref().expect("pred").is_header[b];
        let phi_blk = self
            .pred
            .as_ref()
            .expect("pred")
            .phi_block[b]
            .ok_or(NOT)?;
        if is_header {
            let x = self.s.m.phi(phi_blk, self.s.t.i32);
            self.spill_phis.insert((v, lane), x);
            let edges = self.pred.as_ref().expect("pred").header_edges.get(&b).copied();
            if let Some((pre_dxil, latch_dxil)) = edges
                && latch_dxil != usize::MAX
            {
                self.fill_header_spill_phi(x, b, i, lane, pre_dxil, latch_dxil)?;
            } else {
                self.pred
                    .as_mut()
                    .expect("pred")
                    .pending_header_spills
                    .entry(b)
                    .or_default()
                    .push((x, i, lane));
            }
            return Ok(x);
        }
        let preds: Vec<(usize, Value)> = self.x.p.blocks[b]
            .preds
            .iter()
            .copied()
            .zip(self.x.p.blocks[b].phis[i].args.iter().copied())
            .filter(|&(q, _)| self.block[q].is_some())
            .collect();
        let mut pairs = Vec::with_capacity(preds.len());
        for (q, arg) in preds {
            let y = self.spill_slot(arg, lane, &mut Vec::new(), true)?;
            pairs.push((q, y));
        }
        let def = self.s.m.i32(0);
        let x = self.pred_select_edges_at(phi_blk, b, &pairs, def);
        self.spill_phis.insert((v, lane), x);
        Ok(x)
    }

    /// Selects among incoming `(pred_block, value)` pairs for IR block `b` using `edge_active`,
    /// emitting into `phi_block[b]`. Used for `Source::Phi` / `Source::LanePhi` in predicated mode.
    pub(super) fn pred_select_in_block(
        &mut self,
        b: usize,
        pairs: &[(usize, V)],
        default: V,
    ) -> Result<V, Why> {
        let pred = self.pred.as_ref().expect("pred");
        if pred.is_header[b] {
            return Err("a descriptor chosen by control flow");
        }
        let at = pred.phi_block[b].ok_or("a descriptor chosen by control flow")?;
        Ok(self.pred_select_edges_at(at, b, pairs, default))
    }

    /// Selects among `(pred_block, value)` pairs at DXIL block `at` using `edge_active[(q, b)]`.
    fn pred_select_edges_at(
        &mut self,
        at: usize,
        b: usize,
        pairs: &[(usize, V)],
        default: V,
    ) -> V {
        let mut iter = pairs.iter().copied();
        let Some((_, mut acc)) = iter.next() else {
            return default;
        };
        for (q, val) in iter {
            let cond = self
                .pred
                .as_ref()
                .expect("pred")
                .edge_active
                .get(&(q, b))
                .copied();
            if let Some(c) = cond {
                acc = self.s.m.select(at, c, val, acc);
            }
        }
        acc
    }

    /// Entry point for multi-wave if-converted lowering (replaces `lower_blocks` + `fill_phis`).
    pub(super) fn lower_predicated(
        &mut self,
        order: &[usize],
        entry_block: usize,
    ) -> Result<(), Stop> {
        self.cur = entry_block;
        self.inputs()?;
        if self.x.p.thread_counts {
            self.launched();
        }
        let regions = self.build_regions(order)?;
        self.lower_region_list(&regions, None)?;
        self.s.m.terminate(self.cur, Term::Ret);
        let retry = std::mem::take(&mut self.pred.as_mut().expect("pred").retry);
        if !retry.is_empty() {
            return Err(Stop::Retry(retry));
        }
        Ok(())
    }

    /// Decomposes the reducible CFG into a loop-respecting topological tree of blocks and loops.
    fn build_regions(&mut self, order: &[usize]) -> Result<Vec<Region>, Stop> {
        let p = self.x.p;
        let n = p.blocks.len();
        let reachable = |b: usize| self.rpo[b] != usize::MAX;

        // 1. Collect back edges u -> h (rpo[h] <= rpo[u]) and build each header's natural loop.
        let mut back_sources: HashMap<usize, Vec<usize>> = HashMap::new();
        for &u in order {
            for v in p.blocks[u].exit.successors() {
                if reachable(v) && self.rpo[v] <= self.rpo[u] {
                    back_sources.entry(v).or_default().push(u);
                }
            }
        }

        let mut loop_members: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (&h, sources) in &back_sources {
            let mut members = HashSet::new();
            members.insert(h);
            let mut stack = Vec::new();
            for &u in sources {
                if members.insert(u) {
                    stack.push(u);
                }
            }
            while let Some(x) = stack.pop() {
                for &q in &p.blocks[x].preds {
                    if reachable(q) && members.insert(q) {
                        stack.push(q);
                    }
                }
            }
            loop_members.insert(h, members);
            self.pred.as_mut().expect("pred").is_header[h] = true;
        }

        // 2. For each reachable block, find its innermost enclosing loop (`owner[b]`);
        //    for each loop header `h`, find its immediate parent loop (`parent_loop[h]`).
        let mut owner: Vec<Option<usize>> = vec![None; n];
        for &b in order {
            owner[b] = loop_members
                .iter()
                .filter(|(_, m)| m.contains(&b))
                .min_by_key(|&(&h, m)| (m.len(), self.rpo[h]))
                .map(|(&h, _)| h);
        }
        let mut parent_loop: HashMap<usize, Option<usize>> = HashMap::new();
        for &h in loop_members.keys() {
            let parent = loop_members
                .iter()
                .filter(|&(&other, m)| other != h && m.contains(&h))
                .min_by_key(|&(&other, m)| (m.len(), self.rpo[other]))
                .map(|(&other, _)| other);
            parent_loop.insert(h, parent);
        }

        // 3. Recursively topologically sort each scope (top-level or natural loop).
        fn scope_child(
            b: usize,
            scope: Option<usize>,
            owner: &[Option<usize>],
            parent_loop: &HashMap<usize, Option<usize>>,
        ) -> Option<ScopeNode> {
            let mut cur = owner[b];
            if cur == scope {
                return Some(ScopeNode::Block(b));
            }
            while let Some(h) = cur {
                let p = parent_loop[&h];
                if p == scope {
                    return Some(ScopeNode::Loop(h));
                }
                cur = p;
            }
            None
        }

        fn build_scope(
            p: &Program,
            order: &[usize],
            rpo: &[usize],
            scope: Option<usize>,
            loop_members: &HashMap<usize, HashSet<usize>>,
            owner: &[Option<usize>],
            parent_loop: &HashMap<usize, Option<usize>>,
        ) -> Result<Vec<Region>, Stop> {
            let in_scope = |b: usize| -> bool {
                if rpo[b] == usize::MAX {
                    return false;
                }
                match scope {
                    None => true,
                    Some(h) => loop_members[&h].contains(&b),
                }
            };
            let mut nodes: Vec<ScopeNode> = Vec::new();
            for &b in order {
                if owner[b] == scope {
                    nodes.push(ScopeNode::Block(b));
                }
            }
            for (&sub_h, &par) in parent_loop {
                if par == scope {
                    nodes.push(ScopeNode::Loop(sub_h));
                }
            }

            let head_of = |node: ScopeNode| match node {
                ScopeNode::Block(b) | ScopeNode::Loop(b) => b,
            };

            let mut succs: HashMap<ScopeNode, BTreeSet<ScopeNode>> = HashMap::new();
            let mut indeg: HashMap<ScopeNode, usize> = nodes.iter().map(|&nd| (nd, 0)).collect();
            for &u in order {
                if !in_scope(u) {
                    continue;
                }
                let Some(nu) = scope_child(u, scope, owner, parent_loop) else {
                    continue;
                };
                for v in p.blocks[u].exit.successors() {
                    if !in_scope(v) || Some(v) == scope {
                        continue;
                    }
                    let Some(nv) = scope_child(v, scope, owner, parent_loop) else {
                        continue;
                    };
                    if nu != nv && succs.entry(nu).or_default().insert(nv) {
                        *indeg.entry(nv).or_insert(0) += 1;
                    }
                }
            }

            let mut ready: BTreeSet<(usize, ScopeNode)> = indeg
                .iter()
                .filter(|&(_, &d)| d == 0)
                .map(|(&nd, _)| (rpo[head_of(nd)], nd))
                .collect();
            let mut sorted = Vec::with_capacity(nodes.len());
            while let Some((_, nd)) = ready.pop_first() {
                sorted.push(nd);
                if let Some(nexts) = succs.get(&nd) {
                    for &nv in nexts {
                        let d = indeg.get_mut(&nv).expect("node");
                        *d -= 1;
                        if *d == 0 {
                            ready.insert((rpo[head_of(nv)], nv));
                        }
                    }
                }
            }
            if sorted.len() != nodes.len() {
                let offset = scope.map_or(0, |h| p.blocks[h].offset);
                return Err(unsupported(offset, "irreducible control flow"));
            }

            let mut out = Vec::with_capacity(sorted.len());
            for nd in sorted {
                match nd {
                    ScopeNode::Block(b) => out.push(Region::Block(b)),
                    ScopeNode::Loop(sub_h) => {
                        let members = loop_members[&sub_h].clone();
                        let body = build_scope(
                            p,
                            order,
                            rpo,
                            Some(sub_h),
                            loop_members,
                            owner,
                            parent_loop,
                        )?;
                        let forward_preds: Vec<usize> = p.blocks[sub_h]
                            .preds
                            .iter()
                            .copied()
                            .filter(|&q| rpo[q] != usize::MAX && !members.contains(&q))
                            .collect();
                        let back_edges: Vec<usize> = p.blocks[sub_h]
                            .preds
                            .iter()
                            .copied()
                            .filter(|&q| members.contains(&q))
                            .collect();
                        let mut exit_edges = Vec::new();
                        for &u in order {
                            if !members.contains(&u) {
                                continue;
                            }
                            for v in p.blocks[u].exit.successors() {
                                if rpo[v] != usize::MAX && !members.contains(&v) {
                                    exit_edges.push((u, v));
                                }
                            }
                        }
                        exit_edges.sort_unstable();
                        exit_edges.dedup();
                        out.push(Region::Loop {
                            header: sub_h,
                            body,
                            members,
                            forward_preds,
                            back_edges,
                            exit_edges,
                        });
                    }
                }
            }
            Ok(out)
        }

        build_scope(p, order, &self.rpo, None, &loop_members, &owner, &parent_loop)
    }

    fn lower_region_list(
        &mut self,
        regions: &[Region],
        header_active: Option<(usize, V)>,
    ) -> Result<(), Stop> {
        for r in regions {
            match r {
                Region::Block(b) => {
                    let act = header_active.filter(|&(h, _)| h == *b).map(|(_, a)| a);
                    self.lower_pred_block(*b, act)?;
                }
                Region::Loop {
                    header,
                    body,
                    members,
                    forward_preds,
                    back_edges,
                    exit_edges,
                } => {
                    self.lower_pred_loop(
                        *header,
                        body,
                        members,
                        forward_preds,
                        back_edges,
                        exit_edges,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn lower_pred_block(&mut self, b: usize, loop_header_active: Option<V>) -> Result<(), Stop> {
        let p = self.x.p;
        let act_b = if let Some(act) = loop_header_active {
            // `self.cur` is `header_dxil`; jump to a dedicated body block so `header_dxil`
            // holds only the loop-header phis and stays clean.
            let header_dxil = self.cur;
            let body_b = self.s.m.new_block();
            self.s.m.terminate(header_dxil, Term::Br(body_b));
            self.cur = body_b;
            act
        } else {
            let phi_b = self.s.m.new_block();
            let body_b = self.s.m.new_block();
            self.s.m.terminate(self.cur, Term::Br(phi_b));
            self.s.m.terminate(phi_b, Term::Br(body_b));
            self.block[b] = Some(phi_b);
            self.pred.as_mut().expect("pred").phi_block[b] = Some(phi_b);
            let act = if b == 0 {
                self.s.m.bool(true)
            } else {
                let preds: Vec<usize> = p.blocks[b]
                    .preds
                    .iter()
                    .copied()
                    .filter(|&q| self.block[q].is_some())
                    .collect();
                self.or_edge_active(phi_b, &preds, b)
            };
            self.materialize_non_header_phis(b, phi_b);
            self.cur = body_b;
            act
        };

        self.cur_ir = b;
        self.pred.as_mut().expect("pred").block_active = Some(act_b);
        self.mode16 = self.round16[b];
        let mut i = 0;
        while i < p.blocks[b].insts.len() {
            if self.split_barriers.contains(&(b, i)) {
                self.group_barrier();
            }
            self.inst(b, i)?;
            self.mode16 = super::round16_after(&p.blocks[b].insts[i], self.mode16);
            i += 1;
        }
        if self.split_barriers.contains(&(b, p.blocks[b].insts.len())) {
            self.group_barrier();
        }
        self.pred.as_mut().expect("pred").block_active = None;
        self.mask_ok = self.mask_data == super::MaskData::Console;
        self.tail[b] = Some(self.cur);

        // Compute outgoing per-wave edge predicates without branching.
        let offset = p.blocks[b]
            .insts
            .last()
            .map_or(p.blocks[b].offset, |inst| inst.machine.offset);
        match self.x.st.exits[b] {
            LaneExit::End => {}
            LaneExit::Jump(t) => {
                self.pred
                    .as_mut()
                    .expect("pred")
                    .edge_active
                    .insert((b, t), act_b);
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                if taken == not_taken {
                    self.pred
                        .as_mut()
                        .expect("pred")
                        .edge_active
                        .insert((b, taken), act_b);
                } else {
                    let cond = self
                        .boolean(when, self.cur)
                        .map_err(|w| unsupported(offset, w))?;
                    let c = self.cur;
                    let t_act = self.s.m.bin(c, Bin::And, act_b, cond);
                    let one = self.s.m.bool(true);
                    let not_cond = self.s.m.bin(c, Bin::Xor, cond, one);
                    let f_act = self.s.m.bin(c, Bin::And, act_b, not_cond);
                    let edges = &mut self.pred.as_mut().expect("pred").edge_active;
                    edges.insert((b, taken), t_act);
                    edges.insert((b, not_taken), f_act);
                }
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        }
        Ok(())
    }

    fn or_edge_active(&mut self, at: usize, preds: &[usize], to: usize) -> V {
        let mut acc: Option<V> = None;
        for &q in preds {
            let Some(&e) = self.pred.as_ref().expect("pred").edge_active.get(&(q, to)) else {
                continue;
            };
            acc = Some(match acc {
                None => e,
                Some(prev) => self.s.m.bin(at, Bin::Or, prev, e),
            });
        }
        acc.unwrap_or_else(|| self.s.m.bool(false))
    }

    fn materialize_non_header_phis(&mut self, b: usize, phi_b: usize) {
        let p = self.x.p;
        for (i, phi) in p.blocks[b].phis.iter().enumerate() {
            if matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                self.vals.insert(phi.value, Val::Missing("a lane mask"));
                continue;
            }
            if let Some(&why) = self.bad.get(&phi.value) {
                self.vals.insert(phi.value, Val::Missing(why));
                continue;
            }
            // Fast path: every reachable incoming arg already has a matching DXIL value.
            let bit = phi.reg == Reg::Scc;
            let mut ready = Vec::new();
            let mut all_ready = true;
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.block[q].is_none() {
                    continue;
                }
                match (self.vals.get(&phi.args[n]).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => ready.push((q, x)),
                    _ => {
                        all_ready = false;
                        break;
                    }
                }
            }
            if all_ready && !ready.is_empty() {
                let def = if bit {
                    self.s.m.bool(false)
                } else {
                    self.s.m.i32(0)
                };
                let x = self.pred_select_edges_at(phi_b, b, &ready, def);
                self.vals
                    .insert(phi.value, if bit { Val::Bit(x) } else { Val::I32(x) });
            } else {
                // Defer until first read so dead phis with missing/mask args never fail or emit
                // unused ballots.
                self.pred
                    .as_mut()
                    .expect("pred")
                    .lazy_phis
                    .insert(phi.value, (b, i));
            }
        }
    }

    fn eval_non_header_phi(
        &mut self,
        b: usize,
        i: usize,
        phi_b: usize,
        read: bool,
    ) -> Result<Val, Why> {
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let bit = phi.reg == Reg::Scc;
        let mut pairs = Vec::new();
        for (n, &q) in p.blocks[b].preds.iter().enumerate() {
            if self.block[q].is_none() {
                continue;
            }
            let arg = phi.args[n];
            // Resolve chained lazy phis on demand.
            if self.pred.as_ref().expect("pred").lazy_phis.contains_key(&arg) {
                let _ = self.pred_resolve_lazy_phi(arg);
            }
            let x = match (self.vals.get(&arg).copied(), bit) {
                (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                _ if read && (self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg)) => {
                    if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    }
                }
                (Some(Val::Bit(_)), false) if read => return Err("an SCC joined with SGPR values"),
                _ if read
                    && !bit
                    && self.mask_data == super::MaskData::Console
                    && (self.x.st.masks.contains_key(&arg)
                        || self.x.st.highs.contains_key(&arg)) =>
                {
                    self.mask_dword(arg)?
                }
                (Some(Val::Missing(_)), false)
                    if read && let Some(x) = self.copied_number(phi.value) =>
                {
                    x
                }
                (Some(Val::Missing(why)), _) if read => return Err(why),
                _ if read => return Err("a join of mismatched values"),
                _ if bit => self.s.m.bool(false),
                _ => self.s.m.i32(0),
            };
            pairs.push((q, x));
        }
        let def = if bit {
            self.s.m.bool(false)
        } else {
            self.s.m.i32(0)
        };
        let x = self.pred_select_edges_at(phi_b, b, &pairs, def);
        Ok(if bit { Val::Bit(x) } else { Val::I32(x) })
    }

    fn lower_pred_loop(
        &mut self,
        header: usize,
        body: &[Region],
        members: &HashSet<usize>,
        forward_preds: &[usize],
        back_edges: &[usize],
        exit_edges: &[(usize, usize)],
    ) -> Result<(), Stop> {
        let p = self.x.p;
        let pre_dxil = self.cur;
        let enter_h = if forward_preds.is_empty() {
            self.s.m.bool(true)
        } else {
            self.or_edge_active(pre_dxil, forward_preds, header)
        };

        // Collect all values defined inside `members` that are referenced outside `members`, so
        // we snapshot them when each wave exits the loop (LCSSA).
        let (live_out_vals, live_out_mask_data, live_out_mask_phis) =
            self.loop_live_outs(members);

        let header_dxil = self.s.m.new_block();
        let join_dxil = self.s.m.new_block();
        let skip_dxil = if !forward_preds.is_empty() {
            let any_enter = self.group_any(pre_dxil, enter_h);
            let skip = self.s.m.new_block();
            self.s.m.terminate(pre_dxil, Term::CondBr(any_enter, header_dxil, skip));
            self.s.m.terminate(skip, Term::Br(join_dxil));
            Some(skip)
        } else {
            self.s.m.terminate(pre_dxil, Term::Br(header_dxil));
            None
        };

        // Header DXIL phis.
        let active_phi = self.s.m.phi(header_dxil, self.s.t.i1);
        let iter_phi = self.s.m.phi(header_dxil, self.s.t.i32);
        self.block[header] = Some(header_dxil);
        self.pred.as_mut().expect("pred").phi_block[header] = Some(header_dxil);
        self.pred
            .as_mut()
            .expect("pred")
            .header_edges
            .insert(header, (pre_dxil, usize::MAX));

        let mut header_data_phis: Vec<(V, usize)> = Vec::new();
        for (i, phi) in p.blocks[header].phis.iter().enumerate() {
            let val = match phi.reg {
                Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                Reg::Scc => {
                    let v = self.s.m.phi(header_dxil, self.s.t.i1);
                    header_data_phis.push((v, i));
                    Val::Bit(v)
                }
                _ => {
                    let v = self.s.m.phi(header_dxil, self.s.t.i32);
                    header_data_phis.push((v, i));
                    Val::I32(v)
                }
            };
            self.vals.insert(phi.value, val);
        }

        // Lower loop body (starts with `Region::Block(header)`).
        self.cur = header_dxil;
        self.lower_region_list(body, Some((header, active_phi)))?;

        // Materialize any mask phis defined inside this loop that are only referenced outside
        // it, so they are filled and snapshotted before the loop closes.
        for id in &live_out_mask_phis {
            let offset = self
                .phi_at
                .get(id)
                .map_or(p.blocks[header].offset, |&(b, _)| p.blocks[b].offset);
            self.pred_mask_phi(*id)
                .map_err(|w| unsupported(offset, w))?;
        }

        let latch_dxil = self.cur;
        self.pred
            .as_mut()
            .expect("pred")
            .header_edges
            .insert(header, (pre_dxil, latch_dxil));

        let continue_h = self.or_edge_active(latch_dxil, back_edges, header);
        self.s
            .m
            .set_phi(active_phi, vec![(enter_h, pre_dxil), (continue_h, latch_dxil)]);

        // Fill header data phis.
        let live_out_set: HashSet<Value> = live_out_vals.iter().copied().collect();
        for (v_phi, i) in header_data_phis {
            let phi = &p.blocks[header].phis[i];
            let bit = phi.reg == Reg::Scc;
            let read = self.read_phis.contains(&phi.value) || live_out_set.contains(&phi.value);
            let init_res = self.eval_phi_subset(header, i, forward_preds, pre_dxil, read);
            let back_res = self.eval_phi_subset(header, i, back_edges, latch_dxil, read);
            let (init_v, back_v) = match (init_res, back_res) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(why), _) | (_, Err(why)) if read => {
                    self.pred
                        .as_mut()
                        .expect("pred")
                        .retry
                        .push((phi.value, why));
                    let z = if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    };
                    (z, z)
                }
                _ => {
                    let z = if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    };
                    (z, z)
                }
            };
            self.s
                .m
                .set_phi(v_phi, vec![(init_v, pre_dxil), (back_v, latch_dxil)]);
        }

        // Fill header mask phis and spill phis created while lowering the body.
        let header_masks = self
            .pred
            .as_mut()
            .expect("pred")
            .pending_header_masks
            .remove(&header)
            .unwrap_or_default();
        for (v_phi, i) in header_masks {
            let offset = p.blocks[header].offset;
            self.fill_header_mask_phi(v_phi, header, i, pre_dxil, latch_dxil)
                .map_err(|w| unsupported(offset, w))?;
        }
        let header_spills = self
            .pred
            .as_mut()
            .expect("pred")
            .pending_header_spills
            .remove(&header)
            .unwrap_or_default();
        for (v_phi, i, lane) in header_spills {
            let offset = p.blocks[header].offset;
            self.fill_header_spill_phi(v_phi, header, i, lane, pre_dxil, latch_dxil)
                .map_err(|w| unsupported(offset, w))?;
        }

        // Accumulate exit edges across iterations.
        let mut exiting_l: Option<V> = None;
        for &e in exit_edges {
            let Some(&ea) = self.pred.as_ref().expect("pred").edge_active.get(&e) else {
                continue;
            };
            exiting_l = Some(match exiting_l {
                None => ea,
                Some(prev) => self.s.m.bin(latch_dxil, Bin::Or, prev, ea),
            });
        }
        let exiting_l = exiting_l.unwrap_or_else(|| self.s.m.bool(false));

        let mut exit_accum: Vec<((usize, usize), V)> = Vec::with_capacity(exit_edges.len());
        for &e in exit_edges {
            let ex_phi = self.s.m.phi(header_dxil, self.s.t.i1);
            let e_now = self
                .pred
                .as_ref()
                .expect("pred")
                .edge_active
                .get(&e)
                .copied()
                .unwrap_or_else(|| self.s.m.bool(false));
            let ex_next = self.s.m.bin(latch_dxil, Bin::Or, ex_phi, e_now);
            let f = self.s.m.bool(false);
            self.s
                .m
                .set_phi(ex_phi, vec![(f, pre_dxil), (ex_next, latch_dxil)]);
            exit_accum.push((e, ex_next));
        }

        // Materialize ballots of any masks defined inside this loop that are read as data outside
        // it, before snapshotting live-out values.
        for lo in &live_out_mask_data {
            if !self.ballots.contains_key(lo) {
                let offset = p.blocks[header].offset;
                self.cur = latch_dxil;
                self.cur_ir = header;
                let was = std::mem::replace(&mut self.mask_ok, true);
                let r = self.mask_dword(*lo);
                self.mask_ok = was;
                // Only fail if the ballot is genuinely invalid; a dead mask not usable as data
                // stays unmaterialized unless actually read later.
                let _ = (r, offset);
            }
        }

        // Snapshot all live-out values (vals, lane_bits, mask_phis, ballots) on `exiting_l`.
        let mut saved_vals: Vec<(Value, bool, V)> = Vec::new();
        let mut saved_lanes: Vec<(Value, V)> = Vec::new();
        let mut saved_mask_phis: Vec<(Value, V)> = Vec::new();
        let mut saved_ballots: Vec<(Value, V, V)> = Vec::new();

        for &v in &live_out_vals {
            if self.pred.as_ref().expect("pred").lazy_phis.contains_key(&v) {
                let _ = self.pred_resolve_lazy_phi(v);
            }
            if let Some(val) = self.vals.get(&v).copied() {
                match val {
                    Val::I32(cur_v) => {
                        let phi = self.s.m.phi(header_dxil, self.s.t.i32);
                        let next = self.s.m.select(latch_dxil, exiting_l, cur_v, phi);
                        let z = self.s.m.i32(0);
                        self.s
                            .m
                            .set_phi(phi, vec![(z, pre_dxil), (next, latch_dxil)]);
                        saved_vals.push((v, false, next));
                    }
                    Val::Bit(cur_v) => {
                        let phi = self.s.m.phi(header_dxil, self.s.t.i1);
                        let next = self.s.m.select(latch_dxil, exiting_l, cur_v, phi);
                        let f = self.s.m.bool(false);
                        self.s
                            .m
                            .set_phi(phi, vec![(f, pre_dxil), (next, latch_dxil)]);
                        saved_vals.push((v, true, next));
                    }
                    Val::Missing(_) => {}
                }
            }
            if let Some(&cur_b) = self.lane_bits.get(&v) {
                let phi = self.s.m.phi(header_dxil, self.s.t.i1);
                let next = self.s.m.select(latch_dxil, exiting_l, cur_b, phi);
                let f = self.s.m.bool(false);
                self.s
                    .m
                    .set_phi(phi, vec![(f, pre_dxil), (next, latch_dxil)]);
                saved_lanes.push((v, next));
            }
            if let Some(&cur_m) = self.mask_phis.get(&v) {
                let phi = self.s.m.phi(header_dxil, self.s.t.i1);
                let next = self.s.m.select(latch_dxil, exiting_l, cur_m, phi);
                let f = self.s.m.bool(false);
                self.s
                    .m
                    .set_phi(phi, vec![(f, pre_dxil), (next, latch_dxil)]);
                saved_mask_phis.push((v, next));
            }
            if let Some(&(bx, by)) = self.ballots.get(&v) {
                let px = self.s.m.phi(header_dxil, self.s.t.i32);
                let py = self.s.m.phi(header_dxil, self.s.t.i32);
                let nx = self.s.m.select(latch_dxil, exiting_l, bx, px);
                let ny = self.s.m.select(latch_dxil, exiting_l, by, py);
                let z = self.s.m.i32(0);
                self.s
                    .m
                    .set_phi(px, vec![(z, pre_dxil), (nx, latch_dxil)]);
                self.s
                    .m
                    .set_phi(py, vec![(z, pre_dxil), (ny, latch_dxil)]);
                saved_ballots.push((v, nx, ny));
            }
        }

        // Group-uniform loop continuation check.
        let any_continue = self.group_any(latch_dxil, continue_h);
        let one = self.s.m.i32(1);
        let iter_next = self.s.m.bin(latch_dxil, Bin::Add, iter_phi, one);
        let limit = self.s.m.i32(super::super::loop_limit());
        let ok = self.s.m.cmp(latch_dxil, pred::IULT, iter_next, limit);
        let go = self.s.m.bin(latch_dxil, Bin::And, any_continue, ok);
        let zero = self.s.m.i32(0);
        self.s
            .m
            .set_phi(iter_phi, vec![(zero, pre_dxil), (iter_next, latch_dxil)]);
        self.s
            .m
            .terminate(latch_dxil, Term::CondBr(go, header_dxil, join_dxil));

        // Join block after the loop: merge with `skip_dxil` if the loop could be skipped.
        let merge_i1 = |l: &mut Self, from_latch: V| -> V {
            match skip_dxil {
                Some(skip) => {
                    let phi = l.s.m.phi(join_dxil, l.s.t.i1);
                    let f = l.s.m.bool(false);
                    l.s.m.set_phi(phi, vec![(from_latch, latch_dxil), (f, skip)]);
                    phi
                }
                None => from_latch,
            }
        };
        let merge_i32 = |l: &mut Self, from_latch: V| -> V {
            match skip_dxil {
                Some(skip) => {
                    let phi = l.s.m.phi(join_dxil, l.s.t.i32);
                    let z = l.s.m.i32(0);
                    l.s.m.set_phi(phi, vec![(from_latch, latch_dxil), (z, skip)]);
                    phi
                }
                None => from_latch,
            }
        };

        for (e, ex_next) in exit_accum {
            let merged = merge_i1(self, ex_next);
            self.pred
                .as_mut()
                .expect("pred")
                .edge_active
                .insert(e, merged);
        }
        for (v, is_bit, next) in saved_vals {
            let merged = if is_bit {
                Val::Bit(merge_i1(self, next))
            } else {
                Val::I32(merge_i32(self, next))
            };
            self.vals.insert(v, merged);
            self.floats.remove(&v);
        }
        for (v, next) in saved_lanes {
            let merged = merge_i1(self, next);
            self.lane_bits.insert(v, merged);
        }
        for (v, next) in saved_mask_phis {
            let merged = merge_i1(self, next);
            self.mask_phis.insert(v, merged);
        }
        for (v, nx, ny) in saved_ballots {
            let mx = merge_i32(self, nx);
            let my = merge_i32(self, ny);
            self.ballots.insert(v, (mx, my));
        }

        self.cur = join_dxil;
        Ok(())
    }

    /// Evaluates a subset of incoming predecessors of `phi` at IR block `b` into a single `V`
    /// emitted at DXIL block `at`.
    fn eval_phi_subset(
        &mut self,
        b: usize,
        i: usize,
        subset: &[usize],
        at: usize,
        read: bool,
    ) -> Result<V, Why> {
        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let bit = phi.reg == Reg::Scc;
        let mut pairs = Vec::new();
        for (n, &q) in p.blocks[b].preds.iter().enumerate() {
            if !subset.contains(&q) || self.block[q].is_none() {
                continue;
            }
            let arg = phi.args[n];
            if self.pred.as_ref().expect("pred").lazy_phis.contains_key(&arg) {
                let _ = self.pred_resolve_lazy_phi(arg);
            }
            let x = match (self.vals.get(&arg).copied(), bit) {
                (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                _ if read && (self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg)) => {
                    if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    }
                }
                (Some(Val::Bit(_)), false) if read => return Err("an SCC joined with SGPR values"),
                _ if read
                    && !bit
                    && self.mask_data == super::MaskData::Console
                    && (self.x.st.masks.contains_key(&arg)
                        || self.x.st.highs.contains_key(&arg)) =>
                {
                    let here = std::mem::replace(&mut self.cur, at);
                    let was = std::mem::replace(&mut self.mask_ok, true);
                    let r = self.mask_dword(arg);
                    self.mask_ok = was;
                    self.cur = here;
                    r?
                }
                (Some(Val::Missing(_)), false)
                    if read && let Some(x) = self.copied_number(phi.value) =>
                {
                    x
                }
                (Some(Val::Missing(why)), _) if read => return Err(why),
                _ if read => return Err("a join of mismatched values"),
                _ if bit => self.s.m.bool(false),
                _ => self.s.m.i32(0),
            };
            pairs.push((q, x));
        }
        let def = if bit {
            self.s.m.bool(false)
        } else {
            self.s.m.i32(0)
        };
        Ok(self.pred_select_edges_at(at, b, &pairs, def))
    }

    fn eval_non_header_mask_phi(&mut self, b: usize, i: usize, phi_b: usize) -> Result<V, Why> {
        let p = self.x.p;
        let preds: Vec<usize> = p.blocks[b]
            .preds
            .iter()
            .copied()
            .filter(|&q| self.block[q].is_some())
            .collect();
        self.eval_mask_phi_subset(b, i, &preds, phi_b)
    }

    fn fill_header_mask_phi(
        &mut self,
        v_phi: V,
        header: usize,
        i: usize,
        pre_dxil: usize,
        latch_dxil: usize,
    ) -> Result<(), Why> {
        let p = self.x.p;
        let fwd: Vec<usize> = p.blocks[header]
            .preds
            .iter()
            .copied()
            .filter(|&q| self.block[q].is_some() && self.rpo[q] < self.rpo[header])
            .collect();
        let back: Vec<usize> = p.blocks[header]
            .preds
            .iter()
            .copied()
            .filter(|&q| self.block[q].is_some() && self.rpo[q] >= self.rpo[header])
            .collect();
        let init = self.eval_mask_phi_subset(header, i, &fwd, pre_dxil)?;
        let next = self.eval_mask_phi_subset(header, i, &back, latch_dxil)?;
        self.s
            .m
            .set_phi(v_phi, vec![(init, pre_dxil), (next, latch_dxil)]);
        Ok(())
    }

    fn eval_mask_phi_subset(
        &mut self,
        b: usize,
        i: usize,
        subset: &[usize],
        at: usize,
    ) -> Result<V, Why> {
        const HIGH: Why =
            "a lane-mask join of an untracked mask: data whose high dword is not known";
        const MISSING: Why =
            "a lane-mask join of an untracked mask: a value the translation could not make";
        const BIT: Why = "a lane-mask join of an untracked mask: an SCC or lane bit";
        const NONE: Why = "a lane-mask join of an untracked mask: no value on that edge";

        let p = self.x.p;
        let phi = &p.blocks[b].phis[i];
        let high = self
            .pair_high(phi.value)
            .and_then(|h| self.phi_at.get(&h))
            .map(|&(hb, hi)| &p.blocks[hb].phis[hi]);
        let mut pairs = Vec::new();
        for (n, &q) in p.blocks[b].preds.iter().enumerate() {
            if !subset.contains(&q) || self.block[q].is_none() {
                continue;
            }
            let arg = phi.args[n];
            if self.pred.as_ref().expect("pred").lazy_phis.contains_key(&arg) {
                let _ = self.pred_resolve_lazy_phi(arg);
            }
            if let Some(h) = high {
                let harg = h.args[n];
                if self.pred.as_ref().expect("pred").lazy_phis.contains_key(&harg) {
                    let _ = self.pred_resolve_lazy_phi(harg);
                }
            }
            let mask = self.x.st.masks.get(&arg).copied();
            let x = match (mask, self.vals.get(&arg).copied()) {
                (Some(m), _) => self.boolean(m, at)?,
                (None, _) if self.unneeded_edges().contains(&(phi.value, n)) => {
                    self.s.m.bool(false)
                }
                (None, Some(Val::I32(word))) => {
                    let hi = if self.x.native.wave64 {
                        match high.and_then(|h| self.vals.get(&h.args[n]).copied()) {
                            Some(Val::I32(hi)) => Some(hi),
                            _ => return Err(HIGH),
                        }
                    } else {
                        None
                    };
                    self.lane_bit(at, word, hi)
                }
                (None, _)
                    if let Some(&m) = self.x.st.other.get(&arg)
                        && !self.x.st.bools.support(m).iter().any(|a| {
                            matches!(a, Atom::Opaque(_) | Atom::Quad(_))
                        }) =>
                {
                    self.boolean(m, at)?
                }
                (None, Some(Val::Missing(_))) => return Err(MISSING),
                (None, Some(Val::Bit(_))) => return Err(BIT),
                (None, None) => return Err(NONE),
            };
            pairs.push((q, x));
        }
        let def = self.s.m.bool(false);
        Ok(self.pred_select_edges_at(at, b, &pairs, def))
    }

    fn fill_header_spill_phi(
        &mut self,
        v_phi: V,
        header: usize,
        i: usize,
        lane: u32,
        pre_dxil: usize,
        latch_dxil: usize,
    ) -> Result<(), Why> {
        let p = self.x.p;
        let mut fwd = Vec::new();
        let mut back = Vec::new();
        for (n, &q) in p.blocks[header].preds.iter().enumerate() {
            if self.block[q].is_none() {
                continue;
            }
            let a = p.blocks[header].phis[i].args[n];
            let y = self.spill_slot(a, lane, &mut Vec::new(), true)?;
            if self.rpo[q] < self.rpo[header] {
                fwd.push((q, y));
            } else {
                back.push((q, y));
            }
        }
        let z = self.s.m.i32(0);
        let init = self.pred_select_edges_at(pre_dxil, header, &fwd, z);
        let next = self.pred_select_edges_at(latch_dxil, header, &back, z);
        self.s
            .m
            .set_phi(v_phi, vec![(init, pre_dxil), (next, latch_dxil)]);
        Ok(())
    }

    /// Computes all SSA values defined inside `members` that are referenced outside `members`
    /// (directly, via phis, or via `Bool` guards/exits/masks), plus masks read as data outside
    /// `members` and mask phis referenced outside `members`.
    fn loop_live_outs(&self, members: &HashSet<usize>) -> (Vec<Value>, Vec<Value>, Vec<Value>) {
        let p = self.x.p;
        let st = self.x.st;
        let defined_in = |v: Value| -> bool {
            self.def_block.get(&v).is_some_and(|b| members.contains(b))
                || self.phi_at.get(&v).is_some_and(|(b, _)| members.contains(b))
        };

        let mut live = BTreeSet::new();
        let mut mask_data = BTreeSet::new();
        let mut mask_phis = BTreeSet::new();

        let note_bool =
            |b: Bool, live: &mut BTreeSet<Value>, mask_phis: &mut BTreeSet<Value>| {
                for a in st.bools.support(b) {
                    match a {
                        Atom::Lane(v) | Atom::Uniform(v) | Atom::Opaque(v) => {
                            if defined_in(v) {
                                live.insert(v);
                            }
                            if let Atom::Opaque(v) = a
                                && let Some(h) = self.pair_high(v)
                                && defined_in(h)
                            {
                                live.insert(h);
                            }
                        }
                        Atom::Phi(v) => {
                            if defined_in(v) {
                                live.insert(v);
                                mask_phis.insert(v);
                            }
                        }
                        Atom::Constant(_) | Atom::Quad(_) => {}
                    }
                }
            };

        for (b, block) in p.blocks.iter().enumerate() {
            if members.contains(&b) || self.rpo[b] == usize::MAX {
                continue;
            }
            for phi in &block.phis {
                for &arg in &phi.args {
                    if defined_in(arg) {
                        live.insert(arg);
                        if let Some(&m) = st.masks.get(&arg) {
                            note_bool(m, &mut live, &mut mask_phis);
                            mask_data.insert(arg);
                        } else if let Some(&lo) = st.highs.get(&arg)
                            && defined_in(lo)
                        {
                            live.insert(lo);
                            mask_data.insert(lo);
                            if let Some(&m) = st.masks.get(&lo) {
                                note_bool(m, &mut live, &mut mask_phis);
                            }
                        }
                    }
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if let Some(g) = st.guards[b][i] {
                    note_bool(g, &mut live, &mut mask_phis);
                }
                let reads_mask = super::reads_mask_as_data(st, inst);
                for &(_, v) in &inst.reads {
                    if defined_in(v) {
                        live.insert(v);
                        if reads_mask {
                            if st.masks.contains_key(&v) {
                                mask_data.insert(v);
                            } else if let Some(&lo) = st.highs.get(&v)
                                && defined_in(lo)
                            {
                                live.insert(lo);
                                mask_data.insert(lo);
                            }
                        }
                        if let Some(&m) = st.masks.get(&v) {
                            note_bool(m, &mut live, &mut mask_phis);
                        }
                    }
                }
                // Include `s_cmp_eq_u64`/`s_cmp_lg_u64` or other SALU reading a mask as data.
                if matches!(inst.fields, Fields::Sop { .. }) {
                    for &(_, v) in &inst.reads {
                        if defined_in(v) && st.masks.contains_key(&v) {
                            mask_data.insert(v);
                        } else if let Some(&lo) = st.highs.get(&v)
                            && defined_in(lo)
                        {
                            live.insert(lo);
                            mask_data.insert(lo);
                        }
                    }
                }
            }
            if let LaneExit::Branch { when, .. } = st.exits[b] {
                note_bool(when, &mut live, &mut mask_phis);
            }
        }

        (
            live.into_iter().collect(),
            mask_data.into_iter().collect(),
            mask_phis.into_iter().collect(),
        )
    }
}
