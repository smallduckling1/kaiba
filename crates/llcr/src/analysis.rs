// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.
//! CFG analyses: reverse post-order, dominator tree, dominance frontiers.

use crate::entity::{BlockId, Entity};
use crate::function::Function;

/// Reachable blocks in reverse post-order (entry first).
pub fn reverse_postorder(f: &Function) -> Vec<BlockId> {
    let Some(entry) = f.entry() else { return vec![] };
    let mut visited = vec![false; f.blocks.len()];
    let mut post = Vec::new();
    // Iterative DFS: (block, next successor index).
    let mut stack: Vec<(BlockId, Vec<BlockId>, usize)> = Vec::new();
    visited[entry.index()] = true;
    stack.push((entry, f.successors(entry), 0));
    while let Some((b, succs, i)) = stack.last_mut() {
        if *i < succs.len() {
            let s = succs[*i];
            *i += 1;
            if !visited[s.index()] {
                visited[s.index()] = true;
                let ss = f.successors(s);
                stack.push((s, ss, 0));
            }
        } else {
            post.push(*b);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// Dominator tree computed with the Cooper-Harvey-Kennedy algorithm.
#[derive(Clone, Debug)]
pub struct DomTree {
    entry: Option<BlockId>,
    rpo: Vec<BlockId>,
    rpo_num: Vec<usize>,
    idom: Vec<Option<BlockId>>,
}

impl DomTree {
    pub fn compute(f: &Function) -> DomTree {
        let rpo = reverse_postorder(f);
        let n = f.blocks.len();
        let mut rpo_num = vec![usize::MAX; n];
        for (i, b) in rpo.iter().enumerate() { rpo_num[b.index()] = i; }
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        let Some(&entry) = rpo.first() else {
            return DomTree { entry: None, rpo, rpo_num, idom };
        };
        idom[entry.index()] = Some(entry);
        let preds = f.predecessors();
        let intersect = |idom: &Vec<Option<BlockId>>, mut a: BlockId, mut b: BlockId| {
            while a != b {
                while rpo_num[a.index()] > rpo_num[b.index()] { a = idom[a.index()].unwrap(); }
                while rpo_num[b.index()] > rpo_num[a.index()] { b = idom[b.index()].unwrap(); }
            }
            a
        };
        loop {
            let mut changed = false;
            for &b in &rpo[1..] {
                let mut new: Option<BlockId> = None;
                for &p in &preds[&b] {
                    if idom[p.index()].is_none() { continue; }
                    new = Some(match new { None => p, Some(cur) => intersect(&idom, cur, p) });
                }
                if new != idom[b.index()] { idom[b.index()] = new; changed = true; }
            }
            if !changed { break; }
        }
        DomTree { entry: Some(entry), rpo, rpo_num, idom }
    }

    pub fn is_reachable(&self, b: BlockId) -> bool { self.rpo_num.get(b.index()).map_or(false, |&n| n != usize::MAX) }
    pub fn rpo(&self) -> &[BlockId] { &self.rpo }

    /// Immediate dominator (`None` for the entry block and unreachable blocks).
    pub fn idom(&self, b: BlockId) -> Option<BlockId> {
        if Some(b) == self.entry { None } else { self.idom[b.index()] }
    }

    /// Does `a` dominate `b`? (Reflexive.) False if either is unreachable.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if !self.is_reachable(a) || !self.is_reachable(b) { return false; }
        let mut cur = b;
        loop {
            if cur == a { return true; }
            match self.idom(cur) { Some(p) => cur = p, None => return false }
        }
    }

    /// Dominance frontier of every block, indexed by block index.
    pub fn frontiers(&self, f: &Function) -> Vec<Vec<BlockId>> {
        let mut df: Vec<Vec<BlockId>> = vec![vec![]; f.blocks.len()];
        let preds = f.predecessors();
        for &b in &self.rpo {
            let ps: Vec<_> = preds[&b].iter().copied().filter(|&p| self.is_reachable(p)).collect();
            if ps.len() < 2 { continue; }
            let stop = self.idom(b);
            for p in ps {
                let mut runner = p;
                while Some(runner) != stop {
                    if !df[runner.index()].contains(&b) { df[runner.index()].push(b); }
                    match self.idom(runner) { Some(n) => runner = n, None => break }
                }
            }
        }
        df
    }
}
