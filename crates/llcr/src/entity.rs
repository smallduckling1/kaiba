// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Copyright (C) The Kaiba Authors
//
// This file is part of the Kaiba compiler toolchain. It is licensed under
// GPL-2.0-only with the Kaiba Runtime/Output Exception; see COPYING and
// LICENSE-EXCEPTION at the repository root.

//! Index-based entity arenas. All IR objects live in `PrimaryMap`s and refer
//! to one another with small `Copy` ids, so there are no `Rc`/`RefCell` cycles.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A typed, 32-bit index into a [`PrimaryMap`].
pub trait Entity: Copy + Eq {
    fn new(index: usize) -> Self;
    fn index(self) -> usize;
}

macro_rules! entity_id {
    ($(#[$m:meta])* $name:ident, $prefix:literal) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);
        impl Entity for $name {
            #[inline]
            fn new(index: usize) -> Self {
                $name(u32::try_from(index).expect("entity index overflow"))
            }
            #[inline]
            fn index(self) -> usize { self.0 as usize }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(self, f)
            }
        }
    };
}

entity_id!(/// An SSA value (function parameter or instruction result).
    ValueId, "%v");
entity_id!(/// An instruction within a function.
    InstId, "i");
entity_id!(/// A basic block within a function.
    BlockId, "bb");
entity_id!(/// A function within a module.
    FuncId, "@f");
entity_id!(/// A global variable within a module.
    GlobalId, "@g");
entity_id!(/// An interned aggregate type (struct/array/vector) within a module.
    AggId, "agg");

/// Dense arena keyed by a typed id.
#[derive(Clone)]
pub struct PrimaryMap<K: Entity, V> {
    items: Vec<V>,
    _k: PhantomData<K>,
}

impl<K: Entity, V> Default for PrimaryMap<K, V> {
    fn default() -> Self { Self { items: Vec::new(), _k: PhantomData } }
}

impl<K: Entity, V: fmt::Debug> fmt::Debug for PrimaryMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_list().entries(&self.items).finish() }
}

impl<K: Entity, V> PrimaryMap<K, V> {
    pub fn new() -> Self { Self::default() }
    pub fn push(&mut self, v: V) -> K {
        let k = K::new(self.items.len());
        self.items.push(v);
        k
    }
    pub fn next_key(&self) -> K { K::new(self.items.len()) }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn get(&self, k: K) -> Option<&V> { self.items.get(k.index()) }
    pub fn get_mut(&mut self, k: K) -> Option<&mut V> { self.items.get_mut(k.index()) }
    pub fn keys(&self) -> impl Iterator<Item = K> { (0..self.items.len()).map(K::new) }
    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> { self.items.iter().enumerate().map(|(i, v)| (K::new(i), v)) }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> {
        self.items.iter_mut().enumerate().map(|(i, v)| (K::new(i), v))
    }
}

impl<K: Entity, V> Index<K> for PrimaryMap<K, V> {
    type Output = V;
    #[inline]
    fn index(&self, k: K) -> &V { &self.items[k.index()] }
}
impl<K: Entity, V> IndexMut<K> for PrimaryMap<K, V> {
    #[inline]
    fn index_mut(&mut self, k: K) -> &mut V { &mut self.items[k.index()] }
}
