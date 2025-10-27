use std::{collections::hash_map, ops::Deref};

use super::Dict;
use godot::{builtin::Rid, classes::RdUniform};

#[derive(Debug, Copy, Clone)]
pub(super) enum BufferType {
    StorageBuffer,
    UniformBuffer,
    Texture,
}

pub(super) struct Uniform {
    inner: godot::obj::Gd<RdUniform>,
    buffer_type: BufferType,
}

impl Uniform {
    pub fn new(inner: godot::obj::Gd<RdUniform>, buffer_type: BufferType) -> Self {
        assert_eq!(
            inner.get_ids().is_empty(),
            false,
            "Tried to build a uniform with no buffer attached"
        );

        Self { inner, buffer_type }
    }

    pub fn clone_inner(&self) -> godot::obj::Gd<RdUniform> {
        self.inner.clone()
    }

    pub fn get_buffer_type(&self) -> BufferType {
        self.buffer_type
    }
}

impl Deref for Uniform {
    type Target = RdUniform;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Abstraction over the classic uniform sets.
/// This structure will contains the uniforms created by the `create_*_uniform` functions.
pub(super) struct UniformManager {
    uniform_sets: Dict<Dict<Uniform>>,
}

impl UniformManager {
    pub fn new() -> Self {
        Self {
            uniform_sets: Dict::new(),
        }
    }

    /// Add a new uniform to the the structure. If the specified set doesn't exist, it will be created
    pub fn add_uniform(&mut self, set: &str, label: &str, uniform: Uniform) {
        let set = self
            .uniform_sets
            .entry(set.to_string())
            .or_insert(Dict::new());

        set.insert(label.to_string(), uniform);
    }

    pub fn get_uniform(&self, set: &str, label: &str) -> &Uniform {
        self.uniform_sets
            .get(set)
            .expect(&format!("Uniform set `{set}` was not found"))
            .get(label)
            .expect(&format!("Couldn't find uniform `{label}` in set `{set}`"))
    }

    /// Return an iterator over every uniform
    pub fn get_iter(&self) -> impl Iterator<Item = (&str, &str, &Uniform)> {
        self.uniform_sets.iter().flat_map(|(set_label, set)| {
            set.iter()
                .map(move |(label, uniform)| (set_label.as_str(), label.as_str(), uniform))
        })
    }

    /// Return an iterator over a particular uniform set
    pub fn get_set_iter(&self, set: &str) -> hash_map::Iter<'_, String, Uniform> {
        self.uniform_sets
            .get(set)
            .expect(&format!("Uniform set `{set}` was not found"))
            .iter()
    }
}

pub(super) struct ComputeShader {
    rid: Rid,
}

impl ComputeShader {
    pub fn new(rid: Rid) -> Self {
        assert!(rid != Rid::Invalid, "Trying to create a invalid shader");

        Self { rid }
    }

    pub fn get_rid(&self) -> Rid {
        self.rid
    }
}
