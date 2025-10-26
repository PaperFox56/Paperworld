use std::ops::Deref;

use godot::{builtin::Rid, classes::RdUniform};

#[derive(Debug, Copy, Clone)]
pub(super) enum BufferType {
    StorageBuffer,
    UniformBuffer,
    TextureBuffer,
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
