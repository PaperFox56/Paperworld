use std::collections::HashMap;
use std::collections::hash_map;

use godot::builtin::{Array, PackedByteArray, Rid};
use godot::classes::image::Format;
use godot::classes::rendering_device::{DataFormat, TextureUsageBits, UniformType};
use godot::classes::{
    Image, RdShaderFile, RdTextureFormat, RdTextureView, RdUniform, RenderingDevice,
    RenderingServer, Texture2Drd, TextureRect,
};
use godot::global::godot_print;
use godot::obj::{Gd, NewGd};
use godot::tools::load as godot_load;

mod shader;

use shader::*;

///!
/// Almost every rendering related code will be handled by this module.
///
///
///
/// Here are some important informations about the way this code manage buffers.
/// [`UniformManager`] is the structure in charge of storing the buffers allocated as well as their associated uniforms.
/// They are stored in separate sets that cn be specified when executing a shader.
/// Some sets are reserved by the library and already contains some uniforms. You can add update them and bind them to your own
/// shader but be sure to check that the binding and label you choose are not already taken or really nasty things might happen.
///
/// ### Set `core`:
///     - texture `color_buffer` : 0
///     - storage buffer `data` : 1 (to be changed soon)
///     - uniform buffer `camera`: 2
///

///  This should be a thing in the standard library
type Dict<T> = HashMap<String, T>;

pub struct Renderer {
    rendering_device: Gd<RenderingDevice>,

    shaders: Dict<ComputeShader>,
    uniform_manager: UniformManager,

    viewport_texture: Gd<Texture2Drd>,
}

/// Manage the [`RenderingDevice`] and every GPU related ressource.
/// You only need one of it, please don't create more.
///
impl Renderer {
    /// Build a new Renderer ready to be used
    pub fn new() -> Self {
        let rendering_device = RenderingServer::singleton()
            .get_rendering_device()
            .expect("Couldn't obtain the rendering device");

        Self {
            rendering_device,
            shaders: HashMap::new(),
            uniform_manager: UniformManager::new(),
            viewport_texture: Texture2Drd::new_gd(),
        }
    }

    pub fn init(&mut self) {
        // bind the shaders
        self.load_shader_from_file("voxel_shader", "voxel_shader.glsl")
            .unwrap();

        // texture used as output
        let mut texture_format = RdTextureFormat::new_gd();
        texture_format.set_width(1024);
        texture_format.set_height(1024);
        texture_format.set_format(DataFormat::R32G32B32A32_SFLOAT);
        texture_format.set_usage_bits(
            TextureUsageBits::STORAGE_BIT
                | TextureUsageBits::CAN_COPY_FROM_BIT
                | TextureUsageBits::CAN_UPDATE_BIT
                | TextureUsageBits::SAMPLING_BIT,
        );
        // Empty image just to fill the buffer
        let image = Image::create_empty(1024, 1024, false, Format::RGBAF)
            .expect("Couldn't create the color buffer");
        let image_bytes = image.get_data();

        // I don't know what this is but we need it
        let texture_view = RdTextureView::new_gd();

        self.create_texture_uniform(
            "core",
            "color_buffer",
            UniformType::IMAGE,
            texture_format,
            texture_view,
            0,
        )
        .unwrap();

        self.update_buffer("core", "color_buffer", 0, &image_bytes)
            .unwrap();

        // assign the texture to the viewport
        let texture_rid = self.get_buffer_rid("core", "color_buffer");
        self.viewport_texture.set_texture_rd_rid(texture_rid);
    }

    pub fn bind(&self, viewport: &mut Gd<TextureRect>) {
        viewport.set_texture(&self.viewport_texture);
    }

    /// Generic helper to create an `RdUniform` and register it in the uniform manager
    fn create_uniform_generic(
        &mut self,
        set: &str,
        label: &str,
        binding: i32,
        uniform_type: UniformType,
        buffer_type: BufferType,
        rid: Rid,
    ) -> Result<(), String> {
        match rid {
            Rid::Valid(_) => {
                let mut uniform = RdUniform::new_gd();
                uniform.set_uniform_type(uniform_type);
                uniform.set_binding(binding);
                uniform.add_id(rid);

                self.uniform_manager
                    .add_uniform(set, label, Uniform::new(uniform, buffer_type));

                Ok(())
            }
            Rid::Invalid => Err(format!("Couldn't create {buffer_type:?} `{label}`")),
        }
    }

    /// Create a new `RdUniform` with a dedicated storage buffer in the specified set
    pub fn create_storage_uniform(
        &mut self,
        set: &str,
        label: &str,
        size: u32,
        binding: i32,
    ) -> Result<(), String> {
        let rid = self.rendering_device.storage_buffer_create(size);

        self.create_uniform_generic(
            set,
            label,
            binding,
            UniformType::STORAGE_BUFFER,
            BufferType::StorageBuffer,
            rid,
        )
    }

    /// Create a new `RdUniform` with a dedicated uniform buffer.
    /// Yes the name is weird, i might change it soon
    pub fn create_uniform_uniform(
        &mut self,
        set: &str,
        label: &str,
        size: u32,
        binding: i32,
    ) -> Result<(), String> {
        let rid = self.rendering_device.uniform_buffer_create(size);
        self.create_uniform_generic(
            set,
            label,
            binding,
            UniformType::UNIFORM_BUFFER,
            BufferType::UniformBuffer,
            rid,
        )
    }

    /// Create a new `RdUniform` with a dedicated texture
    pub fn create_texture_uniform(
        &mut self,
        set: &str,
        label: &str,
        u_type: UniformType,
        format: Gd<RdTextureFormat>,
        view: Gd<RdTextureView>,
        binding: i32,
    ) -> Result<(), String> {
        let rid = self.rendering_device.texture_create(&format, &view);
        self.create_uniform_generic(
            set,
            label,
            binding,
            u_type,
            BufferType::Texture,
            rid,
        )
    }

    /// Update a buffer with the given value
    ///
    /// #Panic
    /// Panics if the buffer is not found
    pub fn update_buffer(
        &mut self,
        set: &str,
        label: &str,
        offset: u32,
        data: &PackedByteArray,
    ) -> Result<(), String> {
        let uniform = self.uniform_manager.get_uniform(set, label);

        let buffer = uniform.get_ids().get(0).unwrap();

        let e = match uniform.get_buffer_type() {
            BufferType::StorageBuffer | BufferType::UniformBuffer => self
                .rendering_device
                .buffer_update(buffer, offset, data.len() as u32, data),
            BufferType::Texture => self.rendering_device.texture_update(buffer, offset, data),
        };

        match e {
            godot::global::Error::OK => Ok(()),
            _ => Err(format!(
                "Couldn't update {:?} buffer `{label}`. Godot error {e:?}",
                uniform.get_buffer_type()
            )),
        }
    }

    #[inline]
    fn get_buffer_rid(&self, set: &str, label: &str) -> Rid {
        self.uniform_manager
            .get_uniform(set, label)
            .get_ids()
            .get(0)
            .unwrap()
    }

    pub fn get_buffer_data(&mut self, set: &str, label: &str) -> PackedByteArray {
        let buffer = self.get_buffer_rid(set, label);
        self.rendering_device.buffer_get_data(buffer)
    }

    /// Create a new compute pipeline, bind the required uniforms, and call the shader
    pub fn execute_shader(&mut self, shader_label: &str, sets_to_bind: &[(&str, usize)]) {
        let shader_rid = self.shaders.get(shader_label).unwrap().get_rid();

        // Compute pipeline
        let pipeline = self.rendering_device.compute_pipeline_create(shader_rid);
        let compute_list = self.rendering_device.compute_list_begin();
        self.rendering_device
            .compute_list_bind_compute_pipeline(compute_list, pipeline);

        // bind the required uniform sets
        for (set_label, set_index) in sets_to_bind {
            // We need get the inner uniforms to be bound to the shader
            let mut uniforms: Vec<Gd<RdUniform>> = self
                .uniform_manager
                .get_set_iter(set_label)
                .map(|(_, uniform)| uniform.clone_inner())
                .collect();

            uniforms.sort_by_key(|u| u.get_binding());

            let uniforms = Array::from(uniforms.as_slice());

            let uniform_set = self
                .rendering_device
                .uniform_set_create(&uniforms, shader_rid, 0);
            self.rendering_device.compute_list_bind_uniform_set(
                compute_list,
                uniform_set,
                *set_index as u32,
            );
        }
        self.rendering_device
            .compute_list_dispatch(compute_list, 32, 32, 1);
        self.rendering_device.compute_list_end();

        // execute
        // self.rendering_device.submit();
        // self.rendering_device.sync();
    }

    /// Load a shader file and store the Rid.
    /// Note that the path is relative to `assets/shaders`
    ///
    /// # Panics
    /// Panics if the file can't be accessed.
    pub fn load_shader_from_file(
        &mut self,
        label: &str,
        path_to_shader: &str,
    ) -> Result<(), Error> {
        let full_path = format!("res://assets/shaders/{path_to_shader}");

        let shader_file: Gd<RdShaderFile> = godot_load(&full_path);

        match shader_file.get_spirv() {
            Some(shader_spirv) => {
                let rid = self
                    .rendering_device
                    .shader_create_from_spirv(&shader_spirv);

                let shader = ComputeShader::new(rid);

                self.shaders.insert(label.to_string(), shader);

                Ok(())
            }
            None => Err(Error(format!(
                "Couldn't load shader `{label}` at {full_path}"
            ))),
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        godot_print!("Freeing the acquiered ressources");

        // free every shader
        self.shaders.iter().for_each(|(label, shader)| {
            godot_print!("Freeing shader `{label}`");
            self.rendering_device.free_rid(shader.get_rid());
        });

        // free every buffer
        self.uniform_manager
            .get_iter()
            .for_each(|(set_label, label, uniform)| {
                godot_print!("Freeing uniform `{label}` of set `{set_label}`");
                uniform
                    .get_ids()
                    .iter_shared()
                    .for_each(|rid| self.rendering_device.free_rid(rid));
            });
    }
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}

impl std::error::Error for Error {}

/// Abstraction over the classic uniform sets.
/// This structure will contains the uniforms created by the `create_*_uniform` functions.
struct UniformManager {
    uniform_sets: Dict<Dict<Uniform>>,
}

impl UniformManager {
    fn new() -> Self {
        Self {
            uniform_sets: Dict::new(),
        }
    }

    /// Add a new uniform to the the structure. If the specified set doesn't exist, it will be created
    fn add_uniform(&mut self, set: &str, label: &str, uniform: Uniform) {
        let set = self
            .uniform_sets
            .entry(set.to_string())
            .or_insert(Dict::new());

        set.insert(label.to_string(), uniform);
    }

    fn get_uniform(&self, set: &str, label: &str) -> &Uniform {
        self.uniform_sets
            .get(set)
            .expect(&format!("Uniform set `{set}` was not found"))
            .get(label)
            .expect(&format!("Couldn't find uniform `{label}` in set `{set}`"))
    }

    /// Return an iterator over every uniform
    fn get_iter(&self) -> impl Iterator<Item = (&str, &str, &Uniform)> {
        self.uniform_sets.iter().flat_map(|(set_label, set)| {
            set.iter()
                .map(move |(label, uniform)| (set_label.as_str(), label.as_str(), uniform))
        })
    }

    /// Return an iterator over a particular uniform set
    fn get_set_iter(&self, set: &str) -> hash_map::Iter<'_, String, Uniform> {
        self.uniform_sets
            .get(set)
            .expect(&format!("Uniform set `{set}` was not found"))
            .iter()
    }
}
