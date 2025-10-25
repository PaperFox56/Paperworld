use std::collections::HashMap;

use godot::builtin::{Array, PackedByteArray, Rid};
use godot::classes::image::Format;
use godot::classes::rendering_device::{DataFormat, TextureUsageBits, UniformType};
use godot::classes::{ Image, RdShaderFile, RdTextureFormat, RdTextureView, RdUniform, RenderingDevice, RenderingServer, Texture2Drd, TextureRect};
use godot::global::godot_print;
use godot::obj::{Gd, NewGd};
use godot::tools::load as godot_load;

mod shader;

use shader::*;

pub struct Renderer<'a> {
    rendering_device: Gd<RenderingDevice>,

    shaders: HashMap<&'a str, ComputeShader>,
    uniforms: HashMap<&'a str, Uniform>,

    viewport_texture: Gd<Texture2Drd>,
}

impl<'a> Renderer<'a> {
    /// Build a new Renderer ready to be used
    pub fn new() -> Self {
        let rendering_device = RenderingServer::singleton().get_rendering_device()
            .expect("Couldn't obtain the rendering device");


        Self {
            rendering_device,
            shaders: HashMap::new(),
            uniforms: HashMap::new(),
            viewport_texture: Texture2Drd::new_gd(),
        }
    }

    pub fn init(&mut self) {
        // bind the shaders
        self.load_shader_from_file("voxel_shader", "voxel_shader.glsl").unwrap();

        // texture used as output
        let mut texture_format = RdTextureFormat::new_gd();
        texture_format.set_width(1024);
        texture_format.set_height(1024);
        texture_format.set_format(DataFormat::R32G32B32A32_SFLOAT);
        texture_format.set_usage_bits(TextureUsageBits::STORAGE_BIT |
                                                TextureUsageBits::CAN_COPY_FROM_BIT |
                                                TextureUsageBits::CAN_UPDATE_BIT |
                                                TextureUsageBits::SAMPLING_BIT
        );
        let image = Image::create_empty(1024, 1024, false, Format::RGBAF)
            .expect("Couldn't create the color buffer");
        let image_bytes = image.get_data();

        let texture_view = RdTextureView::new_gd();

        self.create_texture_uniform(
            "color_buffer", 
            UniformType::IMAGE, 
            texture_format,
            texture_view,
            1,
        ).unwrap();

        self.update_buffer("color_buffer", 0,  &image_bytes).unwrap();

        // assign the texture to the viewport
        let texture_rid = self.get_buffer_rid("color_buffer");
        self.viewport_texture.set_texture_rd_rid(texture_rid);
    }

    pub fn bind(&self, viewport: &mut Gd<TextureRect>) {
        viewport.set_texture(&self.viewport_texture);
    }

    /// Create a new `RdUniform` with a dedicated storage buffer
    pub fn create_storage_uniform(&mut self, label: &'a str, u_type: UniformType, size: u32, binding: i32) -> Result<(), String> {
        
        let buffer = self.rendering_device.storage_buffer_create(size);

        match buffer {
            Rid::Valid(_) => {
                let mut uniform = RdUniform::new_gd();
                uniform.set_uniform_type(u_type);
                uniform.set_binding(binding);
                uniform.add_id(buffer);

                self.uniforms.insert(label, Uniform::new(uniform, BufferType::StorageBuffer));

                Ok(())
            }
            Rid::Invalid => Err(format!("Couldn't create buffer `{label}`")),
        }

    }

    /// Create a new `RdUniform` with a dedicated texture
    pub fn create_texture_uniform(&mut self, 
        label: &'a str, 
        u_type: UniformType, 
        format: Gd<RdTextureFormat>, 
        view: Gd<RdTextureView>, 
        binding: i32)
             -> Result<(), String> {

        let texture = self.rendering_device.texture_create(&format, &view);

        if texture == Rid::Invalid {
            return Err(format!("Couldn't create buffer `{label}`"));
        }

        let mut uniform = RdUniform::new_gd();
        uniform.set_uniform_type(u_type);
        uniform.set_binding(binding);
        uniform.add_id(texture);

        self.uniforms.insert(label, Uniform::new(uniform, BufferType::TextureBuffer));

        Ok(())
    }

    /// Update a buffer with the given value
    /// 
    /// #Panic
    /// Panics if the buffer is not found
    pub fn update_buffer(&mut self, label: &str, offset: u32, data: &PackedByteArray) -> Result<(), String> {
        let uniform = self.uniforms.get(label)
            .expect(&format!("Uniform `{label}` not found, consider binding the buffer"));
        
        let buffer = uniform.get_ids().get(0)
            .unwrap();

        let e = match uniform.get_buffer_type() {
            BufferType::StorageBuffer => self.rendering_device.buffer_update(buffer, offset, data.len() as u32, data),
            BufferType::TextureBuffer => self.rendering_device.texture_update(buffer, offset, data),
        };

        match e {
            godot::global::Error::OK => Ok(()),
            _ => Err(format!("Couldn't update {:?} buffer `{label}`. Godot error {e:?}", uniform.get_buffer_type())),
        }
    }

    #[inline]
    fn get_buffer_rid(&self, label: &str) -> Rid {
        self.uniforms.get(label)
            .expect(&format!("Uniform `{label}` not found, consider binding the buffer"))
            .get_ids()
            .get(0)
            .unwrap()
    }

    pub fn get_buffer_data(&mut self, label: &str) -> PackedByteArray {
        let buffer = self.get_buffer_rid(label);
        self.rendering_device.buffer_get_data(buffer)
    }

    /// Create a new compute pipeline, bind the required uniforms, and call the shader
    pub fn execute_shader(&mut self, shader_label: &str, uniform_config: UniformConfig) {
        
        let shader_rid = self.shaders.get(shader_label).unwrap().get_rid();

        // We need get the inner uniforms to be bound to the shader
        let uniforms: Vec<Gd<RdUniform>> = self.uniforms
                            .iter()
                            .filter_map(|(l, u)| match uniform_config {
                                UniformConfig::All => Some(u.clone_inner()),
                                UniformConfig::Custom(list) if list.contains(l) => {
                                    Some(u.clone_inner())
                                }
                                _ => None
                            }).collect();

        let uniforms = Array::from(uniforms.as_slice());

        let uniform_set = self.rendering_device.uniform_set_create(&uniforms, shader_rid, 0);


        // Compute pipeline
        let pipeline =self.rendering_device.compute_pipeline_create(shader_rid);
        let compute_list = self.rendering_device.compute_list_begin();
        self.rendering_device.compute_list_bind_compute_pipeline(compute_list, pipeline);
        self.rendering_device.compute_list_bind_uniform_set(compute_list, uniform_set, 0);
        self.rendering_device.compute_list_dispatch(compute_list, 32, 32, 1);
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
    pub fn load_shader_from_file(&mut self, label: &'a str, path_to_shader: &str) -> Result<(), Error> {
        let full_path = format!("res://assets/shaders/{path_to_shader}");

        let shader_file: Gd<RdShaderFile> = godot_load(&full_path);
        
        match shader_file.get_spirv() {
            Some(shader_spirv) => {
                let rid = self.rendering_device.shader_create_from_spirv(&shader_spirv);

                let shader = ComputeShader::new(rid);

                self.shaders.insert(label, shader);

                Ok(())
            }
            None => Err(Error(format!("Couldn't load shader `{label}` at {full_path}")))
        }
    }
}

impl<'a> Drop for Renderer<'a> {
    fn drop(&mut self) {

        godot_print!("Freeing the acquiered ressources");

        // free every shader
        self.shaders.iter().for_each(|(&label, shader)| {
            godot_print!("Freeing shader `{label}`");
            self.rendering_device.free_rid(shader.get_rid());
        });

        // free every buffer
        self.uniforms.iter().for_each(|(&label, uniform)| {
            godot_print!("Freeing uniform `{label}`");
            uniform.get_ids().iter_shared().for_each(|rid| self.rendering_device.free_rid(rid) );
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

/// Allow to choose different sets of uniforms when running a shader
pub enum UniformConfig<'b> {
    Custom(&'b [&'b str]),
    All,
}