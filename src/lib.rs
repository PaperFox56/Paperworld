mod renderer;
mod player;

use std::{time::Instant};

use godot::{classes::{rendering_device::UniformType, InputEvent, TextureRect}, prelude::*};

use crate::renderer::Renderer;


#[derive(GodotClass)]
#[class(base=Node)]
pub struct Game{
    base: Base<Node>,
    renderer: Renderer<'static>,

    #[export]
    viewport: Option<Gd<TextureRect>>,
    #[export]
    camera: Option<Gd<Camera3D>>,
    clock: Instant
}

#[godot_api]
impl INode for Game {
    fn init(base: Base <Node>) -> Self {

        Self {
            base,
            renderer: Renderer::new(),
            viewport: None,
            camera: None,
            clock: Instant::now()
        }
    }

    fn ready(&mut self) {
        self.renderer.init();

        let viewport = self.viewport.as_mut().unwrap();

        self.renderer.bind(viewport);

        let parameters_bytes = PackedFloat32Array::from([0.]).to_byte_array();

        // Create a storage buffer for our values
        self.renderer.create_storage_uniform("data", UniformType::STORAGE_BUFFER, parameters_bytes.len() as u32, 0).unwrap();
        self.renderer.update_buffer("data", 0, &parameters_bytes).unwrap();
    }

    fn physics_process(&mut self, _delta: f64) {

        let time = self.clock.elapsed().as_secs_f32();
        let parameters_bytes = PackedFloat32Array::from([time]).to_byte_array();
        self.renderer.update_buffer("data", 0, &parameters_bytes).unwrap();
    
        self.renderer.execute_shader("voxel_shader", renderer::UniformConfig::All);
    }

    fn unhandled_key_input(&mut self, event: Gd<InputEvent>) {
        if event.is_action_pressed("quit") {
            self.base().get_tree().unwrap().quit();
        }
    }
}

pub struct RustExtension;

#[gdextension]
unsafe impl ExtensionLibrary for RustExtension {}

