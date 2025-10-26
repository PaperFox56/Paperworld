use engine::renderer::*;

use std::time::Instant;

use godot::{
    classes::{InputEvent, TextureRect, rendering_device::UniformType},
    prelude::*,
};

#[derive(GodotClass)]
#[class(base=Node)]
pub struct Game {
    base: Base<Node>,
    renderer: Renderer,

    #[export]
    viewport: Option<Gd<TextureRect>>,
    #[export]
    camera: Option<Gd<Camera3D>>,
    clock: Instant,
}

#[godot_api]
impl INode for Game {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            renderer: Renderer::new(),
            viewport: None,
            camera: None,
            clock: Instant::now(),
        }
    }

    fn ready(&mut self) {
        self.renderer.init();

        let viewport = self.viewport.as_mut().unwrap();

        self.renderer.bind(viewport);

        let parameters_bytes = PackedFloat32Array::from([0.]).to_byte_array();

        // Create a uniform buffer for our parameters
        self.renderer
            .create_storage_uniform(
                "core",
                "data",
                parameters_bytes.len() as u32,
                1,
            )
            .unwrap();
        self.renderer
            .update_buffer("core", "data", 0, &parameters_bytes)
            .unwrap();

        // Create a uniform buffer for the camera's data
        const CAMERA_DATA_SIZE: u32 = 16;    // number of float values in the struct
        self.renderer
            .create_uniform_uniform(
                "core",
                "camera",
                CAMERA_DATA_SIZE * 4,
                2, 
            )
            .unwrap();
        self.renderer
            .update_buffer("core", "camera", 0, &PackedFloat32Array::from(&[0.; CAMERA_DATA_SIZE as usize]).to_byte_array())
            .unwrap();
    }

    fn physics_process(&mut self, _delta: f64) {
        let time = self.clock.elapsed().as_secs_f32();
        let parameters_bytes = PackedFloat32Array::from([time]).to_byte_array();
        self.renderer
            .update_buffer("core", "data", 0, &parameters_bytes)
            .unwrap();
        self.renderer
            .update_buffer("core", "camera", 0, &self.get_camera_data().to_byte_array())
            .unwrap();

        self.renderer
            .execute_shader("voxel_shader", &[
                ("core", 0),
            ]);
    }

    fn unhandled_key_input(&mut self, event: Gd<InputEvent>) {
        if event.is_action_pressed("quit") {
            self.base().get_tree().unwrap().quit();
        }
    }
}

impl Game {
    fn get_camera_data(&self) -> CameraData {
        let camera = self.camera.as_ref().unwrap();

        let transform = camera.get_camera_transform();

        CameraData {
            position: transform.origin,
            front: transform.basis.col_c(),
            right: transform.basis.col_a(),
            up: transform.basis.col_b(),
            fov: camera.get_fov(),
        }
    }
}

struct CameraData {
    position: Vector3,
    front: Vector3,
    right: Vector3,
    up: Vector3,

    fov: f32,
}

impl CameraData {
    fn new(position: Vector3, front: Vector3, right: Vector3, up: Vector3, fov: f32) -> Self {
        Self {
            position,
            front,
            right,
            up,
            fov,
        }
    }

    fn to_byte_array(&self) -> PackedByteArray {
        let mut out = Vec::new();

        // zeros are added because the GPU is a bitch 

        out.extend(self.position.to_array());
        out.push(0.);
        out.extend(self.front.to_array());
        out.push(0.);
        out.extend(self.right.to_array());
        out.push(0.);
        out.extend(self.up.to_array());
        out.push(self.fov);


        let out = PackedFloat32Array::from(out).to_byte_array();

        //godot_print!("{:?}", out.len());

        out
    }
}

pub struct RustExtension;

#[gdextension]
unsafe impl ExtensionLibrary for RustExtension {}
