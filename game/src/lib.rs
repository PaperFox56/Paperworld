use engine::renderer::*;

use std::time::Instant;

use godot::{
    classes::{InputEvent, Label, TextureRect},
    prelude::*,
};

struct Parameters {
    epsilon: f32,
    ray_offset: f32,
    intersection_offset: f32,
    max_steps: f32,
}

#[derive(GodotClass)]
#[class(base=Node)]
pub struct Game {
    base: Base<Node>,
    renderer: Renderer,

    #[export]
    viewport: Option<Gd<TextureRect>>,
    #[export]
    camera: Option<Gd<Camera3D>>,
    #[export]
    fps_indicator: Option<Gd<Label>>,
    #[export]
    epsilon: f32,
    #[export]
    ray_offset: f32,
    #[export]
    intersection_offset: f32,
    #[export]
    max_steps: f32,

    clock: Instant,
}

#[godot_api]
impl INode for Game {
    fn init(base: Base<Node>) -> Self {
        Self {
            base,
            renderer: Renderer::new(800, 640),
            viewport: None,
            camera: None,
            fps_indicator: None,
            epsilon: 0.001,
            ray_offset: 0.01,
            intersection_offset: 0.001,
            max_steps: 256.0,
            clock: Instant::now(),
        }
    }

    fn ready(&mut self) {
        self.renderer.init();

        let viewport = self.viewport.as_mut().unwrap();

        self.renderer.bind(viewport);

        // Create parameters buffer with time and precision values
        let parameters = [
            0.0, // time
            self.epsilon,
            self.ray_offset,
            self.intersection_offset,
            self.max_steps as f32,
        ];
        let parameters_bytes = PackedFloat32Array::from(&parameters).to_byte_array();

        // Create a uniform buffer for our parameters
        self.renderer
            .create_storage_uniform("core", "global", parameters_bytes.len() as u32, 0)
            .unwrap();
        self.renderer
            .update_buffer("core", "global", 0, &parameters_bytes)
            .unwrap();

        // fill the voxel data
        let voxels_per_units: f32 = 5.;
        let (w, h, d): (i32, i32, i32) = (16, 16, 16);
        let mut voxel_array = PackedInt32Array::new();
        voxel_array.resize((w * h * d) as usize);

        let mut index = 0;

        godot_print!("{}", voxel_array.len());

        for i in 0..w {
            for j in 0..h {
                for k in 0..d {
                    if Vector3::new((i - w / 2) as f32, (j - h / 2) as f32, (k - d / 2) as f32)
                        .length()
                        < 6.
                    {
                        voxel_array.insert(index, 1);
                    }

                    index += 1;
                }
            }
            godot_print!("{i}");
        }

        self.renderer
            .update_buffer(
                "core",
                "voxel_data",
                0,
                &PackedFloat32Array::from(&[voxels_per_units]).to_byte_array(),
            )
            .unwrap();
        self.renderer
            .update_buffer(
                "core",
                "voxel_data",
                16,
                &PackedInt32Array::from(&[w, h, d]).to_byte_array(),
            )
            .unwrap();
        self.renderer
            .update_buffer("core", "voxel_data", 32, &voxel_array.to_byte_array())
            .unwrap();

        // add a custom shader
    }

    fn physics_process(&mut self, delta: f64) {
        let time = self.clock.elapsed().as_secs_f32();
        let parameters = [
            time,
            self.epsilon,
            self.ray_offset,
            self.intersection_offset,
            self.max_steps as f32,
        ];
        let parameters_bytes = PackedFloat32Array::from(&parameters).to_byte_array();
        self.renderer
            .update_buffer("core", "global", 0, &parameters_bytes)
            .unwrap();

        self.renderer.render_frame(self.get_camera_data());

        let fps = (1. / delta) as u32;
        let fps = format!("FPS: {}", fps);

        self.fps_indicator.as_mut().unwrap().set_text(&fps);
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

        CameraData::new(
            transform.origin,
            transform.basis.col_c(),
            transform.basis.col_a(),
            transform.basis.col_b(),
            camera.get_fov(),
        )
    }
}

pub struct RustExtension;

#[gdextension]
unsafe impl ExtensionLibrary for RustExtension {}
