use engine::renderer::*;

use std::time::Instant;

use godot::{
    classes::{InputEvent, Label, TextureRect},
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
    #[export]
    debug_mode: bool,

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
            debug_mode: false,
            clock: Instant::now(),
        }
    }

    fn ready(&mut self) {
        self.renderer.init();

        let viewport = self.viewport.as_mut().unwrap();

        self.renderer.bind(viewport);

        // Create parameters buffer with time and precision values
        let parameters = [0.0; 5];
        let mut parameters_bytes = PackedFloat32Array::from(&parameters).to_byte_array();
        parameters_bytes.push(0);

        // Create a uniform buffer for our parameters
        self.renderer
            .create_storage_uniform("core", "global", parameters_bytes.len() as u32, 0)
            .unwrap();
        self.renderer
            .update_buffer("core", "global", 0, &parameters_bytes)
            .unwrap();

        // fill the voxel data
        let voxels_per_units: f32 = 4.;

        // load the voxel data from file
        let data = std::fs::read("assets/models/voxel_sphere_octree.vox").expect("Couldn't find the model");

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
                4,
                &PackedByteArray::from(&data[16..20]), // grid size
            )
            .unwrap();

        self.renderer
            .update_buffer(
                "core",
                "voxel_data",
                8,
                &PackedByteArray::from(&data[32..36]), // node count
            )
            .unwrap();

        self.renderer
            .update_buffer(
                "core",
                "voxel_data",
                12,
                &PackedByteArray::from(&data[48..]), // octree data
            )
            .unwrap();

        //godot_print!("First node {}", PackedByteArray::from(&data[48..96]).to_int32_array());

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
        let mut parameters_bytes = PackedFloat32Array::from(&parameters).to_byte_array();
        parameters_bytes.push(self.debug_mode as u8);
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
