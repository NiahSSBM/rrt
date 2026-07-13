use image::ImageBuffer;
use image::ImageError;
use image::Rgba;
use image::open;
use nalgebra::Point3;
use nalgebra::Rotation3;
use nalgebra::Vector3;
use nalgebra::{Matrix4, Scale3, Similarity3, TGeneral, Transform3, Translation3, try_convert};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
use std::vec;
use vulkano::shader::ShaderStage;
use winit::event::KeyEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;

use crate::mesh::Mesh3D;
use crate::object::Object;
use crate::shader::ShaderType;
use crate::shader::{AdditionalShaderProperties, Shader};
use crate::shader_cache::ShaderCache;

const PHYSICS_UPDATES_PER_SECOND: f32 = 60.0;
const MOUSE_SENSITIVITY: f32 = 0.05;
const MOVE_SPEED: f32 = 0.05;

pub enum RenderEvent {
    AddMesh(Arc<Mutex<Mesh3D>>),
    UpdateVertexBuffer,
    UpdateTaskGraph,
    UpdateShader,
}

pub enum GameEvent {
    GameClose,
    CursorMoved((f64, f64)),
    KeyEvent(KeyEvent),
}

#[derive(PartialEq)]
enum GameStatus {
    Ok,
    Exit,
}

struct Camera {
    perspective: [[[f32; 4]; 4]; 3],
    position: Point3<f32>,
    front: Vector3<f32>,
    up: Vector3<f32>,
    right: Vector3<f32>,
    world_up: Vector3<f32>,
    yaw: f32,
    pitch: f32,
}

pub struct GameData {
    to_render: mpsc::Sender<RenderEvent>,
    from_render: mpsc::Receiver<GameEvent>,
    persistent_textures: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl GameData {
    pub fn new(
        to_render: mpsc::Sender<RenderEvent>,
        from_render: mpsc::Receiver<GameEvent>,
    ) -> Self {
        Self {
            to_render,
            from_render,
            persistent_textures: Vec::new(),
        }
    }
}

struct GameState {
    camera: Camera,
    objects: Vec<Object>,
    last_physics_update: Instant,
    delta: Duration,
    keys_held: Vec<KeyCode>,
    frame_counter: u64,
}

impl GameState {
    fn new() -> Self {
        Self {
            camera: Camera::new(),
            objects: Vec::new(),
            last_physics_update: Instant::now(),
            delta: Duration::ZERO,
            keys_held: Vec::new(),
            frame_counter: 0,
        }
    }
}

impl Camera {
    fn new() -> Self {
        Self {
            perspective: [
                Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
                Matrix4::look_at_rh(
                    &Point3::new(4.0, 0.0, 0.0),  // Where the camera is
                    &Point3::new(0.0, 0.0, 0.0),  // Where the camera looks
                    &Vector3::new(0.0, 1.0, 0.0), // What axis is up
                )
                .into(),
                Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 100.0).into(),
            ],
            position: Point3::origin(),
            front: *Vector3::z_axis(),
            up: *Vector3::y_axis(),
            right: *Vector3::x_axis(),
            world_up: *Vector3::y_axis(),
            yaw: -90.0,
            pitch: 0.0,
        }
    }
}

fn game_init(data: &mut GameData, state: &mut GameState) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);

    // Create shader cache
    let shader_cache = Arc::new(Mutex::new(ShaderCache::new()));

    let texture_paths = vec!["textures/grid.png"];
    data.persistent_textures = load_images(texture_paths);

    let shader = Shader::new(
        stage_pipeline.clone(),
        Some(shader_cache.clone()),
        vec![
            AdditionalShaderProperties::Perspective(
                state.camera.perspective[0],
                state.camera.perspective[1],
                state.camera.perspective[2],
            ),
            AdditionalShaderProperties::Texture(data.persistent_textures.get(0).unwrap().clone()),
        ],
    );

    // Load objects
    let paths = vec![
        //"models/texturedsphere.glb",
        //"models/sphere.glb",
        //"models/smoothsphere.glb",
        //"models/hqsphere.glb",
        //"models/smoothhqsphere.glb",
        //"models/scaletest.glb",
        //"models/trex.glb",
        "models/macaw.glb",
        //"models/scene.gltf",
    ];
    let mut objects = vec![];
    for path in paths {
        objects.push(Object::from_path(&Path::new(path), shader.clone()));
    }

    let mut i = 0;
    for mut object in objects {
        object.translate(Translation3::new(i as f32, 4.0, -4.0));
        object.scale(Scale3::new(5.0, 5.0, 5.0));
        //object.rotate(Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 1.0));
        object.rotate(Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 1.0));

        object.load();

        state.objects.push(object);
        i += 2;
    }

    for object in &state.objects {
        set_object_render(&object, data);
    }
}

/// Enables rendering for a given object. Recurses through child objects.
fn set_object_render(object: &Object, data: &GameData) {
    if object.mesh.is_some() {
        data.to_render
            .send(RenderEvent::AddMesh(object.mesh.clone().unwrap()))
            .expect("Failed to send mesh data to render thread!");
        data.to_render
            .send(RenderEvent::UpdateVertexBuffer)
            .expect("Failed to request vertex buffer update!");
    }
    for child in &object.children {
        set_object_render(child, data);
    }
}

// Runs as quickly as possible
fn update(_data: &GameData, _state: &mut GameState) -> GameStatus {
    GameStatus::Ok
}

// Runs 60 times per second
fn physics_update(data: &GameData, state: &mut GameState) -> GameStatus {
    let (mut x_delta, mut y_delta) = (0.0, 0.0);

    for event in data.from_render.try_iter() {
        match event {
            GameEvent::GameClose => {
                println!("Game thread exiting...");
                return GameStatus::Exit;
            }
            GameEvent::CursorMoved((cursor_x_delta, cursor_y_delta)) => {
                x_delta += cursor_x_delta as f32;
                y_delta += cursor_y_delta as f32;
            }
            GameEvent::KeyEvent(event) => {
                if event.state.is_pressed() && !event.repeat {
                    match event.physical_key {
                        PhysicalKey::Code(key_code) => state.keys_held.push(key_code),
                        PhysicalKey::Unidentified(_) => (),
                    }
                } else if !event.repeat {
                    match event.physical_key {
                        PhysicalKey::Code(key_code) => {
                            match state.keys_held.iter().position(|k| key_code == *k) {
                                Some(i) => {
                                    state.keys_held.remove(i);
                                }
                                None => println!("Key released but was not pressed!"),
                            }
                        }
                        PhysicalKey::Unidentified(_) => (),
                    }
                }
            }
        }
    }

    state.camera.yaw += x_delta * MOUSE_SENSITIVITY;
    state.camera.pitch += y_delta * MOUSE_SENSITIVITY;

    state.camera.pitch = state.camera.pitch.clamp(-89.0, 89.0);

    let new_front: Vector3<f32> = Vector3::new(
        state.camera.yaw.to_radians().cos() * state.camera.pitch.to_radians().cos(),
        state.camera.pitch.to_radians().sin(),
        state.camera.yaw.to_radians().sin() * state.camera.pitch.to_radians().cos(),
    )
    .normalize();

    state.camera.front = new_front;
    state.camera.right = state.camera.front.cross(&state.camera.world_up).normalize();
    state.camera.up = state.camera.right.cross(&state.camera.front).normalize();

    for key in &state.keys_held {
        if *key == KeyCode::KeyW {
            state.camera.position += state.camera.front * MOVE_SPEED
        } else if *key == KeyCode::KeyA {
            state.camera.position += -state.camera.right * MOVE_SPEED;
        } else if *key == KeyCode::KeyS {
            state.camera.position += -state.camera.front * MOVE_SPEED;
        } else if *key == KeyCode::KeyD {
            state.camera.position += state.camera.right * MOVE_SPEED;
        } else if *key == KeyCode::Space {
            state.camera.position += -state.camera.up * MOVE_SPEED;
        } else if *key == KeyCode::ShiftLeft {
            state.camera.position += state.camera.up * MOVE_SPEED;
        }
    }

    for object in &state.objects {
        object.update_view(
            Matrix4::look_at_rh(
                &state.camera.position,
                &(state.camera.position + state.camera.front),
                &state.camera.up,
            )
            .into(),
        )
    }

    if let Err(e) = data.to_render.send(RenderEvent::UpdateShader) {
        println!("Failed to request shader update: {e}");
    }
    if let Err(e) = data.to_render.send(RenderEvent::UpdateTaskGraph) {
        println!("Failed to request task graph update: {e}");
    }

    state.frame_counter += 1;
    GameStatus::Ok
}

fn load_images(paths: Vec<&str>) -> Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let mut loaded_images: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = Vec::new();
    for path in paths {
        println!("Loading image {path}");
        if let Ok(image) = load_image(path) {
            loaded_images.push(image);
        } else {
            println!("Could not load image: {path}");
            continue;
        }
    }
    loaded_images
}

fn load_image(path: &str) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, ImageError> {
    Ok(open(path)?.to_rgba8())
}

pub fn game_main(mut data: GameData) {
    let mut state = GameState::new();
    game_init(&mut data, &mut state);

    // Main loop
    loop {
        state.delta = state.last_physics_update.elapsed();
        // Run physics update if it's been long enough
        if state.delta >= Duration::from_secs_f32(1.0 / PHYSICS_UPDATES_PER_SECOND) {
            let status = physics_update(&data, &mut state);
            if status == GameStatus::Exit {
                break;
            }
            state.last_physics_update = Instant::now();
        }

        // Run normal update as quick as possible
        let status = update(&data, &mut state);
        if status == GameStatus::Exit {
            break;
        }
    }
}
