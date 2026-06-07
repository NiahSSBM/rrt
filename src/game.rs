use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
use std::vec;

use color::AlphaColor;
use color::palette::css;
use image::ImageBuffer;
use image::ImageError;
use image::Rgba;
use image::open;
use nalgebra::Matrix4;
use nalgebra::Point3;
use nalgebra::Rotation3;
use nalgebra::Vector3;
use rand::TryRngCore;
use stl_io::IndexedMesh;
use vulkano::shader::ShaderStage;
use winit::event::KeyEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;

use crate::mesh::Mesh3D;
use crate::object::Object;
use crate::shader::AdditionalShaderProperties;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex3D;

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

    // Load STL model files
    let stl_paths = vec!["models/horse.stl", "models/pig.stl"];
    let stl_models = load_stls(stl_paths.clone());

    // Load GLTF model files
    let gltf_paths = vec!["models/scene.gltf"];
    let _gltf_models = load_gltfs(gltf_paths);

    let texture_paths = vec!["textures/grid.png", "textures/texture.jpg"];
    data.persistent_textures = load_images(texture_paths);

    // Assemble vertices into models
    let mut i = 0;
    for model in stl_models {
        let tri_shaders = Shader::new(
            stage_pipeline.clone(),
            vec![
                AdditionalShaderProperties::Perspective(
                    state.camera.perspective[0],
                    state.camera.perspective[1],
                    state.camera.perspective[2],
                ),
                AdditionalShaderProperties::Texture(
                    data.persistent_textures
                        .get(i % data.persistent_textures.len())
                        .unwrap()
                        .clone(),
                ),
            ],
        );
        let mut model_verts: Vec<Vertex3D> = vec![];
        let mut model_indices: Vec<u32> = vec![];

        for vertex in model.vertices {
            let colors: [AlphaColor<color::Srgb>; 3] = [css::RED, css::BLUE, css::GREEN];
            let rand = rand::rng().try_next_u32().unwrap() % 3;

            model_verts.push(Vertex3D::new(vertex.into(), colors[rand as usize]));
        }
        for face in model.faces {
            for vertex in face.vertices {
                model_indices.push(vertex as u32);
            }
        }

        let mesh = Arc::new(Mutex::new(Mesh3D::new(
            model_verts.clone(),
            model_indices,
            tri_shaders,
        )));

        let mut object = Object::from_mesh(mesh.clone());
        object.translate(Vector3::new(-1.0 + (i as f32), 1.0, -3.0));
        object.rotate(Rotation3::from_axis_angle(&Vector3::y_axis(), PI / 2.0));
        object.rotate(Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 2.0));

        state.objects.push(object);
        i += 3;
    }

    // Send each mesh to be added to the vertex buffer
    for object in &state.objects {
        if object.mesh.is_some() {
            // Skip empty objects
            data.to_render
                .send(RenderEvent::AddMesh(object.mesh.clone().unwrap()))
                .expect("Failed to send mesh data to render thread!");
            data.to_render
                .send(RenderEvent::UpdateVertexBuffer)
                .expect("Failed to request vertex buffer update!");
        }
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
        if object.mesh.is_some() {
            // Skip empty objects
            let mut mesh = object.mesh.as_ref().unwrap().lock().unwrap();

            if state.frame_counter % 60 <= 30 {
                mesh.shader
                    .set_texture(data.persistent_textures.get(1).unwrap().clone());
            } else {
                mesh.shader
                    .set_texture(data.persistent_textures.get(0).unwrap().clone());
            }

            mesh.shader
                .update_descriptor(AdditionalShaderProperties::Perspective(
                    object.transform.to_homogeneous().into(),
                    Matrix4::look_at_rh(
                        &state.camera.position,
                        &(state.camera.position + state.camera.front),
                        &state.camera.up,
                    )
                    .into(),
                    Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 100.0).into(),
                ));
        }
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

fn load_gltfs(paths: Vec<&str>) {
    for path in paths {
        println!("Loading GLTF {}", path);
        let (document, _buffers, _images) = match gltf::import(path) {
            Ok(g) => g,
            Err(e) => {
                println!("Error opening GLTF file: {e}");
                continue;
            }
        };

        println!("Scenes loaded: {}", document.scenes().count());
        for scene in document.scenes() {
            println!(
                "Loaded Scene {}",
                scene.name().unwrap_or("[unnammed scene]")
            );
            println!("Scene has {} nodes", scene.nodes().len());
            for node in scene.nodes() {
                walk_gltf_nodes(&node, 0);
            }
        }
    }
}

fn walk_gltf_nodes(node: &gltf::Node, depth: usize) {
    for node in node.children() {
        println!(
            "{:width$}Loaded Node {} with {} child nodes",
            "",
            node.name().unwrap_or("[unnammed node]"),
            node.children().len(),
            width = depth
        );
        if let Some(mesh) = node.mesh() {
            println!(
                "{:width$}Loaded Mesh {} with {} primitives",
                "",
                mesh.name().unwrap_or("[unnammed mesh]"),
                mesh.primitives().len(),
                width = depth
            );
            for primitive in mesh.primitives() {
                println!(
                    "{:width$}Loaded {:?} primitive",
                    "",
                    primitive.mode(),
                    width = depth + 1
                );
                if let Some(indices) = primitive.indices() {
                    println!(
                        "{:width$}Primitive contains {} indices and {} attributes",
                        "",
                        indices.count(),
                        primitive.attributes().count(),
                        width = depth + 2
                    );
                }
            }
        }
        if let Some(camera) = node.camera() {
            println!(
                "{:width$}Loaded Camera {}",
                "",
                camera.name().unwrap_or("[unnammed camera]"),
                width = depth
            );
        }
        walk_gltf_nodes(&node, depth + 1);
    }
}

fn load_stls(paths: Vec<&str>) -> Vec<IndexedMesh> {
    let mut loaded_models: Vec<IndexedMesh> = Vec::new();

    for path in paths {
        println!("Loading STL {}", path);
        let mut file = match OpenOptions::new().read(true).open(path) {
            Ok(f) => f,
            Err(e) => {
                println!("Could not open file: {e}");
                continue;
            }
        };

        match stl_io::read_stl(&mut file) {
            Ok(m) => {
                println!("Model validation: {:?}", m.validate());
                println!(
                    "Number of triangles read from file: {:?}",
                    m.faces.iter().size_hint()
                );
                loaded_models.push(m);
            }
            Err(e) => {
                println!("Could not load STL: {e}");
                continue;
            }
        };
    }

    loaded_models
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
