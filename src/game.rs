use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;

use color::AlphaColor;
use color::palette::css;
use nalgebra::Matrix;
use nalgebra::Matrix3;
use nalgebra::Matrix4;
use nalgebra::Point;
use nalgebra::Point3;
use nalgebra::Rotation3;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;
use rand::TryRngCore;
use stl_io::IndexedMesh;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh3D;
use crate::shader::AdditionalShaderProperties;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex3D;

const PHYSICS_UPDATES_PER_SECOND: f32 = 60.0;
const MOUSE_SENSITIVITY: f32 = 0.05;

pub enum RenderEvent {
    AddMesh(Arc<Mutex<Mesh3D>>),
    UpdateVertexBuffer,
    UpdateTaskGraph,
    UpdateShader,
}

pub enum GameEvent {
    GameClose,
    CursorMoved((f64, f64)),
}

#[derive(PartialEq)]
enum GameStatus {
    Ok,
    Exit,
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub from_render: mpsc::Receiver<GameEvent>,
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

struct GameState {
    camera: Camera,
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
    last_physics_update: Instant,
    delta: Duration,
}

impl GameState {
    fn new() -> Self {
        Self {
            camera: Camera::new(),
            meshes: Vec::new(),
            last_physics_update: Instant::now(),
            delta: Duration::ZERO,
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
                Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
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

fn game_init(data: &GameData, state: &mut GameState) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);

    // Load STL model file
    let model_paths = vec!["models/horse.stl"];
    let models = load_stls(model_paths);

    // Assemble verticies into models
    for model in models {
        let tri_shaders = Shader::new(
            stage_pipeline.clone(),
            vec![AdditionalShaderProperties::Perspective(
                state.camera.perspective[0],
                state.camera.perspective[1],
                state.camera.perspective[2],
            )],
        );
        let mut model_verts: Vec<Vertex3D> = vec![];
        let mut model_indicies: Vec<usize> = vec![];

        for vertex in model.vertices {
            let colors: [AlphaColor<color::Srgb>; 3] = [css::RED, css::BLUE, css::GREEN];
            let rand = rand::rng().try_next_u32().unwrap() % 3;

            model_verts.push(Vertex3D::new(vertex.into(), colors[rand as usize]));
        }
        for face in model.faces {
            for tri_indicies in face.vertices {
                model_indicies.push(tri_indicies);
            }
        }

        let mesh = Arc::new(Mutex::new(Mesh3D::new(
            model_verts.clone(),
            model_indicies.iter().map(|i| i.clone() as u32).collect(),
            tri_shaders,
        )));
        state.meshes.push(mesh);
    }

    // Send each mesh to be added to the vertex buffer
    for mesh in &state.meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
        data.to_render
            .send(RenderEvent::UpdateVertexBuffer)
            .expect("Failed to request vertex buffer update!");
    }
}

// Runs as quickly as possible
fn update(data: &GameData, state: &mut GameState) -> GameStatus {
    GameStatus::Ok
}

// Runs 60 times per second
fn physics_update(data: &GameData, state: &mut GameState) -> GameStatus {
    let (mut x_delta, mut y_delta) = (0.0, 0.0);

    let events = data.from_render.try_iter();
    for event in events {
        match event {
            GameEvent::GameClose => {
                println!("Game thread exiting...");
                return GameStatus::Exit;
            }
            GameEvent::CursorMoved(cursor_delta) => {
                x_delta += cursor_delta.0 as f32;
                y_delta += cursor_delta.1 as f32;
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

    let model: Matrix4<f32> = Matrix4::from_axis_angle(&Vector3::z_axis(), -1.0);

    state.camera.perspective = [
        model.into(),
        Rotation3::look_at_rh(&state.camera.front, &state.camera.up).to_homogeneous().into(),
        Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
    ];

    for mesh in state.meshes.clone() {
        mesh.lock()
            .unwrap()
            .shader
            .update_descriptor(AdditionalShaderProperties::Perspective(
                state.camera.perspective[0],
                state.camera.perspective[1],
                state.camera.perspective[2],
            ));
    }

    data.to_render
        .send(RenderEvent::UpdateShader)
        .expect("Failed to request shader update!");
    data.to_render
        .send(RenderEvent::UpdateTaskGraph)
        .expect("Failed to request task graph update!");

    GameStatus::Ok
}

fn load_stls(paths: Vec<&str>) -> Vec<IndexedMesh> {
    let mut loaded_models: Vec<IndexedMesh> = Vec::new();

    for path in paths {
        println!("Loading model {}", path);
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

pub fn game_main(data: GameData) {
    let mut state = GameState::new();
    game_init(&data, &mut state);

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
