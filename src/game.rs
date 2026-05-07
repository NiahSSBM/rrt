use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;

use color::AlphaColor;
use color::palette::css;
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
use crate::object;
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
    objects: Vec<Object>,
    last_physics_update: Instant,
    delta: Duration,
    keys_held: Vec<KeyCode>,
}

impl GameState {
    fn new() -> Self {
        Self {
            camera: Camera::new(),
            objects: Vec::new(),
            last_physics_update: Instant::now(),
            delta: Duration::ZERO,
            keys_held: Vec::new(),
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
        let object = Object::from_mesh(mesh);

        state.objects.push(object);
    }

    // Send each mesh to be added to the vertex buffer
    for object in &state.objects {
        if object.mesh.is_some() {
            // Skip empty objects
            data.to_render
                .send(RenderEvent::AddMesh(object.mesh.clone().unwrap())) // TODO: see if we can get around this clone
                .expect("Failed to send mesh data to render thread!");
            data.to_render
                .send(RenderEvent::UpdateVertexBuffer)
                .expect("Failed to request vertex buffer update!");
        }
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

    for object in &mut state.objects {
        let mut model: Matrix4<f32> = Matrix4::identity();
        model = model.prepend_translation(&Vector3::new(0.0, 1.0, -2.0));
        model = model
            * Rotation3::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI / 2.0)
                .to_homogeneous();
        model = model * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.0).to_homogeneous();
        model = model * Rotation3::from_axis_angle(&Vector3::z_axis(), 0.0).to_homogeneous();
        model = model.scale(1.0);

        object.transform = model.into();
    }

    state.camera.perspective = [
        state.objects.get(0).unwrap().transform.into(), // Model
        Matrix4::look_at_rh(
            &state.camera.position,
            &(state.camera.position + state.camera.front),
            &state.camera.up,
        )
        .into(), // View
        Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(), // Projection
    ];

    for object in &state.objects {
        if object.mesh.is_some() {
            // Skip empty objects
            let mut mesh = object.mesh.as_ref().unwrap().lock().unwrap();

            mesh.shader
                .update_descriptor(AdditionalShaderProperties::Perspective(
                    state.camera.perspective[0],
                    state.camera.perspective[1],
                    state.camera.perspective[2],
                ));
        }
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
