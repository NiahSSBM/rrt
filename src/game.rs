use std::collections::HashMap;
use std::fs::OpenOptions;
use std::ops::Add;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use color::AlphaColor;
use color::palette::css;
use nalgebra::Matrix4;
use nalgebra::Point3;
use nalgebra::Vector3;
use rand::TryRngCore;
use stl_io::IndexedMesh;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh3D;
use crate::shader::AdditionalShaderProperties;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex3D;

pub enum RenderEvent {
    AddMesh(Arc<Mutex<Mesh3D>>),
    UpdateVertexBuffer,
    UpdateTaskGraph,
    UpdateShader,
}

pub enum GameEvent {
    GameClose,
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
}

struct GameState {
    camera: Camera,
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
    view_offset: f32,
}

impl GameState {
    fn new() -> Self {
        Self {
            camera: Camera::new(),
            meshes: Vec::new(),
            view_offset: 0.0,
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

fn update(data: &GameData, state: &mut GameState) -> GameStatus {
    state.view_offset += 0.05;
    state.camera.perspective = [
        Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
        Matrix4::look_at_rh(
            &Point3::new(3.0 * state.view_offset.sin(), state.view_offset.cos() * 3.0, 2.0), // Where the camera is
            &Point3::new(0.0, 0.0, 1.0),   // Where the camera looks
            &Vector3::new(0.0, 0.0, -1.0), // What axis is up
        )
        .into(),
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

    thread::sleep(Duration::from_millis(16));

    match data.from_render.try_recv() {
        Ok(event) => match event {
            GameEvent::GameClose => {
                println!("Game thread exiting...");
                return GameStatus::Exit
            }
        },
        Err(_) => {}
    }

    data.to_render
        .send(RenderEvent::UpdateShader)
        .expect("Failed to request shader update!");
    data.to_render
        .send(RenderEvent::UpdateTaskGraph)
        .expect("Failed to request task graph update!");

    GameStatus::Ok
}

fn physics_update() {}

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
        let status = update(&data, &mut state);
        if status == GameStatus::Exit {
            break
        }
    }
}
