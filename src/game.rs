use bitstream_io::{ByteRead, ByteReader, LittleEndian};
use color::AlphaColor;
use color::palette::css;
use gltf::json::accessor::{ComponentType, Type};
use image::ImageBuffer;
use image::ImageError;
use image::Rgba;
use image::open;
use nalgebra::Point3;
use nalgebra::Rotation3;
use nalgebra::Vector3;
use nalgebra::{Matrix4, max};
use rand::TryRngCore;
use std::collections::{BTreeMap, HashMap};
use std::f32::consts::PI;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
use std::vec;
use stl_io::IndexedMesh;
use vulkano::shader::ShaderStage;
use winit::event::KeyEvent;
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;

use crate::mesh::{Mesh3D, Triangle};
use crate::object::Object;
use crate::shader::AdditionalShaderProperties;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex3D;
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

    // Load STL model files
    let stl_paths = vec![];
    let stl_models = load_stls(stl_paths.clone());

    // Load GLTF model files
    let mut gltf_paths = vec![];
    for i in 0..1 {
        gltf_paths.push("models/parot.glb");
    }
    let gltf_models = load_gltfs(gltf_paths);

    let texture_paths = vec!["textures/grid.png"];
    data.persistent_textures = load_images(texture_paths);

    for model in gltf_models {
        for (i, (vertices, indices, normals, texcoords)) in model {
            let tri_shaders = Shader::new(
                stage_pipeline.clone(),
                Some(shader_cache.clone()),
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
            let mut model_tris: Vec<Triangle> = vec![];

            println!("Number of vertices: {}", vertices.len());
            println!("Number of indices: {}", indices.len());
            println!("Number of normals: {}", normals.len());

            let vertices = vertices.as_chunks::<3>().0;
            let indices = indices.as_chunks::<3>().0;
            let normals = normals.as_chunks::<3>().0;

            for (j, index) in indices.iter().enumerate() {
                model_tris.push(Triangle::new(
                    *index,
                    *normals.get(j).unwrap_or_else(|| &[0.0, 0.0, 0.0]),
                ));
            }

            for (j, vert) in vertices.iter().enumerate() {
                model_verts.push(Vertex3D::new(
                    vert.clone(),
                    normals.get(j).unwrap().clone(),
                    css::WHITE,
                ));
            }

            let mesh = Arc::new(Mutex::new(Mesh3D::new(
                model_verts.clone(),
                model_tris,
                tri_shaders,
            )));

            let mut object = Object::from_mesh(mesh.clone());
            object.translate(Vector3::new(-1.0, 1.0, -3.0));
            //object.rotate(Rotation3::from_axis_angle(&Vector3::y_axis(), PI / 2.0));
            object.rotate(Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 1.0));

            state.objects.push(object);
        }
    }

    // Assemble vertices into models
    let mut i = 0;
    for model in stl_models {
        let tri_shaders = Shader::new(
            stage_pipeline.clone(),
            Some(shader_cache.clone()),
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
        let mut model_tris: Vec<Triangle> = vec![];

        for vertex in model.vertices {
            let colors: [AlphaColor<color::Srgb>; 3] = [css::RED, css::BLUE, css::GREEN];
            let rand = rand::rng().try_next_u32().unwrap() % 3;

            // TODO: Use real normals here
            model_verts.push(Vertex3D::new(
                vertex.into(),
                [0.0, 1.0, 0.0],
                colors[rand as usize],
            ));
        }

        for face in model.faces {
            model_tris.push(Triangle::new(
                face.vertices.map(|v| v.clone() as u32),
                [0.0, 1.0, 0.0],
            ));
        }

        let mesh = Arc::new(Mutex::new(Mesh3D::new(
            model_verts.clone(),
            model_tris,
            tri_shaders,
        )));

        let mut object = Object::from_mesh(mesh.clone());
        object.translate(Vector3::new(-1.0 + (i as f32), 1.0, -3.0));
        object.rotate(Rotation3::from_axis_angle(&Vector3::y_axis(), PI / 2.0));
        object.rotate(Rotation3::from_axis_angle(&Vector3::x_axis(), PI / 2.0));

        //state.objects.push(object);
        //i += 3;
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

            // mesh.shader
            //     .set_texture(data.persistent_textures.get(0).unwrap().clone());
            // if state.frame_counter % 60 <= 30 {
            //     mesh.shader
            //         .set_texture(data.persistent_textures.get(1).unwrap().clone());
            // } else {
            //     mesh.shader
            //         .set_texture(data.persistent_textures.get(0).unwrap().clone());
            // }

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

fn load_gltfs(paths: Vec<&str>) -> Vec<BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)>> {
    let mut out: Vec<BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)>> = Vec::new();
    for path in paths {
        println!("Loading GLTF {}", path);
        let (document, buffers, _images) = match gltf::import(path) {
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

        out.push(gltf_load_model(document, buffers));
    }
    out
}

fn gltf_load_model(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
) -> BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)> {
    let buffer = buffers.get(0).unwrap(); // Not sure what to do with the other buffers
    let meshes = document.meshes();

    let mut vertex_map: BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)> = BTreeMap::new();
    let default_vertex_map = (vec![], vec![], vec![], vec![]);

    let vertex_accessors = get_gltf_accessors(&document, gltf::Semantic::Positions);
    let normal_accessors = get_gltf_accessors(&document, gltf::Semantic::Normals);
    let tex_accessors = get_gltf_accessors(&document, gltf::Semantic::TexCoords(0));
    let index_accessors = get_gltf_index_accessors(&document);

    for (i, accessor) in vertex_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
        }
    }

    for (i, accessor) in normal_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
        }
    }

    for (i, accessor) in tex_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
        }
    }

    for (i, accessor) in index_accessors.iter().enumerate() {
        print_accessor(&accessor, 0);
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
        }
    }

    vertex_map
}

fn gltf_get_accessor_data<T: bitstream_io::Primitive>(
    accessor: &gltf::Accessor,
    buffer: &gltf::buffer::Data,
) -> Vec<T> {
    let mut out = Vec::new();
    let data_width = match accessor.data_type() {
        ComponentType::I8 => 1,
        ComponentType::U8 => 1,
        ComponentType::I16 => 2,
        ComponentType::U16 => 2,
        ComponentType::U32 => 4,
        ComponentType::F32 => 4,
    };
    let data_dimensions = match accessor.dimensions() {
        Type::Scalar => 1,
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 => 4,
        Type::Mat2 => 4,
        Type::Mat3 => 9,
        Type::Mat4 => 16,
    };
    if let Some(view) = accessor.view() {
        let stride = view.stride().unwrap_or(0);
        let buffer_slice = &buffer[view.offset() + accessor.offset()
            ..view.offset() + accessor.offset() + max(accessor.size(), stride) * accessor.count()];
        let mut reader = ByteReader::endian(buffer_slice, LittleEndian);

        for component in 0..accessor.count() {
            for i in 0..data_dimensions {
                let next = reader.read::<T>().unwrap();
                out.push(next);
                if stride != 0 {
                    reader
                        .skip((stride - data_width * data_dimensions) as u32)
                        .unwrap();
                }
            }
        }
    }
    out
}

fn get_gltf_accessors(
    document: &gltf::Document,
    desired_semantic: gltf::Semantic,
) -> Vec<gltf::Accessor> {
    let mut accessors = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(accessor) = primitive.get(&desired_semantic) {
                accessors.push(accessor);
            }
        }
    }
    accessors
}

fn get_gltf_index_accessors(document: &gltf::Document) -> Vec<gltf::Accessor> {
    let mut accessors = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(accessor) = primitive.indices() {
                accessors.push(accessor);
            }
        }
    }
    accessors
}

fn walk_gltf_nodes(node: &gltf::Node, depth: usize) {
    println!(
        "{:width$}Loaded Node {} with {} child nodes",
        "",
        node.name().unwrap_or("[unnammed node]"),
        node.children().len(),
        width = depth
    );
    println!(
        "{:width$}Node transform {:?}",
        "",
        node.transform(),
        width = depth
    );
    if let Some(mesh) = node.mesh() {
        println!(
            "{:width$}Loaded Mesh {} with {} primitives",
            "",
            mesh.name().unwrap_or("[unnammed mesh]"),
            mesh.primitives().len(),
            width = depth + 1
        );
        // Primitives are assumed to be triangles
        // Unsure how this behaves if they are any other type
        for primitive in mesh.primitives() {
            println!(
                "{:width$}Loaded {:?} primitive with {} attributes",
                "",
                primitive.mode(),
                primitive.attributes().len(),
                width = depth + 2
            );

            println!(
                "{:width$}Primitive has {} morph targets",
                "",
                primitive.morph_targets().len(),
                width = depth + 2
            );

            if let Some(accessor) = primitive.indices() {
                println!(
                    "{:width$}Loaded index accessor with {} indices",
                    "",
                    accessor.count(),
                    width = depth + 3
                );
                print_accessor(&accessor, depth + 3)
            }

            for (semantic, accessor) in primitive.attributes() {
                println!(
                    "{:width$}Loaded semantic {:?}",
                    "",
                    semantic,
                    width = depth + 3
                );
                print_accessor(&accessor, depth + 3);
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
    for node in node.children() {
        walk_gltf_nodes(&node, depth + 1);
    }
}

fn print_accessor(accessor: &gltf::Accessor, depth: usize) {
    println!(
        "{:width$}Loaded accessor {}",
        "",
        accessor.name().unwrap_or("[unnammed accessor]"),
        width = depth + 3
    );
    if let Some(sparse) = accessor.sparse() {
        println!("{:width$}Accessor is sparse", "", width = depth + 3);
    }
    if let Some(view) = accessor.view() {
        println!(
            "{:width$}Accessor view offset {}",
            "",
            view.offset(),
            width = depth + 3
        );
        println!(
            "{:width$}Accessor view size {}",
            "",
            view.length(),
            width = depth + 3
        );
        if let Some(stride) = view.stride() {
            println!(
                "{:width$}Accessor view stride {}",
                "",
                stride,
                width = depth + 3
            );
        }
    }
    println!(
        "{:width$}Accessor offset {}",
        "",
        accessor.offset(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor data type {:?}",
        "",
        accessor.data_type(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor component size {}",
        "",
        accessor.size(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor count {}",
        "",
        accessor.count(),
        width = depth + 3
    );
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
