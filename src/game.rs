use std::collections::HashMap;
use std::fs::OpenOptions;
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
use vulkano::buffer::view;
use vulkano::device::Queue;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh2D;
use crate::mesh::Mesh3D;
use crate::shader::AdditionalShaderProperties;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex2D;
use crate::shader::Vertex3D;

pub enum RenderEvent {
    AddMesh(Arc<Mutex<Mesh3D>>),
    UpdateVertexBuffer,
}

pub enum GameEvent {
    GameClose,
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub from_render: mpsc::Receiver<GameEvent>,
    pub render_queue: Arc<Queue>,
}

pub fn game_main(data: GameData) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);

    // Load STL model file
    let mut file = OpenOptions::new().read(true).open("models/horse.stl").unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();
    println!(
        "Number of triangles read from file: {:?}",
        stl.faces.iter().size_hint()
    );
    println!("Model validated: {:?}", stl.validate());

    let mut model_verts: Vec<Vertex3D> = vec![];
    let mut model_indicies: Vec<usize> = vec![];

    for stl_vert in stl.vertices {
        let colors: [AlphaColor<color::Srgb>; 3] = [css::RED, css::BLUE, css::GREEN];
        let rand = rand::rng().try_next_u32().unwrap() % 3;

        model_verts.push(Vertex3D::new(stl_vert.into(), colors[rand as usize]));
    }
    for stl_faces in stl.faces {
        for tri_indicies in stl_faces.vertices {
            model_indicies.push(tri_indicies);
        }
    }

    let mut perspective = AdditionalShaderProperties::Perspective(
        Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
        Matrix4::look_at_rh(
            &Point3::new(4.0, 0.0, 0.0),  // Where the camera is
            &Point3::new(0.0, 0.0, 0.0),  // Where the camera looks
            &Vector3::new(0.0, 1.0, 0.0), // What axis is up
        )
        .into(),
        Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
    );
    let mut tri_shaders = Shader::new(
        stage_pipeline.clone(),
        vec![perspective.clone()],
        data.render_queue.clone(),
    );

    let mut meshes = vec![];
    let mesh = Arc::new(Mutex::new(Mesh3D::new(
        model_verts.clone(),
        model_indicies.iter().map(|i| i.clone() as u32).collect(),
        tri_shaders.clone(),
    )));
    meshes.push(mesh);
    for mesh in &meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
    }

    let mut view_offset: f32 = 0.0;
    loop {
        // First check if the main thread is telling us to exit
        match data.from_render.try_recv() {
            Ok(event) => match event {
                GameEvent::GameClose => {
                    println!("Game thread exiting...");
                    break;
                }
            },
            Err(_) => {}
        }

        //vert_offsets += 0.01;
        view_offset += 0.05;
        perspective = AdditionalShaderProperties::Perspective(
            Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
            Matrix4::look_at_rh(
                &Point3::new(3.0 * view_offset.sin(), view_offset.cos() * 3.0, 2.0), // Where the camera is
                &Point3::new(0.0, 0.0, 1.0),   // Where the camera looks
                &Vector3::new(0.0, 0.0, -1.0), // What axis is up
            )
            .into(),
            Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
        );

        tri_shaders = meshes[0]
            .lock()
            .unwrap()
            .shader
            .update_descriptor(perspective.clone());

        meshes[0].lock().unwrap().shader = tri_shaders;
        thread::sleep(Duration::from_millis(16));
        data.to_render
            .send(RenderEvent::UpdateVertexBuffer)
            .expect("Failed to request vertex buffer update!");
    }
}
