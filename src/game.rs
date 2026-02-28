use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use color::AlphaColor;
use color::palette::css;
use vulkano::device::Queue;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh2D;
use crate::mesh::Mesh3D;
use crate::shader::Shader;
use crate::shader::ShaderType;
use crate::shader::Vertex2D;
use crate::shader::Vertex3D;

pub enum RenderEvent {
    AddMesh(Arc<Mutex<Mesh3D>>),
    UpdateVertexBuffer,
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub render_queue: Arc<Queue>,
    //pub available_shaders: Shaders,
}

pub fn game_main(data: GameData) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);
    let tri_shaders = Shader::new(stage_pipeline, data.render_queue.clone());
    let mut tri_verts = vec![
            Vertex3D::new([0.0, 0.5, 0.0], css::RED),
            Vertex3D::new([-0.5, -0.5, 0.0], css::BLUE),
            Vertex3D::new([-1.0, 0.5, 0.0], css::GREEN),
        ];

    let mut meshes = vec![];
    let mesh = Arc::new(Mutex::new(Mesh3D::new(
        tri_verts.clone(),
        tri_shaders,
    )));
    meshes.push(mesh);
    for mesh in &meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
    }

    let mut vert_offsets: f32 = 0.0;
    loop {
        vert_offsets += 0.01;
        tri_verts = vec![
            Vertex3D::new([0.0 + vert_offsets.sin(), 0.5 + vert_offsets.cos(), 0.0], css::RED),
            Vertex3D::new([-0.5 + vert_offsets.sin(), -0.5 + vert_offsets.cos(), 0.0], css::BLUE),
            Vertex3D::new([-1.0 + vert_offsets.sin(), 0.5 + vert_offsets.cos(), 0.0], css::GREEN),
        ];

        meshes[0].lock().unwrap().verticies = tri_verts;
        thread::sleep(Duration::from_millis(16));
        data.to_render.send(RenderEvent::UpdateVertexBuffer).expect("Failed to request vertex buffer update!");
    }
}
