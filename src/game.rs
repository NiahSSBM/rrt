use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

use color::AlphaColor;
use color::palette::css;
use vulkano::device::Queue;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh2D;
use crate::mesh::Mesh3D;
use crate::shader::ShaderType;
use crate::shader::Shader;
use crate::shader::Vertex2D;
use crate::shader::Vertex3D;

pub enum RenderEvent {
    AddMesh(Mesh3D),
    //LoadShader()
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
    let first_tri_shaders = Shader::new(stage_pipeline, data.render_queue.clone());

    let mut meshes = vec![];
    let mesh = Mesh3D::new(
        vec![
            Vertex3D::new([0.0, 0.5, 0.0], css::RED),
            Vertex3D::new([-0.5, -0.5, 0.0], css::BLUE),
            Vertex3D::new([-1.0, 0.5, 0.0], css::GREEN),
        ],
        first_tri_shaders,
    );
    meshes.push(mesh);
    for mesh in meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
    }
}
