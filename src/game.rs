use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

use color::AlphaColor;
use color::palette::css;
use vulkano::device::Queue;
use vulkano::shader::ShaderStage;

use crate::mesh::Mesh;
use crate::shader::ShaderType;
use crate::shader::Shaders;
use crate::shader::Vertex2D;

pub enum RenderEvent {
    AddMesh(Mesh),
    //LoadShader()
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub render_queue: Arc<Queue>,
    //pub available_shaders: Shaders,
}

pub fn game_main(mut data: GameData) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);
    let first_tri_shaders = Shaders::new(stage_pipeline, data.render_queue.clone());

    //let mut second_tri_shaders = Shaders::new(data.render_queue.clone());
    //second_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::VertexWireframe);
    //second_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::FragmentWireframe);

    let mut meshes = vec![];
    let mesh = Mesh::new(
        vec![
            Vertex2D::new([0.0, 0.5], css::RED),
            Vertex2D::new([-0.5, -0.5], css::BLUE),
            Vertex2D::new([-1.0, 0.5], css::GREEN),
        ],
        first_tri_shaders,
    );
    meshes.push(mesh);
    //let mesh = Mesh::new(
    //    vec![
    //        Vertex2D::new([1.0, 0.5], AlphaColor::WHITE),
    //        Vertex2D::new([0.5, -0.5], AlphaColor::WHITE),
    //        Vertex2D::new([0.0, 0.5], AlphaColor::WHITE),
    //    ],
    //    second_tri_shaders,
    //);
    //meshes.push(mesh);
    for mesh in meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
    }

    loop {
        //let xoffset = rand::rng().random_range(-1.0..1.0);
        //let yoffset = rand::rng().random_range(-1.0..1.0);
        //let mesh = Mesh::new(
        //    vec![
        //        Vertex2D::new([0.05 + xoffset, 0.05 + yoffset]),
        //        Vertex2D::new([0.0 + xoffset, -0.05 + yoffset]),
        //        Vertex2D::new([-0.05 + xoffset, 0.05 + yoffset]),
        //    ],
        //    None,
        //);
        //tri_count += 1;
        //println!("Mesh count {tri_count}");
        //to_render
        //    .send(RenderEvent::AddMesh(mesh.clone()))
        //    .expect("Failed to send mesh data to render thread!");
        //thread::sleep(Duration::from_millis(5));
    }
}
