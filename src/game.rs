use std::sync::Arc;
use std::sync::mpsc;

use crate::mesh::{Mesh, MyVertex};
use crate::shader::ShaderType;
use crate::shader::{Shaders};

pub enum RenderEvent {
    AddMesh(Mesh),
    //LoadShader()
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub render_device: Arc<vulkano::device::Device>,
    pub available_shaders: Shaders
}

pub fn game_main(mut data: GameData) {
    // Initialize shaders
    data.available_shaders.load(ShaderType::VertexDefault, data.render_device.clone());
    data.available_shaders.load(ShaderType::VertexCustom, data.render_device.clone());

    data.available_shaders.load(ShaderType::FragmentDefault, data.render_device.clone());
    data.available_shaders.load(ShaderType::FragmentCustom, data.render_device.clone());
    data.available_shaders.load(ShaderType::FragmentWireframe, data.render_device.clone());

    // Only use needed shaders for each mesh
    let mut first_tri_shaders = Shaders::new();
    first_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::VertexDefault);
    first_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::FragmentDefault);

    let mut second_tri_shaders = Shaders::new();
    second_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::VertexDefault);
    second_tri_shaders.insert_loaded(&data.available_shaders, ShaderType::FragmentDefault);

    let mut meshes = vec![];
    let mesh = Mesh::new(
        vec![
            MyVertex::new([0.0, 0.5]),
            MyVertex::new([-0.5, -0.5]),
            MyVertex::new([-1.0, 0.5]),
        ],
        first_tri_shaders,
    );
    meshes.push(mesh);
    let mesh = Mesh::new(
        vec![
            MyVertex::new([1.0, 0.5]),
            MyVertex::new([0.5, -0.5]),
            MyVertex::new([0.0, 0.5]),
        ],
        second_tri_shaders
    );
    meshes.push(mesh);
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
        //        MyVertex::new([0.05 + xoffset, 0.05 + yoffset]),
        //        MyVertex::new([0.0 + xoffset, -0.05 + yoffset]),
        //        MyVertex::new([-0.05 + xoffset, 0.05 + yoffset]),
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
