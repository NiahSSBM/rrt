use std::collections::HashMap;
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

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub render_queue: Arc<Queue>,
}

pub fn game_main(data: GameData) {
    // Load shaders
    let stage_pipeline = HashMap::from([
        (ShaderStage::Vertex, ShaderType::VertexDefault),
        (ShaderStage::Fragment, ShaderType::FragmentDefault),
    ]);
    let tri_verts = vec![
        // First tri
        Vertex3D::new([0.0, 1.0, 0.0], css::RED),
        Vertex3D::new([-1.0, -1.0, 1.0], css::BLUE),
        Vertex3D::new([1.0, -1.0, 1.0], css::GREEN),
        // Second tri
        Vertex3D::new([0.0, 1.0, 0.0], css::RED),
        Vertex3D::new([-1.0, -1.0, -1.0], css::BLUE),
        Vertex3D::new([1.0, -1.0, -1.0], css::GREEN),
        // Third tri
        Vertex3D::new([0.0, 1.0, 0.0], css::RED),
        Vertex3D::new([-1.0, -1.0, -1.0], css::BLUE),
        Vertex3D::new([-1.0, -1.0, 1.0], css::GREEN),
        // Fourth tri
        Vertex3D::new([0.0, 1.0, 0.0], css::RED),
        Vertex3D::new([1.0, -1.0, 1.0], css::BLUE),
        Vertex3D::new([1.0, -1.0, -1.0], css::GREEN),

    ];
    let mut perspective = AdditionalShaderProperties::Perspective(
            Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
            Matrix4::look_at_rh(
                &Point3::new(2.0, 0.0,  0.0), // Where the camera is
                &Point3::new(0.0, 0.0, 0.0), // Where the camera looks
                &Vector3::new(0.0, 1.0, 0.0), // What axis is up
            )
            .into(),
            Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
        );
    let mut tri_shaders = Shader::new(
            stage_pipeline.clone(),
            vec![perspective],
            data.render_queue.clone(),
        );

    let mut meshes = vec![];
    let mesh = Arc::new(Mutex::new(Mesh3D::new(tri_verts.clone(), tri_shaders)));
    meshes.push(mesh);
    for mesh in &meshes {
        data.to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
    }

    let mut view_offset: f32 = 0.0;
    loop {
        //vert_offsets += 0.01;
        view_offset += 0.1;
        perspective = AdditionalShaderProperties::Perspective(
            Matrix4::new_rotation(Vector3::new(0.0, 0.0, 0.0)).into(),
            Matrix4::look_at_rh(
                &Point3::new(3.0 * view_offset.sin(), 0.0 , view_offset.cos() * 3.0), // Where the camera is
                &Point3::new(0.0, 0.0, 0.0), // Where the camera looks
                &Vector3::new(0.0, -1.0, 0.0), // What axis is up
            )
            .into(),
            Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
        );

        tri_shaders = meshes[0].lock().unwrap().shader.update_descriptor(perspective);
        //tri_shaders = Shader::new(
        //    stage_pipeline.clone(),
        //    vec![perspective],
        //    data.render_queue.clone(),
        //);

        meshes[0].lock().unwrap().shader = tri_shaders;
        thread::sleep(Duration::from_millis(16));
        data.to_render
            .send(RenderEvent::UpdateVertexBuffer)
            .expect("Failed to request vertex buffer update!");
    }
}
