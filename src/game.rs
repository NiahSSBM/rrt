use std::sync::Arc;
use std::{sync::mpsc, thread, time::Duration};

use rand::Rng;

use crate::mesh::{Mesh, MyVertex};
use crate::shader::{Shaders, fs, vs};

pub enum RenderEvent {
    AddMesh(Mesh),
    //LoadShader()
}

pub struct GameData {
    pub to_render: mpsc::Sender<RenderEvent>,
    pub render_device: Arc<vulkano::device::Device>,
}

pub fn game_main(data: GameData) {
    //let mut tri_count = 0;
    let vs = vs::load(data.render_device.clone())
        .expect("Failed to load custom vertex shader module!")
        .entry_point("main")
        .expect("Couldn't find custom vertex shader module entry point!");
    let fs = fs::load(data.render_device.clone())
        .expect("Failed to load custom vertex shader module!")
        .entry_point("main")
        .expect("Couldn't find custom vertex shader module entry point!");

    let mut meshes = vec![];
    let mesh = Mesh::new(
        vec![
            MyVertex::new([0.0, 0.5]),
            MyVertex::new([-0.5, -0.5]),
            MyVertex::new([-1.0, 0.5]),
        ],
        None,
    );
    meshes.push(mesh);
    let mesh = Mesh::new(
        vec![
            MyVertex::new([1.0, 0.5]),
            MyVertex::new([0.5, -0.5]),
            MyVertex::new([0.0, 0.5]),
        ],
        Some(Shaders {
            vs,
            fs,
            descriptor_set: None,
        }),
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
