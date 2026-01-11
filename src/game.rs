use std::{sync::mpsc, thread, time::Duration};

use rand::{Rng};

use crate::mesh::{Mesh, MyVertex};

pub enum RenderEvent {
    AddMesh(Mesh),
}

pub fn game_main(to_render: mpsc::Sender<RenderEvent>) {
	let mut tri_count = 0;
    loop {
        let xoffset = rand::rng().random_range(-1.0..1.0);
        let yoffset = rand::rng().random_range(-1.0..1.0);
        let mesh = Mesh::new(
            vec![
                MyVertex::new([0.05 + xoffset, 0.05 + yoffset]),
                MyVertex::new([0.0 + xoffset, -0.05 + yoffset]),
                MyVertex::new([-0.05 + xoffset, 0.05 + yoffset]),
            ],
            None,
        );
		tri_count += 1;
        //println!("Mesh count {tri_count}");
        to_render
            .send(RenderEvent::AddMesh(mesh.clone()))
            .expect("Failed to send mesh data to render thread!");
        thread::sleep(Duration::from_millis(50));
    }
}
