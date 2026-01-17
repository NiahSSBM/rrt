use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::shader::Shaders;

#[derive(Clone)]
pub struct Mesh {
    pub verticies: Vec<MyVertex>,
    pub shaders: Shaders,
    _id: u32, // unused
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

impl Mesh {
    pub fn new(verts: Vec<MyVertex>, shaders: Shaders) -> Self {
        Self {
            verticies: verts,
            shaders: shaders,
            _id: 0, // unused
        }
    }
}

impl MyVertex {
    pub fn new(position: [f32; 2]) -> Self {
        Self { position: position }
    }
}

pub fn combine_verticies(verts: Vec<Vec<MyVertex>>) -> Vec<MyVertex> {
    let mut out: Vec<MyVertex> = Vec::new();
    for mut vec in verts {
        out.try_reserve(vec.len())
            .unwrap_or_else(|e| panic!("Could not combine verticies: {:?}", e));
        out.append(&mut vec);
    }
    out
}
