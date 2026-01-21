use color::{AlphaColor, Srgb};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::shader::Shaders;

#[derive(Clone)]
pub struct Mesh {
    pub verticies: Vec<Vertex2D>,
    pub shaders: Shaders,
    _id: u32, // unused
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl Mesh {
    pub fn new(verts: Vec<Vertex2D>, shaders: Shaders) -> Self {
        Self {
            verticies: verts,
            shaders: shaders,
            _id: 0, // unused
        }
    }
}

impl Vertex2D {
    pub fn new(position: [f32; 2], color: AlphaColor<Srgb>) -> Self {
        Self {
            position: position,
            color: color.components
        }
    }
}

pub fn combine_verticies(verts: Vec<Vec<Vertex2D>>) -> Vec<Vertex2D> {
    let mut out: Vec<Vertex2D> = Vec::new();
    for mut vec in verts {
        out.try_reserve(vec.len())
            .unwrap_or_else(|e| panic!("Could not combine verticies: {:?}", e));
        out.append(&mut vec);
    }
    out
}
