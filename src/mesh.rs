use image::{ImageBuffer, Rgba};
use crate::shader::{Shader, Vertex3D};

#[derive(Clone, Debug)]
pub struct Triangle {
    pub indices: [u32; 3],
    pub normal: [f32; 3],
}

impl Triangle {
    pub(crate) fn new(indices: [u32; 3], normal: [f32; 3]) -> Self {
        Self { indices, normal }
    }
}

#[derive(Clone)]
pub struct Mesh3D {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub triangles: Vec<Triangle>,
    pub shader: Shader,
}

impl Mesh3D {
    pub fn new(vertices: Vec<Vertex3D>, triangles: Vec<Triangle>, shader: Shader) -> Self {
        let mut indices: Vec<u32> = Vec::with_capacity(vertices.len() * 3);

        for triangle in &triangles {
            indices.extend(triangle.indices);
        }

        Self {
            vertices,
            indices,
            triangles,
            shader,
        }
    }
}
