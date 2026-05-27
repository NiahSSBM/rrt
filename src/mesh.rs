use vulkano::image::Image;
use vulkano_taskgraph::Id;

use crate::shader::{Shader, Vertex3D};

#[derive(Clone)]
pub struct Mesh3D {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub shader: Shader,
}

impl Mesh3D {
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>, shader: Shader) -> Self {
        Self {
            vertices,
            indices,
            shader,
        }
    }
}
