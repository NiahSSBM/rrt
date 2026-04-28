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

pub fn combine_vec<T>(verts: Vec<Vec<T>>) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for mut vec in verts {
        out.try_reserve(vec.len())
            .unwrap_or_else(|e| panic!("Could not combine verticies: {:?}", e));
        out.append(&mut vec);
    }
    out
}
