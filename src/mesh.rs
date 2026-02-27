use crate::shader::{Shader, Vertex2D, Vertex3D};

#[derive(Clone)]
pub struct Mesh2D {
    pub verticies: Vec<Vertex2D>,
    pub shader: Shader,
}

#[derive(Clone)]
pub struct Mesh3D {
    pub verticies: Vec<Vertex3D>,
    pub shader: Shader,
}

impl Mesh2D {
    pub fn new(verts: Vec<Vertex2D>, shader: Shader) -> Self {
        Self {
            verticies: verts,
            shader: shader,
        }
    }
}

impl Mesh3D {
    pub fn new(verts: Vec<Vertex3D>, shader: Shader) -> Self {
        Self {
            verticies: verts,
            shader: shader,
        }
    }
}

pub fn combine_verticies(verts: Vec<Vec<Vertex3D>>) -> Vec<Vertex3D> {
    let mut out: Vec<Vertex3D> = Vec::new();
    for mut vec in verts {
        out.try_reserve(vec.len())
            .unwrap_or_else(|e| panic!("Could not combine verticies: {:?}", e));
        out.append(&mut vec);
    }
    out
}
