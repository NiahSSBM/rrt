use crate::shader::{Shaders, Vertex2D};

#[derive(Clone)]
pub struct Mesh {
    pub verticies: Vec<Vertex2D>,
    pub shaders: Shaders,
    _id: u32, // unused
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

pub fn combine_verticies(verts: Vec<Vec<Vertex2D>>) -> Vec<Vertex2D> {
    let mut out: Vec<Vertex2D> = Vec::new();
    for mut vec in verts {
        out.try_reserve(vec.len())
            .unwrap_or_else(|e| panic!("Could not combine verticies: {:?}", e));
        out.append(&mut vec);
    }
    out
}
