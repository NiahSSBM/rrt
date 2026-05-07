use std::sync::{Arc, Mutex};

use nalgebra::Matrix4;

use crate::mesh::Mesh3D;

pub struct Object {
    pub mesh: Option<Arc<Mutex<Mesh3D>>>,
    pub transform: [[f32; 4]; 4],
}

impl Object {
    pub fn new() -> Self {
        Self {
            mesh: None,
            transform: Matrix4::identity().into(),
        }
    }

    pub fn from_mesh(mesh: Arc<Mutex<Mesh3D>>) -> Self {
        Self {
            mesh: Some(mesh),
            transform: Matrix4::identity().into(),
        }
    }
}
