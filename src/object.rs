use std::sync::{Arc, Mutex};

use nalgebra::{Matrix3, Point3, Rotation3, Transform3, Vector3};

use crate::mesh::Mesh3D;

pub struct Object {
    pub mesh: Option<Arc<Mutex<Mesh3D>>>,
    pub transform: Transform3<f32>,
}

impl Object {
    pub fn new() -> Self {
        Self {
            mesh: None,
            transform: Transform3::identity(),
        }
    }

    pub fn from_mesh(mesh: Arc<Mutex<Mesh3D>>) -> Self {
        Self {
            mesh: Some(mesh),
            transform: Transform3::identity(),
        }
    }

    pub fn translate(&mut self, vector: Vector3<f32>) {
        self.transform = Transform3::from_matrix_unchecked(
            self.transform
                .to_homogeneous()
                .prepend_translation(&vector)
                .into(),
        );
    }

    pub fn rotate(&mut self, rotation: Rotation3<f32>) {
        self.transform *= rotation;
    }
}
