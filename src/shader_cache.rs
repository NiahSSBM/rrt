use std::collections::HashMap;
use std::sync::Arc;
use vulkano::image::Image;
use vulkano::shader::{EntryPoint, ShaderModule};
use vulkano_taskgraph::Id;

pub struct ShaderCache {
    entry_points: HashMap<String, EntryPoint>,
    images: HashMap<String, Id<Image>>,
}

impl ShaderCache {
    pub fn new() -> ShaderCache {
        Self {
            entry_points: HashMap::new(),
            images: HashMap::new(),
        }
    }

    pub fn add_entry(&mut self, name: &str, entry: EntryPoint) {
        self.entry_points.insert(name.into(), entry);
    }

    pub fn add_image(&mut self, name: &str, image: Id<Image>) {
        self.images.insert(name.into(), image);
    }

    pub fn get_entry(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.get(name)
    }

    pub fn get_image(&self, name: &str) -> Option<&Id<Image>> {
        self.images.get(name)
    }
}