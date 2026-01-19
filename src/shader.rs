use std::{collections::HashMap, sync::Arc};
use vulkano::{
    descriptor_set::DescriptorSetWithOffsets,
    shader::{EntryPoint, spirv::ExecutionModel},
};

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
pub enum ShaderType {
    VertexDefault,
    VertexCustom,
    VertexWireframe,

    FragmentDefault,
    FragmentCustom,
    FragmentWireframe,
}

#[derive(Clone)]
pub struct ShaderWithDescriptors {
    pub entry_point: EntryPoint,
    // Temporary Option until I can easily create descriptor sets
    pub descriptor_set: Option<DescriptorSetWithOffsets>,
}

#[derive(Clone)]
pub struct Shaders {
    pub loaded: HashMap<ShaderType, ShaderWithDescriptors>,
}

impl Shaders {
    pub fn new() -> Self {
        Self {
            loaded: HashMap::new(),
        }
    }

    // Takes an already loaded shader and copies it to another struct
    // TODO: Return an actual error when a shader isn't found
    pub fn insert_loaded(&mut self, pre_loaded_shaders: &Self, s_type: ShaderType) {
        self.loaded.insert(
            s_type.clone(),
            pre_loaded_shaders.loaded.get(&s_type).cloned().expect("Error: Shader not loaded!"),
        );
    }

    pub fn get_descriptor_sets(&self) -> Vec<DescriptorSetWithOffsets> {
        let mut out = Vec::new();
        for shader in self.loaded.values() {
            out.push(shader.descriptor_set.clone().unwrap());
        }
        out
    }

    pub fn get_entry(&self, execution_model: ExecutionModel) -> Option<&EntryPoint> {
        let mut entry: Option<&EntryPoint> = None;
        for shader in self.loaded.values() {
            let current = &shader.entry_point;
            if current.info().execution_model == execution_model {
                // I'm not sure yet what to do with multiple shaders
                // It's probably a normal thing to do but IDK yet
                if entry.is_some() {
                    panic!(
                        "Error: More than one {execution_model:?} shader found! Only one {execution_model:?} shader per mesh is supported currently"
                    );
                }
                entry = Some(&shader.entry_point);
            }
        }
        entry
    }

    // TODO: Return an actual error when a shader isn't found
    pub fn load(&mut self, s_type: ShaderType, device: Arc<vulkano::device::Device>) {
        self.loaded.insert(
            s_type.clone(),
            match s_type {
                ShaderType::VertexDefault => ShaderWithDescriptors {
                    entry_point: vs_default::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
                ShaderType::VertexCustom => ShaderWithDescriptors {
                    entry_point: vs_custom::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
                ShaderType::VertexWireframe => ShaderWithDescriptors {
                    entry_point: vs_wireframe::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
                ShaderType::FragmentDefault => ShaderWithDescriptors {
                    entry_point: fs_default::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
                ShaderType::FragmentWireframe => ShaderWithDescriptors {
                    entry_point: fs_wireframe::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
                ShaderType::FragmentCustom => ShaderWithDescriptors {
                    entry_point: fs_custom::load(device)
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    descriptor_set: None,
                },
            },
        );
    }
}

pub mod vs_default {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_default.glsl",
    }
}

pub mod fs_default {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag_default.glsl",
    }
}

pub mod vs_custom {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
    }
}

pub mod fs_custom {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
    }
}

pub mod fs_wireframe {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag_wireframe.glsl",
    }
}

pub mod vs_wireframe {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_wireframe.glsl",
    }
}
