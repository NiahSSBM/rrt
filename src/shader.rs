use vulkano::{descriptor_set::DescriptorSetWithOffsets, shader::EntryPoint};

#[derive(Clone)]
pub struct Shaders {
    pub vs: EntryPoint,
    pub fs: EntryPoint,
    pub descriptor_set: Option<DescriptorSetWithOffsets>,
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

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
    }
}