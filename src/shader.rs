use color::{AlphaColor, Srgb};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use vulkano::{
    Validated, VulkanError, buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, command_buffer::{
        AutoCommandBufferBuilder, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    }, descriptor_set::{
        self, DescriptorSet, DescriptorSetWithOffsets, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    }, device::{Device, Queue}, memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    }, pipeline::PipelineLayout, shader::{EntryPoint, ShaderStage, ShaderStages, spirv::ExecutionModel}, sync::GpuFuture
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

#[derive(
    vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone,
)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl Vertex2D {
    pub fn new(position: [f32; 2], color: AlphaColor<Srgb>) -> Self {
        Self {
            position: position,
            color: color.components,
        }
    }
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
            pre_loaded_shaders
                .loaded
                .get(&s_type)
                .cloned()
                .expect("Error: Shader not loaded!"),
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

// TODO: This only needs to be pub for a temporary descriptor creation in vgfx.rs
pub struct VGFXDescriptorSetLayout {
    pub stage: ShaderStages,
    pub descriptor_type: DescriptorType,
    pub descriptor_count: u32,
}

#[derive(Clone)]
pub struct VGFXDescriptorSetLayoutWithData<T> {
    pub layout: Arc<DescriptorSetLayout>,
    pub data: T,
}

// To create a descriptor set layout we need:
// - The pipeline stages the descriptor set is intended for (Vertex, Fragment, All, etc...)
// - The descriptor type (StorageBuffer, StorageImage, etc...) for each descriptor
// - The descriptor count for each descriptor.
//      This one is a little confusing. A descriptor can contain either describe a single "block" of data, or an array of blocks of data.
//      The descriptor count is NOT the total number of descriptors. It's instead the number of elements within a single descriptor
//      If the data is a single element, this should be 1. If the data is an array, this is the array length.
// - The device the descriptor set is used for
pub fn create_descriptor_set_layout(
    layouts: Vec<VGFXDescriptorSetLayout>,
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
    // Enumerate all our bindings
    let mut bindings: BTreeMap<u32, DescriptorSetLayoutBinding> = BTreeMap::new();
    for i in 0..layouts.len() {
        let binding = DescriptorSetLayoutBinding {
            descriptor_count: layouts[i].descriptor_count,
            stages: layouts[i].stage,
            immutable_samplers: Vec::new(),
            ..DescriptorSetLayoutBinding::descriptor_type(layouts[i].descriptor_type)
        };
        bindings.insert(i as u32, binding);
    }

    // Create layout from our bindings
    let create_info = DescriptorSetLayoutCreateInfo {
        flags: Default::default(),
        bindings: bindings.clone(),
        ..Default::default()
    };

    DescriptorSetLayout::new(device.clone(), create_info)
}

// This function combines a descriptor set layout and associated data, sends it to the GPU, and returns the descriptor set in GPU memory
// We need:
// - A descriptor set layout. create_descriptor_set_layout() does this
// - The data to get sent to the GPU
// - A few memory allocators
// - A device queue
pub fn push_descriptor_sets<T: Send + Sync + BufferContents>(
    sets: Vec<VGFXDescriptorSetLayoutWithData<T>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
) -> (Arc<PipelineLayout>, Vec<DescriptorSetWithOffsets>) {
    // We might only need one of these
    let host_buffer_allocator =
        Arc::new(GenericMemoryAllocator::new_default(queue.device().clone()));
    let device_buffer_allocator =
        Arc::new(GenericMemoryAllocator::new_default(queue.device().clone()));

    let mut host_buffers: Vec<Subbuffer<T>> = Vec::new();
    let mut device_buffers: Vec<Subbuffer<T>> = Vec::new();
    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    let mut descriptor_sets: Vec<DescriptorSetWithOffsets> = Vec::new();

    for set in sets {
        // Copy descriptor data into a buffer located in host memory
        // This will get copied over to device memory later
        let host_buffer = Buffer::from_data(
            host_buffer_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            set.data,
        )
        .unwrap();

        // Create a target memory buffer on the device to copy our descriptor set to
        let device_buffer: Subbuffer<T> = Buffer::new_sized(
            device_buffer_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        // Define our descriptor set
        // This is binds our descriptor layout to the region in device memory the data will end up
        // (I think)
        let descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
            set.layout.clone(),
            set.layout.variable_descriptor_count(),
            vec![WriteDescriptorSet::buffer(0, device_buffer.clone())],
            vec![],
        )
        .unwrap();

        host_buffers.push(host_buffer.clone());
        device_buffers.push(device_buffer.clone());
        descriptor_set_layouts.push(set.layout.clone());
        descriptor_sets.push(DescriptorSetWithOffsets::new(descriptor_set.clone(), []));
    }

    // Create a pipeline to copy our data from the host to the device
    let pipeline_layout = vulkano::pipeline::PipelineLayout::new(
        queue.device().clone(),
        vulkano::pipeline::layout::PipelineLayoutCreateInfo {
            flags: vulkano::pipeline::layout::PipelineLayoutCreateFlags::default(),
            set_layouts: descriptor_set_layouts,
            push_constant_ranges: Vec::new(),
            ..Default::default()
        },
    )
    .unwrap();

    // Setup a one-time command buffer to send our host buffer to the device buffer
    let mut cbb = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    for i in 0..host_buffers.len() {
        cbb.copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            host_buffers[i].clone(),
            device_buffers[i].clone(),
        ))
        .unwrap();
    }
    cbb.bind_descriptor_sets(
        vulkano::pipeline::PipelineBindPoint::Graphics,
        pipeline_layout.clone(),
        0,
        descriptor_sets.clone(),
    )
    .unwrap();
    let cb = cbb.build().unwrap();
    cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    (pipeline_layout, descriptor_sets)
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
