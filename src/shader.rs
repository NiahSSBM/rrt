use bytemuck::{bytes_of, try_cast_slice};
use color::{AlphaColor, Srgb};
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    sync::Arc,
    vec,
};
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, DescriptorSetWithOffsets, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{Device, Queue},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    },
    pipeline::PipelineLayout,
    shader::{EntryPoint, ShaderStage, ShaderStages, spirv::ExecutionModel},
    sync::GpuFuture,
};

// Size in bytes
const STORAGE_BUFFER_MAX_SIZE: usize = 1024;

// When adding new shader types, you must add a new load leaf in Shaders::load()
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
pub struct Shader {
    stage_pipeline: HashMap<ShaderStage, ShaderType>,
    pub stage_entries: HashMap<ShaderStage, EntryPoint>,
    queue: Arc<Queue>,
    pub pipeline_layout: Option<Arc<PipelineLayout>>,
    pub descriptor_sets: HashMap<ShaderStage, BTreeMap<u32, DescriptorSetWithOffsets>>,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

struct VGFXDescriptorSetLayout {
    descriptor_type: DescriptorType,
    descriptor_count: u32,
}

#[derive(Clone)]
struct VGFXDescriptorSetLayoutWithData<'a> {
    layout: Arc<DescriptorSetLayout>,
    data: &'a [u8],
    size: usize,
}

#[derive(Clone)]
struct ShaderLoadData<'a> {
    entry: EntryPoint,
    data: &'a [u8],
    size: usize,
}

// This exists so we can use None::<NoDescriptorSet>
// It allows us to represent there isn't a descriptor set without using a struct in one of our shaders that may or may not exist
#[repr(C)]
#[derive(BufferContents, Clone)]
struct NoDescriptorSet {
    _this_value_is_intentionally_unused: i32,
}

impl Shader {
    // Create allocators that shaders will be loaded with later
    // These allocators are used for the lifetime of the Shaders struct
    pub fn new(stage_pipeline: HashMap<ShaderStage, ShaderType>, queue: Arc<Queue>) -> Self {
        // TODO: Verify requested stages are compatible with each other
        // eg: no duplicates and vertex stage is present

        let shader = Self {
            stage_pipeline,
            stage_entries: HashMap::new(),
            queue: queue.clone(),
            pipeline_layout: None,
            descriptor_sets: HashMap::new(),
            host_buffer_allocator: Arc::new(GenericMemoryAllocator::new_default(
                queue.device().clone(),
            )),
            device_buffer_allocator: Arc::new(GenericMemoryAllocator::new_default(
                queue.device().clone(),
            )),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                queue.device().clone(),
                Default::default(),
            )),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                queue.device().clone(),
                Default::default(),
            )),
        };

        shader.load()
    }

    // Takes an already loaded shader and copies it to another struct
    // TODO: Return an actual error when a shader isn't found
    //pub fn insert_loaded(&mut self, pre_loaded_shaders: &Self, s_type: ShaderType) {
    //    match self.loaded.get(&s_type) {
    //        Some(s) => {
    //            println!(
    //                "WARNING: Multiple loads for {:?} stage! Only one shader can be loaded for each stage. Skipping load...",
    //                s.entry_point.info().execution_model
    //            );
    //            return;
    //        }
    //        None => (),
    //    }

    //    self.loaded.insert(
    //        s_type.clone(),
    //        pre_loaded_shaders
    //            .loaded
    //            .get(&s_type)
    //            .cloned()
    //            .expect("Error: Shader not loaded!"),
    //    );
    //}

    // Returns descriptor sets for all loaded shaders
    //pub fn get_descriptor_sets(&self) -> Vec<DescriptorSetWithOffsets> {
    //    let mut out = Vec::new();
    //    for shader in self.loaded.values() {
    //        for descriptor in &shader.descriptor_sets {
    //            out.push(descriptor.clone());
    //        }
    //    }
    //    out
    //}

    // Returns pipelines for all loaded shaders
    //pub fn get_pipelines(&self) -> Vec<Arc<PipelineLayout>> {
    //    let mut out = Vec::new();
    //    for shader in self.loaded.values() {
    //        out.push(shader.pipeline_layout.clone());
    //    }
    //    out
    //}

    // Returns pipelines for shaders of the same execution model
    // Ex. Only pipelines for vertex shaders
    // This returns a vector, but this will only have multiple elements if shaders were loaded with Shader::load()
    // rather than loaded from Shader::insert_loaded()
    //pub fn get_pipelines_for_model(&self, model: ExecutionModel) -> Vec<Arc<PipelineLayout>> {
    //    let mut out = Vec::new();
    //    for shader in self.loaded.values() {
    //        if model == shader.entry_point.info().execution_model {
    //            out.push(shader.pipeline_layout.clone());
    //        }
    //    }
    //    out
    //}

    // TODO: Return an actual error when a shader isn't found
    // Each shader is matched to descriptor set input data
    // For each new shader, a new match leaf is required
    fn load(&self) -> Self {
        let mut stage_pipeline_data: HashMap<ShaderStage, BTreeMap<u32, ShaderLoadData>> =
            HashMap::new();
        let mut stage_entries: HashMap<ShaderStage, EntryPoint> = HashMap::new();

        let temp_color_data = vs_default::vColor {
            colors: [
                [1.0, 0.0, 0.0, 1.0].into(),
                [0.0, 1.0, 0.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
            ],
        };
        let temp_offset_data = fs_default::colorOffset { offset: 0.0 };

        for (s_stage, s_type) in self.stage_pipeline.clone() {
            let load_data = match s_type {
                ShaderType::VertexDefault => ShaderLoadData {
                    entry: vs_default::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of(&temp_color_data),
                    size: size_of::<vs_default::vColor>(),
                },
                ShaderType::VertexCustom => ShaderLoadData {
                    entry: vs_custom::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of(&temp_color_data),
                    size: size_of::<vs_default::vColor>(),
                },
                ShaderType::VertexWireframe => ShaderLoadData {
                    entry: vs_wireframe::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of::<[u8; 0]>(&[]),
                    size: size_of::<[u8; 0]>(),
                },
                ShaderType::FragmentDefault => ShaderLoadData {
                    entry: fs_default::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of(&temp_offset_data),
                    size: size_of::<fs_default::colorOffset>(),
                },
                ShaderType::FragmentWireframe => ShaderLoadData {
                    entry: fs_wireframe::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of::<[u8; 0]>(&[]),
                    size: size_of::<[u8; 0]>(),
                },
                ShaderType::FragmentCustom => ShaderLoadData {
                    entry: fs_custom::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    data: bytes_of::<[u8; 0]>(&[]),
                    size: size_of::<[u8; 0]>(),
                },
            };

            // This only supports 1 binding per shader for now
            let load_data_bindings: BTreeMap<u32, ShaderLoadData> =
                BTreeMap::from([(0, load_data.clone())]);

            stage_entries.insert(s_stage, load_data.entry.clone());
            stage_pipeline_data.insert(s_stage, load_data_bindings);
        }

        let (descriptor_sets, pipeline_layout) =
            self.load_internal(self.queue.clone(), stage_pipeline_data);

        Shader {
            descriptor_sets,
            pipeline_layout: Some(pipeline_layout),
            stage_entries: stage_entries,
            ..self.clone()
        }
    }

    // This takes some data and an input shader and puts them together on the GPU
    // Returning what needs to be attached to the shader so it can be fully rendered
    fn load_internal(
        &self,
        queue: Arc<Queue>,
        stage_pipeline_data: HashMap<ShaderStage, BTreeMap<u32, ShaderLoadData>>,
    ) -> (
        HashMap<ShaderStage, BTreeMap<u32, DescriptorSetWithOffsets>>,
        Arc<PipelineLayout>,
    ) {
        // Layouts for each stage go in a HashMap. Each layout can have multiple bindings, which go in a BTreeMap
        // Only doing storage buffers for now
        let mut descriptor_set_layout_create_info: HashMap<
            ShaderStage,
            BTreeMap<u32, VGFXDescriptorSetLayout>,
        > = HashMap::new();
        for (stage, map) in &stage_pipeline_data {
            let mut new_map: BTreeMap<u32, VGFXDescriptorSetLayout> = BTreeMap::new();
            for (binding, _) in map {
                new_map.insert(
                    *binding,
                    VGFXDescriptorSetLayout {
                        descriptor_type: DescriptorType::StorageBuffer,
                        descriptor_count: 1,
                    },
                );
            }
            descriptor_set_layout_create_info.insert(*stage, new_map);
        }

        let descriptor_set_layout =
            create_descriptor_set_layout(descriptor_set_layout_create_info, queue.device().clone())
                .unwrap();

        let mut descriptor_layouts_with_data: HashMap<
            ShaderStage,
            BTreeMap<u32, VGFXDescriptorSetLayoutWithData>,
        > = HashMap::new();
        for (stage, map) in &stage_pipeline_data {
            let mut new_map: BTreeMap<u32, VGFXDescriptorSetLayoutWithData> = BTreeMap::new();
            for (binding, data) in map {
                new_map.insert(
                    *binding,
                    VGFXDescriptorSetLayoutWithData {
                        layout: descriptor_set_layout.clone(),
                        data: data.data,
                        size: data.size,
                    },
                );
            }
            descriptor_layouts_with_data.insert(*stage, new_map);
        }

        // TODO: Find a way to use this function to put every shader on the mesh into the same pipeline
        // Currently we're creating a whole pipeline with duplicate descriptor sets for each shader
        let (pipeline_layout, descriptor_sets) = push_descriptor_sets(
            descriptor_layouts_with_data,
            self.host_buffer_allocator.clone(),
            self.device_buffer_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            queue.clone(),
        );

        (descriptor_sets, pipeline_layout)
    }
}

// To create a descriptor set layout we need:
// - The pipeline stages the descriptor set is intended for (Vertex, Fragment, All, etc...)
// - The descriptor type (StorageBuffer, StorageImage, etc...) for each descriptor
// - The descriptor count for each descriptor.
//      This one is a little confusing. A descriptor can contain either describe a single "block" of data, or an array of blocks of data.
//      The descriptor count is NOT the total number of descriptors. It's instead the number of elements within a single descriptor
//      If the data is a single element, this should be 1. If the data is an array, this is the array length.
// - The device the descriptor set is used for
fn create_descriptor_set_layout(
    layouts: HashMap<ShaderStage, BTreeMap<u32, VGFXDescriptorSetLayout>>,
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
    // Enumerate all our bindings
    let mut bindings: BTreeMap<u32, DescriptorSetLayoutBinding> = BTreeMap::new();
    for (stage, layout) in layouts {
        let binding = DescriptorSetLayoutBinding {
            descriptor_count: layout.get(&0).unwrap().descriptor_count,
            stages: ShaderStages::all_graphics(), // We only support 1 stage for now. ShaderStages is a superset of ShaderStage
            immutable_samplers: Vec::new(),
            ..DescriptorSetLayoutBinding::descriptor_type(layout.get(&0).unwrap().descriptor_type)
        };
        // Only use first binding for now. This might be completely wrong
        bindings.insert(0, binding);
    }

    // Create layout from our bindings
    let create_info = DescriptorSetLayoutCreateInfo {
        flags: Default::default(),
        bindings: bindings.clone(),
        ..Default::default()
    };

    DescriptorSetLayout::new(device.clone(), create_info)
}

fn pad(data: &[u8]) -> [u8; STORAGE_BUFFER_MAX_SIZE] {
    let mut out: [u8; STORAGE_BUFFER_MAX_SIZE] = [0; STORAGE_BUFFER_MAX_SIZE];
    for (i, byte) in data.iter().enumerate() {
        out[i] = *byte;
    }
    out
}

// This function combines a descriptor set layout and associated data, sends it to the GPU, and returns the descriptor set in GPU memory
// We need:
// - A descriptor set layout. create_descriptor_set_layout() does this
// - The data to get sent to the GPU
// - A few memory allocators
// - A device queue
fn push_descriptor_sets(
    sets: HashMap<ShaderStage, BTreeMap<u32, VGFXDescriptorSetLayoutWithData>>,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
) -> (
    Arc<PipelineLayout>,
    HashMap<ShaderStage, BTreeMap<u32, DescriptorSetWithOffsets>>,
) {
    // These Vecs might need to be hashmaps
    let mut host_buffers: Vec<Subbuffer<[u8; STORAGE_BUFFER_MAX_SIZE]>> = Vec::new();
    let mut device_buffers: Vec<Subbuffer<[u8; STORAGE_BUFFER_MAX_SIZE]>> = Vec::new();
    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    let mut descriptor_sets: HashMap<ShaderStage, BTreeMap<u32, DescriptorSetWithOffsets>> =
        HashMap::new();

    for (stage, set) in &sets {
        let mut current_descriptor_set_bindings: BTreeMap<u32, DescriptorSetWithOffsets> =
            BTreeMap::new();
        for (binding, set_layout) in set {
            // Data needs to have a known size
            let data: [u8; STORAGE_BUFFER_MAX_SIZE] = pad(set_layout.data);

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
                data,
            )
            .unwrap();

            // Create a target memory buffer on the device to copy our descriptor set to
            let device_buffer: Subbuffer<[u8; STORAGE_BUFFER_MAX_SIZE]> = Buffer::new_sized(
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
                set_layout.layout.clone(),
                set_layout.layout.variable_descriptor_count(),
                vec![WriteDescriptorSet::buffer(0, device_buffer.clone())],
                vec![],
            )
            .unwrap();
            current_descriptor_set_bindings.insert(
                *binding,
                DescriptorSetWithOffsets::new(descriptor_set.clone(), []),
            );

            host_buffers.push(host_buffer.clone());
            device_buffers.push(device_buffer.clone());
            descriptor_set_layouts.push(set.get(&binding).unwrap().layout.clone());
        }
        descriptor_sets.insert(*stage, current_descriptor_set_bindings);
    }

    // This is likely temporary
    // I'm not sure yet if we need to order descriptor sets before they get pushed to the GPU
    // When binding the descriptor sets below, the call wants a vector and not a map
    let mut descriptor_vec: Vec<DescriptorSetWithOffsets> = Vec::new();
    for (_, binding) in descriptor_sets.drain() {
        for (_, set) in binding {
            descriptor_vec.push(set);
        }
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
        descriptor_vec,
    )
    .unwrap();
    let cb = cbb.build().unwrap();

    // Command buffer finished, execute
    cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    (pipeline_layout, descriptor_sets)
}

pub mod vs_default {
    use bytemuck::NoUninit;

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_default.glsl",
        custom_derives: [NoUninit, Copy, Clone]
    }
}

pub mod fs_default {
    use bytemuck::NoUninit;

    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag_default.glsl",
        custom_derives: [NoUninit, Copy, Clone]
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
