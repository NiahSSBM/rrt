use bytemuck::{bytes_of, try_cast_slice};
use color::{AlphaColor, Srgb};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
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
pub struct ShaderWithDescriptors {
    pub entry_point: EntryPoint,
    pub descriptor_sets: Vec<DescriptorSetWithOffsets>,
    pub pipeline_layout: Arc<PipelineLayout>,
}

#[derive(Clone)]
pub struct Shaders {
    stage_pipeline: HashMap<ShaderStage, ShaderType>,
    queue: Arc<Queue>,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

struct VGFXDescriptorSetLayout {
    stage: ShaderStage,
    descriptor_type: DescriptorType,
    descriptor_count: u32,
}

#[derive(Clone)]
struct VGFXDescriptorSetLayoutWithData<'a>{
    layout: Arc<DescriptorSetLayout>,
    data: &'a [u8],
    size: usize,
}

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

impl Shaders {
    // Create allocators that shaders will be loaded with later
    // These allocators are used for the lifetime of the Shaders struct
    pub fn new(stage_pipeline: HashMap<ShaderStage, ShaderType>, queue: Arc<Queue>) -> Self {
        // TODO: Verify requested stages are compatible with each other
        // eg: no duplicates and vertex stage is present

        // TODO: Call self::load here with stage_pipeline
        // Possibly store some returned data for later
        Self {
            stage_pipeline,
            queue: queue.clone(),
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
        }
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

    //// Returns pipelines for all loaded shaders
    //pub fn get_pipelines(&self) -> Vec<Arc<PipelineLayout>> {
    //    let mut out = Vec::new();
    //    for shader in self.loaded.values() {
    //        out.push(shader.pipeline_layout.clone());
    //    }
    //    out
    //}

    //// Returns pipelines for shaders of the same execution model
    //// Ex. Only pipelines for vertex shaders
    //// This returns a vector, but this will only have multiple elements if shaders were loaded with Shader::load()
    //// rather than loaded from Shader::insert_loaded()
    //pub fn get_pipelines_for_model(&self, model: ExecutionModel) -> Vec<Arc<PipelineLayout>> {
    //    let mut out = Vec::new();
    //    for shader in self.loaded.values() {
    //        if model == shader.entry_point.info().execution_model {
    //            out.push(shader.pipeline_layout.clone());
    //        }
    //    }
    //    out
    //}

    //// Returns the shader of the specified type (vertex, fragment, etc...)
    //// Panics if multiple types are found
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
    // Each shader is matched to descriptor set input data
    // For each new shader, a new match leaf is required
    pub fn load(&mut self, stage_pipeline: HashMap<ShaderStage, ShaderType>) {
        let mut stage_pipeline_data: HashMap<ShaderStage, ShaderLoadData> = HashMap::new();

        let temp_color_data = vs_default::vColor {
            colors: [
                [1.0, 0.0, 0.0, 1.0].into(),
                [0.0, 1.0, 0.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
            ],
        };
        let temp_offset_data = fs_default::colorOffset { offset: 0.5 };

        for (s_stage, s_type) in stage_pipeline {
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
                    size: size_of::<[u8; 0]>(),
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

            stage_pipeline_data.insert(s_stage, load_data);
        }

        self.load_internal(self.queue.clone(), stage_pipeline_data);
    }

    // This takes some data and an input shader and puts them together on the GPU
    // Returning what needs to be attached to the shader so it can be fully rendered
    fn load_internal(
        &self,
        queue: Arc<Queue>,
        stage_pipeline_data: HashMap<ShaderStage, ShaderLoadData>,
    ) {
        // Put all stages into a vector describing the type of data we have
        // Only doing storage buffers for now
        let descriptor_set_layout_create_info = stage_pipeline_data
            .keys()
            .map(|s| VGFXDescriptorSetLayout {
                stage: *s,
                descriptor_type: DescriptorType::StorageBuffer,
                descriptor_count: 1,
            })
            .collect();

        let descriptor_set_layout =
            create_descriptor_set_layout(descriptor_set_layout_create_info, queue.device().clone())
                .unwrap();

        // Only create a descriptor set layout if we have data, otherwise it's None, and a descriptor set eventually doesn't get bound
        let descriptor_layouts_with_data: Vec<VGFXDescriptorSetLayoutWithData> = stage_pipeline_data
            .values()
            .map(|d| VGFXDescriptorSetLayoutWithData {
                layout: descriptor_set_layout.clone(),
                data: d.data,
                size: d.size,
            })
            .collect();

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

        // This might not be needed
        ShaderWithDescriptors {
            entry_point: entry_point,
            descriptor_sets: descriptor_sets,
            pipeline_layout: pipeline_layout,
        }
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
    layouts: Vec<VGFXDescriptorSetLayout>,
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
    // Enumerate all our bindings
    let mut bindings: BTreeMap<u32, DescriptorSetLayoutBinding> = BTreeMap::new();
    for i in 0..layouts.len() {
        let binding = DescriptorSetLayoutBinding {
            descriptor_count: layouts[i].descriptor_count,
            stages: layouts[i].stage.into(), // We only support 1 stage for now. ShaderStages is a superset of ShaderStage
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
fn push_descriptor_sets (
    sets: Vec<VGFXDescriptorSetLayoutWithData>,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
) -> (Arc<PipelineLayout>, Vec<DescriptorSetWithOffsets>) {
    let mut host_buffers: Vec<Subbuffer<[u8; STORAGE_BUFFER_MAX_SIZE]>> = Vec::new();
    let mut device_buffers: Vec<Subbuffer<[u8; STORAGE_BUFFER_MAX_SIZE]>> = Vec::new();
    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    let mut descriptor_sets: Vec<DescriptorSetWithOffsets> = Vec::new();

    for set in sets {
        // Data needs to have a known size
        let data: [u8; STORAGE_BUFFER_MAX_SIZE] = try_cast_slice(set.data).expect("ERROR: Storage buffer data is of different size than STORAGE_BUFFER_MAX_SIZE")[0];

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
