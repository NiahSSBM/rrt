use bytemuck::bytes_of;
use color::{AlphaColor, Srgb};
use nalgebra::{Matrix, Matrix4, Point3, Vector, Vector3};
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    sync::Arc,
    vec,
};
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
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
    shader::{EntryPoint, ShaderStage, ShaderStages},
    sync::GpuFuture,
};

// Size in bytes
const STORAGE_BUFFER_BINDING_MAX_SIZE: usize = 1024;

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

#[derive(
    vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone,
)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
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

impl Vertex3D {
    pub fn new(position: [f32; 3], color: AlphaColor<Srgb>) -> Self {
        Self {
            position: position,
            color: color.components,
        }
    }
}

#[derive(Clone)]
pub struct Shader {
    pub stage_pipeline: HashMap<ShaderStage, ShaderType>,
    pub stage_entries: HashMap<ShaderStage, EntryPoint>,
    pub queue: Arc<Queue>,
    pub pipeline_layout: Option<Arc<PipelineLayout>>,
    pub descriptor_sets: BTreeMap<u32, DescriptorSetWithOffsets>,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

struct VGFXDescriptorSetLayout {
    descriptor_type: DescriptorType,
    descriptor_count: u32,
}

// DescriptorSetLayout contains each binding
// This struct contains the data that goes with each binding
#[derive(Clone)]
struct VGFXDescriptorSetLayoutWithData {
    layout: Arc<DescriptorSetLayout>,
    data: BTreeMap<u32, [u8; STORAGE_BUFFER_BINDING_MAX_SIZE]>,
}

// Takes an array of bytes and returns a sized array of max binding size
fn pad(data: &[u8]) -> [u8; STORAGE_BUFFER_BINDING_MAX_SIZE] {
    let mut out: [u8; STORAGE_BUFFER_BINDING_MAX_SIZE] = [0; STORAGE_BUFFER_BINDING_MAX_SIZE];
    for (i, byte) in data.iter().enumerate() {
        out[i] = *byte;
    }
    out
}

impl Shader {
    // TODO: Verify requested stages are compatible with each other
    // eg: no duplicates and vertex stage is present
    pub fn new(stage_pipeline: HashMap<ShaderStage, ShaderType>, queue: Arc<Queue>) -> Self {
        // Start with the values we have now
        // TODO: Re-use these allocators so they don't get re-created every time we make a new shader
        let shader = Self {
            stage_pipeline,
            stage_entries: HashMap::new(),
            queue: queue.clone(),
            pipeline_layout: None,
            descriptor_sets: BTreeMap::new(),
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

        // Shader::load() populates the rest
        shader.load()
    }

    // This is where the data inputs for each shader are defined
    // Data is seperated by bindings. If we put something in binding 0 for a specific shader,
    // other shaders can access that binding. So we need to make sure we don't overlap bindings,
    // each type of data should have it's own binding
    //
    // For each new shader, a new match leaf is required
    fn load(&self) -> Self {
        let mut binding_data: BTreeMap<u32, [u8; STORAGE_BUFFER_BINDING_MAX_SIZE]> =
            BTreeMap::new();
        let mut stage_entries: HashMap<ShaderStage, EntryPoint> = HashMap::new();

        for (s_stage, s_type) in self.stage_pipeline.clone() {
            let entry: EntryPoint;
            let binding: u32;
            let data: [u8; STORAGE_BUFFER_BINDING_MAX_SIZE];

            (entry, binding, data) = match s_type {
                ShaderType::VertexDefault => (
                    vs_default::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of(&vs_default::vInput {
                        colors: [
                            [1.0, 0.0, 0.0, 1.0].into(),
                            [0.0, 1.0, 0.0, 1.0].into(),
                            [0.0, 0.0, 1.0, 1.0].into(),
                        ],
                        mvp: vs_default::MVPBuffer {
                            model: Matrix4::new_rotation(Vector3::new(1.0, 0.8, 0.5)).into(),
                            view: Matrix4::look_at_rh(
                                &Point3::new(2.0, 2.0, 2.0),
                                &Point3::new(0.0, 0.0, 0.0),
                                &Vector3::new(0.0, 0.0, 1.0),
                            )
                            .into(),
                            proj: Matrix4::new_perspective(0.5, 800.0/600.0, 0.1, 10.0).into(),
                        },
                    })),
                ),
                ShaderType::VertexCustom => (
                    vs_custom::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of(&vs_default::vInput {
                        colors: [
                            [1.0, 0.0, 0.0, 1.0].into(),
                            [0.0, 1.0, 0.0, 1.0].into(),
                            [0.0, 0.0, 1.0, 1.0].into(),
                        ],
                        mvp: vs_default::MVPBuffer {
                            model: Matrix4::new(
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0,
                            )
                            .into(),
                            view: Matrix4::new(
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0,
                            )
                            .into(),
                            proj: Matrix4::new(
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0,
                            )
                            .into(),
                        },
                    })),
                ),
                ShaderType::VertexWireframe => (
                    vs_wireframe::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
                ShaderType::FragmentDefault => (
                    fs_default::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    1,
                    pad(bytes_of(&fs_default::colorOffset { offset: -0.5 })),
                ),
                ShaderType::FragmentCustom => (
                    fs_wireframe::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
                ShaderType::FragmentWireframe => (
                    fs_custom::load(self.queue.device().clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
            };

            stage_entries.insert(s_stage, entry.clone());
            binding_data.insert(binding, data);
        }

        let (descriptor_sets, pipeline_layout) =
            self.load_internal(self.queue.clone(), binding_data);

        Shader {
            descriptor_sets,
            pipeline_layout: Some(pipeline_layout),
            stage_entries: stage_entries,
            ..self.clone()
        }
    }

    // This function is split off from Shader::load() because
    // it's an implemtation detail that shouldn't be worred about when adding new shaders
    fn load_internal(
        &self,
        queue: Arc<Queue>,
        binding_data: BTreeMap<u32, [u8; STORAGE_BUFFER_BINDING_MAX_SIZE]>,
    ) -> (BTreeMap<u32, DescriptorSetWithOffsets>, Arc<PipelineLayout>) {
        let mut descriptor_set_layout_create_info: BTreeMap<u32, VGFXDescriptorSetLayout> =
            BTreeMap::new();
        for (binding, _) in &binding_data {
            descriptor_set_layout_create_info.insert(
                *binding,
                VGFXDescriptorSetLayout {
                    descriptor_type: DescriptorType::StorageBuffer, // Only storage buffers for now
                    descriptor_count: 1,
                },
            );
        }

        let descriptor_set_layout =
            create_descriptor_set_layout(descriptor_set_layout_create_info, queue.device().clone())
                .unwrap();

        let descriptor_layouts_with_data = VGFXDescriptorSetLayoutWithData {
            layout: descriptor_set_layout.clone(),
            data: binding_data,
        };

        let (pipeline_layout, descriptor_sets) = push_descriptor_set(
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
// - The descriptor type (StorageBuffer, StorageImage, etc...) for each descriptor
// - The descriptor count for each descriptor.
//      This one is a little confusing. A descriptor can contain either describe a single "block" of data, or an array of blocks of data.
//      The descriptor count is NOT the total number of descriptors. It's instead the number of elements within a single descriptor
//      If the data is a single element, this should be 1. If the data is an array, this is the array length.
// - The device the descriptor set is used for
fn create_descriptor_set_layout(
    layouts: BTreeMap<u32, VGFXDescriptorSetLayout>,
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
    // Enumerate all our bindings
    let mut bindings: BTreeMap<u32, DescriptorSetLayoutBinding> = BTreeMap::new();
    for (binding, layout) in layouts {
        let binding_layout = DescriptorSetLayoutBinding {
            descriptor_count: layout.descriptor_count,
            stages: ShaderStages::all_graphics(), // Every binding is accessable from each shader stage for now
            immutable_samplers: Vec::new(),
            ..DescriptorSetLayoutBinding::descriptor_type(layout.descriptor_type)
        };

        bindings.insert(binding, binding_layout);
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
fn push_descriptor_set(
    descriptor_set_with_data: VGFXDescriptorSetLayoutWithData,
    host_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    device_buffer_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
) -> (Arc<PipelineLayout>, BTreeMap<u32, DescriptorSetWithOffsets>) {
    // Right now we only process one descriptor set layout here
    // Pipeline creation requires a vector of layouts when binding
    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    descriptor_set_layouts.push(descriptor_set_with_data.layout.clone());

    // We need to store each binding in their own buffers as they get pushed to the GPU seperately
    let mut host_buffers: BTreeMap<u32, Subbuffer<[u8; 1024]>> = BTreeMap::new();
    let mut device_buffers: BTreeMap<u32, Subbuffer<[u8; 1024]>> = BTreeMap::new();

    let mut descriptor_sets: BTreeMap<u32, DescriptorSetWithOffsets> = BTreeMap::new();
    let mut descriptor_writes: Vec<WriteDescriptorSet> = Vec::new();

    // Match each descriptor set layout binding with the data we have for each binding =
    for (binding, _) in descriptor_set_with_data.layout.bindings() {
        // Create a host visible buffer with the data we have for this binding
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
            *descriptor_set_with_data.data.get(binding).unwrap(),
        )
        .unwrap();

        // Create a device visible buffer with capacity for our max buffer size
        let device_buffer: Subbuffer<[u8; STORAGE_BUFFER_BINDING_MAX_SIZE]> = Buffer::new_sized(
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

        descriptor_writes.push(WriteDescriptorSet::buffer(*binding, device_buffer.clone()));

        host_buffers.insert(*binding, host_buffer);
        device_buffers.insert(*binding, device_buffer);
    }

    let descriptor_set = DescriptorSet::new_variable(
        descriptor_set_allocator.clone(),
        descriptor_set_with_data.clone().layout,
        descriptor_set_with_data
            .clone()
            .layout
            .variable_descriptor_count(),
        descriptor_writes,
        vec![],
    )
    .unwrap();

    descriptor_sets.insert(0, DescriptorSetWithOffsets::new(descriptor_set.clone(), []));

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

    // Copy buffer for each binding
    for (binding, host_buffer) in host_buffers {
        println!("Host Buffer: {:?}", host_buffer.read());
        cbb.copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            host_buffer,
            device_buffers.get(&binding).unwrap().clone(),
        ))
        .unwrap();
        cbb.bind_descriptor_sets(
            vulkano::pipeline::PipelineBindPoint::Graphics,
            pipeline_layout.clone(),
            0,
            DescriptorSetWithOffsets::new(descriptor_set.clone(), []),
        )
        .unwrap();
    }

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
        custom_derives: [Copy, Clone, bytemuck::NoUninit]
    }
}

pub mod fs_custom {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
        custom_derives: [Copy, Clone, bytemuck::NoUninit]
    }
}

pub mod fs_wireframe {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag_wireframe.glsl",
        custom_derives: [Copy, Clone, bytemuck::NoUninit]
    }
}

pub mod vs_wireframe {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_wireframe.glsl",
        custom_derives: [Copy, Clone, bytemuck::NoUninit]
    }
}
