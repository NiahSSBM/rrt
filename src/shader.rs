use bytemuck::bytes_of;
use color::{AlphaColor, Srgb};
use nalgebra::Matrix4;
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    sync::Arc,
    vec,
};
use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{
        DescriptorSet, DescriptorSetWithOffsets, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::PipelineLayout,
    shader::{EntryPoint, ShaderStage, ShaderStages},
};
use vulkano_taskgraph::{
    Id,
    resource::{Flight, HostAccessType, Resources},
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

#[derive(Clone)]
pub enum AdditionalShaderProperties {
    // Model, View, Projection
    Perspective([[f32; 4]; 4], [[f32; 4]; 4], [[f32; 4]; 4]),
}

impl AdditionalShaderProperties {
    fn perspective_default() -> Self {
        return Self::Perspective(
            Matrix4::identity().into(),
            Matrix4::identity().into(),
            Matrix4::identity().into(),
        );
    }
}

#[derive(
    vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone, Copy,
)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

#[derive(
    vulkano::buffer::BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex, Clone, Copy,
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
    queue: Option<Arc<Queue>>,
    pub pipeline_layout: Option<Arc<PipelineLayout>>,
    pub descriptor_sets: BTreeMap<u32, DescriptorSetWithOffsets>,
    pub additional_properties: Vec<AdditionalShaderProperties>,
    descriptor_set_allocator: Option<Arc<StandardDescriptorSetAllocator>>,
    resources: Option<Arc<Resources>>,
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

// This searches the vec of properties provided and returns the first of the same type of the desired property
// desired_property can contain any data, only the type of the data is relevent
fn get_shader_property(
    desired_property: AdditionalShaderProperties,
    properties: &Vec<AdditionalShaderProperties>,
) -> Option<&AdditionalShaderProperties> {
    for potential in properties {
        if std::mem::discriminant(potential) == std::mem::discriminant(&desired_property) {
            return Some(potential);
        }
    }
    None
}

impl Shader {
    // Calling new() is NOT enough to call load(), you must call build() after calling new()
    // This is so new() can be called from the game thread with no knowledge of the graphics device
    // The game thread will then signal the render thread to call build()
    //
    // TODO: Verify requested stages are compatible with each other
    // eg: no duplicates and vertex stage is present
    pub fn new(
        stage_pipeline: HashMap<ShaderStage, ShaderType>,
        additional_properties: Vec<AdditionalShaderProperties>,
    ) -> Self {
        // Start with the values we have now
        // TODO: Re-use these allocators so they don't get re-created every time we make a new shader
        Self {
            stage_pipeline,
            stage_entries: HashMap::new(),
            queue: None,
            pipeline_layout: None,
            descriptor_sets: BTreeMap::new(),
            additional_properties,
            descriptor_set_allocator: None,
            resources: None,
        }
    }

    pub fn build(&mut self, queue: Arc<Queue>, resources: Arc<Resources>, flight_id: Id<Flight>) {
        self.queue = Some(queue.clone());
        self.resources = Some(resources);
        self.descriptor_set_allocator = Some(Arc::new(StandardDescriptorSetAllocator::new(
            queue.device().clone(),
            Default::default(),
        )));

        self.load(flight_id);
    }

    pub fn rebuild(&mut self, flight_id: Id<Flight>) {
        self.build(
            self.queue.clone().unwrap(),
            self.resources.clone().unwrap(),
            flight_id,
        );
    }

    // This is where the data inputs for each shader are defined
    // Data is seperated by bindings. If we put something in binding 0 for a specific shader,
    // other shaders can access that binding. So we need to make sure we don't overlap bindings,
    // each type of data should have it's own binding
    //
    // For each new shader, a new match leaf is required
    fn load(&mut self, flight_id: Id<Flight>) {
        if self.queue.is_none() || self.resources.is_none() {
            println!("WARNING: Queue or Resources are not initialized while loading shader.");
            return;
        }
        let queue = self.queue.clone().unwrap();
        let device = self.queue.clone().unwrap().device().clone();
        let resources = self.resources.clone().unwrap();

        let mut binding_data: BTreeMap<u32, [u8; STORAGE_BUFFER_BINDING_MAX_SIZE]> =
            BTreeMap::new();
        let mut stage_entries: HashMap<ShaderStage, EntryPoint> = HashMap::new();

        for (s_stage, s_type) in self.stage_pipeline.clone() {
            let entry: EntryPoint;
            let binding: u32;
            let data: [u8; STORAGE_BUFFER_BINDING_MAX_SIZE];

            (entry, binding, data) = match s_type {
                ShaderType::VertexDefault => {
                    let perspective = get_shader_property(
                        AdditionalShaderProperties::perspective_default(),
                        &self.additional_properties,
                    )
                    .expect("No perspective property on shader that requires perspective!");
                    (
                        vs_default::load(device.clone())
                            .unwrap()
                            .entry_point("main")
                            .unwrap(),
                        0,
                        pad(bytes_of(&vs_default::vInput {
                            mvp: {
                                match perspective {
                                    AdditionalShaderProperties::Perspective(model, view, proj) => {
                                        vs_default::MVPBuffer {
                                            model: *model,
                                            view: *view,
                                            proj: *proj,
                                        }
                                    }
                                }
                            },
                        })),
                    )
                }
                ShaderType::VertexCustom => (
                    vs_custom::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of(&vs_custom::vColor {
                        colors: [
                            [1.0, 0.0, 0.0, 1.0].into(),
                            [0.0, 1.0, 0.0, 1.0].into(),
                            [0.0, 0.0, 1.0, 1.0].into(),
                        ],
                    })),
                ),
                ShaderType::VertexWireframe => (
                    vs_wireframe::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
                ShaderType::FragmentDefault => (
                    fs_default::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    1,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
                ShaderType::FragmentCustom => (
                    fs_wireframe::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    0,
                    pad(bytes_of::<[u8; 0]>(&[])),
                ),
                ShaderType::FragmentWireframe => (
                    fs_custom::load(device.clone())
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
            self.load_internal(queue.clone(), binding_data, resources, flight_id);

        self.descriptor_sets = descriptor_sets;
        self.pipeline_layout = Some(pipeline_layout);
        self.stage_entries = stage_entries;
    }

    // This function is split off from Shader::load() because
    // it's an implemtation detail that shouldn't be worred about when adding new shaders
    fn load_internal(
        &self,
        queue: Arc<Queue>,
        binding_data: BTreeMap<u32, [u8; STORAGE_BUFFER_BINDING_MAX_SIZE]>,
        resources: Arc<Resources>,
        flight_id: Id<Flight>,
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
            self.descriptor_set_allocator.clone().unwrap(),
            queue.clone(),
            resources,
            flight_id,
        );

        (descriptor_sets, pipeline_layout)
    }

    // This is temporarily very simple to test what it takes to update descriptor data
    // This is for updating the perspective matrix
    pub fn update_descriptor(&mut self, shader_property: AdditionalShaderProperties) {
        self.additional_properties = vec![shader_property];
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
// - A memory allocators
// - A device queue
// - A flight ID
fn push_descriptor_set(
    descriptor_set_with_data: VGFXDescriptorSetLayoutWithData,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
) -> (Arc<PipelineLayout>, BTreeMap<u32, DescriptorSetWithOffsets>) {
    // Right now we only process one descriptor set layout here
    // Pipeline creation requires a vector of layouts when binding
    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    descriptor_set_layouts.push(descriptor_set_with_data.layout.clone());

    // We need to store each binding in their own buffers as they get pushed to the GPU seperately
    let mut descriptor_sets: BTreeMap<u32, DescriptorSetWithOffsets> = BTreeMap::new();
    let mut descriptor_writes: Vec<WriteDescriptorSet> = Vec::new();

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

    for (binding, _) in descriptor_set_with_data.layout.bindings() {
        // Create a buffer for each binding
        let device_buffer = resources
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(descriptor_set_with_data.data.get(binding).unwrap())
                    .unwrap(),
            )
            .unwrap();

        // Note our write to this buffer
        descriptor_writes.push(WriteDescriptorSet::buffer(
            *binding,
            Subbuffer::new(resources.buffer(device_buffer).unwrap().buffer().clone()),
        ));

        // Wait for GPU to finish flight
        resources.flight(flight_id).unwrap().wait(None).unwrap();

        // Write buffer to GPU
        unsafe {
            vulkano_taskgraph::execute(
                &queue,
                &resources,
                flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[u8]>(device_buffer, ..)
                        .unwrap()
                        .copy_from_slice(descriptor_set_with_data.data.get(binding).unwrap());

                    Ok(())
                },
                [(device_buffer, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();
    }

    // Construct a descriptor set from our device buffer
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
