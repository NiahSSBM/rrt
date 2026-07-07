use crate::shader_cache::ShaderCache;
use bytemuck::bytes_of;
use color::{AlphaColor, Srgb};
use image::{ImageBuffer, Rgba};
use nalgebra::Matrix4;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::Instant;
use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    sync::Arc,
    vec,
};
use vulkano::shader::ShaderModule;
use vulkano::{
    DeviceSize, Validated, VulkanError,
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
    format::Format,
    image::{
        Image, ImageAspect, ImageCreateInfo, ImageLayout, ImageTiling, ImageType, ImageUsage,
        SampleCount,
        sampler::{Sampler, SamplerCreateInfo},
        view::ImageView,
    },
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::PipelineLayout,
    shader::{EntryPoint, ShaderStage, ShaderStages},
    sync::Sharing,
};
use vulkano_taskgraph::{
    Id,
    command_buffer::CopyBufferToImageInfo,
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
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
enum DescriptorData {
    Buffer([u8; STORAGE_BUFFER_BINDING_MAX_SIZE]),
    Texture((Id<Image>, Arc<Sampler>, ImageBuffer<Rgba<u8>, Vec<u8>>)),
    Sampler(Arc<Sampler>),
}

#[derive(Clone)]
pub enum AdditionalShaderProperties {
    // Model, View, Projection
    Perspective([[f32; 4]; 4], [[f32; 4]; 4], [[f32; 4]; 4]),
    Texture(ImageBuffer<Rgba<u8>, Vec<u8>>),
}

impl AdditionalShaderProperties {
    fn perspective_default() -> Self {
        Self::Perspective(
            Matrix4::identity().into(),
            Matrix4::identity().into(),
            Matrix4::identity().into(),
        )
    }

    fn texture_default() -> Self {
        Self::Texture(image::ImageBuffer::from_fn(64, 64, |_, _| {
            image::Rgba([255, 255, 255, 255])
        }))
    }
}

#[derive(
    vulkano::buffer::BufferContents,
    vulkano::pipeline::graphics::vertex_input::Vertex,
    Clone,
    Copy,
    Debug,
)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], normal: [f32; 3], color: AlphaColor<Srgb>) -> Self {
        Self {
            position,
            normal,
            color: color.components,
        }
    }
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
    data: BTreeMap<u32, DescriptorData>,
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
    sampler: Option<Arc<Sampler>>,
    staged_texture: Option<ImageBuffer<Rgba<u8>, Vec<u8>>>,
    texture: Option<Id<Image>>,
    cache: Option<Arc<Mutex<ShaderCache>>>,
    time: Instant,
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
        shader_cache: Option<Arc<Mutex<ShaderCache>>>,
        additional_properties: Vec<AdditionalShaderProperties>,
    ) -> Self {
        Self {
            stage_pipeline,
            stage_entries: HashMap::new(),
            queue: None,
            pipeline_layout: None,
            descriptor_sets: BTreeMap::new(),
            additional_properties,
            descriptor_set_allocator: None,
            resources: None,
            sampler: None,
            staged_texture: None,
            texture: None,
            cache: shader_cache,
            time: Instant::now(),
        }
    }

    pub fn set_texture(&mut self, texture: ImageBuffer<Rgba<u8>, Vec<u8>>) {
        self.staged_texture = Some(texture);
    }

    fn load_texture(
        &mut self,
        texture: ImageBuffer<Rgba<u8>, Vec<u8>>,
        flight_id: Id<Flight>,
    ) -> (Id<Image>, Arc<Sampler>) {
        println!("loading texture");
        let sampler = self.sampler.clone().unwrap_or_else(|| {
            Sampler::new(
                self.queue.clone().unwrap().device().clone(),
                SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
            )
            .unwrap()
        });

        // Create staging buffer
        let host_buffer = self
            .resources
            .clone()
            .unwrap()
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::from_size_alignment(
                    texture.len() as DeviceSize,
                    DeviceSize::from(DeviceAlignment::of::<ImageBuffer<Rgba<u8>, Vec<u8>>>()),
                )
                .unwrap(),
            )
            .unwrap();

        if self.texture.is_some() {
            if let Ok(image_state) = self
                .resources
                .clone()
                .unwrap()
                .image(self.texture.clone().unwrap())
            {
                if image_state.image().extent()
                    != [texture.width() as u32, texture.height() as u32, 1]
                {
                    // TODO: Remove this buffer safely
                    // It is causing validation errors
                    //
                    // If there's a pre-existing texture buffer
                    // AND that buffer has a different size than what we need now
                    // Remove that buffer
                    // unsafe {
                    //     self.resources
                    //         .clone()
                    //         .unwrap()
                    //         .remove_image(self.texture.take().unwrap())
                    //         .unwrap();
                    // }
                }
            }
        }

        // Create final destination buffer
        let device_buffer = self.texture.unwrap_or_else(|| {
            self.resources
                .clone()
                .unwrap()
                .create_image(
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8G8B8A8_SRGB,
                        view_formats: vec![Format::R8G8B8A8_SRGB],
                        extent: [texture.width(), texture.height(), 1],
                        array_layers: 1,
                        mip_levels: 1,
                        samples: SampleCount::Sample1,
                        tiling: ImageTiling::Linear,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        stencil_usage: None,
                        sharing: Sharing::Exclusive,
                        initial_layout: ImageLayout::Undefined,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )
                .unwrap()
        });

        // Wait for GPU to finish flight
        self.resources
            .clone()
            .unwrap()
            .flight(flight_id)
            .unwrap()
            .wait(None)
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &self.queue.clone().unwrap(),
                &self.resources.clone().unwrap(),
                flight_id,
                |cbf, tcx| {
                    // Copy image into staging buffer
                    tcx.write_buffer::<[u8]>(host_buffer, ..)
                        .unwrap()
                        .copy_from_slice(&texture);

                    // Copy staging buffer into device buffer
                    cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: host_buffer,
                        dst_image: device_buffer,
                        ..Default::default()
                    })
                    .unwrap();
                    Ok(())
                },
                vec![(host_buffer, HostAccessType::Write)],
                vec![],
                vec![],
            )
        }
        .unwrap();

        // TODO: remove this buffer safely, or cache it for later use
        // It is causing causing validation errors
        // unsafe {
        //     match self.resources.clone().unwrap().remove_buffer(host_buffer) {
        //         Ok(_) => {}
        //         Err(e) => {
        //             println!("Failed to remove temporary host buffer: {e}")
        //         }
        //     }
        //}

        (device_buffer, sampler)
        //self.image = Some(device_buffer);
    }

    pub fn build(&mut self, queue: Arc<Queue>, resources: Arc<Resources>, flight_id: Id<Flight>) {
        if let None = self.descriptor_set_allocator {
            self.descriptor_set_allocator = Some(Arc::new(StandardDescriptorSetAllocator::new(
                queue.device().clone(),
                Default::default(),
            )))
        }
        self.resources = Some(resources);
        self.queue = Some(queue);

        // Load texture if we have one in staging
        if let Some(t) = self.staged_texture.clone() {
            let (texture, sampler) = self.load_texture(t, flight_id);
            self.texture = Some(texture);
            self.sampler = Some(sampler);
            self.staged_texture = None;
        }

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
    // Data is separated by bindings. If we put something in binding 0 for a specific shader,
    // other shaders can access that binding. So we need to make sure we don't overlap bindings,
    // each type of data should have its own binding
    //
    // For each new shader, a new match leaf is required
    fn load(&mut self, flight_id: Id<Flight>) {
        if self.queue.is_none() | self.resources.is_none() {
            println!("WARNING: Queue or Resources are not initialized while loading shader.");
            return;
        }
        let queue = self.queue.clone().unwrap();
        let device = self.queue.clone().unwrap().device().clone();
        let resources = self.resources.clone().unwrap();

        let mut binding_data: BTreeMap<u32, DescriptorData> = BTreeMap::new();
        let mut stage_entries: HashMap<ShaderStage, EntryPoint> = HashMap::new();

        for (s_stage, s_type) in self.stage_pipeline.clone() {
            let entry: EntryPoint;
            let mut data: BTreeMap<u32, DescriptorData>;

            (entry, data) = match s_type {
                ShaderType::VertexDefault => {
                    let perspective = get_shader_property(
                        AdditionalShaderProperties::perspective_default(),
                        &self.additional_properties,
                    )
                    .expect("No perspective property on shader that requires perspective!");

                    // Load from cache if available
                    let shader_module: EntryPoint;
                    if let Some(c) = &self.cache {
                        let mut lock = c.lock().unwrap();
                        if let Some(s) = lock.get_entry("vs_default") {
                            shader_module = s.clone();
                        } else {
                            shader_module = vs_default::load(device.clone())
                                .unwrap()
                                .entry_point("main")
                                .unwrap();
                            // Add to cache
                            lock.add_entry("vs_default", shader_module.clone())
                        }
                    } else {
                        shader_module = vs_default::load(device.clone())
                            .unwrap()
                            .entry_point("main")
                            .unwrap();
                    }

                    (
                        shader_module.clone(),
                        BTreeMap::from([(
                            0,
                            DescriptorData::Buffer(pad(bytes_of(&vs_default::vInput {
                                mvp: {
                                    match perspective {
                                        AdditionalShaderProperties::Perspective(
                                            model,
                                            view,
                                            proj,
                                        ) => vs_default::MVPBuffer {
                                            model: *model,
                                            view: *view,
                                            proj: *proj,
                                        },
                                        _ => panic!("This branch should never be reached"),
                                    }
                                },
                                time: self.time.elapsed().as_secs_f32(),
                            }))),
                        )]),
                    )
                }
                ShaderType::VertexCustom => (
                    vs_custom::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    BTreeMap::from([(
                        0,
                        DescriptorData::Buffer(pad(bytes_of(&vs_custom::vColor {
                            colors: [
                                [1.0, 0.0, 0.0, 1.0].into(),
                                [0.0, 1.0, 0.0, 1.0].into(),
                                [0.0, 0.0, 1.0, 1.0].into(),
                            ],
                        }))),
                    )]),
                ),
                ShaderType::VertexWireframe => (
                    vs_wireframe::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    BTreeMap::from([(0, DescriptorData::Buffer(pad(bytes_of::<[u8; 0]>(&[]))))]),
                ),
                ShaderType::FragmentDefault => {
                    let default_texture = AdditionalShaderProperties::texture_default();
                    let properties = self.additional_properties.clone(); // Probably an unnecessary clone
                    let texture = get_shader_property(
                        AdditionalShaderProperties::texture_default(),
                        &properties,
                    )
                    .unwrap_or_else(|| {
                        println!("No texture property on shader that requires a texture!");
                        &default_texture
                    });

                    // Load from cache if available
                    let shader_module: EntryPoint;
                    if let Some(c) = &self.cache {
                        let mut lock = c.lock().unwrap();
                        if let Some(s) = lock.get_entry("fs_default") {
                            shader_module = s.clone();
                        } else {
                            shader_module = fs_default::load(device.clone())
                                .unwrap()
                                .entry_point("main")
                                .unwrap();
                            // Add to cache
                            lock.add_entry("fs_default", shader_module.clone())
                        }
                    } else {
                        shader_module = fs_default::load(device.clone())
                            .unwrap()
                            .entry_point("main")
                            .unwrap();
                    }

                    (
                        shader_module,
                        BTreeMap::from([
                            (
                                2,
                                match texture {
                                    AdditionalShaderProperties::Texture(t) => {
                                        if self.texture.is_none() | self.sampler.is_none() {
                                            let (texture, sampler) =
                                                self.load_texture(t.clone(), flight_id);
                                            self.texture = Some(texture);
                                            self.sampler = Some(sampler);
                                        }
                                        DescriptorData::Texture((
                                            self.texture.clone().unwrap(),
                                            self.sampler.clone().unwrap(),
                                            t.clone(),
                                        ))
                                    }
                                    _ => panic!("This should never be reached"),
                                },
                            ),
                            (
                                1,
                                match &self.sampler {
                                    Some(s) => DescriptorData::Sampler(s.clone()),
                                    None => DescriptorData::Buffer(pad(bytes_of::<[u8; 0]>(&[]))),
                                },
                            ),
                        ]),
                    )
                }
                ShaderType::FragmentCustom => (
                    fs_wireframe::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    BTreeMap::from([(0, DescriptorData::Buffer(pad(bytes_of::<[u8; 0]>(&[]))))]),
                ),
                ShaderType::FragmentWireframe => (
                    fs_custom::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                    BTreeMap::from([(0, DescriptorData::Buffer(pad(bytes_of::<[u8; 0]>(&[]))))]),
                ),
            };

            stage_entries.insert(s_stage, entry.clone());
            binding_data.append(&mut data);
        }

        let (descriptor_sets, pipeline_layout) =
            self.load_internal(queue.clone(), binding_data, resources, flight_id);

        self.descriptor_sets = descriptor_sets;
        self.pipeline_layout = Some(pipeline_layout);
        self.stage_entries = stage_entries;
    }

    // This function is split off from Shader::load() because
    // it's an implementation detail that shouldn't be worried about when adding new shaders
    fn load_internal(
        &self,
        queue: Arc<Queue>,
        binding_data: BTreeMap<u32, DescriptorData>,
        resources: Arc<Resources>,
        flight_id: Id<Flight>,
    ) -> (BTreeMap<u32, DescriptorSetWithOffsets>, Arc<PipelineLayout>) {
        let mut descriptor_set_layout_create_info: BTreeMap<u32, VGFXDescriptorSetLayout> =
            BTreeMap::new();

        for (binding, data) in &binding_data {
            descriptor_set_layout_create_info.insert(
                *binding,
                // Set buffer type depending on input data
                match data {
                    DescriptorData::Buffer(_) => VGFXDescriptorSetLayout {
                        descriptor_type: DescriptorType::StorageBuffer,
                        descriptor_count: 1,
                    },
                    DescriptorData::Texture(_) => VGFXDescriptorSetLayout {
                        descriptor_type: DescriptorType::SampledImage,
                        descriptor_count: 1,
                    },
                    DescriptorData::Sampler(_) => VGFXDescriptorSetLayout {
                        descriptor_type: DescriptorType::Sampler,
                        descriptor_count: 1,
                    },
                },
            );
        }

        let layout =
            create_descriptor_set_layout(descriptor_set_layout_create_info, queue.device().clone())
                .unwrap();

        let descriptor_layouts_with_data = VGFXDescriptorSetLayoutWithData {
            layout,
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

    // Searches for an already existing property and replaces it
    // If there is no property, it is added
    pub fn update_descriptor(&mut self, shader_property: AdditionalShaderProperties) {
        let mut existing_index = Some(0);
        for (i, property) in self.additional_properties.iter().enumerate() {
            if std::mem::discriminant(property) == std::mem::discriminant(&shader_property) {
                existing_index = Some(i);
            }
        }

        match existing_index {
            Some(i) => {
                self.additional_properties.remove(i);
            }
            None => (),
        }
        self.additional_properties.push(shader_property);
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
        bindings,
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
                match descriptor_set_with_data.data.get(binding).unwrap() {
                    DescriptorData::Buffer(b) => DeviceLayout::for_value(b).unwrap(),
                    DescriptorData::Texture(t) => {
                        let subresource_layout = resources
                            .image(t.0)
                            .unwrap()
                            .image()
                            .subresource_layout(ImageAspect::Color, 0, 0)
                            .unwrap();
                        DeviceLayout::from_size_alignment(
                            subresource_layout.size,
                            (size_of::<u8>() * 4) as u64,
                        )
                        .unwrap()
                    }
                    DescriptorData::Sampler(_s) => DeviceLayout::for_value(&0).unwrap(),
                },
            )
            .unwrap();

        let mut host_buffer_accesses = vec![];
        let mut image_accesses = vec![];
        let buffer_accesses = vec![];

        match descriptor_set_with_data.data.get(binding).unwrap() {
            DescriptorData::Buffer(_) => {
                descriptor_writes.push(WriteDescriptorSet::buffer(
                    *binding,
                    Subbuffer::new(resources.buffer(device_buffer).unwrap().buffer().clone()),
                ));
                host_buffer_accesses = vec![(device_buffer, HostAccessType::Write)]
            }
            DescriptorData::Texture(t) => {
                descriptor_writes.push(WriteDescriptorSet::image_view(
                    *binding,
                    ImageView::new_default(resources.image(t.0).unwrap().image().clone()).unwrap(),
                ));
                host_buffer_accesses = vec![(device_buffer, HostAccessType::Write)];
                image_accesses = vec![(
                    t.0,
                    AccessTypes::COPY_TRANSFER_WRITE,
                    ImageLayoutType::General,
                )]
            }
            DescriptorData::Sampler(s) => {
                descriptor_writes.push(WriteDescriptorSet::sampler(*binding, s.clone()));
                host_buffer_accesses = vec![(device_buffer, HostAccessType::Write)];
            }
        }

        match descriptor_set_with_data.data.get(binding).unwrap() {
            DescriptorData::Buffer(data) => push_buffer(
                queue.clone(),
                resources.clone(),
                data,
                device_buffer,
                host_buffer_accesses,
                image_accesses,
                buffer_accesses,
                flight_id,
            ),
            _ => {}
        }

        // TODO: Cleanup buffer safely
        // This is causing validation errors
        //
        // Cleanup buffer
        unsafe {
            if let Err(e) = resources.remove_buffer(device_buffer) {
                println!("Failed to remove GPU buffer. This is a memory leak! {e}")
            };
        }
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

fn push_buffer(
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    data: &[u8],
    buffer: Id<Buffer>,
    host_buffer_accesses: Vec<(Id<Buffer>, HostAccessType)>,
    image_accesses: Vec<(Id<Image>, AccessTypes, ImageLayoutType)>,
    buffer_accesses: Vec<(Id<Buffer>, AccessTypes)>,
    flight_id: Id<Flight>,
) {
    // Wait for GPU to finish flight
    resources.flight(flight_id).unwrap().wait(None).unwrap();

    unsafe {
        vulkano_taskgraph::execute(
            &queue,
            &resources,
            flight_id,
            |_cbf, tcx| {
                tcx.write_buffer::<[u8]>(buffer, ..)
                    .unwrap()
                    .copy_from_slice(data);

                Ok(())
            },
            host_buffer_accesses,
            buffer_accesses,
            image_accesses,
        )
    }
    .unwrap_or_else(|e| panic!("Failed to push buffer to GPU: {e}"));
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
// desired_property can contain any data, only the type of the data is relevant
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

pub mod vs_default {
    use bytemuck::NoUninit;

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_default.glsl",
        custom_derives: [NoUninit, Copy, Clone]
    }
}

pub mod fs_default {

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
