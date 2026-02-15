use std::collections::TryReserveError;

use std::sync::Arc;

use color::AlphaColor;
use std::sync::mpsc::Receiver;
use std::time::Instant;
use std::vec;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateFlags,
    QueueCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::ImageLayout::PresentSrc;
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageAspects, ImageSubresourceRange, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderStage;
use vulkano::shader::spirv::ExecutionModel;
use vulkano::swapchain::{
    self, ColorSpace, CompositeAlpha, FullScreenExclusive, PresentMode, Surface,
    SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture, Sharing};
use vulkano::{Validated, VulkanError, VulkanLibrary, single_pass_renderpass};
use winit::event_loop::EventLoop;
use winit::platform::wayland::EventLoopExtWayland;
use winit::window::Window;

use crate::game::RenderEvent;
use crate::mesh::{Mesh, combine_verticies};
use crate::shader::Vertex2D;

#[derive(Default, PartialEq)]
pub enum Platform {
    //ANDROID,
    //IOS,
    //MACOS,
    //ORBITAL,
    WAYLAND,
    //WEB,
    //WINDOWS,
    X11,
    #[default]
    UNKNOWN,
}

#[derive(Default)]
pub struct WindowContext {
    pub window: Option<Arc<Window>>,
    pub vulkan_instance: Option<Arc<Instance>>,
    pub device: Option<Arc<Device>>,
    command_buffers: Option<Vec<Arc<PrimaryAutoCommandBuffer>>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    vertex_buffer_allocator: Option<Arc<GenericMemoryAllocator<FreeListAllocator>>>,
    pub queues: Option<Vec<Arc<Queue>>>,
    pipelines: Vec<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[Vertex2D]>>,
    framebuffer: Option<Vec<Arc<Framebuffer>>>,
    swapchain: Option<Arc<Swapchain>>,
    surface: Option<Arc<Surface>>,
    images: Option<Vec<Arc<Image>>>,
    render_pass: Option<Arc<RenderPass>>,
    meshes: Vec<Mesh>,
    previous_fence_i: u32,
    pub should_resize: bool,
    pub requested_resize: bool,
    pub last_resized: Option<Instant>,
    pub recreate_swapchain: bool,
    viewport: Viewport,
    pub game_thread_receiver: Option<Receiver<RenderEvent>>,
    pub platform: Platform,
}

impl WindowContext {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let vulkan_libary = VulkanLibrary::new()
            .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
        let vulkan_extensions = Surface::required_extensions(event_loop).unwrap_or_else(|err| {
            panic!("Could not determine required Vulkan extensions: {:?}", err)
        });

        let vulkan_instance = Instance::new(
            vulkan_libary,
            InstanceCreateInfo {
                enabled_extensions: vulkan_extensions,
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("Failed to load Vulkan instance: {:?}", err));

        let platform = match event_loop.is_wayland() {
            true => Platform::WAYLAND,
            false => Platform::X11,
        };

        Self {
            vulkan_instance: Some(vulkan_instance.clone()),
            platform: platform,
            ..Default::default()
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> Result<&mut Self, TryReserveError> {
        self.meshes.try_reserve(1)?;
        self.meshes.push(mesh);

        Ok(self)
    }
}

// A swapchain gets recreated when the window is resized, the previous swapchain image reported as suboptimal,
// or fetching the current swapchain image failed for whatever reason
pub fn recreate_swapchain(window_context: &mut WindowContext) {
    // Make sure we have a compatible image format and colorspace for the new swapchain and framebuffer before creating them
    let (format, colorspace) = get_format_and_colorspace(
        window_context.device.clone().unwrap(),
        window_context.surface.clone().unwrap(),
    );

    // Recreate our swapchain using the previous one, only changing the size, colorspace, and image format if applicable
    let (new_swapchain, new_images) = window_context
        .swapchain
        .as_ref()
        .unwrap()
        .recreate(SwapchainCreateInfo {
            image_extent: window_context.window.as_ref().unwrap().inner_size().into(),
            image_color_space: colorspace,
            image_format: format,
            ..window_context.swapchain.as_ref().unwrap().create_info()
        })
        .unwrap_or_else(|err| panic!("Failed to create new swapchain: {:?}", err));

    let new_framebuffers = create_frame_buffer(
        window_context.render_pass.clone().unwrap(),
        &new_images,
        format,
    );

    window_context.swapchain = Some(new_swapchain);
    window_context.framebuffer = Some(new_framebuffers);
    window_context.images = Some(new_images);
}

pub fn resize_window(window_context: &mut WindowContext) {
    window_context.viewport.extent = window_context.window.as_ref().unwrap().inner_size().into();
    window_context.pipelines = create_pipelines(window_context);
    if window_context.pipelines.is_empty() {
        println!(
            "Warning: No pipelines were created when resizing window! Are there any meshes to draw?"
        );
    }

    let new_command_buffers = create_command_buffers(window_context);
    window_context.command_buffers = Some(new_command_buffers);
}

pub fn redraw(window_context: &mut WindowContext) {
    let queues = window_context.queues.as_ref().unwrap();
    let queue = &queues[0];
    let command_buffers = window_context.command_buffers.as_ref().unwrap();
    let swapchain = window_context.swapchain.clone().unwrap();
    let images = window_context.images.clone().unwrap();

    let (image_i, suboptimal, acquire_future) =
        match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok(r) => r,
            Err(err) => {
                // Should be non-fatal
                // We just don't draw using this swapchain, and it will get regenerated next frame
                println!("WARNING: Failed to acquire next swapchain image: {}", err);

                // In testing, this branch only happens when resizing the window when using X11
                // Where the swapchain would get regenerated because it's being resized
                // Here we force a swapchain recreation too, which might be unnecessary, but it's safe
                window_context.recreate_swapchain = true;
                return;
            }
        };

    if suboptimal {
        println!("Suboptimal detected");
        window_context.recreate_swapchain = true;
    }

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];

    if let Some(image_fence) = &fences[image_i as usize] {
        image_fence.wait(None).unwrap();
    }

    let previous_future = match fences[window_context.previous_fence_i as usize].clone() {
        None => {
            let mut now = sync::now(window_context.device.clone().unwrap());
            now.cleanup_finished();
            now.boxed()
        }
        Some(fence) => fence.boxed(),
    };

    let future = previous_future
        .join(acquire_future)
        .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
        .unwrap()
        .then_swapchain_present(
            queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
        )
        .then_signal_fence_and_flush();

    fences[image_i as usize] = match future.map_err(Validated::unwrap) {
        Ok(value) => Some(Arc::new(value)),
        Err(VulkanError::OutOfDate) => {
            window_context.recreate_swapchain = true;
            None
        }
        Err(e) => {
            println!("failed to flush future: {e}");
            None
        }
    };

    window_context.previous_fence_i = image_i;
}

fn get_device_total_memory(device: &Arc<PhysicalDevice>) -> u64 {
    let mut total_memory = 0;
    let heaps = &device.memory_properties().memory_heaps;
    let heap_types = &device.memory_properties().memory_types;
    for heap in heap_types {
        if heap
            .property_flags
            .contains(MemoryPropertyFlags::DEVICE_LOCAL ^ MemoryPropertyFlags::HOST_VISIBLE)
        {
            total_memory += heaps[heap.heap_index as usize].size;
        }
    }
    total_memory
}

fn select_device(
    devices: impl ExactSizeIterator<Item = Arc<PhysicalDevice>>,
) -> Option<Arc<PhysicalDevice>> {
    let mut selected_device: Option<Arc<PhysicalDevice>> = None;
    for device in devices {
        selected_device = match selected_device {
            Some(device) => {
                if device.properties().device_type == PhysicalDeviceType::DiscreteGpu
                    && device.properties().device_type == PhysicalDeviceType::DiscreteGpu
                {
                    if get_device_total_memory(&device) > get_device_total_memory(&device) {
                        Some(device)
                    } else {
                        Some(device)
                    }
                } else if device.properties().device_type == PhysicalDeviceType::DiscreteGpu {
                    Some(device)
                } else {
                    Some(device)
                }
            }
            None => Some(device),
        };
    }
    selected_device
}

fn create_device(
    physical_device: Arc<PhysicalDevice>,
) -> Result<(Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>), Validated<VulkanError>> {
    let device_create_info = DeviceCreateInfo {
        queue_create_infos: vec![QueueCreateInfo {
            flags: QueueCreateFlags::default(),
            ..Default::default()
        }],
        enabled_extensions: DeviceExtensions {
            khr_swapchain: true,
            khr_fragment_shader_barycentric: true,
            ..Default::default()
        },
        enabled_features: DeviceFeatures {
            fragment_shader_barycentric: true,
            ..Default::default()
        },
        ..Default::default()
    };
    return Device::new(physical_device, device_create_info);
}

// This gets a compatible image format and colorspace for the given surface
// We're not always guaranteed to use our prefered format and colorspace, and it can change mid run
// When we re-create swapchains, this is called to make sure our new swapchain is using compatible parameters
fn get_format_and_colorspace(device: Arc<Device>, surface: Arc<Surface>) -> (Format, ColorSpace) {
    // Query supported surface formats
    let formats = match device
        .physical_device()
        .surface_formats(&surface, Default::default())
    {
        Ok(f) => f,
        Err(e) => {
            panic!("Failed to query surface formats: {:?}", e);
        }
    };

    // Prefer B8G8R8A8_SRGB and SrgbNonLinear if available
    // Otherwise just pick the first one
    formats
        .iter()
        .find(|(fmt, cs)| *fmt == Format::B8G8R8A8_SRGB && *cs == ColorSpace::SrgbNonLinear)
        .cloned()
        .unwrap_or_else(|| formats[0])
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
    capabilities: SurfaceCapabilities,
) -> Result<
    (
        Arc<vulkano::swapchain::Swapchain>,
        Vec<Arc<vulkano::image::Image>>,
    ),
    Validated<VulkanError>,
> {
    let (format, colorspace) = get_format_and_colorspace(device.clone(), surface.clone());

    let swapchain_create_info = SwapchainCreateInfo {
        flags: Default::default(),
        min_image_count: capabilities.min_image_count,
        image_format: format,
        image_view_formats: Default::default(),
        image_color_space: colorspace,
        //TODO: image_extent should be the same size as the window
        image_extent: [800, 600],
        image_array_layers: 1,
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        image_sharing: Sharing::Exclusive,
        pre_transform: Default::default(),
        composite_alpha: CompositeAlpha::Opaque,
        present_mode: PresentMode::Fifo,
        present_modes: Default::default(),
        clipped: Default::default(),
        scaling_behavior: Default::default(),
        present_gravity: Default::default(),
        full_screen_exclusive: FullScreenExclusive::Default,
        win32_monitor: Default::default(),
        ..Default::default()
    };

    Swapchain::new(device, surface, swapchain_create_info)
}

// TODO:
// Cache already created shaders so they don't need their own pipeline
// Batch host/device buffer copies so it's not copied for every mesh
// Execute the command buffer once, instead of for each mesh
fn create_pipelines(window_context: &mut WindowContext) -> Vec<Arc<GraphicsPipeline>> {
    let device = window_context.device.as_ref().unwrap();
    let mut completed_pipelines: Vec<Arc<GraphicsPipeline>> = Vec::new();

    if window_context.meshes.is_empty() {
        // Not fatal, a default mesh with 1 vertex is created later in this case
        println!("Warning: No meshes to load!");
    }

    for mesh in &mut window_context.meshes {
        let vs = mesh
            .shader
            .stage_entries
            .get(&ShaderStage::Vertex)
            .expect("Error: No vertex shader found!");
        let fs = mesh
            .shader
            .stage_entries
            .get(&ShaderStage::Fragment)
            .expect("Error: No fragment shader found!");

        let vertex_input_state = Vertex2D::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];

        let subpass =
            Subpass::from(window_context.render_pass.as_ref().unwrap().clone(), 0).unwrap();

        let new_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [window_context.viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                // Our pipeline is pre-computed and is attached to our shader on our mesh
                ..GraphicsPipelineCreateInfo::layout(mesh.shader.pipeline_layout.clone().unwrap())
            },
        )
        .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));

        completed_pipelines.push(new_pipeline);
    }

    completed_pipelines
}

fn create_command_buffers(window_context: &WindowContext) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let pipelines = &window_context.pipelines;
    let vertex_buffer = window_context
        .vertex_buffer
        .as_ref()
        .expect("ERROR: There's no vertex buffer!");
    window_context
        .framebuffer
        .clone()
        .expect("ERROR: There's no frame buffer!")
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                window_context
                    .command_buffer_allocator
                    .as_ref()
                    .unwrap()
                    .clone(),
                window_context.queues.as_ref().unwrap()[0].queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap_or_else(|err| panic!("Could not create framebuffer: {:?}", err));

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())], // Background color
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|err| panic!("Could not begin render pass: {:?}", err));

            // Bind the shader pipeline, verticies, and descriptor sets for each mesh
            // Currently this assumes each mesh is only 1 tri
            for (i, mesh) in window_context.meshes.iter().enumerate() {
                let vertex_buffer_slice = vertex_buffer
                    .clone()
                    .slice((3 * i) as u64..(3 * (i + 1)) as u64);

                // This is likely temporary
                // I'm not sure yet if we need to order descriptor sets before they get pushed to the GPU
                // When binding the descriptor sets below, the call wants a vector and not a map
                let mut descriptor_vec: Vec<vulkano::descriptor_set::DescriptorSetWithOffsets> =
                    Vec::new();
                for (_, binding) in mesh.shader.descriptor_sets.clone().drain() {
                    for (_, set) in binding {
                        descriptor_vec.push(set);
                    }
                }
                println!("Descriptor len: {}", descriptor_vec.len());

                builder
                    .bind_pipeline_graphics(pipelines[i].clone())
                    .unwrap_or_else(|err| panic!("Could not bind graphics pipeline: {:?}", err))
                    .bind_vertex_buffers(0, vertex_buffer_slice.clone())
                    .unwrap_or_else(|err| panic!("Could not bind vertex buffers: {:?}", err))
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipelines[i].layout().clone(),
                        0,
                        descriptor_vec,
                    )
                    .unwrap_or_else(|err| panic!("Could not bind descriptor sets: {:?}", err));

                // SAFETY:
                // Draw functions are marked as unsafe in vulkano as shader safety needs to be followed
                // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
                unsafe {
                    // We have access to the entire vertex buffer, but should only draw the verticies for the mesh who's shader this pipeline represents
                    // Right now this assumes every mesh has the same number of verticies
                    builder
                        .draw(vertex_buffer_slice.len() as u32, 1, 0, 0)
                        .unwrap_or_else(|err| panic!("Could not draw: {:?}", err));
                }
            }

            builder
                .end_render_pass(Default::default())
                .unwrap_or_else(|err| panic!("Could not end render pass: {:?}", err));

            builder.build().unwrap()
        })
        .collect()
}

fn create_render_pass(
    device: Arc<Device>,
    swapchain: Arc<Swapchain>,
) -> Result<Arc<RenderPass>, Validated<vulkano::VulkanError>> {
    single_pass_renderpass!(
        device,
        attachments: {
            rp: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
                initial_layout: PresentSrc,
                final_layout: PresentSrc
            },
        },
        pass: {
            color: [rp],
            depth_stencil: {},
        },
    )
}

fn create_frame_buffer(
    render_pass: Arc<RenderPass>,
    images: &Vec<Arc<Image>>,
    format: Format,
) -> Vec<Arc<Framebuffer>> {
    let image_views: Vec<Arc<ImageView>> = images
        .iter()
        .map(|image| {
            ImageView::new(
                image.clone(),
                ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    format: format,
                    component_mapping: ComponentMapping::identity(),
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        mip_levels: (0..1),
                        array_layers: (0..1),
                    },
                    usage: ImageUsage::COLOR_ATTACHMENT,
                    sampler_ycbcr_conversion: None,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|err| panic!("Could not create image view from image: {:?}", err))
        })
        .collect();

    image_views
        .iter()
        .map(|image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image.clone()],
                    ..Default::default()
                },
            )
            .unwrap_or_else(|err| panic!("Could not create frame buffer: {:?}", err))
        })
        .collect()
}

pub fn init_vulkano(window_context: &mut WindowContext) {
    let window_context = window_context;
    let window = window_context
        .window
        .clone()
        .expect("Error: Window needs to be set BEFORE vulkan is initialized.");
    let vulkan_instance = window_context
        .vulkan_instance
        .as_ref()
        .expect("Attempted to initialize vulkan with no vulkan instance!")
        .clone();

    // Queue available physical devices and select one
    let available_devices = vulkan_instance.enumerate_physical_devices().unwrap();
    for physical_device in vulkan_instance.enumerate_physical_devices().unwrap() {
        println!(
            "Available device: {}",
            physical_device.properties().device_name,
        );
    }
    let selected_device = select_device(available_devices)
        .expect("Could not select a device! Are there not any display devices?");
    println!(
        "Selected device: {}",
        selected_device.as_ref().properties().device_name
    );

    // Create the vulkan device and associated queues
    let (device, queues) = create_device(selected_device.clone())
        .unwrap_or_else(|err| panic!("Could not create graphics device: {:?}", err));
    let queues: Vec<Arc<Queue>> = queues.collect();
    window_context.queues = Some(queues.clone());
    window_context.device = Some(device.clone());
    println!("Successfully created graphics device");

    // Create the surface fom the window provided by winit
    let surface = Surface::from_window(vulkan_instance.clone(), window.clone())
        .unwrap_or_else(|err| panic!("Could not create surface: {:?}", err));
    window_context.surface = Some(surface.clone());
    println!("Successfully created surface");

    // Create the swapchain and images
    let surface_capabilities = selected_device
        .surface_capabilities(&surface, Default::default())
        .unwrap_or_else(|err| panic!("Failed to get surface capabilities: {:?}", err));
    let (swapchain, swapchain_images) =
        create_swapchain(device.clone(), surface, surface_capabilities)
            .unwrap_or_else(|err| panic!("Could not create swapchain: {:?}", err));
    window_context.swapchain = Some(swapchain.clone());
    window_context.images = Some(swapchain_images.clone());
    println!("Successfully created swapchain");

    // Create render pass
    let render_pass = create_render_pass(device.clone(), swapchain.clone())
        .unwrap_or_else(|err| panic!("Could not create render pass: {:?}", err));
    println!("Successfully created render pass");
    window_context.render_pass = Some(render_pass.clone());

    // Create frame buffer
    let framebuffer = create_frame_buffer(
        render_pass.clone(),
        &swapchain_images,
        swapchain.image_format(),
    );
    window_context.framebuffer = Some(framebuffer.clone());
    println!("Successfully created framebuffer");

    // Create command buffer
    // This is intended to be the only command buffer used in the window, which will get shared around whatever needs it
    // It's first used in create_pipeline(), so it needs to be defined before then
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    window_context.command_buffer_allocator = Some(command_buffer_allocator.clone());

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    window_context.viewport = viewport.clone();

    window_context.pipelines = create_pipelines(window_context);
    if window_context.pipelines.is_empty() {
        // This might mean there's no meshes in the pipeline
        println!("Warning: No pipelines were created when initializing!");
    }
    println!("Successfully created graphics pipeline");

    // Put all verticies of all meshes into the vertex buffer
    update_vertex_buffer(window_context);

    let command_buffers = create_command_buffers(window_context);
    println!("Successfully created command buffer");
    window_context.command_buffers = Some(command_buffers);
}

pub fn update_vertex_buffer(window_context: &mut WindowContext) {
    // Create new vertext buffer allocator if there isn't one already
    if window_context.vertex_buffer_allocator.is_none() {
        println!("No vertex buffer alloctor found, creating new one...");
        let vertex_memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            window_context.device.as_ref().unwrap().clone(),
        ));
        window_context.vertex_buffer_allocator = Some(vertex_memory_allocator.clone());
    }

    // Put all verticies in all meshes in the vertex buffer
    let mut verticies = vec![];
    for mesh in &window_context.meshes {
        verticies = combine_verticies(vec![verticies, mesh.verticies.clone()])
    }

    // Buffer::from_iter will panic if there's no verticies, so we'll make one vertex if there isn't one
    if verticies.is_empty() {
        verticies.push(Vertex2D::new([0.0, 0.0], AlphaColor::WHITE));
    }
    let vertex_buffer = Buffer::from_iter(
        window_context
            .vertex_buffer_allocator
            .as_ref()
            .unwrap()
            .clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        verticies,
    )
    .unwrap_or_else(|err| panic!("Could not create vertex buffer: {:?}", err));
    window_context.vertex_buffer = Some(vertex_buffer);
    window_context.pipelines = create_pipelines(window_context);
    window_context.command_buffers = Some(create_command_buffers(window_context));
}
