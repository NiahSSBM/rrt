use std::collections::TryReserveError;

use std::sync::{Arc, Mutex};

use color::AlphaColor;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use std::vec;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBuffer, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferUsage, PrimaryAutoCommandBuffer, RecordingCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateFlags,
    QueueCreateInfo,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::ImageLayout::{self, PresentSrc};
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageAspects, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::{GraphicsPipelineCreateInfo, vertex_input};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{self, Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderStage;
use vulkano::swapchain::{
    self, ColorSpace, CompositeAlpha, FullScreenExclusive, PresentMode, Surface,
    SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture, Sharing};
use vulkano::{Validated, VulkanError, VulkanLibrary, single_pass_renderpass};
use vulkano_taskgraph::graph::{ExecutableTaskGraph, TaskGraph};
use vulkano_taskgraph::resource::{Resources, ResourcesCreateInfo};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::wayland::{ActiveEventLoopExtWayland, EventLoopExtWayland};
use winit::window::{self, Window};

use crate::game::{GameEvent, RenderEvent};
use crate::mesh::{Mesh3D, combine_vec};
use crate::shader::Vertex3D;

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

pub struct WindowContext {
    pub window: Arc<Window>,
    pub vulkan_instance: Arc<Instance>,
    pub device: Arc<Device>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    //task_graph: ExecutableTaskGraph<Self>,
    resources: Arc<Resources>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub vertex_buffer_allocator: Arc<StandardMemoryAllocator>,
    pub index_buffer_allocator: Arc<StandardMemoryAllocator>,
    image_allocator: Arc<StandardMemoryAllocator>,
    pub queues: Vec<Arc<Queue>>,
    pipelines: Vec<Arc<GraphicsPipeline>>,
    vertex_buffer: Subbuffer<[Vertex3D]>,
    index_buffer: Subbuffer<[u32]>,
    depth_buffer: Arc<ImageView>,
    framebuffer: Vec<Arc<Framebuffer>>,
    swapchain: Arc<Swapchain>,
    surface: Arc<Surface>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    pub meshes: Vec<Arc<Mutex<Mesh3D>>>,
    previous_fence_i: u32,
    pub should_resize: bool,
    pub requested_resize: bool,
    pub last_resized: Option<Instant>,
    pub recreate_swapchain: bool,
    viewport: Viewport,
    pub game_thread_receiver: Option<Receiver<RenderEvent>>,
    pub game_thread_sender: Option<Sender<GameEvent>>,
    pub platform: Platform,
}

impl WindowContext {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
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

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
        );

        // Query available physical devices and select one
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

        let resources = Resources::new(&device, &ResourcesCreateInfo::default());

        // Create the surface fom the window provided by winit
        let surface = Surface::from_window(vulkan_instance.clone(), window.clone())
            .unwrap_or_else(|err| panic!("Could not create surface: {:?}", err));

        // Create the swapchain and images
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap_or_else(|err| panic!("Failed to get surface capabilities: {:?}", err));
        let (swapchain, images) =
            create_swapchain(device.clone(), surface.clone(), surface_capabilities)
                .unwrap_or_else(|err| panic!("Could not create swapchain: {:?}", err));

        // Initialize task graph
        let mut task_graph: TaskGraph<WindowContext> = TaskGraph::new(&resources.clone(), 16, 16);

        // Allocators
        let image_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let vertex_buffer_allocator =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let index_buffer_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create depth buffer
        let depth_buffer =
            create_depth_image_view(image_allocator.clone(), window.inner_size().into())
                .unwrap_or_else(|err| panic!("Could not create depth buffer: {:?}", err));

        // Create render pass
        let render_pass =
            create_render_pass(device.clone(), swapchain.image_format(), Format::D16_UNORM)
                .unwrap_or_else(|err| panic!("Could not create render pass: {:?}", err));

        // Create frame buffer
        let framebuffer = create_frame_buffer(
            render_pass.clone(),
            &images,
            depth_buffer.clone(),
            swapchain.image_format(),
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        // Create pipelines
        let pipelines = create_pipelines(device.clone(), vec![], render_pass.clone(), viewport.clone());

        let (vertex_buffer, index_buffer) = update_vertex_buffer(
            vec![],
            vertex_buffer_allocator.clone(),
            index_buffer_allocator.clone(),
        );

        // Create command buffers
        let command_buffers = create_command_buffers(
            &vertex_buffer,
            &index_buffer,
            framebuffer.clone(),
            &pipelines,
            &queues,
            &vec![],
            command_buffer_allocator.clone(),
        );

        Self {
            vulkan_instance: vulkan_instance,
            platform: platform,
            window,
            device,
            command_buffers,
            //task_graph,
            resources,
            command_buffer_allocator,
            vertex_buffer_allocator,
            index_buffer_allocator,
            image_allocator,
            queues: queues,
            pipelines,
            vertex_buffer,
            index_buffer,
            depth_buffer,
            framebuffer,
            swapchain,
            surface,
            images,
            render_pass,
            meshes: vec![],
            previous_fence_i: 0,
            should_resize: false,
            requested_resize: false,
            last_resized: None,
            recreate_swapchain: false,
            viewport,
            game_thread_receiver: None,
            game_thread_sender: None,
        }
    }

    pub fn add_mesh(&mut self, mesh: Arc<Mutex<Mesh3D>>) -> Result<&mut Self, TryReserveError> {
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
        window_context.device.clone(),
        window_context.surface.clone(),
    );

    // Recreate our swapchain using the previous one, only changing the size, colorspace, and image format if applicable
    let (new_swapchain, new_images) = window_context
        .swapchain
        .recreate(SwapchainCreateInfo {
            image_extent: window_context.window.inner_size().into(),
            image_color_space: colorspace,
            image_format: format,
            ..window_context.swapchain.create_info()
        })
        .unwrap_or_else(|err| panic!("Failed to create new swapchain: {:?}", err));

    // This might not be necessary
    let depth_buffer = create_depth_image_view(
        window_context.image_allocator.clone(),
        [
            window_context.viewport.extent[0] as u32,
            window_context.viewport.extent[1] as u32,
        ],
    )
    .unwrap_or_else(|err| panic!("Could not create depth buffer: {:?}", err));
    window_context.depth_buffer = depth_buffer.clone();

    let new_framebuffers = create_frame_buffer(
        window_context.render_pass.clone(),
        &new_images,
        depth_buffer,
        format,
    );

    window_context.swapchain = new_swapchain;
    window_context.framebuffer = new_framebuffers;
    window_context.images = new_images;
}

pub fn resize_window(window_context: &mut WindowContext) {
    window_context.viewport.extent = window_context.window.inner_size().into();
    window_context.pipelines = create_pipelines(
        window_context.device.clone(),
        vec![],
        window_context.render_pass.clone(),
        window_context.viewport.clone(),
    );
    if window_context.pipelines.is_empty() {
        println!(
            "Warning: No pipelines were created when resizing window! Are there any meshes to draw?"
        );
    }

    let new_command_buffers = create_command_buffers(
        &window_context.vertex_buffer,
        &window_context.index_buffer,
        window_context.framebuffer.clone(),
        &window_context.pipelines,
        &window_context.queues,
        &window_context.meshes,
        window_context.command_buffer_allocator.clone(),
    );
    window_context.command_buffers = new_command_buffers;
}

pub fn redraw(window_context: &mut WindowContext) {
    let queues = &window_context.queues;
    let queue = &queues[0];
    let command_buffers = &window_context.command_buffers;
    let swapchain = window_context.swapchain.clone();
    let images = window_context.images.clone();

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
            let mut now = sync::now(window_context.device.clone());
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

    Device::new(physical_device, device_create_info)
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

fn create_depth_image_view(
    allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    extent: [u32; 2],
) -> Result<Arc<ImageView>, Validated<VulkanError>> {
    ImageView::new_default(
        Image::new(
            allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
}

// TODO:
// Cache already created shaders so they don't need their own pipeline
// Batch host/device buffer copies so it's not copied for every mesh
// Execute the command buffer once, instead of for each mesh
fn create_pipelines(
    device: Arc<Device>,
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Vec<Arc<GraphicsPipeline>> {
    let mut completed_pipelines: Vec<Arc<GraphicsPipeline>> = Vec::new();

    if meshes.is_empty() {
        // Not fatal, a default mesh with 1 vertex is created later in this case
        println!("Warning: No meshes to load!");
    }

    for mesh_mutex in &meshes {
        let mesh = mesh_mutex.lock().unwrap();
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

        let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();

        let depth_stencil_state = DepthStencilState {
            depth: Some(DepthState::simple()),
            ..Default::default()
        };

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let new_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport.clone()].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                depth_stencil_state: Some(depth_stencil_state),
                // Our pipeline is pre-computed and is attached to our shader on our mesh
                ..GraphicsPipelineCreateInfo::layout(mesh.shader.pipeline_layout.clone().unwrap())
            },
        )
        .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));

        completed_pipelines.push(new_pipeline);
    }

    completed_pipelines
}

fn create_recording_command_buffer(
    vertex_buffer: &Subbuffer<[Vertex3D]>,
    index_buffer: &Subbuffer<[u32]>,
    frame_buffer: Vec<Arc<Framebuffer>>,
    pipelines: &Vec<Arc<GraphicsPipeline>>,
    queues: &Vec<Arc<Queue>>,
    meshes: &Vec<Arc<Mutex<Mesh3D>>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) -> Vec<Arc<CommandBuffer>> {
    frame_buffer
        .iter()
        .map(|framebuffer| {
            // Start recording a command buffer
            let mut builder = RecordingCommandBuffer::new(
                command_buffer_allocator.clone(),
                queues[0].queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo::default(),
            )
            .unwrap_or_else(|err| panic!("Could not create command buffer: {:?}", err));

            // Begin render pass
            unsafe {
                builder
                    .begin_render_pass(
                        &RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 0.0, 1.0].into()), // Background color
                                Some(ClearValue::Depth(1.0)),      // Depth buffer
                            ],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        &SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap_or_else(|err| panic!("Could not begin render pass: {:?}", err));
            }

            // Bind the shader pipeline, verticies, and descriptor sets for each mesh
            let mut vertex_slice_start: u64;
            let mut vertex_slice_end: u64 = 0;
            let mut index_slice_start: u64;
            let mut index_slice_end: u64 = 0;
            for (i, mesh_mutex) in meshes.iter().enumerate() {
                let mesh = mesh_mutex.lock().unwrap();
                vertex_slice_start = vertex_slice_end;
                vertex_slice_end = vertex_slice_start + mesh.verticies.len() as u64;
                index_slice_start = index_slice_end;
                index_slice_end = index_slice_start + mesh.indicies.len() as u64;

                // Slices must be within buffers
                assert!(vertex_buffer.size() >= vertex_slice_end * size_of::<Vertex3D>() as u64);
                assert!(index_buffer.size() >= index_slice_end * size_of::<u32>() as u64);

                let vertex_buffer_slice = vertex_buffer
                    .clone()
                    .slice(vertex_slice_start..vertex_slice_end);

                let (descriptor_set, offsets) =
                    mesh.shader.descriptor_sets.get(&0).unwrap().as_ref();

                unsafe {
                    builder
                        .bind_pipeline_graphics(pipelines[i].as_ref())
                        .unwrap_or_else(|err| panic!("Could not bind graphics pipeline: {:?}", err))
                        .bind_vertex_buffers(0, &[vertex_buffer_slice.into_bytes()])
                        .unwrap_or_else(|err| panic!("Could not bind vertex buffers: {:?}", err))
                        .bind_index_buffer(&IndexBuffer::U32(index_buffer.clone()))
                        .unwrap_or_else(|err| panic!("Could not bind index buffers: {:?}", err))
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipelines[i].layout(),
                            0,
                            &[descriptor_set.as_raw()],
                            offsets,
                        )
                        .unwrap_or_else(|err| panic!("Could not bind descriptor sets: {:?}", err));
                }

                // SAFETY:
                // Draw functions are marked as unsafe in vulkano as shader safety needs to be followed
                // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
                unsafe {
                    // We have access to the entire vertex buffer, but should only draw the verticies for the mesh who's shader this pipeline represents
                    // Right now this assumes every mesh has the same number of verticies
                    builder
                        .draw_indexed(
                            mesh.indicies.len() as u32,
                            1,
                            index_slice_start as u32,
                            0,
                            0,
                        )
                        .unwrap_or_else(|err| panic!("Could not draw: {:?}", err));
                }
            }

            unsafe {
                builder
                    .end_render_pass(&Default::default())
                    .unwrap_or_else(|err| panic!("Could not end render pass: {:?}", err));
            }

            // Stop recording
            unsafe { Arc::new(builder.end().unwrap()) }
        })
        .collect()
}

fn create_command_buffers(
    vertex_buffer: &Subbuffer<[Vertex3D]>,
    index_buffer: &Subbuffer<[u32]>,
    frame_buffer: Vec<Arc<Framebuffer>>,
    pipelines: &Vec<Arc<GraphicsPipeline>>,
    queues: &Vec<Arc<Queue>>,
    meshes: &Vec<Arc<Mutex<Mesh3D>>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    frame_buffer
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queues[0].queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap_or_else(|err| panic!("Could not create command buffer: {:?}", err));

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()), // Background color
                            Some(ClearValue::Depth(1.0)),      // Depth buffer
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|err| panic!("Could not begin render pass: {:?}", err));

            // Bind the shader pipeline, verticies, and descriptor sets for each mesh
            let mut vertex_slice_start: u64;
            let mut vertex_slice_end: u64 = 0;
            let mut index_slice_start: u64;
            let mut index_slice_end: u64 = 0;
            for (i, mesh_mutex) in meshes.iter().enumerate() {
                let mesh = mesh_mutex.lock().unwrap();
                vertex_slice_start = vertex_slice_end;
                vertex_slice_end = vertex_slice_start + mesh.verticies.len() as u64;
                index_slice_start = index_slice_end;
                index_slice_end = index_slice_start + mesh.indicies.len() as u64;

                // Slices must be within buffers
                assert!(vertex_buffer.size() >= vertex_slice_end * size_of::<Vertex3D>() as u64);
                assert!(index_buffer.size() >= index_slice_end * size_of::<u32>() as u64);

                let vertex_buffer_slice = vertex_buffer
                    .clone()
                    .slice(vertex_slice_start..vertex_slice_end);

                builder
                    .bind_pipeline_graphics(pipelines[i].clone())
                    .unwrap_or_else(|err| panic!("Could not bind graphics pipeline: {:?}", err))
                    .bind_vertex_buffers(0, vertex_buffer_slice.clone())
                    .unwrap_or_else(|err| panic!("Could not bind vertex buffers: {:?}", err))
                    .bind_index_buffer(index_buffer.clone())
                    .unwrap_or_else(|err| panic!("Could not bind index buffers: {:?}", err))
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipelines[i].layout().clone(),
                        0,
                        mesh.shader.descriptor_sets.get(&0).unwrap().clone(),
                    )
                    .unwrap_or_else(|err| panic!("Could not bind descriptor sets: {:?}", err));

                // SAFETY:
                // Draw functions are marked as unsafe in vulkano as shader safety needs to be followed
                // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
                unsafe {
                    // We have access to the entire vertex buffer, but should only draw the verticies for the mesh who's shader this pipeline represents
                    // Right now this assumes every mesh has the same number of verticies
                    builder
                        .draw_indexed(
                            mesh.indicies.len() as u32,
                            1,
                            index_slice_start as u32,
                            0,
                            0,
                        )
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
    primary_format: Format,
    depth_format: Format,
) -> Result<Arc<RenderPass>, Validated<vulkano::VulkanError>> {
    single_pass_renderpass!(
        device,
        attachments: {
            rp: {
                format: primary_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
                initial_layout: PresentSrc,
                final_layout: PresentSrc
            },
            ds: {
                format: depth_format,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal
            }
        },
        pass: {
            color: [rp],
            depth_stencil: {ds},
        },
    )
}

fn create_frame_buffer(
    render_pass: Arc<RenderPass>,
    images: &Vec<Arc<Image>>,
    depth_buffer: Arc<ImageView>,
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
                    attachments: vec![image.clone(), depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap_or_else(|err| panic!("Could not create frame buffer: {:?}", err))
        })
        .collect()
}

pub fn update_vertex_buffer(
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
    vertex_buffer_allocator: Arc<StandardMemoryAllocator>,
    index_buffer_allocator: Arc<StandardMemoryAllocator>,
) -> (Subbuffer<[Vertex3D]>, Subbuffer<[u32]>) {
    // Fill the vertex and index buffers with all of our models
    let mut verticies: Vec<Vertex3D> = vec![];
    let mut indicies: Vec<u32> = vec![];
    for mesh_mutex in &meshes {
        let mesh = mesh_mutex.lock().unwrap();
        verticies = combine_vec(vec![verticies, mesh.verticies.clone()]);
        indicies = combine_vec(vec![indicies, mesh.indicies.clone()]);
    }

    // Buffer::from_iter will panic if there's no verticies or indicies, so we'll make one if either are empty
    if verticies.is_empty() {
        verticies.push(Vertex3D::new([0.0, 0.0, 0.0], AlphaColor::WHITE));
    }
    if indicies.is_empty() {
        indicies.push(0);
    }
    let vertex_buffer = Buffer::from_iter(
        vertex_buffer_allocator,
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

    let index_buffer = Buffer::from_iter(
        index_buffer_allocator,
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indicies,
    )
    .unwrap_or_else(|err| panic!("Could not create index buffer: {:?}", err));

    //update_pipelines(window_context);
    (vertex_buffer, index_buffer)
}

// This is called when we only need to update shaders and perspective
pub fn update_pipelines(window_context: &mut WindowContext) {
    window_context.pipelines = create_pipelines(
        window_context.device.clone(),
        window_context.meshes.clone(),
        window_context.render_pass.clone(),
        window_context.viewport.clone(),
    );
    window_context.command_buffers = create_command_buffers(
        &window_context.vertex_buffer,
        &window_context.index_buffer,
        window_context.framebuffer.clone(),
        &window_context.pipelines,
        &window_context.queues,
        &window_context.meshes,
        window_context.command_buffer_allocator.clone(),
    );
}
