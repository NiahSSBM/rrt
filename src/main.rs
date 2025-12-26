use std::mem;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateFlags, QueueCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageAspects, ImageSubresourceRange, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, ColorSpace, CompositeAlpha, FullScreenExclusive, PresentMode, Surface,
    SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture, Sharing, event};
use vulkano::{Validated, VulkanError, VulkanLibrary, single_pass_renderpass};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    windows: Vec<Arc<Window>>,
    window_contexts: Vec<WindowContext>,
    resume_count: u32,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

#[derive(Default)]
struct WindowContext {
    vulkan_instance: Option<Arc<Instance>>,
    device: Option<Arc<Device>>,
    command_buffers: Option<Vec<Arc<PrimaryAutoCommandBuffer>>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    queues: Option<Vec<Arc<Queue>>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[MyVertex]>>,
    framebuffer: Option<Vec<Arc<Framebuffer>>>,
    swapchain: Option<Arc<Swapchain>>,
    images: Option<Vec<Arc<Image>>>,
    previous_fence_i: u32,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.resume_count += 1;
        if self.resume_count > 1 {
            println!(
                "Resume requested {} times, not recreating window and not resuming",
                self.resume_count
            );
            return;
        }
        assert!(self.windows.len() == 0, "Windows already exist!");
        for i in 0..self.window_contexts.len() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
            );
            self.windows.push(window);
            init_vulkano(&mut self.window_contexts[i], self.windows[i].clone());
        }
        // This locks up the thread
        //self.window.first().unwrap().pre_present_notify();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut i = 0;
        for window in &mut self.windows {
            if window_id == window.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        redraw(&mut self.window_contexts[i]);
                        window.request_redraw();
                    }
                    _ => (), //println!("Event received: {:?}", event),
                }
            }
            i += 1;
        }
    }
}

fn redraw(window_context: &mut WindowContext) {
    let queues = window_context.queues.as_ref().unwrap();
    let queue = &queues[0];
    let command_buffers = create_command_buffers(
        window_context.command_buffer_allocator.clone().unwrap(),
        queue.clone(),
        window_context.pipeline.clone().unwrap(),
        &window_context.framebuffer.clone().unwrap(),
        window_context.vertex_buffer.clone().unwrap(),
    );
    let swapchain = window_context.swapchain.clone().unwrap();
    let images = window_context.images.clone().unwrap();

    let (image_i, suboptimal, acquire_future) =
        match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok(r) => r,
            Err(err) => panic!("Failed to acquire next image: {err}"),
        };

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
            //recreate_swapchain = true;
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
            ..Default::default()
        },
        enabled_features: Default::default(),
        ..Default::default()
    };
    return Device::new(physical_device, device_create_info);
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
    let swapchain_create_info = SwapchainCreateInfo {
        flags: Default::default(),
        min_image_count: capabilities.min_image_count,
        image_format: Format::R8G8B8A8_SRGB,
        image_view_formats: Default::default(),
        image_color_space: ColorSpace::SrgbNonLinear,
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
    return Swapchain::new(device, surface, swapchain_create_info);
}

fn create_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
}

fn create_command_buffers(
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: Subbuffer<[MyVertex]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap_or_else(|err| panic!("Could not create framebuffer: {:?}", err));

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|err| panic!("Could not begin render pass: {:?}", err))
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap_or_else(|err| panic!("Could not bind graphics pipeline: {:?}", err))
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap_or_else(|err| panic!("Could not bind vertex buffers: {:?}", err));

            // Draw functions are marked as unsafe in vulkano as shader safety needs to be followed
            // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
            unsafe {
                builder
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap_or_else(|err| panic!("Could not draw: {:?}", err));
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
            },
        },
        pass: {
            color: [rp],
            depth_stencil: {},
        },
    )
}

fn create_image_views(swapchain_images: Vec<Arc<Image>>) -> Vec<Arc<ImageView>> {
    swapchain_images
        .iter()
        .map(|image| {
            ImageView::new(
                image.clone(),
                ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    format: Format::R8G8B8A8_SRGB,
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
        .collect()
}

fn create_frame_buffer(
    render_pass: Arc<RenderPass>,
    image_views: Vec<Arc<ImageView>>,
) -> Vec<Arc<Framebuffer>> {
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

fn init_vulkano(window_context: &mut WindowContext, window: Arc<Window>) {
    let window_context = window_context;
    let window = window.clone();
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

    // Create image view
    let image_views = create_image_views(swapchain_images.clone());
    image_views
        .iter()
        .for_each(|image_view| println!("Successfully created image veiw"));

    // Create frame buffer
    let framebuffer = create_frame_buffer(render_pass.clone(), image_views);
    window_context.framebuffer = Some(framebuffer.clone());
    println!("Successfully created framebuffer");

    // Create graphics pipeline
    let vs = vs::load(device.clone()).expect("Failed to create vertex shader module!");
    let fs = fs::load(device.clone()).expect("Failed to create fragment shader module!");
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    let pipeline = create_pipeline(device.clone(), vs, fs, render_pass, viewport)
        .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));
    window_context.pipeline = Some(pipeline.clone());
    println!("Successfully created graphics pipeline");

    // Create vertex buffer
    let vertex1 = MyVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = MyVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = MyVertex {
        position: [0.5, -0.25],
    };
    let vertex_memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let vertex_buffer = Buffer::from_iter(
        vertex_memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )
    .unwrap_or_else(|err| panic!("Could not create vertex buffer: {:?}", err));
    window_context.vertex_buffer = Some(vertex_buffer.clone());

    // Create command buffer
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    window_context.command_buffer_allocator = Some(command_buffer_allocator.clone());

    let command_buffers = create_command_buffers(
        command_buffer_allocator,
        queues[0].clone(),
        pipeline,
        &framebuffer,
        vertex_buffer,
    );
    println!("Successfully created command buffer");
    window_context.command_buffers = Some(command_buffers);
}

fn main() {
    let event_loop = EventLoop::new()
        .unwrap_or_else(|err| panic!("Couldn't create window event loop: {:?}", err));
    event_loop.set_control_flow(ControlFlow::Poll);

    let vulkan_libary = VulkanLibrary::new()
        .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
    let vulkan_extensions = Surface::required_extensions(&event_loop)
        .unwrap_or_else(|err| panic!("Could not determine required Vulkan extensions: {:?}", err));

    let vulkan_instance = Instance::new(
        vulkan_libary,
        InstanceCreateInfo {
            enabled_extensions: vulkan_extensions,
            ..Default::default()
        },
    )
    .unwrap_or_else(|err| panic!("Failed to load Vulkan instance: {:?}", err));

    let mut app = App {
        window_contexts: vec![
            WindowContext {
                vulkan_instance: Some(vulkan_instance.clone()),
                ..Default::default()
            },
            WindowContext {
                // Creating a second window context just as a test for now
                vulkan_instance: Some(vulkan_instance),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    event_loop.run_app(&mut app).unwrap_or_else(|err| {
        panic!(
            "Event loop couldn't be created or exited with and error: {:?}",
            err
        )
    });
}
