use std::collections::BTreeMap;

use std::sync::Arc;

use std::vec;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, DescriptorSetWithOffsets, WriteDescriptorSet};
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
    AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::{PipelineLayoutCreateFlags, PipelineLayoutCreateInfo};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::swapchain::{
    self, ColorSpace, CompositeAlpha, FullScreenExclusive, PresentMode, Surface,
    SurfaceCapabilities, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture, Sharing};
use vulkano::{Validated, VulkanError, VulkanLibrary, single_pass_renderpass};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::vs::vColor;

#[derive(Default)]
struct App {
    //windows: Vec<Arc<Window>>,
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
    window: Option<Arc<Window>>,
    vulkan_instance: Option<Arc<Instance>>,
    device: Option<Arc<Device>>,
    command_buffers: Option<Vec<Arc<PrimaryAutoCommandBuffer>>>,
    command_buffer_allocator: Option<Arc<StandardCommandBufferAllocator>>,
    queues: Option<Vec<Arc<Queue>>>,
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[MyVertex]>>,
    device_buffer: Option<Subbuffer<vColor>>,
    descriptor_sets: Option<DescriptorSetWithOffsets>,
    framebuffer: Option<Vec<Arc<Framebuffer>>>,
    swapchain: Option<Arc<Swapchain>>,
    images: Option<Vec<Arc<Image>>>,
    render_pass: Option<Arc<RenderPass>>,
    vs: Option<Arc<ShaderModule>>,
    fs: Option<Arc<ShaderModule>>,
    previous_fence_i: u32,
    resized: bool,
    recreate_swapchain: bool,
    viewport: Viewport,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
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
        for i in 0..self.window_contexts.len() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
            );
            self.window_contexts[i].window = Some(window);
            init_vulkano(&mut self.window_contexts[i]);
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
        for window_context in &mut self.window_contexts {
            let window = window_context.window.clone().unwrap();
            if window_id == window.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        if window_context.resized || window_context.recreate_swapchain {
                            window_context.recreate_swapchain = false;
                            recreate_swapchain(window_context);

                            if window_context.resized {
                                window_context.resized = false;
                                resize_window(window_context);
                            }
                        }

                        redraw(window_context);
                        window.request_redraw();
                    }
                    WindowEvent::Resized(_size) => {
                        window_context.resized = true;
                    }
                    _ => (), //println!("Event received: {:?}", event),
                }
            }
        }
    }
}

fn recreate_swapchain(window_context: &mut WindowContext) {
    let (new_swapchain, new_images) = window_context
        .swapchain
        .as_ref()
        .unwrap()
        .recreate(SwapchainCreateInfo {
            image_extent: window_context.window.as_ref().unwrap().inner_size().into(),
            ..window_context.swapchain.as_ref().unwrap().create_info()
        })
        .unwrap_or_else(|err| panic!("Failed to create new swapchain: {:?}", err));

    window_context.swapchain = Some(new_swapchain);
    let new_framebuffers =
        create_frame_buffer(window_context.render_pass.clone().unwrap(), new_images);
    window_context.framebuffer = Some(new_framebuffers);
}

fn resize_window(window_context: &mut WindowContext) {
    window_context.viewport.extent = window_context.window.as_ref().unwrap().inner_size().into();
    let new_pipeline = create_pipeline(window_context)
        .unwrap_or_else(|err| panic!("Failed to create new pipeline: {:?}", err));
    window_context.pipeline = Some(new_pipeline.clone());
    let new_command_buffers = create_command_buffers(window_context);
    window_context.command_buffers = Some(new_command_buffers);
}

fn redraw(window_context: &mut WindowContext) {
    let queues = window_context.queues.as_ref().unwrap();
    let queue = &queues[0];
    let command_buffers = window_context.command_buffers.as_ref().unwrap();
    let swapchain = window_context.swapchain.clone().unwrap();
    let images = window_context.images.clone().unwrap();

    let (image_i, suboptimal, acquire_future) =
        match swapchain::acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok(r) => r,
            Err(err) => panic!("Failed to acquire next image: {err}"),
        };

    if suboptimal {
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
            ..Default::default()
        },
        enabled_features: DeviceFeatures {
            //descriptor_binding_update_unused_while_pending: true,
            ..Default::default()
        },
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
    window_context: &mut WindowContext,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let device = window_context.device.as_ref().unwrap();
    let vs = window_context
        .vs
        .as_ref()
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = window_context
        .fs
        .as_ref()
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let mut descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>> = Vec::new();
    let mut bindings: BTreeMap<u32, DescriptorSetLayoutBinding> = BTreeMap::new();
    let binding = DescriptorSetLayoutBinding {
        descriptor_count: 1,
        stages: ShaderStages::all_graphics(),
        immutable_samplers: Vec::new(),
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
    };
    for i in 0..stages.len() {
        bindings.insert(i as u32, binding.clone());
        let create_info = DescriptorSetLayoutCreateInfo {
            flags: Default::default(),
            bindings: bindings.clone(),
            ..Default::default()
        };
        let layout = DescriptorSetLayout::new(device.clone(), create_info).unwrap();
        descriptor_set_layouts.push(layout);
    }
    let data = vs::vColor {
        colors: [[1.0, 0.0, 0.0].into(),
                [0.0, 1.0, 0.0].into(),
                [0.0, 0.0, 1.0].into()]
    };

    let allocator = Arc::new(GenericMemoryAllocator::new_default(device.clone()));
    let host_buffer = Buffer::from_data(
        allocator.clone(),
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

    let device_buffer: Subbuffer<vColor> = Buffer::new_sized(
        allocator.clone(),
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

    let allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    ));
    let descriptor_set = DescriptorSet::new_variable(
        allocator,
        descriptor_set_layouts[0].clone(),
        descriptor_set_layouts[0].variable_descriptor_count(),
        vec![WriteDescriptorSet::buffer(0, device_buffer.clone())],
        vec![],
    )
    .unwrap();
    let descriptor_sets = DescriptorSetWithOffsets::new(descriptor_set, []);

    window_context.descriptor_sets = Some(descriptor_sets.clone());
    window_context.device_buffer = Some(device_buffer.clone());

    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            flags: PipelineLayoutCreateFlags::default(),
            set_layouts: descriptor_set_layouts,
            push_constant_ranges: Vec::new(),
            ..Default::default()
        },
    )
    .unwrap();

    let mut cbb = AutoCommandBufferBuilder::primary(
        window_context
            .command_buffer_allocator
            .as_ref()
            .unwrap()
            .clone(),
        window_context.queues.as_ref().unwrap()[0].queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    cbb.copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone()))
        .unwrap();
    cbb.bind_descriptor_sets(
        PipelineBindPoint::Graphics,
        pipeline_layout.clone(),
        0,
        descriptor_sets,
    )
    .unwrap();
    let cb = cbb.build().unwrap();
    cb.execute(window_context.queues.as_ref().unwrap()[0].clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let subpass = Subpass::from(window_context.render_pass.as_ref().unwrap().clone(), 0).unwrap();

    GraphicsPipeline::new(
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
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        },
    )
}

fn create_command_buffers(window_context: &WindowContext) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let pipeline = window_context.pipeline.as_ref().unwrap();
    let vertex_buffer = window_context.vertex_buffer.as_ref().unwrap();
    window_context
        .framebuffer
        .clone()
        .unwrap()
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

            // Bind descriptor sets created earlier for this window/context
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    window_context
                        .descriptor_sets
                        .as_ref()
                        .expect("Descriptor sets not created for window context")
                        .clone(),
                )
                .unwrap_or_else(|err| panic!("Could not bind descriptor sets: {:?}", err));

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
    images: Vec<Arc<Image>>,
) -> Vec<Arc<Framebuffer>> {
    let image_views: Vec<Arc<ImageView>> = images
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

fn init_vulkano(window_context: &mut WindowContext) {
    let window_context = window_context;
    let window = window_context.window.clone().unwrap();
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
    window_context.render_pass = Some(render_pass.clone());

    // Create frame buffer
    let framebuffer = create_frame_buffer(render_pass.clone(), swapchain_images.clone());
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

    // Create graphics pipeline
    let vs = vs::load(device.clone()).expect("Failed to create vertex shader module!");
    let fs = fs::load(device.clone()).expect("Failed to create fragment shader module!");

    window_context.vs = Some(vs.clone());
    window_context.fs = Some(fs.clone());
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    window_context.viewport = viewport.clone();
    let pipeline = create_pipeline(window_context)
        .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));
    window_context.pipeline = Some(pipeline.clone());
    println!("Successfully created graphics pipeline");

    // Create vertex buffer
    let vertex1 = MyVertex {
        position: [0.5, 0.5],
    };
    let vertex2 = MyVertex {
        position: [0.0, -0.5],
    };
    let vertex3 = MyVertex {
        position: [-0.5, 0.5],
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

    let command_buffers = create_command_buffers(window_context);
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
            //WindowContext {
            //    // Creating a second window context just as a test for now
            //    vulkan_instance: Some(vulkan_instance.clone()),
            //    ..Default::default()
            //},
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
