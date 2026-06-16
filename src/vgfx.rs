use std::collections::{BTreeMap, TryReserveError};

use std::sync::{Arc, Mutex};

use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::vec;
use vulkano::buffer::Buffer;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceOwned, Queue,
    QueueCreateFlags, QueueCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageTiling, ImageType, ImageUsage, SampleCount};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Framebuffer;
use vulkano::shader::ShaderStage;
use vulkano::swapchain::{
    ColorSpace, FullScreenExclusive, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
};
use vulkano::sync::Sharing;
use vulkano::{Validated, VulkanError, VulkanLibrary};
use vulkano_taskgraph::graph::{
    AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, NodeId, TaskGraph,
};
use vulkano_taskgraph::resource::{
    AccessTypes, Flight, ImageLayoutType, Resources, ResourcesCreateInfo,
};
use vulkano_taskgraph::{Id, resource_map};
use winit::dpi::PhysicalPosition;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::game::{GameEvent, RenderEvent};
use crate::mesh::{Mesh3D, Triangle};
use crate::scene::SceneTask;
use crate::shader::Vertex3D;

pub struct WindowContext {
    pub window: Arc<Window>,
    pub task_graph: ExecutableTaskGraph<Self>,
    pub scene_node_id: NodeId,
    scene_task: SceneTask,
    pub resources: Arc<Resources>,
    pub flight_id: Id<Flight>,
    buffers: BTreeMap<usize, (Id<Buffer>, Id<Buffer>)>,
    pub queue: Arc<Queue>,
    swapchain_id: Id<Swapchain>,
    depthbuffer_id: Id<Image>,
    virtual_framebuffer_id: Id<Framebuffer>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_depthbuffer_id: Id<Image>,
    swapchain_format: Format,
    pub meshes: Vec<Arc<Mutex<Mesh3D>>>,
    pub requested_resize: bool,
    pub recreate_swapchain: bool,
    pub viewport: Viewport,
    pub game_thread_handle: Option<std::thread::JoinHandle<()>>,
    pub game_thread_receiver: Receiver<RenderEvent>,
    pub game_thread_sender: Sender<GameEvent>,
    pub last_cursor_position: PhysicalPosition<f64>,
}

impl WindowContext {
    pub fn new(
        event_loop: &ActiveEventLoop,
        preferred_device: Option<String>,
        join_handle: JoinHandle<()>,
        (game_thread_sender, game_thread_receiver): (Sender<GameEvent>, Receiver<RenderEvent>),
    ) -> Self {
        // Start by initializing Vulkan
        let vulkan_libary = VulkanLibrary::new()
            .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));

        let vulkan_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..Surface::required_extensions(event_loop).unwrap_or_else(|err| {
                panic!("Could not determine required Vulkan extensions: {:?}", err)
            })
        };

        let vulkan_instance = Instance::new(
            vulkan_libary,
            InstanceCreateInfo {
                enabled_extensions: vulkan_extensions,
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("Failed to load Vulkan instance: {:?}", err));

        // Create window
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
        );

        // Create viewport
        // Viewport determines where we render to
        // Depth range is reversed for better depth precision at far distances
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 1.0..=0.0,
        };

        // Query available physical devices and select one
        let available_devices = vulkan_instance.enumerate_physical_devices().unwrap();
        for physical_device in vulkan_instance.enumerate_physical_devices().unwrap() {
            println!(
                "Available device: {}",
                physical_device.properties().device_name,
            );
        }
        let selected_device = select_device(available_devices, preferred_device)
            .expect("Could not select a device! Are there not any display devices?");
        println!(
            "Selected device: {}",
            selected_device.as_ref().properties().device_name
        );

        // Create the vulkan device and associated queues
        let (device, mut queues) = create_device(selected_device.clone())
            .unwrap_or_else(|err| panic!("Could not create graphics device: {:?}", err));
        let queue: Arc<Queue> = queues.next().expect("There are no graphics queues!");

        // Create the surface fom the window provided by winit
        let surface = Surface::from_window(vulkan_instance.clone(), window.clone())
            .unwrap_or_else(|err| panic!("Could not create surface: {:?}", err));

        // Create the swapchain and images
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap_or_else(|err| panic!("Failed to get surface capabilities: {:?}", err));
        let (swapchain_format, _) = get_format_and_colorspace(device.clone(), surface.clone());

        // Initialize resources
        // Resources contain all global resources for taskgraph
        // Only one will ever exist at any given time
        let resources = Resources::new(&device, &ResourcesCreateInfo::default());

        // Create flight, swapchain, and depth buffer
        let flight_id = resources
            .create_flight(surface_capabilities.min_image_count)
            .unwrap();
        let swapchain_id = resources
            .create_swapchain(
                flight_id,
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format: swapchain_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    present_mode: select_presentmode(device.clone(), &surface),
                    ..Default::default()
                },
            )
            .unwrap();
        let depthbuffer_id = resources
            .create_image(
                get_depthimage_createinfo(viewport.clone()),
                AllocationCreateInfo::default(),
            )
            .unwrap();

        // Create task graph
        let (
            task_graph,
            scene_node_id,
            scene_task,
            buffers,
            virtual_framebuffer_id,
            virtual_swapchain_id,
            virtual_depthbuffer_id,
        ) = create_taskgraph(
            resources.clone(),
            swapchain_format,
            queue.clone(),
            flight_id,
            viewport.clone(),
            vec![create_temp_mesh(
                queue.clone(),
                resources.clone(),
                flight_id,
            )],
        );

        Self {
            window,
            task_graph,
            scene_node_id,
            scene_task,
            resources,
            flight_id,
            buffers,
            queue,
            swapchain_id,
            depthbuffer_id,
            virtual_framebuffer_id,
            virtual_swapchain_id,
            virtual_depthbuffer_id,
            swapchain_format,
            meshes: vec![],
            requested_resize: false,
            recreate_swapchain: false,
            viewport,
            game_thread_handle: Some(join_handle),
            game_thread_receiver,
            game_thread_sender,
            last_cursor_position: PhysicalPosition::default(),
        }
    }

    pub fn add_mesh(&mut self, mesh: Arc<Mutex<Mesh3D>>) -> Result<&mut Self, TryReserveError> {
        self.meshes.try_reserve(1)?;

        mesh.lock().unwrap().shader.build(
            self.queue.clone(),
            self.resources.clone(),
            self.flight_id,
        );

        self.meshes.push(mesh.clone());

        Ok(self)
    }

    pub fn recreate_swapchain(&mut self) {
        self.viewport.extent = self.window.inner_size().into();

        self.swapchain_id = self
            .resources
            .recreate_swapchain(self.swapchain_id, |create_info| SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..create_info
            })
            .unwrap_or_else(|e| panic!("Failed to recreate swapchain: {:?}", e));
    }

    pub fn redraw(&mut self) {
        let resource_map = resource_map!(&self.task_graph,
                self.virtual_swapchain_id => self.swapchain_id,
                self.virtual_depthbuffer_id => self.depthbuffer_id)
        .unwrap();

        self.resources
            .flight(self.flight_id)
            .unwrap()
            .wait(None)
            .unwrap();

        match unsafe {
            self.task_graph
                .execute(resource_map, self, || self.window.pre_present_notify())
        } {
            Ok(()) => {}
            Err(ExecuteError::Swapchain {
                error: Validated::Error(VulkanError::OutOfDate),
                ..
            }) => {
                self.recreate_swapchain = true;
            }
            Err(e) => {
                panic!("Failed to execute next frame: {e:?}");
            }
        }
    }

    pub fn resize_window(&mut self) {
        unsafe { self.resources.remove_image(self.depthbuffer_id) }.unwrap();
        self.viewport.extent = self.window.inner_size().into();
        self.depthbuffer_id = self.resources.add_image(
            Image::new(
                Arc::new(StandardMemoryAllocator::new_default(
                    self.queue.device().clone(),
                )),
                get_depthimage_createinfo(self.viewport.clone()),
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        );
    }

    pub fn recreate_taskgraph(&mut self) {
        let mut task_graph: TaskGraph<WindowContext> = TaskGraph::new(&self.resources, 16, 16);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: self.swapchain_format,
            ..Default::default()
        });

        let virtual_framebuffer_id = task_graph.add_framebuffer();
        let virtual_depthbuffer_id =
            task_graph.add_image(&get_depthimage_createinfo(self.viewport.clone()));

        let mut new_buffers = BTreeMap::new();
        for (i, buffer) in &self.scene_task.buffers {
            let mesh = self.meshes.get(*i);

            if mesh.is_none() {
                return; // No meshes are loaded
            }

            new_buffers.insert(
                *i,
                (
                    mesh.unwrap().lock().unwrap().shader.clone(),
                    buffer.1,
                    buffer.2,
                ),
            );
        }
        self.scene_task.buffers = new_buffers;

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                vulkano_taskgraph::QueueFamilyType::Graphics,
                self.scene_task.clone(),
            )
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    format: self.swapchain_format,
                    ..Default::default()
                },
            )
            .depth_stencil_attachment(
                virtual_depthbuffer_id,
                AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    format: Format::D32_SFLOAT,
                    ..Default::default()
                },
            )
            .build();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        let scene_node = task_graph.task_node_mut(scene_node_id).unwrap();
        let subpass = scene_node.subpass().unwrap().clone();
        scene_node
            .task_mut()
            .downcast_mut::<SceneTask>()
            .unwrap()
            .create_pipelines(self.queue.device().clone(), subpass, self.viewport.clone());

        self.task_graph = task_graph;
        self.scene_node_id = scene_node_id;
        self.virtual_framebuffer_id = virtual_framebuffer_id;
        self.virtual_swapchain_id = virtual_swapchain_id;
    }

    pub fn reload_shaders(&mut self) {
        for mutex in &self.meshes {
            let mut mesh = mutex.lock().unwrap();
            mesh.shader.rebuild(self.flight_id);
        }
    }

    pub fn recreate_buffers(&mut self) {
        (self.scene_task, self.buffers) = create_buffers(
            self.resources.clone(),
            self.queue.clone(),
            self.flight_id,
            self.meshes.clone(),
        );
    }
}

fn get_depthimage_createinfo(viewport: Viewport) -> ImageCreateInfo {
    ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::D32_SFLOAT,
        view_formats: vec![Format::D32_SFLOAT],
        extent: [viewport.extent[0] as u32, viewport.extent[1] as u32, 1],
        array_layers: 1,
        mip_levels: 1,
        samples: SampleCount::Sample1,
        tiling: ImageTiling::Optimal,
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        stencil_usage: None,
        sharing: Sharing::Exclusive,
        initial_layout: Default::default(),
        drm_format_modifiers: vec![],
        drm_format_modifier_plane_layouts: vec![],
        external_memory_handle_types: Default::default(),
        ..Default::default()
    }
}

// Selects a present mode to use
// Prefers Immediate mode if available, allowing the framerate to run uncapped
// Falls back to FIFO otherwise, which is guaranteed to be supported
//
// Panics if the device and surface don't belong to the same Vulkan instance
fn select_presentmode(device: Arc<Device>, surface: &Surface) -> PresentMode {
    let surface_info = vulkano::swapchain::SurfaceInfo {
        present_mode: None,
        full_screen_exclusive: FullScreenExclusive::Default,
        win32_monitor: None,
        ..Default::default()
    };

    let supported_modes = device
        .physical_device()
        .surface_present_modes(surface, surface_info)
        .unwrap_or_else(|e| panic!("No valid present modes could be determined: {e}"));

    if supported_modes.contains(&PresentMode::Immediate) {
        PresentMode::Immediate
    } else {
        PresentMode::Fifo
    }
}

// Create buffers on device and push meshes to device memory
fn create_buffers(
    resources: Arc<Resources>,
    queue: Arc<Queue>,
    flight_id: Id<Flight>,
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
) -> (SceneTask, BTreeMap<usize, (Id<Buffer>, Id<Buffer>)>) {
    let scene_task = SceneTask::new(&meshes, resources.clone(), queue, flight_id);

    let mut buffers = BTreeMap::new();
    for (i, buffer) in &scene_task.buffers {
        buffers.insert(*i, (buffer.1, buffer.2));
    }

    (scene_task, buffers)
}

fn create_taskgraph(
    resources: Arc<Resources>,
    swapchain_format: Format,
    queue: Arc<Queue>,
    flight_id: Id<Flight>,
    viewport: Viewport,
    meshes: Vec<Arc<Mutex<Mesh3D>>>,
) -> (
    ExecutableTaskGraph<WindowContext>,
    NodeId,
    SceneTask,
    BTreeMap<usize, (Id<Buffer>, Id<Buffer>)>,
    Id<Framebuffer>,
    Id<Swapchain>,
    Id<Image>,
) {
    let mut task_graph: TaskGraph<WindowContext> = TaskGraph::new(&resources, 16, 16);

    let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
        image_format: swapchain_format,
        ..Default::default()
    });

    let virtual_framebuffer_id = task_graph.add_framebuffer();
    let virtual_depthbuffer_id = task_graph.add_image(&get_depthimage_createinfo(viewport.clone()));

    // Send meshes to device and get back buffer IDs
    let (scene_task, buffers) = create_buffers(resources, queue.clone(), flight_id, meshes);

    // Assemble our scene
    let scene_node_id = task_graph
        .create_task_node(
            "Scene",
            vulkano_taskgraph::QueueFamilyType::Graphics,
            scene_task.clone(),
        )
        .framebuffer(virtual_framebuffer_id)
        .color_attachment(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            ImageLayoutType::Optimal,
            &AttachmentInfo {
                clear: true,
                format: swapchain_format,
                ..Default::default()
            },
        )
        .depth_stencil_attachment(
            virtual_depthbuffer_id,
            AccessTypes::DEPTH_STENCIL_ATTACHMENT_READ
                | AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ImageLayoutType::Optimal,
            &AttachmentInfo {
                clear: true,
                format: Format::D32_SFLOAT,
                ..Default::default()
            },
        )
        .build();

    let mut task_graph = unsafe {
        task_graph.compile(&CompileInfo {
            queues: &[&queue],
            present_queue: Some(&queue),
            flight_id,
            ..Default::default()
        })
    }
    .unwrap();

    let scene_node = task_graph.task_node_mut(scene_node_id).unwrap();
    let subpass = scene_node.subpass().unwrap().clone();
    scene_node
        .task_mut()
        .downcast_mut::<SceneTask>()
        .unwrap()
        .create_pipelines(queue.device().clone(), subpass, viewport.clone());

    (
        task_graph,
        scene_node_id,
        scene_task,
        buffers,
        virtual_framebuffer_id,
        virtual_swapchain_id,
        virtual_depthbuffer_id,
    )
}

fn create_temp_mesh(
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
) -> Arc<Mutex<Mesh3D>> {
    let stage_pipeline = std::collections::HashMap::from([
        (
            ShaderStage::Vertex,
            crate::shader::ShaderType::VertexDefault,
        ),
        (
            ShaderStage::Fragment,
            crate::shader::ShaderType::FragmentDefault,
        ),
    ]);

    let perspective = crate::shader::AdditionalShaderProperties::Perspective(
        nalgebra::Matrix4::new_rotation(nalgebra::Vector3::new(0.0, 0.0, 0.0)).into(),
        nalgebra::Matrix4::look_at_rh(
            &nalgebra::Point3::new(4.0, 0.0, 0.0),  // Where the camera is
            &nalgebra::Point3::new(0.0, 0.0, 0.0),  // Where the camera looks
            &nalgebra::Vector3::new(0.0, 1.0, 0.0), // What axis is up
        )
        .into(),
        nalgebra::Matrix4::new_perspective(800.0 / 600.0, 800.0 / 600.0, 0.1, 10.0).into(),
    );
    let mut tri_shaders =
        crate::shader::Shader::new(stage_pipeline.clone(), None, vec![perspective.clone()]);

    tri_shaders.build(queue, resources, flight_id);

    let model_verts: Vec<Vertex3D> =
        vec![Vertex3D::new([0.0, 0.0, 0.0], color::palette::css::BLACK)];
    let triangle: Vec<Triangle> = vec![Triangle::new([1, 2, 3], [0.0, 0.0, 0.0])];

    Arc::new(Mutex::new(Mesh3D::new(
        model_verts.clone(),
        triangle,
        tri_shaders.clone(),
    )))
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
    preferred_device: Option<String>,
) -> Option<Arc<PhysicalDevice>> {
    let mut selected_device: Option<Arc<PhysicalDevice>> = None;
    let mut device_override: Option<Arc<PhysicalDevice>> = None;
    for device in devices {
        if device
            .properties()
            .device_name
            .to_ascii_lowercase()
            .contains(
                &preferred_device
                    .clone()
                    .unwrap_or_default()
                    .to_ascii_lowercase(),
            )
            && preferred_device.is_some()
        {
            device_override = Some(device.clone());
        }

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

    if device_override.is_some() {
        device_override
    } else {
        if preferred_device.is_some() {
            println!("Requested device not found...");
        }
        selected_device
    }
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
