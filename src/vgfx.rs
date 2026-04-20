use core::task;
use std::collections::TryReserveError;

use std::sync::{Arc, Mutex};

use color::AlphaColor;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use std::vec;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateFlags,
    QueueCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::{ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::debug::{
    DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Framebuffer;
use vulkano::shader::ShaderStage;
use vulkano::swapchain::{ColorSpace, Surface, Swapchain, SwapchainCreateInfo};
use vulkano::{Validated, VulkanError, VulkanLibrary};
use vulkano_taskgraph::graph::{
    AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, NodeId, TaskGraph,
};
use vulkano_taskgraph::resource::{
    AccessTypes, Flight, ImageLayoutType, Resources, ResourcesCreateInfo,
};
use vulkano_taskgraph::{Id, resource_map};
use winit::event_loop::ActiveEventLoop;
use winit::platform::wayland::{ActiveEventLoopExtWayland, EventLoopExtWayland};
use winit::window::Window;

use crate::game::{GameEvent, RenderEvent};
use crate::mesh::{Mesh3D, combine_vec};
use crate::scene::SceneTask;
use crate::shader::Vertex3D;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

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
    pub device: Arc<Device>,
    pub task_graph: ExecutableTaskGraph<Self>,
    pub scene_node_id: NodeId,
    pub resources: Arc<Resources>,
    pub flight_id: Id<Flight>,
    //pub vertex_buffer_allocator: Arc<StandardMemoryAllocator>,
    //pub index_buffer_allocator: Arc<StandardMemoryAllocator>,
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    pub queues: Vec<Arc<Queue>>,
    swapchain_id: Id<Swapchain>,
    virtual_framebuffer_id: Id<Framebuffer>,
    virtual_swapchain_id: Id<Swapchain>,
    surface: Arc<Surface>,
    pub meshes: Vec<Arc<Mutex<Mesh3D>>>,
    pub should_resize: bool,
    pub requested_resize: bool,
    pub last_resized: Option<Instant>,
    pub recreate_swapchain: bool,
    pub viewport: Viewport,
    pub game_thread_receiver: Option<Receiver<RenderEvent>>,
    pub game_thread_sender: Option<Sender<GameEvent>>,
    pub platform: Platform,
}

impl WindowContext {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
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

        let platform = match event_loop.is_wayland() {
            true => Platform::WAYLAND,
            false => Platform::X11,
        };

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

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

        // Create the surface fom the window provided by winit
        let surface = Surface::from_window(vulkan_instance.clone(), window.clone())
            .unwrap_or_else(|err| panic!("Could not create surface: {:?}", err));

        // Create the swapchain and images
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap_or_else(|err| panic!("Failed to get surface capabilities: {:?}", err));

        let (swapchain_format, _) = get_format_and_colorspace(device.clone(), surface.clone());

        // Initialize task graph
        let resources = Resources::new(&device, &ResourcesCreateInfo::default());
        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();
        let mut task_graph: TaskGraph<WindowContext> = TaskGraph::new(&resources.clone(), 16, 16);

        let swapchain_id = resources
            .create_swapchain(
                flight_id,
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities
                        .min_image_count
                        .max(MIN_SWAPCHAIN_IMAGES),
                    image_format: swapchain_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });

        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let scene_task = SceneTask::new(
            &vec![
                create_temp_mesh(queues[0].clone()),
                create_temp_mesh(queues[0].clone()),
            ],
            resources.clone(),
            queues.clone(),
            flight_id,
            swapchain_id.current_image_id(),
            None,
            None
        );
        let vertex_buffer_id = scene_task.vertex_buffer_id;
        let index_buffer_id = scene_task.index_buffer_id;

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                vulkano_taskgraph::QueueFamilyType::Graphics,
                scene_task,
            )
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE | AccessTypes::COLOR_ATTACHMENT_READ,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    format: swapchain_format,
                    ..Default::default()
                },
            )
            .build();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&queues[0]],
                present_queue: Some(&queues[0]),
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
            .create_pipeline(device.clone(), subpass, viewport.clone());

        // Allocators
        //let vertex_buffer_allocator =
        //    Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        //let index_buffer_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Self {
            platform,
            window,
            device,
            task_graph,
            scene_node_id,
            resources,
            flight_id,
            //vertex_buffer_allocator,
            //index_buffer_allocator,
            vertex_buffer_id,
            index_buffer_id,
            queues,
            swapchain_id,
            virtual_framebuffer_id,
            virtual_swapchain_id,
            surface,
            meshes: vec![],
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
        self.meshes.push(mesh.clone());

        Ok(self)
    }

    pub fn recreate_swapchain(&mut self) {
        self.swapchain_id = self
            .resources
            .recreate_swapchain(self.swapchain_id, |create_info| SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..create_info
            })
            .unwrap_or_else(|e| panic!("Failed to recreate swapchain: {:?}", e));

        self.viewport.extent = self.window.inner_size().into();
    }

    pub fn redraw(&mut self) {
        let flight = self.resources.flight(self.flight_id).unwrap();

        flight.wait(None).unwrap();

        let resource_map =
            resource_map!(&self.task_graph, self.virtual_swapchain_id => self.swapchain_id)
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
        let (format, _) = get_format_and_colorspace(self.device.clone(), self.surface.clone());

        self.resources
            .create_image(
                ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: ImageType::Dim2d,
                    format,
                    view_formats: Default::default(),
                    extent: self
                        .resources
                        .swapchain(self.swapchain_id)
                        .unwrap()
                        .images()[0]
                        .extent(),
                    mip_levels: 1,
                    usage: ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
    }

    pub fn update_taskgraph(&mut self) {
        let mut task_graph: TaskGraph<WindowContext> =
            TaskGraph::new(&self.resources, 16, 16);

        let (swapchain_format, _) =
            get_format_and_colorspace(self.device.clone(), self.surface.clone());

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });

        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let scene_task = SceneTask::new(
            &self.meshes,
            self.resources.clone(),
            self.queues.clone(),
            self.flight_id,
            virtual_swapchain_id.current_image_id(),
            Some(self.vertex_buffer_id),
            Some(self.index_buffer_id),
        );

        let vertex_buffer_id = scene_task.vertex_buffer_id;
        let index_buffer_id = scene_task.index_buffer_id;

        let scene_node_id = task_graph
            .create_task_node(
                "Scene",
                vulkano_taskgraph::QueueFamilyType::Graphics,
                scene_task,
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
                }
            )
            .build();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queues[0]],
                present_queue: Some(&self.queues[0]),
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
            .create_pipeline(self.device.clone(), subpass, self.viewport.clone());

        self.task_graph = task_graph;
        self.virtual_framebuffer_id = virtual_framebuffer_id;
        self.virtual_swapchain_id = virtual_swapchain_id;
        self.scene_node_id = scene_node_id;
        self.vertex_buffer_id = vertex_buffer_id;
        self.index_buffer_id = index_buffer_id;
    }
}

fn create_temp_mesh(queue: Arc<Queue>) -> Arc<Mutex<Mesh3D>> {
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
    let tri_shaders = crate::shader::Shader::new(
        stage_pipeline.clone(),
        vec![perspective.clone()],
        queue.clone(),
    );

    let colors: [AlphaColor<color::Srgb>; 3] = [
        color::palette::css::RED,
        color::palette::css::BLUE,
        color::palette::css::GREEN,
    ];
    let model_verts: Vec<Vertex3D> = vec![
        Vertex3D::new([1.0, 0.0, 0.0], colors[0]),
        Vertex3D::new([0.0, 1.0, 0.0], colors[1]),
        Vertex3D::new([0.0, 0.0, 1.0], colors[2]),
    ];
    let model_indicies: Vec<usize> = vec![0, 1, 2];

    Arc::new(Mutex::new(Mesh3D::new(
        model_verts.clone(),
        model_indicies.iter().map(|i| i.clone() as u32).collect(),
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
        verticies = combine_vec(vec![verticies, mesh.vertices.clone()]);
        indicies = combine_vec(vec![indicies, mesh.indices.clone()]);
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