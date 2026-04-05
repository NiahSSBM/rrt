use std::{
    slice,
    sync::{Arc, Mutex},
};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    device::Queue,
    image::Image,
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::GraphicsPipeline,
};
use vulkano_taskgraph::{
    ClearValues, Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    resource::{Flight, HostAccessType, Resources},
};
use winit::window::Window;

use crate::{
    mesh::{Mesh3D, combine_vec},
    shader::Vertex3D,
    vgfx::WindowContext,
};

pub struct SceneTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer_id: Id<Buffer>,
}

impl SceneTask {
    pub fn new(
        meshes: Vec<Arc<Mutex<Mesh3D>>>,
        resources: Arc<Resources>,
        queues: Vec<Arc<Queue>>,
        flight_id: Id<Flight>,
    ) -> Self {
        let mut vertices: Vec<Vertex3D> = vec![Vertex3D {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }];
        let mut indicies: Vec<u32> = vec![];
        for mesh_mutex in &meshes {
            let mesh = mesh_mutex.lock().unwrap();
            vertices = combine_vec(vec![vertices, mesh.verticies.clone()]);
            indicies = combine_vec(vec![indicies, mesh.indicies.clone()]);
        }

        let vertex_buffer_id = resources
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &queues[0],
                &resources,
                flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[Vertex3D]>(vertex_buffer_id, ..)?
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        SceneTask {
            pipeline: None,
            vertex_buffer_id,
        }
    }
}

impl Task for SceneTask {
    type World = WindowContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>) {
        //clear_values.set(self.bloom_image_id, [0.0; 4]);
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        window_context: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, slice::from_ref(&window_context.viewport))?;
        cbf.bind_pipeline_graphics(self.pipeline.as_ref().unwrap())?;
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[])?;

        unsafe { cbf.draw(3, 1, 0, 0) }?;

        Ok(())
    }
}
