use std::{
    slice,
    sync::{Arc, Mutex},
};

use nalgebra::U32;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexType},
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        GraphicsPipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::Subpass,
    shader::ShaderStage,
};
use vulkano_taskgraph::{
    ClearValues, Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    resource::{Flight, HostAccessType, Resources},
};

use crate::{
    mesh::{Mesh3D, combine_vec},
    shader::{Shader, Vertex3D},
    vgfx::WindowContext,
};

pub struct SceneTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertices: Vec<Vertex3D>,
    indices: Vec<u32>,
    shader: Shader,
    pub vertex_buffer_id: Id<Buffer>,
    pub index_buffer_id: Id<Buffer>,
}

impl SceneTask {
    pub fn new(
        meshes: &Vec<Arc<Mutex<Mesh3D>>>,
        resources: Arc<Resources>,
        queues: Vec<Arc<Queue>>,
        flight_id: Id<Flight>,
        vertex_buffer_id: Option<Id<Buffer>>,
        index_buffer_id: Option<Id<Buffer>>,
    ) -> Self {
        let mut vertices: Vec<Vertex3D> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut shader: Option<Shader> = None;
        for mesh_mutex in meshes {
            let mesh = mesh_mutex.lock().unwrap();
            vertices = combine_vec(vec![vertices, mesh.vertices.clone()]);
            indices = combine_vec(vec![indices, mesh.indicies.clone()]);
            shader = Some(mesh.shader.clone());
        }

        let shader = shader.expect("No meshes when creating new Scene Task!");

        println!("new scenetask");
        println!("vertices len: {}", vertices.len());

        if vertex_buffer_id.is_some() {
            unsafe { resources.remove_buffer(vertex_buffer_id.unwrap()).unwrap() };
        }

        if index_buffer_id.is_some() {
            unsafe { resources.remove_buffer(index_buffer_id.unwrap()).unwrap() };
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

        println!("Vertex buffer id: {:?}", vertex_buffer_id);

        let index_buffer_id = resources
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(indices.as_slice()).unwrap(),
            )
            .unwrap();

        println!("Vertex buffer id: {:?}", index_buffer_id);

        unsafe {
            vulkano_taskgraph::execute(
                &queues[0],
                &resources,
                flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[Vertex3D]>(vertex_buffer_id, ..)?
                        .copy_from_slice(&vertices);
                    tcx.write_buffer::<[u32]>(index_buffer_id, ..)?
                        .copy_from_slice(&indices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write), (index_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        SceneTask {
            pipeline: None,
            vertices,
            indices,
            shader,
            vertex_buffer_id,
            index_buffer_id,
        }
    }

    pub fn create_pipeline(&mut self, device: Arc<Device>, subpass: Subpass, viewport: Viewport) {
        let vs = self
            .shader
            .stage_entries
            .get(&ShaderStage::Vertex)
            .expect("Error: No vertex shader found!");
        let fs = self
            .shader
            .stage_entries
            .get(&ShaderStage::Fragment)
            .expect("Error: No fragment shader found!");

        let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();

        let _depth_stencil_state = DepthStencilState {
            depth: Some(DepthState::simple()),
            ..Default::default()
        };

        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];

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
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState::default()],
                    ..Default::default()
                }),
                subpass: Some(subpass.clone().into()),
                depth_stencil_state: None,
                ..GraphicsPipelineCreateInfo::layout(self.shader.pipeline_layout.clone().unwrap())
            },
        )
        .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));

        self.pipeline = Some(new_pipeline);
    }
}

impl Task for SceneTask {
    type World = WindowContext;

    fn clear_values(&self, _clear_values: &mut ClearValues<'_>) {
        //clear_values.set(self.bloom_image_id, [0.0; 4]);
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        window_context: &Self::World,
    ) -> TaskResult {
        let binding = self.shader.pipeline_layout.clone().unwrap();
        let layout = binding.as_ref();

        let binding = self.shader.descriptor_sets.get(&0).unwrap().clone();
        let raw_descriptor_set = binding.as_ref().0.as_raw();

        unsafe {
            cbf.set_viewport(0, slice::from_ref(&window_context.viewport))?;
            cbf.as_raw().bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                layout,
                0,
                &[raw_descriptor_set],
                &[],
            )?;
            cbf.bind_pipeline_graphics(
                self.pipeline
                    .as_ref()
                    .expect("Attempted to bind pipeline but there's no pipeline!"),
            )?;
            println!("Index buffer len: {}", self.indices.len());
            println!("Vertex buffer id: {:?}", self.vertex_buffer_id);
            cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[])?;
            cbf.bind_index_buffer(self.index_buffer_id, 0, self.indices.len() as u64, IndexType::U32);

            cbf.draw_indexed(self.indices.len() as u32, 1, 0, 0, 0)?;

            cbf.destroy_object(binding.as_ref().0.clone());
        }

        Ok(())
    }
}
