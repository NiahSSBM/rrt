use std::{
    collections::{BTreeMap},
    slice,
    sync::{Arc, Mutex},
};

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
use vulkano::pipeline::graphics::depth_stencil::CompareOp;
use vulkano_taskgraph::{
    ClearValues, Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    resource::{Flight, HostAccessType, Resources},
};

use crate::{
    mesh::Mesh3D,
    shader::{Shader, Vertex3D},
    vgfx::WindowContext,
};

#[derive(Clone)]
pub struct SceneTask {
    pipelines: BTreeMap<usize, Arc<GraphicsPipeline>>,
    pub buffers: BTreeMap<usize, (Shader, Id<Buffer>, Id<Buffer>)>,
}

impl SceneTask {
    // Initializes scene
    // Creates vertex and index buffers and pushes data to them
    pub fn new(
        meshes: &Vec<Arc<Mutex<Mesh3D>>>,
        resources: Arc<Resources>,
        queue: Arc<Queue>,
        flight_id: Id<Flight>,
    ) -> Self {
        let mut buffers = BTreeMap::new();

        for (i, mesh_mutex) in meshes.iter().enumerate() {
            let mesh = mesh_mutex.lock().unwrap();

            // Create buffers on device
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
                    DeviceLayout::for_value(mesh.vertices.as_slice()).unwrap(),
                )
                .unwrap();

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
                    DeviceLayout::for_value(mesh.indices.as_slice()).unwrap(),
                )
                .unwrap();

            // Write buffers to device
            unsafe {
                vulkano_taskgraph::execute(
                    &queue,
                    &resources,
                    flight_id,
                    |_cbf, tcx| {
                        tcx.write_buffer::<[Vertex3D]>(vertex_buffer_id, ..)?
                            .copy_from_slice(mesh.vertices.as_slice());
                        tcx.write_buffer::<[u32]>(index_buffer_id, ..)?
                            .copy_from_slice(mesh.indices.as_slice());

                        Ok(())
                    },
                    [
                        (vertex_buffer_id, HostAccessType::Write),
                        (index_buffer_id, HostAccessType::Write),
                    ],
                    [],
                    [],
                )
            }
            .unwrap();

            buffers.insert(i, (mesh.shader.clone(), vertex_buffer_id, index_buffer_id));
        }

        SceneTask {
            pipelines: BTreeMap::new(),
            buffers,
        }
    }

    pub fn create_pipelines(&mut self, device: Arc<Device>, subpass: Subpass, viewport: Viewport) {
        let mut pipelines = BTreeMap::new();

        for (i, buffer) in &self.buffers {
            let vs = buffer
                .0
                .stage_entries
                .get(&ShaderStage::Vertex)
                .expect("Error: No vertex shader found!");
            let fs = buffer
                .0
                .stage_entries
                .get(&ShaderStage::Fragment)
                .expect("Error: No fragment shader found!");

            let vertex_input_state = Vertex3D::per_vertex().definition(&vs).unwrap();

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
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState {
                            write_enable: true,
                            compare_op: CompareOp::Greater, // Depth buffer is reversed
                        }),
                        ..Default::default()
                    }),
                    ..GraphicsPipelineCreateInfo::layout(buffer.0.pipeline_layout.clone().unwrap())
                },
            )
            .unwrap_or_else(|err| panic!("Could not create graphics pipeline: {:?}", err));

            pipelines.insert(*i, new_pipeline);
        }
        self.pipelines = pipelines;
    }
}

impl Task for SceneTask {
    type World = WindowContext;

    // This is run every frame for our scene
    // The framebuffer is already cleared when it's attached to our scene node, so this doesn't need to do anything
    fn clear_values(&self, _clear_values: &mut ClearValues<'_>) {}

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        window_context: &Self::World,
    ) -> TaskResult {
        for (i, pipeline) in &self.pipelines {
            let shader = &self.buffers.get(&i).unwrap().0;
            let binding = shader.pipeline_layout.clone().unwrap();
            let layout = binding.as_ref();

            // Convert our DescriptorSetWithOffsets to a raw descriptor set
            // This doesn't feel safe
            let binding = shader.descriptor_sets.get(&0).unwrap().clone();
            let raw_descriptor_set = binding.as_ref().0.as_raw();

            unsafe {
                cbf.set_viewport(0, slice::from_ref(&window_context.viewport))?;
                cbf.bind_vertex_buffers(0, &[self.buffers.get(&i).unwrap().1], &[0], &[], &[])?;
                cbf.bind_index_buffer(
                    self.buffers.get(&i).unwrap().2,
                    0,
                    window_context
                        .resources
                        .buffer(self.buffers.get(&i).unwrap().2)
                        .unwrap()
                        .buffer()
                        .size(),
                    IndexType::U32,
                )?;
                cbf.as_raw().bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    layout,
                    0,
                    &[raw_descriptor_set],
                    &[],
                )?;
                cbf.bind_pipeline_graphics(&pipeline)?;

                // Draw the entire index buffer
                cbf.draw_indexed(
                    window_context
                        .resources
                        .buffer(self.buffers.get(&i).unwrap().2)
                        .unwrap()
                        .buffer()
                        .size() as u32,
                    1,
                    0,
                    0,
                    0,
                )?;
            }
        }

        Ok(())
    }
}
