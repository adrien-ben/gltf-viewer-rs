use std::mem::size_of;
use std::sync::Arc;
use vulkan::{ash::vk, create_device_local_buffer_with_data, Buffer, Context, Vertex};

use super::{create_renderer_pipeline, RendererPipelineParameters};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct QuadVertex {
    position: [f32; 2],
    coords: [f32; 2],
}

impl Vertex for QuadVertex {
    fn get_bindings_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<QuadVertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attributes_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
        ]
    }
}

pub struct QuadModel {
    pub vertices: Buffer,
    pub indices: Buffer,
}

impl QuadModel {
    pub fn new(context: &Arc<Context>) -> Self {
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );
        let vertices: [f32; 16] = [
            -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0,
        ];
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        Self { vertices, indices }
    }
}

pub fn create_fullscreen_pipeline(
    context: &Arc<Context>,
    output_format: vk::Format,
    layout: vk::PipelineLayout,
    fragment_shader_name: &'static str,
    fragment_shader_specialization: Option<&vk::SpecializationInfo>,
) -> vk::Pipeline {
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];

    create_renderer_pipeline::<QuadVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "fullscreen",
            fragment_shader_name,
            vertex_shader_specialization: None,
            fragment_shader_specialization,
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            color_attachment_formats: &[output_format],
            depth_attachment_format: None,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling: true,
            parent: None,
        },
    )
}
