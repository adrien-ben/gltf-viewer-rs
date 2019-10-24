mod model;
mod skybox;

pub use self::{model::*, skybox::*};
use ash::vk;
use std::sync::Arc;
use vulkan::*;

#[derive(Copy, Clone)]
struct RendererPipelineParameters<'a> {
    shader_name: &'static str,
    vertex_shader_specialization: Option<&'a vk::SpecializationInfo>,
    fragment_shader_specialization: Option<&'a vk::SpecializationInfo>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    depth_stencil_info: &'a vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment: &'a vk::PipelineColorBlendAttachmentState,
    enable_face_culling: bool,
    parent: Option<vk::Pipeline>,
}

fn create_renderer_pipeline<V: Vertex>(
    context: &Arc<Context>,
    params: RendererPipelineParameters,
) -> vk::Pipeline {
    let vertex_shader_params = params
        .vertex_shader_specialization
        .map_or(ShaderParameters::new(params.shader_name), |s| {
            ShaderParameters::specialized(params.shader_name, s)
        });
    let fragment_shader_params = params
        .fragment_shader_specialization
        .map_or(ShaderParameters::new(params.shader_name), |s| {
            ShaderParameters::specialized(params.shader_name, s)
        });

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(params.msaa_samples)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: params.swapchain_properties.extent.width as _,
        height: params.swapchain_properties.extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: params.swapchain_properties.extent,
    }];
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let cull_mode = if params.enable_face_culling {
        vk::CullModeFlags::BACK
    } else {
        vk::CullModeFlags::NONE
    };

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    create_pipeline::<V>(
        context,
        PipelineParameters {
            vertex_shader_params,
            fragment_shader_params,
            multisampling_info: &multisampling_info,
            viewport_info: &viewport_info,
            rasterizer_info: &rasterizer_info,
            dynamic_state_info: None,
            depth_stencil_info: Some(params.depth_stencil_info),
            color_blend_attachment: params.color_blend_attachment,
            render_pass: params.render_pass,
            layout: params.layout,
            parent: params.parent,
            allow_derivatives: params.parent.is_none(),
        },
    )
}