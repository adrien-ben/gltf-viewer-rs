mod model;
mod skybox;

pub use self::{model::*, skybox::*};
use crate::vulkan::*;
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, rc::Rc};

fn create_pipeline<V: Vertex>(
    context: &Rc<Context>,
    shader_name: &str,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    depth_stencil_info: &vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment: &vk::PipelineColorBlendAttachmentState,
    parent: Option<vk::Pipeline>,
) -> vk::Pipeline {
    let vertex_shader_module = ShaderModule::new(
        Rc::clone(context),
        format!("assets/shaders/{}.vert.spv", &shader_name),
    );
    let fragment_shader_module = ShaderModule::new(
        Rc::clone(context),
        format!("assets/shaders/{}.frag.spv", &shader_name),
    );

    let entry_point_name = CString::new("main").unwrap();
    let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module.module())
        .name(&entry_point_name)
        .build();
    let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module.module())
        .name(&entry_point_name)
        .build();
    let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

    let bindings_descs = V::get_bindings_descriptions();
    let attributes_descs = V::get_attributes_descriptions();
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&bindings_descs)
        .vertex_attribute_descriptions(&attributes_descs);

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_properties.extent.width as _,
        height: swapchain_properties.extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let viewports = [viewport];
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_properties.extent,
    };
    let scissors = [scissor];
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachments = [*color_blend_attachment];

    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let mut pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_info)
        .rasterization_state(&rasterizer_info)
        .multisample_state(&multisampling_info)
        .depth_stencil_state(&depth_stencil_info)
        .color_blend_state(&color_blending_info)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    if let Some(parent) = parent {
        pipeline_info = pipeline_info.base_pipeline_handle(parent);
    } else {
        pipeline_info = pipeline_info.flags(vk::PipelineCreateFlags::ALLOW_DERIVATIVES);
    }
    let pipeline_info = pipeline_info.build();
    let pipeline_infos = [pipeline_info];

    unsafe {
        context
            .device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
            .unwrap()[0]
    }
}
