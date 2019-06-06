use super::{Context, ShaderModule, Vertex};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, rc::Rc};

#[derive(Copy, Clone)]
pub struct PipelineParameters<'a> {
    pub vertex_shader_name: &'static str,
    pub fragment_shader_name: &'static str,
    pub multisampling_info: &'a vk::PipelineMultisampleStateCreateInfo,
    pub viewport_info: &'a vk::PipelineViewportStateCreateInfo,
    pub rasterizer_info: &'a vk::PipelineRasterizationStateCreateInfo,
    pub dynamic_state_info: Option<&'a vk::PipelineDynamicStateCreateInfo>,
    pub depth_stencil_info: Option<&'a vk::PipelineDepthStencilStateCreateInfo>,
    pub color_blend_attachment: &'a vk::PipelineColorBlendAttachmentState,
    pub render_pass: vk::RenderPass,
    pub layout: vk::PipelineLayout,
    pub parent: Option<vk::Pipeline>,
    pub allow_derivatives: bool,
}

pub fn create_pipeline<V: Vertex>(
    context: &Rc<Context>,
    params: PipelineParameters,
) -> vk::Pipeline {
    let vertex_shader_module = ShaderModule::new(
        Rc::clone(context),
        format!("assets/shaders/{}.vert.spv", params.vertex_shader_name),
    );
    let fragment_shader_module = ShaderModule::new(
        Rc::clone(context),
        format!("assets/shaders/{}.frag.spv", params.fragment_shader_name),
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

    let color_blend_attachments = [*params.color_blend_attachment];

    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let mut pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(params.viewport_info)
        .rasterization_state(params.rasterizer_info)
        .multisample_state(params.multisampling_info)
        .color_blend_state(&color_blending_info)
        .layout(params.layout)
        .render_pass(params.render_pass)
        .subpass(0);

    if let Some(depth_stencil_info) = params.depth_stencil_info {
        pipeline_info = pipeline_info.depth_stencil_state(depth_stencil_info)
    }

    if let Some(dynamic_state_info) = params.dynamic_state_info {
        pipeline_info = pipeline_info.dynamic_state(dynamic_state_info);
    }

    if let Some(parent) = params.parent {
        pipeline_info = pipeline_info.base_pipeline_handle(parent);
    }

    if params.allow_derivatives {
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
