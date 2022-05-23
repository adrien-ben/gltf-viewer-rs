use super::{Context, ShaderModule, Vertex};
use ash::vk;
use std::{ffi::CString, sync::Arc};

#[derive(Copy, Clone)]
pub struct PipelineParameters<'a> {
    pub vertex_shader_params: ShaderParameters<'a>,
    pub fragment_shader_params: ShaderParameters<'a>,
    pub multisampling_info: &'a vk::PipelineMultisampleStateCreateInfo,
    pub viewport_info: &'a vk::PipelineViewportStateCreateInfo,
    pub rasterizer_info: &'a vk::PipelineRasterizationStateCreateInfo,
    pub dynamic_state_info: Option<&'a vk::PipelineDynamicStateCreateInfo>,
    pub depth_stencil_info: Option<&'a vk::PipelineDepthStencilStateCreateInfo>,
    pub color_blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    pub render_pass: vk::RenderPass,
    pub subpass: u32,
    pub layout: vk::PipelineLayout,
    pub parent: Option<vk::Pipeline>,
    pub allow_derivatives: bool,
}

pub fn create_pipeline<V: Vertex>(
    context: &Arc<Context>,
    params: PipelineParameters,
) -> vk::Pipeline {
    let entry_point_name = CString::new("main").unwrap();

    let (_vertex_shader_module, vertex_shader_state_info) = create_shader_stage_info(
        context,
        &entry_point_name,
        vk::ShaderStageFlags::VERTEX,
        params.vertex_shader_params,
    );

    let (_fragment_shader_module, fragment_shader_state_info) = create_shader_stage_info(
        context,
        &entry_point_name,
        vk::ShaderStageFlags::FRAGMENT,
        params.fragment_shader_params,
    );

    let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

    let bindings_descs = V::get_bindings_descriptions();
    let attributes_descs = V::get_attributes_descriptions();
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&bindings_descs)
        .vertex_attribute_descriptions(&attributes_descs);

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(params.color_blend_attachments)
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
        .subpass(params.subpass);

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
            .expect("Failed to create graphics pipeline")[0]
    }
}

fn create_shader_stage_info(
    context: &Arc<Context>,
    entry_point_name: &CString,
    stage: vk::ShaderStageFlags,
    params: ShaderParameters,
) -> (ShaderModule, vk::PipelineShaderStageCreateInfo) {
    let extension = get_shader_file_extension(stage);
    let shader_path = format!("crates/viewer/shaders/{}.{}.spv", params.name, extension);
    let module = ShaderModule::new(Arc::clone(context), &shader_path);

    let mut stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(stage)
        .module(module.module())
        .name(entry_point_name);
    if let Some(specialization) = params.specialization {
        stage_info = stage_info.specialization_info(specialization);
    }
    let stage_info = stage_info.build();

    (module, stage_info)
}

fn get_shader_file_extension(stage: vk::ShaderStageFlags) -> &'static str {
    match stage {
        vk::ShaderStageFlags::VERTEX => "vert",
        vk::ShaderStageFlags::FRAGMENT => "frag",
        _ => panic!("Unsupported shader stage"),
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ShaderParameters<'a> {
    name: &'static str,
    specialization: Option<&'a vk::SpecializationInfo>,
}

impl<'a> ShaderParameters<'a> {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            specialization: None,
        }
    }

    pub fn specialized(name: &'static str, specialization: &'a vk::SpecializationInfo) -> Self {
        Self {
            name,
            specialization: Some(specialization),
        }
    }
}
