use crate::{environment::*, model::*, vulkan::*};
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::Matrix4;
use std::{ffi::CString, mem::size_of, rc::Rc};

pub struct Pipelines {
    context: Rc<Context>,
    skybox_layout: vk::PipelineLayout,
    skybox_pipeline: vk::Pipeline,
    model_layout: vk::PipelineLayout,
    opaque_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
}

impl Pipelines {
    pub fn build(
        context: Rc<Context>,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        skybox_descriptors: &Descriptors,
        model_descriptors: &Descriptors,
    ) -> Self {
        let device = context.device();

        let skybox_layout = create_skybox_pipeline_layout(device, skybox_descriptors.layout());

        let skybox_pipeline = create_skybox_pipeline(
            &context,
            swapchain_properties,
            msaa_samples,
            render_pass,
            skybox_layout,
        );

        let model_layout = create_model_pipeline_layout(device, model_descriptors.layout());

        let opaque_pipeline = create_opaque_pipeline(
            &context,
            swapchain_properties,
            msaa_samples,
            render_pass,
            model_layout,
        );

        let transparent_pipeline = create_transparent_pipeline(
            &context,
            swapchain_properties,
            msaa_samples,
            render_pass,
            model_layout,
            opaque_pipeline,
        );

        Self {
            context,
            skybox_layout,
            skybox_pipeline,
            model_layout,
            opaque_pipeline,
            transparent_pipeline,
        }
    }
}

impl Pipelines {
    pub fn skybox_layout(&self) -> vk::PipelineLayout {
        self.skybox_layout
    }

    pub fn skybox_pipeline(&self) -> vk::Pipeline {
        self.skybox_pipeline
    }

    pub fn model_layout(&self) -> vk::PipelineLayout {
        self.model_layout
    }

    pub fn opaque_pipeline(&self) -> vk::Pipeline {
        self.opaque_pipeline
    }

    pub fn transparent_pipeline(&self) -> vk::Pipeline {
        self.transparent_pipeline
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.skybox_pipeline, None);
            device.destroy_pipeline_layout(self.skybox_layout, None);
            device.destroy_pipeline(self.opaque_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
            device.destroy_pipeline_layout(self.model_layout, None);
        }
    }
}

fn create_skybox_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_skybox_pipeline(
    context: &Rc<Context>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
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

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    create_pipeline::<SkyboxVertex>(
        context,
        "skybox",
        swapchain_properties,
        msaa_samples,
        render_pass,
        layout,
        &depth_stencil_info,
        &color_blend_attachment,
        None,
    )
}

fn create_model_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let push_constant_range = [
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: size_of::<Matrix4<f32>>() as _,
        },
        vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: size_of::<Matrix4<f32>>() as _,
            size: size_of::<Material>() as _,
        },
    ];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);

    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_opaque_pipeline(
    context: &Rc<Context>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    create_pipeline::<ModelVertex>(
        context,
        "model",
        swapchain_properties,
        msaa_samples,
        render_pass,
        layout,
        &depth_stencil_info,
        &color_blend_attachment,
        None,
    )
}

fn create_transparent_pipeline(
    context: &Rc<Context>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    parent: vk::Pipeline,
) -> vk::Pipeline {
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    create_pipeline::<ModelVertex>(
        context,
        "model",
        swapchain_properties,
        msaa_samples,
        render_pass,
        layout,
        &depth_stencil_info,
        &color_blend_attachment,
        Some(parent),
    )
}

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
