use crate::renderer::{
    create_renderer_pipeline, fullscreen::*, RendererPipelineParameters, RendererSettings,
};
use std::{mem::size_of, sync::Arc};
use vulkan::ash::{vk, Device};
use vulkan::{Context, Descriptors, SimpleRenderPass, Texture};

/// Tone mapping and gamma correction pass
pub struct FinalPass {
    context: Arc<Context>,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    default_pipeline: vk::Pipeline,
    uncharted_pipeline: vk::Pipeline,
    hejl_richard_pipeline: vk::Pipeline,
    aces_pipeline: vk::Pipeline,
    none_pipeline: vk::Pipeline,
    tone_map_mode: ToneMapMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapMode {
    Default = 0,
    Uncharted,
    HejlRichard,
    Aces,
    None,
}

impl ToneMapMode {
    pub fn all() -> [ToneMapMode; 5] {
        use ToneMapMode::*;
        [Default, Uncharted, HejlRichard, Aces, None]
    }

    pub fn from_value(value: usize) -> Option<Self> {
        use ToneMapMode::*;
        match value {
            0 => Some(Default),
            1 => Some(Uncharted),
            2 => Some(HejlRichard),
            3 => Some(Aces),
            4 => Some(None),
            _ => Option::None,
        }
    }
}

impl FinalPass {
    pub fn create(
        context: Arc<Context>,
        render_pass: &SimpleRenderPass,
        input_image: &Texture,
        settings: RendererSettings,
    ) -> Self {
        let descriptors = create_descriptors(&context, input_image);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let default_pipeline = create_pipeline(
            &context,
            render_pass.get_render_pass(),
            pipeline_layout,
            ToneMapMode::Default,
        );
        let uncharted_pipeline = create_pipeline(
            &context,
            render_pass.get_render_pass(),
            pipeline_layout,
            ToneMapMode::Uncharted,
        );
        let hejl_richard_pipeline = create_pipeline(
            &context,
            render_pass.get_render_pass(),
            pipeline_layout,
            ToneMapMode::HejlRichard,
        );
        let aces_pipeline = create_pipeline(
            &context,
            render_pass.get_render_pass(),
            pipeline_layout,
            ToneMapMode::Aces,
        );
        let none_pipeline = create_pipeline(
            &context,
            render_pass.get_render_pass(),
            pipeline_layout,
            ToneMapMode::None,
        );

        let tone_map_mode = settings.tone_map_mode;

        FinalPass {
            context,
            descriptors,
            pipeline_layout,
            default_pipeline,
            uncharted_pipeline,
            hejl_richard_pipeline,
            aces_pipeline,
            none_pipeline,
            tone_map_mode,
        }
    }
}

impl FinalPass {
    pub fn set_tone_map_mode(&mut self, tone_map_mode: ToneMapMode) {
        self.tone_map_mode = tone_map_mode;
    }

    pub fn set_input_image(&mut self, input_image: &Texture) {
        self.descriptors
            .sets()
            .iter()
            .for_each(|s| update_descriptor_set(&self.context, *s, input_image))
    }

    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, quad_model: &QuadModel) {
        let device = self.context.device();
        // Bind pipeline
        let current_pipeline = match self.tone_map_mode {
            ToneMapMode::Default => self.default_pipeline,
            ToneMapMode::Uncharted => self.uncharted_pipeline,
            ToneMapMode::HejlRichard => self.hejl_richard_pipeline,
            ToneMapMode::Aces => self.aces_pipeline,
            ToneMapMode::None => self.none_pipeline,
        };

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                current_pipeline,
            )
        };

        // Bind buffers
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[quad_model.vertices.buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                quad_model.indices.buffer,
                0,
                vk::IndexType::UINT16,
            );
        }

        // Bind descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                self.descriptors.sets(),
                &[],
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };
    }
}

impl Drop for FinalPass {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.default_pipeline, None);
            device.destroy_pipeline(self.uncharted_pipeline, None);
            device.destroy_pipeline(self.hejl_richard_pipeline, None);
            device.destroy_pipeline(self.aces_pipeline, None);
            device.destroy_pipeline(self.none_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn create_descriptors(context: &Arc<Context>, input_image: &Texture) -> Descriptors {
    let layout = create_descriptor_set_layout(context.device());
    let pool = create_descriptor_pool(context.device());
    let sets = create_descriptor_sets(context, pool, layout, input_image);
    Descriptors::new(Arc::clone(context), layout, pool, sets)
}

fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}
fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let descriptor_count = 1;
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count,
    }];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    input_image: &Texture,
) -> Vec<vk::DescriptorSet> {
    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()
    };

    update_descriptor_set(context, sets[0], input_image);

    sets
}

fn update_descriptor_set(context: &Arc<Context>, set: vk::DescriptorSet, input_image: &Texture) {
    let input_image_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(input_image.view)
        .sampler(
            input_image
                .sampler
                .expect("Post process input image must have a sampler"),
        )
        .build()];

    let descriptor_writes = [vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&input_image_info)
        .build()];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(
    context: &Arc<Context>,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    tone_map_mode: ToneMapMode,
) -> vk::Pipeline {
    let (specialization_info, _map_entries, _data) =
        create_model_frag_shader_specialization(tone_map_mode);

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
            fragment_shader_name: "final",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            render_pass,
            subpass: 0,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling: true,
            parent: None,
        },
    )
}

fn create_model_frag_shader_specialization(
    tone_map_mode: ToneMapMode,
) -> (
    vk::SpecializationInfo,
    Vec<vk::SpecializationMapEntry>,
    Vec<u8>,
) {
    let map_entries = vec![vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: size_of::<u32>(),
    }];

    let data = [tone_map_mode as u32];

    let data = Vec::from(unsafe { util::any_as_u8_slice(&data) });

    let specialization_info = vk::SpecializationInfo::builder()
        .map_entries(&map_entries)
        .data(&data)
        .build();

    (specialization_info, map_entries, data)
}
