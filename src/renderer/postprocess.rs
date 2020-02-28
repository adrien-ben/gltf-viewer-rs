use super::{create_renderer_pipeline, fullscreen::*, RendererPipelineParameters};
use std::{mem::size_of, sync::Arc};
use vulkan::ash::{version::DeviceV1_0, vk, Device};
use vulkan::{Context, Descriptors, SimpleRenderPass, SwapchainProperties, Texture};

/// Tone mapping and gamma correction pass
pub struct FinalPass {
    context: Arc<Context>,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

#[derive(Debug, Clone, Copy)]
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
        swapchain_props: SwapchainProperties,
        render_pass: &SimpleRenderPass,
        input_image: &Texture,
        tone_map_mode: ToneMapMode,
    ) -> Self {
        let descriptors = create_descriptors(&context, input_image);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
            tone_map_mode,
        );

        FinalPass {
            context,
            descriptors,
            pipeline_layout,
            pipeline,
        }
    }
}

impl FinalPass {
    pub fn set_input_image(&mut self, input_image: &Texture) {
        unsafe {
            self.context
                .device()
                .free_descriptor_sets(self.descriptors.pool(), self.descriptors.sets());
        }
        self.descriptors.set_sets(create_descriptor_sets(
            &self.context,
            self.descriptors.pool(),
            self.descriptors.layout(),
            input_image,
        ));
    }

    pub fn rebuild_pipelines(
        &mut self,
        swapchain_properties: SwapchainProperties,
        render_pass: &SimpleRenderPass,
        tone_map_mode: ToneMapMode,
    ) {
        let device = self.context.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }

        self.pipeline = create_pipeline(
            &self.context,
            swapchain_properties,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            tone_map_mode,
        )
    }

    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, quad_model: &QuadModel) {
        let device = self.context.device();
        // Bind pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
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
                &self.descriptors.sets(),
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
            device.destroy_pipeline(self.pipeline, None);
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
        .dst_set(sets[0])
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&input_image_info)
        .build()];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    sets
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
    swapchain_properties: SwapchainProperties,
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
        .color_write_mask(vk::ColorComponentFlags::all())
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
            swapchain_properties,
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
