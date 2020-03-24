use super::{create_renderer_pipeline, LightRenderPass, RendererPipelineParameters};
use ash::{version::DeviceV1_0, vk, Device};
use environment::*;
use std::sync::Arc;
use vulkan::*;

pub struct SkyboxRenderer {
    context: Arc<Context>,
    model: SkyboxModel,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl SkyboxRenderer {
    pub fn create(
        context: Arc<Context>,
        camera_buffers: &[Buffer],
        swapchain_props: SwapchainProperties,
        environment: &Environment,
        msaa_samples: vk::SampleCountFlags,
        render_pass: &LightRenderPass,
    ) -> Self {
        let model = SkyboxModel::new(&context);
        let descriptors = create_descriptors(&context, camera_buffers, &environment);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let pipeline = create_skybox_pipeline(
            &context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            pipeline_layout,
        );

        Self {
            context,
            model,
            descriptors,
            pipeline_layout,
            pipeline,
        }
    }

    pub fn rebuild_pipeline(
        &mut self,
        swapchain_props: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: &LightRenderPass,
    ) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }

        self.pipeline = create_skybox_pipeline(
            &self.context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            self.pipeline_layout,
        );
    }
}

impl SkyboxRenderer {
    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, frame_index: usize) {
        let device = self.context.device();
        // Bind skybox pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            )
        };

        // Bind skybox descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptors.sets()[frame_index..=frame_index],
                &[],
            )
        };

        unsafe {
            device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.model.vertices().buffer],
                &[0],
            );
        }

        unsafe {
            device.cmd_bind_index_buffer(
                command_buffer,
                self.model.indices().buffer,
                0,
                vk::IndexType::UINT32,
            );
        }

        // Draw skybox
        unsafe { device.cmd_draw_indexed(command_buffer, 36, 1, 0, 0, 0) };
    }
}

impl Drop for SkyboxRenderer {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn create_descriptors(
    context: &Arc<Context>,
    uniform_buffers: &[Buffer],
    environment: &Environment,
) -> Descriptors {
    let layout = create_descriptor_set_layout(context.device());
    let pool = create_descriptor_pool(context.device(), uniform_buffers.len() as _);
    let sets = create_descriptor_sets(context, pool, layout, uniform_buffers, environment.skybox());
    Descriptors::new(Arc::clone(context), layout, pool, sets)
}

fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}
fn create_descriptor_pool(device: &Device, descriptor_count: u32) -> vk::DescriptorPool {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count,
        },
    ];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    buffers: &[Buffer],
    cubemap: &Texture,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..buffers.len()).map(|_| layout).collect::<Vec<_>>();

    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()
    };

    sets.iter().zip(buffers.iter()).for_each(|(set, buffer)| {
        let buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];

        let cubemap_info = [vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(cubemap.view)
            .sampler(cubemap.sampler.unwrap())
            .build()];

        let descriptor_writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&cubemap_info)
                .build(),
        ];

        unsafe {
            context
                .device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    });

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

fn create_skybox_pipeline(
    context: &Arc<Context>,
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

    create_renderer_pipeline::<SkyboxVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "skybox",
            fragment_shader_name: "skybox",
            vertex_shader_specialization: None,
            fragment_shader_specialization: None,
            swapchain_properties,
            msaa_samples,
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
