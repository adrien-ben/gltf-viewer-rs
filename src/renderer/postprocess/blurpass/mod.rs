mod renderpass;

use crate::renderer::{create_renderer_pipeline, fullscreen::*, RendererPipelineParameters};
use renderpass::RenderPass;
use std::sync::Arc;
use vulkan::ash::{vk, Device};
use vulkan::{Context, Descriptors, SwapchainProperties, Texture};

/// Blur pass
pub struct BlurPass {
    context: Arc<Context>,
    extent: vk::Extent2D,
    render_pass: RenderPass,
    framebuffer: vk::Framebuffer,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl BlurPass {
    pub fn create(
        context: Arc<Context>,
        swapchain_props: SwapchainProperties,
        input_image: &Texture,
    ) -> Self {
        let render_pass = RenderPass::create(Arc::clone(&context), swapchain_props.extent);
        let framebuffer = render_pass.create_framebuffer();

        let descriptors = create_descriptors(&context, input_image);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
        );

        BlurPass {
            context,
            extent: swapchain_props.extent,
            render_pass,
            framebuffer,
            descriptors,
            pipeline_layout,
            pipeline,
        }
    }
}

impl BlurPass {
    pub fn set_input_image(&mut self, input_image: &Texture) {
        unsafe {
            self.context
                .device()
                .free_descriptor_sets(self.descriptors.pool(), self.descriptors.sets())
                .expect("Failed to free descriptor sets");
        }
        self.descriptors.set_sets(create_descriptor_sets(
            &self.context,
            self.descriptors.pool(),
            self.descriptors.layout(),
            input_image,
        ));
    }

    pub fn set_extent(&mut self, extent: vk::Extent2D) {
        unsafe {
            self.context
                .device()
                .destroy_framebuffer(self.framebuffer, None);
        }

        self.extent = extent;
        self.render_pass = RenderPass::create(Arc::clone(&self.context), extent);
        self.framebuffer = self.render_pass.create_framebuffer();
    }

    pub fn rebuild_pipelines(&mut self, swapchain_properties: SwapchainProperties) {
        let device = self.context.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }

        self.pipeline = create_pipeline(
            &self.context,
            swapchain_properties,
            self.render_pass.get_render_pass(),
            self.pipeline_layout,
        )
    }

    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, quad_model: &QuadModel) {
        let device = self.context.device();

        {
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.render_pass.get_render_pass())
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .clear_values(&clear_values);

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                )
            };
        }

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
                self.descriptors.sets(),
                &[],
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };

        unsafe { device.cmd_end_render_pass(command_buffer) };
    }
}

/// Getters
impl BlurPass {
    pub fn get_output(&self) -> &Texture {
        self.render_pass.get_output_attachment()
    }
}

impl Drop for BlurPass {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
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
            fragment_shader_name: "blur",
            vertex_shader_specialization: None,
            fragment_shader_specialization: None,
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
