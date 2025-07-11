use crate::renderer::attachments::Attachments;
use crate::renderer::fullscreen::*;
use std::sync::Arc;
use vulkan::ash::vk::{RenderingAttachmentInfo, RenderingInfo};
use vulkan::ash::{Device, vk};
use vulkan::{Context, Descriptors, Texture};

const BLUR_OUTPUT_FORMAT: vk::Format = vk::Format::R8_UNORM;

/// Blur pass
pub struct BlurPass {
    context: Arc<Context>,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl BlurPass {
    pub fn create(context: Arc<Context>, input_image: &Texture) -> Self {
        let descriptors = create_descriptors(&context, input_image);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let pipeline = create_pipeline(&context, pipeline_layout);

        BlurPass {
            context,
            descriptors,
            pipeline_layout,
            pipeline,
        }
    }
}

impl BlurPass {
    pub fn set_input_image(&mut self, input_image: &Texture) {
        self.descriptors
            .sets()
            .iter()
            .for_each(|s| update_descriptor_set(&self.context, *s, input_image));
    }

    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        attachments: &Attachments,
        quad_model: &QuadModel,
    ) {
        let device = self.context.device();

        let extent = vk::Extent2D {
            width: attachments.ssao_blur.image.extent.width,
            height: attachments.ssao_blur.image.extent.height,
        };

        unsafe {
            self.context.device().cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    width: extent.width as _,
                    height: extent.height as _,
                    max_depth: 1.0,
                    ..Default::default()
                }],
            );
            self.context.device().cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    extent,
                    ..Default::default()
                }],
            )
        }

        {
            let attachment_info = RenderingAttachmentInfo::default()
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(attachments.ssao_blur.view)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = RenderingInfo::default()
                .color_attachments(std::slice::from_ref(&attachment_info))
                .layer_count(1)
                .render_area(vk::Rect2D {
                    extent,
                    ..Default::default()
                });

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_begin_rendering(command_buffer, &rendering_info)
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

        unsafe {
            self.context
                .dynamic_rendering()
                .cmd_end_rendering(command_buffer)
        };
    }
}

impl Drop for BlurPass {
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
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

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

    let create_info = vk::DescriptorPoolCreateInfo::default()
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
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
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
    let input_image_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(input_image.view)
        .sampler(
            input_image
                .sampler
                .expect("Post process input image must have a sampler"),
        )];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&input_image_info)];

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
    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(context: &Arc<Context>, layout: vk::PipelineLayout) -> vk::Pipeline {
    create_fullscreen_pipeline(context, BLUR_OUTPUT_FORMAT, layout, "blur", None)
}
