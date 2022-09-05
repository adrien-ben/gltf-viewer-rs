use std::{mem::size_of, sync::Arc};

use util::any_as_u8_slice;
use vulkan::{
    ash::{
        vk::{self, RenderingAttachmentInfo, RenderingInfo},
        Device,
    },
    Context, Descriptors,
};

use crate::renderer::{
    attachments::{Attachments, BLOOM_FORMAT, BLOOM_MIP_LEVELS},
    fullscreen::{create_fullscreen_pipeline, QuadModel},
};

const BLOOM_FILTER_RADIUS: f32 = 0.005;

pub struct BloomPass {
    context: Arc<Context>,
    descriptors: Descriptors,
    downsample_pipeline_layout: vk::PipelineLayout,
    upsample_pipeline_layout: vk::PipelineLayout,
    downsample_pipeline: vk::Pipeline,
    upsample_pipeline: vk::Pipeline,
}

impl BloomPass {
    pub fn create(context: Arc<Context>, attachments: &Attachments) -> Self {
        let descriptors = create_descriptors(&context, attachments);
        let downsample_pipeline_layout =
            create_downsample_pipeline_layout(context.device(), descriptors.layout());
        let upsample_pipeline_layout =
            create_upsample_pipeline_layout(context.device(), descriptors.layout());
        let downsample_pipeline = create_downsample_pipeline(&context, downsample_pipeline_layout);
        let upsample_pipeline = create_upsample_pipeline(&context, upsample_pipeline_layout);

        Self {
            context,
            descriptors,
            downsample_pipeline_layout,
            upsample_pipeline_layout,
            downsample_pipeline,
            upsample_pipeline,
        }
    }
}

impl BloomPass {
    pub fn set_attachments(&mut self, attachments: &Attachments) {
        update_descriptor_sets(&self.context, self.descriptors.sets(), attachments);
    }

    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        attachments: &Attachments,
        quad_model: &QuadModel,
    ) {
        self.cmd_downsample(command_buffer, attachments, quad_model);
        self.cmd_upsample(command_buffer, attachments, quad_model);

        attachments.bloom.image.cmd_transition_image_mips_layout(
            command_buffer,
            0,
            1,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }

    fn cmd_downsample(
        &self,
        command_buffer: vk::CommandBuffer,
        attachments: &Attachments,
        quad_model: &QuadModel,
    ) {
        let device = self.context.device();

        let mut input_extent = vk::Extent2D {
            width: attachments.bloom.image.extent.width,
            height: attachments.bloom.image.extent.height,
        };

        let mut input_image = &attachments.get_scene_resolved_color().image;
        let mut input_mip = 0u32;

        for output_mip in 0..attachments.bloom.image.get_mip_levels() as usize {
            let output_extent = attachments.bloom.mips_resolution[output_mip];

            {
                input_image.cmd_transition_image_mips_layout(
                    command_buffer,
                    input_mip as _,
                    1,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );

                attachments.bloom.image.cmd_transition_image_mips_layout(
                    command_buffer,
                    output_mip as _,
                    1,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }

            unsafe {
                self.context.device().cmd_set_viewport(
                    command_buffer,
                    0,
                    &[vk::Viewport {
                        width: output_extent.width as _,
                        height: output_extent.height as _,
                        max_depth: 1.0,
                        ..Default::default()
                    }],
                );
                self.context.device().cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D {
                        extent: output_extent,
                        ..Default::default()
                    }],
                )
            }

            {
                let attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(attachments.bloom.mips_views[output_mip])
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&attachment_info))
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        extent: output_extent,
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
                    self.downsample_pipeline,
                )
            };

            // Bind buffers
            unsafe {
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[quad_model.vertices.buffer],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    command_buffer,
                    quad_model.indices.buffer,
                    0,
                    vk::IndexType::UINT16,
                );
            }

            // Bind descriptor sets
            unsafe {
                let set_index = output_mip as usize;
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.downsample_pipeline_layout,
                    0,
                    &self.descriptors.sets()[set_index..=set_index],
                    &[],
                )
            };

            // Push constants
            unsafe {
                let data = [(input_extent.width as f32, input_extent.height as f32)];
                let data = any_as_u8_slice(&data);
                device.cmd_push_constants(
                    command_buffer,
                    self.downsample_pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    data,
                );
            }

            // Draw
            unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_end_rendering(command_buffer)
            };

            input_extent = output_extent;
            input_image = &attachments.bloom.image;
            input_mip = output_mip as _;
        }
    }

    fn cmd_upsample(
        &self,
        command_buffer: vk::CommandBuffer,
        attachments: &Attachments,
        quad_model: &QuadModel,
    ) {
        let device = self.context.device();

        for output_mip in (0..(attachments.bloom.image.get_mip_levels() - 1) as usize).rev() {
            let output_extent = attachments.bloom.mips_resolution[output_mip];
            let input_mip = output_mip + 1;

            {
                attachments.bloom.image.cmd_transition_image_mips_layout(
                    command_buffer,
                    input_mip as _,
                    1,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );

                attachments.bloom.image.cmd_transition_image_mips_layout(
                    command_buffer,
                    output_mip as _,
                    1,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }

            unsafe {
                self.context.device().cmd_set_viewport(
                    command_buffer,
                    0,
                    &[vk::Viewport {
                        width: output_extent.width as _,
                        height: output_extent.height as _,
                        max_depth: 1.0,
                        ..Default::default()
                    }],
                );
                self.context.device().cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D {
                        extent: output_extent,
                        ..Default::default()
                    }],
                )
            }

            {
                let attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(attachments.bloom.mips_views[output_mip])
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&attachment_info))
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        extent: output_extent,
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
                    self.upsample_pipeline,
                )
            };

            // Bind buffers
            unsafe {
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[quad_model.vertices.buffer],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    command_buffer,
                    quad_model.indices.buffer,
                    0,
                    vk::IndexType::UINT16,
                );
            }

            // Bind descriptor sets
            unsafe {
                let set_index = 1 + input_mip as usize;
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.upsample_pipeline_layout,
                    0,
                    &self.descriptors.sets()[set_index..=set_index],
                    &[],
                )
            };

            // Push constants
            unsafe {
                let data = [BLOOM_FILTER_RADIUS];
                let data = any_as_u8_slice(&data);
                device.cmd_push_constants(
                    command_buffer,
                    self.upsample_pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    data,
                );
            }

            // Draw
            unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_end_rendering(command_buffer)
            };
        }
    }
}

impl Drop for BloomPass {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.upsample_pipeline, None);
            device.destroy_pipeline(self.downsample_pipeline, None);
            device.destroy_pipeline_layout(self.upsample_pipeline_layout, None);
            device.destroy_pipeline_layout(self.downsample_pipeline_layout, None);
        }
    }
}

fn create_descriptors(context: &Arc<Context>, attachments: &Attachments) -> Descriptors {
    let layout = create_descriptor_set_layout(context.device());
    let pool = create_descriptor_pool(context.device());
    let sets = create_descriptor_sets(context, pool, layout, attachments);
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
    let descriptor_count = BLOOM_MIP_LEVELS + 1; // all mips + input image
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
    attachments: &Attachments,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..(BLOOM_MIP_LEVELS + 1))
        .map(|_| layout)
        .collect::<Vec<_>>();
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()
    };

    update_descriptor_sets(context, &sets, attachments);

    sets
}

/// first set will reference the input image that we want to blur
/// subsequent sets will reference the mips of the bloom attachments
fn update_descriptor_sets(
    context: &Arc<Context>,
    sets: &[vk::DescriptorSet],
    attachments: &Attachments,
) {
    let mut image_infos = vec![];

    image_infos.push(
        vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(attachments.get_scene_resolved_color().view)
            .sampler(
                attachments
                    .get_scene_resolved_color()
                    .sampler
                    .expect("Post process input image must have a sampler"),
            )
            .build(),
    );

    for view in &attachments.bloom.mips_views {
        image_infos.push(
            vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(*view)
                .sampler(attachments.bloom.sampler)
                .build(),
        )
    }

    let descriptor_writes = image_infos
        .iter()
        .zip(sets.iter())
        .map(|(info, set)| {
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))
                .build()
        })
        .collect::<Vec<_>>();

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }
}

fn create_downsample_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let push_constant_ranges = [vk::PushConstantRange {
        offset: 0,
        size: (2 * size_of::<f32>()) as _,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_ranges);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_upsample_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let push_constant_ranges = [vk::PushConstantRange {
        offset: 0,
        size: size_of::<f32>() as _,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_ranges);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_downsample_pipeline(context: &Arc<Context>, layout: vk::PipelineLayout) -> vk::Pipeline {
    create_fullscreen_pipeline(context, BLOOM_FORMAT, layout, "downsample", None)
}

fn create_upsample_pipeline(context: &Arc<Context>, layout: vk::PipelineLayout) -> vk::Pipeline {
    create_fullscreen_pipeline(context, BLOOM_FORMAT, layout, "upsample", None)
}
