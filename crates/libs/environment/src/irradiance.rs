use super::{
    create_descriptors, create_env_pipeline, get_view_matrices, EnvPipelineParameters, SkyboxModel,
    SkyboxVertex,
};
use cgmath::{Deg, Matrix4};
use math::*;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;
use util::*;
use vulkan::ash::vk::{self, RenderingAttachmentInfo, RenderingInfo};
use vulkan::{Context, Texture};

pub(crate) fn create_irradiance_map(
    context: &Arc<Context>,
    cubemap: &Texture,
    size: u32,
) -> Texture {
    log::info!("Creating irradiance map");
    let start = Instant::now();

    let device = context.device();

    let skybox_model = SkyboxModel::new(context);

    let descriptors = create_descriptors(context, cubemap);

    let (pipeline_layout, pipeline) = {
        let layout = {
            let layouts = [descriptors.layout()];
            let push_constant_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<Matrix4<f32>>() as _,
            }];
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_range);

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline = {
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: size as _,
                height: size as _,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let viewports = [viewport];
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            };
            let scissors = [scissor];
            let viewport_info = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);

            let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::FRONT)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0);

            create_env_pipeline::<SkyboxVertex>(
                context,
                EnvPipelineParameters {
                    vertex_shader_name: "cubemap",
                    fragment_shader_name: "irradiance",
                    viewport_info: &viewport_info,
                    rasterizer_info: &rasterizer_info,
                    dynamic_state_info: None,
                    layout,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                },
            )
        };

        (layout, pipeline)
    };

    // create cubemap
    let irradiance_map =
        Texture::create_renderable_cubemap(context, size, 1, vk::Format::R32G32B32A32_SFLOAT);

    let views = (0..6)
        .map(|i| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(irradiance_map.image.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: i,
                    layer_count: 1,
                });

            unsafe { device.create_image_view(&create_info, None).unwrap() }
        })
        .collect::<Vec<_>>();

    let view_matrices = get_view_matrices();

    let proj = perspective(Deg(90.0), 1.0, 0.1, 10.0);

    // Render
    context.execute_one_time_commands(|buffer| {
        {
            for face in 0..6 {
                let attachment_info = RenderingAttachmentInfo::default()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(views[face])
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::default()
                    .color_attachments(std::slice::from_ref(&attachment_info))
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: size,
                            height: size,
                        },
                    });

                unsafe {
                    context
                        .dynamic_rendering()
                        .cmd_begin_rendering(buffer, &rendering_info)
                };

                unsafe {
                    device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
                };

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &descriptors.sets()[0..=0],
                        &[],
                    )
                };

                let view = view_matrices[face];
                let view_proj = proj * view;
                unsafe {
                    let push = any_as_u8_slice(&view_proj);
                    device.cmd_push_constants(
                        buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        push,
                    );
                };

                unsafe {
                    device.cmd_bind_vertex_buffers(
                        buffer,
                        0,
                        &[skybox_model.vertices().buffer],
                        &[0],
                    );
                }

                unsafe {
                    device.cmd_bind_index_buffer(
                        buffer,
                        skybox_model.indices().buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Draw skybox
                unsafe { device.cmd_draw_indexed(buffer, 36, 1, 0, 0, 0) };

                // End render pass
                unsafe { context.dynamic_rendering().cmd_end_rendering(buffer) };
            }
        }
    });

    irradiance_map.image.transition_image_layout(
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    // Cleanup
    unsafe {
        views
            .iter()
            .for_each(|v| device.destroy_image_view(*v, None));
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
    }

    let time = start.elapsed().as_millis();
    log::info!("Finished creating irradiance map. Took {} ms", time);

    irradiance_map
}
