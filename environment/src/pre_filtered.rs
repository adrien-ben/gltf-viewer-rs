use super::{
    create_descriptors, create_env_pipeline, create_render_pass, get_view_matrices,
    EnvPipelineParameters, SkyboxModel, SkyboxVertex,
};
use cgmath::{Deg, Matrix4};
use math::*;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;
use util::*;
use vulkan::ash::{version::DeviceV1_0, vk};
use vulkan::{Context, Texture};

pub(crate) fn create_pre_filtered_map(
    context: &Arc<Context>,
    cubemap: &Texture,
    size: u32,
) -> Texture {
    log::info!("Creating pre-filtered map");
    let start = Instant::now();

    let device = context.device();

    let skybox_model = SkyboxModel::new(context);

    let max_mip_levels = (size as f32).log2().floor() as u32 + 1;

    let cubemap_formap = vk::Format::R16G16B16A16_SFLOAT;

    let render_pass = create_render_pass(context, cubemap_formap);

    let descriptors = create_descriptors(context, &cubemap);

    let (pipeline_layout, pipeline) = {
        let layout = {
            let layouts = [descriptors.layout()];
            let push_constant_range = [
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: size_of::<Matrix4<f32>>() as _,
                },
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    offset: size_of::<Matrix4<f32>>() as _,
                    size: size_of::<f32>() as _,
                },
            ];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_range);

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline = {
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);

            let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
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

            let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

            create_env_pipeline::<SkyboxVertex>(
                context,
                EnvPipelineParameters {
                    vertex_shader_name: "cubemap",
                    fragment_shader_name: "pre_filtered",
                    viewport_info: &viewport_info,
                    rasterizer_info: &rasterizer_info,
                    dynamic_state_info: Some(&dynamic_state_info),
                    layout,
                    render_pass,
                },
            )
        };

        (layout, pipeline)
    };

    // create cubemap
    let pre_filtered =
        Texture::create_renderable_cubemap(context, size, max_mip_levels, cubemap_formap);

    let mut views = Vec::new();
    let mut framebuffers = Vec::new();
    for lod in 0..max_mip_levels {
        let mip_factor = 1.0_f32 / (2.0_f32.powi(lod as i32));
        let viewport_size = (size as f32 * mip_factor) as u32;

        let lod_views = (0..6)
            .map(|i| {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(pre_filtered.image.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(cubemap_formap)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: lod,
                        level_count: 1,
                        base_array_layer: i,
                        layer_count: 1,
                    });

                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        let lod_framebuffers = lod_views
            .iter()
            .map(|view| {
                let attachments = [*view];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(viewport_size)
                    .height(viewport_size)
                    .layers(1);
                unsafe { device.create_framebuffer(&create_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        views.push(lod_views);
        framebuffers.push(lod_framebuffers);
    }

    let view_matrices = get_view_matrices();

    let proj = perspective(Deg(90.0), 1.0, 0.1, 10.0);

    // Render
    context.execute_one_time_commands(|buffer| {
        {
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            }];
            unsafe { device.cmd_set_scissor(buffer, 0, &scissor) };

            for lod in 0..max_mip_levels {
                let mip_factor = 1.0_f32 / (2.0_f32.powi(lod as i32));
                let viewport_size = (size as f32 * mip_factor) as u32;

                let viewport = [vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: viewport_size as _,
                    height: viewport_size as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }];
                unsafe { device.cmd_set_viewport(buffer, 0, &viewport) };

                for (face, view) in view_matrices.iter().enumerate() {
                    let framebuffer = framebuffers[lod as usize][face];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(render_pass)
                        .framebuffer(framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: viewport_size,
                                height: viewport_size,
                            },
                        })
                        .clear_values(&clear_values);

                    unsafe {
                        device.cmd_begin_render_pass(
                            buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        )
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

                    let view_proj = proj * view;
                    unsafe {
                        let matrix_constant = any_as_u8_slice(&view_proj);
                        device.cmd_push_constants(
                            buffer,
                            pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            matrix_constant,
                        );

                        let roughness = (lod as f32) / (max_mip_levels as f32 - 1.0);
                        let roughness_constant = any_as_u8_slice(&roughness);
                        device.cmd_push_constants(
                            buffer,
                            pipeline_layout,
                            vk::ShaderStageFlags::FRAGMENT,
                            size_of::<Matrix4<f32>>() as _,
                            roughness_constant,
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
                    unsafe { device.cmd_end_render_pass(buffer) };
                }
            }
        }
    });

    pre_filtered.image.transition_image_layout(
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    // Cleanup
    unsafe {
        views
            .iter()
            .flatten()
            .for_each(|v| device.destroy_image_view(*v, None));
        framebuffers
            .iter()
            .flatten()
            .for_each(|fb| device.destroy_framebuffer(*fb, None));
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);
    }

    let time = start.elapsed().as_millis();
    log::info!("Finished creating pre-filtered map. Took {} ms", time);

    pre_filtered
}
