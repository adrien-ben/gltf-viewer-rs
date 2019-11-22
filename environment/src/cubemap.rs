use super::{
    create_descriptors, create_env_pipeline, create_render_pass, get_view_matrices,
    EnvPipelineParameters, SkyboxModel, SkyboxVertex,
};
use cgmath::{Deg, Matrix4};
use math::*;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use util::*;
use vulkan::ash::{version::DeviceV1_0, vk};
use vulkan::{Context, Texture};

enum CubemapTexturePath {
    SixFaces(SixFaces),
    Equirectangular(String, u32),
}

struct SixFaces {
    px: String,
    nx: String,
    py: String,
    ny: String,
    pz: String,
    nz: String,
}

pub(crate) fn create_skybox_cubemap<P: Into<String>>(
    context: &Arc<Context>,
    path: Option<P>,
) -> Texture {
    let path = match path {
        Some(path) => CubemapTexturePath::Equirectangular(path.into(), 1024),
        _ => CubemapTexturePath::SixFaces(SixFaces {
            px: String::from("assets/env/px.hdr"),
            nx: String::from("assets/env/nx.hdr"),
            py: String::from("assets/env/py.hdr"),
            ny: String::from("assets/env/ny.hdr"),
            pz: String::from("assets/env/pz.hdr"),
            nz: String::from("assets/env/nz.hdr"),
        }),
    };
    create_cubemap(context, path)
}

fn create_cubemap(context: &Arc<Context>, path: CubemapTexturePath) -> Texture {
    use CubemapTexturePath::*;
    match path {
        Equirectangular(path, size) => {
            create_cubemap_from_equirectangular_texture(context, path, size)
        }
        SixFaces(six_faces) => create_cubemap_from_six_faces(context, six_faces),
    }
}

fn create_cubemap_from_equirectangular_texture<P: AsRef<Path>>(
    context: &Arc<Context>,
    path: P,
    size: u32,
) -> Texture {
    log::info!("Creating cubemap from equirectangular texture");
    let start = Instant::now();
    let device = context.device();

    let image = RgbaHdrImageData::load(path);
    let mipmaps = image.get_mipmaps();

    let pixels = {
        let mut pixels = Vec::new();
        pixels.extend_from_slice(&image.pixels);
        mipmaps
            .iter()
            .for_each(|mipmap| pixels.extend_from_slice(&mipmap.pixels));
        pixels
    };

    let texture = Texture::from_rgba_32(context, image.width, image.height, &pixels);

    let mip_levels = compute_mipmap_levels(size, size);
    let cubemap = Texture::create_renderable_cubemap(context, size, mip_levels);

    let skybox_model = SkyboxModel::new(context);

    let render_pass = create_render_pass(context, vk::Format::R32G32B32A32_SFLOAT);

    let descriptors = create_descriptors(context, &texture);

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
                    fragment_shader_name: "spherical",
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

    let mut views = Vec::new();
    let mut framebuffers = Vec::new();
    for mip_level in 0..mip_levels {
        let mip_factor = 1.0_f32 / (2.0_f32.powi(mip_level as i32));
        let viewport_size = (size as f32 * mip_factor) as u32;

        let lod_views = (0..6)
            .map(|i| {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(cubemap.image.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: mip_level,
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

            let mut extent = vk::Extent2D {
                width: size,
                height: size,
            };

            for mip_level in 0..mip_levels {
                unsafe {
                    let source_image_lod = (mip_level as f32 / (mip_levels as f32 - 1.0)) * mipmaps.len() as f32;

                    let push = any_as_u8_slice(&source_image_lod);
                    device.cmd_push_constants(
                        buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        size_of::<Matrix4<f32>>() as _,
                        push,
                    );
                };

                for face in 0..6 {
                    let framebuffer = framebuffers[mip_level as usize][face];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(render_pass)
                        .framebuffer(framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent,
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

                    let scissor = [vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: size,
                            height: size,
                        },
                    }];
                    unsafe { device.cmd_set_scissor(buffer, 0, &scissor) };

                    let viewport = [vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: extent.width as _,
                        height: extent.height as _,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }];
                    unsafe { device.cmd_set_viewport(buffer, 0, &viewport) };

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
                    unsafe { device.cmd_end_render_pass(buffer) };
                }

                extent.width /= 2;
                extent.height /= 2;
            }
        }
    });

    cubemap.image.transition_image_layout(
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
    log::info!(
        "Finished creating cubemap from equirectangular texture. Took {} ms",
        time
    );

    cubemap
}

fn create_cubemap_from_six_faces(context: &Arc<Context>, six_faces: SixFaces) -> Texture {
    let image_px = RgbaHdrImageData::load(six_faces.px);
    let image_nx = RgbaHdrImageData::load(six_faces.nx);
    let image_py = RgbaHdrImageData::load(six_faces.py);
    let image_ny = RgbaHdrImageData::load(six_faces.ny);
    let image_pz = RgbaHdrImageData::load(six_faces.pz);
    let image_nz = RgbaHdrImageData::load(six_faces.nz);

    let mipmaps_px = image_px.get_mipmaps();
    let mipmaps_nx = image_nx.get_mipmaps();
    let mipmaps_py = image_py.get_mipmaps();
    let mipmaps_ny = image_ny.get_mipmaps();
    let mipmaps_pz = image_pz.get_mipmaps();
    let mipmaps_nz = image_nz.get_mipmaps();

    let width = image_px.width;
    let mip_levels = mipmaps_px.len() + 1;

    let mut data = Vec::new();
    data.extend_from_slice(&image_px.pixels);
    data.extend_from_slice(&image_nx.pixels);
    data.extend_from_slice(&image_py.pixels);
    data.extend_from_slice(&image_ny.pixels);
    data.extend_from_slice(&image_pz.pixels);
    data.extend_from_slice(&image_nz.pixels);

    for level in 0..(mip_levels - 1) {
        data.extend_from_slice(&mipmaps_px[level].pixels);
        data.extend_from_slice(&mipmaps_nx[level].pixels);
        data.extend_from_slice(&mipmaps_py[level].pixels);
        data.extend_from_slice(&mipmaps_ny[level].pixels);
        data.extend_from_slice(&mipmaps_pz[level].pixels);
        data.extend_from_slice(&mipmaps_nz[level].pixels);
    }

    Texture::create_cubemap_from_data(&context, width, mip_levels as u32, &data)
}
