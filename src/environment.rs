use crate::{
    math::*,
    util::*,
    vulkan::{
        create_device_local_buffer_with_data, create_pipeline, Buffer, Context, Descriptors,
        PipelineParameters, Texture, Vertex,
    },
};
use ash::{version::DeviceV1_0, vk};
use cgmath::{Deg, Matrix4, Point3, Vector3};
use std::{mem::size_of, path::Path, rc::Rc, time::Instant};

pub struct Environment {
    skybox: Texture,
    irradiance: Texture,
    pre_filtered: Texture,
    brdf_lookup: Texture,
}

impl Environment {
    pub fn new<P: Into<String>>(context: &Rc<Context>, path: Option<P>) -> Self {
        let skybox = create_skybox_cubemap(&context, path);
        let irradiance = create_irradiance_map(&context, &skybox, 32);
        let pre_filtered = create_pre_filtered_map(&context, &skybox, 512);
        let brdf_lookup = create_brdf_lookup(&context, 512);

        Self {
            skybox,
            irradiance,
            pre_filtered,
            brdf_lookup,
        }
    }
}

impl Environment {
    pub fn skybox(&self) -> &Texture {
        &self.skybox
    }

    pub fn irradiance(&self) -> &Texture {
        &self.irradiance
    }

    pub fn pre_filtered(&self) -> &Texture {
        &self.pre_filtered
    }

    pub fn brdf_lookup(&self) -> &Texture {
        &self.brdf_lookup
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SkyboxVertex {
    position: [f32; 3],
}

impl Vertex for SkyboxVertex {
    fn get_bindings_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<SkyboxVertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attributes_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        }]
    }
}

pub struct SkyboxModel {
    vertices: Buffer,
    indices: Buffer,
}

impl SkyboxModel {
    pub fn new(context: &Rc<Context>) -> Self {
        let indices: [u32; 36] = [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ];
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );
        let vertices: [f32; 24] = [
            -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5,
            0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
        ];
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        SkyboxModel { vertices, indices }
    }
}

impl SkyboxModel {
    pub fn vertices(&self) -> &Buffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Buffer {
        &self.indices
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct QuadVertex {
    position: [f32; 2],
    coords: [f32; 2],
}

impl Vertex for QuadVertex {
    fn get_bindings_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<QuadVertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attributes_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
        ]
    }
}

struct QuadModel {
    vertices: Buffer,
    indices: Buffer,
}

impl QuadModel {
    fn new(context: &Rc<Context>) -> Self {
        let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );
        let vertices: [f32; 16] = [
            -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0,
        ];
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        Self { vertices, indices }
    }
}

pub enum CubemapTexturePath {
    SixFaces {
        px: String,
        nx: String,
        py: String,
        ny: String,
        pz: String,
        nz: String,
    },
    Equirectangular(String, u32),
}

fn create_skybox_cubemap<P: Into<String>>(context: &Rc<Context>, path: Option<P>) -> Texture {
    let path = match path {
        Some(path) => CubemapTexturePath::Equirectangular(path.into(), 1024),
        _ => CubemapTexturePath::SixFaces {
            px: String::from("assets/env/px.hdr"),
            nx: String::from("assets/env/nx.hdr"),
            py: String::from("assets/env/py.hdr"),
            ny: String::from("assets/env/ny.hdr"),
            pz: String::from("assets/env/pz.hdr"),
            nz: String::from("assets/env/nz.hdr"),
        },
    };
    create_cubemap(context, path)
}

fn create_cubemap(context: &Rc<Context>, path: CubemapTexturePath) -> Texture {
    use CubemapTexturePath::*;
    match path {
        Equirectangular(path, size) => {
            create_cubemap_from_equirectangular_texture(context, path, size)
        }
        SixFaces {
            px,
            nx,
            py,
            ny,
            pz,
            nz,
        } => {
            let (w, h, px) = load_hdr_image(px);
            let (_, _, nx) = load_hdr_image(nx);
            let (_, _, py) = load_hdr_image(py);
            let (_, _, ny) = load_hdr_image(ny);
            let (_, _, pz) = load_hdr_image(pz);
            let (_, _, nz) = load_hdr_image(nz);

            let data_size = (w * h * 4 * 6) as usize;
            let mut data = Vec::with_capacity(data_size);
            data.extend_from_slice(&px);
            data.extend_from_slice(&nx);
            data.extend_from_slice(&py);
            data.extend_from_slice(&ny);
            data.extend_from_slice(&pz);
            data.extend_from_slice(&nz);

            Texture::create_cubemap_from_data(&context, w, &data)
        }
    }
}

fn create_cubemap_from_equirectangular_texture<P: AsRef<Path>>(
    context: &Rc<Context>,
    path: P,
    size: u32,
) -> Texture {
    log::info!("Creating cubemap from equirectangular texture");
    let start = Instant::now();
    let device = context.device();
    let (w, h, data) = load_hdr_image(path);
    let mip_levels = (size as f32).log2().floor() as u32 + 1;

    let texture = Texture::from_rgba_32(context, w, h, &data);
    let cubemap = Texture::create_renderable_cubemap(context, size, mip_levels);

    let skybox_model = SkyboxModel::new(context);

    let render_pass = create_render_pass(context, vk::Format::R32G32B32A32_SFLOAT);

    let descriptors = create_descriptors(context, &texture);

    let (pipeline_layout, pipeline) = {
        let layout = {
            let layouts = [descriptors.layout()];
            let push_constant_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<Matrix4<f32>>() as _,
            }];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
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
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewports)
                .scissors(&scissors);

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

            create_env_pipeline::<SkyboxVertex>(
                context,
                EnvPipelineParameters {
                    vertex_shader_name: "cubemap",
                    fragment_shader_name: "spherical",
                    viewport_info: &viewport_info,
                    rasterizer_info: &rasterizer_info,
                    dynamic_state_info: None,
                    layout,
                    render_pass,
                },
            )
        };

        (layout, pipeline)
    };

    let views = (0..6)
        .map(|i| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(cubemap.image.image)
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

    let framebuffers = views
        .iter()
        .map(|view| {
            let attachments = [*view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(size)
                .height(size)
                .layers(1);
            unsafe { device.create_framebuffer(&create_info, None).unwrap() }
        })
        .collect::<Vec<_>>();

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

            for face in 0..6 {
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffers[face])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: size,
                            height: size,
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
        }
    });

    cubemap.image.transition_image_layout(
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    cubemap.image.generate_mipmaps(vk::Extent2D {
        width: size,
        height: size,
    });

    // Cleanup
    unsafe {
        views
            .iter()
            .for_each(|v| device.destroy_image_view(*v, None));
        framebuffers
            .iter()
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

fn create_irradiance_map(context: &Rc<Context>, cubemap: &Texture, size: u32) -> Texture {
    log::info!("Creating irradiance map");
    let start = Instant::now();

    let device = context.device();

    let skybox_model = SkyboxModel::new(context);

    let render_pass = create_render_pass(context, vk::Format::R32G32B32A32_SFLOAT);

    let descriptors = create_descriptors(context, &cubemap);

    let (pipeline_layout, pipeline) = {
        let layout = {
            let layouts = [descriptors.layout()];
            let push_constant_range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<Matrix4<f32>>() as _,
            }];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
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
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewports)
                .scissors(&scissors);

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

            create_env_pipeline::<SkyboxVertex>(
                context,
                EnvPipelineParameters {
                    vertex_shader_name: "cubemap",
                    fragment_shader_name: "irradiance",
                    viewport_info: &viewport_info,
                    rasterizer_info: &rasterizer_info,
                    dynamic_state_info: None,
                    layout,
                    render_pass,
                },
            )
        };

        (layout, pipeline)
    };

    // create cubemap
    let irradiance_map = Texture::create_renderable_cubemap(context, size, 1);

    let views = (0..6)
        .map(|i| {
            let create_info = vk::ImageViewCreateInfo::builder()
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

    let framebuffers = views
        .iter()
        .map(|view| {
            let attachments = [*view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(size)
                .height(size)
                .layers(1);
            unsafe { device.create_framebuffer(&create_info, None).unwrap() }
        })
        .collect::<Vec<_>>();

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

            for face in 0..6 {
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffers[face])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: size,
                            height: size,
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
        framebuffers
            .iter()
            .for_each(|fb| device.destroy_framebuffer(*fb, None));
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);
    }

    let time = start.elapsed().as_millis();
    log::info!("Finished creating irradiance map. Took {} ms", time);

    irradiance_map
}

fn create_pre_filtered_map(context: &Rc<Context>, cubemap: &Texture, size: u32) -> Texture {
    log::info!("Creating pre-filtered map");
    let start = Instant::now();

    let device = context.device();

    let skybox_model = SkyboxModel::new(context);

    let max_mip_levels = (size as f32).log2().floor() as u32 + 1;

    let render_pass = create_render_pass(context, vk::Format::R32G32B32A32_SFLOAT);

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
    let pre_filtered = Texture::create_renderable_cubemap(context, size, max_mip_levels);

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
                    .format(vk::Format::R32G32B32A32_SFLOAT)
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

fn create_brdf_lookup(context: &Rc<Context>, size: u32) -> Texture {
    log::info!("Creating brdf lookup");
    let start = Instant::now();

    let device = context.device();

    let quad_model = QuadModel::new(&context);

    let render_pass = create_render_pass(context, vk::Format::R16G16_SFLOAT);

    let (pipeline_layout, pipeline) = {
        let layout = {
            let layout_info = vk::PipelineLayoutCreateInfo::builder();

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline = {
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: size as _,
                height: size as _,
                min_depth: 0.0,
                max_depth: 1.0,
            }];

            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            }];
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewports)
                .scissors(&scissors);

            let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0);

            create_env_pipeline::<QuadVertex>(
                context,
                EnvPipelineParameters {
                    vertex_shader_name: "brdf_lookup",
                    fragment_shader_name: "brdf_lookup",
                    viewport_info: &viewport_info,
                    rasterizer_info: &rasterizer_info,
                    dynamic_state_info: None,
                    layout,
                    render_pass,
                },
            )
        };

        (layout, pipeline)
    };

    let lookup =
        Texture::create_renderable_texture(&context, size, size, vk::Format::R16G16_SFLOAT);

    let framebuffer = {
        let attachments = [lookup.view];
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(size)
            .height(size)
            .layers(1);
        unsafe { device.create_framebuffer(&create_info, None).unwrap() }
    };

    // Render
    context.execute_one_time_commands(|buffer| {
        {
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: size,
                        height: size,
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

            unsafe { device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline) };

            unsafe {
                device.cmd_bind_vertex_buffers(buffer, 0, &[quad_model.vertices.buffer], &[0]);
            }

            unsafe {
                device.cmd_bind_index_buffer(
                    buffer,
                    quad_model.indices.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            // Draw quad
            unsafe { device.cmd_draw_indexed(buffer, 6, 1, 0, 0, 0) };

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer) };
        }
    });

    lookup.image.transition_image_layout(
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    // Cleanup
    unsafe {
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);
    }

    let time = start.elapsed().as_millis();
    log::info!("Finished creating brdf lookup. Took {} ms", time);

    lookup
}

fn get_view_matrices() -> [Matrix4<f32>; 6] {
    [
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ),
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    ]
}

fn create_render_pass(context: &Rc<Context>, format: vk::Format) -> vk::RenderPass {
    let attachments_descs = [vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let color_attachment_ref = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descs = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_ref)
        .build()];

    let subpass_deps = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    unsafe {
        context
            .device()
            .create_render_pass(&render_pass_info, None)
            .unwrap()
    }
}

fn create_descriptors(context: &Rc<Context>, texture: &Texture) -> Descriptors {
    let device = context.device();

    let layout = {
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
    };

    let pool = {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    };

    let sets = {
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

        let cubemap_info = [vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.view)
            .sampler(texture.sampler.unwrap())
            .build()];

        let descriptor_writes = [vk::WriteDescriptorSet::builder()
            .dst_set(sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&cubemap_info)
            .build()];

        unsafe {
            context
                .device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }

        sets
    };

    Descriptors::new(Rc::clone(context), layout, pool, sets)
}

#[derive(Copy, Clone)]
struct EnvPipelineParameters<'a> {
    vertex_shader_name: &'static str,
    fragment_shader_name: &'static str,
    viewport_info: &'a vk::PipelineViewportStateCreateInfo,
    rasterizer_info: &'a vk::PipelineRasterizationStateCreateInfo,
    dynamic_state_info: Option<&'a vk::PipelineDynamicStateCreateInfo>,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

fn create_env_pipeline<V: Vertex>(
    context: &Rc<Context>,
    params: EnvPipelineParameters,
) -> vk::Pipeline {
    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build();

    create_pipeline::<V>(
        context,
        PipelineParameters {
            vertex_shader_name: params.vertex_shader_name,
            fragment_shader_name: params.fragment_shader_name,
            multisampling_info: &multisampling_info,
            viewport_info: params.viewport_info,
            rasterizer_info: params.rasterizer_info,
            dynamic_state_info: params.dynamic_state_info,
            depth_stencil_info: None,
            color_blend_attachment: &color_blend_attachment,
            render_pass: params.render_pass,
            layout: params.layout,
            parent: None,
            allow_derivatives: false,
        },
    )
}
