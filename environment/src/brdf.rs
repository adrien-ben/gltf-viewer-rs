use super::{create_env_pipeline, create_render_pass, EnvPipelineParameters};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;
use vulkan::ash::{version::DeviceV1_0, vk};
use vulkan::{create_device_local_buffer_with_data, Buffer, Context, Texture, Vertex};

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct QuadVertex {
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
    fn new(context: &Arc<Context>) -> Self {
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

pub(crate) fn create_brdf_lookup(context: &Arc<Context>, size: u32) -> Texture {
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
