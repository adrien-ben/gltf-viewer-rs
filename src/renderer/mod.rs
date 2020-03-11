mod fullscreen;
mod model;
mod postprocess;
mod skybox;
mod ssao;

extern crate model as model_crate;

use self::fullscreen::QuadModel;
use self::model::gbufferpass::{GBufferPass, GBufferRenderPass};
pub use self::model::lightpass::{LightPass, LightRenderPass, OutputMode};
use self::model::{ModelData, ModelRenderer};
use self::ssao::{SSAOPass, SSAORenderPass};
pub use self::{postprocess::*, skybox::*};

use super::camera::{Camera, CameraUBO};
use ash::{version::DeviceV1_0, vk};
use environment::Environment;
use imgui::{Context as GuiContext, DrawData};
use imgui_rs_vulkan_renderer::Renderer as GuiRenderer;
use math::cgmath::{Deg, Matrix4, SquareMatrix, Vector3};
use model_crate::Model;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use vulkan::*;

const DEFAULT_SSAO_KERNEL_SIZE: u32 = 32;
const DEFAULT_SSAO_RADIUS: f32 = 0.5;

// TODO: at some point I'll need to put vulkan's render passes and frame buffers into the pass structure
// TODO: try and remember why I did not
pub struct Renderer {
    context: Arc<Context>,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    swapchain_properties: SwapchainProperties,
    environment: Environment,
    camera_uniform_buffers: Vec<Buffer>,
    light_render_pass: LightRenderPass,
    lightpass_framebuffer: vk::Framebuffer,
    skybox_renderer: SkyboxRenderer,
    gbuffer_render_pass: GBufferRenderPass,
    gbuffer_framebuffer: vk::Framebuffer,
    model_renderer: Option<ModelRenderer>,
    ssao_render_pass: SSAORenderPass,
    ssao_framebuffer: vk::Framebuffer,
    ssao_pass: SSAOPass,
    quad_model: QuadModel,
    final_pass: FinalPass,
    gui_renderer: GuiRenderer,
    output_mode: OutputMode,
    emissive_intensity: f32,
    tone_map_mode: ToneMapMode,
}

impl Renderer {
    pub fn create(
        context: Arc<Context>,
        depth_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
        swapchain_properties: SwapchainProperties,
        simple_render_pass: &SimpleRenderPass,
        environment: Environment,
        gui_context: &mut GuiContext,
    ) -> Self {
        let camera_uniform_buffers =
            create_camera_uniform_buffers(&context, swapchain_properties.image_count);

        let light_render_pass = LightRenderPass::create(
            Arc::clone(&context),
            swapchain_properties.extent,
            depth_format,
            msaa_samples,
        );

        let lightpass_framebuffer = light_render_pass.create_framebuffer();

        let skybox_renderer = SkyboxRenderer::create(
            Arc::clone(&context),
            &camera_uniform_buffers,
            swapchain_properties,
            &environment,
            msaa_samples,
            &light_render_pass,
        );

        let gbuffer_render_pass = GBufferRenderPass::create(
            Arc::clone(&context),
            swapchain_properties.extent,
            depth_format,
        );

        let gbuffer_framebuffer = gbuffer_render_pass.create_framebuffer();

        let ssao_render_pass =
            SSAORenderPass::create(Arc::clone(&context), swapchain_properties.extent);
        let ssao_framebuffer = ssao_render_pass.create_framebuffer();
        let ssao_pass = SSAOPass::create(
            Arc::clone(&context),
            swapchain_properties,
            &ssao_render_pass,
            gbuffer_render_pass.get_normals_attachment(),
            gbuffer_render_pass.get_depth_attachment(),
            &camera_uniform_buffers,
            DEFAULT_SSAO_KERNEL_SIZE,
            DEFAULT_SSAO_RADIUS,
        );

        let quad_model = QuadModel::new(&context);

        let tone_map_mode = ToneMapMode::Default;
        let final_pass = FinalPass::create(
            Arc::clone(&context),
            swapchain_properties,
            &simple_render_pass,
            light_render_pass.get_color_attachment(),
            tone_map_mode,
        );

        let gui_renderer = GuiRenderer::new::<Context>(
            context.borrow(),
            crate::viewer::MAX_FRAMES_IN_FLIGHT as _,
            simple_render_pass.get_render_pass(),
            gui_context,
        )
        .expect("Failed to create gui renderer");

        let output_mode = OutputMode::Final;

        Self {
            context,
            depth_format,
            msaa_samples,
            swapchain_properties,
            environment,
            camera_uniform_buffers,
            light_render_pass,
            lightpass_framebuffer,
            skybox_renderer,
            gbuffer_render_pass,
            gbuffer_framebuffer,
            model_renderer: None,
            ssao_render_pass,
            ssao_framebuffer,
            ssao_pass,
            quad_model,
            final_pass,
            gui_renderer,
            output_mode,
            emissive_intensity: 1.0,
            tone_map_mode,
        }
    }
}

fn create_camera_uniform_buffers(context: &Arc<Context>, count: u32) -> Vec<Buffer> {
    (0..count)
        .map(|_| {
            let mut buffer = Buffer::create(
                Arc::clone(context),
                size_of::<CameraUBO>() as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffer.map_memory();
            buffer
        })
        .collect::<Vec<_>>()
}

impl Renderer {
    pub fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        swapchain_properties: SwapchainProperties,
        simple_render_pass: &SimpleRenderPass,
        final_framebuffer: vk::Framebuffer,
        draw_data: &DrawData,
    ) {
        let device = self.context.device();

        // GBuffer pass
        {
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
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(self.gbuffer_render_pass.get_render_pass())
                    .framebuffer(self.gbuffer_framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
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

            if let Some(renderer) = self.model_renderer.as_ref() {
                renderer
                    .gbuffer_pass
                    .cmd_draw(command_buffer, frame_index, &renderer.data);
            }

            unsafe { device.cmd_end_render_pass(command_buffer) };
        }

        // SSAO Pass
        {
            {
                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(self.ssao_render_pass.get_render_pass())
                    .framebuffer(self.ssao_framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
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

            self.ssao_pass
                .cmd_draw(command_buffer, &self.quad_model, frame_index);

            unsafe { device.cmd_end_render_pass(command_buffer) };
        }

        // Scene Pass
        {
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
                    .render_pass(self.light_render_pass.get_render_pass())
                    .framebuffer(self.lightpass_framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
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

            self.skybox_renderer.cmd_draw(command_buffer, frame_index);

            if let Some(renderer) = self.model_renderer.as_ref() {
                renderer
                    .light_pass
                    .cmd_draw(command_buffer, frame_index, &renderer.data);
            }

            unsafe { device.cmd_end_render_pass(command_buffer) };
        }

        // Final pass and UI
        {
            {
                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 1.0, 1.0],
                    },
                }];
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(simple_render_pass.get_render_pass())
                    .framebuffer(final_framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
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

            // Apply post process
            self.final_pass.cmd_draw(command_buffer, &self.quad_model);

            // Draw UI
            self.gui_renderer
                .cmd_draw::<Context>(self.context.borrow(), command_buffer, draw_data)
                .unwrap();

            unsafe { device.cmd_end_render_pass(command_buffer) };
        }
    }

    pub fn set_model(&mut self, model: &Rc<RefCell<Model>>) {
        self.model_renderer.take();

        let model_data = ModelData::create(
            Arc::clone(&self.context),
            Rc::downgrade(model),
            self.swapchain_properties,
        );

        let gbuffer_pass = GBufferPass::create(
            Arc::clone(&self.context),
            &model_data,
            &self.camera_uniform_buffers,
            self.swapchain_properties,
            &self.gbuffer_render_pass,
        );

        let light_pass = LightPass::create(
            Arc::clone(&self.context),
            &model_data,
            &self.camera_uniform_buffers,
            self.swapchain_properties,
            &self.environment,
            &self.ssao_render_pass.get_ao_attachment(),
            self.msaa_samples,
            &self.light_render_pass,
            self.output_mode,
            self.emissive_intensity,
        );

        self.model_renderer = Some(ModelRenderer {
            data: model_data,
            gbuffer_pass,
            light_pass,
        });
    }

    pub fn on_new_swapchain(
        &mut self,
        swapchain_properties: SwapchainProperties,
        simple_render_pass: &SimpleRenderPass,
    ) {
        unsafe {
            self.context
                .device()
                .destroy_framebuffer(self.lightpass_framebuffer, None);
            self.context
                .device()
                .destroy_framebuffer(self.ssao_framebuffer, None);
            self.context
                .device()
                .destroy_framebuffer(self.gbuffer_framebuffer, None);
        }

        // GBuffer
        let gbuffer_render_pass = GBufferRenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            self.depth_format,
        );
        let gbuffer_framebuffer = gbuffer_render_pass.create_framebuffer();

        // SSAO
        let ssao_render_pass =
            SSAORenderPass::create(Arc::clone(&self.context), swapchain_properties.extent);
        let ssao_framebuffer = ssao_render_pass.create_framebuffer();
        self.ssao_pass.set_inputs(
            gbuffer_render_pass.get_normals_attachment(),
            gbuffer_render_pass.get_depth_attachment(),
        );
        self.ssao_pass
            .rebuild_pipelines(swapchain_properties, &ssao_render_pass);

        // Light
        let light_render_pass = LightRenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            self.depth_format,
            self.msaa_samples,
        );
        let lightpass_framebuffer = light_render_pass.create_framebuffer();

        // Skybox
        self.skybox_renderer.rebuild_pipeline(
            swapchain_properties,
            self.msaa_samples,
            &light_render_pass,
        );

        // Model
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer
                .gbuffer_pass
                .rebuild_pipelines(swapchain_properties, &gbuffer_render_pass);

            renderer
                .light_pass
                .set_ao_map(ssao_render_pass.get_ao_attachment());

            renderer.light_pass.rebuild_pipelines(
                &renderer.data,
                swapchain_properties,
                self.msaa_samples,
                &light_render_pass,
                self.output_mode,
                self.emissive_intensity,
            )
        }

        // Final
        self.final_pass
            .set_input_image(light_render_pass.get_color_attachment());
        self.final_pass.rebuild_pipelines(
            swapchain_properties,
            simple_render_pass,
            self.tone_map_mode,
        );

        self.swapchain_properties = swapchain_properties;
        self.gbuffer_render_pass = gbuffer_render_pass;
        self.gbuffer_framebuffer = gbuffer_framebuffer;
        self.ssao_render_pass = ssao_render_pass;
        self.ssao_framebuffer = ssao_framebuffer;
        self.light_render_pass = light_render_pass;
        self.lightpass_framebuffer = lightpass_framebuffer;
    }

    pub fn set_emissive_intensity(&mut self, emissive_intensity: f32) {
        self.emissive_intensity = emissive_intensity;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.light_pass.rebuild_pipelines(
                &renderer.data,
                self.swapchain_properties,
                self.msaa_samples,
                &self.light_render_pass,
                self.output_mode,
                emissive_intensity,
            );
        }
    }

    pub fn set_tone_map_mode(
        &mut self,
        simple_render_pass: &SimpleRenderPass,
        tone_map_mode: ToneMapMode,
    ) {
        self.tone_map_mode = tone_map_mode;
        self.final_pass.rebuild_pipelines(
            self.swapchain_properties,
            simple_render_pass,
            tone_map_mode,
        );
    }

    pub fn set_output_mode(&mut self, output_mode: OutputMode) {
        self.output_mode = output_mode;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.light_pass.rebuild_pipelines(
                &renderer.data,
                self.swapchain_properties,
                self.msaa_samples,
                &self.light_render_pass,
                output_mode,
                self.emissive_intensity,
            );
        }
    }

    pub fn set_ssao_kernel_size(&mut self, size: u32) {
        self.ssao_pass.set_ssao_kernel_size(size);
        self.ssao_pass
            .rebuild_pipelines(self.swapchain_properties, &self.ssao_render_pass);
    }

    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao_pass.set_ssao_radius(radius);
        self.ssao_pass
            .rebuild_pipelines(self.swapchain_properties, &self.ssao_render_pass);
    }

    pub fn update_ubos(&mut self, frame_index: usize, camera: Camera) {
        // Camera
        {
            let aspect = self.swapchain_properties.extent.width as f32
                / self.swapchain_properties.extent.height as f32;

            let view = Matrix4::look_at(
                camera.position(),
                camera.target(),
                Vector3::new(0.0, 1.0, 0.0),
            );
            let proj = math::perspective(Deg(45.0), aspect, 0.01, 100.0);
            let inverted_proj = proj.invert().unwrap();

            let ubos = [CameraUBO::new(view, proj, inverted_proj, camera.position())];
            let buffer = &mut self.camera_uniform_buffers[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy(data_ptr, &ubos);
            }
        }

        // model
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.data.update_buffers(frame_index);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.gui_renderer
            .destroy::<Context>(self.context.borrow())
            .expect("Failed to destroy renderer");
        unsafe {
            self.context
                .device()
                .destroy_framebuffer(self.lightpass_framebuffer, None);
            self.context
                .device()
                .destroy_framebuffer(self.ssao_framebuffer, None);
            self.context
                .device()
                .destroy_framebuffer(self.gbuffer_framebuffer, None);
        }
    }
}

#[derive(Copy, Clone)]
struct RendererPipelineParameters<'a> {
    vertex_shader_name: &'static str,
    fragment_shader_name: &'static str,
    vertex_shader_specialization: Option<&'a vk::SpecializationInfo>,
    fragment_shader_specialization: Option<&'a vk::SpecializationInfo>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    subpass: u32,
    layout: vk::PipelineLayout,
    depth_stencil_info: &'a vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    enable_face_culling: bool,
    parent: Option<vk::Pipeline>,
}

fn create_renderer_pipeline<V: Vertex>(
    context: &Arc<Context>,
    params: RendererPipelineParameters,
) -> vk::Pipeline {
    let vertex_shader_params = params
        .vertex_shader_specialization
        .map_or(ShaderParameters::new(params.vertex_shader_name), |s| {
            ShaderParameters::specialized(params.vertex_shader_name, s)
        });
    let fragment_shader_params = params
        .fragment_shader_specialization
        .map_or(ShaderParameters::new(params.fragment_shader_name), |s| {
            ShaderParameters::specialized(params.fragment_shader_name, s)
        });

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(params.msaa_samples)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: params.swapchain_properties.extent.width as _,
        height: params.swapchain_properties.extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: params.swapchain_properties.extent,
    }];
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let cull_mode = if params.enable_face_culling {
        vk::CullModeFlags::BACK
    } else {
        vk::CullModeFlags::NONE
    };

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    create_pipeline::<V>(
        context,
        PipelineParameters {
            vertex_shader_params,
            fragment_shader_params,
            multisampling_info: &multisampling_info,
            viewport_info: &viewport_info,
            rasterizer_info: &rasterizer_info,
            dynamic_state_info: None,
            depth_stencil_info: Some(params.depth_stencil_info),
            color_blend_attachments: params.color_blend_attachments,
            render_pass: params.render_pass,
            subpass: params.subpass,
            layout: params.layout,
            parent: params.parent,
            allow_derivatives: params.parent.is_none(),
        },
    )
}
