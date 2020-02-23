mod model;
mod postprocess;
mod renderpass;
mod skybox;

extern crate model as model_crate;

pub use self::{model::*, postprocess::*, renderpass::*, skybox::*};
use super::camera::{Camera, CameraUBO};
use ash::{version::DeviceV1_0, vk};
use environment::Environment;
use imgui::{Context as GuiContext, DrawData};
use imgui_rs_vulkan_renderer::Renderer as GuiRenderer;
use math::cgmath::{Deg, Matrix4, Vector3};
use model_crate::Model;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Weak;
use std::sync::Arc;
use vulkan::*;

pub struct Renderer {
    context: Arc<Context>,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    swapchain_properties: SwapchainProperties,
    environment: Environment,
    camera_uniform_buffers: Vec<Buffer>,
    renderer_render_pass: RenderPass,
    offscreen_framebuffer: vk::Framebuffer,
    skybox_renderer: SkyboxRenderer,
    model_renderer: Option<ModelRenderer>,
    post_process_renderer: PostProcessRenderer,
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

        let renderer_render_pass = RenderPass::create(
            Arc::clone(&context),
            swapchain_properties.extent,
            depth_format,
            msaa_samples,
        );

        let offscreen_framebuffer = renderer_render_pass.create_framebuffer();

        let skybox_renderer = SkyboxRenderer::create(
            Arc::clone(&context),
            &camera_uniform_buffers,
            swapchain_properties,
            &environment,
            msaa_samples,
            &renderer_render_pass,
        );

        let tone_map_mode = ToneMapMode::Default;
        let post_process_renderer = PostProcessRenderer::create(
            Arc::clone(&context),
            swapchain_properties,
            &simple_render_pass,
            renderer_render_pass.get_color_attachment(),
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
            renderer_render_pass,
            offscreen_framebuffer,
            skybox_renderer,
            model_renderer: None,
            post_process_renderer,
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

        // begin render pass
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
                .render_pass(self.renderer_render_pass.get_render_pass())
                .framebuffer(self.offscreen_framebuffer)
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

        if let Some(model_renderer) = self.model_renderer.as_ref() {
            model_renderer.cmd_draw(command_buffer, frame_index);
        }

        // End render pass
        unsafe { device.cmd_end_render_pass(command_buffer) };

        // begin render pass
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
        self.post_process_renderer.cmd_draw(command_buffer);

        // Draw UI
        self.gui_renderer
            .cmd_draw::<Context>(self.context.borrow(), command_buffer, draw_data)
            .unwrap();

        // End render pass
        unsafe { device.cmd_end_render_pass(command_buffer) };
    }

    pub fn set_model(&mut self, model: Weak<RefCell<Model>>) {
        self.model_renderer.take();

        let model_renderer = ModelRenderer::create(
            Arc::clone(&self.context),
            model,
            &self.camera_uniform_buffers,
            self.swapchain_properties,
            &self.environment,
            self.msaa_samples,
            &self.renderer_render_pass,
            self.output_mode,
            self.emissive_intensity,
        );

        self.model_renderer = Some(model_renderer);
    }

    pub fn on_new_swapchain(
        &mut self,
        swapchain_properties: SwapchainProperties,
        simple_render_pass: &SimpleRenderPass,
    ) {
        unsafe {
            self.context
                .device()
                .destroy_framebuffer(self.offscreen_framebuffer, None)
        };

        let renderer_render_pass = RenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            self.depth_format,
            self.msaa_samples,
        );

        let offscreen_framebuffer = renderer_render_pass.create_framebuffer();

        self.skybox_renderer.rebuild_pipeline(
            swapchain_properties,
            self.msaa_samples,
            &renderer_render_pass,
        );

        if let Some(model_renderer) = self.model_renderer.as_mut() {
            model_renderer.rebuild_pipelines(
                swapchain_properties,
                self.msaa_samples,
                &renderer_render_pass,
                self.output_mode,
                self.emissive_intensity,
            );
        }

        let post_process_renderer = PostProcessRenderer::create(
            Arc::clone(&self.context),
            swapchain_properties,
            simple_render_pass,
            renderer_render_pass.get_color_attachment(),
            self.tone_map_mode,
        );

        self.swapchain_properties = swapchain_properties;
        self.renderer_render_pass = renderer_render_pass;
        self.offscreen_framebuffer = offscreen_framebuffer;
        self.post_process_renderer = post_process_renderer;
    }

    pub fn set_emissive_intensity(&mut self, emissive_intensity: f32) {
        self.emissive_intensity = emissive_intensity;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.rebuild_pipelines(
                self.swapchain_properties,
                self.msaa_samples,
                &self.renderer_render_pass,
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
        self.post_process_renderer.rebuild_pipelines(
            self.swapchain_properties,
            simple_render_pass,
            tone_map_mode,
        );
    }

    pub fn set_output_mode(&mut self, output_mode: OutputMode) {
        self.output_mode = output_mode;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.rebuild_pipelines(
                self.swapchain_properties,
                self.msaa_samples,
                &self.renderer_render_pass,
                output_mode,
                self.emissive_intensity,
            );
        }
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
            let proj = math::perspective(Deg(45.0), aspect, 0.01, 10.0);

            let ubos = [CameraUBO::new(view, proj, camera.position())];
            let buffer = &mut self.camera_uniform_buffers[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy(data_ptr, &ubos);
            }
        }

        // model
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.update_buffers(frame_index)
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
                .destroy_framebuffer(self.offscreen_framebuffer, None)
        };
    }
}

#[derive(Copy, Clone)]
struct RendererPipelineParameters<'a> {
    shader_name: &'static str,
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
        .map_or(ShaderParameters::new(params.shader_name), |s| {
            ShaderParameters::specialized(params.shader_name, s)
        });
    let fragment_shader_params = params
        .fragment_shader_specialization
        .map_or(ShaderParameters::new(params.shader_name), |s| {
            ShaderParameters::specialized(params.shader_name, s)
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
