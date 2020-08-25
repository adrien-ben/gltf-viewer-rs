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
use self::ssao::*;
pub use self::{postprocess::*, skybox::*};

use super::camera::{Camera, CameraUBO};
use super::config::Config;
use super::gui::Gui;
use ash::{version::DeviceV1_0, vk, Device};
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
use winit::window::Window;

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub enum RenderError {
    DirtySwapchain,
}

// TODO: at some point I'll need to put vulkan's render passes and frame buffers into the pass structure
// TODO: try and remember why I did not
pub struct Renderer {
    context: Arc<Context>,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    simple_render_pass: SimpleRenderPass,
    swapchain: Swapchain,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
    environment: Environment,
    camera_uniform_buffers: Vec<Buffer>,
    light_render_pass: LightRenderPass,
    lightpass_framebuffer: vk::Framebuffer,
    skybox_renderer: SkyboxRenderer,
    gbuffer_render_pass: GBufferRenderPass,
    gbuffer_framebuffer: vk::Framebuffer,
    model_renderer: Option<ModelRenderer>,
    ssao_pass: SSAOPass,
    ssao_blur_pass: BlurPass,
    quad_model: QuadModel,
    final_pass: FinalPass,
    gui_renderer: GuiRenderer,
    output_mode: OutputMode,
    emissive_intensity: f32,
    tone_map_mode: ToneMapMode,
    ssao_enabled: bool,
}

impl Renderer {
    pub fn create(
        context: Arc<Context>,
        config: &Config,
        environment: Environment,
        gui_context: &mut GuiContext,
    ) -> Self {
        let resolution = [config.resolution().width(), config.resolution().height()];

        let swapchain_support_details = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(resolution, config.vsync());
        let depth_format = find_depth_format(&context);
        let msaa_samples = context.get_max_usable_sample_count(config.msaa());
        log::debug!("msaa: {:?} - preferred was {}", msaa_samples, config.msaa());

        let simple_render_pass =
            SimpleRenderPass::create(Arc::clone(&context), swapchain_properties.format.format);

        let swapchain = Swapchain::create(
            Arc::clone(&context),
            swapchain_support_details,
            resolution,
            config.vsync(),
            &simple_render_pass,
        );

        let command_buffers = allocate_command_buffers(&context, swapchain.image_count());

        let in_flight_frames = create_sync_objects(&context);

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

        let ssao_pass = SSAOPass::create(
            Arc::clone(&context),
            swapchain_properties,
            gbuffer_render_pass.get_normals_attachment(),
            gbuffer_render_pass.get_depth_attachment(),
            &camera_uniform_buffers,
        );

        let ssao_blur_pass = BlurPass::create(
            Arc::clone(&context),
            swapchain_properties,
            ssao_pass.get_output(),
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
            MAX_FRAMES_IN_FLIGHT as _,
            simple_render_pass.get_render_pass(),
            gui_context,
        )
        .expect("Failed to create gui renderer");

        let output_mode = OutputMode::Final;

        Self {
            context,
            depth_format,
            msaa_samples,
            simple_render_pass,
            swapchain,
            command_buffers,
            in_flight_frames,
            environment,
            camera_uniform_buffers,
            light_render_pass,
            lightpass_framebuffer,
            skybox_renderer,
            gbuffer_render_pass,
            gbuffer_framebuffer,
            model_renderer: None,
            ssao_pass,
            ssao_blur_pass,
            quad_model,
            final_pass,
            gui_renderer,
            output_mode,
            emissive_intensity: 1.0,
            tone_map_mode,
            ssao_enabled: true,
        }
    }
}

fn find_depth_format(context: &Context) -> vk::Format {
    let candidates = vec![
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    context
        .find_supported_format(
            &candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
        .expect("Failed to find a supported depth format")
}

fn allocate_command_buffers(context: &Context, count: usize) -> Vec<vk::CommandBuffer> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(context.general_command_pool())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count as _);

    unsafe {
        context
            .device()
            .allocate_command_buffers(&allocate_info)
            .unwrap()
    }
}

fn create_sync_objects(context: &Arc<Context>) -> InFlightFrames {
    let device = context.device();
    let mut sync_objects_vec = Vec::new();
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };

        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };

        let in_flight_fence = {
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            unsafe { device.create_fence(&fence_info, None).unwrap() }
        };

        let sync_objects = SyncObjects {
            image_available_semaphore,
            render_finished_semaphore,
            fence: in_flight_fence,
        };
        sync_objects_vec.push(sync_objects)
    }

    InFlightFrames::new(Arc::clone(context), sync_objects_vec)
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
    pub fn render(
        &mut self,
        window: &Window,
        camera: Camera,
        gui: &mut Gui,
    ) -> Result<(), RenderError> {
        log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.context
                .device()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = self
            .swapchain
            .acquire_next_image(None, Some(image_available_semaphore), None);
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Err(RenderError::DirtySwapchain);
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.context.device().reset_fences(&wait_fences).unwrap() };

        // record_command_buffer
        {
            let command_buffer = self.command_buffers[image_index as usize];
            let frame_index = image_index as _;

            unsafe {
                self.context
                    .device()
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                    .unwrap();
            }

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                unsafe {
                    self.context
                        .device()
                        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                        .unwrap()
                };
            }

            let draw_data = gui.render(&window);

            self.cmd_draw(command_buffer, frame_index, draw_data);

            // End command buffer
            unsafe {
                self.context
                    .device()
                    .end_command_buffer(command_buffer)
                    .unwrap()
            };
        }

        self.update_ubos(image_index as _, camera);

        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                self.context
                    .device()
                    .queue_submit(
                        self.context.graphics_queue(),
                        &submit_infos,
                        in_flight_fence,
                    )
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain.swapchain_khr()];
        let images_indices = [image_index];

        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices);

            match self.swapchain.present(&present_info) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Err(RenderError::DirtySwapchain)
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        Ok(())
    }

    fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        draw_data: &DrawData,
    ) {
        let device = self.context.device();

        let extent = self.swapchain.properties().extent;

        if self.ssao_enabled {
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
                            extent,
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
            self.ssao_pass
                .cmd_draw(command_buffer, &self.quad_model, frame_index);

            // SSAO Blur Pass
            self.ssao_blur_pass
                .cmd_draw(command_buffer, &self.quad_model);
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
                        extent,
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
                    .render_pass(self.simple_render_pass.get_render_pass())
                    .framebuffer(self.swapchain.framebuffers()[frame_index])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
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

        let swapchain_properties = self.swapchain.properties();

        let model_data = ModelData::create(
            Arc::clone(&self.context),
            Rc::downgrade(model),
            swapchain_properties,
        );

        let gbuffer_pass = GBufferPass::create(
            Arc::clone(&self.context),
            &model_data,
            &self.camera_uniform_buffers,
            swapchain_properties,
            &self.gbuffer_render_pass,
        );

        let ao_map = if self.ssao_enabled {
            Some(self.ssao_blur_pass.get_output())
        } else {
            None
        };
        let light_pass = LightPass::create(
            Arc::clone(&self.context),
            &model_data,
            &self.camera_uniform_buffers,
            swapchain_properties,
            &self.environment,
            ao_map,
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

    pub fn recreate_swapchain(&mut self, dimensions: [u32; 2], vsync: bool) {
        log::debug!("Recreating swapchain.");

        self.wait_idle_gpu();

        self.destroy_swapchain();

        let swapchain_support_details = SwapchainSupportDetails::new(
            self.context.physical_device(),
            self.context.surface(),
            self.context.surface_khr(),
        );

        self.swapchain = Swapchain::create(
            Arc::clone(&self.context),
            swapchain_support_details,
            dimensions,
            vsync,
            &self.simple_render_pass,
        );

        self.on_new_swapchain();

        self.command_buffers =
            allocate_command_buffers(&self.context, self.swapchain.image_count());
    }

    pub fn wait_idle_gpu(&self) {
        unsafe { self.context.device().device_wait_idle().unwrap() };
    }

    fn destroy_swapchain(&mut self) {
        unsafe {
            self.context
                .device()
                .free_command_buffers(self.context.general_command_pool(), &self.command_buffers);
        }
        self.swapchain.destroy();
    }

    fn on_new_swapchain(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_framebuffer(self.lightpass_framebuffer, None);
            self.context
                .device()
                .destroy_framebuffer(self.gbuffer_framebuffer, None);
        }

        let swapchain_properties = self.swapchain.properties();

        // GBuffer
        let gbuffer_render_pass = GBufferRenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            self.depth_format,
        );
        let gbuffer_framebuffer = gbuffer_render_pass.create_framebuffer();

        // SSAO
        self.ssao_pass.set_extent(swapchain_properties.extent);
        self.ssao_pass.set_inputs(
            gbuffer_render_pass.get_normals_attachment(),
            gbuffer_render_pass.get_depth_attachment(),
        );
        self.ssao_pass.rebuild_pipelines(swapchain_properties);

        // SSAO Blur
        self.ssao_blur_pass.set_extent(swapchain_properties.extent);
        self.ssao_blur_pass
            .set_input_image(self.ssao_pass.get_output());
        self.ssao_blur_pass.rebuild_pipelines(swapchain_properties);

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

            let ao_map = if self.ssao_enabled {
                Some(self.ssao_blur_pass.get_output())
            } else {
                None
            };
            renderer.light_pass.set_ao_map(ao_map);

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
            &self.simple_render_pass,
            self.tone_map_mode,
        );

        self.gbuffer_render_pass = gbuffer_render_pass;
        self.gbuffer_framebuffer = gbuffer_framebuffer;
        self.light_render_pass = light_render_pass;
        self.lightpass_framebuffer = lightpass_framebuffer;
    }

    pub fn set_emissive_intensity(&mut self, emissive_intensity: f32) {
        self.emissive_intensity = emissive_intensity;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.light_pass.rebuild_pipelines(
                &renderer.data,
                self.swapchain.properties(),
                self.msaa_samples,
                &self.light_render_pass,
                self.output_mode,
                emissive_intensity,
            );
        }
    }

    pub fn set_tone_map_mode(&mut self, tone_map_mode: ToneMapMode) {
        self.tone_map_mode = tone_map_mode;
        self.final_pass.rebuild_pipelines(
            self.swapchain.properties(),
            &self.simple_render_pass,
            tone_map_mode,
        );
    }

    pub fn set_output_mode(&mut self, output_mode: OutputMode) {
        self.output_mode = output_mode;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.light_pass.rebuild_pipelines(
                &renderer.data,
                self.swapchain.properties(),
                self.msaa_samples,
                &self.light_render_pass,
                output_mode,
                self.emissive_intensity,
            );
        }
    }

    pub fn enabled_ssao(&mut self, enable: bool) {
        if self.ssao_enabled != enable {
            self.ssao_enabled = enable;
            if let Some(renderer) = self.model_renderer.as_mut() {
                let ao_map = if enable {
                    Some(self.ssao_blur_pass.get_output())
                } else {
                    None
                };
                renderer.light_pass.set_ao_map(ao_map);
            }
        }
    }

    pub fn set_ssao_kernel_size(&mut self, size: u32) {
        self.ssao_pass.set_ssao_kernel_size(size);
        self.ssao_pass
            .rebuild_pipelines(self.swapchain.properties());
    }

    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao_pass.set_ssao_radius(radius);
        self.ssao_pass
            .rebuild_pipelines(self.swapchain.properties());
    }

    pub fn set_ssao_strength(&mut self, strength: f32) {
        self.ssao_pass.set_ssao_strength(strength);
        self.ssao_pass
            .rebuild_pipelines(self.swapchain.properties());
    }

    pub fn update_ubos(&mut self, frame_index: usize, camera: Camera) {
        // Camera
        {
            let extent = self.swapchain.properties().extent;
            let aspect = extent.width as f32 / extent.height as f32;

            let view = Matrix4::look_at(
                camera.position(),
                camera.target(),
                Vector3::new(0.0, 1.0, 0.0),
            );

            const Z_NEAR: f32 = 0.01;
            const Z_FAR: f32 = 100.0;
            let proj = math::perspective(Deg(45.0), aspect, Z_NEAR, Z_FAR);
            let inverted_proj = proj.invert().unwrap();

            let ubo = CameraUBO::new(view, proj, inverted_proj, camera.position(), Z_NEAR, Z_FAR);
            let buffer = &mut self.camera_uniform_buffers[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy(data_ptr, &[ubo]);
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
                .destroy_framebuffer(self.gbuffer_framebuffer, None);
            self.destroy_swapchain();
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

struct InFlightFrames {
    context: Arc<Context>,
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(context: Arc<Context>, sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            context,
            sync_objects,
            current_frame: 0,
        }
    }
}

impl Drop for InFlightFrames {
    fn drop(&mut self) {
        self.sync_objects
            .iter()
            .for_each(|o| o.destroy(self.context.device()));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}
