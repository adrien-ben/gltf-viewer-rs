mod attachments;
mod fullscreen;
mod model;
mod postprocess;
mod skybox;
mod ssao;

extern crate model as model_crate;

use self::attachments::Attachments;
use self::fullscreen::QuadModel;
use self::model::gbufferpass::GBufferPass;
pub use self::model::lightpass::{LightPass, OutputMode};
use self::model::{ModelData, ModelRenderer};
use self::ssao::*;
pub use self::{postprocess::*, skybox::*};

use super::camera::{Camera, CameraUBO};
use super::config::Config;
use super::gui::Gui;
use ash::{vk, Device};
use egui::{ClippedPrimitive, TextureId};
use egui_ash_renderer::{DynamicRendering, Options, Renderer as GuiRenderer};
use environment::Environment;
use math::cgmath::{Deg, Matrix4, SquareMatrix, Vector3};
use model_crate::Model;
use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use vulkan::ash::vk::{RenderingAttachmentInfo, RenderingInfo};
use vulkan::*;
use winit::window::Window;

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const DEFAULT_EMISSIVE_INTENSITY: f32 = 1.0;
const DEFAULT_SSAO_KERNEL_SIZE: u32 = 32;
const DEFAULT_SSAO_RADIUS: f32 = 0.15;
const DEFAULT_SSAO_STRENGTH: f32 = 1.0;
pub const DEFAULT_BLOOM_STRENGTH: f32 = 0.04;

pub enum RenderError {
    DirtySwapchain,
}

#[derive(Clone, Copy, Debug)]
pub struct RendererSettings {
    pub emissive_intensity: f32,
    pub ssao_enabled: bool,
    pub ssao_kernel_size: u32,
    pub ssao_radius: f32,
    pub ssao_strength: f32,
    pub tone_map_mode: ToneMapMode,
    pub output_mode: OutputMode,
    pub bloom_strength: f32,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            emissive_intensity: DEFAULT_EMISSIVE_INTENSITY,
            ssao_enabled: true,
            ssao_kernel_size: DEFAULT_SSAO_KERNEL_SIZE,
            ssao_radius: DEFAULT_SSAO_RADIUS,
            ssao_strength: DEFAULT_SSAO_STRENGTH,
            tone_map_mode: ToneMapMode::Default,
            output_mode: OutputMode::Final,
            bloom_strength: DEFAULT_BLOOM_STRENGTH,
        }
    }
}

pub struct Renderer {
    settings: RendererSettings,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    swapchain: Swapchain,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
    environment: Environment,
    camera_uniform_buffers: Vec<Buffer>,
    attachments: Attachments,
    skybox_renderer: SkyboxRenderer,
    model_renderer: Option<ModelRenderer>,
    ssao_pass: SSAOPass,
    ssao_blur_pass: BlurPass,
    quad_model: QuadModel,
    bloom_pass: BloomPass,
    final_pass: FinalPass,
    gui_renderer: GuiRenderer,
    context: Arc<Context>,
}

impl Renderer {
    pub fn create(
        context: Arc<Context>,
        config: &Config,
        settings: RendererSettings,
        environment: Environment,
    ) -> Self {
        let swapchain_support_details = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );

        let resolution = [config.resolution().width(), config.resolution().height()];
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(resolution, config.vsync());
        let depth_format = find_depth_format(&context);
        let msaa_samples = context.get_max_usable_sample_count(config.msaa());
        log::debug!(
            "msaa: {:?} - preferred was {:?}",
            msaa_samples,
            config.msaa()
        );

        let swapchain = Swapchain::create(
            Arc::clone(&context),
            swapchain_support_details,
            resolution,
            config.vsync(),
        );

        let command_buffers = allocate_command_buffers(&context, swapchain.image_count());

        let in_flight_frames = create_sync_objects(&context);

        let camera_uniform_buffers =
            create_camera_uniform_buffers(&context, swapchain.image_count() as u32);

        let attachments = Attachments::new(
            &context,
            swapchain_properties.extent,
            depth_format,
            msaa_samples,
        );

        let skybox_renderer = SkyboxRenderer::create(
            Arc::clone(&context),
            &camera_uniform_buffers,
            &environment,
            msaa_samples,
            depth_format,
        );

        let ssao_pass = SSAOPass::create(
            Arc::clone(&context),
            &attachments.gbuffer_normals,
            &attachments.gbuffer_depth,
            &camera_uniform_buffers,
            settings,
        );

        let ssao_blur_pass = BlurPass::create(Arc::clone(&context), &attachments.ssao);

        let quad_model = QuadModel::new(&context);

        let bloom_pass = BloomPass::create(Arc::clone(&context), &attachments);

        let final_pass = FinalPass::create(
            Arc::clone(&context),
            swapchain_properties.format.format,
            &attachments,
            settings,
        );

        let gui_renderer = GuiRenderer::with_default_allocator(
            context.instance(),
            context.physical_device(),
            context.device().clone(),
            DynamicRendering {
                color_attachment_format: swapchain_properties.format.format,
                depth_attachment_format: None,
            },
            Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT as _,
                ..Default::default()
            },
        )
        .expect("Failed to create gui renderer");

        Self {
            context,
            settings,
            depth_format,
            msaa_samples,
            swapchain,
            command_buffers,
            in_flight_frames,
            environment,
            camera_uniform_buffers,
            attachments,
            skybox_renderer,
            model_renderer: None,
            ssao_pass,
            ssao_blur_pass,
            quad_model,
            bloom_pass,
            final_pass,
            gui_renderer,
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

        if !self.in_flight_frames.gui_textures_to_free.is_empty() {
            self.gui_renderer
                .free_textures(&self.in_flight_frames.gui_textures_to_free)
                .unwrap();
        }

        let render_data = gui.render(window); // TODO: free textures

        self.in_flight_frames.gui_textures_to_free = render_data.textures_delta.free;

        self.gui_renderer
            .set_textures(
                self.context.graphics_compute_queue(),
                self.context.transient_command_pool(),
                &render_data.textures_delta.set,
            )
            .unwrap();

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

            self.cmd_draw(
                command_buffer,
                frame_index,
                render_data.pixels_per_point,
                &render_data.clipped_primitives,
            );

            // End command buffer
            unsafe {
                self.context
                    .device()
                    .end_command_buffer(command_buffer)
                    .unwrap()
            };
        }

        self.update_ubos(image_index as _, camera);

        // Submit command buffer
        {
            let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(image_available_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);

            let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(render_finished_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);

            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::builder()
                .command_buffer(self.command_buffers[image_index as usize]);

            let submit_info = vk::SubmitInfo2::builder()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
                .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info));

            unsafe {
                self.context
                    .synchronization2()
                    .queue_submit2(
                        self.context.graphics_compute_queue(),
                        std::slice::from_ref(&submit_info),
                        in_flight_fence,
                    )
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain.swapchain_khr()];
        let images_indices = [image_index];

        {
            let signal_semaphores = [render_finished_semaphore];

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
        pixels_per_point: f32,
        gui_primitives: &[ClippedPrimitive],
    ) {
        if self.settings.ssao_enabled {
            // GBuffer pass
            {
                // Prepare attachments for gbuffer
                cmd_transition_images_layouts(
                    command_buffer,
                    &[
                        LayoutTransition {
                            image: &self.attachments.gbuffer_normals.image,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            mips_range: MipsRange::All,
                        },
                        LayoutTransition {
                            image: &self.attachments.gbuffer_depth.image,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            mips_range: MipsRange::All,
                        },
                    ],
                );

                let extent = vk::Extent2D {
                    width: self.attachments.gbuffer_normals.image.extent.width,
                    height: self.attachments.gbuffer_normals.image.extent.height,
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

                let color_attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(self.attachments.gbuffer_normals.view)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let depth_attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    })
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .image_view(self.attachments.gbuffer_depth.view)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment_info))
                    .depth_attachment(&depth_attachment_info)
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });

                unsafe {
                    self.context
                        .dynamic_rendering()
                        .cmd_begin_rendering(command_buffer, &rendering_info)
                };

                if let Some(renderer) = self.model_renderer.as_ref() {
                    renderer
                        .gbuffer_pass
                        .cmd_draw(command_buffer, frame_index, &renderer.data);
                }

                unsafe {
                    self.context
                        .dynamic_rendering()
                        .cmd_end_rendering(command_buffer)
                };
            }

            // Prepare attachments and inputs for ssao
            cmd_transition_images_layouts(
                command_buffer,
                &[
                    LayoutTransition {
                        image: &self.attachments.gbuffer_normals.image,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        mips_range: MipsRange::All,
                    },
                    LayoutTransition {
                        image: &self.attachments.gbuffer_depth.image,
                        old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        mips_range: MipsRange::All,
                    },
                    LayoutTransition {
                        image: &self.attachments.ssao.image,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        mips_range: MipsRange::All,
                    },
                ],
            );

            // SSAO Pass
            self.ssao_pass.cmd_draw(
                command_buffer,
                &self.attachments,
                &self.quad_model,
                frame_index,
            );

            // Prepare attachments and inputs for ssao blur
            cmd_transition_images_layouts(
                command_buffer,
                &[
                    LayoutTransition {
                        image: &self.attachments.ssao.image,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        mips_range: MipsRange::All,
                    },
                    LayoutTransition {
                        image: &self.attachments.ssao_blur.image,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        mips_range: MipsRange::All,
                    },
                ],
            );

            // SSAO Blur Pass
            self.ssao_blur_pass
                .cmd_draw(command_buffer, &self.attachments, &self.quad_model);
        }

        // Prepare attachments and inputs for lighting pass
        let mut transitions = vec![
            LayoutTransition {
                image: &self.attachments.get_scene_resolved_color().image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                mips_range: MipsRange::All,
            },
            LayoutTransition {
                image: &self.attachments.scene_depth.image,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                mips_range: MipsRange::All,
            },
        ];
        if self.settings.ssao_enabled {
            transitions.push(LayoutTransition {
                image: &self.attachments.ssao_blur.image,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mips_range: MipsRange::All,
            });
        }
        cmd_transition_images_layouts(command_buffer, &transitions);

        // Scene Pass
        {
            let extent = vk::Extent2D {
                width: self.attachments.scene_color.image.extent.width,
                height: self.attachments.scene_color.image.extent.height,
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
                let mut color_attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(self.attachments.scene_color.view)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                if let Some(resolve_attachment) = self.attachments.scene_resolve.as_ref() {
                    color_attachment_info = color_attachment_info
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE_KHR)
                        .resolve_image_view(resolve_attachment.view)
                }

                let depth_attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    })
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .image_view(self.attachments.scene_depth.view)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment_info))
                    .depth_attachment(&depth_attachment_info)
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });

                unsafe {
                    self.context
                        .dynamic_rendering()
                        .cmd_begin_rendering(command_buffer, &rendering_info)
                };
            }

            self.skybox_renderer.cmd_draw(command_buffer, frame_index);

            if let Some(renderer) = self.model_renderer.as_ref() {
                renderer
                    .light_pass
                    .cmd_draw(command_buffer, frame_index, &renderer.data);
            }

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_end_rendering(command_buffer)
            };
        }

        // Bloom pass
        {
            self.bloom_pass
                .cmd_draw(command_buffer, &self.attachments, &self.quad_model);
        }

        // Prepare attachments and inputs for final pass (post-processing + ui)
        {
            self.swapchain.images()[frame_index].cmd_transition_image_layout(
                command_buffer,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }

        // Final pass and UI
        {
            let extent = self.swapchain.properties().extent;

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
                let color_attachment_info = RenderingAttachmentInfo::builder()
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(self.swapchain.image_views()[frame_index])
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = RenderingInfo::builder()
                    .color_attachments(std::slice::from_ref(&color_attachment_info))
                    .layer_count(1)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });

                unsafe {
                    self.context
                        .dynamic_rendering()
                        .cmd_begin_rendering(command_buffer, &rendering_info)
                };
            }

            // Apply post process
            self.final_pass.cmd_draw(command_buffer, &self.quad_model);

            // Draw UI
            self.gui_renderer
                .cmd_draw(command_buffer, extent, pixels_per_point, gui_primitives)
                .unwrap();

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_end_rendering(command_buffer)
            };
        }

        // Transition swapchain image for presentation
        {
            self.swapchain.images()[frame_index].cmd_transition_image_layout(
                command_buffer,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        }
    }

    pub fn set_model(&mut self, model: &Rc<RefCell<Model>>) {
        let model_data = ModelData::create(
            Arc::clone(&self.context),
            Rc::downgrade(model),
            self.swapchain.image_count() as u32,
        );

        let ao_map = self
            .settings
            .ssao_enabled
            .then(|| &self.attachments.ssao_blur);

        if let Some(model_renderer) = self.model_renderer.as_mut() {
            model_renderer
                .gbuffer_pass
                .set_model(&model_data, &self.camera_uniform_buffers);

            model_renderer.light_pass.set_model(
                &model_data,
                &self.camera_uniform_buffers,
                &self.environment,
                ao_map,
            );

            model_renderer.data = model_data;
        } else {
            let gbuffer_pass = GBufferPass::create(
                Arc::clone(&self.context),
                &model_data,
                &self.camera_uniform_buffers,
                self.depth_format,
            );

            let light_pass = LightPass::create(
                Arc::clone(&self.context),
                &model_data,
                &self.camera_uniform_buffers,
                &self.environment,
                ao_map,
                self.msaa_samples,
                self.depth_format,
                self.settings,
            );

            self.model_renderer = Some(ModelRenderer {
                data: model_data,
                gbuffer_pass,
                light_pass,
            });
        }
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
        let swapchain_properties = self.swapchain.properties();

        self.attachments = Attachments::new(
            &self.context,
            swapchain_properties.extent,
            self.depth_format,
            self.msaa_samples,
        );

        // SSAO
        self.ssao_pass.set_inputs(
            &self.attachments.gbuffer_normals,
            &self.attachments.gbuffer_depth,
        );

        // SSAO Blur
        self.ssao_blur_pass.set_input_image(&self.attachments.ssao);

        // Model
        if let Some(renderer) = self.model_renderer.as_mut() {
            let ao_map = if self.settings.ssao_enabled {
                Some(&self.attachments.ssao_blur)
            } else {
                None
            };
            renderer.light_pass.set_ao_map(ao_map);
        }

        // Bloom
        self.bloom_pass.set_attachments(&self.attachments);

        // Final
        self.final_pass.set_attachments(&self.attachments);
    }

    pub fn update_settings(&mut self, settings: RendererSettings) {
        log::debug!("Updating renderer settings");
        self.context.graphics_queue_wait_idle();
        if (self.settings.emissive_intensity - settings.emissive_intensity).abs() > f32::EPSILON {
            self.set_emissive_intensity(settings.emissive_intensity);
        }
        if self.settings.tone_map_mode != settings.tone_map_mode {
            self.set_tone_map_mode(settings.tone_map_mode);
        }
        if self.settings.output_mode != settings.output_mode {
            self.set_output_mode(settings.output_mode);
        }
        if self.settings.ssao_enabled != settings.ssao_enabled {
            self.enabled_ssao(settings.ssao_enabled);
        }
        if self.settings.ssao_kernel_size != settings.ssao_kernel_size {
            self.set_ssao_kernel_size(settings.ssao_kernel_size);
        }
        if (self.settings.ssao_radius - settings.ssao_radius).abs() > f32::EPSILON {
            self.set_ssao_radius(settings.ssao_radius);
        }
        if (self.settings.ssao_strength - settings.ssao_strength).abs() > f32::EPSILON {
            self.set_ssao_strength(settings.ssao_strength);
        }
        if (self.settings.bloom_strength - settings.bloom_strength).abs() > f32::EPSILON {
            self.set_bloom_strength(settings.bloom_strength);
        }
    }

    fn set_emissive_intensity(&mut self, emissive_intensity: f32) {
        self.settings.emissive_intensity = emissive_intensity;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer
                .light_pass
                .set_emissive_intensity(emissive_intensity);
        }
    }

    fn set_tone_map_mode(&mut self, tone_map_mode: ToneMapMode) {
        self.settings.tone_map_mode = tone_map_mode;
        self.final_pass.set_tone_map_mode(tone_map_mode);
    }

    fn set_output_mode(&mut self, output_mode: OutputMode) {
        self.settings.output_mode = output_mode;
        if let Some(renderer) = self.model_renderer.as_mut() {
            renderer.light_pass.set_output_mode(output_mode);
        }
    }

    fn enabled_ssao(&mut self, enable: bool) {
        if self.settings.ssao_enabled != enable {
            self.settings.ssao_enabled = enable;
            if let Some(renderer) = self.model_renderer.as_mut() {
                let ao_map = enable.then(|| &self.attachments.ssao_blur);
                renderer.light_pass.set_ao_map(ao_map);
            }
        }
    }

    fn set_ssao_kernel_size(&mut self, size: u32) {
        self.settings.ssao_kernel_size = size;
        self.ssao_pass.set_ssao_kernel_size(size);
    }

    fn set_ssao_radius(&mut self, radius: f32) {
        self.settings.ssao_radius = radius;
        self.ssao_pass.set_ssao_radius(radius);
    }

    fn set_ssao_strength(&mut self, strength: f32) {
        self.settings.ssao_strength = strength;
        self.ssao_pass.set_ssao_strength(strength);
    }

    fn set_bloom_strength(&mut self, strength: f32) {
        self.settings.bloom_strength = strength;
        self.final_pass.set_bloom_strength(strength);
    }

    pub fn update_ubos(&mut self, frame_index: usize, camera: Camera) {
        // Camera
        {
            let extent = self.swapchain.properties().extent;
            let aspect = extent.width as f32 / extent.height as f32;

            let view = Matrix4::look_at_rh(
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
        self.destroy_swapchain();
    }
}

#[derive(Copy, Clone)]
struct RendererPipelineParameters<'a> {
    vertex_shader_name: &'static str,
    fragment_shader_name: &'static str,
    vertex_shader_specialization: Option<&'a vk::SpecializationInfo>,
    fragment_shader_specialization: Option<&'a vk::SpecializationInfo>,
    msaa_samples: vk::SampleCountFlags,
    color_attachment_formats: &'a [vk::Format],
    depth_attachment_format: Option<vk::Format>,
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

    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

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

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    create_pipeline::<V>(
        context,
        PipelineParameters {
            vertex_shader_params,
            fragment_shader_params,
            multisampling_info: &multisampling_info,
            viewport_info: &viewport_info,
            rasterizer_info: &rasterizer_info,
            dynamic_state_info: Some(&dynamic_state_info),
            depth_stencil_info: Some(params.depth_stencil_info),
            color_blend_attachments: params.color_blend_attachments,
            color_attachment_formats: params.color_attachment_formats,
            depth_attachment_format: params.depth_attachment_format,
            layout: params.layout,
            parent: params.parent,
            allow_derivatives: params.parent.is_none(),
        },
    )
}

struct InFlightFrames {
    context: Arc<Context>,
    sync_objects: Vec<SyncObjects>,
    gui_textures_to_free: Vec<TextureId>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(context: Arc<Context>, sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            context,
            sync_objects,
            gui_textures_to_free: Vec::new(),
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
