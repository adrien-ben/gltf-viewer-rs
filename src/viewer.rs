use crate::{camera::*, config::Config, controls::*, gui::Gui, loader::*, renderer::*};
use ash::{version::DeviceV1_0, vk, Device};
use environment::*;
use model::{Model, PlaybackMode};
use std::{cell::RefCell, path::Path, rc::Rc, sync::Arc, time::Instant};
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct Viewer {
    config: Config,
    event_loop: EventLoop<()>,
    window: Window,
    run: bool,

    camera: Camera,
    input_state: InputState,
    model: Option<Rc<RefCell<Model>>>,

    gui: Gui,

    context: Arc<Context>,
    simple_render_pass: SimpleRenderPass,
    swapchain: Swapchain,

    renderer: Renderer,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,

    loader: Loader,
}

impl Viewer {
    pub fn new<P: AsRef<Path>>(config: Config, enable_debug: bool, path: Option<P>) -> Self {
        log::debug!("Creating application.");

        let resolution = [config.resolution().width(), config.resolution().height()];

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("GLTF Viewer")
            .with_inner_size(PhysicalSize::new(resolution[0], resolution[1]))
            .build(&event_loop)
            .unwrap();

        let mut gui = Gui::new(&window);

        let context = Arc::new(Context::new(&window, enable_debug));

        let swapchain_support_details = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(resolution, config.vsync());
        let depth_format = Self::find_depth_format(&context);
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

        let environment =
            Environment::new(&context, config.env().path(), config.env().resolution());

        let renderer = Renderer::create(
            Arc::clone(&context),
            depth_format,
            msaa_samples,
            swapchain_properties,
            &simple_render_pass,
            environment,
            gui.get_context(),
        );

        let command_buffers = Self::allocate_command_buffers(&context, swapchain.image_count());

        let in_flight_frames = Self::create_sync_objects(context.device());

        let loader = Loader::new(Arc::new(context.new_thread()));
        if let Some(p) = path {
            loader.load(p.as_ref().to_path_buf());
        }

        Self {
            event_loop,
            window,
            config,
            run: true,
            camera: Default::default(),
            input_state: Default::default(),
            model: None,
            gui,
            context,
            simple_render_pass,
            swapchain,
            renderer,
            command_buffers,
            in_flight_frames,
            loader,
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

    fn create_sync_objects(device: &Device) -> InFlightFrames {
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
                let fence_info =
                    vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    pub fn run(self) {
        log::debug!("Running application.");

        let Viewer {
            event_loop,
            window,
            config,
            mut run,
            mut camera,
            mut input_state,
            mut model,
            mut gui,
            context,
            simple_render_pass,
            mut swapchain,
            mut renderer,
            mut command_buffers,
            mut in_flight_frames,
            loader,
        } = self;

        let mut time = Instant::now();
        let mut dirty_swapchain = false;

        // Main loop
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            gui.handle_event(&window, &event);
            input_state = input_state.update(&event);

            match event {
                Event::NewEvents(_) => {
                    input_state.reset();
                    gui.update_delta_time();
                }
                // End of event processing
                Event::MainEventsCleared => {
                    // If swapchain must be recreated wait for windows to not be minimized anymore
                    if dirty_swapchain {
                        let PhysicalSize { width, height } = window.inner_size();
                        if width > 0 && height > 0 {
                            // recreate_swapchain()
                            {
                                log::debug!("Recreating swapchain.");

                                unsafe { context.device().device_wait_idle().unwrap() };

                                // cleanup_swapchain();
                                {
                                    let device = context.device();
                                    unsafe {
                                        device.free_command_buffers(
                                            context.general_command_pool(),
                                            &command_buffers,
                                        );
                                    }
                                    swapchain.destroy();
                                }

                                let dimensions = window.inner_size().into();

                                let swapchain_support_details = SwapchainSupportDetails::new(
                                    context.physical_device(),
                                    context.surface(),
                                    context.surface_khr(),
                                );
                                let swapchain_properties = swapchain_support_details
                                    .get_ideal_swapchain_properties(dimensions, config.vsync());

                                renderer
                                    .on_new_swapchain(swapchain_properties, &simple_render_pass);

                                swapchain = Swapchain::create(
                                    Arc::clone(&context),
                                    swapchain_support_details,
                                    dimensions,
                                    config.vsync(),
                                    &simple_render_pass,
                                );

                                command_buffers = Self::allocate_command_buffers(
                                    &context,
                                    swapchain.image_count(),
                                );
                            }

                            dirty_swapchain = false;
                        } else {
                            return;
                        }
                    }

                    let new_time = Instant::now();
                    let delta_s = ((new_time - time).as_nanos() as f64) / 1_000_000_000.0;
                    time = new_time;

                    gui.prepare_frame(&window);

                    // load_new_model()
                    if let Some(loaded_model) = loader.get_model() {
                        gui.set_model_metadata(loaded_model.metadata().clone());
                        model.take();

                        context.graphics_queue_wait_idle();
                        let loaded_model = Rc::new(RefCell::new(loaded_model));
                        renderer.set_model(&loaded_model);
                        model = Some(loaded_model);
                    }

                    // update_model()
                    if let Some(model) = model.as_ref() {
                        let mut model = model.borrow_mut();

                        if gui.should_toggle_animation() {
                            model.toggle_animation();
                        } else if gui.should_stop_animation() {
                            model.stop_animation();
                        } else if gui.should_reset_animation() {
                            model.reset_animation();
                        } else {
                            let playback_mode = if gui.is_infinite_animation_checked() {
                                PlaybackMode::LOOP
                            } else {
                                PlaybackMode::ONCE
                            };

                            model.set_animation_playback_mode(playback_mode);
                            model.set_current_animation(gui.get_selected_animation());
                        }
                        gui.set_animation_playback_state(model.get_animation_playback_state());

                        let delta_s = delta_s as f32 * gui.get_animation_speed();
                        model.update(delta_s);
                    }

                    // update_camera()
                    {
                        if gui.should_reset_camera() {
                            camera = Default::default();
                        }

                        if !gui.is_hovered() {
                            camera.update(&input_state);
                            gui.set_camera(Some(camera));
                        }
                    }

                    // fn update_renderer_settings()
                    {
                        if let Some(emissive_intensity) = gui.get_new_emissive_intensity() {
                            context.graphics_queue_wait_idle();
                            renderer.set_emissive_intensity(emissive_intensity);
                        }
                        if let Some(ssao_enabled) = gui.get_new_ssao_enabled() {
                            context.graphics_queue_wait_idle();
                            renderer.enabled_ssao(ssao_enabled);
                        }
                        if let Some(ssao_kernel_size) = gui.get_new_ssao_kernel_size() {
                            context.graphics_queue_wait_idle();
                            renderer.set_ssao_kernel_size(ssao_kernel_size);
                        }
                        if let Some(ssao_radius) = gui.get_new_ssao_radius() {
                            context.graphics_queue_wait_idle();
                            renderer.set_ssao_radius(ssao_radius);
                        }
                        if let Some(ssao_strength) = gui.get_new_ssao_strength() {
                            context.graphics_queue_wait_idle();
                            renderer.set_ssao_strength(ssao_strength);
                        }
                        if let Some(tone_map_mode) = gui.get_new_renderer_tone_map_mode() {
                            context.graphics_queue_wait_idle();
                            renderer.set_tone_map_mode(&simple_render_pass, tone_map_mode);
                        }
                        if let Some(output_mode) = gui.get_new_renderer_output_mode() {
                            context.graphics_queue_wait_idle();
                            renderer.set_output_mode(output_mode);
                        }
                    }

                    // draw_frame()
                    {
                        log::trace!("Drawing frame.");
                        let sync_objects = in_flight_frames.next().unwrap();
                        let image_available_semaphore = sync_objects.image_available_semaphore;
                        let render_finished_semaphore = sync_objects.render_finished_semaphore;
                        let in_flight_fence = sync_objects.fence;
                        let wait_fences = [in_flight_fence];

                        unsafe {
                            context
                                .device()
                                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                                .unwrap()
                        };

                        let result = swapchain.acquire_next_image(
                            None,
                            Some(image_available_semaphore),
                            None,
                        );
                        let image_index = match result {
                            Ok((image_index, _)) => image_index,
                            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                                dirty_swapchain = true;
                                return;
                            }
                            Err(error) => {
                                panic!("Error while acquiring next image. Cause: {}", error)
                            }
                        };

                        unsafe { context.device().reset_fences(&wait_fences).unwrap() };

                        // record_command_buffer
                        {
                            let command_buffer = command_buffers[image_index as usize];
                            let frame_index = image_index as _;
                            let device = context.device();

                            unsafe {
                                device
                                    .reset_command_buffer(
                                        command_buffer,
                                        vk::CommandBufferResetFlags::empty(),
                                    )
                                    .unwrap();
                            }

                            // begin command buffer
                            {
                                let command_buffer_begin_info =
                                    vk::CommandBufferBeginInfo::builder()
                                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                                unsafe {
                                    device
                                        .begin_command_buffer(
                                            command_buffer,
                                            &command_buffer_begin_info,
                                        )
                                        .unwrap()
                                };
                            }

                            let draw_data = gui.render(&mut run, &window);

                            renderer.cmd_draw(
                                command_buffer,
                                frame_index,
                                swapchain.properties(),
                                &simple_render_pass,
                                swapchain.framebuffers()[frame_index],
                                draw_data,
                            );

                            // End command buffer
                            unsafe { device.end_command_buffer(command_buffer).unwrap() };
                        }

                        renderer.update_ubos(image_index as _, camera);

                        let device = context.device();
                        let wait_semaphores = [image_available_semaphore];
                        let signal_semaphores = [render_finished_semaphore];

                        // Submit command buffer
                        {
                            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                            let command_buffers = [command_buffers[image_index as usize]];
                            let submit_info = vk::SubmitInfo::builder()
                                .wait_semaphores(&wait_semaphores)
                                .wait_dst_stage_mask(&wait_stages)
                                .command_buffers(&command_buffers)
                                .signal_semaphores(&signal_semaphores)
                                .build();
                            let submit_infos = [submit_info];
                            unsafe {
                                device
                                    .queue_submit(
                                        context.graphics_queue(),
                                        &submit_infos,
                                        in_flight_fence,
                                    )
                                    .unwrap()
                            };
                        }

                        let swapchains = [swapchain.swapchain_khr()];
                        let images_indices = [image_index];

                        {
                            let present_info = vk::PresentInfoKHR::builder()
                                .wait_semaphores(&signal_semaphores)
                                .swapchains(&swapchains)
                                .image_indices(&images_indices);
                            let result = swapchain.present(&present_info);
                            match result {
                                Ok(is_suboptimal) if is_suboptimal => {
                                    dirty_swapchain = true;
                                }
                                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                                    dirty_swapchain = true;
                                }
                                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                                _ => {}
                            }
                        }
                    }
                }
                // Window event
                Event::WindowEvent { event, .. } => {
                    match event {
                        // Dropped file
                        WindowEvent::DroppedFile(path) => {
                            log::debug!("File dropped: {:?}", path);
                            loader.load(path);
                        }
                        // Resizing
                        WindowEvent::Resized(new_size) => {
                            log::debug!("Window was resized. New size is {:?}", new_size);
                            dirty_swapchain = true;
                        }
                        // Exit
                        WindowEvent::CloseRequested => run = false,
                        _ => (),
                    }
                }
                // Cleanup
                Event::LoopDestroyed => {
                    log::info!("Stopping application");
                    let device = context.device();
                    // self.cleanup_swapchain();
                    {
                        unsafe { device.device_wait_idle().unwrap() };
                        unsafe {
                            device.free_command_buffers(
                                context.general_command_pool(),
                                &command_buffers,
                            );
                        }
                        swapchain.destroy();
                    }
                    in_flight_frames.destroy(device);
                }
                // Ignored
                _ => (),
            }

            if !run {
                *control_flow = ControlFlow::Exit;
            }
        });
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

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(&device));
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
