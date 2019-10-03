use crate::{camera::*, config::*, controls::*, loader::*, renderer::*};
use ash::{version::DeviceV1_0, vk, Device};
use environment::*;
use math;
use math::cgmath::{Deg, Matrix4, Point3, Vector3};
use std::{
    mem::size_of,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use vulkan::*;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct Viewer {
    config: Config,
    events_loop: EventsLoop,
    _window: Window,
    resize_dimensions: Option<[u32; 2]>,
    path_to_load: Option<PathBuf>,

    camera: Camera,
    input_state: InputState,

    context: Arc<Context>,
    swapchain_properties: SwapchainProperties,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    render_pass: RenderPass,
    swapchain: Swapchain,

    camera_uniform_buffers: Vec<Buffer>,
    environment: Environment,
    skybox_renderer: SkyboxRenderer,
    model_renderer: Option<ModelRenderer>,

    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,

    loader: Loader,
}

impl Viewer {
    pub fn new<P: AsRef<Path>>(config: Config, path: Option<P>) -> Self {
        log::debug!("Creating application.");

        let resolution = [config.resolution().width(), config.resolution().height()];

        let events_loop = EventsLoop::new();
        let window = WindowBuilder::new()
            .with_title("GLTF Viewer")
            .with_dimensions(LogicalSize::new(
                f64::from(resolution[0]),
                f64::from(resolution[1]),
            ))
            .build(&events_loop)
            .unwrap();

        let context = Arc::new(Context::new(&window));

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

        let render_pass = RenderPass::create(
            Arc::clone(&context),
            swapchain_properties.extent,
            swapchain_properties.format.format,
            depth_format,
            msaa_samples,
        );

        let swapchain = Swapchain::create(
            Arc::clone(&context),
            swapchain_support_details,
            resolution,
            config.vsync(),
            &render_pass,
        );

        let camera_uniform_buffers =
            Self::create_camera_uniform_buffers(&context, swapchain_properties.image_count);
        let environment = Environment::new(&context, config.env());

        let skybox_renderer = SkyboxRenderer::create(
            Arc::clone(&context),
            &camera_uniform_buffers,
            swapchain_properties,
            &environment,
            msaa_samples,
            &render_pass,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &context,
            &swapchain,
            render_pass.get_render_pass(),
            &skybox_renderer,
            None,
        );

        let in_flight_frames = Self::create_sync_objects(context.device());

        let loader = Loader::new(Arc::new(context.new_thread()));

        Self {
            events_loop,
            _window: window,
            config,
            resize_dimensions: None,
            path_to_load: path.map(|p| p.as_ref().to_path_buf()),
            camera: Default::default(),
            input_state: Default::default(),
            context,
            swapchain_properties,
            render_pass,
            swapchain,
            depth_format,
            msaa_samples,
            camera_uniform_buffers,
            environment,
            skybox_renderer,
            model_renderer: None,
            command_buffers,
            in_flight_frames,
            loader,
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

    fn create_and_register_command_buffers(
        context: &Context,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        skybox_renderer: &SkyboxRenderer,
        model_renderer: Option<&ModelRenderer>,
    ) -> Vec<vk::CommandBuffer> {
        let device = context.device();

        let buffers = {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(context.general_command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(swapchain.image_count() as _);

            unsafe { device.allocate_command_buffers(&allocate_info).unwrap() }
        };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = swapchain.framebuffers()[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                unsafe {
                    device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
                        .unwrap()
                };
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.7, 0.7, 0.7, 1.0],
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
                        extent: swapchain.properties().extent,
                    })
                    .clear_values(&clear_values);

                unsafe {
                    device.cmd_begin_render_pass(
                        buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                };
            }

            skybox_renderer.cmd_draw(buffer, i);

            if let Some(model_renderer) = model_renderer {
                model_renderer.cmd_draw(buffer, i);
            }

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End command buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
        });

        buffers
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

    pub fn run(&mut self) {
        log::debug!("Running application.");
        let mut time = Instant::now();
        loop {
            let new_time = Instant::now();
            let delta_s = ((new_time - time).as_nanos() as f64) / 1_000_000_000.0;
            time = new_time;

            if self.process_event() {
                break;
            }

            self.load_new_model();
            self.update_model(delta_s as _);
            self.camera.update(&self.input_state);
            self.draw_frame();
        }
        unsafe { self.context.device().device_wait_idle().unwrap() };
    }

    /// Process the events from the `EventsLoop` and return whether the
    /// main loop should stop.
    fn process_event(&mut self) -> bool {
        let mut should_stop = false;
        let mut resize_dimensions = None;
        let mut path_to_load = None;
        let mut input_state = self.input_state;
        input_state.reset();

        self.events_loop.poll_events(|event| {
            input_state = input_state.update(&event);
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => should_stop = true,
                    WindowEvent::Resized(LogicalSize { width, height }) => {
                        resize_dimensions = Some([width as u32, height as u32]);
                    }
                    WindowEvent::DroppedFile(path) => {
                        log::debug!("File dropped: {:?}", path);
                        path_to_load = Some(path);
                    }
                    _ => {}
                }
            }
        });

        self.resize_dimensions = resize_dimensions;
        if path_to_load.is_some() {
            self.loader.load(path_to_load.as_ref().cloned().unwrap());
            self.path_to_load = path_to_load;
        }
        self.input_state = input_state;
        should_stop
    }

    fn load_new_model(&mut self) {
        if let Some(model) = self.loader.get_model() {
            self.model_renderer.take();

            self.context.graphics_queue_wait_idle();
            unsafe {
                self.context.device().free_command_buffers(
                    self.context.general_command_pool(),
                    &self.command_buffers,
                );
            }

            let model_renderer = ModelRenderer::create(
                Arc::clone(&self.context),
                model,
                &self.camera_uniform_buffers,
                self.swapchain_properties,
                &self.environment,
                self.msaa_samples,
                &self.render_pass,
            );

            let command_buffers = Self::create_and_register_command_buffers(
                &self.context,
                &self.swapchain,
                self.render_pass.get_render_pass(),
                &self.skybox_renderer,
                Some(&model_renderer),
            );

            self.model_renderer = Some(model_renderer);
            self.command_buffers = command_buffers;
        }
    }

    fn update_model(&mut self, delta_s: f32) {
        if let Some(model_renderer) = self.model_renderer.as_mut() {
            model_renderer.update_model(delta_s);
        }
    }

    fn draw_frame(&mut self) {
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
                self.recreate_swapchain();
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.context.device().reset_fences(&wait_fences).unwrap() };

        self.update_uniform_buffers(image_index);

        if let Some(model_renderer) = self.model_renderer.as_mut() {
            model_renderer.update_buffers(image_index as _);
        }

        let device = self.context.device();
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
                device
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
            let result = self.swapchain.present(&present_info);
            match result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            if self.resize_dimensions.is_some() {
                self.recreate_swapchain();
            }
        }
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain.");

        if self.has_window_been_minimized() {
            while !self.has_window_been_maximized() {
                self.process_event();
            }
        }

        unsafe { self.context.device().device_wait_idle().unwrap() };

        self.cleanup_swapchain();

        let dimensions = self.resize_dimensions.unwrap_or([
            self.swapchain.properties().extent.width,
            self.swapchain.properties().extent.height,
        ]);

        let swapchain_support_details = SwapchainSupportDetails::new(
            self.context.physical_device(),
            self.context.surface(),
            self.context.surface_khr(),
        );
        let swapchain_properties = swapchain_support_details
            .get_ideal_swapchain_properties(dimensions, self.config.vsync());

        let render_pass = RenderPass::create(
            Arc::clone(&self.context),
            swapchain_properties.extent,
            swapchain_properties.format.format,
            self.depth_format,
            self.msaa_samples,
        );

        self.skybox_renderer.rebuild_pipeline(
            swapchain_properties,
            self.msaa_samples,
            &render_pass,
        );
        if let Some(model_renderer) = self.model_renderer.as_mut() {
            model_renderer.rebuild_pipelines(swapchain_properties, self.msaa_samples, &render_pass);
        }

        let swapchain = Swapchain::create(
            Arc::clone(&self.context),
            swapchain_support_details,
            dimensions,
            self.config.vsync(),
            &render_pass,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &self.context,
            &swapchain,
            render_pass.get_render_pass(),
            &self.skybox_renderer,
            self.model_renderer.as_ref(),
        );

        self.swapchain = swapchain;
        self.swapchain_properties = swapchain_properties;
        self.render_pass = render_pass;
        self.command_buffers = command_buffers;
    }

    fn has_window_been_minimized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x == 0 || y == 0 => true,
            _ => false,
        }
    }

    fn has_window_been_maximized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x > 0 && y > 0 => true,
            _ => false,
        }
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup_swapchain(&mut self) {
        let device = self.context.device();
        unsafe {
            device.free_command_buffers(self.context.general_command_pool(), &self.command_buffers);
        }
        self.swapchain.destroy();
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        // camera ubo
        {
            let aspect = self.swapchain.properties().extent.width as f32
                / self.swapchain.properties().extent.height as f32;

            let view = Matrix4::look_at(
                self.camera.position(),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            );
            let proj = math::perspective(Deg(45.0), aspect, 0.01, 10.0);

            let ubos = [CameraUBO::new(view, proj, self.camera.position())];
            let buffer = &mut self.camera_uniform_buffers[current_image as usize];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy(data_ptr, &ubos);
            }
        }
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        self.cleanup_swapchain();
        let device = self.context.device();
        self.in_flight_frames.destroy(device);
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
