use environment::Environment;
use model::{Model, PlaybackMode};
use std::{cell::RefCell, path::PathBuf, rc::Rc, sync::Arc, time::Instant};
use vulkan::{
    Context,
    winit::{
        dpi::PhysicalSize,
        event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
        keyboard::Key,
        window::Window,
    },
};

use crate::{
    Camera, InputState, Loader, RenderError, Renderer, RendererSettings,
    config::Config,
    gui::{self, Gui},
};

pub struct Viewer {
    config: Config,
    context: Arc<Context>,
    renderer_settings: RendererSettings,
    gui: Gui,
    enable_ui: bool,
    renderer: Renderer,
    model: Option<Rc<RefCell<Model>>>,
    loader: Loader,
    camera: Camera,
    input_state: InputState,
    time: Instant,
    dirty_swapchain: bool,
}

impl Viewer {
    pub fn new(
        config: Config,
        window: &Window,
        enable_debug: bool,
        model_path: Option<PathBuf>,
    ) -> Self {
        let context = Arc::new(Context::new(window, enable_debug));

        let renderer_settings = RendererSettings::new(&context);

        let environment =
            Environment::new(&context, config.env().path(), config.env().resolution());
        let gui = Gui::new(window, renderer_settings);
        let enable_ui = true;
        let renderer = Renderer::create(
            Arc::clone(&context),
            &config,
            renderer_settings,
            environment,
        );

        let model: Option<Rc<RefCell<Model>>> = None;
        let loader = Loader::new(Arc::new(context.new_thread()));
        if let Some(p) = model_path {
            loader.load(p);
        }

        let camera = Camera::default();
        let input_state = InputState::default();
        let time = Instant::now();
        let dirty_swapchain = false;

        Self {
            config,
            context,
            renderer_settings,
            gui,
            enable_ui,
            renderer,
            model,
            loader,
            camera,
            input_state,
            time,
            dirty_swapchain,
        }
    }

    pub fn new_frame(&mut self) {
        self.input_state = self.input_state.reset();
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        self.input_state = self.input_state.handle_window_event(event);
        self.gui.handle_event(window, event);
        match event {
            // Dropped file
            WindowEvent::DroppedFile(path) => {
                log::debug!("File dropped: {:?}", path);
                self.loader.load(path.clone());
            }
            // Resizing
            WindowEvent::Resized(_) => {
                self.dirty_swapchain = true;
            }
            // Key events
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(c),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if c == "h" {
                    self.enable_ui = !self.enable_ui;
                }
            }
            _ => (),
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        self.input_state = self.input_state.handle_device_event(event);
    }

    pub fn end_frame(&mut self, window: &Window) {
        let new_time = Instant::now();
        let delta_s = (new_time - self.time).as_secs_f32();
        self.time = new_time;

        // Load new model
        if let Some(loaded_model) = self.loader.get_model() {
            self.gui.set_model_metadata(loaded_model.metadata().clone());
            self.model.take();

            self.context.graphics_queue_wait_idle();
            let loaded_model = Rc::new(RefCell::new(loaded_model));
            self.renderer.set_model(&loaded_model);
            self.model = Some(loaded_model);
        }

        // Update model
        if let Some(model) = self.model.as_ref() {
            let mut model = model.borrow_mut();

            if self.gui.should_toggle_animation() {
                model.toggle_animation();
            } else if self.gui.should_stop_animation() {
                model.stop_animation();
            } else if self.gui.should_reset_animation() {
                model.reset_animation();
            } else {
                let playback_mode = if self.gui.is_infinite_animation_checked() {
                    PlaybackMode::Loop
                } else {
                    PlaybackMode::Once
                };

                model.set_animation_playback_mode(playback_mode);
                model.set_current_animation(self.gui.get_selected_animation());
            }
            self.gui
                .set_animation_playback_state(model.get_animation_playback_state());

            let delta_s = delta_s * self.gui.get_animation_speed();
            model.update(delta_s);
        }

        // Update camera
        {
            if self.gui.should_reset_camera() {
                self.camera = Default::default();
            }

            self.camera = match self.gui.camera_mode() {
                gui::CameraMode::Orbital => self.camera.to_orbital(),
                gui::CameraMode::Fps => self.camera.to_fps(),
            };

            self.camera.fov = self.gui.camera_fov();
            self.camera.z_near = self.gui.camera_z_near();
            self.camera.z_far = self.gui.camera_z_far();
            self.camera.set_move_speed(self.gui.camera_move_speed());

            if !self.gui.is_hovered() {
                self.camera.update(&self.input_state, delta_s);
                self.gui.set_camera(Some(self.camera));
            }
        }

        // Check if settings changed
        if let Some(new_renderer_settings) = self.gui.get_new_renderer_settings() {
            // recreate swapchain if hdr was toggled
            if self.renderer_settings.hdr_enabled != new_renderer_settings.hdr_enabled {
                self.renderer.recreate_swapchain(
                    window.inner_size().into(),
                    self.config.vsync(),
                    new_renderer_settings.hdr_enabled.unwrap_or_default(),
                );
                self.dirty_swapchain = false;
            }

            // Update renderer
            self.renderer.update_settings(new_renderer_settings);

            self.renderer_settings = new_renderer_settings;
        }

        // If swapchain must be recreated wait for windows to not be minimized anymore
        if self.dirty_swapchain {
            let PhysicalSize { width, height } = window.inner_size();
            if width > 0 && height > 0 {
                self.renderer.recreate_swapchain(
                    window.inner_size().into(),
                    self.config.vsync(),
                    self.renderer_settings.hdr_enabled.unwrap_or_default(),
                );
            } else {
                return;
            }
        }

        let gui = self.enable_ui.then_some(&mut self.gui);
        self.dirty_swapchain = matches!(
            self.renderer.render(window, self.camera, gui),
            Err(RenderError::DirtySwapchain)
        );
    }

    pub fn on_exit(&mut self) {
        self.renderer.wait_idle_gpu();
    }
}
