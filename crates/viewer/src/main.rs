mod camera;
mod config;
mod controls;
mod error;
mod gui;
mod loader;
mod renderer;

use crate::{camera::*, config::Config, controls::*, gui::Gui, loader::*, renderer::*};
use clap::{Arg, Command};
use environment::*;
use model::{Model, PlaybackMode};
use std::{cell::RefCell, error::Error, path::PathBuf, rc::Rc, sync::Arc, time::Instant};
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, WindowBuilder},
};

const TITLE: &str = "Gltf Viewer";

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let matches = create_app().get_matches();

    let config = matches
        .value_of("config")
        .map_or(Ok(Default::default()), config::load_config)?;
    let enable_debug = matches.is_present("debug");
    let file_path = matches.value_of("file").map(PathBuf::from);

    run(config, enable_debug, file_path);

    Ok(())
}

fn run(config: Config, enable_debug: bool, path: Option<PathBuf>) {
    log::debug!("Initializing application.");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_inner_size(PhysicalSize::new(
            config.resolution().width(),
            config.resolution().height(),
        ))
        .with_fullscreen(config.fullscreen().then_some(Fullscreen::Borderless(None)))
        .build(&event_loop)
        .unwrap();

    let context = Arc::new(Context::new(&window, enable_debug));

    let mut renderer_settings = RendererSettings::new(&context);

    let environment = Environment::new(&context, config.env().path(), config.env().resolution());
    let mut gui = Gui::new(&window, renderer_settings);
    let mut renderer = Renderer::create(
        Arc::clone(&context),
        &config,
        renderer_settings,
        environment,
    );

    let mut model: Option<Rc<RefCell<Model>>> = None;
    let loader = Loader::new(Arc::new(context.new_thread()));
    if let Some(p) = path {
        loader.load(p);
    }

    let mut camera = Camera::default();
    let mut input_state = InputState::default();
    let mut time = Instant::now();
    let mut dirty_swapchain = false;

    // Main loop
    log::debug!("Running application.");
    event_loop
        .run(move |event, elwt| {
            input_state = input_state.update(&event);

            match event {
                // Start of event processing
                Event::NewEvents(_) => {}
                // End of event processing
                Event::AboutToWait => {
                    let new_time = Instant::now();
                    let delta_s = (new_time - time).as_secs_f32();
                    time = new_time;

                    // Load new model
                    if let Some(loaded_model) = loader.get_model() {
                        gui.set_model_metadata(loaded_model.metadata().clone());
                        model.take();

                        context.graphics_queue_wait_idle();
                        let loaded_model = Rc::new(RefCell::new(loaded_model));
                        renderer.set_model(&loaded_model);
                        model = Some(loaded_model);
                    }

                    // Update model
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
                                PlaybackMode::Loop
                            } else {
                                PlaybackMode::Once
                            };

                            model.set_animation_playback_mode(playback_mode);
                            model.set_current_animation(gui.get_selected_animation());
                        }
                        gui.set_animation_playback_state(model.get_animation_playback_state());

                        let delta_s = delta_s * gui.get_animation_speed();
                        model.update(delta_s);
                    }

                    // Update camera
                    {
                        if gui.should_reset_camera() {
                            camera = Default::default();
                        }

                        camera = match gui.camera_mode() {
                            gui::CameraMode::Orbital => camera.to_orbital(),
                            gui::CameraMode::Fps => camera.to_fps(),
                        };

                        camera.fov = gui.camera_fov();
                        camera.z_near = gui.camera_z_near();
                        camera.z_far = gui.camera_z_far();
                        camera.set_move_speed(gui.camera_move_speed());

                        if !gui.is_hovered() {
                            camera.update(&input_state, delta_s);
                            gui.set_camera(Some(camera));
                        }
                    }

                    // Check if settings changed
                    if let Some(new_renderer_settings) = gui.get_new_renderer_settings() {
                        // recreate swapchain if hdr was toggled
                        if renderer_settings.hdr_enabled != new_renderer_settings.hdr_enabled {
                            renderer.recreate_swapchain(
                                window.inner_size().into(),
                                config.vsync(),
                                new_renderer_settings.hdr_enabled.unwrap_or_default(),
                            );
                            dirty_swapchain = false;
                        }

                        // Update renderer
                        renderer.update_settings(new_renderer_settings);

                        renderer_settings = new_renderer_settings;
                    }

                    // If swapchain must be recreated wait for windows to not be minimized anymore
                    if dirty_swapchain {
                        let PhysicalSize { width, height } = window.inner_size();
                        if width > 0 && height > 0 {
                            renderer.recreate_swapchain(
                                window.inner_size().into(),
                                config.vsync(),
                                renderer_settings.hdr_enabled.unwrap_or_default(),
                            );
                        } else {
                            return;
                        }
                    }

                    dirty_swapchain = matches!(
                        renderer.render(&window, camera, &mut gui),
                        Err(RenderError::DirtySwapchain)
                    );
                }
                // Window event
                Event::WindowEvent { event, .. } => {
                    gui.handle_event(&window, &event);
                    match event {
                        // Dropped file
                        WindowEvent::DroppedFile(path) => {
                            log::debug!("File dropped: {:?}", path);
                            loader.load(path);
                        }
                        // Resizing
                        WindowEvent::Resized(_) => {
                            dirty_swapchain = true;
                        }
                        // Exit
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        _ => (),
                    }
                }
                // Cleanup
                Event::LoopExiting => {
                    log::info!("Stopping application");
                    renderer.wait_idle_gpu();
                }
                _ => (),
            }
        })
        .unwrap();
}

fn create_app<'a>() -> Command<'a> {
    Command::new("GLTF Viewer")
        .version("1.0")
        .author("Adrien Bennadji")
        .about("Viewer for GLTF 2.0 files.")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Set the path to the configuration file")
                .takes_value(true),
        )
        .arg(
            Arg::new("file")
                .short('f')
                .long("file")
                .value_name("FILE")
                .help("Set the path to gltf model to view")
                .takes_value(true),
        )
        .arg(
            Arg::new("debug")
                .short('d')
                .long("debug")
                .value_name("DEBUG")
                .help("Enable vulkan debug printing")
                .takes_value(false),
        )
}
