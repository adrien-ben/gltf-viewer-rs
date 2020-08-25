mod camera;
mod config;
mod controls;
mod error;
mod gui;
mod loader;
mod renderer;

use crate::{camera::*, config::Config, controls::*, gui::Gui, loader::*, renderer::*};
use clap::{App, Arg};
use environment::*;
use model::{Model, PlaybackMode};
use std::{cell::RefCell, error::Error, path::PathBuf, rc::Rc, sync::Arc, time::Instant};
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
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

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_inner_size(PhysicalSize::new(
            config.resolution().width(),
            config.resolution().height(),
        ))
        .build(&event_loop)
        .unwrap();

    let context = Arc::new(Context::new(&window, enable_debug));

    let environment = Environment::new(&context, config.env().path(), config.env().resolution());
    let mut gui = Gui::new(&window);
    let mut renderer = Renderer::create(
        Arc::clone(&context),
        &config,
        environment,
        gui.get_context(),
    );

    let mut model: Option<Rc<RefCell<Model>>> = None;
    let loader = Loader::new(Arc::new(context.new_thread()));
    if let Some(p) = path {
        loader.load(p);
    }

    let mut camera = Camera::default();
    let mut input_state = InputState::default();
    let mut run = true;
    let mut time = Instant::now();
    let mut dirty_swapchain = false;

    // Main loop
    log::debug!("Running application.");
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        gui.handle_event(&window, &event);
        input_state = input_state.update(&event);

        match event {
            // Start of event processing
            Event::NewEvents(_) => {
                input_state.reset();
                gui.update_delta_time();
            }
            // End of event processing
            Event::MainEventsCleared => {
                let new_time = Instant::now();
                let delta_s = (new_time - time).as_secs_f64();
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
                        renderer.set_tone_map_mode(tone_map_mode);
                    }
                    if let Some(output_mode) = gui.get_new_renderer_output_mode() {
                        context.graphics_queue_wait_idle();
                        renderer.set_output_mode(output_mode);
                    }
                }

                // If swapchain must be recreated wait for windows to not be minimized anymore
                if dirty_swapchain {
                    let PhysicalSize { width, height } = window.inner_size();
                    if width > 0 && height > 0 {
                        renderer.recreate_swapchain(window.inner_size().into(), config.vsync());
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
                renderer.wait_idle_gpu();
            }
            _ => (),
        }

        if !run {
            *control_flow = ControlFlow::Exit;
        }
    });
}

fn create_app<'a, 'b>() -> App<'a, 'b> {
    App::new("GLTF Viewer")
        .version("1.0")
        .author("Adrien Bennadji")
        .about("Viewer for GLTF 2.0 files.")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILE")
                .help("Set the path to the configuration file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("file")
                .short("f")
                .long("file")
                .value_name("FILE")
                .help("Set the path to gltf model to view")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("debug")
                .short("d")
                .long("debug")
                .value_name("DEBUG")
                .help("Enable vulkan debug printing")
                .takes_value(false),
        )
}
