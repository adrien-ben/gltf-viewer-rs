mod camera;
mod config;
mod controls;
mod error;
mod gui;
mod loader;
mod renderer;
mod viewer;

use crate::{camera::*, config::Config, controls::*, loader::*, renderer::*};
use clap::Parser;
use std::{error::Error, path::PathBuf};
use viewer::Viewer;
use vulkan::*;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowId},
};

const TITLE: &str = "Gltf Viewer";

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}

struct App {
    config: Config,
    enable_debug: bool,
    model_path: Option<PathBuf>,
    window: Option<Window>,
    viewer: Option<Viewer>,
}

impl App {
    fn new() -> Result<Self, Box<dyn Error>> {
        let cli = Cli::parse();

        let config = cli
            .config
            .as_ref()
            .map(config::load_config)
            .transpose()?
            .unwrap_or_default();
        let enable_debug = cli.debug;
        let model_path = cli.file;

        Ok(Self {
            config,
            enable_debug,
            model_path,
            window: None,
            viewer: None,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(TITLE)
                    .with_inner_size(PhysicalSize::new(
                        self.config.resolution().width(),
                        self.config.resolution().height(),
                    ))
                    .with_fullscreen(
                        self.config
                            .fullscreen()
                            .then_some(Fullscreen::Borderless(None)),
                    ),
            )
            .expect("Failed to create window");

        self.viewer = Some(Viewer::new(
            self.config.clone(),
            &window,
            self.enable_debug,
            self.model_path.take(),
        ));
        self.window = Some(window);
    }

    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        if let Some(viewer) = self.viewer.as_mut() {
            viewer.new_frame();
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        self.viewer
            .as_mut()
            .unwrap()
            .end_frame(self.window.as_ref().unwrap());
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }

        self.viewer
            .as_mut()
            .unwrap()
            .handle_window_event(self.window.as_ref().unwrap(), &event);
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        self.viewer.as_mut().unwrap().handle_device_event(&event);
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.viewer.as_mut().unwrap().on_exit();
    }
}

#[derive(Parser)]
#[command(name = "GLTF Viewer")]
#[command(version = "1.0")]
#[command(about = "Viewer for GLTF 2.0 files.", long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    file: Option<PathBuf>,
    #[arg(short, long)]
    debug: bool,
}
