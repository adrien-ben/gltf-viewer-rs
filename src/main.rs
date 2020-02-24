mod camera;
mod config;
mod controls;
mod error;
mod gui;
mod loader;
mod renderer;
mod viewer;

use self::viewer::Viewer;
use clap::{App, Arg};
use std::{error::Error, path::Path, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let matches = create_app().get_matches();

    let config = matches
        .value_of("config")
        .map_or(Ok(Default::default()), config::load_config)?;

    let enable_debug = matches.is_present("debug");

    let file_path = matches.value_of("file").map(Path::new);

    Viewer::new(config, enable_debug, file_path).run();

    Ok(())
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
