mod base;
mod camera;
mod config;
mod controls;
mod environment;
mod error;
mod math;
mod model;
mod pipelines;
mod util;
mod vulkan;

use self::{base::BaseApp, error::AppError};
use clap::{App, Arg};
use std::{error::Error, path::Path, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let matches = create_app().get_matches();

    let config = matches
        .value_of("config")
        .map_or(Ok(Default::default()), config::load_config)?;

    let file_path = Path::new(matches.value_of("file").unwrap());
    if !file_path.exists() {
        Err(AppError::FileNotFound(format!("{}", file_path.display())))?
    }

    BaseApp::new(config, file_path).run();

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
                .value_name("FILE")
                .required(true)
                .help("Set the path to gltf model to view")
                .takes_value(true)
                .last(true),
        )
}
