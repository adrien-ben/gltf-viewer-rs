mod base;
mod model;
mod vulkan;

use base::BaseApp;
use std::{env, error::Error, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let path = get_file_path_from_args()?;
    log::info!("Loading file {}", path);

    let (_document, _buffers, _images) = gltf::import(path)?;

    Ok(BaseApp::new().run())
}

fn get_file_path_from_args() -> Result<String, &'static str> {
    let args = env::args().nth(1);
    match args {
        Some(arg) => Ok(arg),
        None => Err("First program argument should be the file to load"),
    }
}
