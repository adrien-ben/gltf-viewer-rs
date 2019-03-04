mod base;
mod camera;
mod math;
mod model;
mod util;
mod vulkan;

use base::BaseApp;
use std::{env, error::Error, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    log::info!("Welcome to gltf-viewer-rs");

    let file_path = env::args()
        .nth(1)
        .expect("First argument should be a path to a gltf file");
    BaseApp::new(file_path).run();

    Ok(())
}
