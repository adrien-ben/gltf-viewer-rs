use crate::error::*;
use serde::Deserialize;
use std::fs::File;

#[derive(Deserialize, Clone)]
pub struct Config {
    resolution: Resolution,
    vsync: Option<bool>,
    msaa: Option<u32>,
    #[serde(default)]
    env: Environment,
}

impl Config {
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    pub fn vsync(&self) -> bool {
        self.vsync.unwrap_or(false)
    }

    pub fn msaa(&self) -> u32 {
        self.msaa.unwrap_or(1)
    }

    pub fn env(&self) -> &Environment {
        &self.env
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            resolution: Default::default(),
            vsync: Some(false),
            msaa: Some(64),
            env: Default::default(),
        }
    }
}

#[derive(Deserialize, Copy, Clone)]
pub struct Resolution {
    width: u32,
    height: u32,
}

impl Resolution {
    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Resolution {
            width: 120,
            height: 120,
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct Environment {
    path: String,
    resolution: Option<u32>,
}

impl Environment {
    const SKYBOX_DEFAULT_PATH: &'static str = "assets/env/equi.hdr";
    const SKYBOX_DEFAULT_RESOLUTION: u32 = 1024;

    pub fn path(&self) -> &String {
        &self.path
    }

    pub fn resolution(&self) -> u32 {
        self.resolution.unwrap_or(Self::SKYBOX_DEFAULT_RESOLUTION)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            path: String::from(Self::SKYBOX_DEFAULT_PATH),
            resolution: None,
        }
    }
}

pub fn load_config(path: &str) -> Result<Config, AppError> {
    let config_file = File::open(path)
        .map_err(|e| AppError::ConfigLoadError(format!("Failed to load file: {}", e)))?;
    serde_yaml::from_reader(config_file)
        .map_err(|e| AppError::ConfigLoadError(format!("Failed to deserialize config: {}", e)))
}
