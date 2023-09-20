use crate::error::*;
use serde::de::Unexpected;
use serde::Deserialize;
use std::fs::File;
use vulkan::MsaaSamples;

#[derive(Deserialize, Clone)]
pub struct Config {
    resolution: Resolution,
    #[serde(default)]
    fullscreen: bool,
    vsync: Option<bool>,
    #[serde(deserialize_with = "deserialize_msaa")]
    msaa: MsaaSamples,
    #[serde(default)]
    env: Environment,
}

fn deserialize_msaa<'de, D>(deserializer: D) -> Result<MsaaSamples, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let samples: u64 = serde::de::Deserialize::deserialize(deserializer)?;

    match samples {
        1 => Ok(MsaaSamples::S1),
        2 => Ok(MsaaSamples::S2),
        4 => Ok(MsaaSamples::S4),
        8 => Ok(MsaaSamples::S8),
        16 => Ok(MsaaSamples::S16),
        32 => Ok(MsaaSamples::S32),
        64 => Ok(MsaaSamples::S64),
        _ => Err(serde::de::Error::invalid_value(
            Unexpected::Unsigned(samples),
            &"msaa should be one of 1, 2, 4, 8, 16, 32 or 64",
        )),
    }
}

impl Config {
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    pub fn fullscreen(&self) -> bool {
        self.fullscreen
    }

    pub fn vsync(&self) -> bool {
        self.vsync.unwrap_or(false)
    }

    pub fn msaa(&self) -> MsaaSamples {
        self.msaa
    }

    pub fn env(&self) -> &Environment {
        &self.env
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            resolution: Default::default(),
            fullscreen: false,
            vsync: Some(false),
            msaa: MsaaSamples::S1,
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
