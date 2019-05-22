use std::{error::Error, fmt};

#[derive(Debug)]
pub enum AppError {
    ConfigLoadError(String),
    FileNotFound(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::ConfigLoadError(message) => {
                write!(f, "Failed to load app configuration: {}", message)
            }
            AppError::FileNotFound(file) => write!(f, "File not found: {}", file),
        }
    }
}

impl Error for AppError {}
