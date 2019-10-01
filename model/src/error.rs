use std::{error::Error, fmt};

#[derive(Debug)]
pub struct ModelLoadingError {
    message: String,
}

impl ModelLoadingError {
    pub fn new<S: Into<String>>(message: S) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelLoadingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Failed to load model. Cause: {}", self.message)
    }
}

impl Error for ModelLoadingError {}
