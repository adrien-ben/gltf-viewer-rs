//! Non-blocking model loader
//!
//! The loader starts a worker thread that will wait for load messages.
//! Once a message is received the thread will load the model and send the
//! loaded model through another channel.
//!
//! When dropping the loader, a stop message is sent to the thread so it can
//! stop listening for load events. Then we wait for the thread to terminate.
//!
//! Users have to call `load` to load a new model and `get_model` to retrieve
//! the loaded model.

use crate::model::Model;
use crate::vulkan::Context;

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;

enum Message {
    Load(PathBuf),
    Stop,
}

pub struct Loader {
    message_sender: Sender<Message>,
    model_receiver: Receiver<Model>,
    thread_handle: Option<JoinHandle<()>>,
}

impl Loader {
    pub fn new(context: Arc<Context>) -> Self {
        let (message_sender, message_receiver) = mpsc::channel();
        let (model_sender, model_receiver) = mpsc::channel();

        let thread_handle = Some(thread::spawn(move || {
            log::info!("Starting loader");
            loop {
                let message = message_receiver.recv().expect("Failed to receive a path");
                match message {
                    Message::Load(path) => {
                        log::info!("Start loading {}", path.as_path().display());
                        let model = Model::create_from_file(&context, &path);
                        match model {
                            Ok(model) => {
                                log::info!("Finish loading {}", path.as_path().display());
                                model_sender.send(model).unwrap();
                            }
                            Err(error) => {
                                log::error!(
                                    "Failed to load {}. Cause: {}",
                                    path.as_path().display(),
                                    error
                                );
                            }
                        }
                    }
                    Message::Stop => break,
                }
            }
            log::info!("Stopping loader");
        }));

        Self {
            message_sender,
            model_receiver,
            thread_handle,
        }
    }

    /// Start loading a new model in the background.
    ///
    /// Call `get_model` to retrieve the loaded model.
    pub fn load(&self, path: PathBuf) {
        self.message_sender
            .send(Message::Load(path))
            .expect("Failed to send load message to loader");
    }

    /// Get the last loaded model.
    ///
    /// If no model is ready, then `None` is returned.
    pub fn get_model(&self) -> Option<Model> {
        match self.model_receiver.try_recv() {
            Ok(model) => Some(model),
            _ => None,
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.message_sender
            .send(Message::Stop)
            .expect("Failed to send stop message to loader thread");
        if let Some(handle) = self.thread_handle.take() {
            handle
                .join()
                .expect("Failed to wait for loader thread termination");
        }
        log::info!("Loader dropped");
    }
}
