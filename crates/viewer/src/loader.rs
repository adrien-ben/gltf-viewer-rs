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

use model::{Model, ModelStagingResources};

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::thread::JoinHandle;
use vulkan::ash::vk;
use vulkan::{Context, PreLoadedResource};

enum Message {
    Load(PathBuf),
    Stop,
}

pub struct Loader {
    message_sender: Sender<Message>,
    model_receiver: Receiver<PreLoadedResource<Model, ModelStagingResources>>,
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
                        let pre_loaded_model = pre_load_model(&context, path.as_path());

                        match pre_loaded_model {
                            Ok(pre_loaded_model) => {
                                log::info!("Finish loading {}", path.as_path().display());
                                model_sender.send(pre_loaded_model).unwrap();
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
            Ok(mut pre_loaded_model) => Some(pre_loaded_model.finish()),
            _ => None,
        }
    }
}

fn pre_load_model<P: AsRef<Path>>(
    context: &Arc<Context>,
    path: P,
) -> Result<PreLoadedResource<Model, ModelStagingResources>, Box<dyn Error>> {
    let device = context.device();

    // Create command buffer
    let command_buffer = {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(context.general_command_pool())
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);

        unsafe { device.allocate_command_buffers(&allocate_info).unwrap()[0] }
    };

    // Begin recording command buffer
    {
        let inheritance_info = vk::CommandBufferInheritanceInfo::default();
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .inheritance_info(&inheritance_info)
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .unwrap()
        };
    }

    // Load model data and prepare command buffer
    let model = Model::create_from_file(Arc::clone(context), command_buffer, path);

    // End recording command buffer
    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    model
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
