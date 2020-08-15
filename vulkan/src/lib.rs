mod buffer;
mod context;
mod debug;
mod descriptor;
mod image;
mod pipeline;
mod renderpass;
mod shader;
mod swapchain;
mod texture;
mod util;
mod vertex;

pub use self::{
    buffer::*, context::*, debug::*, descriptor::*, image::*, pipeline::*, renderpass::*,
    shader::*, swapchain::*, texture::*, util::*, vertex::*,
};

pub use ash;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;
pub use winit;

/// Hold a partially loaded resource.
///
/// The main usecase is to create resource that don't
/// required submitting work through a queue and to bake
/// commands for actions that do required the use of a queue.
/// It means you can pre-load data on a seperate thread and
/// finish loading on the main thread.
///
/// The struct also holds the temporary data that is required
/// when submitting work to the queue (a staging buffer for example).
///
/// To finish loading you need to call the [finish] method. This will
/// submit the command buffer through the main queue then free the command
/// buffer. The temporary data is also dropped at this point.
pub struct PreLoadedResource<R, T> {
    context: Arc<Context>,
    command_buffer: vk::CommandBuffer,
    resource: Option<R>,
    tmp_data: Option<T>,
}

impl<R, T> PreLoadedResource<R, T> {
    pub fn new(
        context: Arc<Context>,
        command_buffer: vk::CommandBuffer,
        resource: R,
        tmp_data: T,
    ) -> Self {
        Self {
            context,
            command_buffer,
            resource: Some(resource),
            tmp_data: Some(tmp_data),
        }
    }
}

impl<R, T> PreLoadedResource<R, T> {
    /// Finish loading the resource.
    ///
    /// Submit the command buffer to the main queue and free it afterwards.
    /// Temporary data is dropped.
    ///
    /// # Returns
    ///
    /// The loaded resource.
    pub fn finish(&mut self) -> R {
        assert!(
            self.resource.is_some(),
            "Resource loading was already finished"
        );

        self.execute_commands();
        self.free_command_buffer();
        self.tmp_data.take();

        self.resource.take().unwrap()
    }

    fn execute_commands(&self) {
        self.context
            .execute_one_time_commands(|primary_command_buffer| unsafe {
                let secondary_command_buffer = [self.command_buffer];
                self.context
                    .device()
                    .cmd_execute_commands(primary_command_buffer, &secondary_command_buffer);
            });
    }

    fn free_command_buffer(&self) {
        let secondary_command_buffer = [self.command_buffer];
        unsafe {
            self.context.device().free_command_buffers(
                self.context.general_command_pool(),
                &secondary_command_buffer,
            )
        }
    }
}
