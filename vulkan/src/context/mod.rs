mod shared;

use self::shared::*;
use ash::{extensions::khr::Surface, version::DeviceV1_0, vk, Device, Instance};
use imgui_rs_vulkan_renderer::RendererVkContext;
use std::sync::Arc;
use winit::window::Window;

pub struct Context {
    shared_context: Arc<SharedContext>,
    general_command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
}

impl Context {
    pub fn new(window: &Window, enable_debug: bool) -> Self {
        let shared_context = Arc::new(SharedContext::new(window, enable_debug));
        let general_command_pool = create_command_pool(
            &shared_context.device(),
            shared_context.queue_families_indices,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );
        let transient_command_pool = create_command_pool(
            &shared_context.device(),
            shared_context.queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        Self {
            shared_context,
            general_command_pool,
            transient_command_pool,
        }
    }

    pub fn new_thread(&self) -> Self {
        let shared_context = Arc::clone(&self.shared_context);
        let general_command_pool = create_command_pool(
            &shared_context.device(),
            shared_context.queue_families_indices,
            vk::CommandPoolCreateFlags::empty(),
        );
        let transient_command_pool = create_command_pool(
            &shared_context.device(),
            shared_context.queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        Self {
            shared_context,
            general_command_pool,
            transient_command_pool,
        }
    }
}

fn create_command_pool(
    device: &Device,
    queue_families_indices: QueueFamiliesIndices,
    create_flags: vk::CommandPoolCreateFlags,
) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_families_indices.graphics_index)
        .flags(create_flags);

    unsafe {
        device
            .create_command_pool(&command_pool_info, None)
            .expect("Failed to create command pool")
    }
}

impl Context {
    pub fn instance(&self) -> &Instance {
        self.shared_context.instance()
    }

    pub fn surface(&self) -> &Surface {
        self.shared_context.surface()
    }

    pub fn surface_khr(&self) -> vk::SurfaceKHR {
        self.shared_context.surface_khr()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.shared_context.physical_device()
    }

    pub fn device(&self) -> &Device {
        self.shared_context.device()
    }

    pub fn queue_families_indices(&self) -> QueueFamiliesIndices {
        self.shared_context.queue_families_indices()
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.shared_context.graphics_queue()
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.shared_context.present_queue()
    }

    pub fn general_command_pool(&self) -> vk::CommandPool {
        self.general_command_pool
    }
}

impl Context {
    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        self.shared_context.get_mem_properties()
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        self.shared_context
            .find_supported_format(candidates, tiling, features)
    }

    /// Return the preferred sample count or the maximim supported below preferred.
    pub fn get_max_usable_sample_count(&self, preferred: u32) -> vk::SampleCountFlags {
        self.shared_context.get_max_usable_sample_count(preferred)
    }

    pub fn get_ubo_alignment<T>(&self) -> u32 {
        self.shared_context.get_ubo_alignment::<T>()
    }

    /// Create a one time use command buffer and pass it to `executor`.
    pub fn execute_one_time_commands<R, F: FnOnce(vk::CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> R {
        self.shared_context
            .execute_one_time_commands(self.transient_command_pool, executor)
    }

    pub fn graphics_queue_wait_idle(&self) {
        self.shared_context.graphics_queue_wait_idle()
    }
}

impl RendererVkContext for Context {
    fn instance(&self) -> &Instance {
        &self.shared_context.instance()
    }

    fn physical_device(&self) -> vk::PhysicalDevice {
        self.shared_context.physical_device()
    }

    fn device(&self) -> &Device {
        &self.shared_context.device()
    }

    fn queue(&self) -> vk::Queue {
        self.shared_context.graphics_queue()
    }

    fn command_pool(&self) -> vk::CommandPool {
        self.transient_command_pool
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let device = self.shared_context.device();
        unsafe {
            device.destroy_command_pool(self.transient_command_pool, None);
            device.destroy_command_pool(self.general_command_pool, None);
        }
    }
}

/// Find a memory type in `mem_properties` that is suitable
/// for `requirements` and supports `required_properties`.
///
/// # Returns
///
/// The index of the memory type from `mem_properties`.
pub fn find_memory_type(
    requirements: vk::MemoryRequirements,
    mem_properties: vk::PhysicalDeviceMemoryProperties,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_properties.memory_type_count {
        if requirements.memory_type_bits & (1 << i) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties)
        {
            return i;
        }
    }
    panic!("Failed to find suitable memory type.")
}
