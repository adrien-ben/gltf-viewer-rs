use super::{debug::*, surface, swapchain::*};
use ash::{
    extensions::{
        ext::DebugReport,
        khr::{Surface, Swapchain as SwapchainLoader},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use std::{
    ffi::{CStr, CString},
    mem::size_of,
};
use winit::Window;

const POSSIBLE_SAMPLE_COUNTS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];

pub struct Context {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugReport, vk::DebugReportCallbackEXT)>,
    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_families_indices: QueueFamiliesIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    general_command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
}

impl Context {
    pub fn new(window: &Window) -> Self {
        let entry = Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry);

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { surface::create_surface(&entry, &instance, &window).unwrap() };

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_families_indices,
            );

        let general_command_pool = Self::create_command_pool(
            &device,
            queue_families_indices,
            vk::CommandPoolCreateFlags::empty(),
        );
        let transient_command_pool = Self::create_command_pool(
            &device,
            queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        Context {
            _entry: entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            queue_families_indices,
            graphics_queue,
            present_queue,
            general_command_pool,
            transient_command_pool,
        }
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(ash::vk_make_version!(0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(ash::vk_make_version!(0, 1, 0))
            .api_version(ash::vk_make_version!(1, 0, 0));

        let mut extension_names = surface::required_extension_names();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugReport::name().as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    /// Pick the first suitable physical device.
    ///
    /// # Requirements
    /// - At least one queue family with one queue supportting graphics.
    /// - At least one queue family with one queue supporting presentation to `surface_khr`.
    /// - Swapchain extension support.
    ///
    /// # Returns
    ///
    /// A tuple containing the physical device and the queue families indices.
    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_families_indices)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extention_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extention_support
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE
            && features.shader_sampled_image_array_dynamic_indexing == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extentions = Self::get_required_device_extensions();

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        for required in required_extentions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

            if !found {
                return false;
            }
        }

        true
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [SwapchainLoader::name()]
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support =
                unsafe { surface.get_physical_device_surface_support(device, index, surface_khr) };
            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    /// Create the logical device to interact with `device`, a graphics queue
    /// and a presentation queue.
    ///
    /// # Returns
    ///
    /// Return a tuple containing the logical device, the graphics queue and the presentation queue.
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .shader_sampled_image_array_dynamic_indexing(true);

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);
        if ENABLE_VALIDATION_LAYERS {
            device_create_info = device_create_info.enabled_layer_names(&layer_names_ptrs)
        }

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (device, graphics_queue, present_queue)
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
                .unwrap()
        }
    }
}

impl Context {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_khr(&self) -> vk::SurfaceKHR {
        self.surface_khr
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue_families_indices(&self) -> QueueFamiliesIndices {
        self.queue_families_indices
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    pub fn general_command_pool(&self) -> vk::CommandPool {
        self.general_command_pool
    }
}

impl Context {
    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    /// Return the preferred sample count or the maximim supported below preferred.
    pub fn get_max_usable_sample_count(&self, preferred: u32) -> vk::SampleCountFlags {
        if !POSSIBLE_SAMPLE_COUNTS.contains(&preferred) {
            panic!(
                "Preferred sample count must be one of {:?}",
                POSSIBLE_SAMPLE_COUNTS
            );
        }

        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) && preferred == 64 {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) && preferred >= 32 {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) && preferred >= 16 {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) && preferred >= 8 {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) && preferred >= 4 {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) && preferred >= 2 {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn get_min_uniform_buffer_offset_alignment(&self) -> u32 {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        props.limits.min_uniform_buffer_offset_alignment as _
    }

    pub fn get_ubo_alignment<T>(&self) -> u32 {
        let min_alignment = self.get_min_uniform_buffer_offset_alignment();
        let t_size = size_of::<T>() as u32;

        if t_size <= min_alignment {
            min_alignment
        } else {
            min_alignment * (t_size as f32 / min_alignment as f32).ceil() as u32
        }
    }

    /// Create a one time use command buffer and pass it to `executor`.
    pub fn execute_one_time_commands<F: FnOnce(vk::CommandBuffer)>(&self, executor: F) {
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.transient_command_pool)
                .command_buffer_count(1);

            unsafe { self.device.allocate_command_buffers(&alloc_info).unwrap()[0] }
        };
        let command_buffers = [command_buffer];

        // Begin recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap()
            };
        }

        // Execute user function
        executor(command_buffer);

        // End recording
        unsafe { self.device.end_command_buffer(command_buffer).unwrap() };

        // Submit and wait
        {
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())
                    .unwrap();
                self.device.queue_wait_idle(self.graphics_queue).unwrap();
            };
        }

        // Free
        unsafe {
            self.device
                .free_command_buffers(self.transient_command_pool, &command_buffers)
        };
    }

    pub fn graphics_queue_wait_idle(&self) {
        unsafe { self.device.queue_wait_idle(self.graphics_queue).unwrap() }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_command_pool(self.transient_command_pool, None);
            self.device
                .destroy_command_pool(self.general_command_pool, None);
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_report_callback(callback, None);
            }
            self.instance.destroy_instance(None);
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

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}
