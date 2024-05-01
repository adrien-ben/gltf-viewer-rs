use crate::{debug::*, swapchain::*, MsaaSamples};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{DynamicRendering, Surface, Swapchain as SwapchainLoader, Synchronization2},
    },
    vk, Device, Entry, Instance,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::{
    ffi::{CStr, CString},
    mem::size_of,
};
use winit::window::Window;

pub const HDR_SURFACE_FORMAT: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
    format: vk::Format::R16G16B16A16_SFLOAT,
    color_space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
};

pub struct SharedContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,
    surface: Surface,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    pub queue_families_indices: QueueFamiliesIndices,
    graphics_compute_queue: vk::Queue,
    present_queue: vk::Queue,
    dynamic_rendering: DynamicRendering,
    synchronization2: Synchronization2,
    has_hdr_support: bool,
}

impl SharedContext {
    pub fn new(window: &Window, enable_debug: bool) -> Self {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = create_instance(&entry, window, enable_debug);

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .expect("Failed to create surface")
        };

        let debug_report_callback = if enable_debug {
            Some(setup_debug_messenger(&entry, &instance))
        } else {
            None
        };

        let (physical_device, queue_families_indices) =
            pick_physical_device(&instance, &surface, surface_khr);

        let (device, graphics_compute_queue, present_queue) =
            create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_families_indices,
            );

        let dynamic_rendering = DynamicRendering::new(&instance, &device);
        let synchronization2 = Synchronization2::new(&instance, &device);

        let has_hdr_support = unsafe {
            surface
                .get_physical_device_surface_formats(physical_device, surface_khr)
                .expect("failed to list physical device surface formats")
                .contains(&HDR_SURFACE_FORMAT)
        };

        Self {
            _entry: entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            queue_families_indices,
            graphics_compute_queue,
            present_queue,
            dynamic_rendering,
            synchronization2,
            has_hdr_support,
        }
    }
}

fn create_instance(entry: &Entry, window: &Window, enable_debug: bool) -> Instance {
    let app_name = CString::new("Vulkan Application").unwrap();
    let engine_name = CString::new("No Engine").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(app_name.as_c_str())
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(engine_name.as_c_str())
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let mut extension_names =
        ash_window::enumerate_required_extensions(window.raw_display_handle())
            .expect("Failed to enumerate required extensions")
            .to_vec();
    extension_names.push(vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    if enable_debug {
        extension_names.push(DebugUtils::name().as_ptr());
    }
    if has_ext_colorspace_support(entry) {
        extension_names.push(vk::ExtSwapchainColorspaceFn::name().as_ptr());
    }

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names);

    unsafe {
        entry
            .create_instance(&instance_create_info, None)
            .expect("Failed to create instance")
    }
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
    let devices = unsafe {
        let mut devices = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices");
        devices.sort_by_key(|d| {
            let props = instance.get_physical_device_properties(*d);
            match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 10,
            }
        });

        devices
    };
    let device = devices
        .into_iter()
        .find(|device| is_device_suitable(instance, surface, surface_khr, *device))
        .expect("No suitable physical device.");

    let props = unsafe { instance.get_physical_device_properties(device) };
    log::debug!("Selected physical device: {:?}", unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
    });

    let (graphics_compute, present) = find_queue_families(instance, surface, surface_khr, device);
    let queue_families_indices = QueueFamiliesIndices {
        graphics_index: graphics_compute.unwrap(),
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
    let (graphics_compute, present) = find_queue_families(instance, surface, surface_khr, device);
    let extention_support = check_device_extension_support(instance, device);
    let is_swapchain_adequate = {
        let details = SwapchainSupportDetails::new(device, surface, surface_khr);
        !details.formats.is_empty() && !details.present_modes.is_empty()
    };
    let features = unsafe { instance.get_physical_device_features(device) };
    graphics_compute.is_some()
        && present.is_some()
        && extention_support
        && is_swapchain_adequate
        && features.sampler_anisotropy == vk::TRUE
}

fn has_ext_colorspace_support(entry: &Entry) -> bool {
    let extension_props = entry
        .enumerate_instance_extension_properties(None)
        .expect("Failed to enumerate instance extention properties");

    extension_props.iter().any(|ext| {
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
        vk::ExtSwapchainColorspaceFn::name() == name
    })
}

fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let required_extentions = get_required_device_extensions();

    let extension_props = unsafe {
        instance
            .enumerate_device_extension_properties(device)
            .expect("Failed to enumerate device extention properties")
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

fn get_required_device_extensions() -> [&'static CStr; 7] {
    [
        SwapchainLoader::name(),
        DynamicRendering::name(),
        vk::KhrDepthStencilResolveFn::name(),
        vk::KhrCreateRenderpass2Fn::name(),
        vk::KhrMultiviewFn::name(),
        vk::KhrMaintenance2Fn::name(),
        vk::KhrSynchronization2Fn::name(),
    ]
}

/// Find a queue family with at least one graphics & compute queue and one with
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
    let mut graphics_compute = None;
    let mut present = None;

    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
        let index = index as u32;

        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && graphics_compute.is_none()
        {
            graphics_compute = Some(index);
        }

        let present_support = unsafe {
            surface
                .get_physical_device_surface_support(device, index, surface_khr)
                .expect("Failed to get surface support")
        };
        if present_support && present.is_none() {
            present = Some(index);
        }

        if graphics_compute.is_some() && present.is_some() {
            break;
        }
    }

    (graphics_compute, present)
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

    let device_extensions = get_required_device_extensions();
    let device_extensions_ptrs = device_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);
    let mut dynamic_rendering_feature =
        vk::PhysicalDeviceDynamicRenderingFeatures::builder().dynamic_rendering(true);
    let mut synchronization2_feature =
        vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);
    let mut device_features_2 = vk::PhysicalDeviceFeatures2::builder()
        .features(device_features.build())
        .push_next(&mut dynamic_rendering_feature)
        .push_next(&mut synchronization2_feature);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions_ptrs)
        .push_next(&mut device_features_2);

    // Build device and queues
    let device = unsafe {
        instance
            .create_device(device, &device_create_info, None)
            .expect("Failed to create logical device.")
    };
    let graphics_compute_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

    (device, graphics_compute_queue, present_queue)
}

impl SharedContext {
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

    pub fn graphics_compute_queue(&self) -> vk::Queue {
        self.graphics_compute_queue
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }

    pub fn dynamic_rendering(&self) -> &DynamicRendering {
        &self.dynamic_rendering
    }

    pub fn synchronization2(&self) -> &Synchronization2 {
        &self.synchronization2
    }

    pub fn has_hdr_support(&self) -> bool {
        self.has_hdr_support
    }
}

impl SharedContext {
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

    /// Return the preferred sample count or the maximum supported below preferred.
    pub fn get_max_usable_sample_count(&self, preferred: MsaaSamples) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let max_sample_count = color_sample_counts.min(depth_sample_counts);

        use MsaaSamples::*;
        if max_sample_count.contains(vk::SampleCountFlags::TYPE_64) && preferred == S64 {
            vk::SampleCountFlags::TYPE_64
        } else if max_sample_count.contains(vk::SampleCountFlags::TYPE_32) && preferred >= S32 {
            vk::SampleCountFlags::TYPE_32
        } else if max_sample_count.contains(vk::SampleCountFlags::TYPE_16) && preferred >= S16 {
            vk::SampleCountFlags::TYPE_16
        } else if max_sample_count.contains(vk::SampleCountFlags::TYPE_8) && preferred >= S8 {
            vk::SampleCountFlags::TYPE_8
        } else if max_sample_count.contains(vk::SampleCountFlags::TYPE_4) && preferred >= S4 {
            vk::SampleCountFlags::TYPE_4
        } else if max_sample_count.contains(vk::SampleCountFlags::TYPE_2) && preferred >= S2 {
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
    pub fn execute_one_time_commands<R, F: FnOnce(vk::CommandBuffer) -> R>(
        &self,
        pool: vk::CommandPool,
        executor: F,
    ) -> R {
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(pool)
                .command_buffer_count(1);

            unsafe {
                self.device
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to allocate command buffer")[0]
            }
        };
        let command_buffers = [command_buffer];

        // Begin recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .expect("Failed to begin command buffer")
            };
        }

        // Execute user function
        let executor_result = executor(command_buffer);

        // End recording
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        };

        // Submit and wait
        {
            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::builder().command_buffer(command_buffer);
            let submit_info = vk::SubmitInfo2::builder()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

            unsafe {
                let queue = self.graphics_compute_queue();
                self.synchronization2
                    .queue_submit2(queue, std::slice::from_ref(&submit_info), vk::Fence::null())
                    .expect("Failed to submit to queue");
                self.device
                    .queue_wait_idle(queue)
                    .expect("Failed to wait for queue to be idle");
            };
        }

        // Free
        unsafe { self.device.free_command_buffers(pool, &command_buffers) };

        executor_result
    }

    pub fn graphics_queue_wait_idle(&self) {
        unsafe {
            self.device
                .queue_wait_idle(self.graphics_compute_queue())
                .expect("Failed to wait for queue to be idle")
        }
    }
}

impl Drop for SharedContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}
