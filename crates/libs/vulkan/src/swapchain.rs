use super::{
    context::Context,
    image::{create_image_view, Image},
};
use ash::{
    khr::{surface, swapchain},
    prelude::VkResult,
    vk, Device,
};
use std::sync::Arc;

pub struct Swapchain {
    context: Arc<Context>,
    swapchain: swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    properties: SwapchainProperties,
    images: Vec<Image>,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    /// Create the swapchain with optimal settings possible with
    /// `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    pub fn create(
        context: Arc<Context>,
        swapchain_support_details: SwapchainSupportDetails,
        dimensions: [u32; 2],
        preferred_format: Option<vk::SurfaceFormatKHR>,
        preferred_vsync: bool,
    ) -> Self {
        log::debug!("Creating swapchain.");

        let properties = swapchain_support_details.get_ideal_swapchain_properties(
            preferred_format,
            dimensions,
            preferred_vsync,
        );

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let min_image_count = properties.min_image_count;

        let queue_families_indices = context.queue_families_indices();
        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(context.surface_khr())
                .min_image_count(min_image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(swapchain_support_details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let swapchain = swapchain::Device::new(context.instance(), context.device());
        let swapchain_khr = unsafe {
            swapchain
                .create_swapchain(&create_info, None)
                .expect("Failed to create swapchain")
        };
        let images = unsafe {
            swapchain
                .get_swapchain_images(swapchain_khr)
                .expect("Failed to get swapchain images")
                .iter()
                .map(|image| {
                    Image::create_swapchain_image(Arc::clone(&context), *image, properties)
                })
                .collect::<Vec<_>>()
        };
        let views = Self::create_views(context.device(), &images, properties);

        let swapchain = Self::new(context, swapchain, swapchain_khr, properties, images, views);

        log::debug!(
            "Created swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            swapchain.image_count(),
        );

        swapchain
    }

    /// Create one image view for each image of the swapchain.
    fn create_views(
        device: &Device,
        swapchain_images: &[Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                create_image_view(
                    device,
                    image.image,
                    vk::ImageViewType::TYPE_2D,
                    1,
                    1,
                    0,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn new(
        context: Arc<Context>,
        swapchain: swapchain::Device,
        swapchain_khr: vk::SwapchainKHR,
        properties: SwapchainProperties,
        images: Vec<Image>,
        image_views: Vec<vk::ImageView>,
    ) -> Self {
        Self {
            context,
            swapchain,
            swapchain_khr,
            properties,
            images,
            image_views,
        }
    }
}

impl Swapchain {
    pub fn swapchain_khr(&self) -> vk::SwapchainKHR {
        self.swapchain_khr
    }

    pub fn properties(&self) -> SwapchainProperties {
        self.properties
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }
}

impl Swapchain {
    pub fn acquire_next_image(
        &self,
        timeout: Option<u64>,
        semaphore: Option<vk::Semaphore>,
        fence: Option<vk::Fence>,
    ) -> VkResult<(u32, bool)> {
        unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                timeout.unwrap_or(std::u64::MAX),
                semaphore.unwrap_or_else(vk::Semaphore::null),
                fence.unwrap_or_else(vk::Fence::null),
            )
        }
    }

    pub fn present(&self, present_info: &vk::PresentInfoKHR) -> VkResult<bool> {
        unsafe {
            self.swapchain
                .queue_present(self.context.present_queue(), present_info)
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.image_views
                .iter()
                .for_each(|v| self.context.device().destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: vk::PhysicalDevice,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .expect("Failed to get physical device surface capabilities")
        };

        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(device, surface_khr)
                .expect("Failed to get physical device surface formats")
        };

        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .expect("Failed to get physical device surface present modes")
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    fn get_ideal_swapchain_properties(
        &self,
        preferred_format: Option<vk::SurfaceFormatKHR>,
        preferred_dimensions: [u32; 2],
        preferred_vsync: bool,
    ) -> SwapchainProperties {
        let format = Self::choose_swapchain_surface_format(&self.formats, preferred_format);
        let present_mode =
            Self::choose_swapchain_surface_present_mode(&self.present_modes, preferred_vsync);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
        let min_image_count = Self::choose_image_count(self.capabilities);
        SwapchainProperties {
            format,
            present_mode,
            extent,
            min_image_count,
        }
    }

    /// Choose the swapchain surface format.
    ///
    /// Will choose the preferred format or R8G8B8A8_SRGB/SRGB_NONLINEAR or
    /// the first available.
    fn choose_swapchain_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
        preferred_format: Option<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        if let Some(format) = preferred_format {
            if available_formats.contains(&format) {
                return format;
            }
        }

        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::R8G8B8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    /// Choose the swapchain present mode.
    ///
    /// If only one is supported then defaults to it (must be FIFO by the specs)
    /// If vsync is requested then we chose the first available among MAILBOX, FIFO_RELAXED, FIFO
    /// Otherwise we go for immediate
    fn choose_swapchain_surface_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
        preferred_vsync: bool,
    ) -> vk::PresentModeKHR {
        if available_present_modes.len() == 1 {
            return available_present_modes[0];
        }

        if preferred_vsync {
            if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if available_present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                vk::PresentModeKHR::FIFO_RELAXED
            } else {
                vk::PresentModeKHR::FIFO
            }
        } else {
            vk::PresentModeKHR::IMMEDIATE
        }
    }

    /// Choose the swapchain extent.
    ///
    /// If a current extent is defined it will be returned.
    /// Otherwise the surface extent clamped between the min
    /// and max image extent will be returned.
    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        preferred_dimensions: [u32; 2],
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != std::u32::MAX {
            return capabilities.current_extent;
        }

        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;
        let width = preferred_dimensions[0].min(max.width).max(min.width);
        let height = preferred_dimensions[1].min(max.height).max(min.height);
        vk::Extent2D { width, height }
    }

    fn choose_image_count(capabilities: vk::SurfaceCapabilitiesKHR) -> u32 {
        let max = capabilities.max_image_count;
        let mut preferred = capabilities.min_image_count + 1;
        if max > 0 && preferred > max {
            preferred = max;
        }
        preferred
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
    min_image_count: u32,
}
