use super::{context::Context, image::create_image_view, renderpass::RenderPass};
use ash::{
    extensions::khr::{Surface, Swapchain as SwapchainLoader},
    prelude::VkResult,
    version::DeviceV1_0,
    vk, Device,
};
use std::rc::Rc;

pub struct Swapchain {
    context: Rc<Context>,
    swapchain: SwapchainLoader,
    swapchain_khr: vk::SwapchainKHR,
    properties: SwapchainProperties,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Swapchain {
    /// Create the swapchain with optimal settings possible with
    /// `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    pub fn create(
        context: Rc<Context>,
        swapchain_support_details: SwapchainSupportDetails,
        dimensions: [u32; 2],
        preferred_vsync: bool,
        render_pass: &RenderPass,
    ) -> Self {
        let properties =
            swapchain_support_details.get_ideal_swapchain_properties(dimensions, preferred_vsync);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let image_count = properties.image_count;

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let queue_families_indices = context.queue_families_indices();
        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(context.surface_khr())
                .min_image_count(image_count)
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

        let swapchain = SwapchainLoader::new(context.instance(), context.device());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        let views = Self::create_views(context.device(), &images, properties);

        let mut swapchain = Self::new(
            context,
            swapchain,
            swapchain_khr,
            properties,
            images,
            views,
            vec![],
        );
        swapchain.create_framebuffers(render_pass);
        swapchain
    }

    /// Create one image view for each image of the swapchain.
    fn create_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                create_image_view(
                    device,
                    *image,
                    vk::ImageViewType::TYPE_2D,
                    1,
                    1,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn new(
        context: Rc<Context>,
        swapchain: SwapchainLoader,
        swapchain_khr: vk::SwapchainKHR,
        properties: SwapchainProperties,
        images: Vec<vk::Image>,
        image_views: Vec<vk::ImageView>,
        framebuffers: Vec<vk::Framebuffer>,
    ) -> Self {
        Self {
            context,
            swapchain,
            swapchain_khr,
            properties,
            images,
            image_views,
            framebuffers,
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

    pub fn framebuffers(&self) -> &[vk::Framebuffer] {
        &self.framebuffers
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }
}

impl Swapchain {
    fn create_framebuffers(&mut self, render_pass: &RenderPass) {
        self.framebuffers = self
            .image_views
            .iter()
            .map(|view| match render_pass.get_color_attachment() {
                Some(texture) => vec![texture.view, render_pass.get_depth_attachment().view, *view],
                _ => vec![*view, render_pass.get_depth_attachment().view],
            })
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass.get_render_pass())
                    .attachments(&attachments)
                    .width(self.properties.extent.width)
                    .height(self.properties.extent.height)
                    .layers(1);
                unsafe {
                    self.context
                        .device()
                        .create_framebuffer(&framebuffer_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>();
    }

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
            self.framebuffers
                .iter()
                .for_each(|f| self.context.device().destroy_framebuffer(*f, None));
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
    pub fn new(device: vk::PhysicalDevice, surface: &Surface, surface_khr: vk::SurfaceKHR) -> Self {
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .unwrap()
        };

        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(device, surface_khr)
                .unwrap()
        };

        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .unwrap()
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
        preferred_vsync: bool,
    ) -> SwapchainProperties {
        let format = Self::choose_swapchain_surface_format(&self.formats);
        let present_mode =
            Self::choose_swapchain_surface_present_mode(&self.present_modes, preferred_vsync);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
        let image_count = Self::choose_image_count(self.capabilities);
        SwapchainProperties {
            format,
            present_mode,
            extent,
            image_count,
        }
    }

    /// Choose the swapchain surface format.
    ///
    /// Will choose B8G8R8A8_UNORM/SRGB_NONLINEAR if possible or
    /// the first available otherwise.
    fn choose_swapchain_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
            return vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    /// Choose the swapchain present mode.
    ///
    /// Will favor MAILBOX if present otherwise FIFO.
    /// If none is present it will fallback to IMMEDIATE.
    fn choose_swapchain_surface_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
        preferred_vsync: bool,
    ) -> vk::PresentModeKHR {
        if preferred_vsync && !available_present_modes.contains(&vk::PresentModeKHR::FIFO) {
            log::warn!("Vsync was requested but FIFO present mode is not supported");
        }

        if preferred_vsync && available_present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
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
    pub image_count: u32,
}
