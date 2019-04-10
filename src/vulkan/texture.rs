use super::{buffer::*, context::*, image::*};
use ash::{version::DeviceV1_0, vk};
use std::{
    mem::{align_of, size_of},
    rc::Rc,
};

pub struct Texture {
    context: Rc<Context>,
    pub image: Image,
    pub view: vk::ImageView,
    pub sampler: Option<vk::Sampler>,
}

impl Texture {
    pub fn new(
        context: Rc<Context>,
        image: Image,
        view: vk::ImageView,
        sampler: Option<vk::Sampler>,
    ) -> Self {
        Texture {
            context,
            image,
            view,
            sampler,
        }
    }

    pub fn from_rgba(context: &Rc<Context>, width: u32, height: u32, data: &[u8]) -> Self {
        let max_mip_levels = ((width.min(height) as f32).log2().floor() + 1.0) as u32;
        let extent = vk::Extent2D { width, height };
        let image_size = (data.len() * size_of::<u8>()) as vk::DeviceSize;
        let device = context.device();

        let buffer = Buffer::create(
            Rc::clone(context),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = device
                .map_memory(buffer.memory, 0, image_size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, buffer.size);
            align.copy_from_slice(&data);
            device.unmap_memory(buffer.memory);
        }

        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            image.transition_image_layout(
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            image.copy_buffer(&buffer, extent);

            image.generate_mipmaps(extent, vk::Format::R8G8B8A8_UNORM, max_mip_levels);
        }

        let image_view = image.create_view(
            max_mip_levels,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        );

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(max_mip_levels as _)
                .build();

            unsafe { device.create_sampler(&sampler_info, None).unwrap() }
        };

        Texture::new(Rc::clone(context), image, image_view, Some(sampler))
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            if let Some(sampler) = self.sampler.take() {
                self.context.device().destroy_sampler(sampler, None);
            }
            self.context.device().destroy_image_view(self.view, None);
        }
    }
}
