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
            1,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::empty(),
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            image.transition_image_layout(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            image.copy_buffer(&buffer, extent);

            image.generate_mipmaps(extent);
        }

        let image_view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

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

    pub fn from_rgba_32(context: &Rc<Context>, width: u32, height: u32, data: &[f32]) -> Self {
        let max_mip_levels = ((width.min(height) as f32).log2().floor() + 1.0) as u32;
        let extent = vk::Extent2D { width, height };
        let image_size = (data.len() * size_of::<f32>()) as vk::DeviceSize;
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
            1,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::empty(),
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            image.transition_image_layout(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            image.copy_buffer(&buffer, extent);

            image.generate_mipmaps(extent);
        }

        let image_view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
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

    pub fn create_cubemap_from_data(context: &Rc<Context>, size: u32, data: &[f32]) -> Self {
        let max_mip_levels = (size as f32).log2().floor() as u32 + 1;
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };

        let image_size = (data.len() * size_of::<f32>()) as vk::DeviceSize;
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
            let mut align = ash::util::Align::new(ptr, align_of::<f32>() as _, buffer.size);
            align.copy_from_slice(&data);
            device.unmap_memory(buffer.memory);
        }

        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            6,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            image.transition_image_layout(
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            image.copy_buffer(&buffer, extent);

            image.generate_mipmaps(extent);
        }

        let image_view = image.create_view(vk::ImageViewType::CUBE, vk::ImageAspectFlags::COLOR);

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .max_anisotropy(0.0)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
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

    pub fn create_renderable_cubemap(context: &Rc<Context>, size: u32, mip_levels: u32) -> Self {
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };

        let device = context.device();

        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            6,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        );

        image.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let image_view = image.create_view(vk::ImageViewType::CUBE, vk::ImageAspectFlags::COLOR);

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .max_anisotropy(0.0)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(mip_levels as _)
                .build();

            unsafe { device.create_sampler(&sampler_info, None).unwrap() }
        };

        Texture::new(Rc::clone(context), image, image_view, Some(sampler))
    }

    pub fn create_renderable_texture(
        context: &Rc<Context>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        let extent = vk::Extent2D { width, height };

        let device = context.device();

        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            1,
            1,
            vk::SampleCountFlags::TYPE_1,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        );

        image.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let image_view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .max_anisotropy(0.0)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(1.0)
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
