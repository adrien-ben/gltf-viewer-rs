use std::sync::Arc;

use vulkan::{ash::vk, Context, Image, ImageParameters, Texture};

pub const GBUFFER_NORMALS_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const AO_MAP_FORMAT: vk::Format = vk::Format::R8_UNORM;
pub const SCENE_COLOR_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
pub const BLOOM_FORMAT: vk::Format = vk::Format::B10G11R11_UFLOAT_PACK32;
pub const BLOOM_MIP_LEVELS: u32 = 5;

pub struct Attachments {
    pub gbuffer_normals: Texture,
    pub gbuffer_depth: Texture,
    pub ssao: Texture,
    pub ssao_blur: Texture,
    pub scene_color: Texture,
    pub scene_depth: Texture,
    pub scene_resolve: Option<Texture>,
    pub bloom: BloomAttachment,
}

pub struct BloomAttachment {
    context: Arc<Context>,
    pub image: Image,
    pub mips_views: Vec<vk::ImageView>,
    pub mips_resolution: Vec<vk::Extent2D>,
    pub sampler: vk::Sampler,
}

impl Drop for BloomAttachment {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_sampler(self.sampler, None);
            self.mips_views
                .iter()
                .for_each(|v| self.context.device().destroy_image_view(*v, None));
        }
    }
}

impl Attachments {
    pub fn new(
        context: &Arc<Context>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let gbuffer_normals = create_gbuffer_normals(context, extent);
        let gbuffer_depth = create_gbuffer_depth(context, depth_format, extent);
        let ssao = create_ssao(context, extent);
        let ssao_blur = create_ssao_blur(context, extent);
        let scene_color = create_scene_color(context, extent, msaa_samples);
        let scene_depth = create_scene_depth(context, depth_format, extent, msaa_samples);
        let scene_resolve = match msaa_samples {
            vk::SampleCountFlags::TYPE_1 => None,
            _ => Some(create_scene_resolve(context, extent)),
        };
        let bloom = create_bloom(context, extent);

        Self {
            gbuffer_normals,
            gbuffer_depth,
            ssao,
            ssao_blur,
            scene_color,
            scene_depth,
            scene_resolve,
            bloom,
        }
    }
}

impl Attachments {
    pub fn get_scene_resolved_color(&self) -> &Texture {
        self.scene_resolve.as_ref().unwrap_or(&self.scene_color)
    }
}

fn create_gbuffer_normals(context: &Arc<Context>, extent: vk::Extent2D) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: vk::SampleCountFlags::TYPE_1,
            format: GBUFFER_NORMALS_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);
    let sampler = Some(create_sampler(
        context,
        vk::Filter::NEAREST,
        vk::Filter::NEAREST,
    ));

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_gbuffer_depth(
    context: &Arc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: vk::SampleCountFlags::TYPE_1,
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::DEPTH);

    let sampler = Some(create_sampler(
        context,
        vk::Filter::NEAREST,
        vk::Filter::NEAREST,
    ));

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_ssao(context: &Arc<Context>, extent: vk::Extent2D) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: vk::SampleCountFlags::TYPE_1,
            format: AO_MAP_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);
    let sampler = Some(create_sampler(
        context,
        vk::Filter::NEAREST,
        vk::Filter::NEAREST,
    ));

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_ssao_blur(context: &Arc<Context>, extent: vk::Extent2D) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: vk::SampleCountFlags::TYPE_1,
            format: AO_MAP_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);
    let sampler = Some(create_sampler(
        context,
        vk::Filter::NEAREST,
        vk::Filter::NEAREST,
    ));

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_scene_color(
    context: &Arc<Context>,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    let image_usage = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
        }
        _ => vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
    };
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: msaa_samples,
            format: SCENE_COLOR_FORMAT,
            usage: image_usage,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

    let sampler = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => Some(create_sampler(
            context,
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
        )),
        _ => None,
    };

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_scene_depth(
    context: &Arc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    let image_usage = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        _ => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
        }
    };
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: msaa_samples,
            format,
            usage: image_usage,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::DEPTH);

    let sampler = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => Some(create_sampler(
            context,
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
        )),
        _ => None,
    };

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_scene_resolve(context: &Arc<Context>, extent: vk::Extent2D) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            format: SCENE_COLOR_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

    let sampler = create_sampler(context, vk::Filter::NEAREST, vk::Filter::NEAREST);

    Texture::new(Arc::clone(context), image, view, Some(sampler))
}

fn create_bloom(context: &Arc<Context>, extent: vk::Extent2D) -> BloomAttachment {
    let mut extent = vk::Extent2D {
        width: extent.width / 2,
        height: extent.height / 2,
    };

    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            format: BLOOM_FORMAT,
            mip_levels: BLOOM_MIP_LEVELS,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    let mips_views =
        image.create_mips_views(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

    let mut mips_resolution = vec![];
    for _ in 0..BLOOM_MIP_LEVELS {
        mips_resolution.push(extent);
        extent.width /= 2;
        extent.height /= 2;
    }

    let sampler = create_sampler(context, vk::Filter::LINEAR, vk::Filter::LINEAR);

    BloomAttachment {
        context: context.clone(),
        image,
        mips_views,
        mips_resolution,
        sampler,
    }
}

fn create_sampler(
    context: &Arc<Context>,
    min_filter: vk::Filter,
    mag_filter: vk::Filter,
) -> vk::Sampler {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(mag_filter)
        .min_filter(min_filter)
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
        .max_lod(1.0);

    unsafe {
        context
            .device()
            .create_sampler(&sampler_info, None)
            .expect("Failed to create sampler")
    }
}
