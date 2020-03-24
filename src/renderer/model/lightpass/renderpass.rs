use std::sync::Arc;
use vulkan::ash::{version::DeviceV1_0, vk, Device};
use vulkan::{Context, Image, ImageParameters, Texture};

const COLOR_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

pub struct RenderPass {
    context: Arc<Context>,
    extent: vk::Extent2D,
    color_attachment: Texture,
    depth_attachment: Texture,
    color_resolve_attachment: Option<Texture>,
    render_pass: vk::RenderPass,
}

impl RenderPass {
    pub fn create(
        context: Arc<Context>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let color_attachment = create_color_texture(&context, COLOR_FORMAT, extent, msaa_samples);
        let depth_attachment = create_depth_texture(&context, depth_format, extent, msaa_samples);
        let color_resolve_attachment = match msaa_samples {
            vk::SampleCountFlags::TYPE_1 => None,
            _ => Some(create_color_resolve_texture(&context, COLOR_FORMAT, extent)),
        };
        let render_pass = create_render_pass(context.device(), depth_format, msaa_samples);

        Self {
            context,
            extent,
            color_attachment,
            depth_attachment,
            color_resolve_attachment,
            render_pass,
        }
    }
}

impl RenderPass {
    pub fn get_color_attachment(&self) -> &Texture {
        self.color_resolve_attachment
            .as_ref()
            .map_or(&self.color_attachment, |a| &a)
    }

    pub fn get_render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl RenderPass {
    pub fn create_framebuffer(&self) -> vk::Framebuffer {
        let attachments = {
            let color = self.color_attachment.view;
            let depth = self.depth_attachment.view;
            match self.color_resolve_attachment.as_ref() {
                Some(color_resolve) => vec![color, depth, color_resolve.view],
                _ => vec![color, depth],
            }
        };

        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(self.render_pass)
            .attachments(&attachments)
            .width(self.extent.width)
            .height(self.extent.height)
            .layers(1);
        unsafe {
            self.context
                .device()
                .create_framebuffer(&framebuffer_info, None)
                .expect("Failed to create framebuffer")
        }
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

fn create_render_pass(
    device: &Device,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
) -> vk::RenderPass {
    // Attachements
    let color_final_layout = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        _ => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let color_store_op = if msaa_samples == vk::SampleCountFlags::TYPE_1 {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    };

    let mut attachment_descs = vec![
        // Color attachment
        vk::AttachmentDescription::builder()
            .format(COLOR_FORMAT)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(color_store_op)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(color_final_layout)
            .build(),
        // Depth attachment
        vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build(),
    ];
    if msaa_samples != vk::SampleCountFlags::TYPE_1 {
        // Resolve attachment
        attachment_descs.push(
            vk::AttachmentDescription::builder()
                .format(COLOR_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build(),
        );
    }

    let render_color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let depth_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let resolve_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    // Subpasses
    let subpasses = {
        let mut subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&render_color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);
        if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            subpass_desc = subpass_desc.resolve_attachments(&resolve_attachment_refs)
        }
        [subpass_desc.build()]
    };

    // Dependencies
    let subpass_deps = [
        vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build(),
        vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build(),
    ];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpasses)
        .dependencies(&subpass_deps);

    unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .expect("Failed to create render pass")
    }
}

fn create_color_texture(
    context: &Arc<Context>,
    format: vk::Format,
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
            format,
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
        vk::SampleCountFlags::TYPE_1 => Some(create_sampler(context)),
        _ => None,
    };

    Texture::new(Arc::clone(context), image, view, sampler)
}

/// Create the depth buffer texture (image, memory and view).
///
/// This function also transitions the image to be ready to be used
/// as a depth/stencil attachement.
fn create_depth_texture(
    context: &Arc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    let image_usage = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
        }
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
        vk::SampleCountFlags::TYPE_1 => Some(create_sampler(context)),
        _ => None,
    };

    Texture::new(Arc::clone(context), image, view, sampler)
}

fn create_color_resolve_texture(
    context: &Arc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
) -> Texture {
    let image = Image::create(
        Arc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

    let sampler = create_sampler(context);

    Texture::new(Arc::clone(context), image, view, Some(sampler))
}

fn create_sampler(context: &Arc<Context>) -> vk::Sampler {
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
        .max_lod(1.0);

    unsafe {
        context
            .device()
            .create_sampler(&sampler_info, None)
            .expect("Failed to create sampler")
    }
}
