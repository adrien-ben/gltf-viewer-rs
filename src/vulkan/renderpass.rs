use super::{Context, Image, ImageParameters, Texture};
use ash::{version::DeviceV1_0, vk, Device};
use std::rc::Rc;

pub struct RenderPass {
    context: Rc<Context>,
    color_attachment: Option<Texture>,
    depth_attachment: Texture,
    render_pass: vk::RenderPass,
}

impl RenderPass {
    pub fn create(
        context: Rc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
        depth_format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let render_pass = create_render_pass(context.device(), format, depth_format, msaa_samples);

        let color_attachment = match msaa_samples {
            vk::SampleCountFlags::TYPE_1 => None,
            _ => Some(create_color_texture(&context, format, extent, msaa_samples)),
        };

        let depth_attachment = create_depth_texture(&context, depth_format, extent, msaa_samples);

        Self {
            context,
            color_attachment,
            depth_attachment,
            render_pass,
        }
    }
}

impl RenderPass {
    pub fn get_color_attachment(&self) -> Option<&Texture> {
        self.color_attachment.as_ref()
    }

    pub fn get_depth_attachment(&self) -> &Texture {
        &self.depth_attachment
    }

    pub fn get_render_pass(&self) -> vk::RenderPass {
        self.render_pass
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
    format: vk::Format,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
) -> vk::RenderPass {
    let final_image_layout = match msaa_samples {
        vk::SampleCountFlags::TYPE_1 => vk::ImageLayout::PRESENT_SRC_KHR,
        _ => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let mut attachment_descs = vec![
        vk::AttachmentDescription::builder()
            .format(format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(final_image_layout)
            .build(),
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
        attachment_descs.push(
            vk::AttachmentDescription::builder()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build(),
        );
    }

    let color_attachment_refs = [vk::AttachmentReference::builder()
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

    let subpass_descs = {
        let mut subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);
        if msaa_samples != vk::SampleCountFlags::TYPE_1 {
            subpass_desc = subpass_desc.resolve_attachments(&resolve_attachment_refs);
        }
        [subpass_desc.build()]
    };

    let subpass_dep = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build();
    let subpass_deps = [subpass_dep];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
}

fn create_color_texture(
    context: &Rc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    let image = Image::create(
        Rc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: msaa_samples,
            format,
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

    Texture::new(Rc::clone(context), image, view, None)
}

/// Create the depth buffer texture (image, memory and view).
///
/// This function also transitions the image to be ready to be used
/// as a depth/stencil attachement.
fn create_depth_texture(
    context: &Rc<Context>,
    format: vk::Format,
    extent: vk::Extent2D,
    msaa_samples: vk::SampleCountFlags,
) -> Texture {
    let image = Image::create(
        Rc::clone(context),
        ImageParameters {
            mem_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            sample_count: msaa_samples,
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::DEPTH);

    Texture::new(Rc::clone(context), image, view, None)
}
