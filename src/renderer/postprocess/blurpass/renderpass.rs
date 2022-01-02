use std::sync::Arc;
use vulkan::ash::{vk, Device};
use vulkan::{Context, Image, ImageParameters, Texture};

const BLUR_OUTPUT_FORMAT: vk::Format = vk::Format::R8_UNORM;

pub struct RenderPass {
    context: Arc<Context>,
    extent: vk::Extent2D,
    output_attachment: Texture,
    render_pass: vk::RenderPass,
}

impl RenderPass {
    pub fn create(context: Arc<Context>, extent: vk::Extent2D) -> Self {
        let output_attachment = create_output_texture(&context, BLUR_OUTPUT_FORMAT, extent);
        let render_pass = create_render_pass(context.device());

        Self {
            context,
            extent,
            output_attachment,
            render_pass,
        }
    }
}

impl RenderPass {
    pub fn get_output_attachment(&self) -> &Texture {
        &self.output_attachment
    }

    pub fn get_render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl RenderPass {
    pub fn create_framebuffer(&self) -> vk::Framebuffer {
        let attachments = [self.output_attachment.view];

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

fn create_render_pass(device: &Device) -> vk::RenderPass {
    // Attachements
    let attachment_descs = [
        // Color attachment
        vk::AttachmentDescription::builder()
            .format(BLUR_OUTPUT_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build(),
    ];

    let render_color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    // Subpasses
    let subpasses = {
        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&render_color_attachment_refs);
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

fn create_output_texture(
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
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            ..Default::default()
        },
    );

    image.transition_image_layout(
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);
    let sampler = Some(create_sampler(context));

    Texture::new(Arc::clone(context), image, view, sampler)
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
