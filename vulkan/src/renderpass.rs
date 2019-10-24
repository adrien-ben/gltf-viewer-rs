use super::Context;
use ash::{version::DeviceV1_0, vk, Device};
use std::sync::Arc;

pub struct SimpleRenderPass {
    context: Arc<Context>,
    render_pass: vk::RenderPass,
}

impl SimpleRenderPass {
    pub fn create(context: Arc<Context>, format: vk::Format) -> Self {
        let render_pass = create_render_pass(context.device(), format);

        Self {
            context,
            render_pass,
        }
    }
}

impl SimpleRenderPass {
    pub fn get_render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl Drop for SimpleRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

fn create_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
    let attachment_descs = [vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descs = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build()];

    let subpass_deps = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .expect("Failed to create render pass")
    }
}
