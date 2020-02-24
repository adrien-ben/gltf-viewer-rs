use ash::prelude::*;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use ash::RawPtr;
use std::ffi::CStr;
use std::mem;

#[derive(Clone)]
pub struct CreateRenderpass2 {
    handle: vk::Device,
    create_renderpass_2_fn: vk::KhrCreateRenderpass2Fn,
}

impl CreateRenderpass2 {
    pub fn new<I: InstanceV1_0, D: DeviceV1_0>(instance: &I, device: &D) -> CreateRenderpass2 {
        let create_renderpass_2_fn = vk::KhrCreateRenderpass2Fn::load(|name| unsafe {
            mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
        });
        CreateRenderpass2 {
            handle: device.handle(),
            create_renderpass_2_fn,
        }
    }

    pub fn name() -> &'static CStr {
        vk::KhrCreateRenderpass2Fn::name()
    }

    #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateRenderPass2.html>"]
    pub unsafe fn create_render_pass2_khr(
        &self,
        create_info: &vk::RenderPassCreateInfo2KHR,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> VkResult<vk::RenderPass> {
        let mut render_pass = mem::zeroed();
        let err_code = self.create_renderpass_2_fn.create_render_pass2_khr(
            self.handle,
            create_info,
            allocation_callbacks.as_raw_ptr(),
            &mut render_pass,
        );
        match err_code {
            vk::Result::SUCCESS => Ok(render_pass),
            _ => Err(err_code),
        }
    }

    #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBeginRenderPass2.html>"]
    pub unsafe fn cmd_begin_render_pass2_khr(
        &self,
        command_buffer: vk::CommandBuffer,
        render_pass_begin_info: &vk::RenderPassBeginInfo,
        subpass_begin_info: &vk::SubpassBeginInfoKHR,
    ) {
        self.create_renderpass_2_fn.cmd_begin_render_pass2_khr(
            command_buffer,
            render_pass_begin_info,
            subpass_begin_info,
        );
    }

    #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdNextSubpass2.html>"]
    pub unsafe fn cmd_next_subpass2_khr(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::SubpassBeginInfoKHR,
        end_info: &vk::SubpassEndInfoKHR,
    ) {
        self.create_renderpass_2_fn
            .cmd_next_subpass2_khr(command_buffer, begin_info, end_info);
    }

    #[doc = "<https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdEndRenderPass2.html>"]
    pub unsafe fn cmd_end_render_pass2_khr(
        &self,
        command_buffer: vk::CommandBuffer,
        end_info: &vk::SubpassEndInfoKHR,
    ) {
        self.create_renderpass_2_fn
            .cmd_end_render_pass2_khr(command_buffer, end_info);
    }

    pub fn fp(&self) -> &vk::KhrCreateRenderpass2Fn {
        &self.create_renderpass_2_fn
    }

    pub fn device(&self) -> vk::Device {
        self.handle
    }
}
