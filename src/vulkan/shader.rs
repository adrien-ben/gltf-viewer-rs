use super::Context;
use ash::{version::DeviceV1_0, vk, Device};
use std::{path::Path, rc::Rc};

pub struct ShaderModule {
    context: Rc<Context>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new<P: AsRef<Path>>(context: Rc<Context>, path: P) -> Self {
        let source = read_shader_from_file(path);
        let module = create_shader_module(context.device(), &source);
        Self { context, module }
    }
}

impl ShaderModule {
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe { device.destroy_shader_module(self.module, None) };
    }
}

fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
    log::debug!("Loading shader file {}", path.as_ref().to_str().unwrap());
    let mut file = std::fs::File::open(path).unwrap();
    ash::util::read_spv(&mut file).unwrap()
}

fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::builder().code(code);
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}
