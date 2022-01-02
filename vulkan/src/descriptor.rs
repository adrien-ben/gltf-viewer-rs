use super::context::Context;
use ash::vk;
use std::sync::Arc;

pub struct Descriptors {
    context: Arc<Context>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl Descriptors {
    pub fn new(
        context: Arc<Context>,
        layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
        sets: Vec<vk::DescriptorSet>,
    ) -> Self {
        Self {
            context,
            layout,
            pool,
            sets,
        }
    }
}

impl Descriptors {
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }

    pub fn sets(&self) -> &[vk::DescriptorSet] {
        &self.sets
    }

    pub fn set_sets(&mut self, sets: Vec<vk::DescriptorSet>) {
        self.sets = sets;
    }
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
