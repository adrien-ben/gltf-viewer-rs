use super::{context::*, image::*};
use ash::{version::DeviceV1_0, vk};
use std::rc::Rc;

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
