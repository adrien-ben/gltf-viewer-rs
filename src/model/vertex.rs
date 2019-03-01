use crate::vulkan::*;
use ash::vk;
use std::mem::size_of;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    pub fn get_bindings_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    pub fn get_attributes_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        }]
    }
}

pub type VertexBuffer = (Buffer, u32);
pub type IndexBuffer = (Buffer, u32, vk::IndexType);
