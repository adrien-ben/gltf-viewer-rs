use crate::vulkan::*;
use ash::vk;
use std::{mem::size_of, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub tangent: [f32; 4],
    pub weights: [f32; 4],
    pub joints: [u32; 4],
}

impl Vertex for ModelVertex {
    fn get_bindings_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<ModelVertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attributes_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48,
            },
            vk::VertexInputAttributeDescription {
                location: 5,
                binding: 0,
                format: vk::Format::R32G32B32A32_UINT,
                offset: 64,
            },
        ]
    }
}

pub struct VertexBuffer {
    buffer: Arc<Buffer>,
    offset: vk::DeviceSize,
    element_count: u32,
}

impl VertexBuffer {
    pub fn new(buffer: Arc<Buffer>, offset: vk::DeviceSize, element_count: u32) -> Self {
        Self {
            buffer,
            offset,
            element_count,
        }
    }
}

impl VertexBuffer {
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    pub fn element_count(&self) -> u32 {
        self.element_count
    }
}

pub struct IndexBuffer {
    buffer: Arc<Buffer>,
    offset: vk::DeviceSize,
    element_count: u32,
    index_type: vk::IndexType,
}

impl IndexBuffer {
    pub fn new(buffer: Arc<Buffer>, offset: vk::DeviceSize, element_count: u32) -> Self {
        Self {
            buffer,
            offset,
            element_count,
            index_type: vk::IndexType::UINT32,
        }
    }
}

impl IndexBuffer {
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    pub fn element_count(&self) -> u32 {
        self.element_count
    }

    pub fn index_type(&self) -> vk::IndexType {
        self.index_type
    }
}
