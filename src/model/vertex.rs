use crate::vulkan::*;
use ash::vk;
use std::{mem::size_of, rc::Rc};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct ModelVertex {
    position: [f32; 3],
    normals: [f32; 3],
    tex_coords: [f32; 2],
    tangents: [f32; 4],
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
        ]
    }
}

pub struct VertexBuffer {
    buffer: Rc<Buffer>,
    offset: vk::DeviceSize,
    element_count: u32,
}

impl VertexBuffer {
    pub fn new(buffer: Rc<Buffer>, offset: vk::DeviceSize, element_count: u32) -> Self {
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
    buffer: Rc<Buffer>,
    offset: vk::DeviceSize,
    element_count: u32,
    index_type: vk::IndexType,
}

impl IndexBuffer {
    pub fn new(
        buffer: Rc<Buffer>,
        offset: vk::DeviceSize,
        element_count: u32,
        index_type: vk::IndexType,
    ) -> Self {
        Self {
            buffer,
            offset,
            element_count,
            index_type,
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
