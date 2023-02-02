use ash::vk;
use std::{mem::size_of, sync::Arc};
use vulkan::*;

const POSITION_LOCATION: u32 = 0;
const NORMAL_LOCATION: u32 = 1;
const TEX_COORDS_0_LOCATION: u32 = 2;
const TEX_COORDS_1_LOCATION: u32 = 3;
const TANGENT_LOCATION: u32 = 4;
const WEIGHTS_LOCATION: u32 = 5;
const JOINTS_LOCATION: u32 = 6;
const COLOR_LOCATION: u32 = 7;

const POSITION_OFFSET: u32 = 0;
const NORMAL_OFFSET: u32 = 12;
const TEX_COORDS_0_OFFSET: u32 = 24;
const TEX_COORDS_1_OFFSET: u32 = 32;
const TANGENT_OFFSET: u32 = 40;
const WEIGHTS_OFFSET: u32 = 56;
const JOINTS_OFFSET: u32 = 72;
const COLOR_OFFSET: u32 = 88;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords_0: [f32; 2],
    pub tex_coords_1: [f32; 2],
    pub tangent: [f32; 4],
    pub weights: [f32; 4],
    pub joints: [u32; 4],
    pub colors: [f32; 4],
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
                location: POSITION_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: POSITION_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: NORMAL_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: NORMAL_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: TEX_COORDS_0_LOCATION,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: TEX_COORDS_0_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: TEX_COORDS_1_LOCATION,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: TEX_COORDS_1_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: TANGENT_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: TANGENT_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: WEIGHTS_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: WEIGHTS_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: JOINTS_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32A32_UINT,
                offset: JOINTS_OFFSET,
            },
            vk::VertexInputAttributeDescription {
                location: COLOR_LOCATION,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: COLOR_OFFSET,
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
