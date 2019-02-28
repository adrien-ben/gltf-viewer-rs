use crate::vulkan::*;
use ash::vk;
use std::{mem::size_of, rc::Rc};

pub const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [-0.5, 0.5, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.0],
    },
    Vertex {
        position: [0.0, -0.5, 0.0],
    },
];

pub const INDICES: [u32; 3] = [0, 1, 2];

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

pub struct Model {
    indices: Buffer,
    vertices: Buffer,
    index_count: u32,
}

impl Model {
    pub fn create_triangle(context: &Rc<Context>) -> Self {
        let vertices = create_device_local_buffer_with_data::<Vertex, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &VERTICES,
        );
        let indices = create_device_local_buffer_with_data::<u32, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &INDICES,
        );
        Model {
            vertices,
            indices,
            index_count: INDICES.len() as _,
        }
    }
}

impl Model {
    pub fn vertices(&self) -> &Buffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Buffer {
        &self.indices
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}
