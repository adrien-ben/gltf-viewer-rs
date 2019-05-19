use crate::{
    util::*,
    vulkan::{create_device_local_buffer_with_data, Buffer, Context, Texture},
};
use ash::vk;
use std::{mem::size_of, path::Path, rc::Rc};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SkyboxVertex {
    position: [f32; 3],
}

impl SkyboxVertex {
    pub fn get_bindings_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<SkyboxVertex>() as _,
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

pub struct SkyboxModel {
    vertices: Buffer,
    indices: Buffer,
}

impl SkyboxModel {
    pub fn new(context: &Rc<Context>) -> Self {
        let indices: [u32; 36] = [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ];
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );
        let vertices: [f32; 24] = [
            -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5,
            0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
        ];
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        SkyboxModel { vertices, indices }
    }
}

impl SkyboxModel {
    pub fn vertices(&self) -> &Buffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Buffer {
        &self.indices
    }
}

pub fn create_skybox_cubemap(context: &Rc<Context>) -> Texture {
    let (w, h, px) = load_hdr_image(Path::new("assets/env/px.hdr"));
    let (_, _, nx) = load_hdr_image(Path::new("assets/env/nx.hdr"));
    let (_, _, py) = load_hdr_image(Path::new("assets/env/py.hdr"));
    let (_, _, ny) = load_hdr_image(Path::new("assets/env/ny.hdr"));
    let (_, _, pz) = load_hdr_image(Path::new("assets/env/pz.hdr"));
    let (_, _, nz) = load_hdr_image(Path::new("assets/env/nz.hdr"));

    let data_size = (w * h * 4 * 6) as usize;
    let mut data = Vec::with_capacity(data_size);
    data.extend_from_slice(&px);
    data.extend_from_slice(&nx);
    data.extend_from_slice(&py);
    data.extend_from_slice(&ny);
    data.extend_from_slice(&pz);
    data.extend_from_slice(&nz);

    Texture::create_cubemap(&context, w, &data)
}
