mod error;
mod vertex;

pub use self::{error::*, vertex::*};
use crate::vulkan::*;
use ash::vk;
use gltf::{accessor::DataType, buffer::Data, mesh::Semantic, Accessor};
use std::{error::Error, path::Path, rc::Rc, result::Result};

pub struct Model {
    vertices: VertexBuffer,
    indices: Option<IndexBuffer>,
}

impl Model {
    pub fn create_from_file<P: AsRef<Path>>(
        context: &Rc<Context>,
        path: P,
    ) -> Result<Self, Box<dyn Error>> {
        let (document, buffers, _) = gltf::import(path)?;

        for scene in document.scenes() {
            for node in scene.nodes() {
                if let Some(mesh) = node.mesh() {
                    if let Some(primitive) = mesh.primitives().nth(0) {
                        let indices = primitive.indices().map(|accessor| {
                            create_index_buffer_from_accessor(context, &accessor, &buffers)
                        });

                        if let Some(positions) = primitive.get(&Semantic::Positions) {
                            let vertices =
                                create_vertex_buffer_from_accessor(context, &positions, &buffers);

                            return Ok(Model { vertices, indices });
                        }
                    }
                }
            }
        }

        Err(Box::new(ModelLoadingError::new(
            "Could not find any renderable primitives",
        )))
    }
}

impl Model {
    pub fn vertices(&self) -> &VertexBuffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Option<IndexBuffer> {
        &self.indices
    }
}

fn create_index_buffer_from_accessor(
    context: &Rc<Context>,
    accessor: &Accessor,
    buffers: &[Data],
) -> IndexBuffer {
    let index_type = match accessor.data_type() {
        DataType::U32 => vk::IndexType::UINT32,
        _ => vk::IndexType::UINT16,
    };

    let indices = read_accessor(&accessor, &buffers);
    let indices = match accessor.data_type() {
        DataType::U8 => {
            let indices = indices
                .iter()
                .map(|val| u16::from(*val))
                .collect::<Vec<_>>();
            create_device_local_buffer_with_data::<u8, _>(
                context,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            )
        }
        _ => create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        ),
    };
    (indices, accessor.count() as _, index_type)
}

fn create_vertex_buffer_from_accessor(
    context: &Rc<Context>,
    accessor: &Accessor,
    buffers: &[Data],
) -> VertexBuffer {
    let vertices = read_accessor(&accessor, &buffers);
    let vertices = create_device_local_buffer_with_data::<u8, _>(
        context,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &vertices,
    );
    (vertices, accessor.count() as _)
}

fn read_accessor(accessor: &Accessor, buffers: &[Data]) -> Vec<u8> {
    let view = accessor.view();
    let buffer = view.buffer();
    let data = &buffers[buffer.index()];

    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or_else(|| accessor.size());

    let mut vertices = Vec::<u8>::new();
    for component_index in 0..accessor.count() {
        let offset = offset + component_index * stride;
        for byte_index in 0..accessor.size() {
            vertices.push(data[offset + byte_index]);
        }
    }
    vertices
}
