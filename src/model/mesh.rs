use super::{IndexBuffer, VertexBuffer};
use crate::{util, vulkan::*};
use ash::vk;
use gltf::{accessor::DataType, buffer::Data, mesh::Semantic, Accessor, Document};
use std::rc::Rc;

pub struct Mesh {
    vertices: VertexBuffer,
    indices: Option<IndexBuffer>,
}

impl Mesh {
    pub fn vertices(&self) -> &VertexBuffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Option<IndexBuffer> {
        &self.indices
    }
}

/// Vertex buffer byte offset / element count
type VertexBufferPart = (usize, usize);

/// Index buffer byte offset / element count / type
type IndexBufferPart = (usize, usize, vk::IndexType);

pub fn create_mesh_from_gltf(
    context: &Rc<Context>,
    document: &Document,
    buffers: &[Data],
) -> Vec<Mesh> {
    // (usize, usize) -> byte offset, element count
    let mut meshes_buffers = Vec::<(Option<IndexBufferPart>, VertexBufferPart)>::new();
    let mut all_vertices = Vec::<u8>::new();
    let mut all_indices = Vec::<u8>::new();

    // Gather vertices and indices from all the meshes in the document
    for mesh in document.meshes() {
        if let Some(primitive) = mesh.primitives().nth(0) {
            let indices = primitive.indices().map(|accessor| {
                let (indices, index_type) = extract_indices_from_accessor(&accessor, &buffers);
                let offset = all_indices.len();
                all_indices.extend_from_slice(&indices);
                (offset, accessor.count(), index_type)
            });

            if let Some(positions) = primitive.get(&Semantic::Positions) {
                let vertices = read_accessor(&positions, &buffers);
                let offset = all_vertices.len();
                all_vertices.extend_from_slice(&vertices);

                meshes_buffers.push((indices, (offset, positions.count())));
            }
        }
    }

    if !meshes_buffers.is_empty() {
        let indices = if all_indices.is_empty() {
            None
        } else {
            Some(Rc::new(create_device_local_buffer_with_data::<u8, _>(
                context,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &all_indices,
            )))
        };

        let vertices = Rc::new(create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &all_vertices,
        ));

        return meshes_buffers
            .iter()
            .map(|buffers| {
                let mesh_vertices = buffers.1;
                let vertex_buffer = VertexBuffer::new(
                    Rc::clone(&vertices),
                    mesh_vertices.0 as _,
                    mesh_vertices.1 as _,
                );

                let mesh_indices = buffers.0;
                let index_buffer = mesh_indices.map(|mesh_indices| {
                    IndexBuffer::new(
                        Rc::clone(indices.as_ref().unwrap()),
                        mesh_indices.0 as _,
                        mesh_indices.1 as _,
                        mesh_indices.2 as _,
                    )
                });

                Mesh {
                    vertices: vertex_buffer,
                    indices: index_buffer,
                }
            })
            .collect::<Vec<_>>();
    }

    Vec::new()
}

fn extract_indices_from_accessor(
    accessor: &Accessor,
    buffers: &[Data],
) -> (Vec<u8>, vk::IndexType) {
    let index_type = match accessor.data_type() {
        DataType::U32 => vk::IndexType::UINT32,
        _ => vk::IndexType::UINT16,
    };

    let indices = read_accessor(&accessor, &buffers);

    if accessor.data_type() == DataType::U8 {
        let u16_indices = indices
            .iter()
            .map(|val| u16::from(*val))
            .collect::<Vec<_>>();

        // TODO: Find something better
        let mut u8_indices = Vec::<u8>::new();
        for i in u16_indices {
            unsafe {
                u8_indices.extend_from_slice(util::any_as_u8_slice(&i));
            }
        }

        return (u8_indices, index_type);
    }

    (indices, index_type)
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
