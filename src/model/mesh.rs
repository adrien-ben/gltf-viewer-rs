use super::{IndexBuffer, VertexBuffer, Material};
use crate::{util::*, vulkan::*};
use ash::vk;
use gltf::{accessor::DataType, buffer::Data, mesh::Semantic, Accessor, Document, mesh::Primitive as GltfPrimitive};
use std::rc::Rc;

pub struct Mesh {
    primitives: Vec<Primitive>,
}

impl Mesh {
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }
}

pub struct Primitive {
    vertices: VertexBuffer,
    indices: Option<IndexBuffer>,
    material: Material,
}

impl Primitive {
    pub fn vertices(&self) -> &VertexBuffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Option<IndexBuffer> {
        &self.indices
    }

    pub fn material(&self) -> Material {
        self.material
    }
}

/// Vertex buffer byte offset / element count
type VertexBufferPart = (usize, usize);

/// Index buffer byte offset / element count / type
type IndexBufferPart = (usize, usize, vk::IndexType);

pub fn create_meshes_from_gltf(
    context: &Rc<Context>,
    document: &Document,
    buffers: &[Data],
) -> Vec<Mesh> {
    // (usize, usize) -> byte offset, element count
    let mut meshes_data = Vec::<Vec<(Option<IndexBufferPart>, VertexBufferPart, Material)>>::new();
    let mut all_vertices = Vec::<u8>::new();
    let mut all_indices = Vec::<u8>::new();

    // Gather vertices and indices from all the meshes in the document
    for mesh in document.meshes() {
        let mut primitives_buffers = Vec::<(Option<IndexBufferPart>, VertexBufferPart, Material)>::new();

        for primitive in mesh.primitives() {
            let indices = primitive.indices().map(|accessor| {
                let (indices, index_type) = extract_indices_from_accessor(&accessor, &buffers);
                let offset = all_indices.len();
                all_indices.extend_from_slice(&indices);
                (offset, accessor.count(), index_type)
            });

            if let Some(accessor) = primitive.get(&Semantic::Positions) {
                let positions = read_accessor(&accessor, &buffers);
                let normals = read_normals(&primitive, &buffers);

                let mut vertices = Vec::<u8>::new();

                for elt_index in 0..accessor.count() {
                    push_vec3(&Some(&positions), elt_index, &mut vertices);
                    push_vec3(&normals.as_ref().map(|v| &v[..]), elt_index, &mut vertices);
                }

                let offset = all_vertices.len();
                all_vertices.extend_from_slice(&vertices);

                let material = Material::from(primitive.material());

                primitives_buffers.push((indices, (offset, accessor.count()), material));
            }
        }

        meshes_data.push(primitives_buffers);
    }

    if !meshes_data.is_empty() {
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

        return meshes_data
            .iter()
            .map(|primitives_buffers| {
                let primitives = primitives_buffers
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

                        Primitive {
                            vertices: vertex_buffer,
                            indices: index_buffer,
                            material: buffers.2,
                        }
                    })
                    .collect::<Vec<_>>();
                Mesh { primitives }
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
                u8_indices.extend_from_slice(any_as_u8_slice(&i));
            }
        }

        return (u8_indices, index_type);
    }

    (indices, index_type)
}

fn read_normals(primitive: &GltfPrimitive, buffers: &[Data]) -> Option<Vec<u8>> {
    primitive
        .get(&Semantic::Normals)
        .map(|normals| read_accessor(&normals, &buffers))
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

fn push_vec3(src: &Option<&[u8]>, index: usize, dest: &mut Vec<u8>) {
    let left = index * 12;
    let right = left + 12;

    if let Some(src) = src {
        dest.extend_from_slice(&src[left..right]);
    } else {
        unsafe {
            let one: [f32; 3] = [1.0, 1.0, 1.0];
            dest.extend_from_slice(any_as_u8_slice(&one));
        }
    };
}
