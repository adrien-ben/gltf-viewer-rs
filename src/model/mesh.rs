use super::{IndexBuffer, Material, VertexBuffer, util::read_accessor};
use crate::{math::*, util::*, vulkan::*};
use ash::vk;
use cgmath::Vector3;
use gltf::{
    accessor::DataType, buffer::Data, mesh::Primitive as GltfPrimitive, mesh::Semantic, Accessor,
    Document,
};
use serde_json::Value;
use std::rc::Rc;

pub struct Mesh {
    primitives: Vec<Primitive>,
    aabb: AABB<f32>,
}

impl Mesh {
    fn new(primitives: Vec<Primitive>) -> Self {
        let aabbs = primitives.iter().map(|p| p.aabb()).collect::<Vec<_>>();
        let aabb = AABB::union(&aabbs).unwrap();
        Mesh { primitives, aabb }
    }
}

impl Mesh {
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    pub fn aabb(&self) -> AABB<f32> {
        self.aabb
    }
}

pub struct Primitive {
    vertices: VertexBuffer,
    indices: Option<IndexBuffer>,
    material: Material,
    aabb: AABB<f32>,
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

    pub fn aabb(&self) -> AABB<f32> {
        self.aabb
    }
}

/// Vertex buffer byte offset / element count
type VertexBufferPart = (usize, usize);

/// Index buffer byte offset / element count / type
type IndexBufferPart = (usize, usize, vk::IndexType);

struct PrimitiveData {
    indices: Option<IndexBufferPart>,
    vertices: VertexBufferPart,
    material: Material,
    aabb: AABB<f32>,
}

pub fn create_meshes_from_gltf(
    context: &Rc<Context>,
    document: &Document,
    buffers: &[Data],
) -> Vec<Mesh> {
    // (usize, usize) -> byte offset, element count
    let mut meshes_data = Vec::<Vec<PrimitiveData>>::new();
    let mut all_vertices = Vec::<u8>::new();
    let mut all_indices = Vec::<u8>::new();

    // Gather vertices and indices from all the meshes in the document
    for mesh in document.meshes() {
        let mut primitives_buffers = Vec::<PrimitiveData>::new();

        for primitive in mesh.primitives() {
            let indices = primitive.indices().map(|accessor| {
                let (indices, index_type) = extract_indices_from_accessor(&accessor, buffers);
                let offset = all_indices.len();
                all_indices.extend_from_slice(&indices);
                (offset, accessor.count(), index_type)
            });

            if let Some(accessor) = primitive.get(&Semantic::Positions) {
                let (positions, aabb) = read_positions(&accessor, buffers);
                let normals = read_normals(&primitive, buffers);
                let texcoords = read_texcoords(&primitive, buffers);
                let tangents = read_tangents(&primitive, buffers);

                let mut vertices = Vec::<u8>::new();

                for elt_index in 0..accessor.count() {
                    push_vec3(&Some(&positions), elt_index, &mut vertices);
                    push_vec3(&normals.as_ref().map(|v| &v[..]), elt_index, &mut vertices);
                    push_vec2(
                        &texcoords.as_ref().map(|c| &c[..]),
                        elt_index,
                        &mut vertices,
                    );
                    // TODO : if tangents are not provided they should be computed using default MikkTSpace algorithms.
                    push_vec4(&tangents.as_ref().map(|t| &t[..]), elt_index, &mut vertices);
                }

                let offset = all_vertices.len();
                all_vertices.extend_from_slice(&vertices);

                let material = Material::from(primitive.material());

                primitives_buffers.push(PrimitiveData {
                    indices,
                    vertices: (offset, accessor.count()),
                    material,
                    aabb,
                });
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
                        let mesh_vertices = buffers.vertices;
                        let vertex_buffer = VertexBuffer::new(
                            Rc::clone(&vertices),
                            mesh_vertices.0 as _,
                            mesh_vertices.1 as _,
                        );

                        let index_buffer = buffers.indices.map(|mesh_indices| {
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
                            material: buffers.material,
                            aabb: buffers.aabb,
                        }
                    })
                    .collect::<Vec<_>>();
                Mesh::new(primitives)
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

fn read_positions(accessor: &Accessor, buffers: &[Data]) -> (Vec<u8>, AABB<f32>) {
    let min = parse_vec3_json_value(&accessor.min().unwrap());
    let max = parse_vec3_json_value(&accessor.max().unwrap());
    let data = read_accessor(accessor, buffers);
    (data, AABB::new(min, max))
}

fn parse_vec3_json_value(value: &Value) -> Vector3<f32> {
    let as_array = value.as_array().unwrap();
    let x = as_array[0].as_f64().unwrap() as _;
    let y = as_array[1].as_f64().unwrap() as _;
    let z = as_array[2].as_f64().unwrap() as _;
    Vector3::new(x, y, z)
}

fn read_normals(primitive: &GltfPrimitive, buffers: &[Data]) -> Option<Vec<u8>> {
    primitive
        .get(&Semantic::Normals)
        .map(|normals| read_accessor(&normals, buffers))
}

fn read_texcoords(primitive: &GltfPrimitive, buffers: &[Data]) -> Option<Vec<u8>> {
    primitive
        .get(&Semantic::TexCoords(0))
        .filter(|texcoords| texcoords.data_type() == DataType::F32)
        .map(|texcoords| read_accessor(&texcoords, buffers))
}

fn read_tangents(primitive: &GltfPrimitive, buffers: &[Data]) -> Option<Vec<u8>> {
    primitive
        .get(&Semantic::Tangents)
        .map(|tangents| read_accessor(&tangents, buffers))
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

fn push_vec2(src: &Option<&[u8]>, index: usize, dest: &mut Vec<u8>) {
    let left = index * 8;
    let right = left + 8;

    if let Some(src) = src {
        dest.extend_from_slice(&src[left..right]);
    } else {
        unsafe {
            let one: [f32; 2] = [0.0, 0.0];
            dest.extend_from_slice(any_as_u8_slice(&one));
        }
    };
}

fn push_vec4(src: &Option<&[u8]>, index: usize, dest: &mut Vec<u8>) {
    let left = index * 16;
    let right = left + 16;

    if let Some(src) = src {
        dest.extend_from_slice(&src[left..right]);
    } else {
        unsafe {
            let one: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
            dest.extend_from_slice(any_as_u8_slice(&one));
        }
    };
}
