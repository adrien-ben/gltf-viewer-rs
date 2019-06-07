use super::{IndexBuffer, Material, ModelVertex, VertexBuffer};
use crate::{math::*, vulkan::*};
use ash::vk;
use cgmath::Vector3;
use gltf::{
    buffer::{Buffer as GltfBuffer, Data},
    mesh::{Bounds, Reader, Semantic},
    Document,
};
use std::{mem::size_of, rc::Rc};

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

/// Index buffer byte offset / element count
type IndexBufferPart = (usize, usize);

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
    let mut meshes_data = Vec::<Vec<PrimitiveData>>::new();
    let mut all_vertices = Vec::<ModelVertex>::new();
    let mut all_indices = Vec::<u32>::new();

    // Gather vertices and indices from all the meshes in the document
    for mesh in document.meshes() {
        let mut primitives_buffers = Vec::<PrimitiveData>::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let indices = read_indices(&reader).map(|indices| {
                let offset = all_indices.len() * size_of::<u32>();
                all_indices.extend_from_slice(&indices);
                (offset, indices.len())
            });

            if let Some(accessor) = primitive.get(&Semantic::Positions) {
                let aabb = get_aabb(&primitive.bounding_box());
                let positions = read_positions(&reader);
                let normals = read_normals(&reader);
                let tex_coords = read_tex_coords(&reader);
                let tangents = read_tangents(&reader);
                let weights = read_weights(&reader);
                let joints = read_joints(&reader);

                let vertices = positions
                    .iter()
                    .enumerate()
                    .map(|(index, position)| {
                        let position = *position;
                        let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                        let tex_coords = *tex_coords.get(index).unwrap_or(&[0.0, 0.0]);
                        let tangent = *tangents.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
                        let weights = *weights.get(index).unwrap_or(&[0.0, 0.0, 0.0, 0.0]);
                        let joints = *joints.get(index).unwrap_or(&[0, 0, 0, 0]);

                        ModelVertex {
                            position,
                            normal,
                            tex_coords,
                            tangent,
                            weights,
                            joints,
                        }
                    })
                    .collect::<Vec<_>>();

                let offset = all_vertices.len() * size_of::<ModelVertex>();
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
            .collect();
    }

    Vec::new()
}

fn read_indices<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Option<Vec<u32>>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_indices()
        .map(|indices| indices.into_u32().collect::<Vec<_>>())
}

fn get_aabb(bounds: &Bounds<[f32; 3]>) -> AABB<f32> {
    let min = bounds.min;
    let min = Vector3::new(min[0], min[1], min[2]);

    let max = bounds.max;
    let max = Vector3::new(max[0], max[1], max[2]);

    AABB::new(min, max)
}

fn read_positions<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 3]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_positions()
        .expect("Position primitives should be present")
        .collect()
}

fn read_normals<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 3]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_normals()
        .map_or(vec![], |normals| normals.collect())
}

fn read_tex_coords<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 2]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_tex_coords(0)
        .map_or(vec![], |coords| coords.into_f32().collect())
}

fn read_tangents<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 4]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_tangents()
        .map_or(vec![], |tangents| tangents.collect())
}

fn read_weights<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[f32; 4]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_weights(0)
        .map_or(vec![], |weights| weights.into_f32().collect())
}

fn read_joints<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<[u32; 4]>
where
    F: Clone + Fn(GltfBuffer<'a>) -> Option<&'s [u8]>,
{
    reader.read_joints(0).map_or(vec![], |joints| {
        joints
            .into_u16()
            .map(|[x, y, z, w]| [u32::from(x), u32::from(y), u32::from(z), u32::from(w)])
            .collect()
    })
}
