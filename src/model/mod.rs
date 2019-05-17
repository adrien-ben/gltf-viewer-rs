mod error;
mod material;
mod mesh;
mod node;
mod texture;
mod vertex;

pub use self::{error::*, material::*, mesh::*, node::*, texture::*, vertex::*};
use crate::{math::*, vulkan::*};
use cgmath::Matrix4;
use std::{error::Error, path::Path, rc::Rc, result::Result};

pub struct Model {
    meshes: Vec<Mesh>,
    nodes: Vec<Node>,
    textures: Vec<Texture>,
}

impl Model {
    pub fn create_from_file<P: AsRef<Path>>(
        context: &Rc<Context>,
        path: P,
    ) -> Result<Self, Box<dyn Error>> {
        log::debug!("Importing gltf file");
        let (document, buffers, images) = gltf::import(path)?;

        log::debug!("Creating the model");
        if document.scenes().len() == 0 {
            return Err(Box::new(ModelLoadingError::new("There is no scene")));
        }

        let meshes = create_meshes_from_gltf(context, &document, &buffers);
        if meshes.is_empty() {
            return Err(Box::new(ModelLoadingError::new(
                "Could not find any renderable primitives",
            )));
        }

        let scene = document
            .default_scene()
            .unwrap_or_else(|| document.scenes().nth(0).unwrap());
        let nodes = traverse_scene_nodes(&scene);
        let textures = texture::create_textures_from_gltf(context, &images);

        let aabb = compute_aabb(&nodes, &meshes);
        let transform = compute_unit_cube_at_origin_transform(aabb);

        let nodes = nodes.iter().map(|n| *n * transform).collect::<Vec<_>>();

        Ok(Model {
            meshes,
            nodes,
            textures,
        })
    }
}

impl Model {
    pub fn mesh(&self, index: usize) -> &Mesh {
        &self.meshes[index]
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn textures(&self) -> &[Texture] {
        &self.textures
    }
}

fn compute_aabb(nodes: &[Node], meshes: &[Mesh]) -> AABB<f32> {
    let aabbs = nodes
        .iter()
        .map(|n| {
            let mesh = &meshes[n.mesh_index()];
            mesh.aabb() * n.transform()
        })
        .collect::<Vec<_>>();
    AABB::union(&aabbs).unwrap()
}

fn compute_unit_cube_at_origin_transform(aabb: AABB<f32>) -> Matrix4<f32> {
    let larger_side = aabb.get_larger_side_size();
    let scale_factor = 1.0_f32 / larger_side;

    let aabb = aabb * scale_factor;
    let center = aabb.get_center();

    let translation = Matrix4::from_translation(-center);
    let scale = Matrix4::from_scale(1.0_f32 / larger_side);
    translation * scale
}
