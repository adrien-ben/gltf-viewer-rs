mod animation;
mod error;
mod material;
mod mesh;
mod node;
mod skin;
mod texture;
mod vertex;

pub use self::{
    animation::*, error::*, material::*, mesh::*, node::*, skin::*, texture::*, vertex::*,
};
use crate::{math::*, vulkan::*};
use cgmath::Matrix4;
use std::{error::Error, path::Path, rc::Rc, result::Result};

pub struct Model {
    meshes: Vec<Mesh>,
    nodes: Nodes,
    global_transform: Matrix4<f32>,
    animations: Vec<Animation>,
    skins: Vec<Skin>,
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

        let animations = load_animations(document.animations(), &buffers);

        let mut skins = create_skins_from_gltf(document.skins(), &buffers);

        let mut nodes = Nodes::from_gltf_nodes(document.nodes(), &scene);

        let global_transform = {
            let aabb = compute_aabb(&nodes, &meshes);
            let transform = compute_unit_cube_at_origin_transform(aabb);
            nodes.transform(Some(transform));
            nodes
                .get_skins_transform()
                .iter()
                .for_each(|(index, transform)| {
                    let skin = &mut skins[*index];
                    skin.compute_joints_matrices(*transform, &nodes.nodes());
                });
            transform
        };

        let textures = texture::create_textures_from_gltf(context, &images);

        Ok(Model {
            meshes,
            nodes,
            global_transform,
            animations,
            skins,
            textures,
        })
    }
}

impl Model {
    pub fn update(&mut self, delta_time: f32) -> bool {
        let updated = if let Some(animation) = &mut self.animations.get_mut(0) {
            animation.animate(&mut self.nodes, delta_time)
        } else {
            false
        };

        if updated {
            self.nodes.transform(Some(self.global_transform));
            self.nodes
                .get_skins_transform()
                .iter()
                .for_each(|(index, transform)| {
                    let skin = &mut self.skins[*index];
                    skin.compute_joints_matrices(*transform, &self.nodes.nodes());
                });
        }

        updated
    }
}

impl Model {
    pub fn mesh(&self, index: usize) -> &Mesh {
        &self.meshes[index]
    }

    pub fn skin(&self, index: usize) -> &Skin {
        &self.skins[index]
    }

    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    pub fn textures(&self) -> &[Texture] {
        &self.textures
    }
}

fn compute_aabb(nodes: &Nodes, meshes: &[Mesh]) -> AABB<f32> {
    let aabbs = nodes
        .nodes()
        .iter()
        .filter(|n| n.mesh_index().is_some())
        .map(|n| {
            let mesh = &meshes[n.mesh_index().unwrap()];
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
