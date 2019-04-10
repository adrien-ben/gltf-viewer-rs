mod error;
mod material;
mod mesh;
mod node;
mod texture;
mod vertex;

pub use self::{error::*, material::*, mesh::*, node::*, texture::*, vertex::*};
use crate::vulkan::*;
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
