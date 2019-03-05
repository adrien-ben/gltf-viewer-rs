mod error;
mod mesh;
mod node;
mod vertex;

pub use self::{error::*, mesh::*, node::*, vertex::*};
use crate::vulkan::*;

use std::{error::Error, path::Path, rc::Rc, result::Result};

pub struct Model {
    meshes: Vec<Mesh>,
    nodes: Vec<Node>,
}

impl Model {
    pub fn create_from_file<P: AsRef<Path>>(
        context: &Rc<Context>,
        path: P,
    ) -> Result<Self, Box<dyn Error>> {
        let (document, buffers, _) = gltf::import(path)?;

        if document.scenes().len() == 0 {
            return Err(Box::new(ModelLoadingError::new("There is no scene")));
        }

        let meshes = create_mesh_from_gltf(context, &document, &buffers);

        if meshes.is_empty() {
            return Err(Box::new(ModelLoadingError::new(
                "Could not find any renderable primitives",
            )));
        }

        let scene = document
            .default_scene()
            .unwrap_or_else(|| document.scenes().nth(0).unwrap());
        let nodes = traverse_scene_nodes(&scene);

        Ok(Model { meshes, nodes })
    }
}

impl Model {
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}
