mod error;
mod mesh;
mod vertex;

pub use self::{error::*, mesh::*, vertex::*};
use crate::vulkan::*;
use std::{error::Error, path::Path, rc::Rc, result::Result};

pub struct Model {
    meshes: Vec<Mesh>,
}

impl Model {
    pub fn create_from_file<P: AsRef<Path>>(
        context: &Rc<Context>,
        path: P,
    ) -> Result<Self, Box<dyn Error>> {
        let (document, buffers, _) = gltf::import(path)?;

        let meshes = create_mesh_from_gltf(context, &document, &buffers);

        if meshes.is_empty() {
            return Err(Box::new(ModelLoadingError::new(
                "Could not find any renderable primitives",
            )));
        }

        Ok(Model { meshes })
    }
}

impl Model {
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }
}
