mod animation;
mod error;
mod light;
mod material;
mod mesh;
pub mod metadata;
mod mikktspace;
mod node;
mod skin;
mod texture;
mod vertex;

use self::mikktspace::generate_tangents;
pub use self::{
    animation::*, error::*, light::*, material::*, mesh::*, node::*, skin::*, texture::*, vertex::*,
};
use cgmath::Matrix4;
use math::*;
use metadata::Metadata;
use std::{error::Error, path::Path, result::Result, sync::Arc};
use vulkan::ash::vk;
use vulkan::{Buffer, Context, PreLoadedResource};

pub struct ModelStagingResources {
    _staged_vertices: Buffer,
    _staged_indices: Option<Buffer>,
    _staged_textures: Vec<Buffer>,
}

pub struct Model {
    metadata: Metadata,
    meshes: Vec<Mesh>,
    nodes: Nodes,
    global_transform: Matrix4<f32>,
    animations: Option<Animations>,
    skins: Vec<Skin>,
    textures: Textures,
    lights: Vec<Light>,
}

impl Model {
    pub fn create_from_file<P: AsRef<Path>>(
        context: Arc<Context>,
        command_buffer: vk::CommandBuffer,
        path: P,
    ) -> Result<PreLoadedResource<Model, ModelStagingResources>, Box<dyn Error>> {
        log::debug!("Importing gltf file");
        let (document, buffers, images) = gltf::import(&path)?;

        let metadata = Metadata::new(path, &document);

        log::debug!("Creating the model");
        if document.scenes().len() == 0 {
            return Err(Box::new(ModelLoadingError::new("There is no scene")));
        }

        let meshes = create_meshes_from_gltf(&context, command_buffer, &document, &buffers);
        if meshes.is_none() {
            return Err(Box::new(ModelLoadingError::new(
                "Could not find any renderable primitives",
            )));
        }

        let Meshes {
            meshes,
            vertices: staged_vertices,
            indices: staged_indices,
        } = meshes.unwrap();

        let scene = document
            .default_scene()
            .unwrap_or_else(|| document.scenes().next().unwrap());

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

        let (textures, staged_textures) = texture::create_textures_from_gltf(
            &context,
            command_buffer,
            document.textures(),
            &images,
        );

        let lights = create_lights_from_gltf(&document);

        let model = Model {
            metadata,
            meshes,
            nodes,
            global_transform,
            animations,
            skins,
            textures,
            lights,
        };

        let model_staging_res = ModelStagingResources {
            _staged_vertices: staged_vertices,
            _staged_indices: staged_indices,
            _staged_textures: staged_textures,
        };

        Ok(PreLoadedResource::new(
            context,
            command_buffer,
            model,
            model_staging_res,
        ))
    }
}

impl Model {
    pub fn update(&mut self, delta_time: f32) -> bool {
        let updated = if let Some(animations) = self.animations.as_mut() {
            animations.update(&mut self.nodes, delta_time)
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

/// Animations methods
impl Model {
    pub fn get_animation_playback_state(&self) -> Option<PlaybackState> {
        self.animations
            .as_ref()
            .map(Animations::get_playback_state)
            .copied()
    }

    pub fn set_current_animation(&mut self, animation_index: usize) {
        if let Some(animations) = self.animations.as_mut() {
            animations.set_current(animation_index);
        }
    }

    pub fn set_animation_playback_mode(&mut self, playback_mode: PlaybackMode) {
        if let Some(animations) = self.animations.as_mut() {
            animations.set_playback_mode(playback_mode);
        }
    }

    pub fn toggle_animation(&mut self) {
        if let Some(animations) = self.animations.as_mut() {
            animations.toggle();
        }
    }

    pub fn stop_animation(&mut self) {
        if let Some(animations) = self.animations.as_mut() {
            animations.stop();
        }
    }

    pub fn reset_animation(&mut self) {
        if let Some(animations) = self.animations.as_mut() {
            animations.reset();
        }
    }
}

/// Getters
impl Model {
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn mesh(&self, index: usize) -> &Mesh {
        &self.meshes[index]
    }

    pub fn primitive_count(&self) -> usize {
        self.meshes.iter().map(Mesh::primitive_count).sum()
    }

    pub fn skins(&self) -> &[Skin] {
        &self.skins
    }

    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    pub fn textures(&self) -> &[Texture] {
        &self.textures.textures
    }

    pub fn lights(&self) -> &[Light] {
        &self.lights
    }
}

fn compute_aabb(nodes: &Nodes, meshes: &[Mesh]) -> Aabb<f32> {
    let aabbs = nodes
        .nodes()
        .iter()
        .filter(|n| n.mesh_index().is_some())
        .map(|n| {
            let mesh = &meshes[n.mesh_index().unwrap()];
            mesh.aabb() * n.transform()
        })
        .collect::<Vec<_>>();
    Aabb::union(&aabbs).unwrap()
}

fn compute_unit_cube_at_origin_transform(aabb: Aabb<f32>) -> Matrix4<f32> {
    let larger_side = aabb.get_larger_side_size();
    let scale_factor = (1.0_f32 / larger_side) * 10.0;

    let aabb = aabb * scale_factor;
    let center = aabb.get_center();

    let translation = Matrix4::from_translation(-center);
    let scale = Matrix4::from_scale(scale_factor);
    translation * scale
}
