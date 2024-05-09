use super::JointsBuffer;
use math::cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector4};
use model::{Light, Material, Model, Type, Workflow, MAX_JOINTS_PER_MESH};
use std::{mem::size_of, sync::Arc};
use vulkan::{ash::vk, Buffer, Context};

pub const MAX_LIGHT_COUNT: usize = 8;
const DEFAULT_LIGHT_DIRECTION: [f32; 4] = [0.0, 0.0, -1.0, 0.0];
const DIRECTIONAL_LIGHT_TYPE: u32 = 0;
const POINT_LIGHT_TYPE: u32 = 1;
const SPOT_LIGHT_TYPE: u32 = 2;
const NO_TEXTURE_ID: u32 = u8::MAX as _;
const METALLIC_ROUGHNESS_WORKFLOW: u32 = 0;
const SPECULAR_GLOSSINESS_WORKFLOW: u32 = 1;

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct LightsUBO {
    pub count: u32,
    pub lights: [LightUniform; MAX_LIGHT_COUNT],
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, align(16))]
pub struct LightUniform {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    intensity: f32,
    range: f32,
    angle_scale: f32,
    angle_offset: f32,
    light_type: u32,
    pad: [u32; 3],
}

impl From<(Matrix4<f32>, Light)> for LightUniform {
    fn from((transform, light): (Matrix4<f32>, Light)) -> Self {
        let position = [transform.w.x, transform.w.y, transform.w.z, 0.0];

        let direction = (transform * Vector4::from(DEFAULT_LIGHT_DIRECTION))
            .normalize()
            .into();

        let color = light.color();
        let color = [color[0], color[1], color[2], 0.0];

        let intensity = light.intensity();

        let range = light.range().unwrap_or(-1.0);

        let (angle_scale, angle_offset) = match light.light_type() {
            Type::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => {
                let outer_cos = outer_cone_angle.cos();
                let angle_scale = 1.0 / math::max(0.001, inner_cone_angle.cos() - outer_cos);
                let angle_offset = -outer_cos * angle_scale;
                (angle_scale, angle_offset)
            }
            _ => (-1.0, -1.0),
        };

        let light_type = match light.light_type() {
            Type::Directional => DIRECTIONAL_LIGHT_TYPE,
            Type::Point => POINT_LIGHT_TYPE,
            Type::Spot { .. } => SPOT_LIGHT_TYPE,
        };

        Self {
            position,
            direction,
            color,
            intensity,
            range,
            angle_scale,
            angle_offset,
            light_type,
            pad: [0, 0, 0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MaterialUniform {
    color: [f32; 4],
    emissive_factor: [f32; 3],
    // - roughness for metallic/roughness workflows
    // - glossiness for specular/glossiness workflows
    roughness_glossiness: f32,
    // Contains the metallic (or specular) factor.
    // - metallic: metallic_specular_and_occlusion[0] (for metallic/roughness workflows)
    // - specular: metallic_specular_and_occlusion[0,1,2] (for specular/glossiness workflows)
    metallic_specular: [f32; 3],
    occlusion: f32,
    alpha_cutoff: f32,
    clearcoat_factor: f32,
    clearcoat_roughness: f32,
    color_texture_channel: u32,
    material_texture_channel: u32,
    emissive_texture_channel: u32,
    normals_texture_channel: u32,
    occlusion_texture_channel: u32,
    clearcoat_factor_texture_channel: u32,
    clearcoat_roughness_texture_channel: u32,
    clearcoat_normal_texture_channel: u32,
    alpha_mode: u32,
    is_unlit: vk::Bool32,
    workflow: u32,
}

impl From<Material> for MaterialUniform {
    fn from(material: Material) -> MaterialUniform {
        let color = material.get_color();
        let emissive_factor = material.get_emissive();

        let workflow = material.get_workflow();

        let roughness_glossiness = match workflow {
            Workflow::MetallicRoughness(workflow) => workflow.get_roughness(),
            Workflow::SpecularGlossiness(workflow) => workflow.get_glossiness(),
        };

        let metallic_specular = match workflow {
            Workflow::MetallicRoughness(workflow) => [workflow.get_metallic(), 0.0, 0.0],
            Workflow::SpecularGlossiness(workflow) => workflow.get_specular(),
        };

        let occlusion = material.get_occlusion();

        let alpha_cutoff = material.get_alpha_cutoff();
        let clearcoat = material.get_clearcoat().unwrap_or_default();
        let clearcoat_factor = clearcoat.factor();
        let clearcoat_roughness = clearcoat.roughness();

        let color_texture_channel = material
            .get_color_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let material_texture_channel = match material.get_workflow() {
            Workflow::MetallicRoughness(workflow) => workflow.get_metallic_roughness_texture(),
            Workflow::SpecularGlossiness(workflow) => workflow.get_specular_glossiness_texture(),
        }
        .map_or(NO_TEXTURE_ID, |t| t.get_channel());
        let emissive_texture_channel = material
            .get_emissive_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let normals_texture_channel = material
            .get_normals_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let occlusion_texture_channel = material
            .get_occlusion_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let clearcoat_factor_texture_channel = clearcoat
            .factor_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());

        let clearcoat_roughness_texture_channel = clearcoat
            .roughness_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());

        let clearcoat_normal_texture_channel = clearcoat
            .normal_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());

        let alpha_mode = material.get_alpha_mode();
        let is_unlit = material.is_unlit().into();
        let workflow = if let Workflow::MetallicRoughness { .. } = workflow {
            METALLIC_ROUGHNESS_WORKFLOW
        } else {
            SPECULAR_GLOSSINESS_WORKFLOW
        };

        MaterialUniform {
            color,
            emissive_factor,
            roughness_glossiness,
            metallic_specular,
            occlusion,
            alpha_cutoff,
            clearcoat_factor,
            clearcoat_roughness,
            color_texture_channel,
            material_texture_channel,
            emissive_texture_channel,
            normals_texture_channel,
            occlusion_texture_channel,
            clearcoat_factor_texture_channel,
            clearcoat_roughness_texture_channel,
            clearcoat_normal_texture_channel,
            alpha_mode,
            is_unlit,
            workflow,
        }
    }
}

pub fn create_transform_ubos(context: &Arc<Context>, model: &Model, count: u32) -> Vec<Buffer> {
    let mesh_node_count = model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.mesh_index().is_some())
        .count() as u32;
    let elem_size = context.get_ubo_alignment::<Matrix4<f32>>();

    (0..count)
        .map(|_| {
            let mut buffer = Buffer::create(
                Arc::clone(context),
                u64::from(elem_size * mesh_node_count),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffer.map_memory();
            buffer
        })
        .collect::<Vec<_>>()
}

pub fn create_skin_ubos(
    context: &Arc<Context>,
    model: &Model,
    count: u32,
) -> (Vec<Buffer>, Vec<Vec<JointsBuffer>>) {
    let skin_node_count = model.skins().len().max(1);
    let elem_size = context.get_ubo_alignment::<JointsBuffer>();

    let buffers = (0..count)
        .map(|_| {
            let mut buffer = Buffer::create(
                Arc::clone(context),
                u64::from(elem_size * skin_node_count as u32),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffer.map_memory();
            buffer
        })
        .collect();

    let matrices = (0..count)
        .map(|_| {
            let mut matrices = Vec::with_capacity(skin_node_count);
            for _ in 0..skin_node_count {
                matrices.push([Matrix4::<f32>::identity(); MAX_JOINTS_PER_MESH]);
            }
            matrices
        })
        .collect();

    (buffers, matrices)
}

/// create a ubo containing model's materials
/// first is a default material (used for primitives that do no reference a material)
/// then the materials actually defined by the model
pub fn create_materials_ubo(context: &Arc<Context>, model: &Model) -> Buffer {
    let material_count = 1 + model.materials().len() as vk::DeviceSize;
    let elem_size = context.get_ubo_alignment::<MaterialUniform>() as vk::DeviceSize;
    let size = elem_size * material_count;
    Buffer::create(
        Arc::clone(context),
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

pub fn create_lights_ubos(context: &Arc<Context>, count: u32) -> Vec<Buffer> {
    let size = size_of::<LightsUBO>();
    (0..count)
        .map(|_| {
            Buffer::create(
                Arc::clone(context),
                size as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .collect::<Vec<_>>()
}
