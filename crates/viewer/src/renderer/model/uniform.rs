use super::JointsBuffer;
use math::cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector4};
use model::{Light, Material, Model, Type, Workflow, MAX_JOINTS_PER_MESH};
use std::{mem::size_of, sync::Arc};
use vulkan::{ash::vk, Buffer, Context};

const DEFAULT_LIGHT_DIRECTION: [f32; 4] = [0.0, 0.0, -1.0, 0.0];
const DIRECTIONAL_LIGHT_TYPE: u32 = 0;
const POINT_LIGHT_TYPE: u32 = 1;
const SPOT_LIGHT_TYPE: u32 = 2;
const NO_TEXTURE_ID: u32 = 3; // MAX u2
const UNLIT_FLAG_LIT: u32 = 0;
const UNLIT_FLAG_UNLIT: u32 = 1;
const METALLIC_ROUGHNESS_WORKFLOW: u32 = 0;
const SPECULAR_GLOSSINESS_WORKFLOW: u32 = 1;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
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
#[derive(Clone, Copy)]
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
    // Contains the texture channels. Each channel taking 2 bits
    // [0-1] Color texture channel
    // [2-3] metallic/roughness or specular/glossiness texture channel
    // [4-5] emissive texture channel
    // [6-7] normals texture channel
    // [8-9] Occlusion texture channel
    // [10-11] Clearcoat factor texture channel
    // [12-13] Clearcoat roughness texture channel
    // [14-15] Clearcoat normal texture channel
    // [16-31] Reserved
    textures_channels: u32,
    // Contains alpha mode, unlit flag and workflow flag
    // [0-7] Alpha mode
    // [8-15] Unlit flag
    // [16-23] Workflow (metallic/roughness or specular/glossiness)
    // [24-31] Reserved
    alpha_mode_unlit_flag_and_workflow: u32,
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

        let textures_channels = {
            let color = material
                .get_color_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());
            let metallic_roughness = match material.get_workflow() {
                Workflow::MetallicRoughness(workflow) => workflow.get_metallic_roughness_texture(),
                Workflow::SpecularGlossiness(workflow) => {
                    workflow.get_specular_glossiness_texture()
                }
            }
            .map_or(NO_TEXTURE_ID, |t| t.get_channel());
            let emissive = material
                .get_emissive_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());
            let normal = material
                .get_normals_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());
            let occlusion = material
                .get_occlusion_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());
            let clearcoat_factor = clearcoat
                .factor_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());

            let clearcoat_roughness = clearcoat
                .roughness_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());

            let clearcoat_normal = clearcoat
                .normal_texture()
                .map_or(NO_TEXTURE_ID, |info| info.get_channel());

            (color << 30)
                | (metallic_roughness << 28)
                | (emissive << 26)
                | (normal << 24)
                | (occlusion << 22)
                | (clearcoat_factor << 20)
                | (clearcoat_roughness << 18)
                | (clearcoat_normal << 16)
        };

        let alpha_mode_unlit_flag_and_workflow = {
            let alpha_mode = material.get_alpha_mode();
            let unlit_flag = if material.is_unlit() {
                UNLIT_FLAG_UNLIT
            } else {
                UNLIT_FLAG_LIT
            };
            let workflow = if let Workflow::MetallicRoughness { .. } = workflow {
                METALLIC_ROUGHNESS_WORKFLOW
            } else {
                SPECULAR_GLOSSINESS_WORKFLOW
            };

            (alpha_mode << 24) | (unlit_flag << 16) | (workflow << 8)
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
            textures_channels,
            alpha_mode_unlit_flag_and_workflow,
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

pub fn create_lights_ubos(context: &Arc<Context>, model: &Model, count: u32) -> Vec<Buffer> {
    let light_count = model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.light_index().is_some())
        .count();

    // Buffer size cannot be 0 so we allocate at least anough space for one light
    // Probably a bad idea but I'd rather avoid creating a specific shader
    let buffer_size = std::cmp::max(1, light_count) * size_of::<LightUniform>();

    (0..count)
        .map(|_| {
            Buffer::create(
                Arc::clone(context),
                buffer_size as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .collect::<Vec<_>>()
}
