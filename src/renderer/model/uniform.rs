use super::JointsBuffer;
use math::cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector4};
use model::{Light, Material, Model, Type, MAX_JOINTS_PER_MESH};
use std::{mem::size_of, sync::Arc};
use vulkan::{ash::vk, Buffer, Context};

const DEFAULT_LIGHT_DIRECTION: [f32; 4] = [0.0, 0.0, -1.0, 0.0];
const DIRECTIONAL_LIGHT_TYPE: u32 = 0;
const POINT_LIGHT_TYPE: u32 = 1;
const SPOT_LIGHT_TYPE: u32 = 2;
const NO_TEXTURE_ID: u32 = std::u8::MAX as u32;

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

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct MaterialUniform {
    color: [f32; 4],
    emissive_and_roughness: [f32; 4],
    metallic: f32,
    occlusion: f32,
    // Contains the texture channels for color metallic/roughness emissive and normal (each taking 8 bytes)
    color_metallicroughness_emissive_normal_texture_channels: u32,
    occlusion_texture_channel_and_alpha_mode: u32,
    alpha_cutoff: f32,
}

impl<'a> From<Material> for MaterialUniform {
    fn from(material: Material) -> MaterialUniform {
        let color = material.get_color();
        let emissive_factor = material.get_emissive();

        let emissive_and_roughness = [
            emissive_factor[0],
            emissive_factor[1],
            emissive_factor[2],
            material.get_roughness(),
        ];

        let metallic = material.get_metallic();

        let color_texture_id = material
            .get_color_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let metallic_roughness_texture_id = material
            .get_metallic_roughness_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let emissive_texture_id = material
            .get_emissive_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let normal_texture_id = material
            .get_normals_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let color_metallicroughness_emissive_normal_texture_channels = (color_texture_id << 24)
            | (metallic_roughness_texture_id << 16)
            | (emissive_texture_id << 8)
            | normal_texture_id;

        let occlusion = material.get_occlusion();
        let occlusion_texture_id = material
            .get_occlusion_texture()
            .map_or(NO_TEXTURE_ID, |info| info.get_channel());
        let alpha_mode = material.get_alpha_mode();
        let occlusion_texture_channel_and_alpha_mode =
            ((occlusion_texture_id as u32) << 24) | (alpha_mode << 16);

        let alpha_cutoff = material.get_alpha_cutoff();

        MaterialUniform {
            color,
            emissive_and_roughness,
            metallic,
            occlusion,
            color_metallicroughness_emissive_normal_texture_channels,
            occlusion_texture_channel_and_alpha_mode,
            alpha_cutoff,
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