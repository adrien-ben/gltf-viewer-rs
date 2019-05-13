use gltf::{
    material::{Material as GltfMaterial, NormalTexture, OcclusionTexture},
    texture::Info,
};

pub const MAX_TEXTURE_COUNT: u32 = 64; // MUST be the same in the fragment shager

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Material {
    color_and_metallic: [f32; 4],
    emissive_and_roughness: [f32; 4],
    occlusion: f32,
    color_texture_id: i32,
    metallic_roughness_texture_id: i32,
    emissive_texture_id: i32,
    normal_texture_id: i32,
    occlusion_texture_id: i32,
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let pbr = material.pbr_metallic_roughness();

        let base_color_factor = pbr.base_color_factor();
        let color_and_metallic = [
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
            pbr.metallic_factor(),
        ];

        let emissive_factor = material.emissive_factor();
        let emissive_and_roughness = [
            emissive_factor[0],
            emissive_factor[1],
            emissive_factor[2],
            pbr.roughness_factor(),
        ];

        let color_texture_id = get_texture_index(pbr.base_color_texture());
        let metallic_roughness_texture_id = get_texture_index(pbr.metallic_roughness_texture());
        let emissive_texture_id = get_texture_index(material.emissive_texture());
        let normal_texture_id = get_normal_texture_index(material.normal_texture());
        let (occlusion, occlusion_texture_id) = get_occlusion(material.occlusion_texture());

        Material {
            color_and_metallic,
            emissive_and_roughness,
            occlusion,
            color_texture_id,
            metallic_roughness_texture_id,
            emissive_texture_id,
            normal_texture_id,
            occlusion_texture_id,
        }
    }
}

fn get_texture_index(texture_info: Option<Info>) -> i32 {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
        .filter(|index| *index < MAX_TEXTURE_COUNT as _)
        .map_or(-1, |index| index as _)
}

fn get_normal_texture_index(texture_info: Option<NormalTexture>) -> i32 {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
        .filter(|index| *index < MAX_TEXTURE_COUNT as _)
        .map_or(-1, |index| index as _)
}

fn get_occlusion(texture_info: Option<OcclusionTexture>) -> (f32, i32) {
    (
        texture_info
            .as_ref()
            .map_or(0.0, |tex_info| tex_info.strength()),
        texture_info
            .map(|tex_info| tex_info.texture())
            .map(|texture| texture.index())
            .filter(|index| *index < MAX_TEXTURE_COUNT as _)
            .map_or(-1, |index| index as _),
    )
}
