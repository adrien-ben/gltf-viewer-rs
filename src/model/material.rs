use gltf::{
    material::{AlphaMode, Material as GltfMaterial, NormalTexture, OcclusionTexture},
    texture::Info,
};

pub const MAX_TEXTURE_COUNT: u32 = 64; // MUST be the same in the fragment shager
const NO_TEXTURE_ID: u8 = std::u8::MAX;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Material {
    color: [f32; 4],
    emissive_and_roughness: [f32; 4],
    metallic: f32,
    occlusion: f32,
    // Contains the texture ids for color metallic/roughness emissive and normal (each taking 8 bytes)
    color_metallicroughness_emissive_normal_texture_ids: u32,
    occlusion_texture_id_and_alpha_mode: u32,
    alpha_cutoff: f32,
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let pbr = material.pbr_metallic_roughness();

        let color = pbr.base_color_factor();

        let emissive_factor = material.emissive_factor();
        let emissive_and_roughness = [
            emissive_factor[0],
            emissive_factor[1],
            emissive_factor[2],
            pbr.roughness_factor(),
        ];

        let metallic = pbr.metallic_factor();

        let color_texture_id = get_texture_index(pbr.base_color_texture());
        let metallic_roughness_texture_id = get_texture_index(pbr.metallic_roughness_texture());
        let emissive_texture_id = get_texture_index(material.emissive_texture());
        let normal_texture_id = get_normal_texture_index(material.normal_texture());
        let color_metallicroughness_emissive_normal_texture_ids = ((color_texture_id as u32) << 24)
            | ((metallic_roughness_texture_id as u32) << 16)
            | ((emissive_texture_id as u32) << 8)
            | (normal_texture_id as u32);

        let (occlusion, occlusion_texture_id) = get_occlusion(material.occlusion_texture());
        let alpha_mode = get_alpha_mode_index(material.alpha_mode());
        let occlusion_texture_id_and_alpha_mode =
            ((occlusion_texture_id as u32) << 24) | (alpha_mode << 16);

        let alpha_cutoff = material.alpha_cutoff();

        Material {
            color,
            emissive_and_roughness,
            metallic,
            occlusion,
            color_metallicroughness_emissive_normal_texture_ids,
            occlusion_texture_id_and_alpha_mode,
            alpha_cutoff,
        }
    }
}

fn get_texture_index(texture_info: Option<Info>) -> u8 {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
        .filter(|index| *index < MAX_TEXTURE_COUNT as _)
        .map_or(NO_TEXTURE_ID, |index| index as _)
}

fn get_normal_texture_index(texture_info: Option<NormalTexture>) -> u8 {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
        .filter(|index| *index < MAX_TEXTURE_COUNT as _)
        .map_or(NO_TEXTURE_ID, |index| index as _)
}

fn get_occlusion(texture_info: Option<OcclusionTexture>) -> (f32, u32) {
    (
        texture_info
            .as_ref()
            .map_or(0.0, |tex_info| tex_info.strength()),
        texture_info
            .map(|tex_info| tex_info.texture())
            .map(|texture| texture.index())
            .filter(|index| *index < MAX_TEXTURE_COUNT as _)
            .map_or(NO_TEXTURE_ID as _, |index| index as _),
    )
}

fn get_alpha_mode_index(alpha_mode: AlphaMode) -> u32 {
    match alpha_mode {
        AlphaMode::Opaque => 0,
        AlphaMode::Mask => 1,
        AlphaMode::Blend => 2,
    }
}
