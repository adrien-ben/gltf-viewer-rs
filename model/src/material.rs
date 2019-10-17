use gltf::{
    material::{AlphaMode, Material as GltfMaterial, NormalTexture, OcclusionTexture},
    texture::Info,
};

const NO_TEXTURE_ID: u32 = std::u8::MAX as u32;
pub const MAX_TEXTURE_COUNT: u32 = NO_TEXTURE_ID as u32 - 1;

const ALPHA_MODE_OPAQUE: u32 = 0;
const ALPHA_MODE_MASK: u32 = 1;
const ALPHA_MODE_BLEND: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    color: [f32; 4],
    emissive: [f32; 3],
    roughness: f32,
    metallic: f32,
    occlusion: f32,
    color_texture_id: Option<usize>,
    metallic_roughness_texture_id: Option<usize>,
    emissive_texture_id: Option<usize>,
    normals_texture_id: Option<usize>,
    occlusion_texture_id: Option<usize>,
    alpha_mode: u32,
    alpha_cutoff: f32,
}

impl Material {
    pub fn is_transparent(&self) -> bool {
        self.alpha_mode == ALPHA_MODE_BLEND
    }

    pub fn get_color_texture_index(&self) -> Option<usize> {
        self.color_texture_id
    }

    pub fn get_metallic_roughness_texture_index(&self) -> Option<usize> {
        self.metallic_roughness_texture_id
    }

    pub fn get_emissive_texture_index(&self) -> Option<usize> {
        self.emissive_texture_id
    }

    pub fn get_normals_texture_index(&self) -> Option<usize> {
        self.normals_texture_id
    }

    pub fn get_occlusion_texture_index(&self) -> Option<usize> {
        self.occlusion_texture_id
    }
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let pbr = material.pbr_metallic_roughness();

        let color = pbr.base_color_factor();

        let emissive = material.emissive_factor();
        let roughness = pbr.roughness_factor();
        let metallic = pbr.metallic_factor();

        let color_texture_id = get_texture_index(pbr.base_color_texture());
        let metallic_roughness_texture_id = get_texture_index(pbr.metallic_roughness_texture());
        let emissive_texture_id = get_texture_index(material.emissive_texture());
        let normals_texture_id = get_normals_texture_index(material.normal_texture());
        let (occlusion, occlusion_texture_id) = get_occlusion(material.occlusion_texture());
        let alpha_mode = get_alpha_mode_index(material.alpha_mode());

        let alpha_cutoff = material.alpha_cutoff();

        Material {
            color,
            emissive,
            roughness,
            metallic,
            occlusion,
            color_texture_id,
            metallic_roughness_texture_id,
            emissive_texture_id,
            normals_texture_id,
            occlusion_texture_id,
            alpha_mode,
            alpha_cutoff,
        }
    }
}

fn get_texture_index(texture_info: Option<Info>) -> Option<usize> {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
}

fn get_normals_texture_index(texture_info: Option<NormalTexture>) -> Option<usize> {
    texture_info
        .map(|tex_info| tex_info.texture())
        .map(|texture| texture.index())
}

fn get_occlusion(texture_info: Option<OcclusionTexture>) -> (f32, Option<usize>) {
    (
        texture_info
            .as_ref()
            .map_or(0.0, |tex_info| tex_info.strength()),
        texture_info
            .map(|tex_info| tex_info.texture())
            .map(|texture| texture.index()),
    )
}

fn get_alpha_mode_index(alpha_mode: AlphaMode) -> u32 {
    match alpha_mode {
        AlphaMode::Opaque => ALPHA_MODE_OPAQUE,
        AlphaMode::Mask => ALPHA_MODE_MASK,
        AlphaMode::Blend => ALPHA_MODE_BLEND,
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct MaterialUniform {
    color: [f32; 4],
    emissive_and_roughness: [f32; 4],
    metallic: f32,
    occlusion: f32,
    // Contains the texture ids for color metallic/roughness emissive and normal (each taking 8 bytes)
    color_metallicroughness_emissive_normal_texture_ids: u32,
    occlusion_texture_id_and_alpha_mode: u32,
    alpha_cutoff: f32,
}

impl<'a> From<Material> for MaterialUniform {
    fn from(material: Material) -> MaterialUniform {
        let color = material.color;
        let emissive_factor = material.emissive;

        let emissive_and_roughness = [
            emissive_factor[0],
            emissive_factor[1],
            emissive_factor[2],
            material.roughness,
        ];

        let metallic = material.metallic;

        let color_texture_id = material
            .color_texture_id
            .map_or(NO_TEXTURE_ID, |i| i as u32);
        let metallic_roughness_texture_id = material
            .metallic_roughness_texture_id
            .map_or(NO_TEXTURE_ID, |i| i as u32);
        let emissive_texture_id = material
            .emissive_texture_id
            .map_or(NO_TEXTURE_ID, |i| i as u32);
        let normal_texture_id = material
            .normals_texture_id
            .map_or(NO_TEXTURE_ID, |i| i as u32);
        let color_metallicroughness_emissive_normal_texture_ids = (color_texture_id << 24)
            | (metallic_roughness_texture_id << 16)
            | (emissive_texture_id << 8)
            | normal_texture_id;

        let occlusion = material.occlusion;
        let occlusion_texture_id = material
            .occlusion_texture_id
            .map_or(NO_TEXTURE_ID, |i| i as u32);
        let alpha_mode = material.alpha_mode;
        let occlusion_texture_id_and_alpha_mode =
            ((occlusion_texture_id as u32) << 24) | (alpha_mode << 16);

        let alpha_cutoff = material.alpha_cutoff;

        MaterialUniform {
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
