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
    color_texture: Option<TextureInfo>,
    metallic_roughness_texture: Option<TextureInfo>,
    emissive_texture: Option<TextureInfo>,
    normals_texture: Option<TextureInfo>,
    occlusion_texture: Option<TextureInfo>,
    alpha_mode: u32,
    alpha_cutoff: f32,
}

#[derive(Clone, Copy, Debug)]
struct TextureInfo {
    index: usize,
    channel: u32,
}

impl Material {
    pub fn is_transparent(&self) -> bool {
        self.alpha_mode == ALPHA_MODE_BLEND
    }

    pub fn get_color_texture_index(&self) -> Option<usize> {
        self.color_texture.map(|info| info.index)
    }

    pub fn get_metallic_roughness_texture_index(&self) -> Option<usize> {
        self.metallic_roughness_texture.map(|info| info.index)
    }

    pub fn get_emissive_texture_index(&self) -> Option<usize> {
        self.emissive_texture.map(|info| info.index)
    }

    pub fn get_normals_texture_index(&self) -> Option<usize> {
        self.normals_texture.map(|info| info.index)
    }

    pub fn get_occlusion_texture_index(&self) -> Option<usize> {
        self.occlusion_texture.map(|info| info.index)
    }
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let pbr = material.pbr_metallic_roughness();

        let color = pbr.base_color_factor();

        let emissive = material.emissive_factor();
        let roughness = pbr.roughness_factor();
        let metallic = pbr.metallic_factor();

        let color_texture = get_texture(pbr.base_color_texture());
        let metallic_roughness_texture = get_texture(pbr.metallic_roughness_texture());
        let emissive_texture = get_texture(material.emissive_texture());
        let normals_texture = get_normals_texture(material.normal_texture());
        let (occlusion, occlusion_texture) = get_occlusion(material.occlusion_texture());
        let alpha_mode = get_alpha_mode_index(material.alpha_mode());

        let alpha_cutoff = material.alpha_cutoff();

        Material {
            color,
            emissive,
            roughness,
            metallic,
            occlusion,
            color_texture,
            metallic_roughness_texture,
            emissive_texture,
            normals_texture,
            occlusion_texture,
            alpha_mode,
            alpha_cutoff,
        }
    }
}

fn get_texture(texture_info: Option<Info>) -> Option<TextureInfo> {
    texture_info.map(|tex_info| TextureInfo {
        index: tex_info.texture().index(),
        channel: tex_info.tex_coord(),
    })
}

fn get_normals_texture(texture_info: Option<NormalTexture>) -> Option<TextureInfo> {
    texture_info.map(|tex_info| TextureInfo {
        index: tex_info.texture().index(),
        channel: tex_info.tex_coord(),
    })
}

fn get_occlusion(texture_info: Option<OcclusionTexture>) -> (f32, Option<TextureInfo>) {
    let strength = texture_info
        .as_ref()
        .map_or(0.0, |tex_info| tex_info.strength());

    let texture = texture_info.map(|tex_info| TextureInfo {
        index: tex_info.texture().index(),
        channel: tex_info.tex_coord(),
    });

    (strength, texture)
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
    // Contains the texture channels for color metallic/roughness emissive and normal (each taking 8 bytes)
    color_metallicroughness_emissive_normal_texture_channels: u32,
    occlusion_texture_channel_and_alpha_mode: u32,
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
            .color_texture
            .map_or(NO_TEXTURE_ID, |info| info.channel);
        let metallic_roughness_texture_id = material
            .metallic_roughness_texture
            .map_or(NO_TEXTURE_ID, |info| info.channel);
        let emissive_texture_id = material
            .emissive_texture
            .map_or(NO_TEXTURE_ID, |info| info.channel);
        let normal_texture_id = material
            .normals_texture
            .map_or(NO_TEXTURE_ID, |info| info.channel);
        let color_metallicroughness_emissive_normal_texture_channels = (color_texture_id << 24)
            | (metallic_roughness_texture_id << 16)
            | (emissive_texture_id << 8)
            | normal_texture_id;

        let occlusion = material.occlusion;
        let occlusion_texture_id = material
            .occlusion_texture
            .map_or(NO_TEXTURE_ID, |info| info.channel);
        let alpha_mode = material.alpha_mode;
        let occlusion_texture_channel_and_alpha_mode =
            ((occlusion_texture_id as u32) << 24) | (alpha_mode << 16);

        let alpha_cutoff = material.alpha_cutoff;

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
