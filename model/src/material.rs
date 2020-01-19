use gltf::{
    material::{AlphaMode, Material as GltfMaterial, NormalTexture, OcclusionTexture},
    texture::Info,
};

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
    double_sided: bool,
    is_unlit: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TextureInfo {
    index: usize,
    channel: u32,
}

impl Material {
    pub fn get_color(&self) -> [f32; 4] {
        self.color
    }

    pub fn get_emissive(&self) -> [f32; 3] {
        self.emissive
    }

    pub fn get_roughness(&self) -> f32 {
        self.roughness
    }

    pub fn get_metallic(&self) -> f32 {
        self.metallic
    }

    pub fn get_occlusion(&self) -> f32 {
        self.occlusion
    }

    pub fn get_alpha_mode(&self) -> u32 {
        self.alpha_mode
    }

    pub fn get_alpha_cutoff(&self) -> f32 {
        self.alpha_cutoff
    }

    pub fn is_double_sided(&self) -> bool {
        self.double_sided
    }

    pub fn get_color_texture(&self) -> Option<TextureInfo> {
        self.color_texture
    }

    pub fn get_metallic_roughness_texture(&self) -> Option<TextureInfo> {
        self.metallic_roughness_texture
    }

    pub fn get_emissive_texture(&self) -> Option<TextureInfo> {
        self.emissive_texture
    }

    pub fn get_normals_texture(&self) -> Option<TextureInfo> {
        self.normals_texture
    }

    pub fn get_occlusion_texture(&self) -> Option<TextureInfo> {
        self.occlusion_texture
    }

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

    pub fn is_unlit(&self) -> bool {
        self.is_unlit
    }
}

impl TextureInfo {
    pub fn get_index(&self) -> usize {
        self.index
    }

    pub fn get_channel(&self) -> u32 {
        self.channel
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

        let double_sided = material.double_sided();

        let is_unlit = material.unlit();

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
            double_sided,
            is_unlit,
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
