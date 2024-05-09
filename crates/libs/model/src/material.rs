use gltf::{
    material::{AlphaMode, Material as GltfMaterial, NormalTexture, OcclusionTexture},
    texture::Info,
    Document,
};

const ALPHA_MODE_OPAQUE: u32 = 0;
const ALPHA_MODE_MASK: u32 = 1;
const ALPHA_MODE_BLEND: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Material {
    color: [f32; 4],
    emissive: [f32; 3],
    occlusion: f32,
    color_texture: Option<TextureInfo>,
    emissive_texture: Option<TextureInfo>,
    normals_texture: Option<TextureInfo>,
    occlusion_texture: Option<TextureInfo>,
    workflow: Workflow,
    alpha_mode: u32,
    alpha_cutoff: f32,
    double_sided: bool,
    is_unlit: bool,
    clearcoat: Option<Clearcoat>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0],
            occlusion: 0.0,
            color_texture: None,
            emissive_texture: None,
            normals_texture: None,
            occlusion_texture: None,
            workflow: Workflow::MetallicRoughness(MetallicRoughnessWorkflow {
                metallic: 1.0,
                roughness: 1.0,
                metallic_roughness_texture: None,
            }),
            alpha_mode: ALPHA_MODE_OPAQUE,
            alpha_cutoff: 0.5,
            double_sided: false,
            is_unlit: false,
            clearcoat: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TextureInfo {
    index: usize,
    channel: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Workflow {
    MetallicRoughness(MetallicRoughnessWorkflow),
    SpecularGlossiness(SpecularGlossinessWorkflow),
}

#[derive(Clone, Copy, Debug)]
pub struct MetallicRoughnessWorkflow {
    metallic: f32,
    roughness: f32,
    metallic_roughness_texture: Option<TextureInfo>,
}

impl MetallicRoughnessWorkflow {
    pub fn get_metallic(&self) -> f32 {
        self.metallic
    }

    pub fn get_roughness(&self) -> f32 {
        self.roughness
    }

    pub fn get_metallic_roughness_texture(&self) -> Option<TextureInfo> {
        self.metallic_roughness_texture
    }

    pub fn get_metallic_roughness_texture_index(&self) -> Option<usize> {
        self.metallic_roughness_texture.map(|info| info.index)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpecularGlossinessWorkflow {
    specular: [f32; 3],
    glossiness: f32,
    specular_glossiness_texture: Option<TextureInfo>,
}

impl SpecularGlossinessWorkflow {
    pub fn get_specular(&self) -> [f32; 3] {
        self.specular
    }

    pub fn get_glossiness(&self) -> f32 {
        self.glossiness
    }

    pub fn get_specular_glossiness_texture(&self) -> Option<TextureInfo> {
        self.specular_glossiness_texture
    }

    pub fn get_specular_glossiness_texture_index(&self) -> Option<usize> {
        self.specular_glossiness_texture.map(|info| info.index)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Clearcoat {
    factor: f32,
    roughness: f32,
    factor_texture: Option<TextureInfo>,
    roughness_texture: Option<TextureInfo>,
    normal_texture: Option<TextureInfo>,
}

impl Clearcoat {
    pub fn factor(&self) -> f32 {
        self.factor
    }

    pub fn roughness(&self) -> f32 {
        self.roughness
    }

    pub fn factor_texture(&self) -> Option<TextureInfo> {
        self.factor_texture
    }

    pub fn factor_texture_index(&self) -> Option<usize> {
        self.factor_texture.map(|info| info.index)
    }

    pub fn roughness_texture(&self) -> Option<TextureInfo> {
        self.roughness_texture
    }

    pub fn roughness_texture_index(&self) -> Option<usize> {
        self.roughness_texture.map(|info| info.index)
    }

    pub fn normal_texture(&self) -> Option<TextureInfo> {
        self.normal_texture
    }

    pub fn normal_texture_index(&self) -> Option<usize> {
        self.normal_texture.map(|info| info.index)
    }
}

impl Material {
    pub fn get_color(&self) -> [f32; 4] {
        self.color
    }

    pub fn get_emissive(&self) -> [f32; 3] {
        self.emissive
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

    pub fn get_clearcoat(&self) -> Option<Clearcoat> {
        self.clearcoat
    }

    pub fn get_workflow(&self) -> Workflow {
        self.workflow
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

pub(crate) fn create_materials_from_gltf(document: &Document) -> Vec<Material> {
    document.materials().map(Material::from).collect()
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let color = match material.pbr_specular_glossiness() {
            Some(pbr) => pbr.diffuse_factor(),
            _ => material.pbr_metallic_roughness().base_color_factor(),
        };

        let emissive_strength = material.emissive_strength().unwrap_or(1.0);
        let emissive = material.emissive_factor();
        let emissive = [
            emissive[0] * emissive_strength,
            emissive[1] * emissive_strength,
            emissive[2] * emissive_strength,
        ];

        let color_texture = match material.pbr_specular_glossiness() {
            Some(pbr) => pbr.diffuse_texture(),
            _ => material.pbr_metallic_roughness().base_color_texture(),
        };
        let color_texture = get_texture(color_texture);
        let emissive_texture = get_texture(material.emissive_texture());
        let normals_texture = get_normals_texture(material.normal_texture());
        let (occlusion, occlusion_texture) = get_occlusion(material.occlusion_texture());

        let workflow = match material.pbr_specular_glossiness() {
            Some(pbr) => Workflow::SpecularGlossiness(SpecularGlossinessWorkflow {
                specular: pbr.specular_factor(),
                glossiness: pbr.glossiness_factor(),
                specular_glossiness_texture: get_texture(pbr.specular_glossiness_texture()),
            }),
            _ => {
                let pbr = material.pbr_metallic_roughness();
                Workflow::MetallicRoughness(MetallicRoughnessWorkflow {
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    metallic_roughness_texture: get_texture(pbr.metallic_roughness_texture()),
                })
            }
        };

        let alpha_mode = get_alpha_mode_index(material.alpha_mode());
        let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);

        let double_sided = material.double_sided();

        let is_unlit = material.unlit();

        let clearcoat = material.clearcoat().map(|m| Clearcoat {
            factor: m.clearcoat_factor(),
            roughness: m.clearcoat_roughness_factor(),
            factor_texture: get_texture(m.clearcoat_texture()),
            roughness_texture: get_texture(m.clearcoat_roughness_texture()),
            normal_texture: get_texture(m.clearcoat_normal_texture()),
        });

        Material {
            color,
            emissive,
            occlusion,
            color_texture,
            emissive_texture,
            normals_texture,
            occlusion_texture,
            workflow,
            alpha_mode,
            alpha_cutoff,
            double_sided,
            is_unlit,
            clearcoat,
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
