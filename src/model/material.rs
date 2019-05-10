use gltf::{material::Material as GltfMaterial, texture::Info};

pub const MAX_TEXTURE_COUNT: u32 = 64; // MUST be the same in the fragment shager

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Material {
    color: [f32; 3],
    metallic: f32,
    roughness: f32,
    color_texture_id: i32,
    metallic_roughness_texture_id: i32,
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let base_color_factor = material.pbr_metallic_roughness().base_color_factor();
        let color = [
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
        ];
        let metallic = material.pbr_metallic_roughness().metallic_factor();
        let roughness = material.pbr_metallic_roughness().roughness_factor();
        let color_texture_id =
            get_texture_index(material.pbr_metallic_roughness().base_color_texture());
        let metallic_roughness_texture_id = get_texture_index(
            material
                .pbr_metallic_roughness()
                .metallic_roughness_texture(),
        );
        Material {
            color,
            metallic,
            roughness,
            color_texture_id,
            metallic_roughness_texture_id,
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
