use gltf::material::Material as GltfMaterial;

pub const MAX_TEXTURE_COUNT: u32 = 64; // MUST be the same in the fragment shager

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Material {
    color: [f32; 3],
    texture_id: i32,
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(material: GltfMaterial) -> Material {
        let base_color_factor = material.pbr_metallic_roughness().base_color_factor();
        let color = [
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
        ];
        let texture_id = material
            .pbr_metallic_roughness()
            .base_color_texture()
            .map(|tex_info| tex_info.texture())
            .map(|texture| texture.index())
            .filter(|index| *index < MAX_TEXTURE_COUNT as _)
            .map_or(-1, |index| index as _);
        Material { color, texture_id }
    }
}
