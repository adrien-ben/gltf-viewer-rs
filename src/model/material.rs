use gltf::material::Material as GltfMaterial;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Material {
    color: [f32; 3]
}

impl<'a> From<GltfMaterial<'a>> for Material {
    fn from(gltf_material: GltfMaterial) -> Material {
        let base_color_factor = gltf_material.pbr_metallic_roughness().base_color_factor();
        let color = [base_color_factor[0], base_color_factor[1], base_color_factor[2]];
        Material { color }
    }
}