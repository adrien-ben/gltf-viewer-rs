use gltf::Document;
use gltf::iter::Lights;
use gltf::khr_lights_punctual::{Kind, Light as GltfLight};

#[derive(Copy, Clone, Debug)]
pub struct Light {
    color: [f32; 3],
    intensity: f32,
    range: Option<f32>,
    light_type: Type,
}

impl Light {
    pub fn color(&self) -> [f32; 3] {
        self.color
    }

    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    pub fn range(&self) -> Option<f32> {
        self.range
    }

    pub fn light_type(&self) -> Type {
        self.light_type
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
    Directional,
    Point,
    Spot {
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    },
}

pub(crate) fn create_lights_from_gltf(document: &Document) -> Vec<Light> {
    document.lights().map_or(vec![], map_gltf_lights)
}

fn map_gltf_lights(lights: Lights) -> Vec<Light> {
    lights.map(map_gltf_light).collect()
}

fn map_gltf_light(light: GltfLight) -> Light {
    let color = light.color();
    let intensity = light.intensity();
    let range = light.range();
    let light_type = map_gltf_light_type(light.kind());

    Light {
        color,
        intensity,
        range,
        light_type,
    }
}

fn map_gltf_light_type(kind: Kind) -> Type {
    match kind {
        Kind::Directional => Type::Directional,
        Kind::Point => Type::Point,
        Kind::Spot {
            inner_cone_angle,
            outer_cone_angle,
        } => Type::Spot {
            inner_cone_angle,
            outer_cone_angle,
        },
    }
}
