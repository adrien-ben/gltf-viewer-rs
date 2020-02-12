use gltf::{
    khr_lights_punctual::{Kind as GltfLightKind, Light as GltfLight},
    material::AlphaMode as GltfAlphaMode,
    mesh::Mode as GltfPrimitiveMode,
    Animation as GltfAnimation, Document, Material as GltfMaterial, Mesh as GltfMesh,
    Node as GltfNode, Primitive as GltfPrimitive, Scene,
};
use std::{fmt, path::Path};

#[derive(Clone, Debug)]
pub struct Metadata {
    name: String,
    path: String,
    scene_count: usize,
    node_count: usize,
    animation_count: usize,
    skin_count: usize,
    mesh_count: usize,
    material_count: usize,
    texture_count: usize,
    light_count: usize,
    nodes: Vec<Node>,
    animations: Vec<Animation>,
}

impl Metadata {
    pub(crate) fn new<P: AsRef<Path>>(path: P, document: &Document) -> Self {
        Metadata {
            name: String::from(path.as_ref().file_name().unwrap().to_str().unwrap()),
            path: String::from(path.as_ref().to_str().unwrap()),
            scene_count: document.scenes().len(),
            node_count: document.nodes().len(),
            animation_count: document.animations().len(),
            skin_count: document.skins().len(),
            mesh_count: document.meshes().len(),
            material_count: document.materials().len(),
            texture_count: document.textures().len(),
            light_count: document.lights().map_or(0, |lights| lights.len()),
            nodes: build_tree(document),
            animations: document.animations().map(Animation::from).collect(),
        }
    }
}

fn build_tree(document: &Document) -> Vec<Node> {
    let mut uid = 0;
    document
        .scenes()
        .map(|s| map_scene_node(&s, &mut uid))
        .collect()
}

fn map_scene_node(scene: &Scene, current_uid: &mut usize) -> Node {
    let children = scene
        .nodes()
        .map(|n| map_node(&n, true, current_uid))
        .collect();
    *current_uid += 1;
    Node {
        uid: *current_uid,
        index: scene.index(),
        name: scene.name().map(String::from),
        kind: NodeKind::Scene,
        children,
    }
}

fn map_node(node: &GltfNode, root: bool, current_uid: &mut usize) -> Node {
    let children = node
        .children()
        .map(|n| map_node(&n, false, current_uid))
        .collect();
    *current_uid += 1;
    Node {
        uid: *current_uid,
        index: node.index(),
        name: node.name().map(String::from),
        kind: map_node_kind(node, root),
        children,
    }
}

fn map_node_kind(node: &GltfNode, root: bool) -> NodeKind {
    NodeKind::Node(NodeData {
        leaf: node.children().len() == 0,
        root,
        mesh: map_mesh_data(node),
        light: map_light_data(node),
    })
}

fn map_mesh_data(node: &GltfNode) -> Option<Mesh> {
    node.mesh().map(Mesh::from)
}

fn map_light_data(node: &GltfNode) -> Option<Light> {
    node.light().map(Light::from)
}

impl Metadata {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn scene_count(&self) -> usize {
        self.scene_count
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn animation_count(&self) -> usize {
        self.animation_count
    }

    pub fn skin_count(&self) -> usize {
        self.skin_count
    }

    pub fn mesh_count(&self) -> usize {
        self.mesh_count
    }

    pub fn material_count(&self) -> usize {
        self.material_count
    }

    pub fn texture_count(&self) -> usize {
        self.texture_count
    }

    pub fn light_count(&self) -> usize {
        self.light_count
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn animations(&self) -> &[Animation] {
        &self.animations
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    uid: usize,
    index: usize,
    name: Option<String>,
    kind: NodeKind,
    children: Vec<Node>,
}

impl Node {
    pub fn uid(&self) -> usize {
        self.uid
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_ref().map(String::as_str)
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    pub fn children(&self) -> &[Node] {
        &self.children
    }
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Scene,
    Node(NodeData),
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            NodeKind::Scene => "Scene",
            NodeKind::Node(..) => "Node",
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Debug)]
pub struct NodeData {
    pub leaf: bool,
    pub root: bool,
    pub mesh: Option<Mesh>,
    pub light: Option<Light>,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub index: usize,
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
}

impl From<GltfMesh<'_>> for Mesh {
    fn from(mesh: GltfMesh) -> Mesh {
        Mesh {
            index: mesh.index(),
            name: mesh.name().map(String::from),
            primitives: mesh.primitives().map(Primitive::from).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Primitive {
    pub index: usize,
    pub mode: PrimitiveMode,
    pub material: Material,
}

impl From<GltfPrimitive<'_>> for Primitive {
    fn from(primitive: GltfPrimitive) -> Primitive {
        Primitive {
            index: primitive.index(),
            mode: PrimitiveMode::from(primitive.mode()),
            material: Material::from(primitive.material()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PrimitiveMode {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

impl fmt::Display for PrimitiveMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PrimitiveMode::Points => "Points",
            PrimitiveMode::Lines => "Lines",
            PrimitiveMode::LineLoop => "LineLoop",
            PrimitiveMode::LineStrip => "LineStrip",
            PrimitiveMode::Triangles => "Triangles",
            PrimitiveMode::TriangleStrip => "TriangleStrip",
            PrimitiveMode::TriangleFan => "TriangleFan",
        };
        write!(f, "{}", name)
    }
}

impl From<GltfPrimitiveMode> for PrimitiveMode {
    fn from(mode: GltfPrimitiveMode) -> PrimitiveMode {
        match mode {
            GltfPrimitiveMode::Points => PrimitiveMode::Points,
            GltfPrimitiveMode::Lines => PrimitiveMode::Lines,
            GltfPrimitiveMode::LineLoop => PrimitiveMode::LineLoop,
            GltfPrimitiveMode::LineStrip => PrimitiveMode::LineStrip,
            GltfPrimitiveMode::Triangles => PrimitiveMode::Triangles,
            GltfPrimitiveMode::TriangleStrip => PrimitiveMode::TriangleStrip,
            GltfPrimitiveMode::TriangleFan => PrimitiveMode::TriangleFan,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Material {
    pub index: Option<usize>,
    pub name: Option<String>,
    pub alpha_cutoff: f32,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    pub base_color: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_color: [f32; 3],
    pub unlit: bool,
}

impl From<GltfMaterial<'_>> for Material {
    fn from(material: GltfMaterial) -> Material {
        Material {
            index: material.index(),
            name: material.name().map(String::from),
            alpha_cutoff: material.alpha_cutoff(),
            alpha_mode: AlphaMode::from(material.alpha_mode()),
            double_sided: material.double_sided(),
            base_color: material.pbr_metallic_roughness().base_color_factor(),
            metallic_factor: material.pbr_metallic_roughness().metallic_factor(),
            roughness_factor: material.pbr_metallic_roughness().roughness_factor(),
            emissive_color: material.emissive_factor(),
            unlit: material.unlit(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            AlphaMode::Opaque => "Opaque",
            AlphaMode::Mask => "Mask",
            AlphaMode::Blend => "Blend",
        };
        write!(f, "{}", name)
    }
}

impl From<GltfAlphaMode> for AlphaMode {
    fn from(mode: GltfAlphaMode) -> AlphaMode {
        match mode {
            GltfAlphaMode::Opaque => AlphaMode::Opaque,
            GltfAlphaMode::Mask => AlphaMode::Mask,
            GltfAlphaMode::Blend => AlphaMode::Blend,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Light {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: Option<f32>,
    pub kind: LightKind,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LightKind {
    Directional,
    Point,
    Spot {
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    },
}

impl From<GltfLight<'_>> for Light {
    fn from(light: GltfLight) -> Light {
        use GltfLightKind::*;
        let kind = match light.kind() {
            Directional => LightKind::Directional,
            Point => LightKind::Point,
            Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => LightKind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            },
        };

        Light {
            color: light.color(),
            intensity: light.intensity(),
            range: light.range(),
            kind,
        }
    }
}

impl fmt::Display for LightKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            LightKind::Directional => "Directional",
            LightKind::Point => "Point",
            LightKind::Spot { .. } => "Spot",
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Debug)]
pub struct Animation {
    pub index: usize,
    pub name: Option<String>,
}

impl From<GltfAnimation<'_>> for Animation {
    fn from(animation: GltfAnimation) -> Self {
        Self {
            index: animation.index(),
            name: animation.name().map(String::from),
        }
    }
}
