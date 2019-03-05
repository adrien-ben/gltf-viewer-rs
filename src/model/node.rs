use cgmath::{Matrix4, Quaternion, SquareMatrix, Vector3};
use gltf::{scene::Transform, Node as GltfNode, Scene};

pub struct Node {
    transform: Matrix4<f32>,
    mesh_index: usize,
}

impl Node {
    pub fn transform(&self) -> Matrix4<f32> {
        self.transform
    }

    pub fn mesh_index(&self) -> usize {
        self.mesh_index
    }
}

/// Traverse the complete node hierarchy of a given scene
pub fn traverse_scene_nodes(scene: &Scene) -> Vec<Node> {
    log::debug!("Traversing hierarchy");

    let mut nodes = Vec::new();
    for node in scene.nodes() {
        let transform = Matrix4::identity();
        traverse_node_hierarchy(&node, &transform, &mut nodes);
    }
    log::debug!("Found {} nodes referencing a mesh", nodes.len());

    nodes.sort_by(|a, b| a.mesh_index().cmp(&b.mesh_index()));
    nodes
}

/// Recursively traverse a node hierarchy from a given node
///
/// Compute and pass down the global transform down
fn traverse_node_hierarchy(
    node: &GltfNode,
    global_transform: &Matrix4<f32>,
    nodes: &mut Vec<Node>,
) {
    log::debug!("Traversing node {}", node.index());

    let global_transform = *global_transform * extract_transform_matrix(node);

    if let Some(mesh) = node.mesh() {
        log::debug!("Current node has a mesh. Index: {}", mesh.index());
        nodes.push(Node {
            transform: global_transform,
            mesh_index: mesh.index(),
        });
    }

    for child in node.children() {
        traverse_node_hierarchy(&child, &global_transform, nodes);
    }
}

/// Extract the transform matrix from a gltf node
fn extract_transform_matrix(node: &GltfNode) -> Matrix4<f32> {
    match node.transform() {
        Transform::Matrix { matrix } => Matrix4::from(matrix),
        Transform::Decomposed {
            translation,
            rotation: [xr, yr, zr, wr],
            scale: [xs, ys, zs],
        } => {
            let translation = Matrix4::from_translation(Vector3::from(translation));
            let rotation = Matrix4::from(Quaternion::new(wr, xr, yr, zr));
            let scale = Matrix4::from_nonuniform_scale(xs, ys, zs);
            translation * rotation * scale
        }
    }
}
