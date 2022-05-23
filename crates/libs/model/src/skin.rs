use super::node::Node;
use gltf::{buffer::Data, iter::Skins as GltfSkins, Skin as GltfSkin};
use math::cgmath::{Matrix4, SquareMatrix};

// Must be kept in sync with the value in model.vert
pub const MAX_JOINTS_PER_MESH: usize = 512;

#[derive(Clone, Debug)]
pub struct Skin {
    joints: Vec<Joint>,
}

impl Skin {
    /// Compute the joints matrices from the nodes matrices.
    pub fn compute_joints_matrices(&mut self, transform: Matrix4<f32>, nodes: &[Node]) {
        self.joints
            .iter_mut()
            .for_each(|j| j.compute_matrix(transform, nodes));
    }
}

impl Skin {
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Joint {
    matrix: Matrix4<f32>,
    inverse_bind_matrix: Matrix4<f32>,
    node_id: usize,
}

impl Joint {
    fn new(inverse_bind_matrix: Matrix4<f32>, node_id: usize) -> Self {
        Joint {
            matrix: Matrix4::identity(),
            inverse_bind_matrix,
            node_id,
        }
    }

    fn compute_matrix(&mut self, transform: Matrix4<f32>, nodes: &[Node]) {
        let global_transform_inverse = transform
            .invert()
            .expect("Transform matrix should be invertible");
        let node_transform = nodes[self.node_id].transform();

        self.matrix = global_transform_inverse * node_transform * self.inverse_bind_matrix;
    }
}

impl Joint {
    pub fn matrix(&self) -> Matrix4<f32> {
        self.matrix
    }
}

pub fn create_skins_from_gltf(gltf_skins: GltfSkins, data: &[Data]) -> Vec<Skin> {
    gltf_skins.map(|s| map_skin(&s, data)).collect::<Vec<_>>()
}

fn map_skin(gltf_skin: &GltfSkin, data: &[Data]) -> Skin {
    let joint_count = gltf_skin.joints().count();
    if joint_count > MAX_JOINTS_PER_MESH {
        log::warn!(
            "Skin {} has more than {} joints ({}). Mesh might not display properly",
            gltf_skin.index(),
            MAX_JOINTS_PER_MESH,
            joint_count
        );
    }

    let inverse_bind_matrices = map_inverse_bind_matrices(gltf_skin, data);
    let node_ids = map_node_ids(gltf_skin);

    let joints = inverse_bind_matrices
        .iter()
        .zip(node_ids)
        .map(|(matrix, node_id)| Joint::new(*matrix, node_id))
        .collect::<Vec<_>>();

    Skin { joints }
}

fn map_inverse_bind_matrices(gltf_skin: &GltfSkin, data: &[Data]) -> Vec<Matrix4<f32>> {
    let iter = gltf_skin
        .reader(|buffer| Some(&data[buffer.index()]))
        .read_inverse_bind_matrices()
        .expect("IBM reader not found for skin");
    iter.map(Matrix4::from).collect::<Vec<_>>()
}

fn map_node_ids(gltf_skin: &GltfSkin) -> Vec<usize> {
    gltf_skin
        .joints()
        .map(|node| node.index())
        .collect::<Vec<_>>()
}
