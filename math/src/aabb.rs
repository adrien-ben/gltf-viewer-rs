use super::{partial_max, partial_min};
use cgmath::{vec3, BaseFloat, Matrix4, Vector3, Vector4};
use std::ops::Mul;

/// Axis aligned bounding box.
#[derive(Copy, Clone, Debug)]
pub struct AABB<S> {
    min: Vector3<S>,
    max: Vector3<S>,
}

impl<S> AABB<S> {
    /// Create a new AABB.
    pub fn new(min: Vector3<S>, max: Vector3<S>) -> Self {
        AABB { min, max }
    }
}

impl<S: BaseFloat> AABB<S> {
    /// Compute the union of several AABBs.
    pub fn union(aabbs: &[AABB<S>]) -> Option<Self> {
        if aabbs.is_empty() {
            None
        } else if aabbs.len() == 1 {
            Some(aabbs[0])
        } else {
            let min_x = partial_min(aabbs.iter().map(|aabb| aabb.min.x)).unwrap();
            let min_y = partial_min(aabbs.iter().map(|aabb| aabb.min.y)).unwrap();
            let min_z = partial_min(aabbs.iter().map(|aabb| aabb.min.z)).unwrap();
            let min = Vector3::new(min_x, min_y, min_z);

            let max_x = partial_max(aabbs.iter().map(|aabb| aabb.max.x)).unwrap();
            let max_y = partial_max(aabbs.iter().map(|aabb| aabb.max.y)).unwrap();
            let max_z = partial_max(aabbs.iter().map(|aabb| aabb.max.z)).unwrap();
            let max = Vector3::new(max_x, max_y, max_z);

            Some(AABB::new(min, max))
        }
    }

    /// Get the size of the larger side of the AABB.
    pub fn get_larger_side_size(&self) -> S {
        let size = self.max - self.min;
        let x = size.x.abs();
        let y = size.y.abs();
        let z = size.z.abs();

        if x > y && x > z {
            x
        } else if y > z {
            y
        } else {
            z
        }
    }

    /// Get the center of the AABB.
    pub fn get_center(&self) -> Vector3<S> {
        let two = S::one() + S::one();
        self.min + (self.max - self.min) / two
    }

    /// Get all vertices of the AABB.
    pub fn get_vertices(&self) -> [Vector3<S>; 8] {
        [
            vec3(self.min.x, self.min.y, self.min.z),
            vec3(self.max.x, self.min.y, self.min.z),
            vec3(self.min.x, self.min.y, self.max.z),
            vec3(self.max.x, self.min.y, self.max.z),
            vec3(self.min.x, self.max.y, self.min.z),
            vec3(self.max.x, self.max.y, self.min.z),
            vec3(self.min.x, self.max.y, self.max.z),
            vec3(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Get all vertices of the AABB transformed with the passed in matrix.
    pub fn get_transformed_vertices(&self, transform: Matrix4<S>) -> [Vector3<S>; 8] {
        let mut vertices = self.get_vertices();

        for vertex in vertices.iter_mut() {
            let transformed = transform * vertex.extend(S::one());
            vertex.x = transformed.x;
            vertex.y = transformed.y;
            vertex.z = transformed.z;
        }

        vertices
    }
}

/// Transform the AABB by multiplying it with a Matrix4.
impl<S: BaseFloat> Mul<Matrix4<S>> for AABB<S> {
    type Output = AABB<S>;

    fn mul(self, rhs: Matrix4<S>) -> Self::Output {
        let min = self.min;
        let min = rhs * Vector4::new(min.x, min.y, min.z, S::one());

        let max = self.max;
        let max = rhs * Vector4::new(max.x, max.y, max.z, S::one());

        AABB::new(min.truncate(), max.truncate())
    }
}

/// Scale the AABB by multiplying it by a BaseFloat
impl<S: BaseFloat> Mul<S> for AABB<S> {
    type Output = AABB<S>;

    fn mul(self, rhs: S) -> Self::Output {
        AABB::new(self.min * rhs, self.max * rhs)
    }
}
