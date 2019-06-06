use cgmath::prelude::*;
use cgmath::{BaseFloat, Matrix4, Quaternion, Rad, Vector3, Vector4};
use std::{cmp::Ordering, ops::Mul};

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

/// Perspective matrix that is suitable for Vulkan.
///
/// It inverts the projected y-axis. And set the depth range to 0..1
/// instead of -1..1. Mind the vertex winding order though.
#[rustfmt::skip]
pub fn perspective<S, F>(fovy: F, aspect: S, near: S, far: S) -> Matrix4<S>
where
    S: BaseFloat,
    F: Into<Rad<S>>,
{
    let two = S::one() + S::one();
    let f = Rad::cot(fovy.into() / two);

    let c0r0 = f / aspect;
    let c0r1 = S::zero();
    let c0r2 = S::zero();
    let c0r3 = S::zero();

    let c1r0 = S::zero();
    let c1r1 = -f;
    let c1r2 = S::zero();
    let c1r3 = S::zero();

    let c2r0 = S::zero();
    let c2r1 = S::zero();
    let c2r2 = -far / (far - near);
    let c2r3 = -S::one();

    let c3r0 = S::zero();
    let c3r1 = S::zero();
    let c3r2 = -(far * near) / (far - near);
    let c3r3 = S::zero();

    Matrix4::new(
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3,
    )
}

/// Clamp `value` between `min` and `max`.
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    let value = if value > max { max } else { value };
    if value < min {
        min
    } else {
        value
    }
}

/// Return the partial minimum from an Iterator of PartialOrd if it exists.
pub fn partial_min<I, S>(iter: I) -> Option<S>
where
    S: PartialOrd,
    I: Iterator<Item = S>,
{
    iter.min_by(|v1, v2| v1.partial_cmp(v2).unwrap_or(Ordering::Equal))
}

/// Return the partial maximum from an Iterator of PartialOrd if it exists.
pub fn partial_max<I, S>(iter: I) -> Option<S>
where
    S: PartialOrd,
    I: Iterator<Item = S>,
{
    iter.max_by(|v1, v2| v1.partial_cmp(v2).unwrap_or(Ordering::Equal))
}

/// slerp from cgmath is bugged.
///
/// This algorithm is suggested in the cgmath issue about slerp
/// https://github.com/rustgd/cgmath/issues/300
pub fn slerp(left: Quaternion<f32>, right: Quaternion<f32>, amount: f32) -> Quaternion<f32> {
    let num2;
    let num3;
    let num = amount;
    let mut num4 = (((left.v.x * right.v.x) + (left.v.y * right.v.y)) + (left.v.z * right.v.z))
        + (left.s * right.s);
    let mut flag = false;
    if num4 < 0.0 {
        flag = true;
        num4 = -num4;
    }
    if num4 > 0.999_999 {
        num3 = 1.0 - num;
        num2 = if flag { -num } else { num };
    } else {
        let num5 = num4.acos();
        let num6 = 1.0 / num5.sin();
        num3 = ((1.0 - num) * num5).sin() * num6;
        num2 = if flag {
            -(num * num5).sin() * num6
        } else {
            (num * num5).sin() * num6
        };
    }
    Quaternion::new(
        (num3 * left.s) + (num2 * right.s),
        (num3 * left.v.x) + (num2 * right.v.x),
        (num3 * left.v.y) + (num2 * right.v.y),
        (num3 * left.v.z) + (num2 * right.v.z),
    )
}
