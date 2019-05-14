use crate::math::clamp;
use cgmath::{Matrix4, Point3};

const MIN_ORBITAL_CAMERA_DISTANCE: f32 = 0.05;

#[derive(Clone, Copy)]
pub struct Camera {
    theta: f32,
    phi: f32,
    r: f32,
}

impl Camera {
    pub fn position(&self) -> Point3<f32> {
        Point3::new(
            self.r * self.phi.sin() * self.theta.sin(),
            self.r * self.phi.cos(),
            self.r * self.phi.sin() * self.theta.cos(),
        )
    }
}

impl Camera {
    pub fn rotate(&mut self, theta: f32, phi: f32) {
        self.theta += theta;
        let phi = self.phi + phi;
        self.phi = clamp(phi, 10.0_f32.to_radians(), 170.0_f32.to_radians());
    }

    pub fn forward(&mut self, r: f32) {
        if (self.r - r).abs() > MIN_ORBITAL_CAMERA_DISTANCE {
            self.r -= r;
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            theta: 0.0_f32.to_radians(),
            phi: 90.0_f32.to_radians(),
            r: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct CameraUBO {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    eye: Point3<f32>,
}

impl CameraUBO {
    pub fn new(view: Matrix4<f32>, proj: Matrix4<f32>, eye: Point3<f32>) -> Self {
        Self { view, proj, eye }
    }
}
