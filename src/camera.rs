use crate::controls::*;
use math::cgmath::{Matrix4, Point3};
use math::clamp;

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
    pub fn update(&mut self, input: &InputState) {
        if input.is_left_clicked() {
            let delta = input.cursor_delta();
            let theta = delta[0] as f32 * (0.2_f32).to_radians();
            let phi = delta[1] as f32 * (0.2_f32).to_radians();
            self.rotate(theta, phi);
        }
        self.forward(input.wheel_delta() * self.r * 0.2);
    }

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
