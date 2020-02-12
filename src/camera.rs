use crate::controls::*;
use math::cgmath::{InnerSpace, Matrix4, Point3, Vector3};
use math::clamp;

const MIN_ORBITAL_CAMERA_DISTANCE: f32 = 0.05;
const TARGET_MOVEMENT_SPEED: f32 = 0.003;

#[derive(Clone, Copy)]
pub struct Camera {
    theta: f32,
    phi: f32,
    r: f32,
    target: Point3<f32>,
}

impl Camera {
    pub fn position(&self) -> Point3<f32> {
        Point3::new(
            self.target[0] + self.r * self.phi.sin() * self.theta.sin(),
            self.target[1] + self.r * self.phi.cos(),
            self.target[2] + self.r * self.phi.sin() * self.theta.cos(),
        )
    }

    pub fn target(&self) -> Point3<f32> {
        self.target
    }
}

impl Camera {
    pub fn update(&mut self, input: &InputState) {
        // Rotation
        if input.is_left_clicked() {
            let delta = input.cursor_delta();
            let theta = delta[0] as f32 * (0.2_f32).to_radians();
            let phi = delta[1] as f32 * (0.2_f32).to_radians();
            self.rotate(theta, phi);
        }

        // Target move
        if input.is_right_clicked() {
            let position = self.position();
            let forward = (self.target - position).normalize();
            let up = Vector3::new(0.0, 1.0, 0.0);
            let right = up.cross(forward).normalize();
            let up = forward.cross(right.normalize());

            let delta = input.cursor_delta();
            if delta[0] != 0.0 {
                self.target += right * delta[0] * self.r * TARGET_MOVEMENT_SPEED;
            }
            if delta[1] != 0.0 {
                self.target += up * delta[1] * self.r * TARGET_MOVEMENT_SPEED;
            }
        }

        // Zoom
        self.forward(input.wheel_delta() * self.r * 0.2);
    }

    fn rotate(&mut self, theta: f32, phi: f32) {
        self.theta += theta;
        let phi = self.phi + phi;
        self.phi = clamp(phi, 10.0_f32.to_radians(), 170.0_f32.to_radians());
    }

    fn forward(&mut self, r: f32) {
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
            target: Point3::new(0.0, 0.0, 0.0),
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
