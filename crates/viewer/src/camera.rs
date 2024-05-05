use crate::controls::*;
use math::cgmath::{Deg, InnerSpace, Matrix3, Matrix4, Point3, Rad, Vector3, Zero};
use math::clamp;

const MIN_ORBITAL_CAMERA_DISTANCE: f32 = 0.5;
const TARGET_MOVEMENT_SPEED: f32 = 0.003;
const ROTATION_SPEED_DEG: f32 = 0.4;
pub const DEFAULT_FPS_MOVE_SPEED: f32 = 6.0;

pub const DEFAULT_FOV: f32 = 45.0;
pub const DEFAULT_Z_NEAR: f32 = 0.01;
pub const DEFAULT_Z_FAR: f32 = 100.0;

#[derive(Debug, Clone, Copy)]

pub struct Camera {
    mode: Mode,
    pub fov: Deg<f32>,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            fov: Deg(DEFAULT_FOV),
            z_near: DEFAULT_Z_NEAR,
            z_far: DEFAULT_Z_FAR,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Orbital(Orbital),
    Fps(Fps),
}

impl Default for Mode {
    fn default() -> Self {
        Self::Orbital(Default::default())
    }
}

impl Camera {
    pub fn update(&mut self, input: &InputState, delta_time_secs: f32) {
        match &mut self.mode {
            Mode::Orbital(c) => c.update(input, delta_time_secs),
            Mode::Fps(c) => c.update(input, delta_time_secs),
        }
    }

    pub fn position(&self) -> Point3<f32> {
        match self.mode {
            Mode::Orbital(c) => c.position(),
            Mode::Fps(c) => c.position(),
        }
    }

    pub fn target(&self) -> Point3<f32> {
        match self.mode {
            Mode::Orbital(c) => c.target(),
            Mode::Fps(c) => c.target(),
        }
    }

    pub fn to_orbital(self) -> Self {
        let mode = match self.mode {
            Mode::Orbital(_) => self.mode,
            Mode::Fps(c) => Mode::Orbital(c.into()),
        };
        Self { mode, ..self }
    }

    pub fn to_fps(self) -> Self {
        let mode = match self.mode {
            Mode::Fps(_) => self.mode,
            Mode::Orbital(c) => Mode::Fps(c.into()),
        };
        Self { mode, ..self }
    }

    pub fn set_move_speed(&mut self, move_speed: f32) {
        if let Mode::Fps(c) = &mut self.mode {
            c.move_speed = move_speed;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Orbital {
    theta: f32,
    phi: f32,
    r: f32,
    target: Point3<f32>,
}

impl Default for Orbital {
    fn default() -> Self {
        Self {
            theta: 0.0_f32.to_radians(),
            phi: 90.0_f32.to_radians(),
            r: 10.0,
            target: Point3::new(0.0, 0.0, 0.0),
        }
    }
}

impl From<Fps> for Orbital {
    fn from(fps: Fps) -> Self {
        let Point3 { x, y, z } = fps.position;
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;

        let r = (xx + yy + zz).sqrt();
        let theta = x.signum() * (z / (f32::EPSILON + (zz + xx).sqrt())).acos();
        let phi = (y / (r + f32::EPSILON)).acos();

        Self {
            r,
            theta,
            phi,
            target: Point3::new(0.0, 0.0, 0.0),
        }
    }
}

impl Orbital {
    fn update(&mut self, input: &InputState, _: f32) {
        // Rotation
        if input.is_left_clicked() {
            let delta = input.cursor_delta();
            let theta = delta[0] * ROTATION_SPEED_DEG.to_radians();
            let phi = delta[1] * ROTATION_SPEED_DEG.to_radians();
            self.rotate(theta, phi);
        }

        // Target move
        if input.is_right_clicked() {
            let position = self.position();
            let forward = (self.target - position).normalize();
            let up = Vector3::unit_y();
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

    fn position(&self) -> Point3<f32> {
        Point3::new(
            self.target[0] + self.r * self.phi.sin() * self.theta.sin(),
            self.target[1] + self.r * self.phi.cos(),
            self.target[2] + self.r * self.phi.sin() * self.theta.cos(),
        )
    }

    fn target(&self) -> Point3<f32> {
        self.target
    }
}

#[derive(Debug, Clone, Copy)]
struct Fps {
    position: Point3<f32>,
    direction: Vector3<f32>,
    move_speed: f32,
}

impl Default for Fps {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 10.0),
            direction: -Vector3::unit_z(),
            move_speed: DEFAULT_FPS_MOVE_SPEED,
        }
    }
}

impl From<Orbital> for Fps {
    fn from(orbital: Orbital) -> Self {
        let position = orbital.position();
        let target = orbital.target();
        let direction = (target - position).normalize();
        Self {
            position,
            direction,
            move_speed: DEFAULT_FPS_MOVE_SPEED,
        }
    }
}

impl Fps {
    fn update(&mut self, input: &InputState, delta_time_secs: f32) {
        let forward = self.direction.normalize();
        let up = Vector3::unit_y();
        let right = up.cross(forward).normalize();
        let up = forward.cross(right.normalize());

        // compute movement
        let mut move_dir = Vector3::zero();
        if input.is_forward_pressed() {
            move_dir += forward;
        }
        if input.is_backward_pressed() {
            move_dir -= forward;
        }
        if input.is_left_pressed() {
            move_dir += right;
        }
        if input.is_right_pressed() {
            move_dir -= right;
        }
        if input.is_up_pressed() {
            move_dir += up;
        }
        if input.is_down_pressed() {
            move_dir -= up;
        }

        if !move_dir.is_zero() {
            move_dir = move_dir.normalize() * delta_time_secs * self.move_speed;
        }

        self.position += move_dir;

        // compute rotation
        if input.is_left_clicked() {
            let delta = input.cursor_delta();

            let rot_speed = delta_time_secs * ROTATION_SPEED_DEG;
            let rot_y = Matrix3::<f32>::from_angle_y(Rad(-delta[0] * rot_speed));
            let rot_x = Matrix3::<f32>::from_axis_angle(right, Rad(delta[1] * rot_speed));

            self.direction = (rot_x * rot_y * forward).normalize();
        }
    }

    fn position(&self) -> Point3<f32> {
        self.position
    }

    fn target(&self) -> Point3<f32> {
        self.position + self.direction.normalize()
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct CameraUBO {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    inverted_proj: Matrix4<f32>,
    eye: Point3<f32>,
    padding: f32,
    z_near: f32,
    z_far: f32,
}

impl CameraUBO {
    pub fn new(
        view: Matrix4<f32>,
        proj: Matrix4<f32>,
        inverted_proj: Matrix4<f32>,
        eye: Point3<f32>,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        Self {
            view,
            proj,
            inverted_proj,
            eye,
            padding: 0.0,
            z_near,
            z_far,
        }
    }
}
