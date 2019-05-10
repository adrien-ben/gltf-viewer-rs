use cgmath::{Matrix4, Point3};

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
