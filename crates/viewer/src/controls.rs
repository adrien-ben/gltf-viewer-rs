use vulkan::winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Copy, Clone, Debug)]
pub struct InputState {
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_left_clicked: bool,
    is_right_clicked: bool,
    cursor_delta: [f32; 2],
    wheel_delta: f32,
}

impl InputState {
    pub fn reset(self) -> Self {
        Self {
            cursor_delta: [0.0, 0.0],
            wheel_delta: 0.0,
            ..self
        }
    }

    pub fn handle_window_event(self, event: &WindowEvent) -> Self {
        let mut is_forward_pressed = None;
        let mut is_backward_pressed = None;
        let mut is_left_pressed = None;
        let mut is_right_pressed = None;
        let mut is_up_pressed = None;
        let mut is_down_pressed = None;
        let mut is_left_clicked = None;
        let mut is_right_clicked = None;
        let mut wheel_delta = self.wheel_delta;

        match event {
            WindowEvent::MouseInput { button, state, .. } => {
                let clicked = matches!(state, ElementState::Pressed);
                match button {
                    MouseButton::Left => is_left_clicked = Some(clicked),
                    MouseButton::Right => is_right_clicked = Some(clicked),
                    _ => {}
                };
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, v_lines),
                ..
            } => {
                wheel_delta += v_lines;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(scancode),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = matches!(state, ElementState::Pressed);
                match scancode {
                    KeyCode::KeyW => is_forward_pressed = Some(pressed),
                    KeyCode::KeyS => is_backward_pressed = Some(pressed),
                    KeyCode::KeyA => is_left_pressed = Some(pressed),
                    KeyCode::KeyD => is_right_pressed = Some(pressed),
                    KeyCode::Space => is_up_pressed = Some(pressed),
                    KeyCode::ControlLeft => is_down_pressed = Some(pressed),
                    _ => {}
                };
            }
            _ => {}
        }

        Self {
            is_forward_pressed: is_forward_pressed.unwrap_or(self.is_forward_pressed),
            is_backward_pressed: is_backward_pressed.unwrap_or(self.is_backward_pressed),
            is_left_pressed: is_left_pressed.unwrap_or(self.is_left_pressed),
            is_right_pressed: is_right_pressed.unwrap_or(self.is_right_pressed),
            is_up_pressed: is_up_pressed.unwrap_or(self.is_up_pressed),
            is_down_pressed: is_down_pressed.unwrap_or(self.is_down_pressed),
            is_left_clicked: is_left_clicked.unwrap_or(self.is_left_clicked),
            is_right_clicked: is_right_clicked.unwrap_or(self.is_right_clicked),
            wheel_delta,
            ..self
        }
    }

    pub fn handle_device_event(self, event: &DeviceEvent) -> Self {
        let mut cursor_delta = self.cursor_delta;

        if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
            cursor_delta[0] += *x as f32;
            cursor_delta[1] += *y as f32;
        }

        Self {
            cursor_delta,
            ..self
        }
    }
}

impl InputState {
    pub fn is_forward_pressed(&self) -> bool {
        self.is_forward_pressed
    }

    pub fn is_backward_pressed(&self) -> bool {
        self.is_backward_pressed
    }

    pub fn is_left_pressed(&self) -> bool {
        self.is_left_pressed
    }

    pub fn is_right_pressed(&self) -> bool {
        self.is_right_pressed
    }

    pub fn is_up_pressed(&self) -> bool {
        self.is_up_pressed
    }

    pub fn is_down_pressed(&self) -> bool {
        self.is_down_pressed
    }

    pub fn is_left_clicked(&self) -> bool {
        self.is_left_clicked
    }

    pub fn is_right_clicked(&self) -> bool {
        self.is_right_clicked
    }

    pub fn cursor_delta(&self) -> [f32; 2] {
        self.cursor_delta
    }

    pub fn wheel_delta(&self) -> f32 {
        self.wheel_delta
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_left_clicked: false,
            is_right_clicked: false,
            cursor_delta: [0.0, 0.0],
            wheel_delta: 0.0,
        }
    }
}
