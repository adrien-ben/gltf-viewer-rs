use vulkan::winit::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};

#[derive(Copy, Clone, Debug)]
pub struct InputState {
    is_left_clicked: bool,
    cursor_delta: [f32; 2],
    wheel_delta: f32,
}

impl InputState {
    pub fn update(self, event: &Event) -> Self {
        let mut is_left_clicked = None;
        let mut wheel_delta = self.wheel_delta;
        let mut cursor_delta = self.cursor_delta;

        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    if *state == ElementState::Pressed {
                        is_left_clicked = Some(true);
                    } else {
                        is_left_clicked = Some(false);
                    }
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, v_lines),
                    ..
                } => {
                    wheel_delta += v_lines;
                }
                _ => {}
            }
        }

        if let Event::DeviceEvent { event, .. } = event {
            if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
                cursor_delta[0] += *x as f32;
                cursor_delta[1] += *y as f32;
            }
        }

        Self {
            is_left_clicked: is_left_clicked.unwrap_or(self.is_left_clicked),
            cursor_delta,
            wheel_delta,
        }
    }

    pub fn reset(&mut self) {
        self.cursor_delta = [0.0, 0.0];
        self.wheel_delta = 0.0;
    }
}

impl InputState {
    pub fn is_left_clicked(&self) -> bool {
        self.is_left_clicked
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
            is_left_clicked: false,
            cursor_delta: [0.0, 0.0],
            wheel_delta: 0.0,
        }
    }
}
