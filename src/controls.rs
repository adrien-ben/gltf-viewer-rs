use vulkan::winit::{
    dpi::LogicalPosition, DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, Touch,
    TouchPhase, WindowEvent,
};

#[derive(Copy, Clone, Debug)]
pub struct InputState {
    is_left_clicked: bool,
    cursor_delta: [f32; 2],
    last_touch_position: [f32; 2],
    wheel_delta: Option<f32>,
}

impl InputState {
    pub fn update(self, event: &Event) -> Self {
        let mut is_left_clicked = None;
        let mut wheel_delta = None;
        let mut cursor_delta = self.cursor_delta;
        let mut last_touch_position = self.last_touch_position;

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
                    wheel_delta = Some(*v_lines);
                }
                WindowEvent::Touch(Touch {
                    location: LogicalPosition { x, y },
                    phase,
                    ..
                }) => {
                    let x = *x as f32;
                    let y = *y as f32;

                    if *phase == TouchPhase::Started {
                        is_left_clicked = Some(true);
                        last_touch_position = [x, y];
                    } else if *phase == TouchPhase::Ended {
                        is_left_clicked = Some(false);
                    }

                    cursor_delta[0] += x - last_touch_position[0];
                    cursor_delta[1] += y - last_touch_position[1];
                    last_touch_position = [x, y];
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
            last_touch_position,
            wheel_delta,
        }
    }

    pub fn reset(&mut self) {
        self.cursor_delta = [0.0, 0.0];
        self.wheel_delta = None;
    }
}

impl InputState {
    pub fn is_left_clicked(&self) -> bool {
        self.is_left_clicked
    }

    pub fn cursor_delta(&self) -> [f32; 2] {
        self.cursor_delta
    }

    pub fn wheel_delta(&self) -> Option<f32> {
        self.wheel_delta
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            is_left_clicked: false,
            cursor_delta: [0.0, 0.0],
            last_touch_position: [0.0, 0.0],
            wheel_delta: None,
        }
    }
}
