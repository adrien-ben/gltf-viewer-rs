use winit::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};

#[derive(Copy, Clone, Debug)]
pub struct InputState {
    is_left_clicked: bool,
    cursor_position: [i32; 2],
    cursor_delta: Option<[i32; 2]>,
    wheel_delta: Option<f32>,
}

impl InputState {
    pub fn update(self, event: &Event) -> Self {
        let mut is_left_clicked = None;
        let mut cursor_position = None;
        let mut wheel_delta = None;

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
                WindowEvent::CursorMoved { position, .. } => {
                    let position: (i32, i32) = (*position).into();
                    cursor_position = Some([position.0, position.1]);
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, v_lines),
                    ..
                } => {
                    wheel_delta = Some(*v_lines);
                }
                _ => {}
            }
        }

        Self {
            is_left_clicked: is_left_clicked.unwrap_or(self.is_left_clicked),
            cursor_position: cursor_position.unwrap_or(self.cursor_position),
            cursor_delta: cursor_position.map(|position| {
                [
                    position[0] - self.cursor_position[0],
                    position[1] - self.cursor_position[1],
                ]
            }),
            wheel_delta,
        }
    }

    pub fn reset(&mut self) {
        self.cursor_delta = None;
        self.wheel_delta = None;
    }
}

impl InputState {
    pub fn is_left_clicked(&self) -> bool {
        self.is_left_clicked
    }

    pub fn cursor_delta(&self) -> Option<[i32; 2]> {
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
            cursor_position: [0, 0],
            cursor_delta: None,
            wheel_delta: None,
        }
    }
}
