use crate::camera::Camera;
use crate::renderer::{OutputMode, RendererSettings, ToneMapMode, DEFAULT_BLOOM_STRENGTH};
use egui::{ClippedPrimitive, Context, TexturesDelta, Ui, ViewportId, Widget};
use egui_winit::State as EguiWinit;
use model::{metadata::*, PlaybackState};
use vulkan::winit::event::WindowEvent;
use vulkan::winit::window::Window as WinitWindow;

const SSAO_KERNEL_SIZES: [u32; 4] = [16, 32, 64, 128];
fn get_kernel_size_index(size: u32) -> usize {
    SSAO_KERNEL_SIZES
        .iter()
        .position(|&v| v == size)
        .unwrap_or_else(|| {
            panic!(
                "Illegal kernel size {:?}. Should be one of {:?}",
                size, SSAO_KERNEL_SIZES
            )
        })
}

pub struct RenderData {
    pub pixels_per_point: f32,
    pub textures_delta: TexturesDelta,
    pub clipped_primitives: Vec<ClippedPrimitive>,
}

pub struct Gui {
    egui: Context,
    egui_winit: EguiWinit,
    model_metadata: Option<Metadata>,
    animation_playback_state: Option<PlaybackState>,
    camera: Option<Camera>,
    state: State,
}

impl Gui {
    pub fn new(window: &WinitWindow, renderer_settings: RendererSettings) -> Self {
        let (egui, egui_winit) = init_egui(window);

        Self {
            egui,
            egui_winit,
            model_metadata: None,
            animation_playback_state: None,
            camera: None,
            state: State::new(renderer_settings),
        }
    }

    pub fn handle_event(&mut self, window: &WinitWindow, event: &WindowEvent) {
        let _ = self.egui_winit.on_window_event(window, event);
    }

    pub fn render(&mut self, window: &WinitWindow) -> RenderData {
        let raw_input = self.egui_winit.take_egui_input(window);

        let previous_state = self.state;

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.egui.run(raw_input, |ctx: &Context| {
            egui::Window::new("Menu")
                .default_open(false)
                .show(ctx, |ui| {
                    build_renderer_settings_window(ui, &mut self.state);
                    ui.separator();
                    build_camera_details_window(ui, &mut self.state, self.camera);
                    ui.separator();
                    build_animation_player_window(
                        ui,
                        &mut self.state,
                        self.model_metadata.as_ref(),
                        self.animation_playback_state,
                    );
                });
        });

        self.state.check_renderer_settings_changed(&previous_state);

        self.state.hovered = self.egui.is_pointer_over_area();

        self.egui_winit
            .handle_platform_output(window, platform_output);

        let clipped_primitives = self.egui.tessellate(shapes, pixels_per_point);

        RenderData {
            pixels_per_point,
            textures_delta,
            clipped_primitives,
        }
    }

    pub fn set_model_metadata(&mut self, metadata: Metadata) {
        self.model_metadata.replace(metadata);
        self.animation_playback_state = None;
        self.state = self.state.reset();
    }

    pub fn set_animation_playback_state(
        &mut self,
        animation_playback_state: Option<PlaybackState>,
    ) {
        self.animation_playback_state = animation_playback_state;
    }

    pub fn set_camera(&mut self, camera: Option<Camera>) {
        self.camera = camera;
    }

    pub fn get_selected_animation(&self) -> usize {
        self.state.selected_animation
    }

    pub fn is_infinite_animation_checked(&self) -> bool {
        self.state.infinite_animation
    }

    pub fn should_toggle_animation(&self) -> bool {
        self.state.toggle_animation
    }

    pub fn should_stop_animation(&self) -> bool {
        self.state.stop_animation
    }

    pub fn should_reset_animation(&self) -> bool {
        self.state.reset_animation
    }

    pub fn get_animation_speed(&self) -> f32 {
        self.state.animation_speed
    }

    pub fn should_reset_camera(&self) -> bool {
        self.state.reset_camera
    }

    pub fn get_new_renderer_settings(&self) -> Option<RendererSettings> {
        if self.state.renderer_settings_changed {
            Some(RendererSettings {
                hdr_enabled: self.state.hdr_enabled,
                emissive_intensity: self.state.emissive_intensity,
                ssao_enabled: self.state.ssao_enabled,
                ssao_kernel_size: SSAO_KERNEL_SIZES[self.state.ssao_kernel_size_index],
                ssao_radius: self.state.ssao_radius,
                ssao_strength: self.state.ssao_strength,
                tone_map_mode: ToneMapMode::from_value(self.state.selected_tone_map_mode)
                    .expect("Unknown tone map mode"),
                output_mode: OutputMode::from_value(self.state.selected_output_mode)
                    .expect("Unknown outpout mode"),
                bloom_strength: self.state.bloom_strength as f32 / 100f32,
            })
        } else {
            None
        }
    }

    pub fn is_hovered(&self) -> bool {
        self.state.hovered
    }
}

fn init_egui(window: &WinitWindow) -> (Context, EguiWinit) {
    let egui = Context::default();
    let egui_winit = EguiWinit::new(egui.clone(), ViewportId::ROOT, &window, None, None);

    (egui, egui_winit)
}

fn build_animation_player_window(
    ui: &mut Ui,
    state: &mut State,
    model_metadata: Option<&Metadata>,
    animation_playback_state: Option<PlaybackState>,
) {
    egui::CollapsingHeader::new("Animation player")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(metadata) = model_metadata {
                let animations_labels = metadata
                    .animations()
                    .iter()
                    .map(|a| {
                        let name = a.name.as_ref().map_or("no name", |n| n);
                        format!("{}: {}", a.index, name)
                    })
                    .collect::<Vec<_>>();

                egui::ComboBox::from_label("Animation").show_index(
                    ui,
                    &mut state.selected_animation,
                    metadata.animation_count(),
                    |i| animations_labels[i].clone(),
                );
            }

            if let Some(playback_state) = animation_playback_state {
                let toggle_text = if playback_state.paused {
                    "Resume"
                } else {
                    "Pause"
                };

                ui.horizontal(|ui| {
                    state.toggle_animation = ui.button(toggle_text).clicked();
                    state.stop_animation = ui.button("Stop").clicked();
                    state.reset_animation = ui.button("Reset").clicked();
                    ui.checkbox(&mut state.infinite_animation, "Loop");
                });

                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut state.animation_speed, 0.05..=3.0).text("Speed"));
                    if ui.button("Default").clicked() {
                        state.animation_speed = 1.0;
                    }
                });

                let progress = playback_state.time / playback_state.total_time;
                egui::ProgressBar::new(progress).ui(ui);
            }
        });
}

fn build_camera_details_window(ui: &mut Ui, state: &mut State, camera: Option<Camera>) {
    egui::CollapsingHeader::new("Camera")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(camera) = camera {
                let p = camera.position();
                let t = camera.target();
                ui.label(format!("Position: {:.3}, {:.3}, {:.3}", p.x, p.y, p.z));
                ui.label(format!("Target: {:.3}, {:.3}, {:.3}", t.x, t.y, t.z));
                state.reset_camera = ui.button("Reset").clicked();
            }
        });
}

fn build_renderer_settings_window(ui: &mut Ui, state: &mut State) {
    egui::CollapsingHeader::new("Renderer settings")
        .default_open(true)
        .show(ui, |ui| {
            {
                ui.heading("Settings");
                ui.separator();

                ui.add_enabled_ui(state.hdr_enabled.is_some(), |ui| {
                    if let Some(hdr_enabled) = state.hdr_enabled.as_mut() {
                        ui.checkbox(hdr_enabled, "Enable HDR");
                    }
                });

                ui.add(
                    egui::Slider::new(&mut state.emissive_intensity, 1.0..=200.0)
                        .text("Emissive intensity")
                        .integer(),
                );
                ui.add(
                    egui::Slider::new(&mut state.bloom_strength, 0..=10)
                        .text("Bloom strength")
                        .integer(),
                );

                ui.checkbox(&mut state.ssao_enabled, "Enable SSAO");
                if state.ssao_enabled {
                    egui::ComboBox::from_label("SSAO Kernel").show_index(
                        ui,
                        &mut state.ssao_kernel_size_index,
                        SSAO_KERNEL_SIZES.len(),
                        |i| SSAO_KERNEL_SIZES[i].to_string(),
                    );
                    ui.add(
                        egui::Slider::new(&mut state.ssao_radius, 0.01..=1.0).text("SSAO Radius"),
                    );
                    ui.add(
                        egui::Slider::new(&mut state.ssao_strength, 0.5..=5.0)
                            .text("SSAO Strength"),
                    );
                }
            }

            {
                ui.heading("Post Processing");
                ui.separator();

                let tone_map_modes = ToneMapMode::all();
                egui::ComboBox::from_label("Tone map mode").show_index(
                    ui,
                    &mut state.selected_tone_map_mode,
                    tone_map_modes.len(),
                    |i| format!("{:?}", tone_map_modes[i]),
                );
            }

            {
                ui.heading("Debug");
                ui.separator();

                let output_modes = OutputMode::all();
                egui::ComboBox::from_label("Output mode").show_index(
                    ui,
                    &mut state.selected_output_mode,
                    output_modes.len(),
                    |i| format!("{:?}", output_modes[i]),
                );
            }
        });
}

#[derive(Clone, Copy)]
struct State {
    selected_animation: usize,
    infinite_animation: bool,
    reset_animation: bool,
    toggle_animation: bool,
    stop_animation: bool,
    animation_speed: f32,

    reset_camera: bool,

    hdr_enabled: Option<bool>,
    selected_output_mode: usize,
    selected_tone_map_mode: usize,
    emissive_intensity: f32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
    ssao_kernel_size_index: usize,
    bloom_strength: u32,
    renderer_settings_changed: bool,

    hovered: bool,
}

impl State {
    fn new(renderer_settings: RendererSettings) -> Self {
        Self {
            hdr_enabled: renderer_settings.hdr_enabled,
            selected_output_mode: renderer_settings.output_mode as _,
            selected_tone_map_mode: renderer_settings.tone_map_mode as _,
            emissive_intensity: renderer_settings.emissive_intensity,
            ssao_enabled: renderer_settings.ssao_enabled,
            ssao_radius: renderer_settings.ssao_radius,
            ssao_strength: renderer_settings.ssao_strength,
            ssao_kernel_size_index: get_kernel_size_index(renderer_settings.ssao_kernel_size),
            ..Default::default()
        }
    }

    fn reset(&self) -> Self {
        Self {
            hdr_enabled: self.hdr_enabled,
            selected_output_mode: self.selected_output_mode,
            selected_tone_map_mode: self.selected_tone_map_mode,
            emissive_intensity: self.emissive_intensity,
            ssao_radius: self.ssao_radius,
            ssao_strength: self.ssao_strength,
            ssao_kernel_size_index: self.ssao_kernel_size_index,
            ssao_enabled: self.ssao_enabled,
            ..Default::default()
        }
    }

    fn check_renderer_settings_changed(&mut self, other: &Self) {
        self.renderer_settings_changed = self.hdr_enabled != other.hdr_enabled
            || self.selected_output_mode != other.selected_output_mode
            || self.selected_tone_map_mode != other.selected_tone_map_mode
            || self.emissive_intensity != other.emissive_intensity
            || self.ssao_enabled != other.ssao_enabled
            || self.ssao_radius != other.ssao_radius
            || self.ssao_strength != other.ssao_strength
            || self.ssao_kernel_size_index != other.ssao_kernel_size_index
            || self.bloom_strength != other.bloom_strength;
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected_animation: 0,
            infinite_animation: true,
            reset_animation: false,
            toggle_animation: false,
            stop_animation: false,
            animation_speed: 1.0,

            reset_camera: false,

            hdr_enabled: None,
            selected_output_mode: 0,
            selected_tone_map_mode: 0,
            emissive_intensity: 1.0,
            ssao_enabled: true,
            ssao_radius: 0.15,
            ssao_strength: 1.0,
            ssao_kernel_size_index: 1,
            bloom_strength: (DEFAULT_BLOOM_STRENGTH * 100f32) as _,
            renderer_settings_changed: false,

            hovered: false,
        }
    }
}
