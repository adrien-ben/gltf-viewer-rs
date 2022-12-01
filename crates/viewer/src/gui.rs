use crate::camera::Camera;
use crate::renderer::{OutputMode, RendererSettings, ToneMapMode, DEFAULT_BLOOM_STRENGTH};
use imgui::*;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use model::{metadata::*, PlaybackState};
use std::borrow::Cow;
use std::time::Instant;
use vulkan::winit::{event::Event, window::Window as WinitWindow};

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

pub struct Gui {
    context: Context,
    winit_platform: WinitPlatform,
    last_frame_instant: Instant,
    model_metadata: Option<Metadata>,
    animation_playback_state: Option<PlaybackState>,
    camera: Option<Camera>,
    state: State,
}

impl Gui {
    pub fn new(window: &WinitWindow, renderer_settings: RendererSettings) -> Self {
        let (context, winit_platform) = init_imgui(window);

        Self {
            context,
            winit_platform,
            last_frame_instant: Instant::now(),
            model_metadata: None,
            animation_playback_state: None,
            camera: None,
            state: State::new(renderer_settings),
        }
    }

    pub fn handle_event(&mut self, window: &WinitWindow, event: &Event<()>) {
        let io = self.context.io_mut();
        let platform = &mut self.winit_platform;

        platform.handle_event(io, window, event);
    }

    pub fn update_delta_time(&mut self) {
        let io = self.context.io_mut();
        let now = Instant::now();
        io.update_delta_time(now - self.last_frame_instant);
        self.last_frame_instant = now;
    }

    pub fn prepare_frame(&mut self, window: &WinitWindow) {
        let io = self.context.io_mut();
        let platform = &mut self.winit_platform;
        platform.prepare_frame(io, window).unwrap();
    }

    pub fn render(&mut self, window: &WinitWindow) -> &DrawData {
        let ui = self.context.frame();

        {
            let ui = &ui;

            ui.window("Menu")
                .collapsed(true, Condition::FirstUseEver)
                .position([0.0, 0.0], Condition::Always)
                .size([350.0, 800.0], Condition::FirstUseEver)
                .focus_on_appearing(false)
                .movable(false)
                .bg_alpha(0.3)
                .build(|| {
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
            self.state.hovered = ui.is_any_item_hovered()
                || ui.is_window_hovered_with_flags(WindowHoveredFlags::ANY_WINDOW);
        }

        self.winit_platform.prepare_render(&ui, window);
        self.context.render()
    }

    pub fn get_context(&mut self) -> &mut Context {
        &mut self.context
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

fn init_imgui(window: &WinitWindow) -> (Context, WinitPlatform) {
    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_bytes!("../../../assets/fonts/mplus-1p-regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                rasterizer_multiply: 1.75,
                glyph_ranges: FontGlyphRanges::default(),
                ..FontConfig::default()
            }),
        },
    ]);
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
    platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

    (imgui, platform)
}

fn build_animation_player_window(
    ui: &Ui,
    state: &mut State,
    model_metadata: Option<&Metadata>,
    animation_playback_state: Option<PlaybackState>,
) {
    if CollapsingHeader::new("Animation player").build(ui) {
        if let Some(metadata) = model_metadata {
            let animations_labels = metadata
                .animations()
                .iter()
                .map(|a| {
                    let name = a.name.as_ref().map_or("no name", |n| n);
                    format!("{}: {}", a.index, name)
                })
                .collect::<Vec<_>>();
            let combo_labels = animations_labels.iter().collect::<Vec<_>>();
            ui.combo_simple_string("Animation", &mut state.selected_animation, &combo_labels);

            if let Some(playback_state) = animation_playback_state {
                let toggle_text = if playback_state.paused {
                    "Resume"
                } else {
                    "Pause"
                };

                state.toggle_animation = ui.button(toggle_text);
                ui.same_line();
                state.stop_animation = ui.button("Stop");
                ui.same_line();
                state.reset_animation = ui.button("Reset");
                ui.same_line();
                ui.checkbox("Loop", &mut state.infinite_animation);

                ui.slider("Speed", 0.05, 3.0, &mut state.animation_speed);
                ui.same_line();
                if ui.button("Default") {
                    state.animation_speed = 1.0;
                }

                let progress = playback_state.time / playback_state.total_time;
                ProgressBar::new(progress).build(ui);
            }
        }
    }
}

fn build_camera_details_window(ui: &Ui, state: &mut State, camera: Option<Camera>) {
    if CollapsingHeader::new("Camera").build(ui) {
        if let Some(camera) = camera {
            let p = camera.position();
            let t = camera.target();
            ui.text(format!("Position: {:.3}, {:.3}, {:.3}", p.x, p.y, p.z));
            ui.text(format!("Target: {:.3}, {:.3}, {:.3}", t.x, t.y, t.z));
            state.reset_camera = ui.button("Reset");
        }
    }
}

fn build_renderer_settings_window(ui: &Ui, state: &mut State) {
    if CollapsingHeader::new("Renderer settings").build(ui) {
        {
            ui.text("Settings");
            ui.separator();

            let emissive_intensity_changed = ui.slider(
                "Emissive intensity",
                1.0,
                200.0,
                &mut state.emissive_intensity,
            );
            state.renderer_settings_changed = emissive_intensity_changed;

            let bloom_strength_changed =
                ui.slider("Bloom strength", 0u32, 10, &mut state.bloom_strength);
            state.renderer_settings_changed |= bloom_strength_changed;

            state.renderer_settings_changed |= ui.checkbox("Enable SSAO", &mut state.ssao_enabled);
            if state.ssao_enabled {
                let ssao_kernel_size_changed = ui.combo(
                    "SSAO Kernel",
                    &mut state.ssao_kernel_size_index,
                    &SSAO_KERNEL_SIZES,
                    |v| Cow::Owned(format!("{} samples", v)),
                );
                state.renderer_settings_changed |= ssao_kernel_size_changed;

                let ssao_radius_changed =
                    ui.slider("SSAO Radius", 0.01, 1.0, &mut state.ssao_radius);
                state.renderer_settings_changed |= ssao_radius_changed;

                let ssao_strength_changed =
                    ui.slider("SSAO Strength", 0.5, 5.0, &mut state.ssao_strength);
                state.renderer_settings_changed |= ssao_strength_changed;
            }
        }

        {
            ui.text("Post Processing");
            ui.separator();

            let tone_map_mode_changed = ui.combo(
                "Tone Map mode",
                &mut state.selected_tone_map_mode,
                &ToneMapMode::all(),
                |mode| Cow::Owned(format!("{:?}", mode)),
            );

            state.renderer_settings_changed |= tone_map_mode_changed;
        }

        {
            ui.text("Debug");
            ui.separator();

            let output_mode_changed = ui.combo(
                "Output mode",
                &mut state.selected_output_mode,
                &OutputMode::all(),
                |mode| Cow::Owned(format!("{:?}", mode)),
            );
            state.renderer_settings_changed |= output_mode_changed;
        }
    }
}

struct State {
    selected_animation: usize,
    infinite_animation: bool,
    reset_animation: bool,
    toggle_animation: bool,
    stop_animation: bool,
    animation_speed: f32,

    reset_camera: bool,

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
