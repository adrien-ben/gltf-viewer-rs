use crate::camera::Camera;
use crate::renderer::{OutputMode, RendererSettings, ToneMapMode};
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
        let mut io = self.context.io_mut();
        let platform = &mut self.winit_platform;

        platform.handle_event(&mut io, window, event);
    }

    pub fn update_delta_time(&mut self) {
        let io = self.context.io_mut();
        self.last_frame_instant = io.update_delta_time(self.last_frame_instant);
    }

    pub fn prepare_frame(&mut self, window: &WinitWindow) {
        let io = self.context.io_mut();
        let platform = &mut self.winit_platform;
        platform.prepare_frame(io, &window).unwrap();
    }

    pub fn render(&mut self, window: &WinitWindow) -> &DrawData {
        let ui = self.context.frame();

        {
            let ui = &ui;

            build_main_menu_bar(ui, &mut self.state);

            if self.state.show_model_descriptor {
                build_model_descriptor_window(ui, &mut self.state, self.model_metadata.as_ref());
            }

            if self.state.show_animation_player {
                build_animation_player_window(
                    ui,
                    &mut self.state,
                    self.model_metadata.as_ref(),
                    self.animation_playback_state,
                );
            }

            if self.state.show_camera_details {
                build_camera_details_window(ui, &mut self.state, self.camera);
            }

            if self.state.show_renderer_settings {
                build_renderer_settings_window(ui, &mut self.state);
            }

            self.state.hovered = ui.is_any_item_hovered()
                || ui.is_window_hovered_with_flags(WindowHoveredFlags::ANY_WINDOW);
        }

        self.winit_platform.prepare_render(&ui, window);
        ui.render()
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
            data: include_bytes!("../assets/fonts/mplus-1p-regular.ttf"),
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

fn build_main_menu_bar(ui: &Ui, state: &mut State) {
    ui.main_menu_bar(|| {
        ui.menu(im_str!("View"), true, || {
            MenuItem::new(im_str!("Model descriptor"))
                .build_with_ref(ui, &mut state.show_model_descriptor);
            MenuItem::new(im_str!("Animation player"))
                .build_with_ref(ui, &mut state.show_animation_player);
            MenuItem::new(im_str!("Camera details"))
                .build_with_ref(ui, &mut state.show_camera_details);
            MenuItem::new(im_str!("Renderer settings"))
                .build_with_ref(ui, &mut state.show_renderer_settings);
        });
    });
}

fn build_model_descriptor_window(ui: &Ui, state: &mut State, model_metadata: Option<&Metadata>) {
    let mut opened = true;
    Window::new(im_str!("Model metadata"))
        .position([20.0, 20.0], Condition::Appearing)
        .size([400.0, 400.0], Condition::Appearing)
        .collapsible(false)
        .opened(&mut opened)
        .build(ui, || {
            let parent_size = ui.window_size();

            ChildWindow::new(0)
                .size([parent_size[0] / 3.0, 0.0])
                .build(ui, || {
                    if let Some(metadata) = model_metadata {
                        build_summary_block_ui(ui, metadata);
                        build_hierarchy_block_ui(ui, metadata, state);
                        build_animation_block_ui(ui, metadata);
                    }
                });

            ui.same_line(0.0);

            if let Some(selected_hierarchy_node) = state.selected_hierarchy_node.as_ref() {
                ChildWindow::new(im_str!("Node details"))
                    .border(true)
                    .build(ui, || build_node_details_ui(ui, selected_hierarchy_node));
            }
        });
    state.show_model_descriptor = opened;
}

fn build_summary_block_ui(ui: &Ui, metadata: &Metadata) {
    if CollapsingHeader::new(&im_str!("Summary"))
        .default_open(true)
        .build(ui)
    {
        ui.text(im_str!("Name: {}", metadata.name()));
        if ui.is_item_hovered() {
            ui.tooltip_text(im_str!("{}", metadata.path()));
        }
        ui.text(im_str!("Scene count: {}", metadata.scene_count()));
        ui.text(im_str!("Node count: {}", metadata.node_count()));
        ui.text(im_str!("Animation count: {}", metadata.animation_count()));
        ui.text(im_str!("Skin count: {}", metadata.skin_count()));
        ui.text(im_str!("Mesh count: {}", metadata.mesh_count()));
        ui.text(im_str!("Material count: {}", metadata.material_count()));
        ui.text(im_str!("Texture count: {}", metadata.texture_count()));
        ui.text(im_str!("Light count: {}", metadata.light_count()));
    }
}

fn build_hierarchy_block_ui(ui: &Ui, metadata: &Metadata, state: &mut State) {
    if CollapsingHeader::new(&im_str!("Hierarchy")).build(ui) {
        for node in metadata.nodes() {
            build_tree_node_ui(ui, node, state);
        }
    }
}

fn build_tree_node_ui(ui: &Ui, node: &Node, state: &mut State) {
    let selected = state
        .selected_hierarchy_node
        .as_ref()
        .map_or(false, |n| n.uid == node.uid());
    let name = node.name().unwrap_or("no name");

    // This flag is there tu make sure we attach the "is_click" to the correct node/leaf
    let mut opened = false;
    TreeNode::new(&im_str!("{}: {}", node.index(), name))
        .leaf(node.children().is_empty())
        .open_on_double_click(true)
        .open_on_arrow(true)
        .selected(selected)
        .build(ui, || {
            // If the node is opened the flag will be set to true
            opened = true;
            if ui.is_item_clicked(MouseButton::Left) {
                state.selected_hierarchy_node = Some(NodeDetails::from(node));
            }
            node.children()
                .iter()
                .for_each(|n| build_tree_node_ui(ui, n, state))
        });

    // If the was not opened then we still want to attach the "is_click"
    if !opened && ui.is_item_clicked(MouseButton::Left) {
        state.selected_hierarchy_node = Some(NodeDetails::from(node));
    }
}

fn build_node_details_ui(ui: &Ui, node_details: &NodeDetails) {
    let title = match node_details.kind {
        NodeKind::Scene => "Scene",
        NodeKind::Node(NodeData { root: true, .. }) => "Root node",
        NodeKind::Node(NodeData { leaf: false, .. }) => "Node",
        NodeKind::Node(NodeData { leaf: true, .. }) => "Leaf",
    };

    ui.text(title);
    ui.separator();
    ui.text(im_str!("Index: {}", node_details.index));
    ui.text(im_str!("Name: {}", node_details.name));
    if let NodeKind::Scene | NodeKind::Node(NodeData { leaf: false, .. }) = node_details.kind {
        ui.text(im_str!("Child count: {}", node_details.child_count));
    }

    if let NodeKind::Node(NodeData {
        mesh: Some(ref mesh),
        ..
    }) = node_details.kind
    {
        build_mesh_details_ui(ui, mesh);
    }

    if let NodeKind::Node(NodeData {
        light: Some(light), ..
    }) = node_details.kind
    {
        build_light_details_ui(ui, light);
    }
}

fn build_mesh_details_ui(ui: &Ui, mesh_data: &Mesh) {
    ui.text("Mesh");
    ui.separator();
    ui.text(im_str!("Index: {}", mesh_data.index));
    ui.text(im_str!(
        "Name: {}",
        mesh_data.name.as_ref().map_or("no name", |s| &s)
    ));
    if CollapsingHeader::new(&im_str!("Primitives")).build(ui) {
        mesh_data
            .primitives
            .iter()
            .for_each(|p| build_primitive_ui(ui, p));
    }
}

fn build_primitive_ui(ui: &Ui, prim: &Primitive) {
    TreeNode::new(&im_str!("{}", prim.index))
        .open_on_double_click(true)
        .open_on_arrow(true)
        .build(ui, || {
            ui.text(im_str!("Mode: {}", prim.mode));
            ui.text("Material:");
            let material = &prim.material;
            ui.indent();
            if let Some(index) = material.index {
                ui.text(im_str!("Index {}", index));
            }
            ui.text(im_str!(
                "Name: {}",
                material.name.as_ref().map_or("no name", |s| &s)
            ));

            ui.text("Base color");
            ui.same_line(0.0);
            ColorButton::new(im_str!("Base color"), material.base_color).build(ui);
            ui.text(im_str!("Alpha mode: {}", material.alpha_mode));
            match material.alpha_mode {
                AlphaMode::Blend | AlphaMode::Mask => {
                    ui.text(im_str!("Alpha cutoff: {}", material.alpha_cutoff))
                }
                _ => (),
            };

            if material.unlit {
                ui.text("Unlit: true");
            } else {
                ui.text("Emissise color");
                ui.same_line(0.0);
                let emissive_color_rgba = [
                    material.emissive_color[0],
                    material.emissive_color[1],
                    material.emissive_color[2],
                    1.0,
                ];
                ColorButton::new(im_str!("Emissive color"), emissive_color_rgba).build(ui);

                ui.text(im_str!("Metalness: {}", material.metallic_factor));
                ui.text(im_str!("Roughness: {}", material.roughness_factor));
            }
        })
}

fn build_light_details_ui(ui: &Ui, light: Light) {
    ui.text("Light");
    ui.separator();

    let color_rgba = [light.color[0], light.color[1], light.color[2], 1.0];
    ui.text(im_str!("Type: {}", light.kind));
    ColorButton::new(im_str!("Color"), color_rgba).build(ui);
    ui.text(im_str!("Intensity: {}", light.intensity));
    ui.text(im_str!(
        "Range: {}",
        light
            .range
            .map_or(String::from("unlimited"), |i| i.to_string())
    ));
    if let LightKind::Spot {
        inner_cone_angle,
        outer_cone_angle,
    } = light.kind
    {
        ui.text(im_str!("Inner cone angle: {}", inner_cone_angle));
        ui.text(im_str!("Outter cone angle: {}", outer_cone_angle));
    }
}

fn build_animation_block_ui(ui: &Ui, metadata: &Metadata) {
    if CollapsingHeader::new(&im_str!("Animations")).build(ui) {
        ui.indent();
        for animation in metadata.animations() {
            let name = animation.name.as_ref().map_or("no name", |n| &n);
            ui.text(im_str!("{}: {}", animation.index, name));
        }
    }
}

fn build_animation_player_window(
    ui: &Ui,
    state: &mut State,
    model_metadata: Option<&Metadata>,
    animation_playback_state: Option<PlaybackState>,
) {
    let mut opened = true;
    Window::new(im_str!("Animation player"))
        .position([20.0, 20.0], Condition::Appearing)
        .size([400.0, 150.0], Condition::Appearing)
        .collapsible(false)
        .opened(&mut opened)
        .build(ui, || {
            if let Some(metadata) = model_metadata {
                let animations_labels = metadata
                    .animations()
                    .iter()
                    .map(|a| {
                        let name = a.name.as_ref().map_or("no name", |n| &n);
                        im_str!("{}: {}", a.index, name)
                    })
                    .collect::<Vec<_>>();
                let combo_labels = animations_labels.iter().map(|l| l).collect::<Vec<_>>();
                ComboBox::new(im_str!("Select animation")).build_simple_string(
                    ui,
                    &mut state.selected_animation,
                    &combo_labels,
                );

                if let Some(playback_state) = animation_playback_state {
                    let toggle_text = if playback_state.paused {
                        "Resume"
                    } else {
                        "Pause"
                    };

                    state.toggle_animation = ui.button(&im_str!("{}", toggle_text), [0.0, 0.0]);
                    ui.same_line(0.0);
                    state.stop_animation = ui.button(im_str!("Stop"), [0.0, 0.0]);
                    ui.same_line(0.0);
                    state.reset_animation = ui.button(im_str!("Reset"), [0.0, 0.0]);
                    ui.same_line(0.0);
                    ui.checkbox(im_str!("Loop"), &mut state.infinite_animation);

                    Slider::new(im_str!("Speed"), 0.05f32..=3.0)
                        .build(ui, &mut state.animation_speed);
                    ui.same_line(0.0);
                    if ui.button(&im_str!("Default"), [0.0, 0.0]) {
                        state.animation_speed = 1.0;
                    }

                    let progress = playback_state.time / playback_state.total_time;
                    ProgressBar::new(progress).build(ui);
                }
            }
        });
    state.show_animation_player = opened;
}

fn build_camera_details_window(ui: &Ui, state: &mut State, camera: Option<Camera>) {
    let mut opened = true;
    Window::new(im_str!("Camera"))
        .position([20.0, 20.0], Condition::Appearing)
        .size([250.0, 100.0], Condition::Appearing)
        .collapsible(false)
        .opened(&mut opened)
        .build(ui, || {
            if let Some(camera) = camera {
                let p = camera.position();
                let t = camera.target();
                ui.text(im_str!("Position: {:.3}, {:.3}, {:.3}", p.x, p.y, p.z));
                ui.text(im_str!("Target: {:.3}, {:.3}, {:.3}", t.x, t.y, t.z));
                state.reset_camera = ui.button(im_str!("Reset"), [0.0, 0.0]);
            }
        });
    state.show_camera_details = opened;
}

fn build_renderer_settings_window(ui: &Ui, state: &mut State) {
    let mut opened = true;
    Window::new(im_str!("Renderer settings"))
        .position([20.0, 20.0], Condition::Appearing)
        .always_auto_resize(true)
        .collapsible(false)
        .opened(&mut opened)
        .build(ui, || {
            {
                ui.text("Settings");
                ui.separator();

                let emissive_intensity_changed =
                    Slider::new(im_str!("Emissive intensity"), 1.0f32..=50.0)
                        .build(ui, &mut state.emissive_intensity);
                state.renderer_settings_changed = emissive_intensity_changed;

                state.renderer_settings_changed |=
                    ui.checkbox(im_str!("Enable SSAO"), &mut state.ssao_enabled);
                if state.ssao_enabled {
                    let ssao_kernel_size_changed = ComboBox::new(im_str!("SSAO Kernel"))
                        .build_simple(
                            ui,
                            &mut state.ssao_kernel_size_index,
                            &SSAO_KERNEL_SIZES,
                            &|v| Cow::Owned(im_str!("{} samples", v)),
                        );
                    state.renderer_settings_changed |= ssao_kernel_size_changed;

                    let ssao_radius_changed = Slider::new(im_str!("SSAO Radius"), 0.01f32..=1.0)
                        .build(ui, &mut state.ssao_radius);
                    state.renderer_settings_changed |= ssao_radius_changed;

                    let ssao_strength_changed = Slider::new(im_str!("SSAO Strength"), 0.5..=5.0f32)
                        .build(ui, &mut state.ssao_strength);
                    state.renderer_settings_changed |= ssao_strength_changed;
                }
            }

            {
                ui.text("Post Processing");
                ui.separator();

                let tone_map_mode_changed = ComboBox::new(im_str!("Tone Map mode")).build_simple(
                    ui,
                    &mut state.selected_tone_map_mode,
                    &ToneMapMode::all(),
                    &|mode| Cow::Owned(im_str!("{:?}", mode)),
                );

                state.renderer_settings_changed |= tone_map_mode_changed;
            }

            {
                ui.text("Debug");
                ui.separator();

                let output_mode_changed = ComboBox::new(im_str!("Output mode")).build_simple(
                    ui,
                    &mut state.selected_output_mode,
                    &OutputMode::all(),
                    &|mode| Cow::Owned(im_str!("{:?}", mode)),
                );
                state.renderer_settings_changed |= output_mode_changed;
            }
        });
    state.show_renderer_settings = opened;
}

struct State {
    show_model_descriptor: bool,
    selected_hierarchy_node: Option<NodeDetails>,

    show_animation_player: bool,
    selected_animation: usize,
    infinite_animation: bool,
    reset_animation: bool,
    toggle_animation: bool,
    stop_animation: bool,
    animation_speed: f32,

    show_camera_details: bool,
    reset_camera: bool,

    show_renderer_settings: bool,
    selected_output_mode: usize,
    selected_tone_map_mode: usize,
    emissive_intensity: f32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
    ssao_kernel_size_index: usize,
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
            show_model_descriptor: self.show_model_descriptor,
            show_animation_player: self.show_animation_player,
            show_camera_details: self.show_camera_details,
            show_renderer_settings: self.show_renderer_settings,
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
            show_model_descriptor: false,
            selected_hierarchy_node: None,

            show_animation_player: false,
            selected_animation: 0,
            infinite_animation: true,
            reset_animation: false,
            toggle_animation: false,
            stop_animation: false,
            animation_speed: 1.0,

            show_camera_details: false,
            reset_camera: false,

            show_renderer_settings: false,
            selected_output_mode: 0,
            selected_tone_map_mode: 0,
            emissive_intensity: 1.0,
            ssao_enabled: true,
            ssao_radius: 0.15,
            ssao_strength: 1.0,
            ssao_kernel_size_index: 1,
            renderer_settings_changed: false,

            hovered: false,
        }
    }
}

struct NodeDetails {
    uid: usize,
    index: usize,
    name: String,
    kind: NodeKind,
    child_count: usize,
}

impl From<&Node> for NodeDetails {
    fn from(node: &Node) -> Self {
        Self {
            uid: node.uid(),
            index: node.index(),
            name: node.name().map_or(String::from("no name"), String::from),
            kind: node.kind().clone(),
            child_count: node.children().len(),
        }
    }
}
