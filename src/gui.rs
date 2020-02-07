use imgui::*;
use model::{metadata::*, PlaybackState};

#[derive(Default)]
pub struct Gui {
    model_metadata: Option<Metadata>,
    animation_playback_state: Option<PlaybackState>,
    state: State,
}

impl Gui {
    pub fn update(&mut self, _run: &mut bool, ui: &mut Ui) {
        let window_alpha = if self.state.is_focused { 0.8 } else { 0.5 };
        Window::new(im_str!("Model metadata"))
            .position([20.0, 20.0], Condition::Appearing)
            .size([400.0, 400.0], Condition::Appearing)
            .movable(false)
            .bg_alpha(window_alpha)
            .build(ui, || {
                let parent_size = ui.window_size();

                ChildWindow::new(0)
                    .size([parent_size[0] / 3.0, 0.0])
                    .build(ui, || {
                        if let Some(metadata) = self.model_metadata.as_ref() {
                            build_summary_block_ui(ui, metadata);
                            build_hierarchy_block_ui(ui, metadata, &mut self.state);
                            build_animation_block_ui(ui, metadata);
                        }
                    });

                ui.same_line(0.0);

                if let Some(selected_hierarchy_node) = self.state.selected_hierarchy_node.as_ref() {
                    ChildWindow::new(im_str!("Node details"))
                        .border(true)
                        .build(ui, || build_node_details_ui(ui, selected_hierarchy_node));
                }

                self.state.is_focused =
                    ui.is_window_focused_with_flags(WindowFocusedFlags::CHILD_WINDOWS);
            });

        Window::new(im_str!("Animation player"))
            .position([440.0, 20.0], Condition::Appearing)
            .size([300.0, 100.0], Condition::Appearing)
            .always_auto_resize(true)
            .collapsed(true, Condition::Appearing)
            .movable(true)
            .build(ui, || {
                if let Some(metadata) = self.model_metadata.as_ref() {
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
                        &mut self.state.selected_animation,
                        &combo_labels,
                    );

                    if let Some(state) = self.animation_playback_state {
                        let toggle_text = match state.paused {
                            true => "Resume",
                            false => "Pause",
                        };

                        self.state.toggle_animation = ui.button(&im_str!("{}", toggle_text), [0.0, 0.0]);
                        ui.same_line(0.0);
                        self.state.stop_animation = ui.button(im_str!("Stop"), [0.0, 0.0]);
                        ui.same_line(0.0);
                        self.state.reset_animation = ui.button(im_str!("Reset"), [0.0, 0.0]);
                        ui.same_line(0.0);
                        ui.checkbox(im_str!("Loop"), &mut self.state.infinite_animation);

                        Slider::new(im_str!("Speed"), 0.05f32..=3.0)
                            .build(ui, &mut self.state.animation_speed);
                        ui.same_line(0.0);
                        if ui.button(&im_str!("Default"), [0.0, 0.0]) {
                            self.state.animation_speed = 1.0;
                        }

                        let progress = state.time / state.total_time;
                        ProgressBar::new(progress).build(ui);
                    }
                }
            });
    }

    pub fn set_model_metadata(&mut self, metadata: Metadata) {
        self.model_metadata.replace(metadata);
        self.animation_playback_state = None;
        self.state = Default::default();
    }

    pub fn set_animation_playback_state(
        &mut self,
        animation_playback_state: Option<PlaybackState>,
    ) {
        self.animation_playback_state = animation_playback_state;
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
}

fn build_summary_block_ui(ui: &Ui, metadata: &Metadata) {
    if ui
        .collapsing_header(&im_str!("Summary"))
        .default_open(true)
        .build()
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
    if ui.collapsing_header(&im_str!("Hierarchy")).build() {
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
    ui.tree_node(&im_str!("{}: {}", node.index(), name))
        .leaf(node.children().is_empty())
        .open_on_double_click(true)
        .open_on_arrow(true)
        .selected(selected)
        .build(|| {
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
    if ui.collapsing_header(&im_str!("Primitives")).build() {
        mesh_data
            .primitives
            .iter()
            .for_each(|p| build_primitive_ui(ui, p));
    }
}

fn build_primitive_ui(ui: &Ui, prim: &Primitive) {
    ui.tree_node(&im_str!("{}", prim.index)).build(|| {
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
    if ui.collapsing_header(&im_str!("Animations")).build() {
        ui.indent();
        for animation in metadata.animations() {
            let name = animation.name.as_ref().map_or("no name", |n| &n);
            ui.text(im_str!("{}: {}", animation.index, name));
        }
    }
}

struct State {
    is_focused: bool,
    selected_hierarchy_node: Option<NodeDetails>,
    selected_animation: usize,
    infinite_animation: bool,
    reset_animation: bool,
    toggle_animation: bool,
    stop_animation: bool,
    animation_speed: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            is_focused: true,
            selected_hierarchy_node: None,
            selected_animation: 0,
            infinite_animation: true,
            reset_animation: false,
            toggle_animation: false,
            stop_animation: false,
            animation_speed: 1.0,
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
