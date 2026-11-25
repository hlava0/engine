//! Visual `egui` Scene Hierarchy & Property Inspector + keyboard entity helpers.

use hecs::{Entity, World};

use crate::ecs::{Name, RigidBody, ScriptComponent, Transform};

/// Runtime scene editor & `egui` property inspector.
#[derive(Debug, Clone)]
pub struct SceneEditor {
    pub selected_index: usize,
    pub paused: bool,
    pub show_physics: bool,
    pub show_editor_ui: bool,
}

impl Default for SceneEditor {
    fn default() -> Self {
        Self {
            selected_index: 0,
            paused: false,
            show_physics: true,
            show_editor_ui: true,
        }
    }
}

impl SceneEditor {
    pub fn new() -> Self {
        Self::default()
    }

    fn sorted_entities(world: &World) -> Vec<Entity> {
        let mut entities: Vec<Entity> = world.query::<&Transform>().iter().map(|(e, _)| e).collect();
        entities.sort();
        entities
    }

    /// Cycle selection to the next entity with a [`Transform`].
    pub fn select_next(&mut self, world: &World) {
        let count = Self::sorted_entities(world).len();
        if count == 0 {
            self.selected_index = 0;
        } else {
            self.selected_index = (self.selected_index + 1) % count;
        }
    }

    /// Return a concise human-readable summary of the currently selected entity.
    pub fn selected_summary(&self, world: &World) -> String {
        let entities = Self::sorted_entities(world);
        if entities.is_empty() {
            return "None".to_string();
        }
        let idx = self.selected_index % entities.len();
        let entity = entities[idx];
        let name = world
            .get::<&Name>(entity)
            .map(|n| n.0.clone())
            .unwrap_or_else(|_| format!("Entity#{}", entity.id()));
        if let Ok(t) = world.get::<&Transform>(entity) {
            format!(
                "{name} ({:.2}, {:.2}, {:.2})",
                t.position[0], t.position[1], t.position[2]
            )
        } else {
            name
        }
    }

    /// Translate the currently selected entity by `delta`, returning `true` if mutated.
    pub fn nudge_selected(&self, world: &mut World, delta: [f32; 3]) -> bool {
        let entities = Self::sorted_entities(world);
        if entities.is_empty() {
            return false;
        }
        let idx = self.selected_index % entities.len();
        let entity = entities[idx];
        if let Ok(mut t) = world.get::<&mut Transform>(entity) {
            t.position[0] += delta[0];
            t.position[1] += delta[1];
            t.position[2] += delta[2];
            true
        } else {
            false
        }
    }

    /// Toggle physics/simulation pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Draw the visual `egui` Scene Hierarchy & Property Inspector window.
    pub fn draw_ui(&mut self, ctx: &egui::Context, world: &mut World) {
        if !self.show_editor_ui {
            return;
        }

        let entities = Self::sorted_entities(world);
        egui::SidePanel::left("scene_editor_panel")
            .default_width(270.0)
            .show(ctx, |ui| {
                ui.heading("Scene Inspector");
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "Resume" } else { "Pause" })
                        .clicked()
                    {
                        self.toggle_pause();
                    }
                    ui.checkbox(&mut self.show_physics, "Physics (Rapier3D)");
                });
                ui.separator();

                ui.label(format!("Entities ({})", entities.len()));
                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, &entity) in entities.iter().enumerate() {
                            let label = world
                                .get::<&Name>(entity)
                                .map(|n| n.0.clone())
                                .unwrap_or_else(|_| format!("Entity#{}", entity.id()));
                            let selected = idx == (self.selected_index % entities.len().max(1));
                            if ui.selectable_label(selected, label).clicked() {
                                self.selected_index = idx;
                            }
                        }
                    });

                ui.separator();
                if let Some(&entity) = entities.get(self.selected_index % entities.len().max(1)) {
                    if let Ok(mut t) = world.get::<&mut Transform>(entity) {
                        ui.label("Transform");
                        ui.horizontal(|ui| {
                            ui.label("Pos");
                            ui.add(egui::DragValue::new(&mut t.position[0]).speed(0.05));
                            ui.add(egui::DragValue::new(&mut t.position[1]).speed(0.05));
                            ui.add(egui::DragValue::new(&mut t.position[2]).speed(0.05));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Rot");
                            ui.add(egui::DragValue::new(&mut t.rotation_euler[0]).speed(0.02));
                            ui.add(egui::DragValue::new(&mut t.rotation_euler[1]).speed(0.02));
                            ui.add(egui::DragValue::new(&mut t.rotation_euler[2]).speed(0.02));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Scale");
                            ui.add(egui::DragValue::new(&mut t.scale[0]).speed(0.05));
                            ui.add(egui::DragValue::new(&mut t.scale[1]).speed(0.05));
                            ui.add(egui::DragValue::new(&mut t.scale[2]).speed(0.05));
                        });
                    }

                    if let Ok(mut rb) = world.get::<&mut RigidBody>(entity) {
                        ui.separator();
                        ui.label("RigidBody");
                        ui.horizontal(|ui| {
                            ui.label("Vel");
                            ui.add(egui::DragValue::new(&mut rb.velocity[0]).speed(0.1));
                            ui.add(egui::DragValue::new(&mut rb.velocity[1]).speed(0.1));
                            ui.add(egui::DragValue::new(&mut rb.velocity[2]).speed(0.1));
                        });
                        ui.add(
                            egui::Slider::new(&mut rb.restitution, 0.0..=1.0).text("Restitution"),
                        );
                        ui.checkbox(&mut rb.use_gravity, "Use Gravity");
                    }

                    if let Ok(mut sc) = world.get::<&mut ScriptComponent>(entity) {
                        ui.separator();
                        ui.label("Rhai Script");
                        ui.text_edit_multiline(&mut sc.source);
                    }
                }
            });
    }
}
