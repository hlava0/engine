//! Lightweight runtime scene inspector and entity manipulator.

use hecs::{Entity, World};

use crate::ecs::{Name, Transform};

/// Runtime inspector state for selecting, summarizing, and nudging ECS entities.
#[derive(Debug, Clone)]
pub struct SceneEditor {
    pub selected_index: usize,
    pub paused: bool,
    pub show_physics: bool,
}

impl Default for SceneEditor {
    fn default() -> Self {
        Self {
            selected_index: 0,
            paused: false,
            show_physics: true,
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
}
