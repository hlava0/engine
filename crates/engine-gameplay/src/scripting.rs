//! `rhai` scripting runtime with AST compilation caching and full ECS binding
//! (`Transform`, `RigidBody` velocity/impulses, `Name`, `AudioSource`).

use std::collections::HashMap;

use hecs::{Entity, World};
use rhai::{AST, Engine as RhaiEngine, Scope};

use crate::ecs::{AudioSource, Name, RigidBody, ScriptComponent, Transform};

/// Compiles, caches, and executes entity [`ScriptComponent`]s against ECS components.
pub struct ScriptEngine {
    engine: RhaiEngine,
    cache: HashMap<String, AST>,
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = RhaiEngine::new();
        engine.set_max_operations(100_000);
        engine.register_fn("clamp", |v: f64, lo: f64, hi: f64| v.clamp(lo, hi));
        engine.register_fn("lerp", |a: f64, b: f64, t: f64| a + (b - a) * t);
        Self {
            engine,
            cache: HashMap::new(),
        }
    }

    /// Execute all [`ScriptComponent`]s attached to entities with a [`Transform`].
    ///
    /// Scripts receive:
    /// - `x, y, z, rx, ry, rz, sx, sy, sz` (`Transform`)
    /// - `vx, vy, vz, impulse_x, impulse_y, impulse_z` (`RigidBody` velocity & additive impulse)
    /// - `audio_freq, audio_trigger` (`AudioSource`)
    /// - `name, dt, time`
    pub fn step_scripts(&mut self, world: &mut World, dt: f32, time: f32) {
        let mut entities: Vec<Entity> = world
            .query::<(&Transform, &ScriptComponent)>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        entities.sort();

        for entity in entities {
            let source = match world.get::<&ScriptComponent>(entity) {
                Ok(s) => s.source.clone(),
                Err(_) => continue,
            };

            if !self.cache.contains_key(&source) {
                match self.engine.compile(&source) {
                    Ok(ast) => {
                        self.cache.insert(source.clone(), ast);
                    }
                    Err(err) => {
                        log::warn!("Rhai script compile error: {err}");
                        continue;
                    }
                }
            }
            let Some(ast) = self.cache.get(&source) else {
                continue;
            };

            let tf = match world.get::<&Transform>(entity) {
                Ok(t) => *t,
                Err(_) => continue,
            };
            let vel = world
                .get::<&RigidBody>(entity)
                .map(|rb| rb.velocity)
                .unwrap_or([0.0; 3]);
            let name = world
                .get::<&Name>(entity)
                .map(|n| n.0.clone())
                .unwrap_or_else(|_| format!("Entity#{}", entity.id()));
            let (audio_freq, audio_trigger) = world
                .get::<&AudioSource>(entity)
                .map(|a| (a.frequency_hz, a.triggered))
                .unwrap_or((440.0, false));

            let mut scope = Scope::new();
            scope.push("x", tf.position[0] as f64);
            scope.push("y", tf.position[1] as f64);
            scope.push("z", tf.position[2] as f64);
            scope.push("rx", tf.rotation_euler[0] as f64);
            scope.push("ry", tf.rotation_euler[1] as f64);
            scope.push("rz", tf.rotation_euler[2] as f64);
            scope.push("sx", tf.scale[0] as f64);
            scope.push("sy", tf.scale[1] as f64);
            scope.push("sz", tf.scale[2] as f64);
            scope.push("vx", vel[0] as f64);
            scope.push("vy", vel[1] as f64);
            scope.push("vz", vel[2] as f64);
            scope.push("impulse_x", 0.0_f64);
            scope.push("impulse_y", 0.0_f64);
            scope.push("impulse_z", 0.0_f64);
            scope.push("audio_freq", audio_freq as f64);
            scope.push("audio_trigger", audio_trigger);
            scope.push("name", name);
            scope.push("dt", dt as f64);
            scope.push("time", time as f64);

            if let Err(err) = self.engine.run_ast_with_scope(&mut scope, ast) {
                log::warn!("Rhai script runtime error: {err}");
                continue;
            }

            if let Ok(mut t) = world.get::<&mut Transform>(entity) {
                if let Some(v) = scope.get_value::<f64>("x") {
                    t.position[0] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("y") {
                    t.position[1] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("z") {
                    t.position[2] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("rx") {
                    t.rotation_euler[0] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("ry") {
                    t.rotation_euler[1] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("rz") {
                    t.rotation_euler[2] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("sx") {
                    t.scale[0] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("sy") {
                    t.scale[1] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("sz") {
                    t.scale[2] = v as f32;
                }
            }

            if let Ok(mut rb) = world.get::<&mut RigidBody>(entity) {
                if let Some(v) = scope.get_value::<f64>("vx") {
                    rb.velocity[0] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("vy") {
                    rb.velocity[1] = v as f32;
                }
                if let Some(v) = scope.get_value::<f64>("vz") {
                    rb.velocity[2] = v as f32;
                }
                let ix = scope.get_value::<f64>("impulse_x").unwrap_or(0.0) as f32;
                let iy = scope.get_value::<f64>("impulse_y").unwrap_or(0.0) as f32;
                let iz = scope.get_value::<f64>("impulse_z").unwrap_or(0.0) as f32;
                if rb.inverse_mass > 0.0 {
                    rb.velocity[0] += ix * rb.inverse_mass;
                    rb.velocity[1] += iy * rb.inverse_mass;
                    rb.velocity[2] += iz * rb.inverse_mass;
                }
            }

            if let Ok(mut audio) = world.get::<&mut AudioSource>(entity) {
                if let Some(f) = scope.get_value::<f64>("audio_freq") {
                    audio.frequency_hz = f as f32;
                }
                if let Some(trig) = scope.get_value::<bool>("audio_trigger") {
                    audio.triggered = trig;
                }
            }
        }
    }
}
