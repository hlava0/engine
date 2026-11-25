//! `rhai` scripting runtime with AST compilation caching and ECS `Transform` binding.

use std::collections::HashMap;

use hecs::World;
use rhai::{AST, Engine as RhaiEngine, Scope};

use crate::ecs::{ScriptComponent, Transform};

/// Compiles, caches, and executes entity [`ScriptComponent`]s against their [`Transform`].
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
        Self {
            engine,
            cache: HashMap::new(),
        }
    }

    /// Execute all [`ScriptComponent`]s attached to entities with a [`Transform`].
    ///
    /// Scripts receive `x, y, z, rx, ry, rz, dt, time` as `f64` variables in scope and any
    /// mutations to `x, y, z, rx, ry, rz` are written back to the entity's [`Transform`].
    pub fn step_scripts(&mut self, world: &mut World, dt: f32, time: f32) {
        for (_entity, (transform, script)) in
            world.query_mut::<(&mut Transform, &ScriptComponent)>()
        {
            if !self.cache.contains_key(&script.source) {
                match self.engine.compile(&script.source) {
                    Ok(ast) => {
                        self.cache.insert(script.source.clone(), ast);
                    }
                    Err(err) => {
                        log::warn!("Rhai script compile error: {err}");
                        continue;
                    }
                }
            }

            let Some(ast) = self.cache.get(&script.source) else {
                continue;
            };

            let mut scope = Scope::new();
            scope.push("x", transform.position[0] as f64);
            scope.push("y", transform.position[1] as f64);
            scope.push("z", transform.position[2] as f64);
            scope.push("rx", transform.rotation_euler[0] as f64);
            scope.push("ry", transform.rotation_euler[1] as f64);
            scope.push("rz", transform.rotation_euler[2] as f64);
            scope.push("dt", dt as f64);
            scope.push("time", time as f64);

            if let Err(err) = self.engine.run_ast_with_scope(&mut scope, ast) {
                log::warn!("Rhai script runtime error: {err}");
                continue;
            }

            if let Some(v) = scope.get_value::<f64>("x") {
                transform.position[0] = v as f32;
            }
            if let Some(v) = scope.get_value::<f64>("y") {
                transform.position[1] = v as f32;
            }
            if let Some(v) = scope.get_value::<f64>("z") {
                transform.position[2] = v as f32;
            }
            if let Some(v) = scope.get_value::<f64>("rx") {
                transform.rotation_euler[0] = v as f32;
            }
            if let Some(v) = scope.get_value::<f64>("ry") {
                transform.rotation_euler[1] = v as f32;
            }
            if let Some(v) = scope.get_value::<f64>("rz") {
                transform.rotation_euler[2] = v as f32;
            }
        }
    }
}
