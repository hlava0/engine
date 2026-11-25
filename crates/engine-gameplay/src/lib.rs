//! Gameplay subsystems: `hecs` ECS, deterministic 3D rigid-body physics, `rodio` audio with headless
//! fallback, `rhai` scripting, and runtime `SceneEditor` inspector.

pub mod audio;
pub mod ecs;
pub mod editor;
pub mod physics;
pub mod scripting;

pub use audio::AudioEngine;
pub use ecs::{
    AudioSource, Collider, GameWorld, MeshRenderer, Name, PointLightComponent, RigidBody,
    ScriptComponent, Transform,
};
pub use editor::SceneEditor;
pub use hecs;
pub use physics::{PhysicsConfig, step_physics};
pub use scripting::ScriptEngine;
