//! Gameplay subsystems: `hecs` ECS, deterministic 3D rigid-body physics, `rodio` audio with headless
//! fallback, `rhai` scripting, and runtime `SceneEditor` inspector.

pub mod audio;
pub mod ecs;
pub mod editor;
pub mod physics;
pub mod scripting;

pub use audio::AudioEngine;
pub use ecs::{
    AudioSource, Collider, GameWorld, JointKind, MeshRenderer, Name, PhysicsJoint,
    PointLightComponent, RigidBody, ScriptComponent, Transform, TriMeshCollider,
};
pub use editor::SceneEditor;
pub use hecs;
pub use physics::{PhysicsConfig, PhysicsWorld, step_physics};
pub use scripting::ScriptEngine;
