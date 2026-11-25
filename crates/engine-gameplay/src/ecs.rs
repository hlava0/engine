//! Core ECS components and `GameWorld` wrapper over `hecs::World`.

use hecs::{DynamicBundle, Entity, World};
use serde::{Deserialize, Serialize};

/// Human-readable entity name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Name(pub String);

impl Name {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// 3D spatial transform with Euler angles (radians, XYZ intrinsic / ZYX extrinsic composition).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation_euler: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_euler: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform {
    pub fn from_position(position: [f32; 3]) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    pub fn from_position_scale(position: [f32; 3], scale: [f32; 3]) -> Self {
        Self {
            position,
            rotation_euler: [0.0, 0.0, 0.0],
            scale,
        }
    }

    /// Convert to a column-major 4x4 TRS matrix (`T * Rz * Ry * Rx * S`).
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        let (sx, cx) = self.rotation_euler[0].sin_cos();
        let (sy, cy) = self.rotation_euler[1].sin_cos();
        let (sz, cz) = self.rotation_euler[2].sin_cos();

        // R = Rz * Ry * Rx
        let r00 = cz * cy;
        let r01 = cz * sy * sx - sz * cx;
        let r02 = cz * sy * cx + sz * sx;

        let r10 = sz * cy;
        let r11 = sz * sy * sx + cz * cx;
        let r12 = sz * sy * cx - cz * sx;

        let r20 = -sy;
        let r21 = cy * sx;
        let r22 = cy * cx;

        let sc = self.scale;
        let t = self.position;

        // Column-major layout: m[col][row]
        [
            [r00 * sc[0], r10 * sc[0], r20 * sc[0], 0.0],
            [r01 * sc[1], r11 * sc[1], r21 * sc[1], 0.0],
            [r02 * sc[2], r12 * sc[2], r22 * sc[2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ]
    }
}

/// Associates an entity with a GPU scene mesh/material and optional render instance slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshRenderer {
    pub mesh_id: u32,
    pub material_id: u32,
    pub instance_index: Option<usize>,
}

/// Dynamic or static rigid-body physics parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub velocity: [f32; 3],
    /// `0.0` denotes an immovable/static body.
    pub inverse_mass: f32,
    pub restitution: f32,
    pub use_gravity: bool,
}

impl RigidBody {
    pub fn dynamic(mass: f32, restitution: f32) -> Self {
        Self {
            velocity: [0.0, 0.0, 0.0],
            inverse_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            restitution: restitution.clamp(0.0, 1.0),
            use_gravity: true,
        }
    }

    pub fn static_body(restitution: f32) -> Self {
        Self {
            velocity: [0.0, 0.0, 0.0],
            inverse_mass: 0.0,
            restitution: restitution.clamp(0.0, 1.0),
            use_gravity: false,
        }
    }
}

/// Collision shape component for `rapier3d` 3D rigid-body physics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Collider {
    Sphere { radius: f32 },
    Box { half_extents: [f32; 3] },
    Capsule { half_height: f32, radius: f32 },
    GroundPlane { y: f32 },
}

/// Optional triangle-mesh collider component for static/kinematic level geometry in `rapier3d`.
#[derive(Debug, Clone, PartialEq)]
pub struct TriMeshCollider {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
}

/// Joint constraint connecting an entity's [`RigidBody`] to another entity in `rapier3d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKind {
    Fixed,
    Spherical,
    Revolute { axis: [f32; 3] },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsJoint {
    pub other: Entity,
    pub kind: JointKind,
    pub anchor_self: [f32; 3],
    pub anchor_other: [f32; 3],
}

/// Point light component attached to an entity's [`Transform`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointLightComponent {
    pub color: [f32; 3],
    pub energy: f32,
    pub range: f32,
}

/// Rhai script attached to an entity; executed each tick by `ScriptEngine`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScriptComponent {
    pub source: String,
}

impl ScriptComponent {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }
}

/// Procedural/triggerable audio source component.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioSource {
    pub frequency_hz: f32,
    pub triggered: bool,
}

/// High-level ECS container wrapping [`hecs::World`].
#[derive(Default)]
pub struct GameWorld {
    pub world: World,
}

impl GameWorld {
    pub fn new() -> Self {
        Self {
            world: World::new(),
        }
    }

    pub fn spawn(&mut self, components: impl DynamicBundle) -> Entity {
        self.world.spawn(components)
    }

    pub fn entity_count(&self) -> u32 {
        self.world.len()
    }
}
