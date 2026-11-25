//! Full 3D rigid-body physics powered by `rapier3d`:
//! box-box, sphere-sphere, sphere-box, capsule, halfspace ground-plane,
//! triangle-mesh colliders, joints (`Fixed`, `Spherical`, `Revolute`), CCD, and raycasting.

use std::collections::{HashMap, HashSet};

use hecs::{Entity, World};
use rapier3d::na::{Quaternion, UnitQuaternion, Vector3};
use rapier3d::prelude::*;

use crate::ecs::{Collider, JointKind, PhysicsJoint, RigidBody, Transform, TriMeshCollider};

/// Global physics simulation parameters.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsConfig {
    pub gravity: [f32; 3],
    pub linear_damping: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.81, 0.0],
            linear_damping: 0.02,
        }
    }
}

/// Persistent `rapier3d` physics simulation world synchronized with `hecs::World`.
pub struct PhysicsWorld {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    entity_to_body: HashMap<Entity, RigidBodyHandle>,
    body_to_entity: HashMap<RigidBodyHandle, Entity>,
    ground_colliders: HashMap<Entity, ColliderHandle>,
    trimesh_colliders: HashMap<Entity, ColliderHandle>,
    joints_created: HashSet<(Entity, Entity)>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            entity_to_body: HashMap::new(),
            body_to_entity: HashMap::new(),
            ground_colliders: HashMap::new(),
            trimesh_colliders: HashMap::new(),
            joints_created: HashSet::new(),
        }
    }

    fn euler_to_iso(pos: [f32; 3], euler: [f32; 3]) -> Isometry<f32> {
        let rot = UnitQuaternion::from_euler_angles(euler[0], euler[1], euler[2]);
        Isometry::from_parts(Translation::new(pos[0], pos[1], pos[2]), rot)
    }

    fn quat_to_euler(q: &UnitQuaternion<f32>) -> [f32; 3] {
        let (rx, ry, rz) = q.euler_angles();
        [rx, ry, rz]
    }

    fn build_collider(col: &Collider, restitution: f32) -> Option<rapier3d::geometry::Collider> {
        let builder = match *col {
            Collider::Sphere { radius } => ColliderBuilder::ball(radius.max(0.01)),
            Collider::Box { half_extents } => ColliderBuilder::cuboid(
                half_extents[0].max(0.01),
                half_extents[1].max(0.01),
                half_extents[2].max(0.01),
            ),
            Collider::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(half_height.max(0.01), radius.max(0.01)),
            Collider::GroundPlane { .. } => return None,
        };
        Some(builder.restitution(restitution).friction(0.5).build())
    }

    /// Synchronize ECS entities into Rapier, step simulation by `dt`, and write transforms/velocities back.
    pub fn step(&mut self, world: &mut World, config: &PhysicsConfig, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.integration_parameters.dt = dt;

        // 1. Register GroundPlane colliders
        for (entity, col) in world.query::<&Collider>().iter() {
            if let Collider::GroundPlane { y } = *col
                && !self.ground_colliders.contains_key(&entity)
            {
                let rapier_col = ColliderBuilder::halfspace(Vector::y_axis())
                    .translation(vector![0.0, y, 0.0])
                    .restitution(0.8)
                    .friction(0.6)
                    .build();
                let handle = self.collider_set.insert(rapier_col);
                self.ground_colliders.insert(entity, handle);
            }
        }

        // 2. Register TriMesh colliders (static world geometry)
        for (entity, (trimesh, tf)) in world.query::<(&TriMeshCollider, Option<&Transform>)>().iter() {
            if !self.trimesh_colliders.contains_key(&entity)
                && !trimesh.vertices.is_empty()
                && !trimesh.indices.is_empty()
            {
                let pts: Vec<Point<f32>> = trimesh
                    .vertices
                    .iter()
                    .map(|v| point![v[0], v[1], v[2]])
                    .collect();
                let mut builder = ColliderBuilder::trimesh(pts, trimesh.indices.clone())
                    .restitution(0.5)
                    .friction(0.6);
                if let Some(t) = tf {
                    builder = builder.position(Self::euler_to_iso(t.position, t.rotation_euler));
                }
                let handle = self.collider_set.insert(builder.build());
                self.trimesh_colliders.insert(entity, handle);
            }
        }

        // 3. Sync RigidBody entities into Rapier
        let mut sorted_bodies: Vec<(Entity, Transform, RigidBody, Option<Collider>)> = world
            .query::<(&Transform, &RigidBody, Option<&Collider>)>()
            .iter()
            .map(|(e, (t, rb, c))| (e, *t, *rb, c.copied()))
            .collect();
        sorted_bodies.sort_by_key(|(e, _, _, _)| *e);

        for (entity, tf, rb, col) in sorted_bodies {
            if let Some(&handle) = self.entity_to_body.get(&entity) {
                if let Some(body) = self.rigid_body_set.get_mut(handle) {
                    // Sync external velocity / position edits (e.g. projectile impulses or editor nudges)
                    body.set_linvel(vector![rb.velocity[0], rb.velocity[1], rb.velocity[2]], true);
                    body.set_position(Self::euler_to_iso(tf.position, tf.rotation_euler), true);
                    body.set_gravity_scale(if rb.use_gravity { 1.0 } else { 0.0 }, true);
                }
            } else {
                let mut builder = if rb.inverse_mass > 0.0 {
                    RigidBodyBuilder::dynamic()
                        .additional_mass(1.0 / rb.inverse_mass)
                        .linear_damping(config.linear_damping)
                        .ccd_enabled(true)
                } else {
                    RigidBodyBuilder::fixed()
                };
                builder = builder
                    .position(Self::euler_to_iso(tf.position, tf.rotation_euler))
                    .linvel(vector![rb.velocity[0], rb.velocity[1], rb.velocity[2]])
                    .gravity_scale(if rb.use_gravity { 1.0 } else { 0.0 });
                let body_handle = self.rigid_body_set.insert(builder.build());
                if let Some(c) = col.as_ref().and_then(|c| Self::build_collider(c, rb.restitution)) {
                    self.collider_set
                        .insert_with_parent(c, body_handle, &mut self.rigid_body_set);
                }
                self.entity_to_body.insert(entity, body_handle);
                self.body_to_entity.insert(body_handle, entity);
            }
        }

        // 4. Sync PhysicsJoint components
        for (entity, joint) in world.query::<&PhysicsJoint>().iter() {
            let key = (entity, joint.other);
            if self.joints_created.contains(&key) {
                continue;
            }
            if let (Some(&b1), Some(&b2)) = (
                self.entity_to_body.get(&entity),
                self.entity_to_body.get(&joint.other),
            ) {
                let a1 = point![joint.anchor_self[0], joint.anchor_self[1], joint.anchor_self[2]];
                let a2 = point![joint.anchor_other[0], joint.anchor_other[1], joint.anchor_other[2]];
                let rapier_joint: GenericJoint = match joint.kind {
                    JointKind::Fixed => FixedJointBuilder::new()
                        .local_anchor1(a1)
                        .local_anchor2(a2)
                        .build()
                        .into(),
                    JointKind::Spherical => SphericalJointBuilder::new()
                        .local_anchor1(a1)
                        .local_anchor2(a2)
                        .build()
                        .into(),
                    JointKind::Revolute { axis } => {
                        let unit_axis = UnitVector::new_normalize(Vector3::new(
                            axis[0], axis[1], axis[2],
                        ));
                        RevoluteJointBuilder::new(unit_axis)
                            .local_anchor1(a1)
                            .local_anchor2(a2)
                            .build()
                            .into()
                    }
                };
                self.impulse_joint_set.insert(b1, b2, rapier_joint, true);
                self.joints_created.insert(key);
            }
        }

        // 5. Step Rapier3D physics pipeline
        let gravity = vector![config.gravity[0], config.gravity[1], config.gravity[2]];
        self.physics_pipeline.step(
            &gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        // 6. Write back Rapier transforms & velocities to ECS
        for (entity, (tf, rb)) in world.query_mut::<(&mut Transform, &mut RigidBody)>() {
            if rb.inverse_mass <= 0.0 {
                continue;
            }
            if let Some(&handle) = self.entity_to_body.get(&entity)
                && let Some(body) = self.rigid_body_set.get(handle)
            {
                let pos = body.translation();
                let rot = body.rotation();
                let vel = body.linvel();
                tf.position = [pos.x, pos.y, pos.z];
                let q = UnitQuaternion::from_quaternion(Quaternion::new(
                    rot.w, rot.i, rot.j, rot.k,
                ));
                tf.rotation_euler = Self::quat_to_euler(&q);
                rb.velocity = [vel.x, vel.y, vel.z];
            }
        }
    }

    /// Cast a ray against all physics colliders, returning `(hit_entity, hit_distance)`.
    pub fn raycast(&self, origin: [f32; 3], dir: [f32; 3], max_toi: f32) -> Option<(Option<Entity>, f32)> {
        let ray = Ray::new(
            point![origin[0], origin[1], origin[2]],
            vector![dir[0], dir[1], dir[2]],
        );
        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                QueryFilter::default(),
            )
            .map(|(col_handle, toi)| {
                let entity = self
                    .collider_set
                    .get(col_handle)
                    .and_then(|c| c.parent())
                    .and_then(|b| self.body_to_entity.get(&b).copied());
                (entity, toi)
            })
    }
}

/// Stateless convenience wrapper running one `rapier3d` step over `world`.
/// For multi-frame simulation with sleeping, joints, and raycasting, prefer [`PhysicsWorld`].
pub fn step_physics(world: &mut World, config: &PhysicsConfig, dt: f32) {
    let mut pw = PhysicsWorld::new();
    pw.step(world, config, dt);
}
