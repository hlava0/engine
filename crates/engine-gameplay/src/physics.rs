//! Deterministic 3D rigid-body physics: semi-implicit Euler integration,
//! sphere/box vs ground plane collision with restitution, and sphere-vs-sphere impulse resolution.

use hecs::{Entity, World};

use crate::ecs::{Collider, RigidBody, Transform};

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

/// Advance all dynamic rigid bodies by `dt` seconds using semi-implicit Euler integration
/// and resolve ground-plane and sphere-sphere collisions deterministically.
pub fn step_physics(world: &mut World, config: &PhysicsConfig, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    // Collect ground plane heights
    let mut ground_planes: Vec<f32> = Vec::new();
    for (_entity, collider) in world.query::<&Collider>().iter() {
        if let Collider::GroundPlane { y } = *collider {
            ground_planes.push(y);
        }
    }

    // 1. Semi-implicit Euler integration + ground-plane collision
    let damping = (1.0 - config.linear_damping * dt).clamp(0.0, 1.0);
    for (_entity, (transform, rb, collider)) in
        world.query_mut::<(&mut Transform, &mut RigidBody, Option<&Collider>)>()
    {
        if rb.inverse_mass <= 0.0 {
            continue;
        }

        if rb.use_gravity {
            rb.velocity[0] += config.gravity[0] * dt;
            rb.velocity[1] += config.gravity[1] * dt;
            rb.velocity[2] += config.gravity[2] * dt;
        }

        rb.velocity[0] *= damping;
        rb.velocity[1] *= damping;
        rb.velocity[2] *= damping;

        transform.position[0] += rb.velocity[0] * dt;
        transform.position[1] += rb.velocity[1] * dt;
        transform.position[2] += rb.velocity[2] * dt;

        // Ground-plane collision for spheres and boxes
        let half_height = match collider {
            Some(Collider::Sphere { radius }) => Some(*radius),
            Some(Collider::Box { half_extents }) => Some(half_extents[1]),
            _ => None,
        };

        if let Some(hh) = half_height {
            for &plane_y in &ground_planes {
                let min_y = plane_y + hh;
                if transform.position[1] < min_y {
                    transform.position[1] = min_y;
                    if rb.velocity[1] < 0.0 {
                        rb.velocity[1] = -rb.velocity[1] * rb.restitution;
                        if rb.velocity[1].abs() < 0.05 {
                            rb.velocity[1] = 0.0;
                        }
                    }
                }
            }
        }
    }

    struct SphereBody {
        entity: Entity,
        pos: [f32; 3],
        vel: [f32; 3],
        inv_mass: f32,
        restitution: f32,
        radius: f32,
    }

    // 2. Sphere-vs-sphere collision & impulse resolution (sorted by Entity for determinism)
    let mut spheres: Vec<SphereBody> = Vec::new();
    for (entity, (transform, rb, collider)) in
        world.query::<(&Transform, &RigidBody, &Collider)>().iter()
    {
        if let Collider::Sphere { radius } = *collider {
            spheres.push(SphereBody {
                entity,
                pos: transform.position,
                vel: rb.velocity,
                inv_mass: rb.inverse_mass,
                restitution: rb.restitution,
                radius,
            });
        }
    }
    spheres.sort_by_key(|s| s.entity);

    let count = spheres.len();
    for i in 0..count {
        for j in (i + 1)..count {
            let inv_m1 = spheres[i].inv_mass;
            let inv_m2 = spheres[j].inv_mass;
            let inv_mass_sum = inv_m1 + inv_m2;
            if inv_mass_sum <= 0.0 {
                continue;
            }

            let p1 = spheres[i].pos;
            let p2 = spheres[j].pos;
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let dz = p2[2] - p1[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let min_dist = spheres[i].radius + spheres[j].radius;

            if dist_sq < min_dist * min_dist {
                let dist = dist_sq.sqrt();
                let normal = if dist > 1e-6 {
                    [dx / dist, dy / dist, dz / dist]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let penetration = min_dist - dist;

                // Positional separation
                let corr = penetration / inv_mass_sum;
                for (c, &n) in normal.iter().enumerate() {
                    spheres[i].pos[c] -= n * corr * inv_m1;
                    spheres[j].pos[c] += n * corr * inv_m2;
                }

                // Relative velocity along normal
                let rv = [
                    spheres[j].vel[0] - spheres[i].vel[0],
                    spheres[j].vel[1] - spheres[i].vel[1],
                    spheres[j].vel[2] - spheres[i].vel[2],
                ];
                let vel_along_normal = rv[0] * normal[0] + rv[1] * normal[1] + rv[2] * normal[2];
                if vel_along_normal < 0.0 {
                    let e = spheres[i].restitution.min(spheres[j].restitution);
                    let j_impulse = -(1.0 + e) * vel_along_normal / inv_mass_sum;
                    for (c, &n) in normal.iter().enumerate() {
                        spheres[i].vel[c] -= n * j_impulse * inv_m1;
                        spheres[j].vel[c] += n * j_impulse * inv_m2;
                    }
                }
            }
        }
    }

    // Write resolved sphere states back into ECS
    for s in spheres {
        if let Ok(mut query) = world.query_one::<(&mut Transform, &mut RigidBody)>(s.entity)
            && let Some((t, r)) = query.get()
        {
            t.position = s.pos;
            r.velocity = s.vel;
        }
    }
}
