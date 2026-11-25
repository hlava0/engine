//! JSON scene serialization & deserialization for `hecs::World` + `SceneEditor`.

use std::path::Path;

use hecs::{Entity, World};
use serde::{Deserialize, Serialize};

use crate::ecs::{
    AudioSource, Collider, MeshRenderer, Name, PointLightComponent, RigidBody, ScriptComponent,
    Transform,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedEntity {
    pub name: Option<Name>,
    pub transform: Option<Transform>,
    pub mesh_renderer: Option<MeshRenderer>,
    pub rigid_body: Option<RigidBody>,
    pub collider: Option<Collider>,
    pub point_light: Option<PointLightComponent>,
    pub script: Option<ScriptComponent>,
    pub audio_source: Option<AudioSource>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SerializedScene {
    pub entities: Vec<SerializedEntity>,
}

/// Serialize all entities in `world` to a pretty JSON string.
pub fn scene_to_json(world: &World) -> Result<String, serde_json::Error> {
    let mut ids: Vec<Entity> = world.iter().map(|e| e.entity()).collect();
    ids.sort();
    let entities = ids
        .into_iter()
        .map(|e| SerializedEntity {
            name: world.get::<&Name>(e).ok().map(|v| (*v).clone()),
            transform: world.get::<&Transform>(e).ok().map(|v| *v),
            mesh_renderer: world.get::<&MeshRenderer>(e).ok().map(|v| *v),
            rigid_body: world.get::<&RigidBody>(e).ok().map(|v| *v),
            collider: world.get::<&Collider>(e).ok().map(|v| *v),
            point_light: world.get::<&PointLightComponent>(e).ok().map(|v| *v),
            script: world.get::<&ScriptComponent>(e).ok().map(|v| (*v).clone()),
            audio_source: world.get::<&AudioSource>(e).ok().map(|v| *v),
        })
        .collect();
    serde_json::to_string_pretty(&SerializedScene { entities })
}

/// Deserialize a JSON string into a fresh `hecs::World`.
pub fn scene_from_json(json: &str) -> Result<World, serde_json::Error> {
    let parsed: SerializedScene = serde_json::from_str(json)?;
    let mut world = World::new();
    for ent in parsed.entities {
        let mut builder = hecs::EntityBuilder::new();
        if let Some(v) = ent.name {
            builder.add(v);
        }
        if let Some(v) = ent.transform {
            builder.add(v);
        }
        if let Some(v) = ent.mesh_renderer {
            builder.add(v);
        }
        if let Some(v) = ent.rigid_body {
            builder.add(v);
        }
        if let Some(v) = ent.collider {
            builder.add(v);
        }
        if let Some(v) = ent.point_light {
            builder.add(v);
        }
        if let Some(v) = ent.script {
            builder.add(v);
        }
        if let Some(v) = ent.audio_source {
            builder.add(v);
        }
        world.spawn(builder.build());
    }
    Ok(world)
}

/// Save `world` to a `.json` file on disk.
pub fn save_scene_file<P: AsRef<Path>>(world: &World, path: P) -> std::io::Result<()> {
    let json = scene_to_json(world)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load a `hecs::World` from a `.json` file on disk.
pub fn load_scene_file<P: AsRef<Path>>(path: P) -> std::io::Result<World> {
    let json = std::fs::read_to_string(path)?;
    scene_from_json(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
