//! glTF 2.0 and binary GLB scene loader producing engine-ready meshes, PBR materials, and instances.

use std::path::Path;

use engine_gpu_driven::GpuVertex;
use engine_gpu_driven::types::{mat4_identity, mat4_mul};

use crate::{AssetError, Result};

/// A flattened mesh primitive extracted from a glTF 2.0 asset.
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    pub name: String,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    /// `xyz` = local bounding sphere center, `w` = radius.
    pub bounding_sphere: [f32; 4],
}

/// PBR metallic-roughness material factors extracted from a glTF 2.0 asset.
#[derive(Debug, Clone)]
pub struct LoadedMaterial {
    pub name: String,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 3],
}

impl Default for LoadedMaterial {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

/// A scene graph instance referencing a [`LoadedMesh`] and [`LoadedMaterial`] with a world transform.
#[derive(Debug, Clone)]
pub struct LoadedInstance {
    pub name: String,
    pub mesh_index: usize,
    pub material_index: usize,
    /// Column-major 4x4 world transform matrix.
    pub transform: [[f32; 4]; 4],
}

/// Complete scene extracted from a glTF 2.0 / GLB document.
#[derive(Debug, Clone, Default)]
pub struct LoadedScene {
    pub meshes: Vec<LoadedMesh>,
    pub materials: Vec<LoadedMaterial>,
    pub instances: Vec<LoadedInstance>,
}

/// Load a glTF 2.0 (`.gltf` or `.glb`) file from disk.
pub fn load_gltf_from_path<P: AsRef<Path>>(path: P) -> Result<LoadedScene> {
    let (document, buffers, _images) = gltf::import(path)?;
    extract_scene(&document, &buffers)
}

/// Load a self-contained glTF 2.0 / binary `.glb` from an in-memory byte slice.
pub fn load_gltf_from_slice(bytes: &[u8]) -> Result<LoadedScene> {
    let (document, buffers, _images) = gltf::import_slice(bytes)?;
    extract_scene(&document, &buffers)
}

fn extract_scene(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<LoadedScene> {
    // 1. Extract materials (always guarantee at least one default material at the end if needed)
    let mut materials: Vec<LoadedMaterial> = document
        .materials()
        .enumerate()
        .map(|(idx, mat)| {
            let pbr = mat.pbr_metallic_roughness();
            let base = pbr.base_color_factor();
            LoadedMaterial {
                name: mat
                    .name()
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("material_{idx}")),
                albedo: [base[0], base[1], base[2]],
                roughness: pbr.roughness_factor(),
                metallic: pbr.metallic_factor(),
                emissive: mat.emissive_factor(),
            }
        })
        .collect();

    let mut default_material_idx: Option<usize> = None;
    let mut get_or_insert_default_material = |mats: &mut Vec<LoadedMaterial>| -> usize {
        if let Some(idx) = default_material_idx {
            idx
        } else {
            let idx = mats.len();
            mats.push(LoadedMaterial::default());
            default_material_idx = Some(idx);
            idx
        }
    };

    // 2. Extract all mesh primitives, tracking `(loaded_mesh_index, material_index)` per glTF mesh
    let mut meshes = Vec::new();
    let mut mesh_primitives: Vec<Vec<(usize, usize)>> = Vec::new();

    for (mesh_idx, mesh) in document.meshes().enumerate() {
        let base_name = mesh
            .name()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("mesh_{mesh_idx}"));
        let mut prim_entries = Vec::new();

        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            if positions.is_empty() {
                continue;
            }

            let indices: Vec<u32> = match reader.read_indices() {
                Some(read_indices) => read_indices.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            if indices.is_empty() {
                continue;
            }

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(iter) => iter.collect(),
                None => compute_flat_normals(&positions, &indices),
            };

            let mut vertices = Vec::with_capacity(positions.len());
            for (i, &pos) in positions.iter().enumerate() {
                let normal = normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                let uv = uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                vertices.push(GpuVertex { pos, normal, uv });
            }

            let bounding_sphere = compute_bounding_sphere(&positions);
            let loaded_mesh_idx = meshes.len();
            let prim_name = if prim_idx == 0 {
                base_name.clone()
            } else {
                format!("{base_name}_prim{prim_idx}")
            };

            meshes.push(LoadedMesh {
                name: prim_name,
                vertices,
                indices,
                bounding_sphere,
            });

            let mat_idx = match primitive.material().index() {
                Some(idx) => idx,
                None => get_or_insert_default_material(&mut materials),
            };
            prim_entries.push((loaded_mesh_idx, mat_idx));
        }

        mesh_primitives.push(prim_entries);
    }

    if materials.is_empty() {
        materials.push(LoadedMaterial::default());
    }

    // 3. Walk the scene node hierarchy, composing parent * local 4x4 column-major matrices
    let mut instances = Vec::new();
    let scene_opt = document
        .default_scene()
        .or_else(|| document.scenes().next());

    if let Some(scene) = scene_opt {
        for root in scene.nodes() {
            walk_node(
                &root,
                mat4_identity(),
                &mesh_primitives,
                &mut instances,
            );
        }
    } else {
        for node in document.nodes() {
            walk_node(
                &node,
                mat4_identity(),
                &mesh_primitives,
                &mut instances,
            );
        }
    }

    if meshes.is_empty() {
        return Err(AssetError::InvalidAsset(
            "glTF document contains no triangle meshes".to_string(),
        ));
    }

    Ok(LoadedScene {
        meshes,
        materials,
        instances,
    })
}

fn walk_node(
    node: &gltf::Node<'_>,
    parent_transform: [[f32; 4]; 4],
    mesh_primitives: &[Vec<(usize, usize)>],
    instances: &mut Vec<LoadedInstance>,
) {
    let local = node.transform().matrix();
    let world = mat4_mul(parent_transform, local);

    if let Some(mesh) = node.mesh()
        && let Some(prims) = mesh_primitives.get(mesh.index())
    {
        let node_name = node
            .name()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("node_{}", node.index()));
        for &(mesh_index, material_index) in prims {
            instances.push(LoadedInstance {
                name: node_name.clone(),
                mesh_index,
                material_index,
                transform: world,
            });
        }
    }

    for child in node.children() {
        walk_node(&child, world, mesh_primitives, instances);
    }
}

fn compute_flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut accum = vec![[0.0f32; 3]; positions.len()];

    for tri in indices.as_chunks::<3>().0 {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &idx in &[i0, i1, i2] {
            accum[idx][0] += n[0];
            accum[idx][1] += n[1];
            accum[idx][2] += n[2];
        }
    }

    for n in &mut accum {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-8 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }
    accum
}

fn compute_bounding_sphere(positions: &[[f32; 3]]) -> [f32; 4] {
    let mut min_p = [f32::INFINITY; 3];
    let mut max_p = [f32::NEG_INFINITY; 3];
    for p in positions {
        for c in 0..3 {
            min_p[c] = min_p[c].min(p[c]);
            max_p[c] = max_p[c].max(p[c]);
        }
    }
    let center = [
        0.5 * (min_p[0] + max_p[0]),
        0.5 * (min_p[1] + max_p[1]),
        0.5 * (min_p[2] + max_p[2]),
    ];
    let mut radius_sq = 0.0f32;
    for p in positions {
        let dx = p[0] - center[0];
        let dy = p[1] - center[1];
        let dz = p[2] - center[2];
        radius_sq = radius_sq.max(dx * dx + dy * dy + dz * dz);
    }
    [center[0], center[1], center[2], radius_sq.sqrt().max(1e-4)]
}
