//! Data-oriented scene representation, procedural mesh builders, camera state, and GPU sync.

use std::f32::consts::PI;

use engine_gpu_driven::types::{
    extract_frustum_planes_reverse_z, halton, mat4_identity, mat4_look_at_rh, mat4_mul,
    mat4_perspective_reverse_z, mat4_translation_scale,
};
use engine_gpu_driven::{FrameUniforms, GpuInstanceData, GpuMeshDrawInfo, GpuScene, GpuVertex};
use engine_rhi::ash::{self, vk};
use engine_rhi::{RhiDevice, sync};
use engine_rt_nrd::{GpuRtLight, GpuRtMaterial};

/// CPU-side mesh geometry and bounding sphere.
#[derive(Debug, Clone)]
pub struct MeshDesc {
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub bounding_sphere: [f32; 4],
}

/// CPU-side PBR material description.
#[derive(Debug, Clone, Copy)]
pub struct MaterialDesc {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: [f32; 3],
}

/// CPU-side renderable instance with current and previous-frame transforms for motion vectors.
#[derive(Debug, Clone, Copy)]
pub struct InstanceDesc {
    pub transform: [[f32; 4]; 4],
    pub prev_transform: [[f32; 4]; 4],
    pub bounding_sphere: [f32; 4],
    pub mesh_id: u32,
    pub material_id: u32,
    pub animated: bool,
    pub base_pos: [f32; 3],
    pub base_scale: [f32; 3],
}

/// CPU-side light description (directional or omni point light).
#[derive(Debug, Clone, Copy)]
pub struct LightDesc {
    pub pos: [f32; 3],
    pub energy: f32,
    pub dir: [f32; 3],
    pub is_point: bool,
    pub color: [f32; 3],
    pub range: f32,
}

/// Camera state using Reverse-Z perspective and 16-sample Halton subpixel jitter.
#[derive(Debug, Clone)]
pub struct CameraState {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_rad: f32,
    pub near: f32,
    pub far: f32,
    pub view: [[f32; 4]; 4],
    pub prev_view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub prev_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub jitter: [f32; 2],
    pub prev_jitter: [f32; 2],
    pub frame_index: u32,
}

impl CameraState {
    pub fn new(
        position: [f32; 3],
        target: [f32; 3],
        fov_y_deg: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let aspect = width as f32 / height.max(1) as f32;
        let view = mat4_look_at_rh(position, target, [0.0, 1.0, 0.0]);
        let proj = mat4_perspective_reverse_z(fov_y_deg.to_radians(), aspect, 0.1, 200.0);
        let view_proj = mat4_mul(proj, view);
        Self {
            position,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y_rad: fov_y_deg.to_radians(),
            near: 0.1,
            far: 200.0,
            view,
            prev_view: view,
            proj,
            prev_proj: proj,
            view_proj,
            prev_view_proj: view_proj,
            inv_view: mat4_inverse(view),
            inv_proj: mat4_inverse(proj),
            inv_view_proj: mat4_inverse(view_proj),
            frustum_planes: extract_frustum_planes_reverse_z(view_proj),
            jitter: [0.0, 0.0],
            prev_jitter: [0.0, 0.0],
            frame_index: 0,
        }
    }

    /// Advance camera matrices and Halton-16 subpixel jitter for the current frame.
    pub fn update(&mut self, width: u32, height: u32) {
        self.prev_view = self.view;
        self.prev_proj = self.proj;
        self.prev_view_proj = self.view_proj;
        self.prev_jitter = self.jitter;

        let aspect = width as f32 / height.max(1) as f32;
        self.view = mat4_look_at_rh(self.position, self.target, self.up);
        self.proj = mat4_perspective_reverse_z(self.fov_y_rad, aspect, self.near, self.far);
        self.view_proj = mat4_mul(self.proj, self.view);
        self.inv_view = mat4_inverse(self.view);
        self.inv_proj = mat4_inverse(self.proj);
        self.inv_view_proj = mat4_inverse(self.view_proj);
        self.frustum_planes = extract_frustum_planes_reverse_z(self.view_proj);

        let sample_idx = (self.frame_index % 16) + 1;
        let jx = (halton(sample_idx, 2) - 0.5) / width.max(1) as f32;
        let jy = (halton(sample_idx, 3) - 0.5) / height.max(1) as f32;
        self.jitter = [jx, jy];
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

/// Data-oriented CPU scene container synced to GPU buffers.
pub struct SceneWorld {
    pub meshes: Vec<MeshDesc>,
    pub materials: Vec<MaterialDesc>,
    pub instances: Vec<InstanceDesc>,
    pub lights: Vec<LightDesc>,
    pub camera: CameraState,
    pub time: f32,
    pub dirty_instances: bool,
}

impl SceneWorld {
    /// Advance animated instances and mark GPU instance buffer dirty if changed.
    pub fn step(&mut self, dt: f32) {
        self.time += dt;
        let mut any_animated = false;
        for (idx, inst) in self.instances.iter_mut().enumerate() {
            inst.prev_transform = inst.transform;
            if inst.animated {
                let phase = self.time * 1.5 + idx as f32 * 0.7;
                let bob_y = inst.base_pos[1] + 0.25 * phase.sin();
                let shift_x = inst.base_pos[0] + 0.35 * (phase * 0.7).cos();
                inst.transform = mat4_translation_scale(
                    [shift_x, bob_y, inst.base_pos[2]],
                    inst.base_scale,
                );
                any_animated = true;
            }
        }
        if any_animated {
            self.dirty_instances = true;
        }
    }

    /// Build `GpuMeshDrawInfo`, flattened `GpuVertex` and `u32` index arrays.
    pub fn flatten_geometry(&self) -> (Vec<GpuVertex>, Vec<u32>, Vec<GpuMeshDrawInfo>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut draw_infos = Vec::with_capacity(self.meshes.len());

        for mesh in &self.meshes {
            let vertex_offset = vertices.len() as i32;
            let first_index = indices.len() as u32;
            let index_count = mesh.indices.len() as u32;
            vertices.extend_from_slice(&mesh.vertices);
            indices.extend_from_slice(&mesh.indices);
            draw_infos.push(GpuMeshDrawInfo {
                bounding_sphere: mesh.bounding_sphere,
                index_count,
                first_index,
                vertex_offset,
                _pad: 0,
            });
        }
        (vertices, indices, draw_infos)
    }

    /// Fill a caller-owned scratch vector with `GpuInstanceData` (zero heap allocation when pre-sized).
    pub fn fill_gpu_instances(&self, out: &mut Vec<GpuInstanceData>) {
        out.clear();
        for inst in &self.instances {
            let mat = self.materials[inst.material_id as usize];
            out.push(GpuInstanceData {
                model: inst.transform,
                prev_model: inst.prev_transform,
                albedo_metallic: [mat.albedo[0], mat.albedo[1], mat.albedo[2], mat.metallic],
                mesh_id: inst.mesh_id,
                material_id: inst.material_id,
                roughness: mat.roughness,
                _pad: 0,
            });
        }
    }

    /// Build `GpuInstanceData` array matching current `instances` and `materials`.
    pub fn gpu_instances(&self) -> Vec<GpuInstanceData> {
        let mut out = Vec::with_capacity(self.instances.len());
        self.fill_gpu_instances(&mut out);
        out
    }

    /// Build `GpuRtMaterial` array for ray tracing.
    pub fn gpu_materials(&self) -> Vec<GpuRtMaterial> {
        self.materials
            .iter()
            .map(|m| GpuRtMaterial {
                albedo_roughness: [m.albedo[0], m.albedo[1], m.albedo[2], m.roughness],
                emission_metallic: [m.emission[0], m.emission[1], m.emission[2], m.metallic],
            })
            .collect()
    }

    /// Build `GpuRtLight` array for ray tracing.
    pub fn gpu_lights(&self) -> Vec<GpuRtLight> {
        self.lights
            .iter()
            .map(|l| GpuRtLight {
                pos_energy: [l.pos[0], l.pos[1], l.pos[2], l.energy],
                dir_type: [l.dir[0], l.dir[1], l.dir[2], if l.is_point { 1.0 } else { 0.0 }],
                color_range: [l.color[0], l.color[1], l.color[2], l.range],
            })
            .collect()
    }

    /// Allocate and upload initial `GpuScene` buffers.
    pub fn build_gpu_scene(
        &self,
        rhi: &mut RhiDevice,
    ) -> Result<GpuScene, engine_rhi::RhiError> {
        let (vertices, indices, meshes) = self.flatten_geometry();
        let instances = self.gpu_instances();
        GpuScene::new(rhi, &vertices, &indices, &meshes, &instances)
    }

    /// Sync changed/animated instance transforms and frame uniforms to `GpuScene` for `frame_slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn sync_to_gpu(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        scene: &mut GpuScene,
        gpu_instances: &[GpuInstanceData],
        width: u32,
        height: u32,
        frame_slot: usize,
        is_first_frame: bool,
    ) -> Result<FrameUniforms, engine_rhi::RhiError> {
        self.camera.update(width, height);

        if self.dirty_instances {
            let bytes = bytemuck::cast_slice(gpu_instances);
            if !bytes.is_empty() && bytes.len() <= 65536 {
                if !is_first_frame {
                    // Order previous frame's cull/vertex reads before overwriting instance_buffer
                    let pre_update = [sync::buffer_barrier(
                        scene.instance_buffer.buffer,
                        0,
                        bytes.len() as vk::DeviceSize,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                            | vk::PipelineStageFlags2::VERTEX_SHADER,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    )];
                    unsafe { sync::cmd_pipeline_barrier2(device, cmd, &[], &pre_update, &[]) };
                }
                unsafe {
                    device.cmd_update_buffer(cmd, scene.instance_buffer.buffer, 0, bytes);
                }
                let barriers = [sync::buffer_barrier(
                    scene.instance_buffer.buffer,
                    0,
                    bytes.len() as vk::DeviceSize,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                )];
                unsafe { sync::cmd_pipeline_barrier2(device, cmd, &[], &barriers, &[]) };
            }
            self.dirty_instances = false;
        }

        let uniforms = FrameUniforms {
            view: self.camera.view,
            proj: self.camera.proj,
            view_proj: self.camera.view_proj,
            prev_view_proj: self.camera.prev_view_proj,
            frustum_planes: self.camera.frustum_planes,
            jitter: [
                self.camera.jitter[0] * 2.0,
                self.camera.jitter[1] * 2.0,
                self.camera.prev_jitter[0] * 2.0,
                self.camera.prev_jitter[1] * 2.0,
            ],
            screen_size: [width, height],
            hiz_size: [width, height],
        };
        scene.update_frame_uniforms(frame_slot, &uniforms)?;
        Ok(uniforms)
    }
}

/// Build a unit cube mesh centered at the origin (`[-0.5, 0.5]^3`).
pub fn create_cube_mesh() -> MeshDesc {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
    ];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for (pos, uv) in corners.into_iter().zip(uvs) {
            vertices.push(GpuVertex { pos, normal, uv });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshDesc {
        vertices,
        indices,
        bounding_sphere: [0.0, 0.0, 0.0, 0.866_025_4],
    }
}

/// Build a unit XZ plane mesh (`[-0.5, 0.5]^2` at `y = 0`).
pub fn create_plane_mesh() -> MeshDesc {
    let normal = [0.0, 1.0, 0.0];
    let vertices = vec![
        GpuVertex { pos: [-0.5, 0.0, 0.5], normal, uv: [0.0, 0.0] },
        GpuVertex { pos: [0.5, 0.0, 0.5], normal, uv: [1.0, 0.0] },
        GpuVertex { pos: [0.5, 0.0, -0.5], normal, uv: [1.0, 1.0] },
        GpuVertex { pos: [-0.5, 0.0, -0.5], normal, uv: [0.0, 1.0] },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    MeshDesc {
        vertices,
        indices,
        bounding_sphere: [0.0, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2],
    }
}

/// Build a UV sphere mesh of radius 0.5 centered at the origin.
pub fn create_sphere_mesh(rings: u32, sectors: u32) -> MeshDesc {
    let rings = rings.max(4);
    let sectors = sectors.max(4);
    let radius = 0.5f32;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * PI;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let theta = u * 2.0 * PI;
            let nx = sin_phi * theta.cos();
            let ny = cos_phi;
            let nz = sin_phi * theta.sin();
            vertices.push(GpuVertex {
                pos: [nx * radius, ny * radius, nz * radius],
                normal: [nx, ny, nz],
                uv: [u, v],
            });
        }
    }

    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }

    MeshDesc {
        vertices,
        indices,
        bounding_sphere: [0.0, 0.0, 0.0, radius],
    }
}

/// Create a rich procedural demo scene exercising frustum culling, Hi-Z occlusion, RT GI, NRD, and CoopML.
pub fn create_demo_scene(width: u32, height: u32) -> SceneWorld {
    let cube = create_cube_mesh();
    let plane = create_plane_mesh();
    let sphere = create_sphere_mesh(16, 24);
    let meshes = vec![cube, plane, sphere]; // 0 = cube, 1 = plane, 2 = sphere

    let materials = vec![
        // 0: matte concrete floor
        MaterialDesc { albedo: [0.65, 0.65, 0.68], roughness: 0.8, metallic: 0.0, emission: [0.0; 3] },
        // 1: occluder wall
        MaterialDesc { albedo: [0.35, 0.38, 0.42], roughness: 0.7, metallic: 0.0, emission: [0.0; 3] },
        // 2: brushed gold metal
        MaterialDesc { albedo: [0.95, 0.78, 0.35], roughness: 0.18, metallic: 0.95, emission: [0.0; 3] },
        // 3: chrome mirror sphere
        MaterialDesc { albedo: [0.92, 0.94, 0.96], roughness: 0.08, metallic: 1.0, emission: [0.0; 3] },
        // 4: terracotta diffuse cube
        MaterialDesc { albedo: [0.85, 0.32, 0.22], roughness: 0.55, metallic: 0.0, emission: [0.0; 3] },
        // 5: warm emissive beacon
        MaterialDesc { albedo: [1.0, 0.7, 0.3], roughness: 0.3, metallic: 0.0, emission: [3.5, 2.2, 0.8] },
    ];

    let make_inst = |pos: [f32; 3], scale: [f32; 3], mesh_id: u32, material_id: u32, animated: bool| {
        let m = mat4_translation_scale(pos, scale);
        let max_s = scale[0].max(scale[1]).max(scale[2]);
        let r = meshes[mesh_id as usize].bounding_sphere[3] * max_s;
        InstanceDesc {
            transform: m,
            prev_transform: m,
            bounding_sphere: [0.0, 0.0, 0.0, r],
            mesh_id,
            material_id,
            animated,
            base_pos: pos,
            base_scale: scale,
        }
    };

    let instances = vec![
        // Ground plane
        make_inst([0.0, -1.0, 0.0], [18.0, 1.0, 18.0], 1, 0, false),
        // Occluder wall in front of the occluded cube
        make_inst([-1.2, 0.4, -1.8], [3.2, 2.8, 0.4], 0, 1, false),
        // Occluded cube directly behind the occluder wall
        make_inst([-1.2, 0.0, -4.2], [1.0, 1.0, 1.0], 0, 4, false),
        // Behind-camera instance (frustum culled)
        make_inst([0.0, 0.5, 14.0], [1.5, 1.5, 1.5], 0, 4, false),
        // Metallic gold sphere (animated)
        make_inst([1.4, 0.2, 0.2], [1.4, 1.4, 1.4], 2, 2, true),
        // Chrome sphere
        make_inst([0.2, -0.1, 1.2], [1.1, 1.1, 1.1], 2, 3, false),
        // Terracotta rough cube
        make_inst([2.2, -0.35, -1.2], [1.3, 1.3, 1.3], 0, 4, false),
        // Emissive cube
        make_inst([-2.6, 0.0, 1.2], [0.7, 0.7, 0.7], 0, 5, false),
    ];

    let lights = vec![
        LightDesc {
            pos: [0.0, 0.0, 0.0],
            energy: 4.5,
            dir: [-0.45, -0.82, -0.35],
            is_point: false,
            color: [1.0, 0.96, 0.88],
            range: 1000.0,
        },
        LightDesc {
            pos: [-2.2, 2.8, 2.5],
            energy: 8.0,
            dir: [0.0, -1.0, 0.0],
            is_point: true,
            color: [0.85, 0.92, 1.0],
            range: 24.0,
        },
        LightDesc {
            pos: [2.5, 1.5, 1.8],
            energy: 5.0,
            dir: [0.0, -1.0, 0.0],
            is_point: true,
            color: [1.0, 0.82, 0.65],
            range: 18.0,
        },
    ];

    let camera = CameraState::new([0.0, 1.8, 5.5], [0.0, 0.2, 0.0], 60.0, width, height);

    SceneWorld {
        meshes,
        materials,
        instances,
        lights,
        camera,
        time: 0.0,
        dirty_instances: false,
    }
}

/// 4x4 matrix inverse (column-major).
pub fn mat4_inverse(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let a = [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ];
    let mut inv = [0.0f32; 16];
    inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] - a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] + a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];

    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if det.abs() < 1e-12 {
        return mat4_identity();
    }
    let inv_det = 1.0 / det;
    for v in &mut inv {
        *v *= inv_det;
    }
    [
        [inv[0], inv[1], inv[2], inv[3]],
        [inv[4], inv[5], inv[6], inv[7]],
        [inv[8], inv[9], inv[10], inv[11]],
        [inv[12], inv[13], inv[14], inv[15]],
    ]
}
