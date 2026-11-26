use bytemuck::{Pod, Zeroable};

/// Per-vertex attributes pulled via BDA in `gbuffer.vert`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Per-instance transform, bounding/material metadata pulled via BDA in `cull.comp` and `gbuffer.vert`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuInstanceData {
    pub model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
    pub albedo_metallic: [f32; 4],
    pub mesh_id: u32,
    pub material_id: u32,
    pub roughness: f32,
    pub _pad: u32,
}

/// Per-mesh draw metadata and local bounding sphere pulled via BDA in `cull.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMeshDrawInfo {
    /// `xyz` = local center, `w` = local radius.
    pub bounding_sphere: [f32; 4],
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub _pad: u32,
}

/// Mirrors `VkDrawIndexedIndirectCommand` for `vkCmdDrawIndexedIndirectCount`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Per-frame camera, frustum, and jitter uniforms pulled via BDA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub frustum_planes: [[f32; 4]; 6],
    /// `xy` = current NDC subpixel jitter, `zw` = previous NDC subpixel jitter.
    pub jitter: [f32; 4],
    pub screen_size: [u32; 2],
    pub hiz_size: [u32; 2],
}

/// Push constants for `shaders/cull.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CullPushConstants {
    pub frame_uniforms_ptr: u64,
    pub instances_ptr: u64,
    pub meshes_ptr: u64,
    pub indirect_draws_ptr: u64,
    pub draw_count_ptr: u64,
    pub draw_instance_ids_ptr: u64,
    pub instance_count: u32,
    pub hiz_bindless_index: u32,
    pub hiz_sampler_index: u32,
    pub cull_flags: u32,
}

/// Push constants for `shaders/hiz_downsample.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct HizDownsamplePushConstants {
    pub src_texture_idx: u32,
    pub src_sampler_idx: u32,
    pub dst_image_idx: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub is_first_mip: u32,
}

/// Per-material bindless texture indices & alpha-mask settings pulled via BDA in `gbuffer.frag`.
/// `u32::MAX` (`0xFFFFFFFF`) means "no texture bound; use scalar fallback".
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterialTextures {
    pub albedo_tex: u32,
    pub normal_tex: u32,
    pub metallic_roughness_tex: u32,
    pub emissive_tex: u32,
    /// `0` = OPAQUE, `1` = MASK (discard below `alpha_cutoff`), `2` = BLEND.
    pub alpha_mode: u32,
    pub alpha_cutoff: f32,
    pub _pad: [u32; 2],
}

impl Default for GpuMaterialTextures {
    fn default() -> Self {
        Self {
            albedo_tex: u32::MAX,
            normal_tex: u32::MAX,
            metallic_roughness_tex: u32::MAX,
            emissive_tex: u32::MAX,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            _pad: [0; 2],
        }
    }
}

/// Push constants for `shaders/gbuffer.vert` and `shaders/gbuffer.frag`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GeomPushConstants {
    pub frame_uniforms_ptr: u64,
    pub vertices_ptr: u64,
    pub instances_ptr: u64,
    pub draw_instance_ids_ptr: u64,
    pub material_textures_ptr: u64,
    pub default_sampler_idx: u32,
    pub _pad: u32,
}

/// Cull flag bits for `CullPushConstants::cull_flags`.
pub const CULL_FLAG_FRUSTUM: u32 = 1 << 0;
pub const CULL_FLAG_HIZ: u32 = 1 << 1;

/// 4x4 column-major matrix multiplication (`a * b`).
pub fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

/// Identity 4x4 matrix.
pub fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Translation + uniform/non-uniform scale 4x4 matrix (column-major).
pub fn mat4_translation_scale(t: [f32; 3], s: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [s[0], 0.0, 0.0, 0.0],
        [0.0, s[1], 0.0, 0.0],
        [0.0, 0.0, s[2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// Right-handed look-at view matrix (camera looks down `-Z`).
pub fn mat4_look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

/// Vulkan Reverse-Z perspective projection matrix (`near -> 1.0`, `far -> 0.0`, Y-down clip).
pub fn mat4_perspective_reverse_z(
    fov_y_radians: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y_radians * 0.5).tan();
    // Reverse-Z finite far: z_ndc = (near / (far - near)) * (-z_view - far) / (-z_view)
    // At -z_view = near: z_ndc = (near / (far - near)) * (near - far) / near ... wait:
    // z_clip = a * z_view + b, w_clip = -z_view
    // At z_view = -near: z_clip / near = 1  => -a * near + b = near
    // At z_view = -far:  z_clip / far  = 0  => -a * far + b = 0 => b = a * far
    // => a * (far - near) = near => a = near / (far - near), b = near * far / (far - near)
    let a = near / (far - near);
    let b = (near * far) / (far - near);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, -1.0],
        [0.0, 0.0, b, 0.0],
    ]
}

/// Extract 6 normalized inward-pointing frustum planes from a Vulkan Reverse-Z `view_proj` matrix.
pub fn extract_frustum_planes_reverse_z(vp: [[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let row = |r: usize| [vp[0][r], vp[1][r], vp[2][r], vp[3][r]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    let norm = |p: [f32; 4]| {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len > 0.0 {
            [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
        } else {
            p
        }
    };

    [
        norm(add(r3, r0)), // Left
        norm(sub(r3, r0)), // Right
        norm(add(r3, r1)), // Bottom
        norm(sub(r3, r1)), // Top
        norm(sub(r3, r2)), // Near (z_clip <= w_clip)
        norm(r2),          // Far  (z_clip >= 0)
    ]
}

/// Halton radical inverse in base `b` for subpixel jitter generation.
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn verify_gpu_struct_layouts() {
        assert_eq!(size_of::<GpuVertex>(), 32);
        assert_eq!(offset_of!(GpuVertex, pos), 0);
        assert_eq!(offset_of!(GpuVertex, normal), 12);
        assert_eq!(offset_of!(GpuVertex, uv), 24);

        assert_eq!(size_of::<GpuInstanceData>(), 160);
        assert_eq!(offset_of!(GpuInstanceData, model), 0);
        assert_eq!(offset_of!(GpuInstanceData, prev_model), 64);
        assert_eq!(offset_of!(GpuInstanceData, albedo_metallic), 128);
        assert_eq!(offset_of!(GpuInstanceData, mesh_id), 144);
        assert_eq!(offset_of!(GpuInstanceData, material_id), 148);
        assert_eq!(offset_of!(GpuInstanceData, roughness), 152);

        assert_eq!(size_of::<GpuMeshDrawInfo>(), 32);
        assert_eq!(offset_of!(GpuMeshDrawInfo, bounding_sphere), 0);
        assert_eq!(offset_of!(GpuMeshDrawInfo, index_count), 16);
        assert_eq!(offset_of!(GpuMeshDrawInfo, first_index), 20);
        assert_eq!(offset_of!(GpuMeshDrawInfo, vertex_offset), 24);

        assert_eq!(
            size_of::<DrawIndexedIndirectCommand>(),
            size_of::<ash::vk::DrawIndexedIndirectCommand>()
        );
        assert_eq!(size_of::<DrawIndexedIndirectCommand>(), 20);

        assert_eq!(size_of::<FrameUniforms>(), 384);
        assert_eq!(offset_of!(FrameUniforms, frustum_planes), 256);
        assert_eq!(offset_of!(FrameUniforms, jitter), 352);
        assert_eq!(offset_of!(FrameUniforms, screen_size), 368);
        assert_eq!(offset_of!(FrameUniforms, hiz_size), 376);

        assert_eq!(size_of::<CullPushConstants>(), 64);
        assert_eq!(align_of::<CullPushConstants>(), 8);
        assert_eq!(size_of::<HizDownsamplePushConstants>(), 24);
        assert_eq!(size_of::<GeomPushConstants>(), 48);
        assert_eq!(size_of::<GpuMaterialTextures>(), 32);
    }
}
