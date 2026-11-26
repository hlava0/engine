use bytemuck::{Pod, Zeroable};

/// Material parameters pulled via BDA in `rt_gi.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuRtMaterial {
    /// `rgb` = albedo, `a` = roughness.
    pub albedo_roughness: [f32; 4],
    /// `rgb` = emission, `a` = metallic.
    pub emission_metallic: [f32; 4],
}

/// Analytic light parameters pulled via BDA in `rt_gi.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuRtLight {
    /// `xyz` = position, `w` = energy.
    pub pos_energy: [f32; 4],
    /// `xyz` = direction, `w` = type (`0.0` = directional, `1.0` = omni, `2.0` = spot).
    pub dir_type: [f32; 4],
    /// `rgb` = color, `a` = range.
    pub color_range: [f32; 4],
}

/// Per-frame ray-tracing & composition parameters pulled via BDA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RtParams {
    pub inv_view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    /// `xyz` = camera world position, `w` = NRD `denoising_range`.
    pub cam_pos_range: [f32; 4],
    /// `xy` = current NDC jitter, `z` = frame_index as f32, `w` = samples per pixel.
    pub jitter_frame: [f32; 4],
    pub screen_size: [u32; 2],
    pub light_count: u32,
    pub _pad: u32,
}

/// Push constants for `shaders/rt_gi.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RtGiPushConstants {
    pub params_ptr: u64,
    pub materials_ptr: u64,
    pub lights_ptr: u64,
}

/// Push constants for `shaders/nrd_compose.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ComposePushConstants {
    pub params_ptr: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn verify_rt_struct_layouts() {
        assert_eq!(size_of::<GpuRtMaterial>(), 32);
        assert_eq!(size_of::<GpuRtLight>(), 48);
        assert_eq!(size_of::<RtParams>(), 240);
        assert_eq!(offset_of!(RtParams, cam_pos_range), 192);
        assert_eq!(offset_of!(RtParams, jitter_frame), 208);
        assert_eq!(offset_of!(RtParams, screen_size), 224);
        assert_eq!(offset_of!(RtParams, light_count), 232);
        assert_eq!(size_of::<RtGiPushConstants>(), 24);
        assert_eq!(size_of::<ComposePushConstants>(), 8);
    }
}
