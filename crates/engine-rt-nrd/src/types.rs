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

/// Bitflags passed via `RtParams::_pad` (`rt_flags`) to `shaders/rt_gi.comp`.
pub const RT_FLAG_DISABLE_GI: u32 = 1 << 0;
pub const RT_FLAG_CHECKERBOARD: u32 = 1 << 1;
pub const RT_FLAG_RESTIR_GI: u32 = 1 << 2;
pub const RT_FLAG_SER_AVAILABLE: u32 = 1 << 3;

/// Per-pixel ReSTIR GI temporal reservoir stored in a ping-ponged BDA buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RestirReservoir {
    /// `xyz` = secondary hit direction (or world offset), `w` = hit distance.
    pub dir_hit: [f32; 4],
    /// `rgb` = demodulated indirect radiance, `a` = target PDF (luminance).
    pub radiance_pdf: [f32; 4],
    /// `x` = M (sample count), `y` = W (unbiased weight), `z` = view_z, `w` = normal_hash.
    pub m_w_vz_norm: [f32; 4],
}

/// Push constants for `shaders/rt_gi.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RtGiPushConstants {
    pub params_ptr: u64,
    pub materials_ptr: u64,
    pub lights_ptr: u64,
    pub prev_reservoirs_ptr: u64,
    pub cur_reservoirs_ptr: u64,
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
        assert_eq!(size_of::<RestirReservoir>(), 48);
        assert_eq!(size_of::<RtGiPushConstants>(), 40);
        assert_eq!(size_of::<ComposePushConstants>(), 8);
    }
}
