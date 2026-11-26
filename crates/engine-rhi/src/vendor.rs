//! GPU vendor detection, subgroup size control, and hardware capability profiling
//! for Intel (Xe / Xe2 / Arc), NVIDIA (RTX / Tensor Core), and AMD (RDNA / WMMA).

use std::fmt;

use ash::vk;

use crate::device::CooperativeMatrixProps;

/// PCI vendor IDs for major GPU IHVs.
pub const VENDOR_ID_INTEL: u32 = 0x8086;
pub const VENDOR_ID_NVIDIA: u32 = 0x10DE;
pub const VENDOR_ID_AMD: u32 = 0x1002;

/// Minimum `DEVICE_LOCAL | HOST_VISIBLE` heap size (256 MiB) required to classify ReBAR / SAM.
pub const REBAR_MIN_HEAP_BYTES: u64 = 256 * 1024 * 1024;

/// Minimum acceleration structure scratch offset alignment floor (bytes).
pub const MIN_AS_SCRATCH_ALIGNMENT_FLOOR: u32 = 256;

/// Detected GPU hardware vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Intel,
    Nvidia,
    Amd,
    Other(u32),
}

impl GpuVendor {
    /// Map a PCI vendor ID to `GpuVendor`.
    pub const fn from_vendor_id(vendor_id: u32) -> Self {
        match vendor_id {
            VENDOR_ID_INTEL => Self::Intel,
            VENDOR_ID_NVIDIA => Self::Nvidia,
            VENDOR_ID_AMD => Self::Amd,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intel => write!(f, "Intel (0x8086)"),
            Self::Nvidia => write!(f, "NVIDIA (0x10DE)"),
            Self::Amd => write!(f, "AMD (0x1002)"),
            Self::Other(id) => write!(f, "Other (0x{id:04X})"),
        }
    }
}

/// Preferred `VK_KHR_cooperative_matrix` tile shape selected for the active GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoopMatrixTile {
    /// Intel Xe / Xe2 XMX (`8x16x16` `FLOAT16` subgroup).
    M8N16K16,
    /// NVIDIA Tensor Core / AMD RDNA3+ WMMA (`16x16x16` `FLOAT16` subgroup).
    M16N16K16,
    /// Scalar compute fallback when cooperative matrix hardware is unavailable.
    Fallback,
}

impl fmt::Display for CoopMatrixTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::M8N16K16 => write!(f, "M8N16K16 (Intel XMX)"),
            Self::M16N16K16 => write!(f, "M16N16K16 (NVIDIA Tensor / AMD WMMA)"),
            Self::Fallback => write!(f, "Fallback (Scalar)"),
        }
    }
}

/// Raw probed inputs used to construct a [`VendorProfile`].
#[derive(Debug, Clone)]
pub struct VendorProbeInput<'a> {
    pub vendor_id: u32,
    pub driver_id: vk::DriverId,
    pub driver_name: String,
    pub driver_info: String,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub required_subgroup_size_stages: vk::ShaderStageFlags,
    pub subgroup_size_control: bool,
    pub khr_cooperative_matrix: bool,
    pub shader_float16: bool,
    pub cooperative_matrix_properties: &'a [CooperativeMatrixProps],
    pub has_rebar: bool,
    pub min_as_scratch_alignment: u32,
    pub nv_ray_tracing_invocation_reorder: bool,
    pub ext_descriptor_buffer: bool,
}

/// Vendor-specific optimization and hardware hardening profile derived from Vulkan 1.4 probing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VendorProfile {
    pub vendor: GpuVendor,
    pub driver_id: vk::DriverId,
    pub driver_name: String,
    pub driver_info: String,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub required_compute_subgroup_size: Option<u32>,
    pub required_rt_subgroup_size: Option<u32>,
    pub preferred_coop_tile: CoopMatrixTile,
    pub has_rebar: bool,
    pub min_as_scratch_alignment: u32,
    pub prefer_tlas_refit: bool,
    pub nv_ray_tracing_invocation_reorder: bool,
    pub ext_descriptor_buffer: bool,
}

impl VendorProfile {
    /// Derive a [`VendorProfile`] from probed physical device properties and extensions.
    pub fn from_probe(input: VendorProbeInput<'_>) -> Self {
        let vendor = GpuVendor::from_vendor_id(input.vendor_id);

        let can_require_compute = input.subgroup_size_control
            && input
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE);

        let fits_compute_subgroup = |size: u32| -> bool {
            can_require_compute
                && size.is_power_of_two()
                && size >= input.min_subgroup_size
                && size <= input.max_subgroup_size
        };

        let required_compute_subgroup_size = match vendor {
            // Intel Xe / Xe2: lock SIMD16 for compute compaction & reductions
            GpuVendor::Intel => fits_compute_subgroup(16).then_some(16),
            // AMD RDNA: prefer Wave32 over Wave64 for lower register pressure & faster ballots
            GpuVendor::Amd => fits_compute_subgroup(32).then_some(32),
            // NVIDIA: native Warp32
            GpuVendor::Nvidia => fits_compute_subgroup(32).then_some(32),
            GpuVendor::Other(_) => None,
        };

        let required_rt_subgroup_size = match vendor {
            // Intel Xe / Xe2: lock SIMD16 to avoid SIMD8 spilling on divergent ray queries
            GpuVendor::Intel => fits_compute_subgroup(16).then_some(16),
            // AMD RDNA: Wave32 for coherent ray traversal
            GpuVendor::Amd => fits_compute_subgroup(32).then_some(32),
            // NVIDIA: fixed Warp32 hardware, no override required
            GpuVendor::Nvidia => None,
            GpuVendor::Other(_) => None,
        };

        let preferred_coop_tile = select_coop_matrix_tile(
            vendor,
            input.khr_cooperative_matrix && input.shader_float16,
            input.cooperative_matrix_properties,
        );

        let min_as_scratch_alignment = input
            .min_as_scratch_alignment
            .max(MIN_AS_SCRATCH_ALIGNMENT_FLOOR);

        let prefer_tlas_refit = matches!(
            vendor,
            GpuVendor::Intel | GpuVendor::Nvidia | GpuVendor::Amd
        );

        Self {
            vendor,
            driver_id: input.driver_id,
            driver_name: input.driver_name,
            driver_info: input.driver_info,
            subgroup_size: input.subgroup_size,
            min_subgroup_size: input.min_subgroup_size,
            max_subgroup_size: input.max_subgroup_size,
            required_compute_subgroup_size,
            required_rt_subgroup_size,
            preferred_coop_tile,
            has_rebar: input.has_rebar,
            min_as_scratch_alignment,
            prefer_tlas_refit,
            nv_ray_tracing_invocation_reorder: input.nv_ray_tracing_invocation_reorder,
            ext_descriptor_buffer: input.ext_descriptor_buffer,
        }
    }
}

/// Returns `true` if `p` matches an `M x N x K` `FLOAT16` subgroup cooperative matrix shape.
pub fn is_f16_subgroup_coop_shape(p: &CooperativeMatrixProps, m: u32, n: u32, k: u32) -> bool {
    p.m_size == m
        && p.n_size == n
        && p.k_size == k
        && p.a_type == vk::ComponentTypeKHR::FLOAT16
        && p.b_type == vk::ComponentTypeKHR::FLOAT16
        && p.c_type == vk::ComponentTypeKHR::FLOAT16
        && p.result_type == vk::ComponentTypeKHR::FLOAT16
        && p.scope == vk::ScopeKHR::SUBGROUP
}

fn select_coop_matrix_tile(
    vendor: GpuVendor,
    coop_f16_enabled: bool,
    props: &[CooperativeMatrixProps],
) -> CoopMatrixTile {
    if !coop_f16_enabled {
        return CoopMatrixTile::Fallback;
    }

    let has_8x16x16 = props
        .iter()
        .any(|p| is_f16_subgroup_coop_shape(p, 8, 16, 16));
    let has_16x16x16 = props
        .iter()
        .any(|p| is_f16_subgroup_coop_shape(p, 16, 16, 16));

    match vendor {
        GpuVendor::Intel => {
            if has_8x16x16 {
                CoopMatrixTile::M8N16K16
            } else if has_16x16x16 {
                CoopMatrixTile::M16N16K16
            } else {
                CoopMatrixTile::Fallback
            }
        }
        GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Other(_) => {
            if has_16x16x16 {
                CoopMatrixTile::M16N16K16
            } else if has_8x16x16 {
                CoopMatrixTile::M8N16K16
            } else {
                CoopMatrixTile::Fallback
            }
        }
    }
}

/// Detect Resizable BAR / Smart Access Memory (`DEVICE_LOCAL | HOST_VISIBLE` heap >= 256 MiB).
pub fn detect_rebar(mem_props: &vk::PhysicalDeviceMemoryProperties) -> bool {
    let required =
        vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
    let type_count = mem_props.memory_type_count as usize;
    let heap_count = mem_props.memory_heap_count as usize;

    mem_props.memory_types[..type_count].iter().any(|mt| {
        if !mt.property_flags.contains(required) {
            return false;
        }
        let heap_idx = mt.heap_index as usize;
        heap_idx < heap_count && mem_props.memory_heaps[heap_idx].size >= REBAR_MIN_HEAP_BYTES
    })
}

/// Align `value` up to the next multiple of `alignment` (must be power of two or > 0).
#[inline]
pub const fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coop_prop(m: u32, n: u32, k: u32) -> CooperativeMatrixProps {
        CooperativeMatrixProps {
            m_size: m,
            n_size: n,
            k_size: k,
            a_type: vk::ComponentTypeKHR::FLOAT16,
            b_type: vk::ComponentTypeKHR::FLOAT16,
            c_type: vk::ComponentTypeKHR::FLOAT16,
            result_type: vk::ComponentTypeKHR::FLOAT16,
            scope: vk::ScopeKHR::SUBGROUP,
            saturating_accumulation: false,
        }
    }

    #[test]
    fn vendor_profile_intel_arc_xe2() {
        let coop_props = [make_coop_prop(8, 16, 16)];
        let profile = VendorProfile::from_probe(VendorProbeInput {
            vendor_id: VENDOR_ID_INTEL,
            driver_id: vk::DriverId::INTEL_OPEN_SOURCE_MESA,
            driver_name: "Intel open-source Mesa driver".to_string(),
            driver_info: "Mesa 25.0.0".to_string(),
            subgroup_size: 16,
            min_subgroup_size: 8,
            max_subgroup_size: 32,
            required_subgroup_size_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_size_control: true,
            khr_cooperative_matrix: true,
            shader_float16: true,
            cooperative_matrix_properties: &coop_props,
            has_rebar: true,
            min_as_scratch_alignment: 128,
            nv_ray_tracing_invocation_reorder: false,
            ext_descriptor_buffer: true,
        });

        assert_eq!(profile.vendor, GpuVendor::Intel);
        assert_eq!(profile.required_compute_subgroup_size, Some(16));
        assert_eq!(profile.required_rt_subgroup_size, Some(16));
        assert_eq!(profile.preferred_coop_tile, CoopMatrixTile::M8N16K16);
        assert!(profile.has_rebar);
        assert_eq!(profile.min_as_scratch_alignment, 256); // clamped to >= 256
        assert!(profile.prefer_tlas_refit);
        assert!(!profile.nv_ray_tracing_invocation_reorder);
        assert!(profile.ext_descriptor_buffer);
    }

    #[test]
    fn vendor_profile_nvidia_rtx() {
        let coop_props = [make_coop_prop(16, 16, 16)];
        let profile = VendorProfile::from_probe(VendorProbeInput {
            vendor_id: VENDOR_ID_NVIDIA,
            driver_id: vk::DriverId::NVIDIA_PROPRIETARY,
            driver_name: "NVIDIA".to_string(),
            driver_info: "565.77".to_string(),
            subgroup_size: 32,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            required_subgroup_size_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_size_control: true,
            khr_cooperative_matrix: true,
            shader_float16: true,
            cooperative_matrix_properties: &coop_props,
            has_rebar: true,
            min_as_scratch_alignment: 128,
            nv_ray_tracing_invocation_reorder: true,
            ext_descriptor_buffer: true,
        });

        assert_eq!(profile.vendor, GpuVendor::Nvidia);
        assert_eq!(profile.required_compute_subgroup_size, Some(32));
        assert_eq!(profile.required_rt_subgroup_size, None);
        assert_eq!(profile.preferred_coop_tile, CoopMatrixTile::M16N16K16);
        assert!(profile.has_rebar);
        assert_eq!(profile.min_as_scratch_alignment, 256);
        assert!(profile.prefer_tlas_refit);
        assert!(profile.nv_ray_tracing_invocation_reorder);
    }

    #[test]
    fn vendor_profile_amd_rdna3() {
        let coop_props = [make_coop_prop(16, 16, 16)];
        let profile = VendorProfile::from_probe(VendorProbeInput {
            vendor_id: VENDOR_ID_AMD,
            driver_id: vk::DriverId::MESA_RADV,
            driver_name: "radv".to_string(),
            driver_info: "Mesa 25.0.0".to_string(),
            subgroup_size: 64,
            min_subgroup_size: 32,
            max_subgroup_size: 64,
            required_subgroup_size_stages: vk::ShaderStageFlags::COMPUTE,
            subgroup_size_control: true,
            khr_cooperative_matrix: true,
            shader_float16: true,
            cooperative_matrix_properties: &coop_props,
            has_rebar: true,
            min_as_scratch_alignment: 512,
            nv_ray_tracing_invocation_reorder: false,
            ext_descriptor_buffer: true,
        });

        assert_eq!(profile.vendor, GpuVendor::Amd);
        assert_eq!(profile.required_compute_subgroup_size, Some(32));
        assert_eq!(profile.required_rt_subgroup_size, Some(32));
        assert_eq!(profile.preferred_coop_tile, CoopMatrixTile::M16N16K16);
        assert_eq!(profile.min_as_scratch_alignment, 512);
        assert!(profile.prefer_tlas_refit);
    }

    #[test]
    fn vendor_profile_fallback_without_coop_or_subgroup_control() {
        let profile = VendorProfile::from_probe(VendorProbeInput {
            vendor_id: VENDOR_ID_INTEL,
            driver_id: vk::DriverId::INTEL_OPEN_SOURCE_MESA,
            driver_name: "Intel".to_string(),
            driver_info: "old".to_string(),
            subgroup_size: 16,
            min_subgroup_size: 8,
            max_subgroup_size: 32,
            required_subgroup_size_stages: vk::ShaderStageFlags::empty(),
            subgroup_size_control: false,
            khr_cooperative_matrix: false,
            shader_float16: true,
            cooperative_matrix_properties: &[],
            has_rebar: false,
            min_as_scratch_alignment: 64,
            nv_ray_tracing_invocation_reorder: false,
            ext_descriptor_buffer: false,
        });

        assert_eq!(profile.required_compute_subgroup_size, None);
        assert_eq!(profile.required_rt_subgroup_size, None);
        assert_eq!(profile.preferred_coop_tile, CoopMatrixTile::Fallback);
        assert!(!profile.has_rebar);
        assert_eq!(profile.min_as_scratch_alignment, 256);
    }

    #[test]
    fn align_up_math() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(1000, 512), 1024);
    }
}
