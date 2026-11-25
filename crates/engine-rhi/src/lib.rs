//! Vulkan 1.4 RHI: instance/device bootstrap, capability probing, BDA buffers, bindless descriptors, images, and Synchronization2 helpers.

pub mod async_queues;
pub mod bindless;
pub mod buffer;
pub mod device;
pub mod error;
pub mod image;
pub mod pipeline_cache;
pub mod swapchain;
pub mod sync;
pub mod vendor;

pub use ash;
pub use ash_window;
pub use gpu_allocator;
pub use raw_window_handle;

/// Number of frames kept in flight concurrently for CPU/GPU async pipelining.
pub const FRAMES_IN_FLIGHT: usize = 2;

pub use async_queues::AsyncQueueContext;
pub use bindless::{BindlessConfig, BindlessTable};
pub use buffer::GpuBuffer;
pub use device::{
    API_VERSION_1_3, API_VERSION_1_4, CooperativeMatrixProps, DeviceCapabilities,
    QueueFamilyIndices, RhiConfig, RhiDevice,
};
pub use error::{Result, RhiError};
pub use image::GpuImage;
pub use pipeline_cache::PipelineCache;
pub use swapchain::Swapchain;
pub use vendor::{
    CoopMatrixTile, GpuVendor, VendorProbeInput, VendorProfile, align_up,
    is_f16_subgroup_coop_shape,
};

mod shader_registry {
    include!(concat!(env!("OUT_DIR"), "/shaders/registry.rs"));
}

/// Load compiled SPIR-V words for a shader relative to `shaders/` (e.g. `"cull.comp"`).
pub fn load_shader_words(name: &str) -> Result<Vec<u32>> {
    let bytes = shader_registry::shader_spv_bytes(name)
        .ok_or_else(|| RhiError::ShaderNotFound(name.to_string()))?;
    Ok(spirv_bytes_to_words(bytes))
}

/// Convert raw SPIR-V bytes (must be 4-byte aligned in length) into `Vec<u32>`.
pub fn spirv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "SPIR-V byte length must be a multiple of 4");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect()
}
