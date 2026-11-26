//! GPU-driven rendering: BDA scene buffers, compute frustum + Hi-Z occlusion culling, and `vkCmdDrawIndexedIndirectCount` G-buffer pass.

pub mod pipeline;
pub mod scene;
pub mod skinning;
pub mod types;

pub use pipeline::{GBufferTargets, GpuDrivenPipeline};
pub use scene::GpuScene;
pub use skinning::{ComputeSkinningPass, GpuSkinWeight, SkinningPushConstants};
pub use types::{
    CULL_FLAG_FRUSTUM, CULL_FLAG_HIZ, CullPushConstants, DrawIndexedIndirectCommand, FrameUniforms,
    GeomPushConstants, GpuInstanceData, GpuMaterialTextures, GpuMeshDrawInfo, GpuVertex,
    HizDownsamplePushConstants,
};
