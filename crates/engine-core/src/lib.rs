//! Data-oriented scene/ECS, GPU timestamp profiler, and full Vulkan 1.4 engine loop.

pub mod engine;
pub mod hud;
pub mod pacer;
pub mod profiler;
pub mod scene;

use engine_compute_ml::CoopMlError;
use engine_rhi::RhiError;
use engine_rhi::ash::vk;
use engine_rt_nrd::RtNrdError;
use thiserror::Error;

pub use engine::{Engine, FrameStats, GraphicsSettings};
pub use hud::HudOverlayState;
pub use pacer::{FramePacer, PacerStats};
pub use profiler::{GpuProfiler, PassTiming};
pub use scene::{
    CameraState, InstanceDesc, LightDesc, MaterialDesc, MeshDesc, SceneWorld, create_cube_mesh,
    create_demo_scene, create_plane_mesh, create_sphere_mesh,
};

/// Errors returned by `engine-core`.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("RHI error: {0}")]
    Rhi(#[from] RhiError),
    #[error("RT/NRD error: {0}")]
    RtNrd(#[from] RtNrdError),
    #[error("CoopML error: {0}")]
    CoopMl(#[from] CoopMlError),
    #[error("Vulkan error: {0}")]
    Vk(#[from] vk::Result),
}
