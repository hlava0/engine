//! Hardware ray tracing and spatiotemporal denoiser integration (optional NVIDIA NRD + built-in GPL-3.0 SVGF).

pub mod accel;
pub mod compose;
pub mod error;
#[cfg(has_nrd)]
pub mod ffi;
pub mod nrd;
pub mod rt_pass;
pub mod svgf;
pub mod types;

pub use accel::{AccelStructureManager, BlasEntry};
pub use compose::NrdComposePass;
pub use error::{Result, RtNrdError};
pub use nrd::{DenoiserMode, NrdDenoiser, NrdInputs};
pub use rt_pass::RtGiPass;
pub use svgf::SvgfDenoiser;
pub use types::{ComposePushConstants, GpuRtLight, GpuRtMaterial, RtGiPushConstants, RtParams};
