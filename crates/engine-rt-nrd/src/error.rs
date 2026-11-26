use thiserror::Error;

#[derive(Debug, Error)]
pub enum RtNrdError {
    #[error("RHI error: {0}")]
    Rhi(#[from] engine_rhi::RhiError),

    #[error("Vulkan error: {0}")]
    Vk(#[from] ash::vk::Result),

    #[error("NRD error (result code {0})")]
    Nrd(u32),

    #[error("Ray tracing / acceleration structure not supported on this device")]
    RayTracingUnsupported,

    #[error("Invalid NRD configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, RtNrdError>;
