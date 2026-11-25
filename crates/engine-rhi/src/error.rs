use ash::vk;
use thiserror::Error;

/// Errors returned by `engine-rhi`.
#[derive(Debug, Error)]
pub enum RhiError {
    #[error("Vulkan loading error: {0}")]
    Loading(#[from] ash::LoadingError),

    #[error("Vulkan error: {0}")]
    Vk(#[from] vk::Result),

    #[error("GPU allocator error: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),

    #[error("No suitable Vulkan physical device found")]
    NoSuitableDevice,

    #[error("Missing required Vulkan feature or extension: {0}")]
    MissingCapability(String),

    #[error("Invalid buffer operation: {0}")]
    InvalidBuffer(String),

    #[error("Invalid image operation: {0}")]
    InvalidImage(String),

    #[error("Compiled shader not found: {0}")]
    ShaderNotFound(String),

    #[error("CString conversion error: {0}")]
    NulError(#[from] std::ffi::NulError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RhiError>;
