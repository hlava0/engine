//! glTF 2.0 / GLB scene loader and GPU texture staging/bindless upload pipeline.

pub mod gltf_loader;
pub mod texture;

use engine_rhi::RhiError;
use thiserror::Error;

pub use gltf_loader::{
    LoadedInstance, LoadedMaterial, LoadedMesh, LoadedScene, load_gltf_from_path,
    load_gltf_from_slice,
};
pub use texture::{
    TextureData, load_texture_from_memory, load_texture_from_path, upload_texture_to_gpu,
};

/// Errors returned by `engine-assets`.
#[derive(Debug, Error)]
pub enum AssetError {
    #[error("glTF error: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("Image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("RHI error: {0}")]
    Rhi(#[from] RhiError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid asset: {0}")]
    InvalidAsset(String),
}

pub type Result<T> = std::result::Result<T, AssetError>;
