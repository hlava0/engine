//! Persistent `VkPipelineCache` wrapper with atomic disk load/save.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::device::DeviceCapabilities;
use crate::error::Result;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Wrapper around `vk::PipelineCache` that loads cached pipeline binaries from disk on startup
/// and persists updated binaries back to disk on `save()` / `destroy()`.
pub struct PipelineCache {
    handle: vk::PipelineCache,
    path: PathBuf,
}

impl PipelineCache {
    /// Compute the default cache file path keyed by GPU vendor, device ID, and driver version.
    pub fn default_path(caps: &DeviceCapabilities) -> PathBuf {
        PathBuf::from(format!(
            "target/pipeline_cache_{:04x}_{:04x}_{}.bin",
            caps.vendor_id, caps.device_id, caps.driver_version
        ))
    }

    /// Load an existing pipeline cache from `path` if present, or create a fresh empty cache.
    pub fn load_or_create(device: &ash::Device, path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let initial_data = match std::fs::read(&path) {
            Ok(bytes) if !bytes.is_empty() => {
                log::info!(
                    "Loaded {} bytes of VkPipelineCache data from {}",
                    bytes.len(),
                    path.display()
                );
                bytes
            }
            _ => Vec::new(),
        };

        let create_info = if initial_data.is_empty() {
            vk::PipelineCacheCreateInfo::default()
        } else {
            vk::PipelineCacheCreateInfo::default().initial_data(&initial_data)
        };

        let handle = match unsafe { device.create_pipeline_cache(&create_info, None) } {
            Ok(cache) => cache,
            Err(err) if !initial_data.is_empty() => {
                log::warn!(
                    "Failed to create VkPipelineCache from {} ({err}); creating empty cache",
                    path.display()
                );
                unsafe {
                    device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
                }
            }
            Err(err) => return Err(err.into()),
        };

        Ok(Self { handle, path })
    }

    /// Underlying Vulkan `vk::PipelineCache` handle.
    #[inline]
    pub fn handle(&self) -> vk::PipelineCache {
        self.handle
    }

    /// Path on disk where this cache is persisted.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Persist the current pipeline cache binary to disk atomically.
    pub fn save(&self, device: &ash::Device) -> Result<()> {
        if self.handle == vk::PipelineCache::null() {
            return Ok(());
        }
        let data = unsafe { device.get_pipeline_cache_data(self.handle)? };
        if data.is_empty() {
            return Ok(());
        }

        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_name = format!(
            "{}.tmp.{}.{}",
            self.path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("pipeline_cache.bin"),
            std::process::id(),
            seq
        );
        let tmp_path = self.path.with_file_name(tmp_name);
        std::fs::write(&tmp_path, &data)?;
        if let Err(err) = std::fs::rename(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(err.into());
        }

        log::info!(
            "Saved {} bytes of VkPipelineCache data to {}",
            data.len(),
            self.path.display()
        );
        Ok(())
    }

    /// Save updated cache data to disk (best-effort) and destroy the Vulkan handle.
    pub fn destroy(&mut self, device: &ash::Device) {
        if self.handle == vk::PipelineCache::null() {
            return;
        }
        if let Err(err) = self.save(device) {
            log::warn!(
                "Failed to save VkPipelineCache to {}: {err}",
                self.path.display()
            );
        }
        unsafe {
            device.destroy_pipeline_cache(self.handle, None);
        }
        self.handle = vk::PipelineCache::null();
    }
}
