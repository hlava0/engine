//! CPU image decoding (PNG/JPEG/HDR/TGA) and Vulkan GPU staging upload into `BindlessTable`.

use std::path::Path;

use ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{BindlessTable, GpuBuffer, GpuImage, RhiDevice, sync};

use crate::{AssetError, Result};

/// Decoded 8-bit RGBA image data ready for GPU upload.
#[derive(Debug, Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Decode an image file (PNG, JPEG, HDR, TGA) from disk into RGBA8 [`TextureData`].
pub fn load_texture_from_path<P: AsRef<Path>>(path: P) -> Result<TextureData> {
    let img = image::open(path)?.to_rgba8();
    let (width, height) = img.dimensions();
    Ok(TextureData {
        width,
        height,
        rgba8: img.into_raw(),
    })
}

/// Decode an image from an in-memory byte slice into RGBA8 [`TextureData`].
pub fn load_texture_from_memory(bytes: &[u8]) -> Result<TextureData> {
    let img = image::load_from_memory(bytes)?.to_rgba8();
    let (width, height) = img.dimensions();
    Ok(TextureData {
        width,
        height,
        rgba8: img.into_raw(),
    })
}

/// Create a `R8G8B8A8_SRGB` [`GpuImage`], copy `tex` via a CPU-to-GPU staging [`GpuBuffer`]
/// using `Synchronization2` barriers, and register the image + sampler at `slot` in `bindless`.
///
/// The caller must keep the returned staging [`GpuBuffer`] alive until `cmd` has finished
/// executing on the GPU, then destroy it.
pub fn upload_texture_to_gpu(
    rhi: &mut RhiDevice,
    cmd: vk::CommandBuffer,
    bindless: &BindlessTable,
    sampler: vk::Sampler,
    slot: u32,
    tex: &TextureData,
) -> Result<(GpuImage, GpuBuffer)> {
    let expected_bytes = (tex.width as usize)
        .saturating_mul(tex.height as usize)
        .saturating_mul(4);
    if tex.width == 0 || tex.height == 0 || tex.rgba8.len() != expected_bytes {
        return Err(AssetError::InvalidAsset(format!(
            "Invalid TextureData dimensions {}x{} with {} bytes (expected {})",
            tex.width,
            tex.height,
            tex.rgba8.len(),
            expected_bytes
        )));
    }

    let device = rhi.device.clone();

    let image = GpuImage::create_2d(
        &device,
        rhi.allocator_mut(),
        tex.width,
        tex.height,
        1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        "asset_texture_srgb",
    )?;

    let mut staging = match GpuBuffer::new(
        &device,
        rhi.allocator_mut(),
        "asset_texture_staging",
        tex.rgba8.len() as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    ) {
        Ok(buf) => buf,
        Err(err) => {
            let _ = image.destroy(&device, rhi.allocator_mut());
            return Err(err.into());
        }
    };

    staging.mapped_slice_mut()?.copy_from_slice(&tex.rgba8);

    // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
    let to_dst = [sync::image_barrier(
        image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags2::TOP_OF_PIPE,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        image.subresource_range(),
    )];
    unsafe {
        sync::cmd_pipeline_barrier2(&device, cmd, &[], &[], &to_dst);
    }

    let copy_region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width: tex.width,
            height: tex.height,
            depth: 1,
        });

    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            staging.buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    // Transition TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    let to_read = [sync::image_barrier(
        image.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ,
        image.subresource_range(),
    )];
    unsafe {
        sync::cmd_pipeline_barrier2(&device, cmd, &[], &[], &to_read);
    }

    bindless.write_sampled_image(
        &device,
        slot,
        image.view,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    bindless.write_sampler(&device, slot, sampler);

    Ok((image, staging))
}
