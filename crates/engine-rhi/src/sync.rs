use ash::vk;

/// Construct a `vk::MemoryBarrier2` for global memory dependencies.
#[inline]
pub fn memory_barrier<'a>(
    src_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    dst_access_mask: vk::AccessFlags2,
) -> vk::MemoryBarrier2<'a> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
}

/// Construct a `vk::BufferMemoryBarrier2` for a buffer range.
#[inline]
pub fn buffer_barrier<'a>(
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    src_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    dst_access_mask: vk::AccessFlags2,
) -> vk::BufferMemoryBarrier2<'a> {
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)
}

/// Construct a `vk::ImageMemoryBarrier2` for an image subresource range.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn image_barrier<'a>(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    dst_access_mask: vk::AccessFlags2,
    subresource_range: vk::ImageSubresourceRange,
) -> vk::ImageMemoryBarrier2<'a> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
}

/// Record a `vkCmdPipelineBarrier2` using Synchronization2.
///
/// # Safety
/// `cmd` must be a valid command buffer in the recording state on `device`.
#[inline]
pub unsafe fn cmd_pipeline_barrier2(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    memory_barriers: &[vk::MemoryBarrier2<'_>],
    buffer_barriers: &[vk::BufferMemoryBarrier2<'_>],
    image_barriers: &[vk::ImageMemoryBarrier2<'_>],
) {
    let dependency_info = vk::DependencyInfo::default()
        .memory_barriers(memory_barriers)
        .buffer_memory_barriers(buffer_barriers)
        .image_memory_barriers(image_barriers);

    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dependency_info);
    }
}
