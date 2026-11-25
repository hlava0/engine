use ash::vk;
use bytemuck::cast_slice;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{FRAMES_IN_FLIGHT, GpuBuffer, Result, RhiDevice, sync};

use crate::types::{
    DrawIndexedIndirectCommand, FrameUniforms, GpuInstanceData, GpuMaterialTextures,
    GpuMeshDrawInfo, GpuVertex,
};

/// GPU-resident scene buffers with Buffer Device Addresses (BDA) for GPU-driven culling & drawing.
pub struct GpuScene {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer: GpuBuffer,
    pub instance_buffer: GpuBuffer,
    pub mesh_buffer: GpuBuffer,
    pub indirect_draw_buffer: GpuBuffer,
    pub draw_count_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
    pub draw_instance_ids_buffer: GpuBuffer,
    pub frame_uniforms_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
    pub material_textures_buffer: Option<GpuBuffer>,
    pub default_sampler_idx: u32,
    pub instance_count: u32,
}

impl GpuScene {
    /// Upload static/initial scene geometry, meshes, and instances to GPU buffers and allocate indirect draw buffers.
    pub fn new(
        rhi: &mut RhiDevice,
        vertices: &[GpuVertex],
        indices: &[u32],
        meshes: &[GpuMeshDrawInfo],
        instances: &[GpuInstanceData],
    ) -> Result<Self> {
        let instance_count = instances.len() as u32;
        assert!(instance_count > 0, "GpuScene requires at least 1 instance");

        let vertex_bytes = cast_slice(vertices);
        let index_bytes = cast_slice(indices);
        let mesh_bytes = cast_slice(meshes);
        let instance_bytes = cast_slice(instances);

        let device = rhi.device.clone();

        let as_input_flag = if rhi.capabilities.khr_acceleration_structure {
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };

        let vertex_buffer = create_and_upload_gpu_buffer(
            rhi,
            "scene-vertices",
            vertex_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | as_input_flag,
        )?;
        let index_buffer = create_and_upload_gpu_buffer(
            rhi,
            "scene-indices",
            index_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | as_input_flag,
        )?;
        let mesh_buffer = create_and_upload_gpu_buffer(
            rhi,
            "scene-meshes",
            mesh_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let instance_buffer = create_and_upload_gpu_buffer(
            rhi,
            "scene-instances",
            instance_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let indirect_size =
            (instance_count as usize * std::mem::size_of::<DrawIndexedIndirectCommand>()) as vk::DeviceSize;
        let indirect_draw_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "scene-indirect-draws",
            indirect_size,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        )?;

        let mut draw_count_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for slot in 0..FRAMES_IN_FLIGHT {
            draw_count_vec.push(GpuBuffer::new(
                &device,
                rhi.allocator_mut(),
                &format!("scene-draw-count-{slot}"),
                std::mem::size_of::<u32>() as vk::DeviceSize,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?);
        }
        let draw_count_buffers: [GpuBuffer; FRAMES_IN_FLIGHT] = draw_count_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        let draw_ids_size = (instance_count as usize * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let draw_instance_ids_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "scene-draw-instance-ids",
            draw_ids_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        )?;

        let mut frame_uniforms_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for slot in 0..FRAMES_IN_FLIGHT {
            frame_uniforms_vec.push(GpuBuffer::new(
                &device,
                rhi.allocator_mut(),
                &format!("scene-frame-uniforms-{slot}"),
                std::mem::size_of::<FrameUniforms>() as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let frame_uniforms_buffers: [GpuBuffer; FRAMES_IN_FLIGHT] = frame_uniforms_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        Ok(Self {
            vertex_buffer,
            index_buffer,
            instance_buffer,
            mesh_buffer,
            indirect_draw_buffer,
            draw_count_buffers,
            draw_instance_ids_buffer,
            frame_uniforms_buffers,
            material_textures_buffer: None,
            default_sampler_idx: 0,
            instance_count,
        })
    }

    /// Write `FrameUniforms` into the ring-buffered host-visible BDA uniform buffer for `frame_slot`.
    pub fn update_frame_uniforms(
        &mut self,
        frame_slot: usize,
        uniforms: &FrameUniforms,
    ) -> Result<()> {
        let slot = frame_slot % FRAMES_IN_FLIGHT;
        let bytes = bytemuck::bytes_of(uniforms);
        let mapped = self.frame_uniforms_buffers[slot].mapped_slice_mut()?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Return the BDA address of `frame_uniforms_buffers` for `frame_slot`.
    #[inline]
    pub fn frame_uniforms_address(&self, frame_slot: usize) -> u64 {
        self.frame_uniforms_buffers[frame_slot % FRAMES_IN_FLIGHT].device_address
    }

    /// Return the `draw_count` buffer for `frame_slot`.
    #[inline]
    pub fn draw_count_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        &self.draw_count_buffers[frame_slot % FRAMES_IN_FLIGHT]
    }

    /// Read back the culled draw count written by `cull.comp` for `frame_slot`
    /// (call after the fence for `frame_slot` has completed).
    pub fn read_draw_count(&mut self, frame_slot: usize) -> Result<u32> {
        let slot = frame_slot % FRAMES_IN_FLIGHT;
        let mapped = self.draw_count_buffers[slot].mapped_slice_mut()?;
        Ok(*bytemuck::from_bytes::<u32>(&mapped[..4]))
    }

    /// Upload per-material bindless texture indices for `gbuffer.frag`.
    pub fn set_material_textures(
        &mut self,
        rhi: &mut RhiDevice,
        materials: &[GpuMaterialTextures],
        default_sampler_idx: u32,
    ) -> Result<()> {
        let device = rhi.device.clone();
        if let Some(old) = self.material_textures_buffer.take() {
            old.destroy(&device, rhi.allocator_mut())?;
        }
        self.default_sampler_idx = default_sampler_idx;
        if materials.is_empty() {
            return Ok(());
        }
        let bytes = cast_slice(materials);
        let mut buf = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "gbuffer_material_textures",
            bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        buf.mapped_slice_mut()?[..bytes.len()].copy_from_slice(bytes);
        self.material_textures_buffer = Some(buf);
        Ok(())
    }

    /// Return the BDA address of `material_textures_buffer` (or `0` if none).
    #[inline]
    pub fn material_textures_address(&self) -> u64 {
        self.material_textures_buffer
            .as_ref()
            .map_or(0, |b| b.device_address)
    }

    /// Destroy all GPU buffers owned by this scene.
    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        let alloc = rhi.allocator_mut();
        if let Some(mat_buf) = self.material_textures_buffer {
            mat_buf.destroy(&device, alloc)?;
        }
        self.vertex_buffer.destroy(&device, alloc)?;
        self.index_buffer.destroy(&device, alloc)?;
        self.instance_buffer.destroy(&device, alloc)?;
        self.mesh_buffer.destroy(&device, alloc)?;
        self.indirect_draw_buffer.destroy(&device, alloc)?;
        for buf in self.draw_count_buffers {
            buf.destroy(&device, alloc)?;
        }
        self.draw_instance_ids_buffer.destroy(&device, alloc)?;
        for buf in self.frame_uniforms_buffers {
            buf.destroy(&device, alloc)?;
        }
        Ok(())
    }
}

fn create_and_upload_gpu_buffer(
    rhi: &mut RhiDevice,
    name: &str,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<GpuBuffer> {
    let size = data.len() as vk::DeviceSize;
    let device = rhi.device.clone();

    let mut staging = GpuBuffer::new(
        &device,
        rhi.allocator_mut(),
        "staging-upload",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )?;
    staging.mapped_slice_mut()?[..data.len()].copy_from_slice(data);

    let dst = GpuBuffer::new(
        &device,
        rhi.allocator_mut(),
        name,
        size,
        usage,
        MemoryLocation::GpuOnly,
    )?;

    let pool = rhi.create_command_pool(
        rhi.queue_families.graphics,
        vk::CommandPoolCreateFlags::TRANSIENT,
    )?;
    let cmd = rhi.allocate_command_buffer(pool)?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device.begin_command_buffer(cmd, &begin_info)?;
        let region = [vk::BufferCopy::default().size(size)];
        device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &region);
        let barrier = [sync::buffer_barrier(
            dst.buffer,
            0,
            size,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
        )];
        sync::cmd_pipeline_barrier2(&device, cmd, &[], &barrier, &[]);
        device.end_command_buffer(cmd)?;
    }

    rhi.submit_and_wait(rhi.graphics_queue, cmd)?;
    unsafe { device.destroy_command_pool(pool, None) };
    staging.destroy(&device, rhi.allocator_mut())?;

    Ok(dst)
}
