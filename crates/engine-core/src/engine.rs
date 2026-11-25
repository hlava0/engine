//! Main GPU-driven Vulkan 1.4 + RT + NRD + CoopML engine loop with 2-frames-in-flight pipelining.

use std::time::Instant;

use engine_compute_ml::CoopMlPipeline;
use engine_gpu_driven::{GpuDrivenPipeline, GpuInstanceData, GpuScene};
use engine_rhi::ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{
    BindlessConfig, BindlessTable, CoopMatrixTile, FRAMES_IN_FLIGHT, GpuBuffer, GpuImage,
    RhiConfig, RhiDevice, VendorProfile, sync,
};
use engine_rt_nrd::{
    AccelStructureManager, NrdComposePass, NrdDenoiser, NrdInputs, RtGiPass, RtParams,
};

use crate::EngineError;
use crate::profiler::{GpuProfiler, PassTiming};
use crate::scene::SceneWorld;

const DENOISING_RANGE: f32 = 1000.0;

/// Per-frame CPU & GPU execution statistics.
///
/// With 2-frames-in-flight async pipelining, `draw_count` and `gpu_timings` reflect the
/// most recently completed GPU frame for this slot (`frame_index - FRAMES_IN_FLIGHT`),
/// avoiding any post-submit CPU stall.
#[derive(Debug, Clone)]
pub struct FrameStats {
    pub frame_index: u32,
    pub cpu_record_us: f64,
    pub draw_count: u32,
    pub gpu_timings: Vec<PassTiming>,
}

struct FrameSlot {
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    profiler: GpuProfiler,
    submitted: bool,
}

/// Top-level headless/render engine owning all Vulkan 1.4 subsystems.
pub struct Engine {
    rhi: RhiDevice,
    bindless: Option<BindlessTable>,
    gbuffer_pipe: Option<GpuDrivenPipeline>,
    gpu_scene: Option<GpuScene>,
    accel: Option<AccelStructureManager>,
    rt_pass: Option<RtGiPass>,
    nrd: Option<NrdDenoiser>,
    compose: Option<NrdComposePass>,
    coop_ml: Option<CoopMlPipeline>,
    output_image: Option<GpuImage>,
    frames: Vec<FrameSlot>,
    gpu_instances_scratch: Vec<GpuInstanceData>,
    pub world: SceneWorld,
    pub width: u32,
    pub height: u32,
    frame_index: u32,
    has_hiz_history: bool,
}

impl Engine {
    /// Initialize all engine subsystems for the given `SceneWorld` and resolution.
    pub fn new(
        world: SceneWorld,
        width: u32,
        height: u32,
        enable_validation: bool,
    ) -> Result<Self, EngineError> {
        let mut rhi = RhiDevice::new(RhiConfig {
            app_name: "vulkan-gpu-driven-engine".to_string(),
            enable_validation,
            enable_sync_validation: enable_validation,
        })?;

        let bindless = BindlessTable::new(&rhi.device, BindlessConfig::default())?;
        let gbuffer_pipe = GpuDrivenPipeline::new(&mut rhi, &bindless, width, height)?;

        let (vertices, _, meshes) = world.flatten_geometry();
        let gpu_instances_scratch = world.gpu_instances();
        let gpu_scene = world.build_gpu_scene(&mut rhi)?;

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let command_pool = rhi.create_command_pool(
                rhi.queue_families.graphics,
                vk::CommandPoolCreateFlags::TRANSIENT,
            )?;
            let cmd = rhi.allocate_command_buffer(command_pool)?;
            let fence = unsafe {
                rhi.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            };
            let profiler = GpuProfiler::new(&rhi, 16)?;
            frames.push(FrameSlot {
                command_pool,
                cmd,
                fence,
                profiler,
                submitted: false,
            });
        }

        // Build initial BLAS + TLAS using slot 0's command buffer before the async loop starts
        let init_cmd = frames[0].cmd;
        let init_pool = frames[0].command_pool;
        unsafe {
            rhi.device.begin_command_buffer(
                init_cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let accel = AccelStructureManager::build_initial(
            &mut rhi,
            init_cmd,
            &gpu_scene,
            &meshes,
            &gpu_instances_scratch,
            vertices.len() as u32,
        )?;
        unsafe { rhi.device.end_command_buffer(init_cmd)? };
        rhi.submit_and_wait(rhi.graphics_queue, init_cmd)?;
        unsafe {
            rhi.device
                .reset_command_pool(init_pool, vk::CommandPoolResetFlags::empty())?;
        }

        let materials = world.gpu_materials();
        let lights = world.gpu_lights();
        let rt_pass = RtGiPass::new(&mut rhi, width, height, &materials, &lights)?;
        let nrd = NrdDenoiser::new(&mut rhi, width, height)?;
        let compose = NrdComposePass::new(&mut rhi, width, height)?;

        rt_pass.bind_resources(&rhi.device, &accel, &gbuffer_pipe.targets);
        compose.bind_resources(&rhi.device, &nrd, &rt_pass, &gbuffer_pipe.targets);

        let coop_ml = CoopMlPipeline::new(&mut rhi, false)?;

        let dev = rhi.device.clone();
        let output_image = GpuImage::create_2d(
            &dev,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "engine_final_output",
        )?;

        Ok(Self {
            rhi,
            bindless: Some(bindless),
            gbuffer_pipe: Some(gbuffer_pipe),
            gpu_scene: Some(gpu_scene),
            accel: Some(accel),
            rt_pass: Some(rt_pass),
            nrd: Some(nrd),
            compose: Some(compose),
            coop_ml: Some(coop_ml),
            output_image: Some(output_image),
            frames,
            gpu_instances_scratch,
            world,
            width,
            height,
            frame_index: 0,
            has_hiz_history: false,
        })
    }

    /// Wait for all in-flight GPU work to finish.
    pub fn wait_idle(&self) -> Result<(), EngineError> {
        self.rhi.wait_idle()?;
        Ok(())
    }

    /// Returns `true` if `CoopMlPipeline` is using hardware `VK_KHR_cooperative_matrix`.
    pub fn uses_cooperative_matrix(&self) -> bool {
        self.coop_ml
            .as_ref()
            .is_some_and(CoopMlPipeline::uses_cooperative_matrix)
    }

    /// Returns the active [`CoopMatrixTile`] selected for `CoopMlPipeline`.
    pub fn coop_ml_tile(&self) -> CoopMatrixTile {
        self.coop_ml
            .as_ref()
            .map_or(CoopMatrixTile::Fallback, CoopMlPipeline::active_tile)
    }

    /// Returns the probed [`VendorProfile`] of the active Vulkan device.
    pub fn vendor_profile(&self) -> &VendorProfile {
        &self.rhi.capabilities.vendor_profile
    }

    /// Returns the physical device name string.
    pub fn device_name(&self) -> &str {
        &self.rhi.capabilities.device_name
    }

    /// Number of Vulkan validation errors observed so far.
    pub fn validation_errors(&self) -> u32 {
        self.rhi.validation_errors()
    }

    /// Execute one full pipelined frame (Scene sync -> Accel -> Cull+GBuffer+HiZ -> RT GI -> NRD -> Compose -> CoopML).
    ///
    /// Uses 2-frames-in-flight async pipelining: waits only on frame `N - 2`'s fence, reads back
    /// its completed GPU timestamps & draw count without stalling, records frame `N`, and submits
    /// asynchronously via `vkQueueSubmit2`.
    pub fn render_frame(&mut self, dt: f32) -> Result<FrameStats, EngineError> {
        let frame_idx = self.frame_index;
        let slot_idx = (frame_idx as usize) % FRAMES_IN_FLIGHT;
        let is_first = frame_idx == 0;

        // 1. Wait only on this slot's fence (frame N - FRAMES_IN_FLIGHT) and reset pool
        let fence = self.frames[slot_idx].fence;
        let pool = self.frames[slot_idx].command_pool;
        let cmd = self.frames[slot_idx].cmd;
        unsafe {
            self.rhi.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.rhi.device.reset_fences(&[fence])?;
            self.rhi
                .device
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
        }

        // 2. Non-blocking readback of completed GPU timestamps & draw count from this slot
        let gpu_scene = self.gpu_scene.as_mut().expect("gpu_scene");
        let (draw_count, gpu_timings) = if self.frames[slot_idx].submitted {
            let timings = self.frames[slot_idx].profiler.readback(&self.rhi.device)?;
            let draws = gpu_scene.read_draw_count(slot_idx)?;
            (draws, timings)
        } else {
            (0, Vec::new())
        };

        // 3. Step simulation and fill pre-allocated scratch buffer (zero heap allocation)
        self.world.step(dt);
        let any_transform_dirty = self.world.dirty_instances;
        if any_transform_dirty {
            self.world
                .fill_gpu_instances(&mut self.gpu_instances_scratch);
        }

        let record_start = Instant::now();

        unsafe {
            self.rhi.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let uniforms = self.world.sync_to_gpu(
            &self.rhi.device,
            cmd,
            gpu_scene,
            &self.gpu_instances_scratch,
            self.width,
            self.height,
            slot_idx,
            is_first,
        )?;

        let rt_params = RtParams {
            inv_view: self.world.camera.inv_view,
            inv_proj: self.world.camera.inv_proj,
            inv_view_proj: self.world.camera.inv_view_proj,
            cam_pos_range: [
                self.world.camera.position[0],
                self.world.camera.position[1],
                self.world.camera.position[2],
                DENOISING_RANGE,
            ],
            jitter_frame: [
                self.world.camera.jitter[0] * 2.0,
                self.world.camera.jitter[1] * 2.0,
                frame_idx as f32,
                1.0,
            ],
            screen_size: [self.width, self.height],
            light_count: self.world.lights.len() as u32,
            _pad: 0,
        };

        let profiler = &mut self.frames[slot_idx].profiler;
        profiler.reset(&self.rhi.device, cmd);

        // 1. Accel (TLAS refit/rebuild for animated instances; skips cleanly on static frames)
        let p_accel = profiler.begin_pass(&self.rhi.device, cmd, "Accel");
        if !is_first {
            self.accel.as_mut().expect("accel").update_tlas(
                &self.rhi,
                cmd,
                &self.gpu_instances_scratch,
                slot_idx,
                any_transform_dirty,
            )?;
        }
        profiler.end_pass(&self.rhi.device, cmd, p_accel);

        // 2. GPU-driven Cull + GBuffer + Hi-Z
        let bindless = self.bindless.as_ref().expect("bindless");
        let gbuffer_pipe = self.gbuffer_pipe.as_mut().expect("gbuffer_pipe");
        let p_gbuf = profiler.begin_pass(&self.rhi.device, cmd, "Cull+GBuffer+HiZ");
        gbuffer_pipe.record_frame(
            &self.rhi,
            bindless,
            cmd,
            gpu_scene,
            &uniforms,
            self.has_hiz_history,
            slot_idx,
        )?;
        profiler.end_pass(&self.rhi.device, cmd, p_gbuf);

        // 3. Hardware Ray Traced GI + Reflections
        let rt_pass = self.rt_pass.as_mut().expect("rt_pass");
        let p_rt = profiler.begin_pass(&self.rhi.device, cmd, "RayTraceGI");
        rt_pass.record(&self.rhi.device, cmd, &gbuffer_pipe.targets, &rt_params, slot_idx)?;
        profiler.end_pass(&self.rhi.device, cmd, p_rt);

        // 4. NVIDIA NRD Denoiser
        let nrd = self.nrd.as_mut().expect("nrd");
        let nrd_inputs = NrdInputs {
            diff_rad_hitdist: &rt_pass.nrd_diff_rad_hitdist,
            spec_rad_hitdist: &rt_pass.nrd_spec_rad_hitdist,
            normal_roughness: &rt_pass.nrd_normal_roughness,
            view_z: &rt_pass.nrd_view_z,
            motion: &gbuffer_pipe.targets.motion,
            diff_confidence: &rt_pass.nrd_diff_confidence,
            spec_confidence: &rt_pass.nrd_spec_confidence,
        };
        let p_nrd = profiler.begin_pass(&self.rhi.device, cmd, "NRD");
        nrd.record(
            &self.rhi.device,
            cmd,
            &nrd_inputs,
            self.world.camera.proj,
            self.world.camera.prev_proj,
            self.world.camera.view,
            self.world.camera.prev_view,
            self.world.camera.jitter,
            self.world.camera.prev_jitter,
            DENOISING_RANGE,
            (dt * 1000.0).max(0.1),
            is_first,
            slot_idx,
        )?;
        profiler.end_pass(&self.rhi.device, cmd, p_nrd);

        // 5. NRD Remodulation & Sky Composition
        let compose = self.compose.as_mut().expect("compose");
        let p_comp = profiler.begin_pass(&self.rhi.device, cmd, "Compose");
        compose.record(&self.rhi.device, cmd, rt_pass, slot_idx);
        profiler.end_pass(&self.rhi.device, cmd, p_comp);

        // 6. Cooperative Matrix ML Post-Processing & ACES Tone-Mapping
        let coop_ml = self.coop_ml.as_mut().expect("coop_ml");
        let out_img = self.output_image.as_ref().expect("output_image");
        let p_ml = profiler.begin_pass(&self.rhi.device, cmd, "CoopML");
        coop_ml.dispatch_slot(
            &self.rhi.device,
            cmd,
            &compose.out_hdr,
            out_img,
            self.width,
            self.height,
            slot_idx,
        );
        profiler.end_pass(&self.rhi.device, cmd, p_ml);

        unsafe { self.rhi.device.end_command_buffer(cmd)? };
        let cpu_record_us = record_start.elapsed().as_secs_f64() * 1_000_000.0;

        // 4. Submit asynchronously via vkQueueSubmit2 signaling this slot's fence
        let cmd_submit = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit_info = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_submit)];
        unsafe {
            self.rhi
                .device
                .queue_submit2(self.rhi.graphics_queue, &submit_info, fence)?;
        }

        self.frames[slot_idx].submitted = true;
        self.frame_index += 1;
        self.has_hiz_history = true;

        Ok(FrameStats {
            frame_index: frame_idx,
            cpu_record_us,
            draw_count,
            gpu_timings,
        })
    }

    /// Read back the final post-processed `output_image` as `[r, g, b, a]` `f32` pixels.
    ///
    /// Automatically drains all in-flight GPU work via `wait_idle()` before copying.
    pub fn readback_output_rgba(&mut self) -> Result<Vec<[f32; 4]>, EngineError> {
        self.wait_idle()?;

        let pixel_count = (self.width * self.height) as usize;
        let byte_size = (pixel_count * 8) as vk::DeviceSize;
        let dev = self.rhi.device.clone();

        let staging = GpuBuffer::new(
            &dev,
            self.rhi.allocator_mut(),
            "engine_output_readback",
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;

        let out_img = self.output_image.as_ref().expect("output_image");
        let pool = self.frames[0].command_pool;
        let cmd = self.frames[0].cmd;
        unsafe {
            dev.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let to_src = [sync::image_barrier(
                out_img.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                out_img.subresource_range(),
            )];
            sync::cmd_pipeline_barrier2(&dev, cmd, &[], &[], &to_src);

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            dev.cmd_copy_image_to_buffer(
                cmd,
                out_img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.buffer,
                &[region],
            );

            let to_gen = [sync::image_barrier(
                out_img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::NONE,
                out_img.subresource_range(),
            )];
            sync::cmd_pipeline_barrier2(&dev, cmd, &[], &[], &to_gen);
            dev.end_command_buffer(cmd)?;
        }

        self.rhi.submit_and_wait(self.rhi.graphics_queue, cmd)?;

        let bytes = staging
            .allocation
            .as_ref()
            .expect("staging allocation")
            .mapped_slice()
            .expect("mapped_slice");
        let words: &[u16] = bytemuck::cast_slice(bytes);
        let mut rgba = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            rgba.push([
                f16_to_f32(words[i * 4]),
                f16_to_f32(words[i * 4 + 1]),
                f16_to_f32(words[i * 4 + 2]),
                f16_to_f32(words[i * 4 + 3]),
            ]);
        }

        staging.destroy(&dev, self.rhi.allocator_mut())?;
        Ok(rgba)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        let dev = self.rhi.device.clone();

        for slot in self.frames.drain(..) {
            slot.profiler.destroy(&dev);
            unsafe {
                dev.destroy_fence(slot.fence, None);
                dev.destroy_command_pool(slot.command_pool, None);
            }
        }
        if let Some(img) = self.output_image.take() {
            let _ = img.destroy(&dev, self.rhi.allocator_mut());
        }
        if let Some(mut ml) = self.coop_ml.take() {
            let _ = ml.destroy(&mut self.rhi);
        }
        if let Some(c) = self.compose.take() {
            let _ = c.destroy(&mut self.rhi);
        }
        if let Some(n) = self.nrd.take() {
            let _ = n.destroy(&mut self.rhi);
        }
        if let Some(rt) = self.rt_pass.take() {
            let _ = rt.destroy(&mut self.rhi);
        }
        if let Some(a) = self.accel.take() {
            let _ = a.destroy(&mut self.rhi);
        }
        if let Some(s) = self.gpu_scene.take() {
            let _ = s.destroy(&mut self.rhi);
        }
        if let Some(g) = self.gbuffer_pipe.take() {
            let _ = g.destroy(&mut self.rhi);
        }
        if let Some(b) = self.bindless.take() {
            b.destroy(&dev);
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 127 - 14;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    f32::from_bits((sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13))
}
