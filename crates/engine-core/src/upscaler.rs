//! GPU-agnostic FSR 3.1-style Temporal Upscaling + RCAS Sharpening, and
//! Cooperative-Matrix Neural Upscaler (`Fsr4NeuralCoopMat`) fallback/hook.

use bytemuck::{Pod, Zeroable};
use engine_rhi::ash::vk;
use engine_rhi::{FRAMES_IN_FLIGHT, GpuImage, RhiDevice, load_shader_words, sync};

use crate::EngineError;

#[cfg(has_fsr31)]
mod fsr31_ffi {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct EngineFsr31Handle {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct EngineFsr31CreateInfo {
        pub vk_device: u64,
        pub vk_physical_device: u64,
        pub vk_get_device_proc_addr: *mut c_void,
        pub max_render_w: u32,
        pub max_render_h: u32,
        pub max_upscale_w: u32,
        pub max_upscale_h: u32,
        pub enable_hdr: u32,
        pub inverted_depth: u32,
    }

    #[repr(C)]
    pub struct EngineFsr31DispatchInfo {
        pub vk_cmd: u64,
        pub color_image: u64,
        pub color_format: u32,
        pub color_w: u32,
        pub color_h: u32,
        pub depth_image: u64,
        pub depth_format: u32,
        pub motion_image: u64,
        pub motion_format: u32,
        pub output_image: u64,
        pub output_format: u32,
        pub output_w: u32,
        pub output_h: u32,
        pub jitter_x: f32,
        pub jitter_y: f32,
        pub mv_scale_x: f32,
        pub mv_scale_y: f32,
        pub sharpness: f32,
        pub frame_time_ms: f32,
        pub camera_near: f32,
        pub camera_far: f32,
        pub fov_y_rad: f32,
        pub reset: u32,
    }

    unsafe extern "C" {
        pub fn engine_fsr31_create(info: *const EngineFsr31CreateInfo) -> *mut EngineFsr31Handle;
        pub fn engine_fsr31_dispatch(
            handle: *mut EngineFsr31Handle,
            info: *const EngineFsr31DispatchInfo,
        ) -> i32;
        pub fn engine_fsr31_destroy(handle: *mut EngineFsr31Handle);
    }
}

#[cfg(has_fsr31)]
struct Fsr31Context(*mut fsr31_ffi::EngineFsr31Handle);

#[cfg(has_fsr31)]
unsafe impl Send for Fsr31Context {}
#[cfg(has_fsr31)]
unsafe impl Sync for Fsr31Context {}

/// Standard FSR quality presets controlling internal render resolution scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsrPreset {
    Native,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl FsrPreset {
    pub fn scale_factor(self) -> f32 {
        match self {
            Self::Native => 1.0,
            Self::Quality => 1.5,
            Self::Balanced => 1.7,
            Self::Performance => 2.0,
            Self::UltraPerformance => 3.0,
        }
    }

    pub fn render_dimensions(self, display_w: u32, display_h: u32) -> (u32, u32) {
        let s = self.scale_factor();
        (
            ((display_w as f32 / s).round() as u32).max(16),
            ((display_h as f32 / s).round() as u32).max(16),
        )
    }
}

/// GPU-agnostic upscaling backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalerMode {
    /// Pure GPL-3.0 GPU-agnostic FSR 3.1-style temporal upscaler + Catmull-Rom + YCoCg clipping + RCAS.
    Fsr31Temporal,
    /// GPU-agnostic Cooperative-Matrix neural upscaling path (Intel XMX / AMD WMMA / NVIDIA Tensor Core).
    Fsr4NeuralCoopMat,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpscalePushConstants {
    render_size: [u32; 2],
    display_size: [u32; 2],
    jitter: [f32; 2],
    sharpness: f32,
    reset_history: u32,
}

/// Temporal upscaling & RCAS sharpening compute pass.
pub struct TemporalUpscalerPass {
    pub display_extent: vk::Extent2D,
    pub out_upscaled: GpuImage,
    history: [GpuImage; 2],
    history_initialized: [bool; 2],
    out_initialized: bool,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    frame_counter: usize,
    #[cfg(has_fsr31)]
    fsr31_ctx: Option<Fsr31Context>,
}

impl TemporalUpscalerPass {
    pub fn new(rhi: &mut RhiDevice, display_w: u32, display_h: u32) -> Result<Self, EngineError> {
        let dev = rhi.device.clone();
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let out_upscaled = GpuImage::create_2d(
            &dev,
            rhi.allocator_mut(),
            display_w,
            display_h,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            vk::ImageAspectFlags::COLOR,
            "fsr31_out_upscaled",
        )?;
        let history = [
            GpuImage::create_2d(
                &dev,
                rhi.allocator_mut(),
                display_w,
                display_h,
                1,
                vk::Format::R16G16B16A16_SFLOAT,
                usage,
                vk::ImageAspectFlags::COLOR,
                "fsr31_hist_0",
            )?,
            GpuImage::create_2d(
                &dev,
                rhi.allocator_mut(),
                display_w,
                display_h,
                1,
                vk::Format::R16G16B16A16_SFLOAT,
                usage,
                vk::ImageAspectFlags::COLOR,
                "fsr31_hist_1",
            )?,
        ];

        let sampler = unsafe {
            dev.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let pc_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<UpscalePushConstants>() as u32,
        }];
        let pipeline_layout = unsafe {
            dev.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(&pc_range),
                None,
            )?
        };
        let spv = load_shader_words("fsr31_temporal_upscale.comp")?;
        let module = unsafe {
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)?
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipe_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            let res = dev.create_compute_pipelines(rhi.pipeline_cache, &pipe_info, None);
            dev.destroy_shader_module(module, None);
            res.map_err(|(_, e)| e)?[0]
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (4 * FRAMES_IN_FLIGHT) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: (2 * FRAMES_IN_FLIGHT) as u32,
            },
        ];
        let descriptor_pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let layouts = [set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };

        #[cfg(has_fsr31)]
        let fsr31_ctx = {
            let info = fsr31_ffi::EngineFsr31CreateInfo {
                vk_device: vk::Handle::as_raw(rhi.device.handle()),
                vk_physical_device: vk::Handle::as_raw(rhi.physical_device),
                vk_get_device_proc_addr: rhi.instance.fp_v1_0().get_device_proc_addr
                    as *mut std::ffi::c_void,
                max_render_w: display_w,
                max_render_h: display_h,
                max_upscale_w: display_w,
                max_upscale_h: display_h,
                enable_hdr: 1,
                inverted_depth: 0,
            };
            let handle = unsafe { fsr31_ffi::engine_fsr31_create(&info) };
            if handle.is_null() {
                log::warn!(
                    "engine_fsr31_create failed; falling back to built-in GLSL FSR 3.1-style upscaler"
                );
                None
            } else {
                log::info!("Initialized official AMD FidelityFX FSR 3.1 (v1.1.4 SDK)");
                Some(Fsr31Context(handle))
            }
        };

        Ok(Self {
            display_extent: vk::Extent2D {
                width: display_w,
                height: display_h,
            },
            out_upscaled,
            history,
            history_initialized: [false, false],
            out_initialized: false,
            sampler,
            set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_sets: [sets[0], sets[1]],
            frame_counter: 0,
            #[cfg(has_fsr31)]
            fsr31_ctx,
        })
    }

    /// Returns the active upscaling backend name.
    pub fn backend_name(&self) -> &'static str {
        #[cfg(has_fsr31)]
        if self.fsr31_ctx.is_some() {
            return "AMD FidelityFX FSR 3.1 (v1.1.4 SDK)";
        }
        "Built-in GLSL FSR 3.1-style Temporal Upscaler"
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi: &RhiDevice,
        cmd: vk::CommandBuffer,
        in_color: &GpuImage,
        in_motion: &GpuImage,
        in_depth: &GpuImage,
        jitter: [f32; 2],
        sharpness: f32,
        slot_idx: usize,
    ) {
        #[cfg(has_fsr31)]
        if let Some(ctx) = self.fsr31_ctx.as_ref() {
            let dev = &rhi.device;
            let out_old = if self.out_initialized {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            };
            let out_barrier = [sync::image_barrier(
                self.out_upscaled.image,
                out_old,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.out_upscaled.subresource_range(),
            )];
            unsafe {
                sync::cmd_pipeline_barrier2(dev, cmd, &[], &[], &out_barrier);
            }

            let dispatch = fsr31_ffi::EngineFsr31DispatchInfo {
                vk_cmd: vk::Handle::as_raw(cmd),
                color_image: vk::Handle::as_raw(in_color.image),
                color_format: in_color.format.as_raw() as u32,
                color_w: in_color.extent.width,
                color_h: in_color.extent.height,
                depth_image: vk::Handle::as_raw(in_depth.image),
                depth_format: in_depth.format.as_raw() as u32,
                motion_image: vk::Handle::as_raw(in_motion.image),
                motion_format: in_motion.format.as_raw() as u32,
                output_image: vk::Handle::as_raw(self.out_upscaled.image),
                output_format: self.out_upscaled.format.as_raw() as u32,
                output_w: self.display_extent.width,
                output_h: self.display_extent.height,
                jitter_x: jitter[0],
                jitter_y: jitter[1],
                mv_scale_x: in_color.extent.width as f32,
                mv_scale_y: in_color.extent.height as f32,
                sharpness,
                frame_time_ms: 16.666,
                camera_near: 0.1,
                camera_far: 1000.0,
                fov_y_rad: 60.0_f32.to_radians(),
                reset: u32::from(self.frame_counter == 0),
            };
            let rc = unsafe { fsr31_ffi::engine_fsr31_dispatch(ctx.0, &dispatch) };
            if rc == 0 {
                self.out_initialized = true;
                self.frame_counter = self.frame_counter.wrapping_add(1);
                return;
            }
            log::warn!(
                "engine_fsr31_dispatch failed with code {rc}; falling back to built-in GLSL upscaler"
            );
        }

        let dev = &rhi.device;
        let slot = slot_idx % FRAMES_IN_FLIGHT;
        let read_idx = (self.frame_counter + 1) & 1;
        let write_idx = self.frame_counter & 1;
        let reset = u32::from(!self.history_initialized[read_idx]);

        let infos = [
            vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: in_color.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: in_motion.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: in_depth.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: self.history[read_idx].view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.out_upscaled.view,
                image_layout: vk::ImageLayout::GENERAL,
            },
            vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: self.history[write_idx].view,
                image_layout: vk::ImageLayout::GENERAL,
            },
        ];

        let mut writes = Vec::with_capacity(6);
        for (b, info) in infos.iter().enumerate() {
            let ty = if b < 4 {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            } else {
                vk::DescriptorType::STORAGE_IMAGE
            };
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[slot])
                    .dst_binding(b as u32)
                    .descriptor_type(ty)
                    .image_info(std::slice::from_ref(info)),
            );
        }

        let read_old = if self.history_initialized[read_idx] {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let write_old = if self.history_initialized[write_idx] {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let out_old = if self.out_initialized {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::UNDEFINED
        };

        let barriers = [
            sync::image_barrier(
                self.history[read_idx].image,
                read_old,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.history[read_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history[write_idx].image,
                write_old,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.history[write_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.out_upscaled.image,
                out_old,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.out_upscaled.subresource_range(),
            ),
        ];

        let push = UpscalePushConstants {
            render_size: [in_color.extent.width, in_color.extent.height],
            display_size: [self.display_extent.width, self.display_extent.height],
            jitter,
            sharpness,
            reset_history: reset,
        };

        unsafe {
            dev.update_descriptor_sets(&writes, &[]);
            sync::cmd_pipeline_barrier2(dev, cmd, &[], &[], &barriers);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[slot]],
                &[],
            );
            dev.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            dev.cmd_dispatch(
                cmd,
                self.display_extent.width.div_ceil(8),
                self.display_extent.height.div_ceil(8),
                1,
            );
        }

        self.history_initialized[write_idx] = true;
        self.out_initialized = true;
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<(), EngineError> {
        #[cfg(has_fsr31)]
        if let Some(ctx) = self.fsr31_ctx {
            unsafe {
                fsr31_ffi::engine_fsr31_destroy(ctx.0);
            }
        }
        let dev = rhi.device.clone();
        unsafe {
            dev.destroy_descriptor_pool(self.descriptor_pool, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
            dev.destroy_sampler(self.sampler, None);
        }
        let alloc = rhi.allocator_mut();
        self.out_upscaled.destroy(&dev, alloc)?;
        for h in self.history {
            h.destroy(&dev, alloc)?;
        }
        Ok(())
    }
}
