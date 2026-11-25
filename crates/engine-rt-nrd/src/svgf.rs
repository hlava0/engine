//! Pure GPL-3.0 built-in SVGF spatiotemporal denoiser fallback.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use engine_rhi::{FRAMES_IN_FLIGHT, GpuImage, RhiDevice, load_shader_words, sync};

use crate::error::Result;
use crate::nrd::NrdInputs;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SvgfPushConstants {
    width: u32,
    height: u32,
    is_first_frame: u32,
    alpha: f32,
    denoising_range: f32,
    depth_sigma: f32,
    normal_power: f32,
    _pad: f32,
}

#[derive(Clone, Copy)]
struct TrackedImageState {
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

impl Default for TrackedImageState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access: vk::AccessFlags2::NONE,
        }
    }
}

/// Built-in GPL-3.0 spatiotemporal SVGF denoiser (reprojection EMA + 3x3 edge-stopping bilateral filter).
pub struct SvgfDenoiser {
    extent: vk::Extent2D,
    history_diff: [GpuImage; 2],
    history_spec: [GpuImage; 2],
    history_view_z: [GpuImage; 2],
    history_states: [TrackedImageState; 2],
    out_initialized: bool,

    nearest_sampler: vk::Sampler,
    linear_sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    history_parity: usize,
    frame_index: u32,
}

impl SvgfDenoiser {
    pub fn new(rhi: &mut RhiDevice, width: u32, height: u32) -> Result<Self> {
        let device = rhi.device.clone();

        let make_hist = |alloc: &mut engine_rhi::gpu_allocator::vulkan::Allocator,
                         fmt: vk::Format,
                         name: &str| {
            GpuImage::create_2d(
                &device,
                alloc,
                width,
                height,
                1,
                fmt,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                name,
            )
        };

        let history_diff = [
            make_hist(rhi.allocator_mut(), vk::Format::R16G16B16A16_SFLOAT, "svgf-hist-diff-0")?,
            make_hist(rhi.allocator_mut(), vk::Format::R16G16B16A16_SFLOAT, "svgf-hist-diff-1")?,
        ];
        let history_spec = [
            make_hist(rhi.allocator_mut(), vk::Format::R16G16B16A16_SFLOAT, "svgf-hist-spec-0")?,
            make_hist(rhi.allocator_mut(), vk::Format::R16G16B16A16_SFLOAT, "svgf-hist-spec-1")?,
        ];
        let history_view_z = [
            make_hist(rhi.allocator_mut(), vk::Format::R16_SFLOAT, "svgf-hist-viewz-0")?,
            make_hist(rhi.allocator_mut(), vk::Format::R16_SFLOAT, "svgf-hist-viewz-1")?,
        ];

        let nearest_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let linear_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        // Bindings 0..=7: COMBINED_IMAGE_SAMPLER (5 inputs + 3 history)
        // Bindings 8..=12: STORAGE_IMAGE (2 outputs + 3 next_history)
        let mut bindings = Vec::with_capacity(13);
        for b in 0..8u32 {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        for b in 8..13u32 {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 8 * FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 5 * FRAMES_IN_FLIGHT as u32,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        let layouts = [descriptor_set_layout; FRAMES_IN_FLIGHT];
        let descriptor_sets_vec = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };
        let descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = descriptor_sets_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SvgfPushConstants>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_constant_ranges(&pc_range),
                None,
            )?
        };

        let shader_words = load_shader_words("svgf_denoise.comp")?;
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shader_words),
                None,
            )?
        };
        let req_sg = rhi
            .capabilities
            .vendor_profile
            .required_compute_subgroup_size
            .filter(|&sz| 64u32.is_multiple_of(sz));
        let mut sg_info = rhi.subgroup_stage_create_info(vk::ShaderStageFlags::COMPUTE, req_sg);
        let stage_flags = rhi.subgroup_stage_flags(vk::ShaderStageFlags::COMPUTE, req_sg, 8);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(stage_flags)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        if let Some(ref mut sg) = sg_info {
            stage = stage.push_next(sg);
        }
        let pipe_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            let res = device.create_compute_pipelines(rhi.pipeline_cache, &pipe_info, None);
            device.destroy_shader_module(module, None);
            res.map_err(|(_, e)| e)?[0]
        };

        log::info!(
            "Built-in SVGF (GPL-3.0) spatiotemporal denoiser initialized ({width}x{height})"
        );

        Ok(Self {
            extent: vk::Extent2D { width, height },
            history_diff,
            history_spec,
            history_view_z,
            history_states: [TrackedImageState::default(); 2],
            out_initialized: false,
            nearest_sampler,
            linear_sampler,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            history_parity: 0,
            frame_index: 0,
        })
    }

    /// Record the SVGF spatiotemporal denoiser dispatch with `Synchronization2` barriers.
    ///
    /// Leaves `out_diff` and `out_spec` in `vk::ImageLayout::GENERAL` ready for `SHADER_STORAGE_READ`.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        inputs: &NrdInputs<'_>,
        out_diff: &GpuImage,
        out_spec: &GpuImage,
        denoising_range: f32,
        reset_history: bool,
        frame_slot: usize,
    ) -> Result<()> {
        let slot = frame_slot % FRAMES_IN_FLIGHT;
        let read_idx = self.history_parity & 1;
        let write_idx = 1 - read_idx;
        let dset = self.descriptor_sets[slot];

        let img_infos = [
            // 0: in_diff_rad_hitdist
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(inputs.diff_rad_hitdist.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 1: in_spec_rad_hitdist
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(inputs.spec_rad_hitdist.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 2: in_normal_roughness
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(inputs.normal_roughness.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 3: in_view_z
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(inputs.view_z.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 4: in_motion
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(inputs.motion.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 5: history_diff (bilinear for smooth reprojection)
            vk::DescriptorImageInfo::default()
                .sampler(self.linear_sampler)
                .image_view(self.history_diff[read_idx].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 6: history_spec (bilinear for smooth reprojection)
            vk::DescriptorImageInfo::default()
                .sampler(self.linear_sampler)
                .image_view(self.history_spec[read_idx].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 7: history_view_z (nearest for sharp disocclusion rejection)
            vk::DescriptorImageInfo::default()
                .sampler(self.nearest_sampler)
                .image_view(self.history_view_z[read_idx].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // 8: out_diff
            vk::DescriptorImageInfo::default()
                .image_view(out_diff.view)
                .image_layout(vk::ImageLayout::GENERAL),
            // 9: out_spec
            vk::DescriptorImageInfo::default()
                .image_view(out_spec.view)
                .image_layout(vk::ImageLayout::GENERAL),
            // 10: next_history_diff
            vk::DescriptorImageInfo::default()
                .image_view(self.history_diff[write_idx].view)
                .image_layout(vk::ImageLayout::GENERAL),
            // 11: next_history_spec
            vk::DescriptorImageInfo::default()
                .image_view(self.history_spec[write_idx].view)
                .image_layout(vk::ImageLayout::GENERAL),
            // 12: next_history_view_z
            vk::DescriptorImageInfo::default()
                .image_view(self.history_view_z[write_idx].view)
                .image_layout(vk::ImageLayout::GENERAL),
        ];

        let mut writes = Vec::with_capacity(img_infos.len());
        for (b, info) in img_infos.iter().enumerate() {
            let ty = if b < 8 {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            } else {
                vk::DescriptorType::STORAGE_IMAGE
            };
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(b as u32)
                    .descriptor_type(ty)
                    .image_info(std::slice::from_ref(info)),
            );
        }

        let read_state = self.history_states[read_idx];
        let write_state = self.history_states[write_idx];
        let (out_old_layout, out_src_stage, out_src_access) = if self.out_initialized {
            (
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        } else {
            (
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
            )
        };

        let pre_barriers = [
            sync::image_barrier(
                self.history_diff[read_idx].image,
                read_state.layout,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                read_state.stage,
                read_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.history_diff[read_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history_spec[read_idx].image,
                read_state.layout,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                read_state.stage,
                read_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.history_spec[read_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history_view_z[read_idx].image,
                read_state.layout,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                read_state.stage,
                read_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.history_view_z[read_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history_diff[write_idx].image,
                write_state.layout,
                vk::ImageLayout::GENERAL,
                write_state.stage,
                write_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.history_diff[write_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history_spec[write_idx].image,
                write_state.layout,
                vk::ImageLayout::GENERAL,
                write_state.stage,
                write_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.history_spec[write_idx].subresource_range(),
            ),
            sync::image_barrier(
                self.history_view_z[write_idx].image,
                write_state.layout,
                vk::ImageLayout::GENERAL,
                write_state.stage,
                write_state.access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.history_view_z[write_idx].subresource_range(),
            ),
            sync::image_barrier(
                out_diff.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                out_src_access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                out_diff.subresource_range(),
            ),
            sync::image_barrier(
                out_spec.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                out_src_access,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                out_spec.subresource_range(),
            ),
        ];

        let pc = SvgfPushConstants {
            width: self.extent.width,
            height: self.extent.height,
            is_first_frame: u32::from(reset_history || self.frame_index == 0),
            alpha: 0.15,
            denoising_range,
            depth_sigma: 0.05,
            normal_power: 32.0,
            _pad: 0.0,
        };

        let post_barriers = [
            sync::image_barrier(
                out_diff.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                out_diff.subresource_range(),
            ),
            sync::image_barrier(
                out_spec.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                out_spec.subresource_range(),
            ),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &pre_barriers);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[dset],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_dispatch(
                cmd,
                self.extent.width.div_ceil(8),
                self.extent.height.div_ceil(8),
                1,
            );
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &post_barriers);
        }

        self.history_states[read_idx] = TrackedImageState {
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_SAMPLED_READ,
        };
        self.history_states[write_idx] = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
        };
        self.out_initialized = true;
        self.history_parity ^= 1;
        self.frame_index += 1;

        Ok(())
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.nearest_sampler, None);
            device.destroy_sampler(self.linear_sampler, None);
        }
        let alloc = rhi.allocator_mut();
        for img in self.history_diff {
            img.destroy(&device, alloc)?;
        }
        for img in self.history_spec {
            img.destroy(&device, alloc)?;
        }
        for img in self.history_view_z {
            img.destroy(&device, alloc)?;
        }
        Ok(())
    }
}
