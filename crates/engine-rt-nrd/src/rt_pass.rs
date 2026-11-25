use ash::vk;
use bytemuck::cast_slice;
use engine_gpu_driven::GBufferTargets;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{FRAMES_IN_FLIGHT, GpuBuffer, GpuImage, RhiDevice, load_shader_words, sync};

use crate::accel::AccelStructureManager;
use crate::error::Result;
use crate::types::{GpuRtLight, GpuRtMaterial, RtGiPushConstants, RtParams};

/// Hardware ray-query GI + specular reflection + shadow pass producing demodulated NRD inputs.
pub struct RtGiPass {
    pub extent: vk::Extent2D,
    pub nrd_diff_rad_hitdist: GpuImage,
    pub nrd_spec_rad_hitdist: GpuImage,
    pub nrd_normal_roughness: GpuImage,
    pub nrd_view_z: GpuImage,
    pub nrd_diff_confidence: GpuImage,
    pub nrd_spec_confidence: GpuImage,

    pub params_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
    pub materials_buffer: GpuBuffer,
    pub lights_buffer: GpuBuffer,

    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    first_frame: bool,
}

impl RtGiPass {
    pub fn new(
        rhi: &mut RhiDevice,
        width: u32,
        height: u32,
        materials: &[GpuRtMaterial],
        lights: &[GpuRtLight],
    ) -> Result<Self> {
        let device = rhi.device.clone();

        let nrd_diff_rad_hitdist = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-diff-rad-hitdist",
        )?;
        let nrd_spec_rad_hitdist = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-spec-rad-hitdist",
        )?;
        let nrd_normal_roughness = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-normal-roughness",
        )?;
        let nrd_view_z = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-view-z",
        )?;
        let nrd_diff_confidence = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-diff-confidence",
        )?;
        let nrd_spec_confidence = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-in-spec-confidence",
        )?;

        let mut params_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for slot in 0..FRAMES_IN_FLIGHT {
            params_vec.push(GpuBuffer::new(
                &device,
                rhi.allocator_mut(),
                &format!("rt-params-{slot}"),
                std::mem::size_of::<RtParams>() as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let params_buffers: [GpuBuffer; FRAMES_IN_FLIGHT] =
            params_vec.try_into().unwrap_or_else(|_| unreachable!());

        let default_mat = [GpuRtMaterial::default()];
        let mat_slice = if materials.is_empty() { &default_mat } else { materials };
        let mut materials_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "rt-materials",
            std::mem::size_of_val(mat_slice) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        materials_buffer.mapped_slice_mut()?.copy_from_slice(cast_slice(mat_slice));

        let default_light = [GpuRtLight::default()];
        let light_slice = if lights.is_empty() { &default_light } else { lights };
        let mut lights_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "rt-lights",
            std::mem::size_of_val(light_slice) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        lights_buffer.mapped_slice_mut()?.copy_from_slice(cast_slice(light_slice));

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
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
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 6,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)? }[0];

        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RtGiPushConstants>() as u32)];
        let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(&pc_range);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipe_layout_info, None)? };

        let shader_words = load_shader_words("rt_gi.comp")?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&shader_words), None)?
        };
        let req_sg = rhi
            .capabilities
            .vendor_profile
            .required_rt_subgroup_size
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

        Ok(Self {
            extent: vk::Extent2D { width, height },
            nrd_diff_rad_hitdist,
            nrd_spec_rad_hitdist,
            nrd_normal_roughness,
            nrd_view_z,
            nrd_diff_confidence,
            nrd_spec_confidence,
            params_buffers,
            materials_buffer,
            lights_buffer,
            sampler,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            first_frame: true,
        })
    }

    /// Update the pre-allocated descriptor set to point at `tlas` and `gbuffer` (zero allocations).
    pub fn bind_resources(
        &self,
        device: &ash::Device,
        accel: &AccelStructureManager,
        gbuffer: &GBufferTargets,
    ) {
        let mut accel_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&accel.tlas));

        let gbuf_albedo_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.albedo_metallic.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let gbuf_normal_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.normal_roughness.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let gbuf_viewz_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.view_z.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let gbuf_depth_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.depth.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let out_diff_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_diff_rad_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_spec_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_spec_rad_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_nr_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_normal_roughness.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_vz_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_view_z.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_dc_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_diff_confidence.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_sc_info = [vk::DescriptorImageInfo::default()
            .image_view(self.nrd_spec_confidence.view)
            .image_layout(vk::ImageLayout::GENERAL)];

        let mut accel_desc = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_write);
        accel_desc.descriptor_count = 1;

        let writes = [
            accel_desc,
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&gbuf_albedo_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&gbuf_normal_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&gbuf_viewz_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&gbuf_depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_diff_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_spec_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_nr_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_vz_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_dc_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_sc_info),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Return the BDA address of `params_buffers` for `frame_slot`.
    #[inline]
    pub fn params_device_address(&self, frame_slot: usize) -> u64 {
        self.params_buffers[frame_slot % FRAMES_IN_FLIGHT].device_address
    }

    /// Update `RtParams` for `frame_slot` and record G-buffer transitions + `rt_gi.comp` dispatch.
    pub fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        gbuffer: &GBufferTargets,
        params: &RtParams,
        frame_slot: usize,
    ) -> Result<()> {
        let slot = frame_slot % FRAMES_IN_FLIGHT;
        self.params_buffers[slot]
            .mapped_slice_mut()?
            .copy_from_slice(bytemuck::bytes_of(params));

        let out_old_layout = if self.first_frame {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let out_src_stage = if self.first_frame {
            vk::PipelineStageFlags2::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        };

        let pre_barriers = [
            // G-buffer color attachments -> SHADER_READ_ONLY_OPTIMAL
            sync::image_barrier(
                gbuffer.albedo_metallic.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                gbuffer.albedo_metallic.subresource_range(),
            ),
            sync::image_barrier(
                gbuffer.normal_roughness.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                gbuffer.normal_roughness.subresource_range(),
            ),
            sync::image_barrier(
                gbuffer.view_z.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                gbuffer.view_z.subresource_range(),
            ),
            // G-buffer motion -> SHADER_READ_ONLY_OPTIMAL for NRD
            sync::image_barrier(
                gbuffer.motion.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                gbuffer.motion.subresource_range(),
            ),
            // NRD outputs -> GENERAL for storage writes
            sync::image_barrier(
                self.nrd_diff_rad_hitdist.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_diff_rad_hitdist.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_spec_rad_hitdist.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_spec_rad_hitdist.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_normal_roughness.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_normal_roughness.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_view_z.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_view_z.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_diff_confidence.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_diff_confidence.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_spec_confidence.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.nrd_spec_confidence.subresource_range(),
            ),
        ];

        let pc = RtGiPushConstants {
            params_ptr: self.params_buffers[slot].device_address,
            materials_ptr: self.materials_buffer.device_address,
            lights_ptr: self.lights_buffer.device_address,
        };

        unsafe {
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &pre_barriers);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
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
        }

        // Transition RT outputs to SHADER_READ_ONLY_OPTIMAL for NRD
        let post_barriers = [
            sync::image_barrier(
                self.nrd_diff_rad_hitdist.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_diff_rad_hitdist.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_spec_rad_hitdist.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_spec_rad_hitdist.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_normal_roughness.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_normal_roughness.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_view_z.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_view_z.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_diff_confidence.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_diff_confidence.subresource_range(),
            ),
            sync::image_barrier(
                self.nrd_spec_confidence.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.nrd_spec_confidence.subresource_range(),
            ),
        ];
        unsafe {
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &post_barriers);
        }

        self.first_frame = false;
        Ok(())
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        let alloc = rhi.allocator_mut();
        for buf in self.params_buffers {
            buf.destroy(&device, alloc)?;
        }
        self.materials_buffer.destroy(&device, alloc)?;
        self.lights_buffer.destroy(&device, alloc)?;
        self.nrd_diff_rad_hitdist.destroy(&device, alloc)?;
        self.nrd_spec_rad_hitdist.destroy(&device, alloc)?;
        self.nrd_normal_roughness.destroy(&device, alloc)?;
        self.nrd_view_z.destroy(&device, alloc)?;
        self.nrd_diff_confidence.destroy(&device, alloc)?;
        self.nrd_spec_confidence.destroy(&device, alloc)?;
        Ok(())
    }
}
