use ash::vk;
use engine_gpu_driven::GBufferTargets;
use engine_rhi::{GpuImage, RhiDevice, load_shader_words, sync};

use crate::error::Result;
use crate::nrd::NrdDenoiser;
use crate::rt_pass::RtGiPass;
use crate::types::ComposePushConstants;

/// Final NRD YCoCg decode + material remodulation + sky preservation pass outputting linear HDR.
pub struct NrdComposePass {
    pub extent: vk::Extent2D,
    pub out_hdr: GpuImage,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    first_frame: bool,
}

impl NrdComposePass {
    pub fn new(rhi: &mut RhiDevice, width: u32, height: u32) -> Result<Self> {
        let device = rhi.device.clone();

        let out_hdr = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "nrd-compose-hdr",
        )?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        let bindings = [
            // 0: in_nrd_diff
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 1: in_nrd_spec
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 2: in_raw_sky
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 3: gbuf_albedo_metallic
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 4: gbuf_normal_roughness
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 5: gbuf_view_z
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 6: gbuf_depth
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 7: out_hdr
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout)),
            )?
        }[0];

        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ComposePushConstants>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_constant_ranges(&pc_range),
                None,
            )?
        };

        let shader_words = load_shader_words("nrd_compose.comp")?;
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
            let res = device.create_compute_pipelines(vk::PipelineCache::null(), &pipe_info, None);
            device.destroy_shader_module(module, None);
            res.map_err(|(_, e)| e)?[0]
        };

        Ok(Self {
            extent: vk::Extent2D { width, height },
            out_hdr,
            sampler,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            first_frame: true,
        })
    }

    /// Bind NRD outputs, raw sky source, and G-buffer textures (zero allocations).
    pub fn bind_resources(
        &self,
        device: &ash::Device,
        nrd: &NrdDenoiser,
        rt_pass: &RtGiPass,
        gbuffer: &GBufferTargets,
    ) {
        let diff_info = [vk::DescriptorImageInfo::default()
            .image_view(nrd.out_diff_rad_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let spec_info = [vk::DescriptorImageInfo::default()
            .image_view(nrd.out_spec_rad_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let raw_sky_info = [vk::DescriptorImageInfo::default()
            .image_view(rt_pass.nrd_diff_rad_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let albedo_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.albedo_metallic.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.normal_roughness.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let viewz_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.view_z.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let depth_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(gbuffer.depth.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let out_info = [vk::DescriptorImageInfo::default()
            .image_view(self.out_hdr.view)
            .image_layout(vk::ImageLayout::GENERAL)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&diff_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&spec_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&raw_sky_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&albedo_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&viewz_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_info),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Record the composition pass, leaving `out_hdr` in `vk::ImageLayout::GENERAL`.
    pub fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        rt_pass: &RtGiPass,
    ) {
        let out_old_layout = if self.first_frame {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::GENERAL
        };
        let out_src_stage = if self.first_frame {
            vk::PipelineStageFlags2::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER
        };

        let pre_barriers = [
            sync::image_barrier(
                rt_pass.nrd_diff_rad_hitdist.image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                rt_pass.nrd_diff_rad_hitdist.subresource_range(),
            ),
            sync::image_barrier(
                self.out_hdr.image,
                out_old_layout,
                vk::ImageLayout::GENERAL,
                out_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.out_hdr.subresource_range(),
            ),
        ];

        let post_barriers = [sync::image_barrier(
            rt_pass.nrd_diff_rad_hitdist.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            rt_pass.nrd_diff_rad_hitdist.subresource_range(),
        )];

        let pc = ComposePushConstants {
            params_ptr: rt_pass.params_buffer.device_address,
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
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &post_barriers);
        }

        self.first_frame = false;
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
        self.out_hdr.destroy(&device, rhi.allocator_mut())?;
        Ok(())
    }
}
