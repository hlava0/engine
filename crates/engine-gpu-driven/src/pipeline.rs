use ash::vk;
use engine_rhi::{BindlessTable, GpuImage, Result, RhiDevice, load_shader_words, sync};

use crate::scene::GpuScene;
use crate::types::{
    CULL_FLAG_FRUSTUM, CULL_FLAG_HIZ, CullPushConstants, FrameUniforms, GeomPushConstants,
    HizDownsamplePushConstants,
};

const BINDLESS_SAMPLER_HIZ: u32 = 0;
const BINDLESS_TEX_HIZ_FULL: u32 = 0;
const BINDLESS_TEX_DEPTH: u32 = 1;
const BINDLESS_TEX_HIZ_MIP_BASE: u32 = 2;
const BINDLESS_IMG_HIZ_MIP_BASE: u32 = 0;

/// G-buffer render targets and full Hi-Z mip pyramid.
pub struct GBufferTargets {
    pub extent: vk::Extent2D,
    pub albedo_metallic: GpuImage,
    pub normal_roughness: GpuImage,
    pub motion: GpuImage,
    pub view_z: GpuImage,
    pub depth: GpuImage,
    pub hiz: GpuImage,
}

impl GBufferTargets {
    pub fn new(rhi: &mut RhiDevice, width: u32, height: u32) -> Result<Self> {
        let device = rhi.device.clone();
        let alloc = rhi.allocator_mut();

        let albedo_metallic = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuffer-albedo-metallic",
        )?;

        let normal_roughness = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuffer-normal-roughness",
        )?;

        let motion = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            1,
            vk::Format::R16G16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            vk::ImageAspectFlags::COLOR,
            "gbuffer-motion",
        )?;

        let view_z = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            1,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuffer-view-z",
        )?;

        let depth = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            1,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
            "gbuffer-depth",
        )?;

        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;
        let hiz = GpuImage::create_2d(
            &device,
            alloc,
            width,
            height,
            mip_levels,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "gbuffer-hiz",
        )?
        .with_mip_views(&device)?;

        Ok(Self {
            extent: vk::Extent2D { width, height },
            albedo_metallic,
            normal_roughness,
            motion,
            view_z,
            depth,
            hiz,
        })
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        let alloc = rhi.allocator_mut();
        self.albedo_metallic.destroy(&device, alloc)?;
        self.normal_roughness.destroy(&device, alloc)?;
        self.motion.destroy(&device, alloc)?;
        self.view_z.destroy(&device, alloc)?;
        self.depth.destroy(&device, alloc)?;
        self.hiz.destroy(&device, alloc)?;
        Ok(())
    }
}

/// GPU-driven compute culling, Dynamic Rendering G-buffer pass, and Hi-Z pyramid generator.
pub struct GpuDrivenPipeline {
    pub targets: GBufferTargets,
    pub hiz_sampler: vk::Sampler,
    cull_layout: vk::PipelineLayout,
    cull_pipeline: vk::Pipeline,
    hiz_layout: vk::PipelineLayout,
    hiz_pipeline: vk::Pipeline,
    geom_layout: vk::PipelineLayout,
    geom_pipeline: vk::Pipeline,
    first_frame: bool,
}

impl GpuDrivenPipeline {
    pub fn new(
        rhi: &mut RhiDevice,
        bindless: &BindlessTable,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let targets = GBufferTargets::new(rhi, width, height)?;
        let device = &rhi.device;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod((targets.hiz.mip_levels - 1) as f32);
        let hiz_sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        // Register Hi-Z & depth views in the bindless table
        bindless.write_sampler(device, BINDLESS_SAMPLER_HIZ, hiz_sampler);
        bindless.write_sampled_image(
            device,
            BINDLESS_TEX_HIZ_FULL,
            targets.hiz.view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        bindless.write_sampled_image(
            device,
            BINDLESS_TEX_DEPTH,
            targets.depth.view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        for mip in 0..targets.hiz.mip_levels {
            let view = targets.hiz.mip_view(mip);
            bindless.write_sampled_image(
                device,
                BINDLESS_TEX_HIZ_MIP_BASE + mip,
                view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            bindless.write_storage_image(
                device,
                BINDLESS_IMG_HIZ_MIP_BASE + mip,
                view,
                vk::ImageLayout::GENERAL,
            );
        }

        let set_layouts = [bindless.layout];

        // 1. Cull compute pipeline (local_size_x = 64, total = 64)
        let (cull_layout, cull_pipeline) = create_compute_pipeline(
            rhi,
            "cull.comp",
            &set_layouts,
            std::mem::size_of::<CullPushConstants>() as u32,
            64,
            64,
        )?;

        // 2. Hi-Z downsample compute pipeline (local_size_x = 8, local_size_y = 8, total = 64)
        let (hiz_layout, hiz_pipeline) = create_compute_pipeline(
            rhi,
            "hiz_downsample.comp",
            &set_layouts,
            std::mem::size_of::<HizDownsamplePushConstants>() as u32,
            8,
            64,
        )?;

        // 3. G-buffer Dynamic Rendering graphics pipeline
        let (geom_layout, geom_pipeline) = create_gbuffer_pipeline(device)?;

        Ok(Self {
            targets,
            hiz_sampler,
            cull_layout,
            cull_pipeline,
            hiz_layout,
            hiz_pipeline,
            geom_layout,
            geom_pipeline,
            first_frame: true,
        })
    }

    /// Record a full GPU-driven frame:
    /// 1. Clear draw_count -> cull.comp
    /// 2. Indirect G-buffer pass (`vkCmdDrawIndexedIndirectCount`, Dynamic Rendering, Reverse-Z)
    /// 3. Hi-Z mip pyramid generation (`hiz_downsample.comp`)
    pub fn record_frame(
        &mut self,
        rhi: &RhiDevice,
        bindless: &BindlessTable,
        cmd: vk::CommandBuffer,
        scene: &mut GpuScene,
        uniforms: &FrameUniforms,
        enable_hiz: bool,
    ) -> Result<()> {
        scene.update_frame_uniforms(uniforms)?;
        let device = &rhi.device;
        let extent = self.targets.extent;

        // 1. Clear draw_count to 0 and barrier before cull.comp
        unsafe {
            device.cmd_fill_buffer(cmd, scene.draw_count_buffer.buffer, 0, 4, 0);
        }
        let fill_barrier = [sync::buffer_barrier(
            scene.draw_count_buffer.buffer,
            0,
            4,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )];
        unsafe {
            sync::cmd_pipeline_barrier2(device, cmd, &[], &fill_barrier, &[]);
        }

        // Dispatch cull.comp
        let mut cull_flags = CULL_FLAG_FRUSTUM;
        if enable_hiz && !self.first_frame {
            cull_flags |= CULL_FLAG_HIZ;
        }
        let cull_pc = CullPushConstants {
            frame_uniforms_ptr: scene.frame_uniforms_buffer.device_address,
            instances_ptr: scene.instance_buffer.device_address,
            meshes_ptr: scene.mesh_buffer.device_address,
            indirect_draws_ptr: scene.indirect_draw_buffer.device_address,
            draw_count_ptr: scene.draw_count_buffer.device_address,
            draw_instance_ids_ptr: scene.draw_instance_ids_buffer.device_address,
            instance_count: scene.instance_count,
            hiz_bindless_index: BINDLESS_TEX_HIZ_FULL,
            hiz_sampler_index: BINDLESS_SAMPLER_HIZ,
            cull_flags,
        };
        let groups_x = scene.instance_count.div_ceil(64);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_layout,
                0,
                &[bindless.set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.cull_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&cull_pc),
            );
            device.cmd_dispatch(cmd, groups_x, 1, 1);
        }

        // 2. Barrier: cull.comp -> indirect draw + G-buffer attachments
        let cull_to_draw_bufs = [
            sync::buffer_barrier(
                scene.indirect_draw_buffer.buffer,
                0,
                scene.indirect_draw_buffer.size,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
            sync::buffer_barrier(
                scene.draw_count_buffer.buffer,
                0,
                4,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
            sync::buffer_barrier(
                scene.draw_instance_ids_buffer.buffer,
                0,
                scene.draw_instance_ids_buffer.size,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];

        let color_src_stage = if self.first_frame {
            vk::PipelineStageFlags2::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        };
        let depth_src_stage = if self.first_frame {
            vk::PipelineStageFlags2::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        };

        let gbuf_img_barriers = [
            sync::image_barrier(
                self.targets.albedo_metallic.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                color_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                self.targets.albedo_metallic.subresource_range(),
            ),
            sync::image_barrier(
                self.targets.normal_roughness.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                color_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                self.targets.normal_roughness.subresource_range(),
            ),
            sync::image_barrier(
                self.targets.motion.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                color_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                self.targets.motion.subresource_range(),
            ),
            sync::image_barrier(
                self.targets.view_z.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                color_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                self.targets.view_z.subresource_range(),
            ),
            sync::image_barrier(
                self.targets.depth.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                depth_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                self.targets.depth.subresource_range(),
            ),
        ];

        unsafe {
            sync::cmd_pipeline_barrier2(device, cmd, &[], &cull_to_draw_bufs, &gbuf_img_barriers);
        }

        // Dynamic Rendering G-buffer pass
        let color_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };
        let depth_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0, // Reverse-Z clear
                stencil: 0,
            },
        };

        let color_attachments = [
            vk::RenderingAttachmentInfo::default()
                .image_view(self.targets.albedo_metallic.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear),
            vk::RenderingAttachmentInfo::default()
                .image_view(self.targets.normal_roughness.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear),
            vk::RenderingAttachmentInfo::default()
                .image_view(self.targets.motion.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear),
            vk::RenderingAttachmentInfo::default()
                .image_view(self.targets.view_z.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear),
        ];

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.targets.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        }];

        let geom_pc = GeomPushConstants {
            frame_uniforms_ptr: scene.frame_uniforms_buffer.device_address,
            vertices_ptr: scene.vertex_buffer.device_address,
            instances_ptr: scene.instance_buffer.device_address,
            draw_instance_ids_ptr: scene.draw_instance_ids_buffer.device_address,
        };

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.geom_pipeline);
            device.cmd_set_viewport(cmd, 0, &viewport);
            device.cmd_set_scissor(cmd, 0, &scissor);
            device.cmd_bind_index_buffer(cmd, scene.index_buffer.buffer, 0, vk::IndexType::UINT32);
            device.cmd_push_constants(
                cmd,
                self.geom_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&geom_pc),
            );
            device.cmd_draw_indexed_indirect_count(
                cmd,
                scene.indirect_draw_buffer.buffer,
                0,
                scene.draw_count_buffer.buffer,
                0,
                scene.instance_count,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            device.cmd_end_rendering(cmd);
        }

        // 3. Barrier: Depth -> SHADER_READ_ONLY_OPTIMAL; Hi-Z all mips -> GENERAL
        let hiz_old_layout = if self.first_frame {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let hiz_src_stage = if self.first_frame {
            vk::PipelineStageFlags2::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        };

        let pre_hiz_barriers = [
            sync::image_barrier(
                self.targets.depth.image,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                self.targets.depth.subresource_range(),
            ),
            sync::image_barrier(
                self.targets.hiz.image,
                hiz_old_layout,
                vk::ImageLayout::GENERAL,
                hiz_src_stage,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                self.targets.hiz.subresource_range(),
            ),
        ];

        unsafe {
            sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &pre_hiz_barriers);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.hiz_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.hiz_layout,
                0,
                &[bindless.set],
                &[],
            );
        }

        // Generate Hi-Z mip chain (mip 0 copies depth, mip 1..N downsample 2x2 min)
        for mip in 0..self.targets.hiz.mip_levels {
            let mip_w = (extent.width >> mip).max(1);
            let mip_h = (extent.height >> mip).max(1);
            let src_texture_idx = if mip == 0 {
                BINDLESS_TEX_DEPTH
            } else {
                BINDLESS_TEX_HIZ_MIP_BASE + (mip - 1)
            };

            let hiz_pc = HizDownsamplePushConstants {
                src_texture_idx,
                src_sampler_idx: BINDLESS_SAMPLER_HIZ,
                dst_image_idx: BINDLESS_IMG_HIZ_MIP_BASE + mip,
                dst_width: mip_w,
                dst_height: mip_h,
                is_first_mip: u32::from(mip == 0),
            };

            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.hiz_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&hiz_pc),
                );
                device.cmd_dispatch(cmd, mip_w.div_ceil(8), mip_h.div_ceil(8), 1);

                let mip_barrier = [sync::image_barrier(
                    self.targets.hiz.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    self.targets.hiz.mip_subresource_range(mip),
                )];
                sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &mip_barrier);
            }
        }

        self.first_frame = false;
        Ok(())
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        unsafe {
            rhi.device.destroy_pipeline(self.cull_pipeline, None);
            rhi.device.destroy_pipeline_layout(self.cull_layout, None);
            rhi.device.destroy_pipeline(self.hiz_pipeline, None);
            rhi.device.destroy_pipeline_layout(self.hiz_layout, None);
            rhi.device.destroy_pipeline(self.geom_pipeline, None);
            rhi.device.destroy_pipeline_layout(self.geom_layout, None);
            rhi.device.destroy_sampler(self.hiz_sampler, None);
        }
        self.targets.destroy(rhi)?;
        Ok(())
    }
}

fn create_compute_pipeline(
    rhi: &RhiDevice,
    shader_name: &str,
    set_layouts: &[vk::DescriptorSetLayout],
    push_constant_size: u32,
    local_size_x: u32,
    workgroup_size: u32,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let device = &rhi.device;
    let words = load_shader_words(shader_name)?;
    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let module = unsafe { device.create_shader_module(&module_info, None)? };

    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size)];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(set_layouts)
        .push_constant_ranges(&pc_range);
    let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
        Ok(l) => l,
        Err(err) => {
            unsafe { device.destroy_shader_module(module, None) };
            return Err(err.into());
        }
    };

    let req_sg = rhi
        .capabilities
        .vendor_profile
        .required_compute_subgroup_size
        .filter(|&sz| workgroup_size.is_multiple_of(sz));
    let mut sg_info = rhi.subgroup_stage_create_info(vk::ShaderStageFlags::COMPUTE, req_sg);
    let stage_flags =
        rhi.subgroup_stage_flags(vk::ShaderStageFlags::COMPUTE, req_sg, local_size_x);

    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .flags(stage_flags)
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    if let Some(ref mut sg) = sg_info {
        stage = stage.push_next(sg);
    }

    let info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout)];
    let pipe_res = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
    unsafe { device.destroy_shader_module(module, None) };

    match pipe_res {
        Ok(pipes) => Ok((layout, pipes[0])),
        Err((_, err)) => {
            unsafe { device.destroy_pipeline_layout(layout, None) };
            Err(err.into())
        }
    }
}

fn create_gbuffer_pipeline(device: &ash::Device) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_words = load_shader_words("gbuffer.vert")?;
    let frag_words = load_shader_words("gbuffer.frag")?;

    let vert_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_words), None)?
    };
    let frag_module = match unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_words), None)
    } {
        Ok(m) => m,
        Err(err) => {
            unsafe { device.destroy_shader_module(vert_module, None) };
            return Err(err.into());
        }
    };

    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<GeomPushConstants>() as u32)];
    let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_range);
    let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
        Ok(l) => l,
        Err(err) => {
            unsafe {
                device.destroy_shader_module(vert_module, None);
                device.destroy_shader_module(frag_module, None);
            }
            return Err(err.into());
        }
    };

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // Vertex pulling via BDA: empty vertex input state
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL); // Reverse-Z

    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);
    let blend_attachments = [blend_attachment; 4];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats = [
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R16G16_SFLOAT,
        vk::Format::R32_SFLOAT,
    ];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(vk::Format::D32_SFLOAT);

    let gfx_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering_info)];

    let pipe_res =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &gfx_info, None) };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    match pipe_res {
        Ok(pipes) => Ok((layout, pipes[0])),
        Err((_, err)) => {
            unsafe { device.destroy_pipeline_layout(layout, None) };
            Err(err.into())
        }
    }
}
