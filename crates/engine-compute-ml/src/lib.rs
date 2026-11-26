//! Cooperative Matrix (`VK_KHR_cooperative_matrix`) compute post-processing with subgroup fallback.

use bytemuck::{Pod, Zeroable};
use engine_rhi::ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{
    CoopMatrixTile, GpuBuffer, GpuImage, RhiDevice, RhiError, ash, is_f16_subgroup_coop_shape,
    load_shader_words, sync,
};
use thiserror::Error;

/// Errors returned by `engine-compute-ml`.
#[derive(Debug, Error)]
pub enum CoopMlError {
    #[error("RHI error: {0}")]
    Rhi(#[from] RhiError),
    #[error("Vulkan error: {0}")]
    Vk(#[from] vk::Result),
}

pub type Result<T> = std::result::Result<T, CoopMlError>;

/// Push constants shared by `coop_ml_post.comp` and `coop_ml_fallback.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CoopMlPushConstants {
    pub weights_addr: u64,
    pub width: u32,
    pub height: u32,
    pub exposure: f32,
    pub residual_strength: f32,
}

/// 2-layer neural color-grading and ACES tone-mapping post-processing pipeline.
///
/// Selects the optimal `VK_KHR_cooperative_matrix` tile shape from `VendorProfile`:
/// - `M8N16K16`: Intel Xe / Xe2 XMX (`coop_ml_post.comp`, SIMD16)
/// - `M16N16K16`: NVIDIA Tensor Core / AMD RDNA3+ WMMA (`coop_ml_post_16x16.comp`, Wave32/Warp32)
/// - `Fallback`: scalar compute shader (`coop_ml_fallback.comp`)
pub struct CoopMlPipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    weight_buffer: Option<GpuBuffer>,
    active_tile: CoopMatrixTile,
}

impl CoopMlPipeline {
    /// Returns `true` if the device supports the `8x16x16` `FLOAT16` subgroup cooperative matrix shape.
    pub fn supports_coop_matrix_8x16x16_f16(rhi: &RhiDevice) -> bool {
        let caps = &rhi.capabilities;
        if !caps.khr_cooperative_matrix || !caps.shader_float16 {
            return false;
        }
        caps.cooperative_matrix_properties
            .iter()
            .any(|p| is_f16_subgroup_coop_shape(p, 8, 16, 16))
    }

    /// Returns `true` if the device supports the `16x16x16` `FLOAT16` subgroup cooperative matrix shape.
    pub fn supports_coop_matrix_16x16x16_f16(rhi: &RhiDevice) -> bool {
        let caps = &rhi.capabilities;
        if !caps.khr_cooperative_matrix || !caps.shader_float16 {
            return false;
        }
        caps.cooperative_matrix_properties
            .iter()
            .any(|p| is_f16_subgroup_coop_shape(p, 16, 16, 16))
    }

    /// Create a new `CoopMlPipeline`. Pass `force_fallback = true` to force the scalar compute path.
    pub fn new(rhi: &mut RhiDevice, force_fallback: bool) -> Result<Self> {
        let active_tile = if force_fallback {
            CoopMatrixTile::Fallback
        } else {
            rhi.capabilities.vendor_profile.preferred_coop_tile
        };

        let (shader_name, req_subgroup_size, local_size_x) = match active_tile {
            CoopMatrixTile::M8N16K16 => {
                log::info!("CoopMlPipeline: using VK_KHR_cooperative_matrix (8x16x16 FLOAT16 Intel XMX, SIMD16)");
                ("coop_ml_post.comp", Some(16u32), 32u32)
            }
            CoopMatrixTile::M16N16K16 => {
                log::info!("CoopMlPipeline: using VK_KHR_cooperative_matrix (16x16x16 FLOAT16 Tensor/WMMA, Wave32)");
                ("coop_ml_post_16x16.comp", Some(32u32), 32u32)
            }
            CoopMatrixTile::Fallback => {
                log::info!("CoopMlPipeline: using scalar compute fallback");
                (
                    "coop_ml_fallback.comp",
                    rhi.capabilities.vendor_profile.required_compute_subgroup_size,
                    8u32,
                )
            }
        };

        let spv_words = load_shader_words(shader_name)?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        let shader_module = unsafe { rhi.device.create_shader_module(&shader_info, None)? };

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { rhi.device.create_descriptor_set_layout(&set_layout_info, None)? };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { rhi.device.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let descriptor_set = unsafe { rhi.device.allocate_descriptor_sets(&alloc_info)?[0] };

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CoopMlPushConstants>() as u32)];
        let pipe_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(&push_ranges);
        let pipeline_layout =
            unsafe { rhi.device.create_pipeline_layout(&pipe_layout_info, None)? };

        let mut sg_info =
            rhi.subgroup_stage_create_info(vk::ShaderStageFlags::COMPUTE, req_subgroup_size);
        let stage_flags = rhi.subgroup_stage_flags(
            vk::ShaderStageFlags::COMPUTE,
            req_subgroup_size,
            local_size_x,
        );

        let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
            .flags(stage_flags)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        if let Some(ref mut sg) = sg_info {
            stage_info = stage_info.push_next(sg);
        }
        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        let pipeline_res = unsafe {
            rhi.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
        };
        unsafe { rhi.device.destroy_shader_module(shader_module, None) };
        let pipeline = pipeline_res.map_err(|(_, e)| e)?[0];

        // Allocate BDA weight buffer (W1: 16x16 floats, W2: 16x16 floats = 2048 bytes)
        let weights = build_default_weights();
        let dev = rhi.device.clone();
        let mut weight_buffer = GpuBuffer::new(
            &dev,
            rhi.allocator_mut(),
            "coop_ml_weights",
            std::mem::size_of_val(&weights) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        weight_buffer.mapped_slice_mut()?.copy_from_slice(bytemuck::bytes_of(&weights));

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            pipeline,
            weight_buffer: Some(weight_buffer),
            active_tile,
        })
    }

    /// Returns the active [`CoopMatrixTile`] selected for this pipeline.
    pub fn active_tile(&self) -> CoopMatrixTile {
        self.active_tile
    }

    /// Returns `true` if this pipeline uses `VK_KHR_cooperative_matrix`.
    pub fn uses_cooperative_matrix(&self) -> bool {
        self.active_tile != CoopMatrixTile::Fallback
    }

    /// Dispatch the neural post-processing pass with default exposure (1.0) and residual strength (0.25).
    pub fn dispatch(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        input_hdr: &GpuImage,
        output_image: &GpuImage,
        width: u32,
        height: u32,
    ) {
        self.dispatch_with_params(device, cmd, input_hdr, output_image, width, height, 1.0, 0.25);
    }

    /// Dispatch the neural post-processing pass with explicit exposure and residual strength.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_params(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        input_hdr: &GpuImage,
        output_image: &GpuImage,
        width: u32,
        height: u32,
        exposure: f32,
        residual_strength: f32,
    ) {
        let in_info = [vk::DescriptorImageInfo::default()
            .image_view(input_hdr.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let out_info = [vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&in_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let pre_barriers = [
            sync::image_barrier(
                input_hdr.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                input_hdr.subresource_range(),
            ),
            sync::image_barrier(
                output_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                output_image.subresource_range(),
            ),
        ];
        unsafe { sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &pre_barriers) };

        let weights_addr = self
            .weight_buffer
            .as_ref()
            .map(|b| b.device_address)
            .unwrap_or(0);
        let pc = CoopMlPushConstants {
            weights_addr,
            width,
            height,
            exposure,
            residual_strength,
        };

        unsafe {
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

            let (gx, gy) = match self.active_tile {
                CoopMatrixTile::M8N16K16 => (width.div_ceil(8), height),
                CoopMatrixTile::M16N16K16 => (width.div_ceil(16), height),
                CoopMatrixTile::Fallback => (width.div_ceil(8), height.div_ceil(8)),
            };
            device.cmd_dispatch(cmd, gx.max(1), gy.max(1), 1);
        }

        let post_barriers = [sync::image_barrier(
            output_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            output_image.subresource_range(),
        )];
        unsafe { sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &post_barriers) };
    }

    /// Destroy all Vulkan resources owned by this pipeline.
    pub fn destroy(&mut self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        if let Some(buf) = self.weight_buffer.take() {
            buf.destroy(&device, rhi.allocator_mut())?;
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
        Ok(())
    }
}

/// Deterministic 2-layer MLP weights for neural contrast/shadow-lift/filmic color grading.
/// Layout: `[W1 (16x16 row-major), W2 (16x16 row-major)]`.
fn build_default_weights() -> [f32; 512] {
    let mut w = [0.0f32; 512];
    let (w1, w2) = w.split_at_mut(256);

    let mut set_w1 = |row: usize, col: usize, val: f32| {
        w1[row * 16 + col] = val;
    };
    let mut set_w2 = |row: usize, col: usize, val: f32| {
        w2[row * 16 + col] = val;
    };

    // W1[feature_k, hidden_col]: slight local contrast + saturation boost on RGB
    set_w1(0, 0, 0.25);
    set_w1(3, 0, -0.05);
    set_w1(9, 0, 0.08);

    set_w1(1, 1, 0.25);
    set_w1(3, 1, -0.05);
    set_w1(9, 1, 0.08);

    set_w1(2, 2, 0.25);
    set_w1(3, 2, -0.05);
    set_w1(10, 2, 0.08);

    // Hidden 3..6: shadow lift via log(1 + c)
    set_w1(4, 3, 0.20);
    set_w1(11, 3, 0.01);
    set_w1(5, 4, 0.20);
    set_w1(11, 4, 0.01);
    set_w1(6, 5, 0.20);
    set_w1(11, 5, 0.01);

    // Hidden 6: highlight roll-off term
    set_w1(7, 6, 0.15);
    set_w1(8, 6, -0.05);

    // W2[hidden_k, out_col]
    set_w2(0, 0, 0.45);
    set_w2(1, 1, 0.45);
    set_w2(2, 2, 0.45);

    set_w2(3, 0, 0.35);
    set_w2(4, 1, 0.35);
    set_w2(5, 2, 0.35);

    // Subtle warm-highlight / cool-shadow roll-off
    set_w2(6, 0, -0.04);
    set_w2(6, 1, -0.05);
    set_w2(6, 2, -0.06);

    w
}
