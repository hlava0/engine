use ash::vk;
use engine_rhi::{GpuImage, RhiDevice};

use crate::error::Result;
#[cfg(not(has_nrd))]
use crate::error::RtNrdError;
use crate::svgf::SvgfDenoiser;

#[cfg(has_nrd)]
use engine_rhi::gpu_allocator::MemoryLocation;
#[cfg(has_nrd)]
use engine_rhi::{FRAMES_IN_FLIGHT, GpuBuffer, sync};

#[cfg(has_nrd)]
use crate::error::RtNrdError;
#[cfg(has_nrd)]
use crate::ffi::{
    NRD_DENOISER_REBLUR_DIFFUSE_SPECULAR, NRD_DESCRIPTOR_STORAGE_TEXTURE,
    NRD_RES_IN_DIFF_CONFIDENCE, NRD_RES_IN_DIFF_RADIANCE_HITDIST, NRD_RES_IN_MV,
    NRD_RES_IN_NORMAL_ROUGHNESS, NRD_RES_IN_SPEC_CONFIDENCE, NRD_RES_IN_SPEC_RADIANCE_HITDIST,
    NRD_RES_IN_VIEWZ, NRD_RES_OUT_DIFF_RADIANCE_HITDIST, NRD_RES_OUT_SPEC_RADIANCE_HITDIST,
    NRD_RES_PERMANENT_POOL, NRD_RES_TRANSIENT_POOL, NRD_RESULT_SUCCESS, NrdCommonSettingsC,
    NrdDispatchInfo, NrdInstance, NrdInstanceInfo, NrdLibraryInfo, NrdPipelineInfo,
    NrdReblurSettingsC, NrdResourceBinding, NrdTextureInfo, nrd_shim_create_instance,
    nrd_shim_destroy_instance, nrd_shim_get_compute_dispatches, nrd_shim_get_instance_info,
    nrd_shim_get_library_info, nrd_shim_set_common_settings, nrd_shim_set_reblur_settings,
};

#[cfg(has_nrd)]
const MAX_DISPATCHES: usize = 48;
#[cfg(has_nrd)]
const MAX_RESOURCES_PER_FRAME: usize = 512;

/// Selects which spatiotemporal denoiser backend [`NrdDenoiser`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenoiserMode {
    /// Use NVIDIA NRD (REBLUR) when compiled with the NRD SDK (`cfg(has_nrd)`),
    /// otherwise fall back to the built-in GPL-3.0 SVGF denoiser.
    #[default]
    Auto,
    /// Require NVIDIA NRD (REBLUR); returns an error if the NRD SDK was not compiled in.
    Nrd,
    /// Force the built-in GPL-3.0 SVGF spatiotemporal denoiser.
    Svgf,
}

/// Input textures provided to the denoiser each frame.
/// All must be in `vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL` before [`NrdDenoiser::record`].
pub struct NrdInputs<'a> {
    pub diff_rad_hitdist: &'a GpuImage,
    pub spec_rad_hitdist: &'a GpuImage,
    pub normal_roughness: &'a GpuImage,
    pub view_z: &'a GpuImage,
    pub motion: &'a GpuImage,
    pub diff_confidence: &'a GpuImage,
    pub spec_confidence: &'a GpuImage,
}

enum DenoiserBackend {
    #[cfg(has_nrd)]
    Nrd(Box<NrdBackend>),
    Svgf(Box<SvgfDenoiser>),
}

/// Unified spatiotemporal denoiser supporting optional NVIDIA NRD (`REBLUR_DIFFUSE_SPECULAR`)
/// and a built-in pure GPL-3.0 SVGF fallback.
pub struct NrdDenoiser {
    pub extent: vk::Extent2D,
    pub out_diff_rad_hitdist: GpuImage,
    pub out_spec_rad_hitdist: GpuImage,
    backend: DenoiserBackend,
}

impl NrdDenoiser {
    /// Create a denoiser using [`DenoiserMode::Auto`] (NRD if available, else built-in SVGF).
    pub fn new(rhi: &mut RhiDevice, width: u32, height: u32) -> Result<Self> {
        Self::new_with_mode(rhi, width, height, DenoiserMode::Auto)
    }

    /// Create a denoiser with an explicitly requested [`DenoiserMode`].
    pub fn new_with_mode(
        rhi: &mut RhiDevice,
        width: u32,
        height: u32,
        mode: DenoiserMode,
    ) -> Result<Self> {
        let device = rhi.device.clone();

        let out_diff_rad_hitdist = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-out-diff-rad-hitdist",
        )?;
        let out_spec_rad_hitdist = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            width,
            height,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            "nrd-out-spec-rad-hitdist",
        )?;

        let backend = match mode {
            DenoiserMode::Auto => {
                #[cfg(has_nrd)]
                {
                    DenoiserBackend::Nrd(Box::new(NrdBackend::new(rhi, width, height)?))
                }
                #[cfg(not(has_nrd))]
                {
                    DenoiserBackend::Svgf(Box::new(SvgfDenoiser::new(rhi, width, height)?))
                }
            }
            DenoiserMode::Nrd => {
                #[cfg(has_nrd)]
                {
                    DenoiserBackend::Nrd(Box::new(NrdBackend::new(rhi, width, height)?))
                }
                #[cfg(not(has_nrd))]
                {
                    let alloc = rhi.allocator_mut();
                    let _ = out_diff_rad_hitdist.destroy(&device, alloc);
                    let _ = out_spec_rad_hitdist.destroy(&device, alloc);
                    return Err(RtNrdError::InvalidConfig(
                        "DenoiserMode::Nrd requested, but crate was built without NVIDIA NRD SDK"
                            .to_string(),
                    ));
                }
            }
            DenoiserMode::Svgf => {
                DenoiserBackend::Svgf(Box::new(SvgfDenoiser::new(rhi, width, height)?))
            }
        };

        Ok(Self {
            extent: vk::Extent2D { width, height },
            out_diff_rad_hitdist,
            out_spec_rad_hitdist,
            backend,
        })
    }

    /// Returns `true` if this denoiser instance is backed by NVIDIA NRD (REBLUR).
    #[inline]
    pub fn is_using_nrd(&self) -> bool {
        #[cfg(has_nrd)]
        {
            matches!(self.backend, DenoiserBackend::Nrd(_))
        }
        #[cfg(not(has_nrd))]
        {
            false
        }
    }

    /// Human-readable name of the active denoiser backend.
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        if self.is_using_nrd() {
            "NVIDIA NRD (REBLUR)"
        } else {
            "Built-in SVGF (GPL-3.0)"
        }
    }

    /// Configure denoiser settings and record compute dispatches with `Synchronization2` barriers.
    ///
    /// Leaves `out_diff_rad_hitdist` and `out_spec_rad_hitdist` in `vk::ImageLayout::GENERAL`
    /// ready for `NrdComposePass`.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        inputs: &NrdInputs<'_>,
        view_to_clip: [[f32; 4]; 4],
        view_to_clip_prev: [[f32; 4]; 4],
        world_to_view: [[f32; 4]; 4],
        world_to_view_prev: [[f32; 4]; 4],
        jitter: [f32; 2],
        jitter_prev: [f32; 2],
        denoising_range: f32,
        time_delta_ms: f32,
        reset_history: bool,
        frame_slot: usize,
    ) -> Result<()> {
        match &mut self.backend {
            #[cfg(has_nrd)]
            DenoiserBackend::Nrd(nrd) => nrd.record(
                device,
                cmd,
                inputs,
                &self.out_diff_rad_hitdist,
                &self.out_spec_rad_hitdist,
                view_to_clip,
                view_to_clip_prev,
                world_to_view,
                world_to_view_prev,
                jitter,
                jitter_prev,
                denoising_range,
                time_delta_ms,
                reset_history,
                frame_slot,
            ),
            DenoiserBackend::Svgf(svgf) => {
                let _ = (
                    view_to_clip,
                    view_to_clip_prev,
                    world_to_view,
                    world_to_view_prev,
                    jitter,
                    jitter_prev,
                    time_delta_ms,
                );
                svgf.record(
                    device,
                    cmd,
                    inputs,
                    &self.out_diff_rad_hitdist,
                    &self.out_spec_rad_hitdist,
                    denoising_range,
                    reset_history,
                    frame_slot,
                )
            }
        }
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        match self.backend {
            #[cfg(has_nrd)]
            DenoiserBackend::Nrd(nrd) => nrd.destroy(rhi)?,
            DenoiserBackend::Svgf(svgf) => svgf.destroy(rhi)?,
        }
        let device = rhi.device.clone();
        let alloc = rhi.allocator_mut();
        self.out_diff_rad_hitdist.destroy(&device, alloc)?;
        self.out_spec_rad_hitdist.destroy(&device, alloc)?;
        Ok(())
    }
}

#[cfg(has_nrd)]
#[derive(Clone, Copy)]
struct TrackedImageState {
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

#[cfg(has_nrd)]
impl Default for TrackedImageState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            access: vk::AccessFlags2::NONE,
        }
    }
}

/// Self-contained NVIDIA NRD (`REBLUR_DIFFUSE_SPECULAR`) compute pass graph.
#[cfg(has_nrd)]
struct NrdBackend {
    instance: *mut NrdInstance,
    extent: vk::Extent2D,

    perm_pool: Vec<GpuImage>,
    trans_pool: Vec<GpuImage>,
    perm_states: Vec<TrackedImageState>,
    trans_states: Vec<TrackedImageState>,
    out_diff_state: TrackedImageState,
    out_spec_state: TrackedImageState,

    dummy_sampled: GpuImage,
    dummy_storage: GpuImage,
    dummies_initialized: bool,

    samplers: Vec<vk::Sampler>,
    cbufs: [Vec<GpuBuffer>; FRAMES_IN_FLIGHT],

    set0_layout: vk::DescriptorSetLayout,
    set1_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,

    descriptor_pool: vk::DescriptorPool,
    set0_per_slot: [Vec<vk::DescriptorSet>; FRAMES_IN_FLIGHT],
    set1_per_slot: [Vec<vk::DescriptorSet>; FRAMES_IN_FLIGHT],

    // Reusable per-frame scratch buffers to avoid hot-loop heap allocations
    dispatches_scratch: Vec<NrdDispatchInfo>,
    resources_scratch: Vec<NrdResourceBinding>,
    tex_infos_scratch: Vec<vk::DescriptorImageInfo>,
    store_infos_scratch: Vec<vk::DescriptorImageInfo>,

    max_tex_slots: u32,
    max_store_slots: u32,
    tex_offset: u32,
    store_offset: u32,
    frame_index: u32,
}

#[cfg(has_nrd)]
impl NrdBackend {
    fn new(rhi: &mut RhiDevice, width: u32, height: u32) -> Result<Self> {
        let mut lib_info = NrdLibraryInfo::default();
        unsafe {
            nrd_shim_get_library_info(&mut lib_info);
        }
        log::info!(
            "NRD v{}.{}.{} initialized (normalEncoding={}, roughnessEncoding={})",
            lib_info.version_major,
            lib_info.version_minor,
            lib_info.version_build,
            lib_info.normal_encoding,
            lib_info.roughness_encoding
        );

        let mut instance: *mut NrdInstance = std::ptr::null_mut();
        let res = unsafe {
            nrd_shim_create_instance(NRD_DENOISER_REBLUR_DIFFUSE_SPECULAR, 0, &mut instance)
        };
        if res != NRD_RESULT_SUCCESS || instance.is_null() {
            return Err(RtNrdError::Nrd(res));
        }

        let mut inst_info = NrdInstanceInfo::default();
        let mut pipe_infos = vec![NrdPipelineInfo::default(); 64];
        let mut perm_infos = vec![NrdTextureInfo::default(); 64];
        let mut trans_infos = vec![NrdTextureInfo::default(); 64];
        let mut sampler_types = vec![0u32; 8];

        let res = unsafe {
            nrd_shim_get_instance_info(
                instance,
                &mut inst_info,
                pipe_infos.as_mut_ptr(),
                pipe_infos.len() as u32,
                perm_infos.as_mut_ptr(),
                perm_infos.len() as u32,
                trans_infos.as_mut_ptr(),
                trans_infos.len() as u32,
                sampler_types.as_mut_ptr(),
                sampler_types.len() as u32,
            )
        };
        if res != NRD_RESULT_SUCCESS {
            unsafe { nrd_shim_destroy_instance(instance) };
            return Err(RtNrdError::Nrd(res));
        }

        pipe_infos.truncate(inst_info.pipelines_num as usize);
        perm_infos.truncate(inst_info.permanent_pool_size as usize);
        trans_infos.truncate(inst_info.transient_pool_size as usize);
        sampler_types.truncate(inst_info.samplers_num as usize);

        let device = rhi.device.clone();

        // Samplers
        let mut samplers = Vec::with_capacity(sampler_types.len());
        for &st in &sampler_types {
            let filter = if st == 0 {
                vk::Filter::NEAREST
            } else {
                vk::Filter::LINEAR
            };
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            samplers.push(unsafe { device.create_sampler(&info, None)? });
        }

        // Pool textures
        let mut perm_pool = Vec::with_capacity(perm_infos.len());
        for (i, td) in perm_infos.iter().enumerate() {
            let shift = td.downsample_factor.saturating_sub(1);
            let w = (width >> shift).max(1);
            let h = (height >> shift).max(1);
            perm_pool.push(GpuImage::create_2d(
                &device,
                rhi.allocator_mut(),
                w,
                h,
                1,
                nrd_format_to_vk(td.format)?,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                &format!("nrd-perm-{i}"),
            )?);
        }

        let mut trans_pool = Vec::with_capacity(trans_infos.len());
        for (i, td) in trans_infos.iter().enumerate() {
            let shift = td.downsample_factor.saturating_sub(1);
            let w = (width >> shift).max(1);
            let h = (height >> shift).max(1);
            trans_pool.push(GpuImage::create_2d(
                &device,
                rhi.allocator_mut(),
                w,
                h,
                1,
                nrd_format_to_vk(td.format)?,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                &format!("nrd-trans-{i}"),
            )?);
        }

        let dummy_sampled = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            1,
            1,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            vk::ImageAspectFlags::COLOR,
            "nrd-dummy-sampled",
        )?;
        let dummy_storage = GpuImage::create_2d(
            &device,
            rhi.allocator_mut(),
            1,
            1,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            vk::ImageAspectFlags::COLOR,
            "nrd-dummy-storage",
        )?;

        // Persistent host-visible constant buffers ring-buffered per frame-in-flight slot
        let cb_size = (inst_info.constant_buffer_max_data_size as u64).max(256);
        let mut cbufs_slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for slot in 0..FRAMES_IN_FLIGHT {
            let mut slot_cbufs = Vec::with_capacity(MAX_DISPATCHES);
            for i in 0..MAX_DISPATCHES {
                slot_cbufs.push(GpuBuffer::new(
                    &device,
                    rhi.allocator_mut(),
                    &format!("nrd-cbuf-s{slot}-{i}"),
                    cb_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?);
            }
            cbufs_slots.push(slot_cbufs);
        }
        let cbufs: [Vec<GpuBuffer>; FRAMES_IN_FLIGHT] = cbufs_slots
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        // Shared descriptor set layouts matching NRD's SPIR-V binding offsets
        let max_tex_slots = inst_info.per_set_textures_max_num.max(1);
        let max_store_slots = inst_info.per_set_storage_textures_max_num.max(1);
        let tex_offset = lib_info.texture_offset + inst_info.resources_base_register_index;
        let store_offset =
            lib_info.storage_texture_and_buffer_offset + inst_info.resources_base_register_index;

        let mut set0_bindings = Vec::with_capacity((max_tex_slots + max_store_slots) as usize);
        for t in 0..max_tex_slots {
            set0_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(tex_offset + t)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        for s in 0..max_store_slots {
            set0_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(store_offset + s)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        let set0_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set0_bindings),
                None,
            )?
        };

        let cb_binding = lib_info.constant_buffer_offset + inst_info.constant_buffer_register_index;
        let sampler_base = lib_info.sampler_offset + inst_info.samplers_base_register_index;
        let mut set1_bindings = Vec::with_capacity(1 + samplers.len());
        set1_bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(cb_binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
        for s in 0..samplers.len() {
            set1_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(sampler_base + s as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        let set1_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set1_bindings),
                None,
            )?
        };

        // NRD spaces: resourcesSpaceIndex (0) and constantBufferAndSamplersSpaceIndex (1)
        let set_layouts = if inst_info.resources_space_index == 0 {
            [set0_layout, set1_layout]
        } else {
            [set1_layout, set0_layout]
        };
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )?
        };

        // Create compute pipelines from NRD's embedded SPIR-V
        let mut pipelines = Vec::with_capacity(pipe_infos.len());
        for p in &pipe_infos {
            if p.spirv_bytecode.is_null() || p.spirv_size == 0 {
                return Err(RtNrdError::InvalidConfig(
                    "NRD pipeline missing SPIR-V bytecode".to_string(),
                ));
            }
            let bytes = unsafe {
                std::slice::from_raw_parts(p.spirv_bytecode.cast::<u8>(), p.spirv_size as usize)
            };
            let words = engine_rhi::spirv_bytes_to_words(bytes);
            let module = unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&words),
                    None,
                )?
            };
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)];
            let pipe = unsafe {
                let res = device.create_compute_pipelines(rhi.pipeline_cache, &info, None);
                device.destroy_shader_module(module, None);
                res.map_err(|(_, e)| e)?[0]
            };
            pipelines.push(pipe);
        }

        // Pre-allocate descriptor pool & per-slot sets for MAX_DISPATCHES * FRAMES_IN_FLIGHT
        let total_dispatches = MAX_DISPATCHES as u32 * FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: total_dispatches * max_tex_slots,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: total_dispatches * max_store_slots,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: total_dispatches,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: total_dispatches * samplers.len() as u32,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(total_dispatches * 2)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        let set0_layouts = vec![set0_layout; MAX_DISPATCHES];
        let set1_layouts = vec![set1_layout; MAX_DISPATCHES];
        let mut set0_slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut set1_slots = Vec::with_capacity(FRAMES_IN_FLIGHT);

        for slot_cbufs in &cbufs {
            let set0_list = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set0_layouts),
                )?
            };
            let set1_list = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set1_layouts),
                )?
            };

            // Pre-populate Set 1 (persistent per-slot constant buffer + immutable sampler bindings)
            for (d, &set1) in set1_list.iter().enumerate() {
                let buf_info = [vk::DescriptorBufferInfo::default()
                    .buffer(slot_cbufs[d].buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)];
                let sampler_infos: Vec<vk::DescriptorImageInfo> = samplers
                    .iter()
                    .map(|&s| vk::DescriptorImageInfo::default().sampler(s))
                    .collect();

                let mut writes = Vec::with_capacity(1 + sampler_infos.len());
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set1)
                        .dst_binding(cb_binding)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buf_info),
                );
                for (s, info) in sampler_infos.iter().enumerate() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set1)
                            .dst_binding(sampler_base + s as u32)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(std::slice::from_ref(info)),
                    );
                }
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }
            }

            set0_slots.push(set0_list);
            set1_slots.push(set1_list);
        }

        let set0_per_slot: [Vec<vk::DescriptorSet>; FRAMES_IN_FLIGHT] = set0_slots
            .try_into()
            .unwrap_or_else(|_| unreachable!());
        let set1_per_slot: [Vec<vk::DescriptorSet>; FRAMES_IN_FLIGHT] = set1_slots
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        let perm_states = vec![TrackedImageState::default(); perm_pool.len()];
        let trans_states = vec![TrackedImageState::default(); trans_pool.len()];

        Ok(Self {
            instance,
            extent: vk::Extent2D { width, height },
            perm_pool,
            trans_pool,
            perm_states,
            trans_states,
            out_diff_state: TrackedImageState::default(),
            out_spec_state: TrackedImageState::default(),
            dummy_sampled,
            dummy_storage,
            dummies_initialized: false,
            samplers,
            cbufs,
            set0_layout,
            set1_layout,
            pipeline_layout,
            pipelines,
            descriptor_pool,
            set0_per_slot,
            set1_per_slot,
            dispatches_scratch: vec![NrdDispatchInfo::default(); MAX_DISPATCHES],
            resources_scratch: vec![NrdResourceBinding::default(); MAX_RESOURCES_PER_FRAME],
            tex_infos_scratch: Vec::with_capacity(max_tex_slots as usize),
            store_infos_scratch: Vec::with_capacity(max_store_slots as usize),
            max_tex_slots,
            max_store_slots,
            tex_offset,
            store_offset,
            frame_index: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        inputs: &NrdInputs<'_>,
        out_diff_rad_hitdist: &GpuImage,
        out_spec_rad_hitdist: &GpuImage,
        view_to_clip: [[f32; 4]; 4],
        view_to_clip_prev: [[f32; 4]; 4],
        world_to_view: [[f32; 4]; 4],
        world_to_view_prev: [[f32; 4]; 4],
        jitter: [f32; 2],
        jitter_prev: [f32; 2],
        denoising_range: f32,
        time_delta_ms: f32,
        reset_history: bool,
        frame_slot: usize,
    ) -> Result<()> {
        let slot = frame_slot % FRAMES_IN_FLIGHT;
        if !self.dummies_initialized {
            let init_dummies = [
                sync::image_barrier(
                    self.dummy_sampled.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    self.dummy_sampled.subresource_range(),
                ),
                sync::image_barrier(
                    self.dummy_storage.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    self.dummy_storage.subresource_range(),
                ),
            ];
            unsafe {
                sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &init_dummies);
            }
            self.dummies_initialized = true;
        }

        let common = NrdCommonSettingsC {
            view_to_clip: flatten_mat4(view_to_clip),
            view_to_clip_prev: flatten_mat4(view_to_clip_prev),
            world_to_view: flatten_mat4(world_to_view),
            world_to_view_prev: flatten_mat4(world_to_view_prev),
            motion_vector_scale: [1.0, 1.0, 0.0],
            motion_vector_bias: [0.0, 0.0, 0.0],
            camera_jitter: jitter,
            camera_jitter_prev: jitter_prev,
            resource_size: [self.extent.width as u16, self.extent.height as u16],
            resource_size_prev: [self.extent.width as u16, self.extent.height as u16],
            rect_size: [self.extent.width as u16, self.extent.height as u16],
            rect_size_prev: [self.extent.width as u16, self.extent.height as u16],
            view_z_scale: 1.0,
            time_delta_between_frames: time_delta_ms,
            denoising_range,
            disocclusion_threshold: 0.015,
            frame_index: self.frame_index,
            accumulation_mode: if reset_history || self.frame_index == 0 {
                2 // CLEAR_AND_RESTART
            } else {
                0 // CONTINUE
            },
            is_motion_vector_in_world_space: 0,
            is_history_confidence_available: 1,
            enable_validation: 0,
        };

        let res = unsafe { nrd_shim_set_common_settings(self.instance, &common) };
        if res != NRD_RESULT_SUCCESS {
            return Err(RtNrdError::Nrd(res));
        }

        let reblur = NrdReblurSettingsC::default();
        let res = unsafe { nrd_shim_set_reblur_settings(self.instance, 0, &reblur) };
        if res != NRD_RESULT_SUCCESS {
            return Err(RtNrdError::Nrd(res));
        }

        let mut dispatch_count = 0u32;
        let mut resource_count = 0u32;

        let res = unsafe {
            nrd_shim_get_compute_dispatches(
                self.instance,
                0,
                self.dispatches_scratch.as_mut_ptr(),
                self.dispatches_scratch.len() as u32,
                &mut dispatch_count,
                self.resources_scratch.as_mut_ptr(),
                self.resources_scratch.len() as u32,
                &mut resource_count,
            )
        };
        if res != NRD_RESULT_SUCCESS {
            return Err(RtNrdError::Nrd(res));
        }
        if dispatch_count as usize > MAX_DISPATCHES {
            return Err(RtNrdError::InvalidConfig(format!(
                "NRD dispatch count {dispatch_count} exceeds MAX_DISPATCHES {MAX_DISPATCHES}"
            )));
        }

        let mut motion_state = TrackedImageState {
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_SAMPLED_READ,
        };

        for d in 0..(dispatch_count as usize) {
            let dd = self.dispatches_scratch[d];

            if !dd.constant_buffer_data.is_null() && dd.constant_buffer_data_size > 0 {
                let src = unsafe {
                    std::slice::from_raw_parts(
                        dd.constant_buffer_data,
                        dd.constant_buffer_data_size as usize,
                    )
                };
                let dst = self.cbufs[slot][d].mapped_slice_mut()?;
                dst[..src.len()].copy_from_slice(src);
            }

            let dd_res = unsafe {
                std::slice::from_raw_parts(dd.resources, dd.resources_num as usize)
            };

            let mut barriers: Vec<vk::ImageMemoryBarrier2<'_>> = Vec::with_capacity(dd_res.len());
            self.tex_infos_scratch.clear();
            self.store_infos_scratch.clear();

            for rb in dd_res {
                let is_storage = rb.descriptor_type == NRD_DESCRIPTOR_STORAGE_TEXTURE;
                let (req_layout, req_access) = if is_storage {
                    (
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    )
                } else {
                    (
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                    )
                };

                let view = match rb.resource_type {
                    NRD_RES_IN_DIFF_RADIANCE_HITDIST => inputs.diff_rad_hitdist.view,
                    NRD_RES_IN_SPEC_RADIANCE_HITDIST => inputs.spec_rad_hitdist.view,
                    NRD_RES_IN_NORMAL_ROUGHNESS => inputs.normal_roughness.view,
                    NRD_RES_IN_VIEWZ => inputs.view_z.view,
                    NRD_RES_IN_MV => {
                        maybe_push_barrier(
                            inputs.motion,
                            &mut motion_state,
                            req_layout,
                            req_access,
                            &mut barriers,
                        );
                        inputs.motion.view
                    }
                    NRD_RES_IN_DIFF_CONFIDENCE => inputs.diff_confidence.view,
                    NRD_RES_IN_SPEC_CONFIDENCE => inputs.spec_confidence.view,
                    NRD_RES_OUT_DIFF_RADIANCE_HITDIST => {
                        maybe_push_barrier(
                            out_diff_rad_hitdist,
                            &mut self.out_diff_state,
                            req_layout,
                            req_access,
                            &mut barriers,
                        );
                        out_diff_rad_hitdist.view
                    }
                    NRD_RES_OUT_SPEC_RADIANCE_HITDIST => {
                        maybe_push_barrier(
                            out_spec_rad_hitdist,
                            &mut self.out_spec_state,
                            req_layout,
                            req_access,
                            &mut barriers,
                        );
                        out_spec_rad_hitdist.view
                    }
                    NRD_RES_PERMANENT_POOL => {
                        let idx = rb.index_in_pool as usize;
                        maybe_push_barrier(
                            &self.perm_pool[idx],
                            &mut self.perm_states[idx],
                            req_layout,
                            req_access,
                            &mut barriers,
                        );
                        self.perm_pool[idx].view
                    }
                    NRD_RES_TRANSIENT_POOL => {
                        let idx = rb.index_in_pool as usize;
                        maybe_push_barrier(
                            &self.trans_pool[idx],
                            &mut self.trans_states[idx],
                            req_layout,
                            req_access,
                            &mut barriers,
                        );
                        self.trans_pool[idx].view
                    }
                    _ => {
                        if is_storage {
                            self.dummy_storage.view
                        } else {
                            self.dummy_sampled.view
                        }
                    }
                };

                let info = vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(req_layout);
                if is_storage {
                    self.store_infos_scratch.push(info);
                } else {
                    self.tex_infos_scratch.push(info);
                }
            }

            while self.tex_infos_scratch.len() < self.max_tex_slots as usize {
                self.tex_infos_scratch.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(self.dummy_sampled.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            while self.store_infos_scratch.len() < self.max_store_slots as usize {
                self.store_infos_scratch.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(self.dummy_storage.view)
                        .image_layout(vk::ImageLayout::GENERAL),
                );
            }

            let set0 = self.set0_per_slot[slot][d];
            let set1 = self.set1_per_slot[slot][d];

            let mut writes = Vec::with_capacity(
                self.tex_infos_scratch.len() + self.store_infos_scratch.len(),
            );
            for (t, info) in self.tex_infos_scratch.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set0)
                        .dst_binding(self.tex_offset + t as u32)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(std::slice::from_ref(info)),
                );
            }
            for (s, info) in self.store_infos_scratch.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set0)
                        .dst_binding(self.store_offset + s as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info)),
                );
            }

            unsafe {
                device.update_descriptor_sets(&writes, &[]);
                if !barriers.is_empty() {
                    sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &barriers);
                }
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines[dd.pipeline_index as usize],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set0, set1],
                    &[],
                );
                device.cmd_dispatch(cmd, dd.grid_width as u32, dd.grid_height as u32, 1);
            }
        }

        // Transition NRD outputs to GENERAL for nrd_compose.comp imageLoad
        let mut post_barriers = Vec::with_capacity(2);
        maybe_push_barrier(
            out_diff_rad_hitdist,
            &mut self.out_diff_state,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            &mut post_barriers,
        );
        maybe_push_barrier(
            out_spec_rad_hitdist,
            &mut self.out_spec_state,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            &mut post_barriers,
        );
        if !post_barriers.is_empty() {
            unsafe {
                sync::cmd_pipeline_barrier2(device, cmd, &[], &[], &post_barriers);
            }
        }

        self.frame_index += 1;
        Ok(())
    }

    fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let device = rhi.device.clone();
        unsafe {
            for p in self.pipelines {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set0_layout, None);
            device.destroy_descriptor_set_layout(self.set1_layout, None);
            for s in self.samplers {
                device.destroy_sampler(s, None);
            }
            if !self.instance.is_null() {
                nrd_shim_destroy_instance(self.instance);
            }
        }

        let alloc = rhi.allocator_mut();
        for slot_cbufs in self.cbufs {
            for cb in slot_cbufs {
                cb.destroy(&device, alloc)?;
            }
        }
        for img in self.perm_pool {
            img.destroy(&device, alloc)?;
        }
        for img in self.trans_pool {
            img.destroy(&device, alloc)?;
        }
        self.dummy_sampled.destroy(&device, alloc)?;
        self.dummy_storage.destroy(&device, alloc)?;
        Ok(())
    }
}

#[cfg(has_nrd)]
fn maybe_push_barrier<'a>(
    image: &'a GpuImage,
    state: &mut TrackedImageState,
    req_layout: vk::ImageLayout,
    req_access: vk::AccessFlags2,
    barriers: &mut Vec<vk::ImageMemoryBarrier2<'a>>,
) {
    let layout_changed = state.layout != req_layout;
    let has_write_hazard = state.access.contains(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        || (req_access.contains(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            && !state.access.is_empty());

    if layout_changed || has_write_hazard {
        barriers.push(sync::image_barrier(
            image.image,
            state.layout,
            req_layout,
            state.stage,
            state.access,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            req_access,
            image.subresource_range(),
        ));
    }
    state.layout = req_layout;
    state.stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
    state.access = req_access;
}

#[cfg(has_nrd)]
fn flatten_mat4(m: [[f32; 4]; 4]) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]
}

#[cfg(has_nrd)]
fn nrd_format_to_vk(fmt: u32) -> Result<vk::Format> {
    // Matches nrd::Format enum in NRDDescs.h
    let vk_fmt = match fmt {
        0 => vk::Format::R8_UNORM,
        1 => vk::Format::R8_SNORM,
        2 => vk::Format::R8_UINT,
        3 => vk::Format::R8_SINT,
        4 => vk::Format::R8G8_UNORM,
        5 => vk::Format::R8G8_SNORM,
        6 => vk::Format::R8G8_UINT,
        7 => vk::Format::R8G8_SINT,
        8 => vk::Format::R8G8B8A8_UNORM,
        9 => vk::Format::R8G8B8A8_SNORM,
        10 => vk::Format::R8G8B8A8_UINT,
        11 => vk::Format::R8G8B8A8_SINT,
        12 => vk::Format::R8G8B8A8_SRGB,
        13 => vk::Format::R16_UNORM,
        14 => vk::Format::R16_SNORM,
        15 => vk::Format::R16_UINT,
        16 => vk::Format::R16_SINT,
        17 => vk::Format::R16_SFLOAT,
        18 => vk::Format::R16G16_UNORM,
        19 => vk::Format::R16G16_SNORM,
        20 => vk::Format::R16G16_UINT,
        21 => vk::Format::R16G16_SINT,
        22 => vk::Format::R16G16_SFLOAT,
        23 => vk::Format::R16G16B16A16_UNORM,
        24 => vk::Format::R16G16B16A16_SNORM,
        25 => vk::Format::R16G16B16A16_UINT,
        26 => vk::Format::R16G16B16A16_SINT,
        27 => vk::Format::R16G16B16A16_SFLOAT,
        28 => vk::Format::R32_UINT,
        29 => vk::Format::R32_SINT,
        30 => vk::Format::R32_SFLOAT,
        31 => vk::Format::R32G32_UINT,
        32 => vk::Format::R32G32_SINT,
        33 => vk::Format::R32G32_SFLOAT,
        34 => vk::Format::R32G32B32_UINT,
        35 => vk::Format::R32G32B32_SINT,
        36 => vk::Format::R32G32B32_SFLOAT,
        37 => vk::Format::R32G32B32A32_UINT,
        38 => vk::Format::R32G32B32A32_SINT,
        39 => vk::Format::R32G32B32A32_SFLOAT,
        40 => vk::Format::A2B10G10R10_UNORM_PACK32,
        41 => vk::Format::A2B10G10R10_UINT_PACK32,
        42 => vk::Format::B10G11R11_UFLOAT_PACK32,
        43 => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        _ => {
            return Err(RtNrdError::InvalidConfig(format!(
                "Unsupported NRD format enum {fmt}"
            )));
        }
    };
    Ok(vk_fmt)
}
