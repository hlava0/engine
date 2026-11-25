use std::ffi::{CStr, CString, c_void};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;
use gpu_allocator::AllocatorDebugSettings;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::{Result, RhiError};
use crate::pipeline_cache::PipelineCache;
use crate::vendor::{VendorProbeInput, VendorProfile, detect_rebar};

/// Vulkan 1.3 API version constant (`vk::make_api_version(0, 1, 3, 0)`).
pub const API_VERSION_1_3: u32 = vk::make_api_version(0, 1, 3, 0);

/// Vulkan 1.4 API version constant (`vk::make_api_version(0, 1, 4, 0)`).
pub const API_VERSION_1_4: u32 = vk::make_api_version(0, 1, 4, 0);

/// Configuration for initializing `RhiDevice`.
#[derive(Debug, Clone)]
pub struct RhiConfig {
    pub app_name: String,
    pub enable_validation: bool,
    pub enable_sync_validation: bool,
    pub surface_extensions: Vec<&'static CStr>,
    pub enable_swapchain: bool,
}

impl Default for RhiConfig {
    fn default() -> Self {
        Self {
            app_name: "vulkan-gpu-driven-engine".to_string(),
            enable_validation: cfg!(debug_assertions),
            enable_sync_validation: cfg!(debug_assertions),
            surface_extensions: Vec::new(),
            enable_swapchain: false,
        }
    }
}

/// Supported cooperative matrix tile configuration probed from `VK_KHR_cooperative_matrix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooperativeMatrixProps {
    pub m_size: u32,
    pub n_size: u32,
    pub k_size: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub scope: vk::ScopeKHR,
    pub saturating_accumulation: bool,
}

/// Probed Vulkan 1.4, ray tracing, and cooperative matrix capabilities of the selected GPU.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub device_name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,

    // Vulkan 1.2 / 1.3 / 1.4 core features
    pub buffer_device_address: bool,
    pub runtime_descriptor_array: bool,
    pub descriptor_binding_partially_bound: bool,
    pub descriptor_binding_sampled_image_update_after_bind: bool,
    pub descriptor_binding_storage_image_update_after_bind: bool,
    pub descriptor_binding_storage_buffer_update_after_bind: bool,
    pub shader_sampled_image_array_non_uniform_indexing: bool,
    pub scalar_block_layout: bool,
    pub draw_indirect_count: bool,
    pub timeline_semaphore: bool,
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    pub shader_int64: bool,
    pub shader_float16: bool,
    pub vulkan_memory_model: bool,
    pub vulkan_memory_model_device_scope: bool,

    // Ray tracing extensions
    pub khr_acceleration_structure: bool,
    pub khr_ray_tracing_pipeline: bool,
    pub khr_ray_query: bool,
    pub khr_deferred_host_operations: bool,
    pub khr_pipeline_library: bool,
    pub khr_ray_tracing_position_fetch: bool,

    // Cooperative matrix
    pub khr_cooperative_matrix: bool,
    pub cooperative_matrix_properties: Vec<CooperativeMatrixProps>,

    // Vulkan 1.3 subgroup size control
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    pub required_subgroup_size_stages: vk::ShaderStageFlags,
    pub max_compute_workgroup_subgroups: u32,

    // Compute shader derivatives (required by NRD SPIR-V)
    pub khr_compute_shader_derivatives: bool,

    // Presentation / swapchain support
    pub khr_swapchain: bool,

    // Vendor-specific hardware optimization profile
    pub vendor_profile: VendorProfile,
}

/// Queue family indices for graphics, compute, and transfer.
#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
}

/// Atomic counters for Vulkan validation messages.
#[derive(Debug, Default)]
pub struct ValidationCounters {
    pub errors: AtomicU32,
    pub warnings: AtomicU32,
}

/// Core Vulkan 1.4 RHI device owning instance, physical/logical device, queues, extension loaders, and allocator.
pub struct RhiDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub capabilities: DeviceCapabilities,
    pub queue_families: QueueFamilyIndices,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,

    pub acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    pub ray_tracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub cooperative_matrix: Option<ash::khr::cooperative_matrix::Instance>,
    pub swapchain_loader: Option<ash::khr::swapchain::Device>,
    pub surface_loader: Option<ash::khr::surface::Instance>,
    pub pipeline_cache: vk::PipelineCache,

    pipeline_cache_owner: PipelineCache,
    allocator: ManuallyDrop<Allocator>,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_counters: Arc<ValidationCounters>,
    validation_user_data: *mut Arc<ValidationCounters>,
}

impl RhiDevice {
    /// Initialize Vulkan 1.4 (falling back gracefully to Vulkan 1.3), select best physical device,
    /// probe capabilities, and create logical device + allocator.
    pub fn new(config: RhiConfig) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let instance_version = unsafe { entry.try_enumerate_instance_version()? }
            .unwrap_or(vk::API_VERSION_1_0);
        if instance_version < API_VERSION_1_3 {
            return Err(RhiError::MissingCapability(format!(
                "Vulkan 1.3 or newer is required (loader reports {}.{}.{})",
                vk::api_version_major(instance_version),
                vk::api_version_minor(instance_version),
                vk::api_version_patch(instance_version),
            )));
        }
        let requested_api_version = if instance_version >= API_VERSION_1_4 {
            API_VERSION_1_4
        } else {
            log::info!(
                "Vulkan 1.4 not advertised by loader ({}.{}.{}); falling back to Vulkan 1.3",
                vk::api_version_major(instance_version),
                vk::api_version_minor(instance_version),
                vk::api_version_patch(instance_version),
            );
            API_VERSION_1_3
        };

        let validation_counters = Arc::new(ValidationCounters::default());
        let validation_user_data = Box::into_raw(Box::new(Arc::clone(&validation_counters)));

        let app_name = CString::new(config.app_name.as_str())?;
        let engine_name = CString::new("vulkan-gpu-driven-engine")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(requested_api_version);

        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let available_exts = unsafe { entry.enumerate_instance_extension_properties(None)? };

        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";
        let has_validation_layer = available_layers.iter().any(|l| {
            (unsafe { CStr::from_ptr(l.layer_name.as_ptr()) }) == validation_layer_name
        });
        let has_debug_utils = available_exts.iter().any(|e| {
            (unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }) == ash::ext::debug_utils::NAME
        });
        let has_validation_features = available_exts.iter().any(|e| {
            (unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
                == ash::ext::validation_features::NAME
        });

        let use_validation = config.enable_validation && has_validation_layer && has_debug_utils;
        if config.enable_validation && !use_validation {
            log::warn!(
                "Validation requested but VK_LAYER_KHRONOS_validation or VK_EXT_debug_utils is unavailable"
            );
        }

        let mut layer_names: Vec<*const i8> = Vec::new();
        let mut ext_names: Vec<*const i8> = Vec::new();

        for &surf_ext in &config.surface_extensions {
            if !ext_names
                .iter()
                .any(|&p| unsafe { CStr::from_ptr(p) } == surf_ext)
            {
                ext_names.push(surf_ext.as_ptr());
            }
        }

        if use_validation {
            layer_names.push(validation_layer_name.as_ptr());
            ext_names.push(ash::ext::debug_utils::NAME.as_ptr());
            if config.enable_sync_validation && has_validation_features {
                ext_names.push(ash::ext::validation_features::NAME.as_ptr());
            }
        }

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback))
            .user_data(validation_user_data.cast::<c_void>());

        let sync_enables = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&sync_enables);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&ext_names);

        if use_validation {
            instance_info = instance_info.push_next(&mut debug_create_info);
            if config.enable_sync_validation && has_validation_features {
                instance_info = instance_info.push_next(&mut validation_features);
            }
        }

        let instance = match unsafe { entry.create_instance(&instance_info, None) } {
            Ok(inst) => inst,
            Err(err) => {
                unsafe { drop(Box::from_raw(validation_user_data)) };
                return Err(err.into());
            }
        };

        let (debug_utils, debug_messenger) = if use_validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe { du.create_debug_utils_messenger(&debug_create_info, None)? };
            (Some(du), messenger)
        } else {
            (None, vk::DebugUtilsMessengerEXT::null())
        };

        let (physical_device, capabilities, queue_families) =
            select_physical_device(&entry, &instance, config.enable_swapchain)?;

        let (device, graphics_queue, compute_queue, transfer_queue) = create_logical_device(
            &instance,
            physical_device,
            &capabilities,
            queue_families,
            config.enable_swapchain,
        )?;

        let acceleration_structure = if capabilities.khr_acceleration_structure {
            Some(ash::khr::acceleration_structure::Device::new(
                &instance, &device,
            ))
        } else {
            None
        };

        let ray_tracing_pipeline = if capabilities.khr_ray_tracing_pipeline {
            Some(ash::khr::ray_tracing_pipeline::Device::new(
                &instance, &device,
            ))
        } else {
            None
        };

        let cooperative_matrix = if capabilities.khr_cooperative_matrix {
            Some(ash::khr::cooperative_matrix::Instance::new(
                &entry, &instance,
            ))
        } else {
            None
        };

        let surface_loader = if config.enable_swapchain || !config.surface_extensions.is_empty() {
            Some(ash::khr::surface::Instance::new(&entry, &instance))
        } else {
            None
        };

        let swapchain_loader = if config.enable_swapchain {
            Some(ash::khr::swapchain::Device::new(&instance, &device))
        } else {
            None
        };

        let pipeline_cache_path = PipelineCache::default_path(&capabilities);
        let pipeline_cache_owner = PipelineCache::load_or_create(&device, pipeline_cache_path)?;
        let pipeline_cache = pipeline_cache_owner.handle();

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: capabilities.buffer_device_address,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            capabilities,
            queue_families,
            graphics_queue,
            compute_queue,
            transfer_queue,
            acceleration_structure,
            ray_tracing_pipeline,
            cooperative_matrix,
            swapchain_loader,
            surface_loader,
            pipeline_cache,
            pipeline_cache_owner,
            allocator: ManuallyDrop::new(allocator),
            debug_utils,
            debug_messenger,
            validation_counters,
            validation_user_data,
        })
    }

    /// Borrow the `gpu-allocator` instance mutably.
    #[inline]
    pub fn allocator_mut(&mut self) -> &mut Allocator {
        &mut self.allocator
    }

    /// Number of Vulkan validation errors observed since device creation.
    #[inline]
    pub fn validation_errors(&self) -> u32 {
        self.validation_counters.errors.load(Ordering::Relaxed)
    }

    /// Number of Vulkan validation warnings observed since device creation.
    #[inline]
    pub fn validation_warnings(&self) -> u32 {
        self.validation_counters.warnings.load(Ordering::Relaxed)
    }

    /// Persist the active `VkPipelineCache` to disk immediately.
    pub fn save_pipeline_cache(&self) -> Result<()> {
        self.pipeline_cache_owner.save(&self.device)
    }

    /// Wait for the logical device to become idle.
    #[inline]
    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }

    /// Create a command pool for the given queue family.
    pub fn create_command_pool(
        &self,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<vk::CommandPool> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(flags);
        Ok(unsafe { self.device.create_command_pool(&info, None)? })
    }

    /// Allocate a primary command buffer from `pool`.
    pub fn allocate_command_buffer(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe { self.device.allocate_command_buffers(&info)? };
        Ok(cmds[0])
    }

    /// Build a `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` for `stage` if subgroup size
    /// control is enabled and `required_size` is valid for this device.
    pub fn subgroup_stage_create_info(
        &self,
        stage: vk::ShaderStageFlags,
        required_size: Option<u32>,
    ) -> Option<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'static>> {
        let req = required_size?;
        let caps = &self.capabilities;
        let vp = &caps.vendor_profile;
        if !caps.subgroup_size_control
            || !caps.required_subgroup_size_stages.contains(stage)
            || !req.is_power_of_two()
            || req < vp.min_subgroup_size
            || req > vp.max_subgroup_size
        {
            return None;
        }
        Some(
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(req),
        )
    }

    /// Return `PipelineShaderStageCreateFlags` with `REQUIRE_FULL_SUBGROUPS` set when supported
    /// and valid for `local_size_x` and `required_size`.
    pub fn subgroup_stage_flags(
        &self,
        stage: vk::ShaderStageFlags,
        required_size: Option<u32>,
        local_size_x: u32,
    ) -> vk::PipelineShaderStageCreateFlags {
        let caps = &self.capabilities;
        if !caps.compute_full_subgroups || stage != vk::ShaderStageFlags::COMPUTE {
            return vk::PipelineShaderStageCreateFlags::empty();
        }
        let effective_size = required_size.unwrap_or(caps.vendor_profile.max_subgroup_size);
        if effective_size > 0 && local_size_x.is_multiple_of(effective_size) {
            vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        } else {
            vk::PipelineShaderStageCreateFlags::empty()
        }
    }

    /// Submit a recorded primary command buffer to `queue` and wait for completion via a fence.
    pub fn submit_and_wait(&self, queue: vk::Queue, cmd: vk::CommandBuffer) -> Result<()> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };

        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        let submit_res = unsafe { self.device.queue_submit(queue, &submit, fence) };
        if let Err(err) = submit_res {
            unsafe { self.device.destroy_fence(fence, None) };
            return Err(err.into());
        }

        let wait_res = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) };
        unsafe { self.device.destroy_fence(fence, None) };
        wait_res?;
        Ok(())
    }
}

impl Drop for RhiDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.pipeline_cache_owner.destroy(&self.device);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some(du) = &self.debug_utils
                && self.debug_messenger != vk::DebugUtilsMessengerEXT::null()
            {
                du.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
            drop(Box::from_raw(self.validation_user_data));
        }
    }
}

fn select_physical_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    require_swapchain: bool,
) -> Result<(vk::PhysicalDevice, DeviceCapabilities, QueueFamilyIndices)> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    if devices.is_empty() {
        return Err(RhiError::NoSuitableDevice);
    }

    let mut best: Option<(i32, vk::PhysicalDevice, DeviceCapabilities, QueueFamilyIndices)> = None;

    for pdev in devices {
        let Some(queues) = find_queue_families(instance, pdev) else {
            continue;
        };
        let caps = probe_capabilities(entry, instance, pdev);

        // Require at least Vulkan 1.3, BDA, Dynamic Rendering, Synchronization2, DrawIndirectCount
        if caps.api_version < API_VERSION_1_3
            || !caps.buffer_device_address
            || !caps.dynamic_rendering
            || !caps.synchronization2
            || !caps.draw_indirect_count
            || (require_swapchain && !caps.khr_swapchain)
        {
            continue;
        }

        let mut score = match caps.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 10_000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1_000,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 100,
            _ => 10,
        };
        if caps.api_version >= API_VERSION_1_4 {
            score += 50;
        }
        if caps.khr_ray_tracing_pipeline {
            score += 500;
        }
        if caps.khr_cooperative_matrix {
            score += 200;
        }

        if best.as_ref().is_none_or(|(best_score, _, _, _)| score > *best_score) {
            best = Some((score, pdev, caps, queues));
        }
    }

    best.map(|(_, pdev, caps, queues)| (pdev, caps, queues))
        .ok_or(RhiError::NoSuitableDevice)
}

fn find_queue_families(
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
) -> Option<QueueFamilyIndices> {
    let props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };

    let graphics = props
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))? as u32;

    // Prefer dedicated compute queue, fall back to any compute-capable queue.
    let compute = props
        .iter()
        .position(|q| {
            q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .or_else(|| {
            props
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        })? as u32;

    // Prefer dedicated transfer queue, fall back to graphics.
    let transfer = props
        .iter()
        .position(|q| {
            q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && !q.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .or_else(|| {
            props
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER))
        })
        .unwrap_or(graphics as usize) as u32;

    Some(QueueFamilyIndices {
        graphics,
        compute,
        transfer,
    })
}

fn probe_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
) -> DeviceCapabilities {
    let props = unsafe { instance.get_physical_device_properties(pdev) };
    let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let exts = unsafe {
        instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
    };
    let has_ext = |name: &CStr| {
        exts.iter()
            .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name)
    };

    let khr_acceleration_structure = has_ext(ash::khr::acceleration_structure::NAME);
    let khr_ray_tracing_pipeline = has_ext(ash::khr::ray_tracing_pipeline::NAME);
    let khr_ray_query = has_ext(ash::khr::ray_query::NAME);
    let khr_deferred_host_operations = has_ext(ash::khr::deferred_host_operations::NAME);
    let khr_pipeline_library = has_ext(ash::khr::pipeline_library::NAME);
    let khr_ray_tracing_position_fetch = has_ext(ash::khr::ray_tracing_position_fetch::NAME);
    let khr_cooperative_matrix = has_ext(ash::khr::cooperative_matrix::NAME);
    let khr_compute_shader_derivatives = has_ext(c"VK_KHR_compute_shader_derivatives");
    let khr_swapchain = has_ext(ash::khr::swapchain::NAME);
    let ext_descriptor_buffer = has_ext(ash::ext::descriptor_buffer::NAME);
    let nv_ray_tracing_invocation_reorder = has_ext(c"VK_NV_ray_tracing_invocation_reorder");

    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut subgroup_size_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();

    let mut props2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut driver_props)
        .push_next(&mut subgroup_props)
        .push_next(&mut subgroup_size_props);
    if khr_acceleration_structure {
        props2 = props2.push_next(&mut as_props);
    }
    unsafe { instance.get_physical_device_properties2(pdev, &mut props2) };
    let _ = props2;

    let driver_name = unsafe { CStr::from_ptr(driver_props.driver_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let driver_info = unsafe { CStr::from_ptr(driver_props.driver_info.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
    let has_rebar = detect_rebar(&mem_props);

    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut coop_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut rt_pos_feat = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default();
    let mut csd_feat = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default();

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut v12)
        .push_next(&mut v13);

    if khr_cooperative_matrix {
        features2 = features2.push_next(&mut coop_feat);
    }
    if khr_ray_tracing_position_fetch {
        features2 = features2.push_next(&mut rt_pos_feat);
    }
    if khr_compute_shader_derivatives {
        // VK_KHR_compute_shader_derivatives reuses the same sType/layout as NV for computeDerivativeGroupQuads
        csd_feat.s_type = vk::StructureType::PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV;
        features2 = features2.push_next(&mut csd_feat);
    }

    unsafe { instance.get_physical_device_features2(pdev, &mut features2) };
    let shader_int64 = features2.features.shader_int64 != 0;
    let _ = features2;

    let cooperative_matrix_properties = if khr_cooperative_matrix {
        let coop_inst = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        unsafe {
            coop_inst
                .get_physical_device_cooperative_matrix_properties(pdev)
                .unwrap_or_default()
        }
        .into_iter()
        .map(|p| CooperativeMatrixProps {
            m_size: p.m_size,
            n_size: p.n_size,
            k_size: p.k_size,
            a_type: p.a_type,
            b_type: p.b_type,
            c_type: p.c_type,
            result_type: p.result_type,
            scope: p.scope,
            saturating_accumulation: p.saturating_accumulation != 0,
        })
        .collect()
    } else {
        Vec::new()
    };

    let subgroup_size_control = v13.subgroup_size_control != 0;
    let compute_full_subgroups = v13.compute_full_subgroups != 0;
    let khr_coop_enabled = khr_cooperative_matrix && coop_feat.cooperative_matrix != 0;
    let shader_float16 = v12.shader_float16 != 0;

    let vendor_profile = VendorProfile::from_probe(VendorProbeInput {
        vendor_id: props.vendor_id,
        driver_id: driver_props.driver_id,
        driver_name,
        driver_info,
        subgroup_size: subgroup_props.subgroup_size,
        min_subgroup_size: subgroup_size_props.min_subgroup_size,
        max_subgroup_size: subgroup_size_props.max_subgroup_size,
        required_subgroup_size_stages: subgroup_size_props.required_subgroup_size_stages,
        subgroup_size_control,
        khr_cooperative_matrix: khr_coop_enabled,
        shader_float16,
        cooperative_matrix_properties: &cooperative_matrix_properties,
        has_rebar,
        min_as_scratch_alignment: as_props.min_acceleration_structure_scratch_offset_alignment,
        nv_ray_tracing_invocation_reorder,
        ext_descriptor_buffer,
    });

    DeviceCapabilities {
        device_name,
        device_type: props.device_type,
        api_version: props.api_version,
        driver_version: props.driver_version,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        buffer_device_address: v12.buffer_device_address != 0,
        runtime_descriptor_array: v12.runtime_descriptor_array != 0,
        descriptor_binding_partially_bound: v12.descriptor_binding_partially_bound != 0,
        descriptor_binding_sampled_image_update_after_bind: v12
            .descriptor_binding_sampled_image_update_after_bind
            != 0,
        descriptor_binding_storage_image_update_after_bind: v12
            .descriptor_binding_storage_image_update_after_bind
            != 0,
        descriptor_binding_storage_buffer_update_after_bind: v12
            .descriptor_binding_storage_buffer_update_after_bind
            != 0,
        shader_sampled_image_array_non_uniform_indexing: v12
            .shader_sampled_image_array_non_uniform_indexing
            != 0,
        scalar_block_layout: v12.scalar_block_layout != 0,
        draw_indirect_count: v12.draw_indirect_count != 0,
        timeline_semaphore: v12.timeline_semaphore != 0,
        synchronization2: v13.synchronization2 != 0,
        dynamic_rendering: v13.dynamic_rendering != 0,
        shader_int64,
        shader_float16,
        vulkan_memory_model: v12.vulkan_memory_model != 0,
        vulkan_memory_model_device_scope: v12.vulkan_memory_model_device_scope != 0,
        khr_acceleration_structure,
        khr_ray_tracing_pipeline,
        khr_ray_query,
        khr_deferred_host_operations,
        khr_pipeline_library,
        khr_ray_tracing_position_fetch: khr_ray_tracing_position_fetch
            && rt_pos_feat.ray_tracing_position_fetch != 0,
        khr_cooperative_matrix: khr_coop_enabled,
        cooperative_matrix_properties,
        subgroup_size_control,
        compute_full_subgroups,
        required_subgroup_size_stages: subgroup_size_props.required_subgroup_size_stages,
        max_compute_workgroup_subgroups: subgroup_size_props.max_compute_workgroup_subgroups,
        khr_compute_shader_derivatives: khr_compute_shader_derivatives
            && csd_feat.compute_derivative_group_quads != 0,
        khr_swapchain,
        vendor_profile,
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
    caps: &DeviceCapabilities,
    queues: QueueFamilyIndices,
    enable_swapchain: bool,
) -> Result<(ash::Device, vk::Queue, vk::Queue, vk::Queue)> {
    let mut unique_families = vec![queues.graphics];
    if !unique_families.contains(&queues.compute) {
        unique_families.push(queues.compute);
    }
    if !unique_families.contains(&queues.transfer) {
        unique_families.push(queues.transfer);
    }

    let priority = [1.0f32];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = unique_families
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priority)
        })
        .collect();

    let mut enabled_exts: Vec<*const i8> = Vec::new();
    if enable_swapchain {
        enabled_exts.push(ash::khr::swapchain::NAME.as_ptr());
    }
    if caps.khr_acceleration_structure {
        enabled_exts.push(ash::khr::acceleration_structure::NAME.as_ptr());
    }
    if caps.khr_ray_tracing_pipeline {
        enabled_exts.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
    }
    if caps.khr_ray_query {
        enabled_exts.push(ash::khr::ray_query::NAME.as_ptr());
    }
    if caps.khr_deferred_host_operations {
        enabled_exts.push(ash::khr::deferred_host_operations::NAME.as_ptr());
    }
    if caps.khr_pipeline_library {
        enabled_exts.push(ash::khr::pipeline_library::NAME.as_ptr());
    }
    if caps.khr_ray_tracing_position_fetch {
        enabled_exts.push(ash::khr::ray_tracing_position_fetch::NAME.as_ptr());
    }
    if caps.khr_cooperative_matrix {
        enabled_exts.push(ash::khr::cooperative_matrix::NAME.as_ptr());
    }
    if caps.khr_compute_shader_derivatives {
        enabled_exts.push(c"VK_KHR_compute_shader_derivatives".as_ptr());
    }

    let core_features = vk::PhysicalDeviceFeatures::default().shader_int64(caps.shader_int64);

    let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(caps.buffer_device_address)
        .descriptor_indexing(true)
        .runtime_descriptor_array(caps.runtime_descriptor_array)
        .descriptor_binding_partially_bound(caps.descriptor_binding_partially_bound)
        .descriptor_binding_sampled_image_update_after_bind(
            caps.descriptor_binding_sampled_image_update_after_bind,
        )
        .descriptor_binding_storage_image_update_after_bind(
            caps.descriptor_binding_storage_image_update_after_bind,
        )
        .descriptor_binding_storage_buffer_update_after_bind(
            caps.descriptor_binding_storage_buffer_update_after_bind,
        )
        .shader_sampled_image_array_non_uniform_indexing(
            caps.shader_sampled_image_array_non_uniform_indexing,
        )
        .scalar_block_layout(caps.scalar_block_layout)
        .draw_indirect_count(caps.draw_indirect_count)
        .timeline_semaphore(caps.timeline_semaphore)
        .shader_float16(caps.shader_float16)
        .vulkan_memory_model(caps.vulkan_memory_model)
        .vulkan_memory_model_device_scope(caps.vulkan_memory_model_device_scope);

    let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(caps.synchronization2)
        .dynamic_rendering(caps.dynamic_rendering)
        .subgroup_size_control(caps.subgroup_size_control)
        .compute_full_subgroups(caps.compute_full_subgroups)
        .shader_demote_to_helper_invocation(true);

    let mut as_feat = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
        .acceleration_structure(true);
    let mut rt_pipe_feat = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
        .ray_tracing_pipeline(true);
    let mut rq_feat = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut rt_pos_feat = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
        .ray_tracing_position_fetch(true);
    let mut coop_feat =
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut csd_feat = vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
        .compute_derivative_group_quads(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(core_features)
        .push_next(&mut v12)
        .push_next(&mut v13);

    if caps.khr_acceleration_structure {
        features2 = features2.push_next(&mut as_feat);
    }
    if caps.khr_ray_tracing_pipeline {
        features2 = features2.push_next(&mut rt_pipe_feat);
    }
    if caps.khr_ray_query {
        features2 = features2.push_next(&mut rq_feat);
    }
    if caps.khr_ray_tracing_position_fetch {
        features2 = features2.push_next(&mut rt_pos_feat);
    }
    if caps.khr_cooperative_matrix {
        features2 = features2.push_next(&mut coop_feat);
    }
    if caps.khr_compute_shader_derivatives {
        features2 = features2.push_next(&mut csd_feat);
    }

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&enabled_exts)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(pdev, &device_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(queues.graphics, 0) };
    let compute_queue = unsafe { device.get_device_queue(queues.compute, 0) };
    let transfer_queue = unsafe { device.get_device_queue(queues.transfer, 0) };

    Ok((device, graphics_queue, compute_queue, transfer_queue))
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    let data = unsafe { &*p_callback_data };
    let msg = if data.p_message.is_null() {
        "<null>"
    } else {
        unsafe { CStr::from_ptr(data.p_message) }
            .to_str()
            .unwrap_or("<invalid utf-8>")
    };

    if !p_user_data.is_null() {
        let counters = unsafe { &*p_user_data.cast::<Arc<ValidationCounters>>() };
        if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            counters.errors.fetch_add(1, Ordering::Relaxed);
        } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
            counters.warnings.fetch_add(1, Ordering::Relaxed);
        }
    }

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan Validation] {msg}");
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan Validation] {msg}");
    }

    vk::FALSE
}
