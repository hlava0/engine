#![allow(dead_code)]

use std::ffi::c_char;

#[repr(C)]
pub struct NrdInstance {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NrdLibraryInfo {
    pub sampler_offset: u32,
    pub texture_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
    pub version_major: u8,
    pub version_minor: u8,
    pub version_build: u8,
    pub normal_encoding: u8,
    pub roughness_encoding: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NrdTextureInfo {
    pub format: u32,
    pub downsample_factor: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdPipelineInfo {
    pub spirv_bytecode: *const std::ffi::c_void,
    pub spirv_size: u64,
    pub texture_descriptors_num: u32,
    pub storage_texture_descriptors_num: u32,
    pub has_constant_data: u8,
}

impl Default for NrdPipelineInfo {
    fn default() -> Self {
        Self {
            spirv_bytecode: std::ptr::null(),
            spirv_size: 0,
            texture_descriptors_num: 0,
            storage_texture_descriptors_num: 0,
            has_constant_data: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdInstanceInfo {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers_num: u32,
    pub pipelines_num: u32,
    pub permanent_pool_size: u32,
    pub transient_pool_size: u32,
    pub per_set_textures_max_num: u32,
    pub per_set_storage_textures_max_num: u32,
    pub pipelines: *const NrdPipelineInfo,
    pub permanent_pool: *const NrdTextureInfo,
    pub transient_pool: *const NrdTextureInfo,
    pub samplers: *const u32,
}

impl Default for NrdInstanceInfo {
    fn default() -> Self {
        Self {
            constant_buffer_and_samplers_space_index: 0,
            resources_space_index: 0,
            constant_buffer_register_index: 0,
            samplers_base_register_index: 0,
            resources_base_register_index: 0,
            constant_buffer_max_data_size: 0,
            samplers_num: 0,
            pipelines_num: 0,
            permanent_pool_size: 0,
            transient_pool_size: 0,
            per_set_textures_max_num: 0,
            per_set_storage_textures_max_num: 0,
            pipelines: std::ptr::null(),
            permanent_pool: std::ptr::null(),
            transient_pool: std::ptr::null(),
            samplers: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NrdResourceBinding {
    pub descriptor_type: u32,
    pub resource_type: u32,
    pub index_in_pool: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdDispatchInfo {
    pub name: *const c_char,
    pub identifier: u32,
    pub resources: *const NrdResourceBinding,
    pub resources_num: u32,
    pub constant_buffer_data: *const u8,
    pub constant_buffer_data_size: u32,
    pub constant_buffer_matches_prev: u8,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
}

impl Default for NrdDispatchInfo {
    fn default() -> Self {
        Self {
            name: std::ptr::null(),
            identifier: 0,
            resources: std::ptr::null(),
            resources_num: 0,
            constant_buffer_data: std::ptr::null(),
            constant_buffer_data_size: 0,
            constant_buffer_matches_prev: 0,
            pipeline_index: 0,
            grid_width: 0,
            grid_height: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdCommonSettingsC {
    pub view_to_clip: [f32; 16],
    pub view_to_clip_prev: [f32; 16],
    pub world_to_view: [f32; 16],
    pub world_to_view_prev: [f32; 16],
    pub motion_vector_scale: [f32; 3],
    pub motion_vector_bias: [f32; 3],
    pub camera_jitter: [f32; 2],
    pub camera_jitter_prev: [f32; 2],
    pub resource_size: [u16; 2],
    pub resource_size_prev: [u16; 2],
    pub rect_size: [u16; 2],
    pub rect_size_prev: [u16; 2],
    pub view_z_scale: f32,
    pub time_delta_between_frames: f32,
    pub denoising_range: f32,
    pub disocclusion_threshold: f32,
    pub frame_index: u32,
    pub accumulation_mode: u8,
    pub is_motion_vector_in_world_space: u8,
    pub is_history_confidence_available: u8,
    pub enable_validation: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdReblurSettingsC {
    pub max_accumulated_frame_num: u32,
    pub max_fast_accumulated_frame_num: u32,
    pub max_stabilized_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub antilag_luminance_sigma_scale: f32,
    pub antilag_luminance_sensitivity: f32,
    pub enable_anti_firefly: u8,
}

impl Default for NrdReblurSettingsC {
    fn default() -> Self {
        Self {
            max_accumulated_frame_num: 30,
            max_fast_accumulated_frame_num: 6,
            max_stabilized_frame_num: 63,
            history_fix_frame_num: 3,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_blur_radius: 1.0,
            max_blur_radius: 30.0,
            antilag_luminance_sigma_scale: 2.0,
            antilag_luminance_sensitivity: 3.0,
            enable_anti_firefly: 1,
        }
    }
}

pub const NRD_RESULT_SUCCESS: u32 = 0;
pub const NRD_DENOISER_REBLUR_DIFFUSE_SPECULAR: u32 = 6;

pub const NRD_DESCRIPTOR_TEXTURE: u32 = 0;
pub const NRD_DESCRIPTOR_STORAGE_TEXTURE: u32 = 1;

// nrd::ResourceType enum values from NRDDescs.h
pub const NRD_RES_IN_MV: u32 = 0;
pub const NRD_RES_IN_NORMAL_ROUGHNESS: u32 = 1;
pub const NRD_RES_IN_VIEWZ: u32 = 2;
pub const NRD_RES_IN_DIFF_CONFIDENCE: u32 = 3;
pub const NRD_RES_IN_SPEC_CONFIDENCE: u32 = 4;
pub const NRD_RES_IN_DISOCCLUSION_THRESHOLD_MIX: u32 = 5;
pub const NRD_RES_IN_DIFF_RADIANCE_HITDIST: u32 = 6;
pub const NRD_RES_IN_SPEC_RADIANCE_HITDIST: u32 = 7;
pub const NRD_RES_OUT_DIFF_RADIANCE_HITDIST: u32 = 18;
pub const NRD_RES_OUT_SPEC_RADIANCE_HITDIST: u32 = 19;
pub const NRD_RES_OUT_VALIDATION: u32 = 29;
pub const NRD_RES_TRANSIENT_POOL: u32 = 30;
pub const NRD_RES_PERMANENT_POOL: u32 = 31;

// nrd::NormalEncoding enum values from NRDDescs.h
pub const NRD_NORMAL_ENCODING_RGBA8_UNORM: u8 = 0;
pub const NRD_NORMAL_ENCODING_RGBA8_SNORM: u8 = 1;
pub const NRD_NORMAL_ENCODING_R10_G10_B10_A2_UNORM: u8 = 2;
pub const NRD_NORMAL_ENCODING_RGBA16_UNORM: u8 = 3;
pub const NRD_NORMAL_ENCODING_RGBA16_SNORM: u8 = 4;

unsafe extern "C" {
    pub fn nrd_shim_get_library_info(out_info: *mut NrdLibraryInfo);
    pub fn nrd_shim_create_instance(
        denoiser_type: u32,
        identifier: u32,
        out_instance: *mut *mut NrdInstance,
    ) -> u32;
    pub fn nrd_shim_destroy_instance(instance: *mut NrdInstance);
    pub fn nrd_shim_get_instance_info(
        instance: *const NrdInstance,
        out_info: *mut NrdInstanceInfo,
        out_pipelines: *mut NrdPipelineInfo,
        max_pipelines: u32,
        out_perm_pool: *mut NrdTextureInfo,
        max_perm_pool: u32,
        out_trans_pool: *mut NrdTextureInfo,
        max_trans_pool: u32,
        out_samplers: *mut u32,
        max_samplers: u32,
    ) -> u32;
    pub fn nrd_shim_set_common_settings(
        instance: *mut NrdInstance,
        settings: *const NrdCommonSettingsC,
    ) -> u32;
    pub fn nrd_shim_set_reblur_settings(
        instance: *mut NrdInstance,
        identifier: u32,
        settings: *const NrdReblurSettingsC,
    ) -> u32;
    pub fn nrd_shim_get_compute_dispatches(
        instance: *mut NrdInstance,
        identifier: u32,
        out_dispatches: *mut NrdDispatchInfo,
        max_dispatches: u32,
        out_dispatches_num: *mut u32,
        out_resources: *mut NrdResourceBinding,
        max_resources: u32,
        out_resources_num: *mut u32,
    ) -> u32;
}
