#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NrdInstance NrdInstance;

typedef struct NrdLibraryInfo {
    uint32_t sampler_offset;
    uint32_t texture_offset;
    uint32_t constant_buffer_offset;
    uint32_t storage_texture_and_buffer_offset;
    uint8_t version_major;
    uint8_t version_minor;
    uint8_t version_build;
    uint8_t normal_encoding;
    uint8_t roughness_encoding;
} NrdLibraryInfo;

typedef struct NrdTextureInfo {
    uint32_t format;
    uint16_t downsample_factor;
} NrdTextureInfo;

typedef struct NrdPipelineInfo {
    const void* spirv_bytecode;
    uint64_t spirv_size;
    uint32_t texture_descriptors_num;
    uint32_t storage_texture_descriptors_num;
    uint8_t has_constant_data;
} NrdPipelineInfo;

typedef struct NrdInstanceInfo {
    uint32_t constant_buffer_and_samplers_space_index;
    uint32_t resources_space_index;
    uint32_t constant_buffer_register_index;
    uint32_t samplers_base_register_index;
    uint32_t resources_base_register_index;
    uint32_t constant_buffer_max_data_size;
    uint32_t samplers_num;
    uint32_t pipelines_num;
    uint32_t permanent_pool_size;
    uint32_t transient_pool_size;
    uint32_t per_set_textures_max_num;
    uint32_t per_set_storage_textures_max_num;
    const NrdPipelineInfo* pipelines;
    const NrdTextureInfo* permanent_pool;
    const NrdTextureInfo* transient_pool;
    const uint32_t* samplers;
} NrdInstanceInfo;

typedef struct NrdResourceBinding {
    uint32_t descriptor_type; // 0 = TEXTURE, 1 = STORAGE_TEXTURE
    uint32_t resource_type;   // nrd::ResourceType
    uint16_t index_in_pool;
} NrdResourceBinding;

typedef struct NrdDispatchInfo {
    const char* name;
    uint32_t identifier;
    const NrdResourceBinding* resources;
    uint32_t resources_num;
    const uint8_t* constant_buffer_data;
    uint32_t constant_buffer_data_size;
    uint8_t constant_buffer_matches_prev;
    uint16_t pipeline_index;
    uint16_t grid_width;
    uint16_t grid_height;
} NrdDispatchInfo;

typedef struct NrdCommonSettingsC {
    float view_to_clip[16];
    float view_to_clip_prev[16];
    float world_to_view[16];
    float world_to_view_prev[16];
    float motion_vector_scale[3];
    float motion_vector_bias[3];
    float camera_jitter[2];
    float camera_jitter_prev[2];
    uint16_t resource_size[2];
    uint16_t resource_size_prev[2];
    uint16_t rect_size[2];
    uint16_t rect_size_prev[2];
    float view_z_scale;
    float time_delta_between_frames;
    float denoising_range;
    float disocclusion_threshold;
    uint32_t frame_index;
    uint8_t accumulation_mode; // 0 = CONTINUE, 1 = RESTART, 2 = CLEAR_AND_RESTART
    uint8_t is_motion_vector_in_world_space;
    uint8_t is_history_confidence_available;
    uint8_t enable_validation;
} NrdCommonSettingsC;

typedef struct NrdReblurSettingsC {
    uint32_t max_accumulated_frame_num;
    uint32_t max_fast_accumulated_frame_num;
    uint32_t max_stabilized_frame_num;
    uint32_t history_fix_frame_num;
    float diffuse_prepass_blur_radius;
    float specular_prepass_blur_radius;
    float min_blur_radius;
    float max_blur_radius;
    float antilag_luminance_sigma_scale;
    float antilag_luminance_sensitivity;
    uint8_t enable_anti_firefly;
} NrdReblurSettingsC;

void nrd_shim_get_library_info(NrdLibraryInfo* out_info);

uint32_t nrd_shim_create_instance(uint32_t denoiser_type, uint32_t identifier, NrdInstance** out_instance);
void nrd_shim_destroy_instance(NrdInstance* instance);

uint32_t nrd_shim_get_instance_info(
    const NrdInstance* instance,
    NrdInstanceInfo* out_info,
    NrdPipelineInfo* out_pipelines,
    uint32_t max_pipelines,
    NrdTextureInfo* out_perm_pool,
    uint32_t max_perm_pool,
    NrdTextureInfo* out_trans_pool,
    uint32_t max_trans_pool,
    uint32_t* out_samplers,
    uint32_t max_samplers);

uint32_t nrd_shim_set_common_settings(NrdInstance* instance, const NrdCommonSettingsC* settings);
uint32_t nrd_shim_set_reblur_settings(NrdInstance* instance, uint32_t identifier, const NrdReblurSettingsC* settings);

uint32_t nrd_shim_get_compute_dispatches(
    NrdInstance* instance,
    uint32_t identifier,
    NrdDispatchInfo* out_dispatches,
    uint32_t max_dispatches,
    uint32_t* out_dispatches_num,
    NrdResourceBinding* out_resources,
    uint32_t max_resources,
    uint32_t* out_resources_num);

#ifdef __cplusplus
}
#endif
