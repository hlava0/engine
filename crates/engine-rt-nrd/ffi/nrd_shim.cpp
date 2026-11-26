#include "nrd_shim.h"

#include <NRD.h>

#include <cstdlib>
#include <cstring>

namespace {
void* NRD_CALL shim_alloc(void*, size_t size, size_t) {
    return std::malloc(size);
}

void* NRD_CALL shim_realloc(void*, void* ptr, size_t size, size_t) {
    return std::realloc(ptr, size);
}

void NRD_CALL shim_free(void*, void* ptr) {
    std::free(ptr);
}
} // namespace

extern "C" {

void nrd_shim_get_library_info(NrdLibraryInfo* out_info) {
    if (!out_info) {
        return;
    }
    const nrd::LibraryDesc* lib = nrd::GetLibraryDesc();
    out_info->sampler_offset = lib->spirvBindingOffsets.samplerOffset;
    out_info->texture_offset = lib->spirvBindingOffsets.textureOffset;
    out_info->constant_buffer_offset = lib->spirvBindingOffsets.constantBufferOffset;
    out_info->storage_texture_and_buffer_offset = lib->spirvBindingOffsets.storageTextureAndBufferOffset;
    out_info->version_major = lib->versionMajor;
    out_info->version_minor = lib->versionMinor;
    out_info->version_build = lib->versionBuild;
    out_info->normal_encoding = static_cast<uint8_t>(lib->normalEncoding);
    out_info->roughness_encoding = static_cast<uint8_t>(lib->roughnessEncoding);
}

uint32_t nrd_shim_create_instance(uint32_t denoiser_type, uint32_t identifier, NrdInstance** out_instance) {
    if (!out_instance) {
        return static_cast<uint32_t>(nrd::Result::INVALID_ARGUMENT);
    }

    nrd::DenoiserDesc denoiser_desc{};
    denoiser_desc.identifier = identifier;
    denoiser_desc.denoiser = static_cast<nrd::Denoiser>(denoiser_type);

    nrd::InstanceCreationDesc creation_desc{};
    creation_desc.allocationCallbacks.Allocate = shim_alloc;
    creation_desc.allocationCallbacks.Reallocate = shim_realloc;
    creation_desc.allocationCallbacks.Free = shim_free;
    creation_desc.allocationCallbacks.userArg = nullptr;
    creation_desc.denoisers = &denoiser_desc;
    creation_desc.denoisersNum = 1;

    nrd::Instance* inst = nullptr;
    nrd::Result res = nrd::CreateInstance(creation_desc, inst);
    *out_instance = reinterpret_cast<NrdInstance*>(inst);
    return static_cast<uint32_t>(res);
}

void nrd_shim_destroy_instance(NrdInstance* instance) {
    if (instance) {
        nrd::DestroyInstance(*reinterpret_cast<nrd::Instance*>(instance));
    }
}

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
    uint32_t max_samplers) {
    if (!instance || !out_info) {
        return static_cast<uint32_t>(nrd::Result::INVALID_ARGUMENT);
    }

    const nrd::InstanceDesc* desc = nrd::GetInstanceDesc(*reinterpret_cast<const nrd::Instance*>(instance));
    if (!desc) {
        return static_cast<uint32_t>(nrd::Result::FAILURE);
    }

    out_info->constant_buffer_and_samplers_space_index = desc->constantBufferAndSamplersSpaceIndex;
    out_info->resources_space_index = desc->resourcesSpaceIndex;
    out_info->constant_buffer_register_index = desc->constantBufferRegisterIndex;
    out_info->samplers_base_register_index = desc->samplersBaseRegisterIndex;
    out_info->resources_base_register_index = desc->resourcesBaseRegisterIndex;
    out_info->constant_buffer_max_data_size = desc->constantBufferMaxDataSize;
    out_info->samplers_num = desc->samplersNum;
    out_info->pipelines_num = desc->pipelinesNum;
    out_info->permanent_pool_size = desc->permanentPoolSize;
    out_info->transient_pool_size = desc->transientPoolSize;
    out_info->per_set_textures_max_num = desc->descriptorPoolDesc.perSetTexturesMaxNum;
    out_info->per_set_storage_textures_max_num = desc->descriptorPoolDesc.perSetStorageTexturesMaxNum;

    if (out_pipelines) {
        uint32_t count = desc->pipelinesNum < max_pipelines ? desc->pipelinesNum : max_pipelines;
        for (uint32_t i = 0; i < count; ++i) {
            const nrd::PipelineDesc& p = desc->pipelines[i];
            out_pipelines[i].spirv_bytecode = p.computeShaderSPIRV.bytecode;
            out_pipelines[i].spirv_size = p.computeShaderSPIRV.size;
            out_pipelines[i].has_constant_data = p.hasConstantData ? 1 : 0;

            uint32_t tex_num = 0;
            uint32_t store_num = 0;
            for (uint32_t r = 0; r < p.resourceRangesNum; ++r) {
                if (p.resourceRanges[r].descriptorType == nrd::DescriptorType::TEXTURE) {
                    tex_num += p.resourceRanges[r].descriptorsNum;
                } else {
                    store_num += p.resourceRanges[r].descriptorsNum;
                }
            }
            out_pipelines[i].texture_descriptors_num = tex_num;
            out_pipelines[i].storage_texture_descriptors_num = store_num;
        }
    }

    if (out_perm_pool) {
        uint32_t count = desc->permanentPoolSize < max_perm_pool ? desc->permanentPoolSize : max_perm_pool;
        for (uint32_t i = 0; i < count; ++i) {
            out_perm_pool[i].format = static_cast<uint32_t>(desc->permanentPool[i].format);
            out_perm_pool[i].downsample_factor = desc->permanentPool[i].downsampleFactor;
        }
    }

    if (out_trans_pool) {
        uint32_t count = desc->transientPoolSize < max_trans_pool ? desc->transientPoolSize : max_trans_pool;
        for (uint32_t i = 0; i < count; ++i) {
            out_trans_pool[i].format = static_cast<uint32_t>(desc->transientPool[i].format);
            out_trans_pool[i].downsample_factor = desc->transientPool[i].downsampleFactor;
        }
    }

    if (out_samplers) {
        uint32_t count = desc->samplersNum < max_samplers ? desc->samplersNum : max_samplers;
        for (uint32_t i = 0; i < count; ++i) {
            out_samplers[i] = static_cast<uint32_t>(desc->samplers[i]);
        }
    }

    out_info->pipelines = out_pipelines;
    out_info->permanent_pool = out_perm_pool;
    out_info->transient_pool = out_trans_pool;
    out_info->samplers = out_samplers;

    return static_cast<uint32_t>(nrd::Result::SUCCESS);
}

uint32_t nrd_shim_set_common_settings(NrdInstance* instance, const NrdCommonSettingsC* settings) {
    if (!instance || !settings) {
        return static_cast<uint32_t>(nrd::Result::INVALID_ARGUMENT);
    }

    nrd::CommonSettings cs{};
    std::memcpy(cs.viewToClipMatrix, settings->view_to_clip, sizeof(cs.viewToClipMatrix));
    std::memcpy(cs.viewToClipMatrixPrev, settings->view_to_clip_prev, sizeof(cs.viewToClipMatrixPrev));
    std::memcpy(cs.worldToViewMatrix, settings->world_to_view, sizeof(cs.worldToViewMatrix));
    std::memcpy(cs.worldToViewMatrixPrev, settings->world_to_view_prev, sizeof(cs.worldToViewMatrixPrev));
    std::memcpy(cs.motionVectorScale, settings->motion_vector_scale, sizeof(cs.motionVectorScale));
    std::memcpy(cs.motionVectorBias, settings->motion_vector_bias, sizeof(cs.motionVectorBias));
    std::memcpy(cs.cameraJitter, settings->camera_jitter, sizeof(cs.cameraJitter));
    std::memcpy(cs.cameraJitterPrev, settings->camera_jitter_prev, sizeof(cs.cameraJitterPrev));
    std::memcpy(cs.resourceSize, settings->resource_size, sizeof(cs.resourceSize));
    std::memcpy(cs.resourceSizePrev, settings->resource_size_prev, sizeof(cs.resourceSizePrev));
    std::memcpy(cs.rectSize, settings->rect_size, sizeof(cs.rectSize));
    std::memcpy(cs.rectSizePrev, settings->rect_size_prev, sizeof(cs.rectSizePrev));
    cs.viewZScale = settings->view_z_scale;
    cs.timeDeltaBetweenFrames = settings->time_delta_between_frames;
    cs.denoisingRange = settings->denoising_range;
    cs.disocclusionThreshold = settings->disocclusion_threshold;
    cs.frameIndex = settings->frame_index;
    cs.accumulationMode = static_cast<nrd::AccumulationMode>(settings->accumulation_mode);
    cs.isMotionVectorInWorldSpace = settings->is_motion_vector_in_world_space != 0;
    cs.isHistoryConfidenceAvailable = settings->is_history_confidence_available != 0;
    cs.enableValidation = settings->enable_validation != 0;

    return static_cast<uint32_t>(nrd::SetCommonSettings(*reinterpret_cast<nrd::Instance*>(instance), cs));
}

uint32_t nrd_shim_set_reblur_settings(NrdInstance* instance, uint32_t identifier, const NrdReblurSettingsC* settings) {
    if (!instance || !settings) {
        return static_cast<uint32_t>(nrd::Result::INVALID_ARGUMENT);
    }

    nrd::ReblurSettings rs{};
    rs.maxAccumulatedFrameNum = settings->max_accumulated_frame_num;
    rs.maxFastAccumulatedFrameNum = settings->max_fast_accumulated_frame_num;
    rs.maxStabilizedFrameNum = settings->max_stabilized_frame_num;
    rs.historyFixFrameNum = settings->history_fix_frame_num;
    rs.diffusePrepassBlurRadius = settings->diffuse_prepass_blur_radius;
    rs.specularPrepassBlurRadius = settings->specular_prepass_blur_radius;
    rs.minBlurRadius = settings->min_blur_radius;
    rs.maxBlurRadius = settings->max_blur_radius;
    rs.antilagSettings.luminanceSigmaScale = settings->antilag_luminance_sigma_scale;
    rs.antilagSettings.luminanceSensitivity = settings->antilag_luminance_sensitivity;
    rs.enableAntiFirefly = settings->enable_anti_firefly != 0;

    return static_cast<uint32_t>(nrd::SetDenoiserSettings(*reinterpret_cast<nrd::Instance*>(instance), identifier, &rs));
}

uint32_t nrd_shim_get_compute_dispatches(
    NrdInstance* instance,
    uint32_t identifier,
    NrdDispatchInfo* out_dispatches,
    uint32_t max_dispatches,
    uint32_t* out_dispatches_num,
    NrdResourceBinding* out_resources,
    uint32_t max_resources,
    uint32_t* out_resources_num) {
    if (!instance || !out_dispatches_num || !out_resources_num) {
        return static_cast<uint32_t>(nrd::Result::INVALID_ARGUMENT);
    }

    nrd::Identifier id = identifier;
    const nrd::DispatchDesc* descs = nullptr;
    uint32_t count = 0;
    nrd::Result res = nrd::GetComputeDispatches(*reinterpret_cast<nrd::Instance*>(instance), &id, 1, descs, count);
    if (res != nrd::Result::SUCCESS) {
        return static_cast<uint32_t>(res);
    }

    *out_dispatches_num = count;
    uint32_t total_res = 0;
    for (uint32_t i = 0; i < count; ++i) {
        total_res += descs[i].resourcesNum;
    }
    *out_resources_num = total_res;

    if (!out_dispatches || !out_resources || count > max_dispatches || total_res > max_resources) {
        return static_cast<uint32_t>(nrd::Result::SUCCESS);
    }

    uint32_t res_cursor = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const nrd::DispatchDesc& src = descs[i];
        NrdDispatchInfo& dst = out_dispatches[i];
        dst.name = src.name;
        dst.identifier = src.identifier;
        dst.resources = &out_resources[res_cursor];
        dst.resources_num = src.resourcesNum;
        dst.constant_buffer_data = src.constantBufferData;
        dst.constant_buffer_data_size = src.constantBufferDataSize;
        dst.constant_buffer_matches_prev = src.constantBufferDataMatchesPreviousDispatch ? 1 : 0;
        dst.pipeline_index = src.pipelineIndex;
        dst.grid_width = src.gridWidth;
        dst.grid_height = src.gridHeight;

        for (uint32_t r = 0; r < src.resourcesNum; ++r) {
            out_resources[res_cursor].descriptor_type = static_cast<uint32_t>(src.resources[r].descriptorType);
            out_resources[res_cursor].resource_type = static_cast<uint32_t>(src.resources[r].type);
            out_resources[res_cursor].index_in_pool = src.resources[r].indexInPool;
            ++res_cursor;
        }
    }

    return static_cast<uint32_t>(nrd::Result::SUCCESS);
}

} // extern "C"
