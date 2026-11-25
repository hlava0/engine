#include "fsr31_shim.h"

#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Stub for FrameInterpolationSwapchainVK (unused by upscaler-only path).
FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const* /*config*/) {
    return FFX_OK;
}

static void fsr31_msg_callback(FfxMsgType /*type*/, const wchar_t *message) {
    std::fputs("[fsr31_sdk] ", stderr);
    if (message) {
        for (size_t i = 0; message[i]; ++i) {
            std::fputc((char)(message[i] & 0x7f), stderr);
        }
    }
    std::fputc('\n', stderr);
}

static PFN_vkVoidFunction VKAPI_CALL shim_get_device_proc_addr(VkDevice device, const char *pName) {
    PFN_vkVoidFunction fn = vkGetDeviceProcAddr(device, pName);
    if (!fn && pName && std::strcmp(pName, "vkGetBufferMemoryRequirements2KHR") == 0) {
        fn = vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2");
    }
    return fn;
}

struct EngineFsr31Handle {
    VkDeviceContext dev_ctx;
    FfxFsr3UpscalerContext ctx;
    void *scratch;
    FfxInterface backend;
};

static FfxResource make_tex_resource(
    uint64_t img_handle,
    uint32_t vk_format,
    uint32_t w,
    uint32_t h,
    const wchar_t *name,
    FfxResourceStates state,
    uint32_t usage
) {
    FfxResourceDescription desc = {};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.format = ffxGetSurfaceFormatVK((VkFormat)vk_format);
    desc.width = w;
    desc.height = h;
    desc.depth = 1;
    desc.mipCount = 1;
    desc.flags = FFX_RESOURCE_FLAGS_NONE;
    desc.usage = (FfxResourceUsage)usage;
    return ffxGetResourceVK((void *)(uintptr_t)img_handle, desc, name, state);
}

extern "C" EngineFsr31Handle *engine_fsr31_create(const EngineFsr31CreateInfo *info) {
    if (!info) return nullptr;

    VkDeviceContext tmp_ctx = {};
    tmp_ctx.vkDevice = (VkDevice)(uintptr_t)info->vk_device;
    tmp_ctx.vkPhysicalDevice = (VkPhysicalDevice)(uintptr_t)info->vk_physical_device;
    // Use shim_get_device_proc_addr so Vulkan 1.1+ core functions (e.g. vkGetBufferMemoryRequirements2)
    // satisfy *KHR queries even when legacy KHR extensions are not explicitly enabled.
    tmp_ctx.vkDeviceProcAddr = shim_get_device_proc_addr;

    size_t scratch_size = ffxGetScratchMemorySizeVK(tmp_ctx.vkPhysicalDevice, 1);
    void *scratch = std::calloc(1, scratch_size);
    if (!scratch) return nullptr;

    auto *h = (EngineFsr31Handle *)std::calloc(1, sizeof(EngineFsr31Handle));
    if (!h) {
        std::free(scratch);
        return nullptr;
    }
    h->dev_ctx = tmp_ctx;
    h->scratch = scratch;

    FfxDevice ffx_dev = ffxGetDeviceVK(&h->dev_ctx);
    if (ffxGetInterfaceVK(&h->backend, ffx_dev, scratch, scratch_size, 1) != FFX_OK) {
        std::free(scratch);
        std::free(h);
        return nullptr;
    }

    FfxFsr3UpscalerContextDescription desc = {};
    desc.backendInterface = h->backend;
    desc.fpMessage = fsr31_msg_callback;
    desc.maxRenderSize.width = info->max_render_w;
    desc.maxRenderSize.height = info->max_render_h;
    desc.maxUpscaleSize.width = info->max_upscale_w;
    desc.maxUpscaleSize.height = info->max_upscale_h;
    desc.flags = 0;
    if (info->enable_hdr) {
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if (info->inverted_depth) {
        desc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
    }

    FfxErrorCode rc = ffxFsr3UpscalerContextCreate(&h->ctx, &desc);
    if (rc != FFX_OK) {
        std::fprintf(stderr, "[fsr31_shim] ffxFsr3UpscalerContextCreate failed with FfxErrorCode=%d\n", (int)rc);
        ffxFsr3UpscalerContextDestroy(&h->ctx);
        std::free(scratch);
        std::free(h);
        return nullptr;
    }
    return h;
}

extern "C" int32_t engine_fsr31_dispatch(EngineFsr31Handle *handle, const EngineFsr31DispatchInfo *info) {
    if (!handle || !info) return -1;

    FfxFsr3UpscalerDispatchDescription d = {};
    d.commandList = ffxGetCommandListVK((VkCommandBuffer)(uintptr_t)info->vk_cmd);
    d.color = make_tex_resource(
        info->color_image, info->color_format, info->color_w, info->color_h,
        L"FSR3_Color", FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY
    );
    d.depth = make_tex_resource(
        info->depth_image, info->depth_format, info->color_w, info->color_h,
        L"FSR3_Depth", FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY
    );
    d.motionVectors = make_tex_resource(
        info->motion_image, info->motion_format, info->color_w, info->color_h,
        L"FSR3_Motion", FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY
    );
    d.output = make_tex_resource(
        info->output_image, info->output_format, info->output_w, info->output_h,
        L"FSR3_Output", FFX_RESOURCE_STATE_UNORDERED_ACCESS, FFX_RESOURCE_USAGE_UAV
    );

    d.jitterOffset.x = info->jitter_x;
    d.jitterOffset.y = info->jitter_y;
    d.motionVectorScale.x = info->mv_scale_x;
    d.motionVectorScale.y = info->mv_scale_y;
    d.renderSize.width = info->color_w;
    d.renderSize.height = info->color_h;
    d.upscaleSize.width = info->output_w;
    d.upscaleSize.height = info->output_h;
    d.enableSharpening = info->sharpness > 0.0f;
    d.sharpness = info->sharpness;
    d.frameTimeDelta = info->frame_time_ms;
    d.preExposure = 1.0f;
    d.reset = info->reset != 0;
    d.cameraNear = info->camera_near;
    d.cameraFar = info->camera_far;
    d.cameraFovAngleVertical = info->fov_y_rad;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = 0;

    return (int32_t)ffxFsr3UpscalerContextDispatch(&handle->ctx, &d);
}

extern "C" void engine_fsr31_destroy(EngineFsr31Handle *handle) {
    if (!handle) return;
    ffxFsr3UpscalerContextDestroy(&handle->ctx);
    std::free(handle->scratch);
    std::free(handle);
}
