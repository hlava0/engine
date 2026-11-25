#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngineFsr31Handle EngineFsr31Handle;

typedef struct EngineFsr31CreateInfo {
    uint64_t vk_device;
    uint64_t vk_physical_device;
    void *vk_get_device_proc_addr;
    uint32_t max_render_w;
    uint32_t max_render_h;
    uint32_t max_upscale_w;
    uint32_t max_upscale_h;
    uint32_t enable_hdr;
    uint32_t inverted_depth;
} EngineFsr31CreateInfo;

typedef struct EngineFsr31DispatchInfo {
    uint64_t vk_cmd;
    uint64_t color_image;
    uint32_t color_format;
    uint32_t color_w;
    uint32_t color_h;
    uint64_t depth_image;
    uint32_t depth_format;
    uint64_t motion_image;
    uint32_t motion_format;
    uint64_t output_image;
    uint32_t output_format;
    uint32_t output_w;
    uint32_t output_h;
    float jitter_x;
    float jitter_y;
    float mv_scale_x;
    float mv_scale_y;
    float sharpness;
    float frame_time_ms;
    float camera_near;
    float camera_far;
    float fov_y_rad;
    uint32_t reset;
} EngineFsr31DispatchInfo;

EngineFsr31Handle *engine_fsr31_create(const EngineFsr31CreateInfo *info);
int32_t engine_fsr31_dispatch(EngineFsr31Handle *handle, const EngineFsr31DispatchInfo *info);
void engine_fsr31_destroy(EngineFsr31Handle *handle);

#ifdef __cplusplus
}
#endif
