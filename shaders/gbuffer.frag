#version 460

layout(location = 0) in vec4 in_cur_clip_unjittered;
layout(location = 1) in vec4 in_prev_clip_unjittered;
layout(location = 2) in vec3 in_world_normal;
layout(location = 3) in vec2 in_uv;
layout(location = 4) in vec4 in_albedo_metallic;
layout(location = 5) in float in_roughness;
layout(location = 6) in float in_view_z;
layout(location = 7) flat in uint in_material_id;

layout(location = 0) out vec4 out_albedo_metallic;
layout(location = 1) out vec4 out_normal_roughness;
layout(location = 2) out vec2 out_motion;
layout(location = 3) out float out_view_z;

void main() {
    vec2 cur_uv = (in_cur_clip_unjittered.xy / in_cur_clip_unjittered.w) * 0.5 + 0.5;
    vec2 prev_uv = (in_prev_clip_unjittered.xy / in_prev_clip_unjittered.w) * 0.5 + 0.5;

    out_albedo_metallic = in_albedo_metallic;
    out_normal_roughness = vec4(normalize(in_world_normal), in_roughness);
    out_motion = prev_uv - cur_uv;
    out_view_z = in_view_z;
}
