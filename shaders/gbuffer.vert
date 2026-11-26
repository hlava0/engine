#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct GpuVertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct GpuInstanceData {
    mat4 model;
    mat4 prev_model;
    vec4 albedo_metallic;
    uint mesh_id;
    uint material_id;
    float roughness;
    uint _pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer FrameUniformsBuf {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    mat4 prev_view_proj;
    vec4 frustum_planes[6];
    vec4 jitter;
    uvec2 screen_size;
    uvec2 hiz_size;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBuf {
    GpuVertex data[];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer InstanceBuf {
    GpuInstanceData data[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer DrawInstanceIdBuf {
    uint ids[];
};

layout(push_constant, scalar) uniform PushConstants {
    FrameUniformsBuf frame_uniforms;
    VertexBuf vertices;
    InstanceBuf instances;
    DrawInstanceIdBuf draw_instance_ids;
} pc;

layout(location = 0) out vec4 out_cur_clip_unjittered;
layout(location = 1) out vec4 out_prev_clip_unjittered;
layout(location = 2) out vec3 out_world_normal;
layout(location = 3) out vec2 out_uv;
layout(location = 4) out vec4 out_albedo_metallic;
layout(location = 5) out float out_roughness;
layout(location = 6) out float out_view_z;
layout(location = 7) flat out uint out_material_id;

void main() {
    uint instance_id = pc.draw_instance_ids.ids[gl_InstanceIndex];
    GpuInstanceData inst = pc.instances.data[instance_id];
    GpuVertex v = pc.vertices.data[gl_VertexIndex];

    vec4 world_pos = inst.model * vec4(v.pos, 1.0);
    vec4 prev_world_pos = inst.prev_model * vec4(v.pos, 1.0);

    vec4 cur_clip = pc.frame_uniforms.view_proj * world_pos;
    vec4 prev_clip = pc.frame_uniforms.prev_view_proj * prev_world_pos;

    vec4 jittered_clip = cur_clip;
    jittered_clip.xy += pc.frame_uniforms.jitter.xy * jittered_clip.w;
    gl_Position = jittered_clip;

    out_cur_clip_unjittered = cur_clip;
    out_prev_clip_unjittered = prev_clip;
    out_world_normal = normalize(mat3(inst.model) * v.normal);
    out_uv = v.uv;
    out_albedo_metallic = inst.albedo_metallic;
    out_roughness = inst.roughness;
    out_view_z = -(pc.frame_uniforms.view * world_pos).z;
    out_material_id = inst.material_id;
}
