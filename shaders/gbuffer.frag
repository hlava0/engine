#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

struct GpuMaterialTextures {
    uint albedo_tex;
    uint normal_tex;
    uint metallic_roughness_tex;
    uint emissive_tex;
    uint alpha_mode;
    float alpha_cutoff;
    uint _pad0;
    uint _pad1;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer MaterialTexturesBuf {
    GpuMaterialTextures mats[];
};

layout(push_constant, scalar) uniform PushConstants {
    uvec2 frame_uniforms_ptr;
    uvec2 vertices_ptr;
    uvec2 instances_ptr;
    uvec2 draw_instance_ids_ptr;
    MaterialTexturesBuf material_textures;
    uint default_sampler_idx;
    uint _pad;
} pc;

layout(set = 0, binding = 0) uniform texture2D bindless_textures[];
layout(set = 0, binding = 3) uniform sampler bindless_samplers[];

layout(location = 0) in vec4 in_cur_clip_unjittered;
layout(location = 1) in vec4 in_prev_clip_unjittered;
layout(location = 2) in vec3 in_world_normal;
layout(location = 3) in vec2 in_uv;
layout(location = 4) in vec4 in_albedo_metallic;
layout(location = 5) in float in_roughness;
layout(location = 6) in float in_view_z;
layout(location = 7) flat in uint in_material_id;
layout(location = 8) in vec3 in_world_pos;

layout(location = 0) out vec4 out_albedo_metallic;
layout(location = 1) out vec4 out_normal_roughness;
layout(location = 2) out vec2 out_motion;
layout(location = 3) out float out_view_z;

vec3 perturb_normal(vec3 n, vec3 pos, vec2 uv, vec3 map_n) {
    vec3 dp1 = dFdx(pos);
    vec3 dp2 = dFdy(pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(t, t), dot(b, b)) + 1e-8);
    mat3 tbn = mat3(t * invmax, b * invmax, n);
    return normalize(tbn * map_n);
}

void main() {
    vec2 cur_uv = (in_cur_clip_unjittered.xy / in_cur_clip_unjittered.w) * 0.5 + 0.5;
    vec2 prev_uv = (in_prev_clip_unjittered.xy / in_prev_clip_unjittered.w) * 0.5 + 0.5;

    vec3 albedo = in_albedo_metallic.rgb;
    float metallic = in_albedo_metallic.a;
    float roughness = in_roughness;
    vec3 normal = normalize(in_world_normal);

    uvec2 mt_addr = uvec2(pc.material_textures);
    if (mt_addr.x != 0u || mt_addr.y != 0u) {
        GpuMaterialTextures mt = pc.material_textures.mats[in_material_id];
        uint s_idx = nonuniformEXT(pc.default_sampler_idx);

        if (mt.albedo_tex != 0xFFFFFFFFu) {
            vec4 tex_col = texture(
                sampler2D(bindless_textures[nonuniformEXT(mt.albedo_tex)], bindless_samplers[s_idx]),
                in_uv
            );
            if (mt.alpha_mode == 1u && tex_col.a < mt.alpha_cutoff) {
                discard;
            }
            albedo *= tex_col.rgb;
        }
        if (mt.metallic_roughness_tex != 0xFFFFFFFFu) {
            vec4 mr = texture(
                sampler2D(bindless_textures[nonuniformEXT(mt.metallic_roughness_tex)], bindless_samplers[s_idx]),
                in_uv
            );
            roughness = clamp(roughness * mr.g, 0.04, 1.0);
            metallic = clamp(metallic * mr.b, 0.0, 1.0);
        }
        if (mt.normal_tex != 0xFFFFFFFFu) {
            vec3 map_n = texture(
                sampler2D(bindless_textures[nonuniformEXT(mt.normal_tex)], bindless_samplers[s_idx]),
                in_uv
            ).xyz * 2.0 - 1.0;
            normal = perturb_normal(normal, in_world_pos, in_uv, map_n);
        }
    }

    out_albedo_metallic = vec4(albedo, metallic);
    out_normal_roughness = vec4(normal, roughness);
    out_motion = prev_uv - cur_uv;
    out_view_z = in_view_z;
}
