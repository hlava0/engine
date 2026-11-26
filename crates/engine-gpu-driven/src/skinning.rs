//! GPU compute skinning pass transforming rest-pose `GpuVertex` + bone matrices into
//! dynamic `GpuVertex` buffers ready for `gbuffer.vert` and BLAS refit.

use bytemuck::{Pod, Zeroable};
use engine_rhi::ash::vk;
use engine_rhi::{Result, RhiDevice, load_shader_words, sync};

/// Per-vertex 4-bone skinning indices and weights (`32` bytes, `scalar` aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuSkinWeight {
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SkinningPushConstants {
    pub rest_verts_ptr: u64,
    pub weights_ptr: u64,
    pub bones_ptr: u64,
    pub out_verts_ptr: u64,
    pub vertex_count: u32,
    pub _pad: u32,
}

pub struct ComputeSkinningPass {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ComputeSkinningPass {
    pub fn new(rhi: &RhiDevice) -> Result<Self> {
        let dev = &rhi.device;
        let pc_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<SkinningPushConstants>() as u32,
        }];
        let pipeline_layout = unsafe {
            dev.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_range),
                None,
            )?
        };
        let spv = load_shader_words("skinning.comp")?;
        let module = unsafe {
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)?
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipe_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            let res = dev.create_compute_pipelines(rhi.pipeline_cache, &pipe_info, None);
            dev.destroy_shader_module(module, None);
            res.map_err(|(_, e)| e)?[0]
        };
        Ok(Self {
            pipeline_layout,
            pipeline,
        })
    }

    pub fn dispatch(&self, rhi: &RhiDevice, cmd: vk::CommandBuffer, push: &SkinningPushConstants) {
        if push.vertex_count == 0 {
            return;
        }
        let dev = &rhi.device;
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push),
            );
            dev.cmd_dispatch(cmd, push.vertex_count.div_ceil(64), 1, 1);

            let post = [sync::memory_barrier(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::VERTEX_SHADER
                    | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            )];
            sync::cmd_pipeline_barrier2(dev, cmd, &post, &[], &[]);
        }
    }

    pub fn destroy(self, rhi: &RhiDevice) {
        unsafe {
            rhi.device.destroy_pipeline(self.pipeline, None);
            rhi.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
