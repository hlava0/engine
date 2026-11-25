//! GPU-rendered bitmap-font HUD, dynamic bow crosshair, and Graphics Settings Menu overlay.

use bytemuck::{Pod, Zeroable};
use engine_rhi::ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{FRAMES_IN_FLIGHT, GpuBuffer, GpuImage, RhiDevice, load_shader_words, sync};

use crate::EngineError;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudDataGpu {
    width: u32,
    height: u32,
    menu_open: u32,
    selected_row: u32,
    bow_charge: f32,
    hit_flash: f32,
    row_count: u32,
    enabled: u32,
    hud_lines: [[u32; 16]; 2],
    menu_lines: [[u32; 16]; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudPushConstants {
    buf_addr: u64,
}

/// High-level CPU state for the in-game HUD and Graphics Settings Menu.
#[derive(Debug, Clone, Default)]
pub struct HudOverlayState {
    pub enabled: bool,
    pub menu_open: bool,
    pub selected_row: u32,
    pub bow_charge: f32,
    pub hit_flash: f32,
    pub hud_lines: [String; 2],
    pub menu_lines: Vec<String>,
}

fn pack_ascii_line(text: &str) -> [u32; 16] {
    let mut words = [0u32; 16];
    for (i, b) in text.bytes().take(64).enumerate() {
        words[i / 4] |= (b as u32) << ((i % 4) * 8);
    }
    words
}

/// Compute pass that draws the HUD, crosshair, and Graphics Menu directly onto `output_image`.
pub struct HudOverlayPass {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    bound_views: [vk::ImageView; FRAMES_IN_FLIGHT],
    data_buffers: [GpuBuffer; FRAMES_IN_FLIGHT],
}

impl HudOverlayPass {
    pub fn new(rhi: &mut RhiDevice) -> Result<Self, EngineError> {
        let dev = rhi.device.clone();

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<HudPushConstants>() as u32,
        }];
        let layouts = [set_layout];
        let pipeline_layout = unsafe {
            dev.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };

        let spv = load_shader_words("hud_overlay.comp")?;
        let module = unsafe {
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)?
        };
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let pipe_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline = unsafe {
            let res = dev.create_compute_pipelines(rhi.pipeline_cache, &pipe_info, None);
            dev.destroy_shader_module(module, None);
            res.map_err(|(_, e)| e)?[0]
        };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: FRAMES_IN_FLIGHT as u32,
        }];
        let descriptor_pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let set_layouts = [set_layout; FRAMES_IN_FLIGHT];
        let sets_vec = unsafe {
            dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )?
        };
        let descriptor_sets = [sets_vec[0], sets_vec[1]];

        let buf_size = std::mem::size_of::<HudDataGpu>() as vk::DeviceSize;
        let data_buffers = [
            GpuBuffer::new(
                &dev,
                rhi.allocator_mut(),
                "hud_data_0",
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )?,
            GpuBuffer::new(
                &dev,
                rhi.allocator_mut(),
                "hud_data_1",
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )?,
        ];

        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_sets,
            bound_views: [vk::ImageView::null(); FRAMES_IN_FLIGHT],
            data_buffers,
        })
    }

    pub fn dispatch(
        &mut self,
        rhi: &RhiDevice,
        cmd: vk::CommandBuffer,
        target: &GpuImage,
        slot_idx: usize,
        state: &HudOverlayState,
    ) {
        if !state.enabled {
            return;
        }
        let width = target.extent.width;
        let height = target.extent.height;
        let slot = slot_idx % FRAMES_IN_FLIGHT;
        let dev = &rhi.device;

        if self.bound_views[slot] != target.view {
            let img_info = [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: target.view,
                image_layout: vk::ImageLayout::GENERAL,
            }];
            let write = [vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[slot])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&img_info)];
            unsafe {
                dev.update_descriptor_sets(&write, &[]);
            }
            self.bound_views[slot] = target.view;
        }

        let mut gpu_data = HudDataGpu {
            width,
            height,
            menu_open: u32::from(state.menu_open),
            selected_row: state.selected_row,
            bow_charge: state.bow_charge.clamp(0.0, 1.0),
            hit_flash: state.hit_flash.clamp(0.0, 1.0),
            row_count: state.menu_lines.len().min(8) as u32,
            enabled: 1,
            hud_lines: [
                pack_ascii_line(&state.hud_lines[0]),
                pack_ascii_line(&state.hud_lines[1]),
            ],
            menu_lines: [[0u32; 16]; 8],
        };
        for (i, line) in state.menu_lines.iter().take(8).enumerate() {
            gpu_data.menu_lines[i] = pack_ascii_line(line);
        }

        if let Ok(slice) = self.data_buffers[slot].mapped_slice_mut() {
            slice[..std::mem::size_of::<HudDataGpu>()]
                .copy_from_slice(bytemuck::bytes_of(&gpu_data));
        }

        let pre_barrier = [sync::image_barrier(
            target.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            target.subresource_range(),
        )];
        let push = HudPushConstants {
            buf_addr: self.data_buffers[slot].device_address,
        };

        unsafe {
            sync::cmd_pipeline_barrier2(dev, cmd, &[], &[], &pre_barrier);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[slot]],
                &[],
            );
            dev.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            dev.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
        }
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<(), EngineError> {
        let dev = rhi.device.clone();
        for buf in self.data_buffers {
            buf.destroy(&dev, rhi.allocator_mut())?;
        }
        unsafe {
            dev.destroy_descriptor_pool(self.descriptor_pool, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
        }
        Ok(())
    }
}
