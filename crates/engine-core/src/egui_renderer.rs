//! Vulkan 1.3/1.4 Dynamic Rendering backend for `egui` visual editor & inspector UI.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use egui::{ClippedPrimitive, TexturesDelta};
use engine_rhi::ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{FRAMES_IN_FLIGHT, GpuBuffer, GpuImage, RhiDevice, load_shader_words, sync};

use crate::EngineError;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EguiPushConstants {
    screen_size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EguiVertexGpu {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [u8; 4],
}

struct EguiTexture {
    image: GpuImage,
    set: vk::DescriptorSet,
}

/// Renders `egui` primitives onto a swapchain image using Vulkan Dynamic Rendering.
pub struct EguiVulkanRenderer {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    textures: HashMap<egui::TextureId, EguiTexture>,
    vertex_buffers: [Option<GpuBuffer>; FRAMES_IN_FLIGHT],
    index_buffers: [Option<GpuBuffer>; FRAMES_IN_FLIGHT],
}

impl EguiVulkanRenderer {
    pub fn new(rhi: &RhiDevice, swapchain_format: vk::Format) -> Result<Self, EngineError> {
        let dev = &rhi.device;

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<EguiPushConstants>() as u32,
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

        let vert_spv = load_shader_words("egui.vert")?;
        let frag_spv = load_shader_words("egui.frag")?;
        let vert_mod = unsafe {
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)?
        };
        let frag_mod = unsafe {
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)?
        };

        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_mod)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_mod)
                .name(entry),
        ];

        let binding_desc = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<EguiVertexGpu>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attr_descs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_desc)
            .vertex_attribute_descriptions(&attr_descs);
        let input_asm = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_att = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

        let color_formats = [swapchain_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        let pipe_info = [vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering_info)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_asm)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(pipeline_layout)];

        let pipeline = unsafe {
            let res = dev.create_graphics_pipelines(rhi.pipeline_cache, &pipe_info, None);
            dev.destroy_shader_module(vert_mod, None);
            dev.destroy_shader_module(frag_mod, None);
            res.map_err(|(_, e)| e)?[0]
        };

        let sampler = unsafe {
            dev.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 32,
        }];
        let descriptor_pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(32)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            descriptor_pool,
            textures: HashMap::new(),
            vertex_buffers: [None, None],
            index_buffers: [None, None],
        })
    }

    /// Upload `egui` texture deltas using an immediate staging transfer.
    pub fn update_textures(
        &mut self,
        rhi: &mut RhiDevice,
        cmd_pool: vk::CommandPool,
        deltas: &TexturesDelta,
    ) -> Result<(), EngineError> {
        if deltas.set.is_empty() && deltas.free.is_empty() {
            return Ok(());
        }
        let dev = rhi.device.clone();

        for (id, delta) in &deltas.set {
            let (w, h, rgba): (u32, u32, Vec<u8>) = match &delta.image {
                egui::ImageData::Color(c) => (
                    c.size[0] as u32,
                    c.size[1] as u32,
                    c.pixels.iter().flat_map(|p| p.to_array()).collect(),
                ),
                egui::ImageData::Font(f) => (
                    f.size[0] as u32,
                    f.size[1] as u32,
                    f.srgba_pixels(None).flat_map(|p| p.to_array()).collect(),
                ),
            };
            if w == 0 || h == 0 {
                continue;
            }
            if delta.pos.is_some() {
                continue;
            }
            if let Some(old) = self.textures.remove(id) {
                old.image.destroy(&dev, rhi.allocator_mut())?;
            }

            let img = GpuImage::create_2d(
                &dev,
                rhi.allocator_mut(),
                w,
                h,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
                "egui_tex",
            )?;
            let staging = GpuBuffer::new(
                &dev,
                rhi.allocator_mut(),
                "egui_tex_staging",
                rgba.len() as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?;
            let mut staging = staging;
            staging.mapped_slice_mut()?[..rgba.len()].copy_from_slice(&rgba);

            let cmd = rhi.allocate_command_buffer(cmd_pool)?;
            unsafe {
                dev.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let to_dst = [sync::image_barrier(
                    img.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    img.subresource_range(),
                )];
                sync::cmd_pipeline_barrier2(&dev, cmd, &[], &[], &to_dst);

                let region = [vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                }];
                dev.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );

                let to_read = [sync::image_barrier(
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    img.subresource_range(),
                )];
                sync::cmd_pipeline_barrier2(&dev, cmd, &[], &[], &to_read);
                dev.end_command_buffer(cmd)?;
            }
            rhi.submit_and_wait(rhi.graphics_queue, cmd)?;
            staging.destroy(&dev, rhi.allocator_mut())?;

            let set = unsafe {
                let sets = dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&[self.set_layout]),
                )?;
                let info = [vk::DescriptorImageInfo {
                    sampler: self.sampler,
                    image_view: img.view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];
                dev.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(sets[0])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info)],
                    &[],
                );
                sets[0]
            };

            self.textures.insert(*id, EguiTexture { image: img, set });
        }

        for id in &deltas.free {
            if let Some(old) = self.textures.remove(id) {
                old.image.destroy(&dev, rhi.allocator_mut())?;
            }
        }
        Ok(())
    }

    /// Record Dynamic Rendering commands for `primitives` onto `target_view` (already in `COLOR_ATTACHMENT_OPTIMAL`).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi: &mut RhiDevice,
        cmd: vk::CommandBuffer,
        target_view: vk::ImageView,
        extent: vk::Extent2D,
        pixels_per_point: f32,
        primitives: &[ClippedPrimitive],
        slot_idx: usize,
    ) -> Result<(), EngineError> {
        let mut vertices = Vec::<EguiVertexGpu>::new();
        let mut indices = Vec::<u32>::new();
        for prim in primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
                for v in &mesh.vertices {
                    vertices.push(EguiVertexGpu {
                        pos: [v.pos.x, v.pos.y],
                        uv: [v.uv.x, v.uv.y],
                        color: v.color.to_array(),
                    });
                }
                indices.extend_from_slice(&mesh.indices);
            }
        }
        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let slot = slot_idx % FRAMES_IN_FLIGHT;
        let v_bytes = bytemuck::cast_slice::<EguiVertexGpu, u8>(&vertices);
        let i_bytes = bytemuck::cast_slice::<u32, u8>(&indices);
        let dev = rhi.device.clone();

        let need_v_realloc = self.vertex_buffers[slot]
            .as_ref()
            .is_none_or(|b| (b.size as usize) < v_bytes.len());
        if need_v_realloc {
            if let Some(old) = self.vertex_buffers[slot].take() {
                old.destroy(&dev, rhi.allocator_mut())?;
            }
            self.vertex_buffers[slot] = Some(GpuBuffer::new(
                &dev,
                rhi.allocator_mut(),
                "egui_vb",
                (v_bytes.len().max(4096) * 2) as vk::DeviceSize,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let need_i_realloc = self.index_buffers[slot]
            .as_ref()
            .is_none_or(|b| (b.size as usize) < i_bytes.len());
        if need_i_realloc {
            if let Some(old) = self.index_buffers[slot].take() {
                old.destroy(&dev, rhi.allocator_mut())?;
            }
            self.index_buffers[slot] = Some(GpuBuffer::new(
                &dev,
                rhi.allocator_mut(),
                "egui_ib",
                (i_bytes.len().max(4096) * 2) as vk::DeviceSize,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }

        let vb = self.vertex_buffers[slot].as_mut().expect("vb");
        vb.mapped_slice_mut()?[..v_bytes.len()].copy_from_slice(v_bytes);
        let ib = self.index_buffers[slot].as_mut().expect("ib");
        ib.mapped_slice_mut()?[..i_bytes.len()].copy_from_slice(i_bytes);

        let color_att = [vk::RenderingAttachmentInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_att);

        let dev = &rhi.device;
        let screen_pts = [
            extent.width as f32 / pixels_per_point,
            extent.height as f32 / pixels_per_point,
        ];
        let push = EguiPushConstants {
            screen_size: screen_pts,
        };

        unsafe {
            dev.cmd_begin_rendering(cmd, &rendering_info);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            dev.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            dev.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
            dev.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);
            dev.cmd_bind_index_buffer(cmd, ib.buffer, 0, vk::IndexType::UINT32);

            let mut v_off = 0i32;
            let mut i_off = 0u32;
            for prim in primitives {
                if let egui::epaint::Primitive::Mesh(mesh) = &prim.primitive {
                    if let Some(tex) = self.textures.get(&mesh.texture_id) {
                        let min_x = (prim.clip_rect.min.x * pixels_per_point).max(0.0) as i32;
                        let min_y = (prim.clip_rect.min.y * pixels_per_point).max(0.0) as i32;
                        let max_x = ((prim.clip_rect.max.x * pixels_per_point) as i32)
                            .min(extent.width as i32);
                        let max_y = ((prim.clip_rect.max.y * pixels_per_point) as i32)
                            .min(extent.height as i32);
                        if max_x > min_x && max_y > min_y {
                            dev.cmd_set_scissor(
                                cmd,
                                0,
                                &[vk::Rect2D {
                                    offset: vk::Offset2D { x: min_x, y: min_y },
                                    extent: vk::Extent2D {
                                        width: (max_x - min_x) as u32,
                                        height: (max_y - min_y) as u32,
                                    },
                                }],
                            );
                            dev.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[tex.set],
                                &[],
                            );
                            dev.cmd_draw_indexed(
                                cmd,
                                mesh.indices.len() as u32,
                                1,
                                i_off,
                                v_off,
                                0,
                            );
                        }
                    }
                    v_off += mesh.vertices.len() as i32;
                    i_off += mesh.indices.len() as u32;
                }
            }
            dev.cmd_end_rendering(cmd);
        }
        Ok(())
    }

    pub fn destroy(mut self, rhi: &mut RhiDevice) -> Result<(), EngineError> {
        let dev = rhi.device.clone();
        for (_, tex) in self.textures.drain() {
            tex.image.destroy(&dev, rhi.allocator_mut())?;
        }
        for b in self.vertex_buffers.into_iter().flatten() {
            b.destroy(&dev, rhi.allocator_mut())?;
        }
        for b in self.index_buffers.into_iter().flatten() {
            b.destroy(&dev, rhi.allocator_mut())?;
        }
        unsafe {
            dev.destroy_descriptor_pool(self.descriptor_pool, None);
            dev.destroy_sampler(self.sampler, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
        }
        Ok(())
    }
}
