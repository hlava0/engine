use engine_compute_ml::CoopMlPipeline;
use engine_rhi::ash::vk;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{GpuBuffer, GpuImage, RhiConfig, RhiDevice, sync};

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 127 - 14;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    f32::from_bits((sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13))
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

fn run_coop_ml_test(force_fallback: bool) -> (bool, Vec<[f32; 4]>) {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: "coop-ml-test".to_string(),
        enable_validation: true,
        enable_sync_validation: true,
        ..Default::default()
    })
    .expect("RhiDevice init");

    let width = 64u32;
    let height = 64u32;

    let mut pipeline = CoopMlPipeline::new(&mut rhi, force_fallback).expect("CoopMlPipeline::new");
    let uses_coop = pipeline.uses_cooperative_matrix();

    let dev = rhi.device.clone();
    let input_hdr = GpuImage::create_2d(
        &dev,
        rhi.allocator_mut(),
        width,
        height,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        "test_in_hdr",
    )
    .expect("create input_hdr");

    let output_ldr = GpuImage::create_2d(
        &dev,
        rhi.allocator_mut(),
        width,
        height,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "test_out_ldr",
    )
    .expect("create output_ldr");

    let pixel_count = (width * height) as usize;
    let byte_size = (pixel_count * 8) as vk::DeviceSize;

    let mut staging_in = GpuBuffer::new(
        &dev,
        rhi.allocator_mut(),
        "staging_in",
        byte_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )
    .expect("staging_in");

    let mut in_pixels = vec![0u16; pixel_count * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let r = 0.2 + 0.8 * (x as f32 / width as f32);
            let g = 0.1 + 0.6 * (y as f32 / height as f32);
            let b = 0.3;
            in_pixels[idx] = f32_to_f16(r);
            in_pixels[idx + 1] = f32_to_f16(g);
            in_pixels[idx + 2] = f32_to_f16(b);
            in_pixels[idx + 3] = f32_to_f16(1.0);
        }
    }
    staging_in
        .mapped_slice_mut()
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&in_pixels));

    let staging_out = GpuBuffer::new(
        &dev,
        rhi.allocator_mut(),
        "staging_out",
        byte_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
    )
    .expect("staging_out");

    let pool = rhi
        .create_command_pool(
            rhi.queue_families.compute,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )
        .expect("create_command_pool");
    let cmd = rhi.allocate_command_buffer(pool).expect("alloc cmd");

    unsafe {
        rhi.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }

    let to_dst = [sync::image_barrier(
        input_hdr.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::PipelineStageFlags2::TOP_OF_PIPE,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        input_hdr.subresource_range(),
    )];
    unsafe { sync::cmd_pipeline_barrier2(&rhi.device, cmd, &[], &[], &to_dst) };

    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    unsafe {
        rhi.device.cmd_copy_buffer_to_image(
            cmd,
            staging_in.buffer,
            input_hdr.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    let to_gen = [sync::image_barrier(
        input_hdr.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_WRITE,
        input_hdr.subresource_range(),
    )];
    unsafe { sync::cmd_pipeline_barrier2(&rhi.device, cmd, &[], &[], &to_gen) };

    pipeline.dispatch(&rhi.device, cmd, &input_hdr, &output_ldr, width, height);

    let to_src = [sync::image_barrier(
        output_ldr.image,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_READ,
        output_ldr.subresource_range(),
    )];
    unsafe { sync::cmd_pipeline_barrier2(&rhi.device, cmd, &[], &[], &to_src) };

    unsafe {
        rhi.device.cmd_copy_image_to_buffer(
            cmd,
            output_ldr.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_out.buffer,
            &[region],
        );
        rhi.device.end_command_buffer(cmd).unwrap();
    }

    rhi.submit_and_wait(rhi.compute_queue, cmd)
        .expect("submit_and_wait");

    let out_bytes = staging_out
        .allocation
        .as_ref()
        .unwrap()
        .mapped_slice()
        .unwrap();
    let out_u16: &[u16] = bytemuck::cast_slice(out_bytes);
    let mut out_rgba = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        out_rgba.push([
            f16_to_f32(out_u16[i * 4]),
            f16_to_f32(out_u16[i * 4 + 1]),
            f16_to_f32(out_u16[i * 4 + 2]),
            f16_to_f32(out_u16[i * 4 + 3]),
        ]);
    }

    assert_eq!(
        rhi.validation_errors(),
        0,
        "Expected 0 Vulkan validation errors"
    );

    unsafe { dev.destroy_command_pool(pool, None) };
    staging_in.destroy(&dev, rhi.allocator_mut()).unwrap();
    staging_out.destroy(&dev, rhi.allocator_mut()).unwrap();
    input_hdr.destroy(&dev, rhi.allocator_mut()).unwrap();
    output_ldr.destroy(&dev, rhi.allocator_mut()).unwrap();
    pipeline.destroy(&mut rhi).unwrap();

    (uses_coop, out_rgba)
}

#[test]
fn coop_ml_hardware_and_fallback_paths() {
    let (uses_coop, coop_pixels) = run_coop_ml_test(false);
    println!("CoopML hardware path: uses_cooperative_matrix={uses_coop}");
    assert!(
        uses_coop,
        "Expected VK_KHR_cooperative_matrix (8x16x16 FLOAT16) to be active on Arc B580"
    );

    let (uses_coop_fb, fb_pixels) = run_coop_ml_test(true);
    println!("CoopML fallback path: uses_cooperative_matrix={uses_coop_fb}");
    assert!(
        !uses_coop_fb,
        "Expected fallback path when force_fallback=true"
    );

    let mut non_zero = 0usize;
    let mut max_diff = 0.0f32;
    for (a, b) in coop_pixels.iter().zip(fb_pixels.iter()) {
        for c in 0..4 {
            assert!(a[c].is_finite(), "coop pixel not finite: {a:?}");
            assert!(b[c].is_finite(), "fallback pixel not finite: {b:?}");
        }
        if a[0] > 0.01 || a[1] > 0.01 || a[2] > 0.01 {
            non_zero += 1;
        }
        for c in 0..3 {
            max_diff = max_diff.max((a[c] - b[c]).abs());
        }
    }

    println!(
        "CoopML non-zero pixels: {non_zero}/{}, max_diff(coop vs fallback)={max_diff:.5}",
        coop_pixels.len()
    );
    assert_eq!(non_zero, coop_pixels.len());
    assert!(
        max_diff < 0.02,
        "Hardware fp16 CoopML and fp32 fallback diverged by {max_diff}"
    );
}
