use engine_core::{Engine, create_demo_scene};

#[test]
fn full_engine_pipelined_frames_headless() {
    let _ = env_logger::builder().is_test(true).try_init();

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const FRAMES: u32 = 12;

    let world = create_demo_scene(WIDTH, HEIGHT);
    let total_instances = world.instances.len() as u32;
    let mut engine = Engine::new(world, WIDTH, HEIGHT, true).expect("Engine::new");

    assert!(
        engine.uses_cooperative_matrix(),
        "Expected VK_KHR_cooperative_matrix to be active"
    );

    let mut last_draws = 0u32;
    let mut completed_gpu_frames = 0u32;
    for frame in 0..FRAMES {
        let stats = engine.render_frame(1.0 / 60.0).expect("render_frame");
        println!(
            "Frame {frame}: cpu_record={:.1} us, draws={}/{}, gpu_timings={:?}",
            stats.cpu_record_us, stats.draw_count, total_instances, stats.gpu_timings
        );
        assert!(stats.cpu_record_us > 0.0);
        if !stats.gpu_timings.is_empty() {
            completed_gpu_frames += 1;
            last_draws = stats.draw_count;
        }
    }

    engine.wait_idle().expect("wait_idle");

    assert!(
        completed_gpu_frames >= FRAMES - 2,
        "Expected at least {} completed GPU timing readbacks, got {}",
        FRAMES - 2,
        completed_gpu_frames
    );

    // Demo scene has 8 instances including 1 behind camera and 1 occluded behind the wall
    assert!(
        last_draws > 0 && last_draws < total_instances,
        "Expected culling to reduce draw count ({last_draws}/{total_instances})"
    );

    let pixels = engine.readback_output_rgba().expect("readback_output_rgba");
    let mut non_zero = 0usize;
    for p in &pixels {
        for c in 0..4 {
            assert!(p[c].is_finite(), "Non-finite output pixel: {p:?}");
        }
        if p[0] > 0.005 || p[1] > 0.005 || p[2] > 0.005 {
            non_zero += 1;
        }
    }
    println!("Non-zero output pixels: {non_zero}/{}", pixels.len());
    assert!(
        non_zero > pixels.len() / 4,
        "Expected substantial non-zero image"
    );

    assert_eq!(
        engine.validation_errors(),
        0,
        "Expected 0 Vulkan validation errors or sync hazards"
    );
}

#[test]
fn hud_menu_overlay_and_settings_toggle_headless() {
    let _ = env_logger::builder().is_test(true).try_init();
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;

    let world = create_demo_scene(WIDTH, HEIGHT);
    let mut engine = Engine::new(world, WIDTH, HEIGHT, true).expect("Engine::new");
    engine.hud_state.enabled = true;
    engine.hud_state.menu_open = true;
    engine.hud_state.selected_row = 1;
    engine.hud_state.bow_charge = 0.75;
    engine.hud_state.hit_flash = 0.5;
    engine.hud_state.hud_lines = [
        "SCORE:300  HITS:2  ARROWS:1/16".to_string(),
        "FPS:240  GPU:1.20MS  NRD  [M] MENU".to_string(),
    ];
    engine.hud_state.menu_lines = vec![
        "GRAPHICS & GAME MENU".to_string(),
        "1. RT GI + REFLECTIONS : OFF".to_string(),
        "2. HI-Z OCCLUSION CULL : ON".to_string(),
    ];
    engine.graphics_settings.enable_rt_gi = false;
    engine.graphics_settings.fov_deg = 75.0;

    for _ in 0..4 {
        engine.render_frame(1.0 / 60.0).expect("render_frame");
    }
    engine.wait_idle().expect("wait_idle");
    let pixels = engine.readback_output_rgba().expect("readback_output_rgba");
    assert!(pixels.iter().any(|p| p[0] > 0.01 && p[1] > 0.01));
    assert_eq!(engine.validation_errors(), 0);
}

#[test]
fn fsr31_temporal_upscaler_two_frames_validation() {
    use engine_core::upscaler::{FsrPreset, TemporalUpscalerPass};
    use engine_rhi::ash::vk;
    use engine_rhi::{GpuImage, RhiConfig, RhiDevice, sync};

    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: "fsr31-upscale-test".to_string(),
        enable_validation: true,
        enable_sync_validation: true,
        ..Default::default()
    })
    .expect("RhiDevice");

    let (rw, rh) = FsrPreset::Performance.render_dimensions(128, 128);
    assert_eq!((rw, rh), (64, 64));

    let dev = rhi.device.clone();
    let mut upscaler = TemporalUpscalerPass::new(&mut rhi, 128, 128).expect("TemporalUpscalerPass");
    eprintln!("Upscaler backend: {}", upscaler.backend_name());

    let in_color = GpuImage::create_2d(
        &dev,
        rhi.allocator_mut(),
        rw,
        rh,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        "fsr_in_col",
    )
    .unwrap();
    let in_motion = GpuImage::create_2d(
        &dev,
        rhi.allocator_mut(),
        rw,
        rh,
        1,
        vk::Format::R16G16_SFLOAT,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        "fsr_in_mv",
    )
    .unwrap();
    let in_depth = GpuImage::create_2d(
        &dev,
        rhi.allocator_mut(),
        rw,
        rh,
        1,
        vk::Format::R32_SFLOAT,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        "fsr_in_depth",
    )
    .unwrap();

    let pool = rhi
        .create_command_pool(rhi.queue_families.graphics, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .unwrap();
    let cmd = rhi.allocate_command_buffer(pool).unwrap();

    for f in 0..2 {
        unsafe {
            dev.reset_command_pool(pool, vk::CommandPoolResetFlags::empty()).unwrap();
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            if f == 0 {
                let init = [
                    sync::image_barrier(
                        in_color.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        in_color.subresource_range(),
                    ),
                    sync::image_barrier(
                        in_motion.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        in_motion.subresource_range(),
                    ),
                    sync::image_barrier(
                        in_depth.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                        in_depth.subresource_range(),
                    ),
                ];
                sync::cmd_pipeline_barrier2(&dev, cmd, &[], &[], &init);
            }
        }
        upscaler.record(
            &rhi,
            cmd,
            &in_color,
            &in_motion,
            &in_depth,
            [0.0, 0.0],
            0.5,
            f,
        );
        unsafe { dev.end_command_buffer(cmd).unwrap() };
        rhi.submit_and_wait(rhi.graphics_queue, cmd).unwrap();
    }

    unsafe {
        dev.device_wait_idle().unwrap();
        dev.destroy_command_pool(pool, None);
    }
    upscaler.destroy(&mut rhi).unwrap();
    in_color.destroy(&dev, rhi.allocator_mut()).unwrap();
    in_motion.destroy(&dev, rhi.allocator_mut()).unwrap();
    in_depth.destroy(&dev, rhi.allocator_mut()).unwrap();
    assert_eq!(rhi.validation_errors(), 0);
}
