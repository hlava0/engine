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
