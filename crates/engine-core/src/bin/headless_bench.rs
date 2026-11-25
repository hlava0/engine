use std::time::Instant;

use engine_core::{Engine, FramePacer, create_demo_scene};

fn main() {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    const WARMUP_FRAMES: u32 = 2;
    const BENCH_FRAMES: u32 = 60;

    let enable_validation =
        cfg!(debug_assertions) || std::env::args().any(|arg| arg == "--validate");

    let init_start = Instant::now();
    let world = create_demo_scene(WIDTH, HEIGHT);
    let mut engine = Engine::new(world, WIDTH, HEIGHT, enable_validation).expect("Engine::new");
    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "=== Headless GPU-Driven Vulkan 1.4 Smoothness Benchmark ({WIDTH}x{HEIGHT}, {WARMUP_FRAMES} warmup + {BENCH_FRAMES} pipelined frames, validation={enable_validation}) ==="
    );
    let vp = engine.vendor_profile();
    println!(
        "GPU: {} | Vendor: {} | Init: {:.1} ms",
        engine.device_name(),
        vp.vendor,
        init_ms
    );
    println!(
        "Driver: {} ({}) [id={:?}]",
        vp.driver_name, vp.driver_info, vp.driver_id
    );
    println!(
        "Subgroups: default={} range=[{}, {}] | req_compute={:?} req_rt={:?}",
        vp.subgroup_size,
        vp.min_subgroup_size,
        vp.max_subgroup_size,
        vp.required_compute_subgroup_size,
        vp.required_rt_subgroup_size
    );
    println!(
        "Hardware: ReBAR={} | AS_scratch_align={}B | TLAS_refit={} | NV_SER={} | EXT_desc_buf={}",
        vp.has_rebar,
        vp.min_as_scratch_alignment,
        vp.prefer_tlas_refit,
        vp.nv_ray_tracing_invocation_reorder,
        vp.ext_descriptor_buffer
    );
    println!(
        "CoopML: hardware={} | active_tile={} (preferred={})",
        engine.uses_cooperative_matrix(),
        engine.coop_ml_tile(),
        vp.preferred_coop_tile
    );

    // Warmup frames (prime Hi-Z history, NRD temporal pool, and in-flight slots)
    for _ in 0..WARMUP_FRAMES {
        engine
            .render_frame(1.0 / 60.0)
            .expect("warmup render_frame");
    }

    let mut pacer = FramePacer::new(1.0 / 60.0, 0.1, BENCH_FRAMES as usize);
    let mut sum_cpu_record_us = 0.0f64;
    let mut sum_gpu_ms = 0.0f64;
    let mut gpu_samples = 0u32;
    let mut last_draws = 0u32;
    let mut last_gpu_timings = Vec::new();

    for _ in 0..BENCH_FRAMES {
        let frame_start = Instant::now();
        let stats = engine
            .render_frame(pacer.fixed_dt() as f32)
            .expect("render_frame");
        let frame_dt_sec = frame_start.elapsed().as_secs_f64();
        let _sim_steps = pacer.begin_frame(frame_dt_sec);

        sum_cpu_record_us += stats.cpu_record_us;
        if !stats.gpu_timings.is_empty() {
            let total_gpu_ms: f64 = stats.gpu_timings.iter().map(|t| t.gpu_ms).sum();
            sum_gpu_ms += total_gpu_ms;
            gpu_samples += 1;
            last_draws = stats.draw_count;
            last_gpu_timings = stats.gpu_timings;
        }
    }

    engine.wait_idle().expect("wait_idle");

    let pacer_stats = pacer.stats();
    let avg_cpu_record_us = sum_cpu_record_us / BENCH_FRAMES as f64;
    let avg_gpu_ms = if gpu_samples > 0 {
        sum_gpu_ms / gpu_samples as f64
    } else {
        0.0
    };

    println!(
        "Smoothness ({} frames): avg_fps={:.1} | 1%_low_fps={:.1} | avg_frame_ms={:.3} | p95_frame_ms={:.3} | p99_frame_ms={:.3} | jitter_ms={:.3}",
        pacer_stats.sample_count,
        pacer_stats.avg_fps,
        pacer_stats.one_percent_low_fps,
        pacer_stats.avg_frame_ms,
        pacer_stats.p95_frame_ms,
        pacer_stats.p99_frame_ms,
        pacer_stats.jitter_ms
    );
    print!(
        "Pipeline: avg_cpu_record_us={:.1} us | avg_gpu_ms={:.3} ms | draws={} |",
        avg_cpu_record_us, avg_gpu_ms, last_draws
    );
    for t in &last_gpu_timings {
        print!(" {}={:.3}ms", t.name, t.gpu_ms);
    }
    println!();

    let pixels = engine.readback_output_rgba().expect("readback_output_rgba");
    let non_zero = pixels
        .iter()
        .filter(|p| p[0] > 0.001 || p[1] > 0.001 || p[2] > 0.001)
        .count();
    println!(
        "Output pixels: {}/{} non-zero | validation_errors={}",
        non_zero,
        pixels.len(),
        engine.validation_errors()
    );
    assert_eq!(engine.validation_errors(), 0, "Expected 0 validation errors");
    assert!(non_zero > 0, "Expected non-zero rendered output");
}
