use engine_core::{Engine, create_demo_scene};

fn main() {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;
    const FRAMES: u32 = 5;

    let world = create_demo_scene(WIDTH, HEIGHT);
    let mut engine = Engine::new(world, WIDTH, HEIGHT, true).expect("Engine::new");

    println!(
        "=== Headless GPU-Driven Vulkan 1.4 Benchmark ({WIDTH}x{HEIGHT}, {FRAMES} frames) ==="
    );
    let vp = engine.vendor_profile();
    println!("GPU: {} | Vendor: {}", engine.device_name(), vp.vendor);
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

    for _ in 0..FRAMES {
        let stats = engine.render_frame(1.0 / 60.0).expect("render_frame");
        let total_gpu_ms: f64 = stats.gpu_timings.iter().map(|t| t.gpu_ms).sum();
        print!(
            "Frame {}: cpu_record={:.1} us | draws={} | total_gpu={:.3} ms |",
            stats.frame_index, stats.cpu_record_us, stats.draw_count, total_gpu_ms
        );
        for t in &stats.gpu_timings {
            print!(" {}={:.3}ms", t.name, t.gpu_ms);
        }
        println!();
    }

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
