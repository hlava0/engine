use ash::vk;
use engine_gpu_driven::types::{
    extract_frustum_planes_reverse_z, halton, mat4_look_at_rh, mat4_mul,
    mat4_perspective_reverse_z, mat4_translation_scale,
};
use engine_gpu_driven::{
    FrameUniforms, GpuDrivenPipeline, GpuInstanceData, GpuMeshDrawInfo, GpuScene, GpuVertex,
};
use engine_rhi::{BindlessConfig, BindlessTable, RhiConfig, RhiDevice};

fn unit_cube_mesh() -> (Vec<GpuVertex>, Vec<u32>, GpuMeshDrawInfo) {
    // 24 vertices (4 per face, outward normals, CCW winding)
    #[rustfmt::skip]
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z (front)
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        // -Z (back)
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        // +X (right)
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        // -X (left)
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        // +Y (top)
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        // -Y (bottom)
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
    ];

    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for (i, pos) in corners.into_iter().enumerate() {
            vertices.push(GpuVertex {
                pos,
                normal,
                uv: uvs[i],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mesh = GpuMeshDrawInfo {
        bounding_sphere: [0.0, 0.0, 0.0, (0.5f32 * 0.5 * 3.0).sqrt()],
        index_count: indices.len() as u32,
        first_index: 0,
        vertex_offset: 0,
        _pad: 0,
    };

    (vertices, indices, mesh)
}

#[test]
fn headless_gpu_driven_frustum_and_hiz_cull() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: "gpu-driven-pass-test".to_string(),
        enable_validation: true,
        enable_sync_validation: true,
    })
    .expect("Failed to initialize RhiDevice");

    assert!(rhi.capabilities.draw_indirect_count);
    assert!(rhi.capabilities.buffer_device_address);
    assert!(rhi.capabilities.dynamic_rendering);
    assert!(rhi.capabilities.synchronization2);

    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 256;

    let bindless = BindlessTable::new(&rhi.device, BindlessConfig::default())
        .expect("Failed to create BindlessTable");
    let mut pipeline = GpuDrivenPipeline::new(&mut rhi, &bindless, WIDTH, HEIGHT)
        .expect("Failed to create GpuDrivenPipeline");

    let (vertices, indices, mesh) = unit_cube_mesh();

    // 4 instances:
    // 0: Large occluder wall in front of camera
    // 1: Small box directly behind wall (visible to frustum, occluded by Hi-Z)
    // 2: Visible box beside the wall
    // 3: Box behind camera (culled by frustum)
    let m0 = mat4_translation_scale([0.0, 0.0, 1.0], [2.0, 2.0, 0.2]);
    let m1 = mat4_translation_scale([0.0, 0.0, -2.0], [0.5, 0.5, 0.5]);
    let m2 = mat4_translation_scale([2.0, 0.0, 0.0], [0.5, 0.5, 0.5]);
    let m3 = mat4_translation_scale([0.0, 0.0, 10.0], [1.0, 1.0, 1.0]);

    let instances = [
        GpuInstanceData {
            model: m0,
            prev_model: m0,
            albedo_metallic: [0.8, 0.2, 0.2, 0.0],
            mesh_id: 0,
            material_id: 0,
            roughness: 0.5,
            _pad: 0,
        },
        GpuInstanceData {
            model: m1,
            prev_model: m1,
            albedo_metallic: [0.2, 0.8, 0.2, 0.0],
            mesh_id: 0,
            material_id: 1,
            roughness: 0.4,
            _pad: 0,
        },
        GpuInstanceData {
            model: m2,
            prev_model: m2,
            albedo_metallic: [0.2, 0.2, 0.8, 0.5],
            mesh_id: 0,
            material_id: 2,
            roughness: 0.3,
            _pad: 0,
        },
        GpuInstanceData {
            model: m3,
            prev_model: m3,
            albedo_metallic: [0.8, 0.8, 0.2, 0.0],
            mesh_id: 0,
            material_id: 3,
            roughness: 0.6,
            _pad: 0,
        },
    ];

    let mut scene = GpuScene::new(&mut rhi, &vertices, &indices, &[mesh], &instances)
        .expect("Failed to create GpuScene");

    let view = mat4_look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let proj = mat4_perspective_reverse_z(60.0f32.to_radians(), 1.0, 0.1, 100.0);
    let view_proj = mat4_mul(proj, view);
    let frustum_planes = extract_frustum_planes_reverse_z(view_proj);

    let pool = rhi
        .create_command_pool(
            rhi.queue_families.graphics,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .expect("create_command_pool");
    let cmd = rhi
        .allocate_command_buffer(pool)
        .expect("allocate_command_buffer");

    // Helper to run one frame and read back draw_count
    let mut run_frame = |frame_idx: u32, enable_hiz: bool| -> u32 {
        let jx = (halton(frame_idx + 1, 2) - 0.5) * (2.0 / WIDTH as f32);
        let jy = (halton(frame_idx + 1, 3) - 0.5) * (2.0 / HEIGHT as f32);
        let uniforms = FrameUniforms {
            view,
            proj,
            view_proj,
            prev_view_proj: view_proj,
            frustum_planes,
            jitter: [jx, jy, 0.0, 0.0],
            screen_size: [WIDTH, HEIGHT],
            hiz_size: [WIDTH, HEIGHT],
        };

        unsafe {
            rhi.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer");
            rhi.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin_command_buffer");
        }

        pipeline
            .record_frame(&rhi, &bindless, cmd, &mut scene, &uniforms, enable_hiz)
            .expect("record_frame");

        unsafe {
            rhi.device
                .end_command_buffer(cmd)
                .expect("end_command_buffer");
        }
        rhi.submit_and_wait(rhi.graphics_queue, cmd)
            .expect("submit_and_wait");

        scene.read_draw_count().expect("read_draw_count")
    };

    // Frame 0: frustum cull only -> instances 0, 1, 2 pass (instance 3 culled)
    let count0 = run_frame(0, false);
    println!("Frame 0 (frustum only) draw_count = {count0}");
    assert_eq!(count0, 3, "Expected 3 instances to pass frustum cull on Frame 0");

    // Frame 1: frustum + Hi-Z cull -> instances 0, 2 pass (instance 1 occluded, instance 3 frustum-culled)
    let count1 = run_frame(1, true);
    println!("Frame 1 (frustum + Hi-Z) draw_count = {count1}");
    assert_eq!(
        count1, 2,
        "Expected 2 instances to pass frustum + Hi-Z cull on Frame 1"
    );

    rhi.wait_idle().expect("wait_idle");
    unsafe {
        rhi.device.destroy_command_pool(pool, None);
    }
    scene.destroy(&mut rhi).expect("scene destroy");
    pipeline.destroy(&mut rhi).expect("pipeline destroy");
    bindless.destroy(&rhi.device);

    assert_eq!(
        rhi.validation_errors(),
        0,
        "Vulkan validation errors detected!"
    );
    assert_eq!(
        rhi.validation_warnings(),
        0,
        "Vulkan validation warnings detected!"
    );
}
