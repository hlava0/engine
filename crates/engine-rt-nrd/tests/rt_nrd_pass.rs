use ash::vk;
use engine_gpu_driven::types::{
    extract_frustum_planes_reverse_z, halton, mat4_look_at_rh, mat4_mul,
    mat4_perspective_reverse_z, mat4_translation_scale,
};
use engine_gpu_driven::{
    FrameUniforms, GpuDrivenPipeline, GpuInstanceData, GpuMeshDrawInfo, GpuScene, GpuVertex,
};
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{BindlessConfig, BindlessTable, GpuBuffer, RhiConfig, RhiDevice, sync};
use engine_rt_nrd::{
    AccelStructureManager, DenoiserMode, GpuRtLight, GpuRtMaterial, NrdComposePass, NrdDenoiser,
    NrdInputs, RtGiPass, RtParams,
};

fn unit_cube_mesh() -> (Vec<GpuVertex>, Vec<u32>, GpuMeshDrawInfo) {
    #[rustfmt::skip]
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
        ([1.0, 0.0, 0.0], [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]),
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
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

fn mat4_inverse(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let a: [f32; 16] = [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ];
    let mut inv = [0.0f32; 16];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
        + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
        - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
        + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
        - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
        - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
        + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
        - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
        + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
        + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
        - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
        + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
        - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
        - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
        + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
        - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
        + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    let inv_det = if det.abs() > 1e-12 { 1.0 / det } else { 0.0 };
    for v in &mut inv {
        *v *= inv_det;
    }

    [
        [inv[0], inv[1], inv[2], inv[3]],
        [inv[4], inv[5], inv[6], inv[7]],
        [inv[8], inv[9], inv[10], inv[11]],
        [inv[12], inv[13], inv[14], inv[15]],
    ]
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits & 0x7C00) >> 10;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            f32::from_bits(sign | ((e as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | (((exp as u32) + (127 - 15)) << 23) | (mant << 13))
    }
}

fn run_rt_denoiser_test(mode: DenoiserMode) {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: format!("rt-nrd-pass-test-{mode:?}"),
        enable_validation: true,
        enable_sync_validation: true,
        ..Default::default()
    })
    .expect("Failed to initialize RhiDevice");

    if !rhi.capabilities.khr_acceleration_structure || !rhi.capabilities.khr_ray_query {
        println!("Skipping RT+NRD test: device does not support KHR_acceleration_structure / KHR_ray_query");
        return;
    }

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const DENOISING_RANGE: f32 = 1000.0;

    let bindless = BindlessTable::new(&rhi.device, BindlessConfig::default())
        .expect("Failed to create BindlessTable");
    let mut gbuffer_pipe = GpuDrivenPipeline::new(&mut rhi, &bindless, WIDTH, HEIGHT)
        .expect("Failed to create GpuDrivenPipeline");

    let (vertices, indices, mesh) = unit_cube_mesh();
    let m_floor = mat4_translation_scale([0.0, -1.0, 0.0], [8.0, 0.2, 8.0]);
    let m_cube = mat4_translation_scale([0.0, 0.0, 0.0], [1.2, 1.2, 1.2]);
    let m_metal = mat4_translation_scale([1.8, -0.2, -0.6], [0.8, 1.4, 0.8]);

    let instances = [
        GpuInstanceData {
            model: m_floor,
            prev_model: m_floor,
            albedo_metallic: [0.7, 0.7, 0.7, 0.0],
            mesh_id: 0,
            material_id: 0,
            roughness: 0.7,
            _pad: 0,
        },
        GpuInstanceData {
            model: m_cube,
            prev_model: m_cube,
            albedo_metallic: [0.8, 0.25, 0.2, 0.0],
            mesh_id: 0,
            material_id: 1,
            roughness: 0.5,
            _pad: 0,
        },
        GpuInstanceData {
            model: m_metal,
            prev_model: m_metal,
            albedo_metallic: [0.9, 0.85, 0.7, 0.9],
            mesh_id: 0,
            material_id: 2,
            roughness: 0.2,
            _pad: 0,
        },
    ];

    let materials = [
        GpuRtMaterial {
            albedo_roughness: [0.7, 0.7, 0.7, 0.7],
            emission_metallic: [0.0, 0.0, 0.0, 0.0],
        },
        GpuRtMaterial {
            albedo_roughness: [0.8, 0.25, 0.2, 0.5],
            emission_metallic: [0.0, 0.0, 0.0, 0.0],
        },
        GpuRtMaterial {
            albedo_roughness: [0.9, 0.85, 0.7, 0.2],
            emission_metallic: [0.0, 0.0, 0.0, 0.9],
        },
    ];

    let lights = [
        GpuRtLight {
            pos_energy: [0.0, 0.0, 0.0, 4.0],
            dir_type: [-0.4, -0.8, -0.45, 0.0], // directional sun
            color_range: [1.0, 0.95, 0.85, 1000.0],
        },
        GpuRtLight {
            pos_energy: [-2.0, 2.5, 2.0, 6.0],
            dir_type: [0.0, -1.0, 0.0, 1.0], // omni point light
            color_range: [0.9, 0.95, 1.0, 20.0],
        },
    ];

    let mut scene = GpuScene::new(&mut rhi, &vertices, &indices, &[mesh], &instances)
        .expect("Failed to create GpuScene");

    let pool = rhi
        .create_command_pool(
            rhi.queue_families.graphics,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .expect("Failed to create command pool");

    // Build initial BLAS + TLAS
    let init_cmd = rhi.allocate_command_buffer(pool).expect("alloc init cmd");
    unsafe {
        rhi.device
            .begin_command_buffer(
                init_cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }
    let mut accel = AccelStructureManager::build_initial(
        &mut rhi,
        init_cmd,
        &scene,
        &[mesh],
        &instances,
        vertices.len() as u32,
    )
    .expect("Failed to build BLAS/TLAS");
    unsafe { rhi.device.end_command_buffer(init_cmd).unwrap() };
    rhi.submit_and_wait(rhi.graphics_queue, init_cmd)
        .expect("submit BLAS/TLAS build");

    let mut rt_pass = RtGiPass::new(&mut rhi, WIDTH, HEIGHT, &materials, &lights)
        .expect("Failed to create RtGiPass");
    let mut nrd = NrdDenoiser::new_with_mode(&mut rhi, WIDTH, HEIGHT, mode)
        .expect("Failed to create NrdDenoiser");
    println!(
        "Testing denoiser mode={mode:?} -> backend='{}' (is_using_nrd={})",
        nrd.backend_name(),
        nrd.is_using_nrd()
    );
    if mode == DenoiserMode::Svgf {
        assert!(!nrd.is_using_nrd());
        assert_eq!(nrd.backend_name(), "Built-in SVGF (GPL-3.0)");
    }
    let mut compose =
        NrdComposePass::new(&mut rhi, WIDTH, HEIGHT).expect("Failed to create NrdComposePass");

    rt_pass.bind_resources(&rhi.device, &accel, &gbuffer_pipe.targets);
    compose.bind_resources(&rhi.device, &nrd, &rt_pass, &gbuffer_pipe.targets);

    let readback_size = (WIDTH * HEIGHT * 8) as vk::DeviceSize; // RGBA16F = 8 bytes/pixel
    let dev = rhi.device.clone();
    let mut readback = GpuBuffer::new(
        &dev,
        rhi.allocator_mut(),
        "hdr-readback",
        readback_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
    )
    .expect("Failed to create readback buffer");

    let cam_pos = [0.0f32, 1.5, 4.5];
    let view = mat4_look_at_rh(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let proj = mat4_perspective_reverse_z(60.0f32.to_radians(), 1.0, 0.1, 100.0);
    let view_proj = mat4_mul(proj, view);
    let inv_view = mat4_inverse(view);
    let inv_proj = mat4_inverse(proj);
    let inv_view_proj = mat4_inverse(view_proj);
    let frustum_planes = extract_frustum_planes_reverse_z(view_proj);

    let mut prev_jitter = [0.0f32; 2];

    for frame in 0..2u32 {
        let jx = (halton(frame + 1, 2) - 0.5) / WIDTH as f32;
        let jy = (halton(frame + 1, 3) - 0.5) / HEIGHT as f32;

        let uniforms = FrameUniforms {
            view,
            proj,
            view_proj,
            prev_view_proj: view_proj,
            frustum_planes,
            jitter: [jx * 2.0, jy * 2.0, prev_jitter[0] * 2.0, prev_jitter[1] * 2.0],
            screen_size: [WIDTH, HEIGHT],
            hiz_size: [WIDTH, HEIGHT],
        };
        let slot = frame as usize;
        scene
            .update_frame_uniforms(slot, &uniforms)
            .expect("update_frame_uniforms");

        let rt_params = RtParams {
            inv_view,
            inv_proj,
            inv_view_proj,
            cam_pos_range: [cam_pos[0], cam_pos[1], cam_pos[2], DENOISING_RANGE],
            jitter_frame: [jx * 2.0, jy * 2.0, frame as f32, 1.0],
            screen_size: [WIDTH, HEIGHT],
            light_count: lights.len() as u32,
            _pad: 0,
        };

        let cmd = rhi.allocate_command_buffer(pool).expect("alloc frame cmd");
        unsafe {
            rhi.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        if frame > 0 {
            accel
                .update_tlas(&rhi, cmd, &instances, slot, true)
                .expect("update_tlas");
        }

        gbuffer_pipe
            .record_frame(&rhi, &bindless, cmd, &mut scene, &uniforms, false, slot)
            .expect("gbuffer record_frame");

        rt_pass
            .record(&rhi.device, cmd, &gbuffer_pipe.targets, &rt_params, slot)
            .expect("rt_pass record");

        let nrd_inputs = NrdInputs {
            diff_rad_hitdist: &rt_pass.nrd_diff_rad_hitdist,
            spec_rad_hitdist: &rt_pass.nrd_spec_rad_hitdist,
            normal_roughness: &rt_pass.nrd_normal_roughness,
            view_z: &rt_pass.nrd_view_z,
            motion: &gbuffer_pipe.targets.motion,
            diff_confidence: &rt_pass.nrd_diff_confidence,
            spec_confidence: &rt_pass.nrd_spec_confidence,
        };
        nrd.record(
            &rhi.device,
            cmd,
            &nrd_inputs,
            proj,
            proj,
            view,
            view,
            [jx, jy],
            prev_jitter,
            DENOISING_RANGE,
            16.67,
            frame == 0,
            slot,
        )
        .expect("nrd record");

        compose.record(&rhi.device, cmd, &rt_pass, slot);

        if frame == 1 {
            let copy_barrier = [sync::image_barrier(
                compose.out_hdr.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                compose.out_hdr.subresource_range(),
            )];
            unsafe {
                sync::cmd_pipeline_barrier2(&rhi.device, cmd, &[], &[], &copy_barrier);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: WIDTH,
                        height: HEIGHT,
                        depth: 1,
                    });
                rhi.device.cmd_copy_image_to_buffer(
                    cmd,
                    compose.out_hdr.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    &[region],
                );
            }
        }

        unsafe { rhi.device.end_command_buffer(cmd).unwrap() };
        rhi.submit_and_wait(rhi.graphics_queue, cmd)
            .expect("submit frame");
        prev_jitter = [jx, jy];
    }

    // Verify HDR output has finite, non-zero pixels
    let bytes = readback.mapped_slice_mut().expect("readback slice");
    let words: &[u16] = bytemuck::cast_slice(bytes);
    let mut non_zero_pixels = 0usize;
    let mut max_lum = 0.0f32;
    for px in words.as_chunks::<4>().0 {
        let r = f16_to_f32(px[0]);
        let g = f16_to_f32(px[1]);
        let b = f16_to_f32(px[2]);
        assert!(r.is_finite() && g.is_finite() && b.is_finite(), "Non-finite HDR pixel: ({r}, {g}, {b})");
        let lum = r + g + b;
        if lum > 0.001 {
            non_zero_pixels += 1;
        }
        max_lum = max_lum.max(lum);
    }
    println!("HDR output: {non_zero_pixels}/{} non-zero pixels, max RGB sum = {max_lum:.3}", (WIDTH * HEIGHT) as usize);
    assert!(non_zero_pixels > 100, "Expected lit HDR pixels, got {non_zero_pixels}");

    assert_eq!(rhi.validation_errors(), 0, "Expected 0 Vulkan validation errors");
    assert_eq!(rhi.validation_warnings(), 0, "Expected 0 Vulkan validation warnings");

    // Clean teardown
    rhi.wait_idle().expect("wait_idle");
    let device = rhi.device.clone();
    unsafe { device.destroy_command_pool(pool, None) };
    readback.destroy(&device, rhi.allocator_mut()).unwrap();
    compose.destroy(&mut rhi).unwrap();
    nrd.destroy(&mut rhi).unwrap();
    rt_pass.destroy(&mut rhi).unwrap();
    accel.destroy(&mut rhi).unwrap();
    scene.destroy(&mut rhi).unwrap();
    gbuffer_pipe.destroy(&mut rhi).unwrap();
    bindless.destroy(&device);
}

#[test]
fn headless_rt_nrd_two_frames_zero_validation_errors() {
    run_rt_denoiser_test(DenoiserMode::Auto);
}

#[test]
fn headless_rt_svgf_two_frames_zero_validation_errors() {
    run_rt_denoiser_test(DenoiserMode::Svgf);
}
