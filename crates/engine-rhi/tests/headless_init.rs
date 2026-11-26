use ash::vk;
use bytemuck::{Pod, Zeroable};
use engine_rhi::device::API_VERSION_1_4;
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{
    BindlessConfig, BindlessTable, GpuBuffer, RhiConfig, RhiDevice, spirv_bytes_to_words, sync,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoopPushConstants {
    out_buf: u64,
    count: u32,
    _pad: u32,
}

#[test]
fn headless_vulkan14_init_and_compute_bda_sync2() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: "headless-init-test".to_string(),
        enable_validation: true,
        enable_sync_validation: true,
    })
    .expect("Failed to initialize RhiDevice");

    let caps = &rhi.capabilities;
    println!("Selected GPU: {} ({:?})", caps.device_name, caps.device_type);
    println!(
        "Vulkan API version: {}.{}.{}",
        vk::api_version_major(caps.api_version),
        vk::api_version_minor(caps.api_version),
        vk::api_version_patch(caps.api_version)
    );
    println!(
        "RT extensions: AS={}, Pipeline={}, RayQuery={}, PosFetch={}",
        caps.khr_acceleration_structure,
        caps.khr_ray_tracing_pipeline,
        caps.khr_ray_query,
        caps.khr_ray_tracing_position_fetch
    );
    println!(
        "Cooperative Matrix: supported={}, configs={}",
        caps.khr_cooperative_matrix,
        caps.cooperative_matrix_properties.len()
    );
    for (i, cm) in caps.cooperative_matrix_properties.iter().enumerate() {
        println!(
            "  CoopMatrix[{i}]: {}x{}x{} A={:?} B={:?} C={:?} R={:?} scope={:?}",
            cm.m_size, cm.n_size, cm.k_size, cm.a_type, cm.b_type, cm.c_type, cm.result_type, cm.scope
        );
    }

    assert!(caps.api_version >= API_VERSION_1_4);
    assert!(caps.buffer_device_address);
    assert!(caps.dynamic_rendering);
    assert!(caps.synchronization2);
    assert!(caps.runtime_descriptor_array);
    assert!(caps.descriptor_binding_partially_bound);
    assert!(caps.khr_acceleration_structure);
    assert!(caps.khr_ray_tracing_pipeline);
    assert!(caps.khr_ray_query);
    assert!(caps.khr_cooperative_matrix);
    assert!(!caps.cooperative_matrix_properties.is_empty());

    // Bindless table smoke check
    let bindless = BindlessTable::new(&rhi.device, BindlessConfig::default())
        .expect("Failed to create BindlessTable");

    // Allocate a host-visible BDA buffer
    const COUNT: u32 = 64;
    let buf_size = (COUNT as usize * std::mem::size_of::<u32>()) as vk::DeviceSize;
    let mut bda_buf = {
        let device = rhi.device.clone();
        GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "test-bda-buffer",
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu,
        )
        .expect("Failed to allocate BDA GpuBuffer")
    };
    assert_ne!(bda_buf.device_address, 0);
    bda_buf
        .mapped_slice_mut()
        .expect("mapped slice")
        .fill(0);

    // Load compiled SPIR-V compute shader from OUT_DIR
    let spv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/noop.comp.spv"));
    let spv_words = spirv_bytes_to_words(spv_bytes);
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
    let shader_module = unsafe { rhi.device.create_shader_module(&shader_info, None) }
        .expect("create_shader_module");

    let pc_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<NoopPushConstants>() as u32)];
    let pipe_layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_range);
    let pipeline_layout = unsafe { rhi.device.create_pipeline_layout(&pipe_layout_info, None) }
        .expect("create_pipeline_layout");

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");
    let compute_info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout)];
    let pipeline = unsafe {
        rhi.device
            .create_compute_pipelines(vk::PipelineCache::null(), &compute_info, None)
    }
    .expect("create_compute_pipelines")[0];

    // Command pool & command buffer
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(rhi.queue_families.compute)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let cmd_pool = unsafe { rhi.device.create_command_pool(&pool_info, None) }
        .expect("create_command_pool");

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { rhi.device.allocate_command_buffers(&alloc_info) }
        .expect("allocate_command_buffers")[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        rhi.device
            .begin_command_buffer(cmd, &begin_info)
            .expect("begin_command_buffer");

        rhi.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

        let pc = NoopPushConstants {
            out_buf: bda_buf.device_address,
            count: COUNT,
            _pad: 0,
        };
        rhi.device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
        );
        rhi.device.cmd_dispatch(cmd, 1, 1, 1);

        // Synchronization2 barrier: compute shader write -> host read
        let mem_barrier = [sync::memory_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        )];
        sync::cmd_pipeline_barrier2(&rhi.device, cmd, &mem_barrier, &[], &[]);

        rhi.device
            .end_command_buffer(cmd)
            .expect("end_command_buffer");
    }

    let cmd_submit = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
    let submit_info = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_submit)];
    unsafe {
        rhi.device
            .queue_submit2(rhi.compute_queue, &submit_info, vk::Fence::null())
            .expect("queue_submit2");
    }
    rhi.wait_idle().expect("wait_idle");

    // Verify GPU wrote 1..=64 via BDA
    let out_bytes = bda_buf.mapped_slice_mut().expect("mapped_slice_mut");
    let out_words: &[u32] = bytemuck::cast_slice(out_bytes);
    for (i, &val) in out_words.iter().enumerate() {
        assert_eq!(val, (i as u32) + 1, "mismatch at index {i}");
    }

    // Cleanup
    unsafe {
        rhi.device.destroy_command_pool(cmd_pool, None);
        rhi.device.destroy_pipeline(pipeline, None);
        rhi.device.destroy_pipeline_layout(pipeline_layout, None);
        rhi.device.destroy_shader_module(shader_module, None);
    }
    {
        let device = rhi.device.clone();
        bda_buf
            .destroy(&device, rhi.allocator_mut())
            .expect("destroy GpuBuffer");
    }
    bindless.destroy(&rhi.device);

    assert_eq!(rhi.validation_errors(), 0, "Expected 0 Vulkan validation errors");
    assert_eq!(
        rhi.validation_warnings(),
        0,
        "Expected 0 Vulkan validation warnings"
    );
}
