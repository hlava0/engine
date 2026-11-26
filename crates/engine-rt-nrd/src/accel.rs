use ash::vk;
use engine_gpu_driven::{GpuInstanceData, GpuMeshDrawInfo, GpuScene, GpuVertex};
use engine_rhi::gpu_allocator::MemoryLocation;
use engine_rhi::{GpuBuffer, RhiDevice, align_up, sync};

use crate::error::{Result, RtNrdError};

/// Bottom-Level Acceleration Structure for a single mesh in `GpuScene`.
pub struct BlasEntry {
    pub handle: vk::AccelerationStructureKHR,
    pub device_address: u64,
    pub buffer: GpuBuffer,
    pub scratch: GpuBuffer,
}

/// Manages BLASes (one per mesh) and a reusable TLAS for hardware ray-query tracing.
pub struct AccelStructureManager {
    pub blases: Vec<BlasEntry>,
    pub tlas: vk::AccelerationStructureKHR,
    pub tlas_device_address: u64,
    pub tlas_buffer: GpuBuffer,
    pub tlas_scratch: GpuBuffer,
    pub tlas_instances_buffer: GpuBuffer,
    pub max_instances: u32,
    pub current_instances: u32,
    tlas_built_once: bool,
}

impl AccelStructureManager {
    /// Allocate BLAS and TLAS resources and record the initial BLAS + TLAS build commands into `cmd`.
    ///
    /// Note: `cmd` must be in the recording state. The caller must submit `cmd` and wait before
    /// destroying `AccelStructureManager`.
    pub fn build_initial(
        rhi: &mut RhiDevice,
        cmd: vk::CommandBuffer,
        scene: &GpuScene,
        meshes: &[GpuMeshDrawInfo],
        instances: &[GpuInstanceData],
        total_vertex_count: u32,
    ) -> Result<Self> {
        let as_device = rhi
            .acceleration_structure
            .as_ref()
            .ok_or(RtNrdError::RayTracingUnsupported)?
            .clone();
        let device = rhi.device.clone();

        if meshes.is_empty() || instances.is_empty() {
            return Err(RtNrdError::InvalidConfig(
                "AccelStructureManager requires at least 1 mesh and 1 instance".to_string(),
            ));
        }

        let scratch_align = rhi.capabilities.vendor_profile.min_as_scratch_alignment as u64;

        let mut blases = Vec::with_capacity(meshes.len());
        for (i, mesh) in meshes.iter().enumerate() {
            let entry = build_blas_entry(
                &device,
                &as_device,
                rhi.allocator_mut(),
                cmd,
                scene,
                mesh,
                total_vertex_count,
                i,
                scratch_align,
            )?;
            blases.push(entry);
        }

        // Barrier: BLAS builds -> TLAS build
        let blas_to_tlas = [sync::memory_barrier(
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        )];
        unsafe {
            sync::cmd_pipeline_barrier2(&device, cmd, &blas_to_tlas, &[], &[]);
        }

        let max_instances = instances.len() as u32;
        let inst_buf_size =
            (max_instances as usize * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>())
                as vk::DeviceSize;

        let mut tlas_instances_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "tlas-instances",
            inst_buf_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
        )?;
        write_tlas_instances(&mut tlas_instances_buffer, instances, &blases)?;

        let tlas_geom = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: tlas_instances_buffer.device_address,
                    }),
            });
        let tlas_geoms = [tlas_geom];

        let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&tlas_geoms);

        let max_prim_counts = [max_instances];
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build_info,
                &max_prim_counts,
                &mut size_info,
            );
        }

        let tlas_buffer = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "tlas-storage",
            size_info.acceleration_structure_size.max(256),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            MemoryLocation::GpuOnly,
        )?;

        let req_scratch = size_info
            .build_scratch_size
            .max(size_info.update_scratch_size)
            .max(scratch_align);
        let scratch_size = align_up(req_scratch, scratch_align) + scratch_align;
        let tlas_scratch = GpuBuffer::new(
            &device,
            rhi.allocator_mut(),
            "tlas-scratch",
            scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        )?;
        let tlas_scratch_addr = align_up(tlas_scratch.device_address, scratch_align);

        let tlas_create = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(tlas_buffer.buffer)
            .offset(0)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        let tlas = unsafe { as_device.create_acceleration_structure(&tlas_create, None)? };

        let tlas_addr_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(tlas);
        let tlas_device_address =
            unsafe { as_device.get_acceleration_structure_device_address(&tlas_addr_info) };

        let mut build_cmd_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(tlas)
            .geometries(&tlas_geoms)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: tlas_scratch_addr,
            });

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(max_instances)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        let ranges = [range];
        let range_slices: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&ranges];

        unsafe {
            as_device.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_mut(&mut build_cmd_info),
                &range_slices,
            );
        }

        // Barrier: TLAS build -> compute shader ray query
        let tlas_to_compute = [sync::memory_barrier(
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        )];
        unsafe {
            sync::cmd_pipeline_barrier2(&device, cmd, &tlas_to_compute, &[], &[]);
        }

        Ok(Self {
            blases,
            tlas,
            tlas_device_address,
            tlas_buffer,
            tlas_scratch,
            tlas_instances_buffer,
            max_instances,
            current_instances: max_instances,
            tlas_built_once: true,
        })
    }

    /// Update/refit the TLAS with updated instance transforms (zero allocations).
    pub fn update_tlas(
        &mut self,
        rhi: &RhiDevice,
        cmd: vk::CommandBuffer,
        instances: &[GpuInstanceData],
    ) -> Result<()> {
        let as_device = rhi
            .acceleration_structure
            .as_ref()
            .ok_or(RtNrdError::RayTracingUnsupported)?;

        if instances.len() > self.max_instances as usize {
            return Err(RtNrdError::InvalidConfig(format!(
                "TLAS instance count {} exceeds max {}",
                instances.len(),
                self.max_instances
            )));
        }

        write_tlas_instances(&mut self.tlas_instances_buffer, instances, &self.blases)?;
        let count = instances.len() as u32;
        let use_update = self.tlas_built_once
            && count == self.current_instances
            && rhi.capabilities.vendor_profile.prefer_tlas_refit;
        self.current_instances = count;
        self.tlas_built_once = true;

        let scratch_align = rhi.capabilities.vendor_profile.min_as_scratch_alignment as u64;
        let tlas_scratch_addr = align_up(self.tlas_scratch.device_address, scratch_align);

        let tlas_geom = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.tlas_instances_buffer.device_address,
                    }),
            });
        let tlas_geoms = [tlas_geom];

        let mut build_cmd_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(if use_update {
                vk::BuildAccelerationStructureModeKHR::UPDATE
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD
            })
            .src_acceleration_structure(if use_update {
                self.tlas
            } else {
                vk::AccelerationStructureKHR::null()
            })
            .dst_acceleration_structure(self.tlas)
            .geometries(&tlas_geoms)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: tlas_scratch_addr,
            });

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        let ranges = [range];
        let range_slices: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&ranges];

        unsafe {
            as_device.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_mut(&mut build_cmd_info),
                &range_slices,
            );
        }

        let tlas_to_compute = [sync::memory_barrier(
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        )];
        unsafe {
            sync::cmd_pipeline_barrier2(&rhi.device, cmd, &tlas_to_compute, &[], &[]);
        }

        Ok(())
    }

    pub fn destroy(self, rhi: &mut RhiDevice) -> Result<()> {
        let as_device = rhi
            .acceleration_structure
            .as_ref()
            .ok_or(RtNrdError::RayTracingUnsupported)?
            .clone();
        let device = rhi.device.clone();
        let alloc = rhi.allocator_mut();

        unsafe {
            as_device.destroy_acceleration_structure(self.tlas, None);
        }
        self.tlas_buffer.destroy(&device, alloc)?;
        self.tlas_scratch.destroy(&device, alloc)?;
        self.tlas_instances_buffer.destroy(&device, alloc)?;

        for blas in self.blases {
            unsafe {
                as_device.destroy_acceleration_structure(blas.handle, None);
            }
            blas.buffer.destroy(&device, alloc)?;
            blas.scratch.destroy(&device, alloc)?;
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_blas_entry(
    device: &ash::Device,
    as_device: &ash::khr::acceleration_structure::Device,
    allocator: &mut engine_rhi::gpu_allocator::vulkan::Allocator,
    cmd: vk::CommandBuffer,
    scene: &GpuScene,
    mesh: &GpuMeshDrawInfo,
    total_vertex_count: u32,
    mesh_index: usize,
    scratch_align: u64,
) -> Result<BlasEntry> {
    let vertex_stride = std::mem::size_of::<GpuVertex>() as u64;
    let vertex_byte_offset = (mesh.vertex_offset as i64 * vertex_stride as i64) as u64;
    let index_byte_offset = mesh.first_index as u64 * std::mem::size_of::<u32>() as u64;
    let primitive_count = mesh.index_count / 3;
    let max_vertex = total_vertex_count
        .saturating_sub(mesh.vertex_offset.max(0) as u32)
        .saturating_sub(1);

    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: scene.vertex_buffer.device_address + vertex_byte_offset,
        })
        .vertex_stride(vertex_stride)
        .max_vertex(max_vertex)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: scene.index_buffer.device_address + index_byte_offset,
        });

    let geom = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles });
    let geoms = [geom];

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geoms);

    let max_prim_counts = [primitive_count];
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_device.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &max_prim_counts,
            &mut size_info,
        );
    }

    let buffer = GpuBuffer::new(
        device,
        allocator,
        &format!("blas-{mesh_index}-storage"),
        size_info.acceleration_structure_size.max(256),
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let req_scratch = size_info.build_scratch_size.max(scratch_align);
    let scratch_size = align_up(req_scratch, scratch_align) + scratch_align;
    let scratch = GpuBuffer::new(
        device,
        allocator,
        &format!("blas-{mesh_index}-scratch"),
        scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
    )?;
    let scratch_addr = align_up(scratch.device_address, scratch_align);

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer.buffer)
        .offset(0)
        .size(size_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
    let handle = unsafe { as_device.create_acceleration_structure(&create_info, None)? };

    let addr_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
    let device_address = unsafe { as_device.get_acceleration_structure_device_address(&addr_info) };

    let mut build_cmd_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(handle)
        .geometries(&geoms)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });

    let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(primitive_count)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0);
    let ranges = [range];
    let range_slices: [&[vk::AccelerationStructureBuildRangeInfoKHR]; 1] = [&ranges];

    unsafe {
        as_device.cmd_build_acceleration_structures(
            cmd,
            std::slice::from_mut(&mut build_cmd_info),
            &range_slices,
        );
    }

    Ok(BlasEntry {
        handle,
        device_address,
        buffer,
        scratch,
    })
}

fn write_tlas_instances(
    buffer: &mut GpuBuffer,
    instances: &[GpuInstanceData],
    blases: &[BlasEntry],
) -> Result<()> {
    let slice = buffer.mapped_slice_mut()?;
    let elem_size = std::mem::size_of::<vk::AccelerationStructureInstanceKHR>();

    for (i, inst) in instances.iter().enumerate() {
        let blas = blases.get(inst.mesh_id as usize).ok_or_else(|| {
            RtNrdError::InvalidConfig(format!("Instance {i} references invalid mesh_id {}", inst.mesh_id))
        })?;
        let m = inst.model;
        let transform = vk::TransformMatrixKHR {
            matrix: [
                m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
            ],
        };
        let as_inst = vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(inst.material_id, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.device_address,
            },
        };
        let bytes = unsafe {
            std::slice::from_raw_parts((&as_inst as *const vk::AccelerationStructureInstanceKHR).cast::<u8>(), elem_size)
        };
        let dst = &mut slice[i * elem_size..(i + 1) * elem_size];
        dst.copy_from_slice(bytes);
    }

    Ok(())
}
