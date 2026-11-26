use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::{Result, RhiError};

/// A GPU buffer backed by `gpu-allocator` with Buffer Device Address (BDA) support.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub device_address: u64,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Allocate a new GPU buffer with `SHADER_DEVICE_ADDRESS` enabled.
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self> {
        if size == 0 {
            return Err(RhiError::InvalidBuffer(
                "Buffer size must be greater than zero".to_string(),
            ));
        }

        let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&create_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(alloc) => alloc,
            Err(err) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(RhiError::Allocation(err));
            }
        };

        if let Err(err) = unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) } {
            let _ = allocator.free(allocation);
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(RhiError::Vk(err));
        }

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { device.get_buffer_device_address(&addr_info) };

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            device_address,
            size,
        })
    }

    /// Return a mutable host-mapped byte slice if the allocation is CPU-accessible.
    pub fn mapped_slice_mut(&mut self) -> Result<&mut [u8]> {
        let alloc = self.allocation.as_mut().ok_or_else(|| {
            RhiError::InvalidBuffer("Buffer allocation has already been freed".to_string())
        })?;
        let slice = alloc.mapped_slice_mut().ok_or_else(|| {
            RhiError::InvalidBuffer("Buffer memory is not host-mapped".to_string())
        })?;
        let len = (self.size as usize).min(slice.len());
        Ok(&mut slice[..len])
    }

    /// Free the underlying allocation and destroy the Vulkan buffer.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<()> {
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc)?;
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        Ok(())
    }
}
