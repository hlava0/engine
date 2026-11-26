use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::{Result, RhiError};

/// A 2D GPU image backed by `gpu-allocator` with a default `vk::ImageView` and optional per-mip views.
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_views: Vec<vk::ImageView>,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub mip_levels: u32,
    pub aspect: vk::ImageAspectFlags,
}

impl GpuImage {
    /// Allocate a 2D `GpuOnly` image and create a default `vk::ImageView` covering all mip levels.
    #[allow(clippy::too_many_arguments)]
    pub fn create_2d(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        name: &str,
    ) -> Result<Self> {
        if width == 0 || height == 0 || mip_levels == 0 {
            return Err(RhiError::InvalidImage(
                "Image width, height, and mip_levels must be greater than zero".to_string(),
            ));
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(alloc) => alloc,
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                return Err(RhiError::Allocation(err));
            }
        };

        if let Err(err) =
            unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        {
            let _ = allocator.free(allocation);
            unsafe { device.destroy_image(image, None) };
            return Err(RhiError::Vk(err));
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(v) => v,
            Err(err) => {
                unsafe { device.destroy_image(image, None) };
                let _ = allocator.free(allocation);
                return Err(RhiError::Vk(err));
            }
        };

        Ok(Self {
            image,
            view,
            mip_views: Vec::new(),
            allocation: Some(allocation),
            format,
            extent: vk::Extent2D { width, height },
            mip_levels,
            aspect,
        })
    }

    /// Create per-mip `vk::ImageView`s (`0..mip_levels`), useful for Hi-Z mip chain generation.
    pub fn with_mip_views(mut self, device: &ash::Device) -> Result<Self> {
        self.mip_views.reserve(self.mip_levels as usize);
        for mip in 0..self.mip_levels {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: self.aspect,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            match unsafe { device.create_image_view(&view_info, None) } {
                Ok(v) => self.mip_views.push(v),
                Err(err) => {
                    for created in self.mip_views.drain(..) {
                        unsafe { device.destroy_image_view(created, None) };
                    }
                    return Err(RhiError::Vk(err));
                }
            }
        }
        Ok(self)
    }

    /// Return the per-mip `vk::ImageView` for `mip`.
    #[inline]
    pub fn mip_view(&self, mip: u32) -> vk::ImageView {
        self.mip_views[mip as usize]
    }

    /// Full subresource range covering all mip levels.
    #[inline]
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: 0,
            level_count: self.mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        }
    }

    /// Subresource range for a single mip level.
    #[inline]
    pub fn mip_subresource_range(&self, mip: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: mip,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }

    /// Destroy all image views, the Vulkan image, and free the underlying allocation in safe order.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<()> {
        unsafe {
            for view in self.mip_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
            }
        }
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc)?;
        }
        Ok(())
    }
}
