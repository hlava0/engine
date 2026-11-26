//! `VK_KHR_swapchain` + `VK_KHR_surface` management for windowed presentation.

use ash::vk;

use crate::FRAMES_IN_FLIGHT;
use crate::device::RhiDevice;
use crate::error::{Result, RhiError};

/// Owns a `vk::SurfaceKHR`, `vk::SwapchainKHR`, its images/views, and presentation semaphores.
pub struct Swapchain {
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub image_available: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub render_finished: Vec<vk::Semaphore>,

    swapchain_loader: ash::khr::swapchain::Device,
}

struct SwapchainResources {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_finished: Vec<vk::Semaphore>,
}

impl Swapchain {
    /// Create a new swapchain for `surface` with initial dimensions `(width, height)`.
    ///
    /// Requires `RhiDevice` to have been initialized with `enable_swapchain: true` and the
    /// appropriate window-system `surface_extensions`.
    pub fn new(rhi: &RhiDevice, surface: vk::SurfaceKHR, width: u32, height: u32) -> Result<Self> {
        let surface_loader = rhi.surface_loader.as_ref().ok_or_else(|| {
            RhiError::MissingCapability(
                "VK_KHR_surface loader not initialized on RhiDevice".to_string(),
            )
        })?;
        let swapchain_loader = rhi
            .swapchain_loader
            .as_ref()
            .ok_or_else(|| {
                RhiError::MissingCapability(
                    "VK_KHR_swapchain loader not initialized on RhiDevice".to_string(),
                )
            })?
            .clone();

        let supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                rhi.physical_device,
                rhi.queue_families.graphics,
                surface,
            )?
        };
        if !supported {
            return Err(RhiError::MissingCapability(
                "Graphics queue family does not support presentation to this surface".to_string(),
            ));
        }

        let sem_info = vk::SemaphoreCreateInfo::default();
        let mut image_available_vec = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            match unsafe { rhi.device.create_semaphore(&sem_info, None) } {
                Ok(sem) => image_available_vec.push(sem),
                Err(err) => {
                    for s in image_available_vec {
                        unsafe { rhi.device.destroy_semaphore(s, None) };
                    }
                    return Err(err.into());
                }
            }
        }
        let image_available: [vk::Semaphore; FRAMES_IN_FLIGHT] = image_available_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!());

        let res = match create_swapchain_resources(
            rhi,
            surface_loader,
            &swapchain_loader,
            surface,
            width,
            height,
            vk::SwapchainKHR::null(),
        ) {
            Ok(r) => r,
            Err(err) => {
                for sem in image_available {
                    unsafe { rhi.device.destroy_semaphore(sem, None) };
                }
                return Err(err);
            }
        };

        Ok(Self {
            surface,
            swapchain: res.swapchain,
            format: res.format,
            color_space: res.color_space,
            extent: res.extent,
            images: res.images,
            image_views: res.image_views,
            image_available,
            render_finished: res.render_finished,
            swapchain_loader,
        })
    }

    /// Return the `image_available` semaphore for `frame_slot`.
    #[inline]
    pub fn image_available_semaphore(&self, frame_slot: usize) -> vk::Semaphore {
        self.image_available[frame_slot % FRAMES_IN_FLIGHT]
    }

    /// Return the `render_finished` semaphore for swapchain `image_index`.
    #[inline]
    pub fn render_finished_semaphore(&self, image_index: u32) -> vk::Semaphore {
        self.render_finished[image_index as usize]
    }

    /// Acquire the next presentable swapchain image for `frame_slot`.
    ///
    /// Returns `Ok((image_index, suboptimal))`. Returns `Err(RhiError::Vk(vk::Result::ERROR_OUT_OF_DATE_KHR))`
    /// when the swapchain is out of date and must be recreated via [`Swapchain::recreate`].
    pub fn acquire_next_image(&self, frame_slot: usize) -> Result<(u32, bool)> {
        let sem = self.image_available_semaphore(frame_slot);
        match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                sem,
                vk::Fence::null(),
            )
        } {
            Ok((image_index, suboptimal)) => Ok((image_index, suboptimal)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Err(RhiError::Vk(vk::Result::ERROR_OUT_OF_DATE_KHR))
            }
            Err(err) => Err(RhiError::Vk(err)),
        }
    }

    /// Present `image_index` on `queue`, waiting on its per-image `render_finished` semaphore.
    ///
    /// Returns `Ok(true)` if the swapchain is suboptimal or out-of-date and should be recreated.
    pub fn present(&self, queue: vk::Queue, image_index: u32) -> Result<bool> {
        let wait_semaphores = [self.render_finished[image_index as usize]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match unsafe { self.swapchain_loader.queue_present(queue, &present_info) } {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(err) => Err(RhiError::Vk(err)),
        }
    }

    /// Recreate the swapchain for new window dimensions `(width, height)`, cleanly destroying
    /// the old views, per-image semaphores, and swapchain handle.
    pub fn recreate(&mut self, rhi: &RhiDevice, width: u32, height: u32) -> Result<()> {
        let surface_loader = rhi.surface_loader.as_ref().ok_or_else(|| {
            RhiError::MissingCapability(
                "VK_KHR_surface loader not initialized on RhiDevice".to_string(),
            )
        })?;

        rhi.wait_idle()?;

        let new_res = create_swapchain_resources(
            rhi,
            surface_loader,
            &self.swapchain_loader,
            self.surface,
            width,
            height,
            self.swapchain,
        )?;

        unsafe {
            for view in self.image_views.drain(..) {
                rhi.device.destroy_image_view(view, None);
            }
            for sem in self.render_finished.drain(..) {
                rhi.device.destroy_semaphore(sem, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = new_res.swapchain;
        self.format = new_res.format;
        self.color_space = new_res.color_space;
        self.extent = new_res.extent;
        self.images = new_res.images;
        self.image_views = new_res.image_views;
        self.render_finished = new_res.render_finished;

        Ok(())
    }

    /// Destroy all semaphores, image views, swapchain, and surface in safe Vulkan teardown order.
    pub fn destroy(self, rhi: &RhiDevice) {
        let _ = rhi.wait_idle();
        unsafe {
            for sem in self.image_available {
                rhi.device.destroy_semaphore(sem, None);
            }
            for sem in self.render_finished {
                rhi.device.destroy_semaphore(sem, None);
            }
            for view in self.image_views {
                rhi.device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            if self.surface != vk::SurfaceKHR::null()
                && let Some(surface_loader) = rhi.surface_loader.as_ref()
            {
                surface_loader.destroy_surface(self.surface, None);
            }
        }
    }
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| {
            formats.iter().copied().find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
        })
        .or_else(|| {
            formats.iter().copied().find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::B8G8R8A8_UNORM
            })
        })
        .unwrap_or(formats[0])
}

fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else if modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
        vk::PresentModeKHR::FIFO_RELAXED
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn choose_composite_alpha(supported: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    for candidate in [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ] {
        if supported.contains(candidate) {
            return candidate;
        }
    }
    vk::CompositeAlphaFlagsKHR::OPAQUE
}

fn create_swapchain_resources(
    rhi: &RhiDevice,
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    width: u32,
    height: u32,
    old_swapchain: vk::SwapchainKHR,
) -> Result<SwapchainResources> {
    let caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(rhi.physical_device, surface)?
    };
    let formats = unsafe {
        surface_loader.get_physical_device_surface_formats(rhi.physical_device, surface)?
    };
    if formats.is_empty() {
        return Err(RhiError::MissingCapability(
            "Surface reported no supported formats".to_string(),
        ));
    }
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(rhi.physical_device, surface)?
    };

    let surface_format = choose_surface_format(&formats);
    let present_mode = choose_present_mode(&present_modes);

    let extent = if caps.current_extent.width != u32::MAX && caps.current_extent.height != u32::MAX
    {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: width
                .max(1)
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: height
                .max(1)
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let mut image_count = caps.min_image_count.saturating_add(1);
    if caps.max_image_count > 0 {
        image_count = image_count.min(caps.max_image_count);
    }

    let desired_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
    let image_usage = if caps.supported_usage_flags.contains(desired_usage) {
        desired_usage
    } else {
        desired_usage & caps.supported_usage_flags
    };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(image_usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(choose_composite_alpha(caps.supported_composite_alpha))
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };
    let images = match unsafe { swapchain_loader.get_swapchain_images(swapchain) } {
        Ok(imgs) => imgs,
        Err(err) => {
            unsafe { swapchain_loader.destroy_swapchain(swapchain, None) };
            return Err(err.into());
        }
    };

    let mut image_views = Vec::with_capacity(images.len());
    for &image in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        match unsafe { rhi.device.create_image_view(&view_info, None) } {
            Ok(view) => image_views.push(view),
            Err(err) => {
                for v in image_views {
                    unsafe { rhi.device.destroy_image_view(v, None) };
                }
                unsafe { swapchain_loader.destroy_swapchain(swapchain, None) };
                return Err(err.into());
            }
        }
    }

    let sem_info = vk::SemaphoreCreateInfo::default();
    let mut render_finished = Vec::with_capacity(images.len());
    for _ in 0..images.len() {
        match unsafe { rhi.device.create_semaphore(&sem_info, None) } {
            Ok(sem) => render_finished.push(sem),
            Err(err) => {
                for s in render_finished {
                    unsafe { rhi.device.destroy_semaphore(s, None) };
                }
                for v in image_views {
                    unsafe { rhi.device.destroy_image_view(v, None) };
                }
                unsafe { swapchain_loader.destroy_swapchain(swapchain, None) };
                return Err(err.into());
            }
        }
    }

    Ok(SwapchainResources {
        swapchain,
        format: surface_format.format,
        color_space: surface_format.color_space,
        extent,
        images,
        image_views,
        render_finished,
    })
}
