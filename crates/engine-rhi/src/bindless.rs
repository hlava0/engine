use ash::vk;

use crate::error::Result;

/// Binding indices inside the global bindless descriptor set.
pub const BINDING_SAMPLED_IMAGES: u32 = 0;
pub const BINDING_STORAGE_IMAGES: u32 = 1;
pub const BINDING_STORAGE_BUFFERS: u32 = 2;
pub const BINDING_SAMPLERS: u32 = 3;

/// Configuration for the bindless descriptor table capacities.
#[derive(Debug, Clone, Copy)]
pub struct BindlessConfig {
    pub max_sampled_images: u32,
    pub max_storage_images: u32,
    pub max_storage_buffers: u32,
    pub max_samplers: u32,
}

impl Default for BindlessConfig {
    fn default() -> Self {
        Self {
            max_sampled_images: 4096,
            max_storage_images: 1024,
            max_storage_buffers: 1024,
            max_samplers: 256,
        }
    }
}

/// Global bindless descriptor set layout, pool, and set using `UPDATE_AFTER_BIND` and `PARTIALLY_BOUND`.
pub struct BindlessTable {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl BindlessTable {
    /// Create a new bindless descriptor table for sampled images, storage images, storage buffers, and samplers.
    pub fn new(device: &ash::Device, config: BindlessConfig) -> Result<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLED_IMAGES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(config.max_sampled_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_IMAGES)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(config.max_storage_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_BUFFERS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(config.max_storage_buffers)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLERS)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(config.max_samplers)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut flags_info);

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: config.max_sampled_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: config.max_storage_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: config.max_storage_buffers,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: config.max_samplers,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                unsafe { device.destroy_descriptor_set_layout(layout, None) };
                return Err(err.into());
            }
        };

        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets = match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets,
            Err(err) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(layout, None);
                }
                return Err(err.into());
            }
        };

        Ok(Self {
            layout,
            pool,
            set: sets[0],
        })
    }

    /// Update a sampled image slot in the bindless table.
    pub fn write_sampled_image(
        &self,
        device: &ash::Device,
        index: u32,
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        let info = [vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(layout)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_SAMPLED_IMAGES)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    /// Update a storage image slot in the bindless table.
    pub fn write_storage_image(
        &self,
        device: &ash::Device,
        index: u32,
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        let info = [vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(layout)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_STORAGE_IMAGES)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    /// Update a storage buffer slot in the bindless table.
    pub fn write_storage_buffer(
        &self,
        device: &ash::Device,
        index: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(range)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_STORAGE_BUFFERS)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    /// Update a sampler slot in the bindless table.
    pub fn write_sampler(&self, device: &ash::Device, index: u32, sampler: vk::Sampler) {
        let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_SAMPLERS)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    /// Destroy the descriptor pool and layout.
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
