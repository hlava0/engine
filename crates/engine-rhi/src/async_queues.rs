//! Async `transfer_queue` and `compute_queue` execution using Vulkan 1.2/1.4 timeline semaphores.

use ash::vk;

use crate::device::RhiDevice;
use crate::error::Result;

/// Manages timeline-semaphore-synchronized async `transfer_queue` and `compute_queue` submissions.
pub struct AsyncQueueContext {
    pub transfer_timeline: vk::Semaphore,
    pub compute_timeline: vk::Semaphore,
    pub transfer_pool: vk::CommandPool,
    pub compute_pool: vk::CommandPool,
    transfer_value: u64,
    compute_value: u64,
}

impl AsyncQueueContext {
    pub fn new(rhi: &RhiDevice) -> Result<Self> {
        let dev = &rhi.device;

        let mut tl_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let tl_info = vk::SemaphoreCreateInfo::default().push_next(&mut tl_type);
        let transfer_timeline = unsafe { dev.create_semaphore(&tl_info, None)? };
        let compute_timeline = match unsafe { dev.create_semaphore(&tl_info, None) } {
            Ok(s) => s,
            Err(e) => {
                unsafe { dev.destroy_semaphore(transfer_timeline, None) };
                return Err(e.into());
            }
        };

        let transfer_pool = rhi.create_command_pool(
            rhi.queue_families.transfer,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )?;
        let compute_pool = rhi.create_command_pool(
            rhi.queue_families.compute,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )?;

        Ok(Self {
            transfer_timeline,
            compute_timeline,
            transfer_pool,
            compute_pool,
            transfer_value: 0,
            compute_value: 0,
        })
    }

    /// Record and submit work asynchronously on `rhi.transfer_queue`, returning a timeline ticket.
    pub fn submit_transfer<F>(&mut self, rhi: &RhiDevice, record: F) -> Result<u64>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<()>,
    {
        let dev = &rhi.device;
        let cmd = rhi.allocate_command_buffer(self.transfer_pool)?;
        unsafe {
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        record(dev, cmd)?;
        unsafe { dev.end_command_buffer(cmd)? };

        self.transfer_value += 1;
        let ticket = self.transfer_value;
        let sig = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline)
            .value(ticket)
            .stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)];
        let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(&sig)];
        unsafe {
            dev.queue_submit2(rhi.transfer_queue, &submit, vk::Fence::null())?;
        }
        Ok(ticket)
    }

    /// Record and submit work asynchronously on `rhi.compute_queue`, optionally waiting on a
    /// `transfer_timeline` ticket, and returning a `compute_timeline` ticket.
    pub fn submit_compute<F>(
        &mut self,
        rhi: &RhiDevice,
        wait_transfer_ticket: Option<u64>,
        record: F,
    ) -> Result<u64>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<()>,
    {
        let dev = &rhi.device;
        let cmd = rhi.allocate_command_buffer(self.compute_pool)?;
        unsafe {
            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        record(dev, cmd)?;
        unsafe { dev.end_command_buffer(cmd)? };

        self.compute_value += 1;
        let ticket = self.compute_value;

        let mut waits = Vec::new();
        if let Some(t) = wait_transfer_ticket {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.transfer_timeline)
                    .value(t)
                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
            );
        }
        let sig = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.compute_timeline)
            .value(ticket)
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];
        let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(&sig)];
        unsafe {
            dev.queue_submit2(rhi.compute_queue, &submit, vk::Fence::null())?;
        }
        Ok(ticket)
    }

    /// Block CPU until the given `transfer_timeline` ticket completes.
    pub fn wait_transfer_cpu(&self, rhi: &RhiDevice, ticket: u64) -> Result<()> {
        let sems = [self.transfer_timeline];
        let vals = [ticket];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&vals);
        unsafe {
            rhi.device.wait_semaphores(&info, u64::MAX)?;
        }
        Ok(())
    }

    /// Block CPU until the given `compute_timeline` ticket completes.
    pub fn wait_compute_cpu(&self, rhi: &RhiDevice, ticket: u64) -> Result<()> {
        let sems = [self.compute_timeline];
        let vals = [ticket];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&vals);
        unsafe {
            rhi.device.wait_semaphores(&info, u64::MAX)?;
        }
        Ok(())
    }

    /// Build a GPU `SemaphoreSubmitInfo` to wait on a `transfer_timeline` ticket in `graphics_queue`.
    pub fn transfer_wait_info(
        &self,
        ticket: u64,
        dst_stage: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline)
            .value(ticket)
            .stage_mask(dst_stage)
    }

    /// Build a GPU `SemaphoreSubmitInfo` to wait on a `compute_timeline` ticket in `graphics_queue`.
    pub fn compute_wait_info(
        &self,
        ticket: u64,
        dst_stage: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.compute_timeline)
            .value(ticket)
            .stage_mask(dst_stage)
    }

    pub fn destroy(self, rhi: &RhiDevice) {
        let dev = &rhi.device;
        unsafe {
            dev.destroy_command_pool(self.transfer_pool, None);
            dev.destroy_command_pool(self.compute_pool, None);
            dev.destroy_semaphore(self.transfer_timeline, None);
            dev.destroy_semaphore(self.compute_timeline, None);
        }
    }
}
