//! GPU timestamp profiler using `VK_QUERY_TYPE_TIMESTAMP`.

use engine_rhi::RhiDevice;
use engine_rhi::ash::{self, vk};

/// Per-pass GPU execution timing in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct PassTiming {
    pub name: &'static str,
    pub gpu_ms: f64,
}

/// Manages a `vk::QueryPool` for per-pass GPU timestamp profiling.
pub struct GpuProfiler {
    query_pool: vk::QueryPool,
    max_queries: u32,
    timestamp_period_ns: f64,
    recorded_passes: Vec<(&'static str, u32, u32)>,
}

impl GpuProfiler {
    /// Create a timestamp query pool supporting up to `max_passes` passes per frame.
    pub fn new(rhi: &RhiDevice, max_passes: u32) -> Result<Self, vk::Result> {
        let max_queries = (max_passes * 2).max(2);
        let props = unsafe {
            rhi.instance
                .get_physical_device_properties(rhi.physical_device)
        };
        let timestamp_period_ns = props.limits.timestamp_period as f64;

        let pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(max_queries);
        let query_pool = unsafe { rhi.device.create_query_pool(&pool_info, None)? };

        Ok(Self {
            query_pool,
            max_queries,
            timestamp_period_ns,
            recorded_passes: Vec::with_capacity(max_passes as usize),
        })
    }

    /// Reset all queries in the pool at the start of a command buffer.
    pub fn reset(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        self.recorded_passes.clear();
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, self.max_queries);
        }
    }

    /// Record a start timestamp for `name` and return the pass index.
    pub fn begin_pass(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        name: &'static str,
    ) -> u32 {
        let start_q = (self.recorded_passes.len() as u32) * 2;
        let end_q = start_q + 1;
        assert!(end_q < self.max_queries, "GpuProfiler query pool exhausted");

        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                start_q,
            );
        }
        let idx = self.recorded_passes.len() as u32;
        self.recorded_passes.push((name, start_q, end_q));
        idx
    }

    /// Record an end timestamp for `pass_idx`.
    pub fn end_pass(&self, device: &ash::Device, cmd: vk::CommandBuffer, pass_idx: u32) {
        let (_, _, end_q) = self.recorded_passes[pass_idx as usize];
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                end_q,
            );
        }
    }

    /// Read back all recorded pass timings after the command buffer fence has completed.
    pub fn readback(&self, device: &ash::Device) -> Result<Vec<PassTiming>, vk::Result> {
        if self.recorded_passes.is_empty() {
            return Ok(Vec::new());
        }
        let query_count = (self.recorded_passes.len() as u32) * 2;
        let mut results = vec![0u64; query_count as usize];
        unsafe {
            device.get_query_pool_results(
                self.query_pool,
                0,
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }

        let timings = self
            .recorded_passes
            .iter()
            .map(|&(name, s, e)| {
                let delta = results[e as usize].saturating_sub(results[s as usize]);
                let gpu_ms = (delta as f64 * self.timestamp_period_ns) / 1_000_000.0;
                PassTiming { name, gpu_ms }
            })
            .collect();
        Ok(timings)
    }

    /// Destroy the underlying `vk::QueryPool`.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.query_pool, None) };
    }
}
