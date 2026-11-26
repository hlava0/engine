//! Fixed-timestep accumulator and rolling frame-time pacer/statistics.

/// Rolling frame-time and FPS smoothness statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct PacerStats {
    pub sample_count: usize,
    pub avg_fps: f64,
    pub avg_frame_ms: f64,
    pub p95_frame_ms: f64,
    pub p99_frame_ms: f64,
    pub one_percent_low_fps: f64,
    pub jitter_ms: f64,
}

/// Fixed-timestep accumulator with spiral-of-death clamping and rolling frame-time ring buffer.
pub struct FramePacer {
    fixed_dt: f64,
    max_frame_dt: f64,
    accumulator: f64,
    samples_ms: Vec<f64>,
    sample_cursor: usize,
    sample_count: usize,
    capacity: usize,
    scratch_sorted: Vec<f64>,
}

impl Default for FramePacer {
    fn default() -> Self {
        Self::new(1.0 / 60.0, 0.1, 120)
    }
}

impl FramePacer {
    /// Create a new `FramePacer` with `fixed_dt` (seconds), `max_frame_dt` clamp (seconds),
    /// and a rolling ring buffer of `history_size` samples.
    pub fn new(fixed_dt: f64, max_frame_dt: f64, history_size: usize) -> Self {
        let capacity = history_size.max(1);
        let fixed_dt = fixed_dt.max(1e-6);
        let max_frame_dt = max_frame_dt.max(fixed_dt);
        Self {
            fixed_dt,
            max_frame_dt,
            accumulator: 0.0,
            samples_ms: vec![0.0; capacity],
            sample_cursor: 0,
            sample_count: 0,
            capacity,
            scratch_sorted: Vec::with_capacity(capacity),
        }
    }

    /// Configured fixed simulation timestep in seconds.
    #[inline]
    pub fn fixed_dt(&self) -> f64 {
        self.fixed_dt
    }

    /// Configured maximum frame delta clamp in seconds.
    #[inline]
    pub fn max_frame_dt(&self) -> f64 {
        self.max_frame_dt
    }

    /// Record a raw frame duration (in seconds), clamp to `max_frame_dt` to prevent spiral-of-death,
    /// accumulate fixed timesteps, and return the number of fixed simulation steps to execute.
    pub fn begin_frame(&mut self, raw_dt_sec: f64) -> u32 {
        let clamped_raw = raw_dt_sec.max(0.0);
        self.record_sample_ms(clamped_raw * 1000.0);

        let sim_dt = clamped_raw.min(self.max_frame_dt);
        self.accumulator += sim_dt;

        let mut steps = 0u32;
        while self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            steps += 1;
        }
        steps
    }

    /// Record a single frame-time sample (in milliseconds) into the rolling ring buffer.
    #[inline]
    pub fn record_sample_ms(&mut self, frame_ms: f64) {
        self.samples_ms[self.sample_cursor] = frame_ms.max(0.0);
        self.sample_cursor = (self.sample_cursor + 1) % self.capacity;
        if self.sample_count < self.capacity {
            self.sample_count += 1;
        }
    }

    /// Interpolation alpha in `[0.0, 1.0]` representing fractional progress toward the next fixed step.
    #[inline]
    pub fn interpolation_alpha(&self) -> f64 {
        (self.accumulator / self.fixed_dt).clamp(0.0, 1.0)
    }

    /// Compute rolling FPS, average/percentile frame times, 1% low FPS, and jitter (stddev)
    /// without allocating on the heap.
    pub fn stats(&mut self) -> PacerStats {
        let n = self.sample_count;
        if n == 0 {
            return PacerStats::default();
        }

        let slice = &self.samples_ms[..n];
        let sum: f64 = slice.iter().sum();
        let avg_frame_ms = sum / n as f64;
        let avg_fps = if avg_frame_ms > 0.0 {
            1000.0 / avg_frame_ms
        } else {
            0.0
        };

        let variance = slice
            .iter()
            .map(|&x| {
                let d = x - avg_frame_ms;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let jitter_ms = variance.sqrt();

        self.scratch_sorted.clear();
        self.scratch_sorted.extend_from_slice(slice);
        self.scratch_sorted.sort_by(f64::total_cmp);

        let p95_idx = ((n as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let p99_idx = ((n as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        let p95_frame_ms = self.scratch_sorted[p95_idx];
        let p99_frame_ms = self.scratch_sorted[p99_idx];

        let slowest_count = (n / 100).max(1);
        let slowest_avg_ms =
            self.scratch_sorted[n - slowest_count..].iter().sum::<f64>() / slowest_count as f64;
        let one_percent_low_fps = if slowest_avg_ms > 0.0 {
            1000.0 / slowest_avg_ms
        } else {
            0.0
        };

        PacerStats {
            sample_count: n,
            avg_fps,
            avg_frame_ms,
            p95_frame_ms,
            p99_frame_ms,
            one_percent_low_fps,
            jitter_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_timestep_accumulator_and_spiral_clamp() {
        let mut pacer = FramePacer::default();
        // Half a 60 Hz frame -> 0 steps, alpha ~ 0.5
        assert_eq!(pacer.begin_frame(1.0 / 120.0), 0);
        assert!((pacer.interpolation_alpha() - 0.5).abs() < 1e-6);

        // Another half frame -> 1 step, alpha ~ 0.0
        assert_eq!(pacer.begin_frame(1.0 / 120.0), 1);
        assert!(pacer.interpolation_alpha() < 1e-6);

        // Huge hitch (2.0s) clamped to max_frame_dt (0.1s) -> 6 steps at 60 Hz
        let steps = pacer.begin_frame(2.0);
        assert_eq!(steps, 6);
    }

    #[test]
    fn rolling_stats_percentiles_and_jitter() {
        let mut pacer = FramePacer::new(1.0 / 60.0, 0.1, 100);
        for i in 1..=100 {
            pacer.record_sample_ms(i as f64);
        }
        let s = pacer.stats();
        assert_eq!(s.sample_count, 100);
        assert!((s.avg_frame_ms - 50.5).abs() < 1e-6);
        assert!((s.p95_frame_ms - 95.0).abs() < 1e-6);
        assert!((s.p99_frame_ms - 99.0).abs() < 1e-6);
        assert!((s.one_percent_low_fps - 10.0).abs() < 1e-6);
        assert!(s.jitter_ms > 28.0 && s.jitter_ms < 29.5);
    }
}
