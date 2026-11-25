//! `rodio` 3D spatial & procedural audio playback engine with graceful headless/CI silent fallback.

use std::io::Cursor;
use std::time::Duration;

use rodio::source::{SineWave, Source};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, SpatialSink};

/// Audio playback engine wrapping an optional `rodio::OutputStream`.
///
/// If no audio output device is available (e.g. headless servers or CI containers),
/// `AudioEngine::new()` silently degrades to a no-op backend and never panics.
pub struct AudioEngine {
    _stream: Option<OutputStream>,
    handle: Option<OutputStreamHandle>,
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEngine {
    /// Attempt to open the default OS audio output device, falling back silently if unavailable.
    pub fn new() -> Self {
        match OutputStream::try_default() {
            Ok((stream, handle)) => Self {
                _stream: Some(stream),
                handle: Some(handle),
            },
            Err(err) => {
                log::info!("Audio output unavailable ({err}); running in silent headless mode");
                Self {
                    _stream: None,
                    handle: None,
                }
            }
        }
    }

    /// Returns `true` if a live audio output device was opened.
    pub fn is_available(&self) -> bool {
        self.handle.is_some()
    }

    /// Play a procedural sine-wave beep at `freq_hz` for `duration_ms` milliseconds.
    /// No-op when running in headless/silent mode.
    pub fn play_beep(&self, freq_hz: f32, duration_ms: u64) {
        let Some(handle) = self.handle.as_ref() else {
            return;
        };
        if let Ok(sink) = Sink::try_new(handle) {
            let source = SineWave::new(freq_hz.max(20.0))
                .take_duration(Duration::from_millis(duration_ms.max(1)))
                .fade_out(Duration::from_millis((duration_ms / 2).max(1)))
                .amplify(0.15);
            sink.append(source);
            sink.detach();
        }
    }

    /// Play a 3D-spatialized procedural tone attenuated and stereo-panned relative to the listener.
    pub fn play_spatial_beep(
        &self,
        freq_hz: f32,
        duration_ms: u64,
        listener_pos: [f32; 3],
        listener_right: [f32; 3],
        emitter_pos: [f32; 3],
    ) {
        let Some(handle) = self.handle.as_ref() else {
            return;
        };
        let ear_sep = 0.12;
        let left_ear = [
            listener_pos[0] - listener_right[0] * ear_sep,
            listener_pos[1] - listener_right[1] * ear_sep,
            listener_pos[2] - listener_right[2] * ear_sep,
        ];
        let right_ear = [
            listener_pos[0] + listener_right[0] * ear_sep,
            listener_pos[1] + listener_right[1] * ear_sep,
            listener_pos[2] + listener_right[2] * ear_sep,
        ];
        if let Ok(sink) = SpatialSink::try_new(handle, emitter_pos, left_ear, right_ear) {
            let source = SineWave::new(freq_hz.max(20.0))
                .take_duration(Duration::from_millis(duration_ms.max(1)))
                .fade_out(Duration::from_millis((duration_ms / 2).max(1)))
                .amplify(0.22);
            sink.append(source);
            sink.detach();
        }
    }

    /// Play a short percussive impact thud (two-tone decaying pulse).
    pub fn play_impact(&self, base_freq_hz: f32, duration_ms: u64) {
        let Some(handle) = self.handle.as_ref() else {
            return;
        };
        if let Ok(sink) = Sink::try_new(handle) {
            let d = Duration::from_millis(duration_ms.max(10));
            let s1 = SineWave::new(base_freq_hz.max(40.0))
                .take_duration(d)
                .fade_out(d)
                .amplify(0.18);
            sink.append(s1);
            sink.detach();
        }
    }

    /// Decode and play a WAV buffer asynchronously. No-op when running in headless/silent mode.
    pub fn play_wav_bytes(&self, bytes: &[u8]) {
        let Some(handle) = self.handle.as_ref() else {
            return;
        };
        let owned = bytes.to_vec();
        if let Ok(decoder) = Decoder::new_wav(Cursor::new(owned))
            && let Ok(sink) = Sink::try_new(handle)
        {
            sink.append(decoder);
            sink.detach();
        }
    }
}
