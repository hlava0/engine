//! `rodio` audio playback engine with graceful headless/CI silent fallback.

use std::io::Cursor;
use std::time::Duration;

use rodio::source::{SineWave, Source};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};

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
                .amplify(0.15);
            sink.append(source);
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
