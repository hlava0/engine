# Third-Party Notices

This project is licensed under **GPL-3.0-only** (`LICENSE`). When optional external SDKs are fetched locally via helper scripts, their respective open-source licenses apply:

## AMD FidelityFX SDK (FSR 3.1, v1.1.4)
- **Source**: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (`v1.1.4`)
- **License**: MIT License (compatible with GPL-3.0-only)
- **Integration**: Optional C/C++ shim in `crates/engine-core/ffi/` auto-detected by `crates/engine-core/build.rs` via `FSR31_SDK_DIR` or `.fsr31-sdk/` (fetch/build helper: `scripts/fetch_fsr31.sh`). When absent or when `ENGINE_DISABLE_FSR31=1` is set, the engine builds and runs with its built-in GPL-3.0 GLSL temporal upscaler (`shaders/fsr31_temporal_upscale.comp`).
