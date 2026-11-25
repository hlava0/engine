# Third-Party Notices & License Compliance

This repository (`hlava0/engine`) is licensed under the **GNU General Public License, Version 3 (`GPL-3.0-only`)** — see [`LICENSE`](LICENSE) — with a **GPL-3.0 Section 7 Additional Permission** for linking the external NVIDIA NRD SDK — see [`LICENSE-NRD-EXCEPTION.md`](LICENSE-NRD-EXCEPTION.md).

## External Optional Proprietary Dependency: NVIDIA Real-Time Denoisers (NRD) SDK

This repository does **not** include, vendor, or redistribute any files from the NVIDIA Real-Time Denoisers (NRD) SDK (headers, shaders, or static libraries).

- `crates/engine-rt-nrd` includes a built-in pure GPL-3.0 SVGF spatiotemporal denoiser by default, and optionally links against a user-supplied local installation of the NVIDIA NRD SDK configured via the `NRD_SDK_DIR` environment variable (can also be disabled via `ENGINE_DISABLE_NRD=1`).
- The NVIDIA NRD SDK is proprietary software provided by NVIDIA Corporation under the **NVIDIA RTX SDKs License** and is **not** subject to the GNU GPL v3.0.
- Under [`LICENSE-NRD-EXCEPTION.md`](LICENSE-NRD-EXCEPTION.md) (GPL-3.0 Section 7), linking with NVIDIA NRD does **not** require NRD to be disclosed in source form or licensed under the GPL, preserving compliance with Section 4(e) of the NVIDIA RTX SDKs License.
- Users building `engine-rt-nrd` or `engine-core` with NRD enabled must obtain the NVIDIA NRD SDK separately from NVIDIA and comply with all terms of the NVIDIA RTX SDKs License (including no implied endorsement by NVIDIA).

## External Optional Open-Source Dependency: AMD FidelityFX SDK (FSR 3.1, v1.1.4 — MIT License)

This repository does **not** vendor or redistribute AMD FidelityFX SDK binaries or headers directly.

- `crates/engine-core` includes a built-in pure GPL-3.0 GLSL temporal upscaler (`shaders/fsr31_temporal_upscale.comp`), and optionally links against a local build of the **AMD FidelityFX SDK v1.1.4** (`https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`) configured via `FSR31_SDK_DIR` or `.fsr31-sdk/` (fetched/built locally via `scripts/fetch_fsr31.sh`, or disabled via `ENGINE_DISABLE_FSR31=1`).
- The AMD FidelityFX SDK is licensed under the **MIT License**, which is compatible with GNU GPL v3.0.

## Architectural Reference: Godot Engine (MIT License — GPL-3.0 Compatible)

Portions of the ray-tracing and NRD host integration design were informed by and migrated from a Godot 4 integration prototype:

> Copyright (c) 2014-present Godot Engine contributors.
> Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
> Licensed under the MIT License (https://godotengine.org/license), which is compatible with GNU GPL v3.0.

## Rust Crate Dependencies (All GPL-3.0 Compatible)

All Cargo dependencies resolved in `Cargo.lock` use open-source licenses compatible with **GNU GPL v3.0**:

- `ash` / `ash-window` — `MIT OR Apache-2.0`
- `raw-window-handle` — `MIT OR Apache-2.0 OR Zlib`
- `winit` — `Apache-2.0`
- `gpu-allocator` — `MIT OR Apache-2.0`
- `bytemuck` / `bytemuck_derive` — `Zlib OR Apache-2.0 OR MIT`
- `gltf` — `MIT OR Apache-2.0`
- `image` — `MIT OR Apache-2.0`
- `hecs` — `MIT OR Apache-2.0`
- `rhai` — `MIT OR Apache-2.0` (uses `smartstring` under `MPL-2.0+`, which is GPL-3.0 compatible)
- `rodio` — `MIT OR Apache-2.0`
- `thiserror` — `MIT OR Apache-2.0`
- `log` / `env_logger` — `MIT OR Apache-2.0`
- `cc` — `MIT OR Apache-2.0`
- `libloading` — `ISC`
- Transitive crates: `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `Unlicense`, `ISC`, `Zlib`, `0BSD`, `CC0-1.0`, `MPL-2.0+`, `LGPL-2.1-or-later`, `Unicode-3.0`.
