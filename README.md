# vulkan-gpu-driven-engine

Lean, modular, code-first 3D rendering engine in **Rust** targeting **Vulkan 1.4** natively via `ash` and `gpu-allocator`. Designed to minimize per-frame CPU overhead by offloading culling, indirect draw generation, ray traversal, denoising, and post-processing to the GPU.

## Architecture

- `crates/engine-rhi`: Vulkan 1.4 instance/device bootstrap (Dynamic Rendering, Synchronization2, Buffer Device Address, Descriptor Indexing, Hardware Ray Tracing / Ray Query, Cooperative Matrix probing), `gpu-allocator` integration, BDA buffer helpers, bindless descriptor table, and `build.rs` SPIR-V compilation.
- `crates/engine-gpu-driven`: GPU scene buffers, two-phase compute frustum + Hi-Z occlusion culling, and `vkCmdDrawIndexedIndirectCount` G-buffer pass with jittered motion vectors.
- `crates/engine-rt-nrd`: BLAS/TLAS acceleration structure builder/refitter, ray-query GI pass, NVIDIA NRD (`REBLUR`/`RELAX`/`SIGMA`) denoiser integration, and material demodulation/remodulation composition pass.
- `crates/engine-compute-ml`: `VK_KHR_cooperative_matrix` compute pipeline abstraction (runtime tile-shape probing with subgroup fallback) for neural post-processing.
- `crates/engine-core`: Lightweight data-oriented scene representation, procedural test scenes, GPU timestamp profiler (`VK_QUERY_TYPE_TIMESTAMP`), and `headless_bench` CLI harness.
- `shaders/`: GLSL compute, vertex, and fragment shaders compiled to SPIR-V at build time.

## Build Requirements

- **Rust**: `rustc` and `cargo` (Edition 2024)
- **Vulkan 1.4 SDK**: Vulkan loader, validation layers (`VK_LAYER_KHRONOS_validation`), and `glslangValidator` in `PATH`
- **NVIDIA NRD**: User-supplied local installation of the [NVIDIA Real-Time Denoisers (NRD) SDK](https://github.com/NVIDIA-RTX/NRD) (not bundled in this repository; governed by the NVIDIA RTX SDKs License). You can either build it manually with SPIR-V enabled or run the opt-in helper `./scripts/fetch_nrd.sh --accept-nvidia-license`, then set `NRD_SDK_DIR` to the directory containing `Include/` and `lib/` (`libNRD.a`, `libShaderMakeBlob.a`).

## Verification

```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo run -p engine-core --bin headless_bench
```

## License & Third-Party Notices

This project is licensed under the **GNU General Public License v3.0 (`GPL-3.0-only`)** — see [`LICENSE`](LICENSE).

- **Pure `GPL-3.0-only`**: `crates/engine-rhi`, `crates/engine-gpu-driven`, `crates/engine-compute-ml`.
- **`GPL-3.0-only` with GPL-3.0 Section 7 NVIDIA NRD Linking Exception**: `crates/engine-rt-nrd` and `crates/engine-core` — see [`LICENSE-NRD-EXCEPTION.md`](LICENSE-NRD-EXCEPTION.md). This allows linking against a user-supplied external NVIDIA NRD SDK without violating Section 4(e) of the NVIDIA RTX SDKs License.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full third-party license compliance details.
