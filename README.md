# vulkan-gpu-driven-engine

Lean, modular, code-first 3D rendering and gameplay engine in **Rust** targeting **Vulkan 1.3 / 1.4** natively via `ash` and `gpu-allocator`. Designed to minimize per-frame CPU overhead by offloading culling, indirect draw generation, ray traversal, denoising, and post-processing to the GPU.

## Architecture

- `crates/engine-rhi`: Vulkan 1.3 / 1.4 instance/device bootstrap (Dynamic Rendering, Synchronization2, Buffer Device Address, Descriptor Indexing, Hardware Ray Tracing / Ray Query, Cooperative Matrix probing), `gpu-allocator` integration, BDA buffer helpers, bindless descriptor table, `VK_KHR_swapchain` presentation, persistent `VkPipelineCache`, and `build.rs` SPIR-V compilation.
- `crates/engine-gpu-driven`: GPU scene buffers, two-phase compute frustum + Hi-Z occlusion culling, and `vkCmdDrawIndexedIndirectCount` G-buffer pass with jittered motion vectors.
- `crates/engine-rt-nrd`: BLAS/TLAS acceleration structure builder/refitter, ray-query GI pass, optional NVIDIA NRD (`REBLUR`) denoiser with a built-in pure GPL-3.0 SVGF spatiotemporal fallback, and material demodulation/remodulation composition pass.
- `crates/engine-compute-ml`: `VK_KHR_cooperative_matrix` compute pipeline abstraction (runtime tile-shape probing with subgroup fallback) for neural post-processing.
- `crates/engine-assets`: glTF 2.0 / GLB scene loader (hierarchical transforms, PBR metallic-roughness + emissive materials, automatic flat normal generation) and `image`-backed texture decoding with GPU staging upload into `BindlessTable`.
- `crates/engine-gameplay`: `hecs` ECS (`GameWorld`), deterministic 3D rigid-body physics (semi-implicit Euler, ground-plane & sphere-sphere impulse resolution), `rodio` audio with silent headless fallback, `rhai` scripting runtime with AST caching, and runtime `SceneEditor` inspector.
- `crates/engine-core`: Data-oriented `SceneWorld`, `FramePacer` + 2-frames-in-flight async pipelining, GPU timestamp profiler (`VK_QUERY_TYPE_TIMESTAMP`), windowed swapchain presentation, `headless_bench` CLI benchmark, and interactive `viewer` binary.
- `shaders/`: GLSL compute, vertex, and fragment shaders compiled to SPIR-V at build time.

## Build Requirements

- **Rust**: `rustc` and `cargo` (Edition 2024)
- **Vulkan 1.3 / 1.4 SDK**: Vulkan loader, validation layers (`VK_LAYER_KHRONOS_validation`), and `glslangValidator` in `PATH`
- **NVIDIA NRD (Optional)**: Built-in GPL-3.0 SVGF denoising works out of the box with zero proprietary dependencies (or force it via `ENGINE_DISABLE_NRD=1`). To enable NVIDIA NRD (`REBLUR`), supply a local installation of the [NVIDIA Real-Time Denoisers (NRD) SDK](https://github.com/NVIDIA-RTX/NRD) (governed by the NVIDIA RTX SDKs License) — either build it manually with SPIR-V enabled or run `./scripts/fetch_nrd.sh --accept-nvidia-license`, then set `NRD_SDK_DIR` to the directory containing `Include/` and `lib/` (`libNRD.a`, `libShaderMakeBlob.a`).

## Archery Range Game

Run the playable 3D Archery Range game with walking, jumping, bow charging, projectile arrow physics, target scoring, audio chimes, and an in-game Graphics Settings Menu:

```bash
cargo run --release -p engine-core --bin archery_game
```

### Archery Game Controls

| Input | Action |
|---|---|
| `W` / `A` / `S` / `D` | Walk |
| `Space` / `Shift` | Jump / Sprint |
| Click window / `Tab` | Lock or unlock mouse look (Arrow keys also look) |
| Hold `Left Mouse` or `F` | Draw bow & release to shoot arrow |
| `M` | Toggle in-game Graphics & Game Menu (`Up/Down` select, `Left/Right` change) |
| `Esc` | Close menu / unlock cursor / quit |

## Interactive Viewer

Run the windowed interactive viewer & ECS/physics/scripting sandbox:

```bash
cargo run -p engine-core --bin viewer
# Or load a glTF 2.0 / GLB scene:
cargo run -p engine-core --bin viewer -- --gltf path/to/model.glb
```

### Viewer Controls

| Input | Action |
|---|---|
| `W` / `A` / `S` / `D` | Move camera forward / left / back / right |
| `Space` / `Shift` | Move camera up / down |
| Arrow keys or Left-Mouse drag | Look around (yaw / pitch) |
| `P` | Pause / resume physics & scripts |
| `Tab` | Cycle `SceneEditor` selected entity |
| `I` / `K` / `J` / `L` | Nudge selected entity (`-Z` / `+Z` / `-X` / `+X`) |
| `Esc` | Quit |

## Verification

```bash
cargo check --workspace
ENGINE_DISABLE_NRD=1 cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --nocapture
cargo run -p engine-core --bin headless_bench
```

## License & Third-Party Notices

This project is licensed under the **GNU General Public License v3.0 (`GPL-3.0-only`)** — see [`LICENSE`](LICENSE).

- **Pure `GPL-3.0-only`**: `crates/engine-rhi`, `crates/engine-gpu-driven`, `crates/engine-compute-ml`, `crates/engine-assets`, `crates/engine-gameplay`.
- **`GPL-3.0-only` with GPL-3.0 Section 7 NVIDIA NRD Linking Exception**: `crates/engine-rt-nrd` and `crates/engine-core` — see [`LICENSE-NRD-EXCEPTION.md`](LICENSE-NRD-EXCEPTION.md). This allows linking against a user-supplied external NVIDIA NRD SDK without violating Section 4(e) of the NVIDIA RTX SDKs License.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full third-party license compliance details.
