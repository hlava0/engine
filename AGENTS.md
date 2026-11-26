# Project: GPU-Driven Vulkan 1.4 Engine (Rust)

## 1. Objective & Hardware Philosophy
Build a lean, modular, code-first 3D rendering and game engine in **Rust** targeting **Vulkan 1.4** natively, integrating and improving the existing ray-tracing and NRD codebase located at `/home/viktor/ray_tracing_denoising_code`.

- **CPU-Bottleneck Mitigation (Primary Constraint):** The host system is CPU-bound. Offload work aggressively from the CPU to the GPU. Per-frame CPU work must stay minimal (updating camera/frame uniform buffers and recording a handful of indirect/compute commands). Frustum/Hi-Z occlusion culling, LOD selection, draw generation, ray traversal, and matrix math must execute on the GPU.
- **Unrestricted Vulkan 1.4 Access:** Do NOT use `wgpu` or wrappers that hide KHR/vendor extensions. Use **`ash`** (1:1 Vulkan 1.4 bindings) + **`gpu-allocator`** so every core 1.4 feature and extension is directly exposed:
  - Dynamic Rendering (`vkCmdBeginRendering` — zero legacy `VkRenderPass`/`VkFramebuffer`)
  - Synchronization2 (`vkCmdPipelineBarrier2`)
  - Buffer Device Address (`VK_KHR_buffer_device_address` — pass 64-bit GPU pointers in push constants)
  - Bindless Descriptor Indexing / Descriptor Buffers (`VK_EXT_descriptor_indexing` / `VK_EXT_descriptor_buffer`)
  - Hardware Ray Tracing (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_query`)
  - Cooperative Matrices (`VK_KHR_cooperative_matrix` — expose query & dispatch plumbing for hardware matrix units such as Intel XMX for neural upscaling/post-processing)
- **Zero Hidden Editor State:** 100% of scene graphs, materials, shaders, and pipeline configurations live in version-controlled code and plain-text files so AI agents have complete visibility.

---

## 2. Antigravity Model Routing, Effort & Quota Protocol
Strictly follow this four-tier escalation chain to minimize token burn and avoid expensive models:

1. **Tier 1 — `Neon` (Orchestrator Mode):**
   - Owns system architecture, existing-code audit (`/home/viktor/ray_tracing_denoising_code`), crate boundaries, task decomposition, dependency ordering, and verification criteria.
   - Breaks work into small, self-contained tasks and dispatches them to `Hydrogen` with an explicit reasoning **effort level**:
     - `low`: Cargo workspace setup, structs, math helpers, boilerplate wrappers, Git/GitHub publishing steps.
     - `medium`: Descriptor layouts, buffer management, asset loaders, scene ECS wiring, targeted refactors.
     - `high`: Vulkan 1.4 Synchronization2 barriers, BDA shader interfaces, indirect compute culling, BLAS/TLAS builders, NRD pipeline fixes, `VK_KHR_cooperative_matrix` kernels.
     - **`xhigh` Restriction (STRICT):** Do **not** assign `xhigh` for debugging, incremental edits, or tasks under 500 lines. Only use `xhigh` when:
       1. Generating **at least 500 lines of cohesive, new code in a single pass** (e.g., a complete subsystem or shader suite), **OR**
       2. Authoring **complex/large benchmark scenes** *after* the core engine milestones are finished and verified.
2. **Tier 2 — `Hydrogen` (Primary Implementer):**
   - Executes all code generation, refactoring of `/home/viktor/ray_tracing_denoising_code`, shader authoring (GLSL/HLSL -> SPIR-V), and validation loops at the effort level specified by `Neon`.
   - Must self-correct using `cargo check`, `cargo test`, shader compiler diagnostics, and Vulkan validation logs before reporting back to `Neon`.
3. **Tier 3 — `Neon` (Non-Orchestrator / Direct Implementation Mode):**
   - **Trigger:** If `Hydrogen` fails to resolve a blocker after **3 consecutive logged attempts** (e.g., a Vulkan Synchronization2 hazard, NRD packing/reprojection bug, or Rust lifetime/FFI issue), switch `Neon` from orchestrator mode into **direct implementation/debugging mode** to inspect the isolated failure, write the fix directly, and verify it passes—**before** ever considering `Carbon`.
4. **Tier 4 — `Carbon` (Last-Resort Escalation — USE RARELY):**
   - `Carbon` is expensive and strictly prohibited for routine coding, refactors, or first-pass debugging.
   - **Trigger:** Invoke `Carbon` *only* after both `Hydrogen` (3 attempts) and `Neon` in direct non-orchestrator mode fail to resolve a specific, reproducible blocker.
   - **Rule:** Pass `Carbon` only the minimal isolated reproduction, exact compiler/validation output, and affected code slice. Require a concise root-cause patch, then immediately return control to `Neon`/`Hydrogen`.

---

## 3. Existing Code Integration (`/home/viktor/ray_tracing_denoising_code`)
Do **not** rewrite the ray tracer or NRD integration from scratch. First audit `/home/viktor/ray_tracing_denoising_code` and migrate/upgrade it into `crates/engine-rt-nrd`:

1. **Audit & Bottleneck Identification (`Neon`):**
   - Inspect all Rust/C++ files, shaders, NRD bindings (`REBLUR`/`RELAX`/`SIGMA`), resource pools, and synchronization logic in `/home/viktor/ray_tracing_denoising_code`.
   - Identify CPU bottlenecks (per-frame descriptor allocations, redundant staging uploads, synchronous waits), Vulkan validation/barrier hazards, and NRD signal-quality issues.
2. **Targeted Improvements (`Hydrogen`):**
   - Upgrade resource binding to Buffer Device Address (`BDA`) + bindless tables and replace legacy barriers with `Synchronization2`.
   - Verify NRD front-end/back-end shader contracts (`NRD.hlsli` / `NRDDescs.h`): proper diffuse/specular albedo demodulation (`NRD_MaterialFactors`), normalized hit-distance trimming (`REBLUR_FrontEnd_GetNormHitDist` / `NRD_FrontEnd_TrimHitDistance`), jittered motion vectors, and gradient-based lighting confidence inputs (`IN_DIFF_CONFIDENCE` / `IN_SPEC_CONFIDENCE`) to eliminate temporal ghosting and fireflies.
   - Isolate NRD behind a clean, reusable Rust module so denoising runs as a self-contained compute pass graph over G-buffer + RT radiance inputs.

---

## 4. Workspace Architecture
Organize the repository as a modular Cargo workspace for fast incremental builds and small AI context windows:

- `crates/engine-rhi`: Vulkan 1.4 instance/device bootstrap, physical device extension/property probing (`VK_KHR_cooperative_matrix`, RT pipeline/ray-query properties), `gpu-allocator` wrapper, BDA buffer helpers, bindless descriptor table, and `build.rs` SPIR-V compilation/hot-reload.
- `crates/engine-gpu-driven`: GPU scene buffer (transforms, bounding spheres/AABBs, mesh/draw metadata), two-phase compute frustum + Hi-Z occlusion culling shaders, and `vkCmdDrawIndexedIndirectCount` submission.
- `crates/engine-rt-nrd`: Refactored and improved ray-tracing + NVIDIA NRD denoiser pipeline migrated from `/home/viktor/ray_tracing_denoising_code` (`BLAS`/`TLAS` builder/refitter, RT/ray-query pass, NRD resource pool & compute pass dispatcher, and material demodulation/remodulation composition pass).
- `crates/engine-compute-ml`: `VK_KHR_cooperative_matrix` compute pipeline abstraction (querying supported M×N×K tile shapes at runtime with subgroup fallback) for neural post-processing or upscaling.
- `crates/engine-core`: Lightweight data-oriented scene/ECS, fixed-timestep loop, GPU timestamp profiler (`VK_QUERY_TYPE_TIMESTAMP`) verifying sub-millisecond CPU frame recording, and a headless/CLI debug & scene-generation harness.
- `shaders/`: GLSL/HLSL compute, vertex/fragment, and ray-tracing shaders compiled to SPIR-V with debug symbols via `build.rs`.

---

## 5. Mandatory Verification Loop
No milestone or task is complete until verified:
1. `cargo check --workspace` and `cargo clippy --workspace` pass with zero errors.
2. All shaders compile cleanly to SPIR-V during `cargo build` with layout assertions matching Rust `#[repr(C)]` BDA structs.
3. Headless validation test (`VK_LAYER_KHRONOS_validation` + Synchronization Validation enabled) executes a test frame through the GPU-driven culling, RT, and NRD passes with **zero** validation errors or sync hazards.
4. GPU timestamp queries confirm CPU frame-recording overhead remains minimal while culling, RT, and denoising execute on the GPU.

---

## 6. Post-Completion GitHub Repository Creation (`low` effort -> `Hydrogen`)
Once **all milestones and verification checks pass**, create a new GitHub repository on the authenticated account using the token from `gh auth token` and push the completed project:

1. **Clean Repository Hygiene:**
   - Ensure `.gitignore` ignores `/target`, compiled `.spv` build outputs, and any local IDE/cache artifacts, and include a clear `README.md` documenting architecture, build requirements, and controls.
   - **Security Rule:** Never print, echo, log, write to disk, or embed the raw token in `.git/config` remote URLs. Pass it strictly in-memory via `GH_TOKEN` at execution time (calling `gh auth token` right before pushing so short-lived CLI tokens remain valid).
2. **Initialize, Create Repo & Push:**
   ```bash
   git init -b main
   git add .
   git commit -m "Initial commit: GPU-driven Vulkan 1.4 engine with RT and NRD"
   GH_TOKEN="$(gh auth token)" gh repo create vulkan-gpu-driven-engine --private --source=. --remote=origin --push

(If git push requires explicit HTTPS header auth without gh credential helper, pass the token transiently via -c http.extraheader rather than storing it in the remote URL, then confirm git status and the remote URL are clean.)
7. Execution Plan (Start Here)

Neon:

    Audit Phase (Orchestrator Mode): Inspect /home/viktor/ray_tracing_denoising_code and the local environment (rustc, cargo, Vulkan SDK / dxc / glslangValidator, vulkaninfo, gh auth status), then produce a concise audit report listing what to reuse, what to fix, and the exact task breakdown.

    Milestone 1 (medium effort -> Hydrogen): Initialize the Cargo workspace and crates/engine-rhi (Vulkan 1.4 device init with BDA, Dynamic Rendering, Synchronization2, RT, and Cooperative Matrix probing + headless validation test).

    Milestone 2 (high effort -> Hydrogen): Implement bindless BDA buffers and GPU-driven compute frustum/Hi-Z culling + vkCmdDrawIndexedIndirectCount geometry pass outputting G-buffer + jittered motion vectors.

    Milestone 3 (high or xhigh if >=500 lines single-shot -> Hydrogen): Migrate, refactor, and improve /home/viktor/ray_tracing_denoising_code into crates/engine-rt-nrd, fixing barrier hazards, BDA bindings, hit-distance packing, and confidence masks.

    Milestone 4 (high effort -> Hydrogen): Implement crates/engine-compute-ml (VK_KHR_cooperative_matrix) and crates/engine-core GPU timestamp profiling + procedural/complex test scenes (xhigh allowed for complex post-engine scene authoring).

    Milestone 5 (low effort -> Hydrogen): Create the GitHub repository via GH_TOKEN="$(gh auth token)" gh repo create ... --push and verify the pushed repository URL.

