# GNU GPL v3.0 Section 7 Additional Permission — NVIDIA NRD Linking Exception

Copyright (c) 2026 hlava0 and contributors

This project is licensed under the **GNU General Public License, Version 3** (`GPL-3.0-only`; see [`LICENSE`](LICENSE)).

## Additional Permission under GNU GPL version 3 section 7

As a special exception under Section 7 of the GNU General Public License, Version 3, the copyright holders of this repository give you permission to link, combine, or compile the source code of this repository (in particular `crates/engine-rt-nrd` and `crates/engine-core`) with the proprietary **NVIDIA Real-Time Denoisers (NRD) SDK** (obtained separately from NVIDIA Corporation under the NVIDIA RTX SDKs License), and to distribute the resulting combined executable or library, **without** requiring the NVIDIA NRD SDK to be licensed under the GNU GPL or disclosed in source code form.

This exception applies **only** to the NVIDIA NRD SDK (and its accompanying ShaderMake blob library) and does not grant permission to link or combine this repository with any other proprietary software unless separately permitted.

If you modify this repository, you may extend this exception to your version of the repository, but you are not obligated to do so. If you do not wish to do so, delete this exception statement from your version (noting that you may then no longer distribute binaries linked with NVIDIA NRD under the NVIDIA RTX SDKs License).

## Scope by Crate

- **Pure `GPL-3.0-only` (no NRD dependency):**
  - `crates/engine-rhi`
  - `crates/engine-gpu-driven`
  - `crates/engine-compute-ml`
  - `shaders/` (except NRD-specific composition bindings)
- **`GPL-3.0-only` + Section 7 NVIDIA NRD Linking Exception:**
  - `crates/engine-rt-nrd`
  - `crates/engine-core`
