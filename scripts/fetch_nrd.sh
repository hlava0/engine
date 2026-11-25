#!/usr/bin/env bash
set -euo pipefail

# Opt-in helper to download and build the NVIDIA Real-Time Denoisers (NRD) SDK locally.
# NVIDIA NRD is proprietary software governed by the NVIDIA RTX SDKs License and is NOT
# included in this repository. Users must review and accept NVIDIA's license terms before
# downloading or building it.

NRD_URL="https://github.com/NVIDIA-RTX/NRD.git"
DEST_DIR="${1:-$(pwd)/.nrd-sdk}"

if [[ "${1:-}" == "--accept-nvidia-license" ]]; then
    DEST_DIR="${2:-$(pwd)/.nrd-sdk}"
elif [[ "${2:-}" == "--accept-nvidia-license" ]]; then
    DEST_DIR="$1"
else
    echo "NVIDIA Real-Time Denoisers (NRD) SDK is proprietary software provided by NVIDIA"
    echo "under the NVIDIA RTX SDKs License: https://github.com/NVIDIA-RTX/NRD/blob/main/LICENSE.txt"
    echo ""
    echo "To confirm you have read and accept the NVIDIA RTX SDKs License and wish to"
    echo "clone and build NRD locally, re-run with:"
    echo "  $0 --accept-nvidia-license [target_dir]"
    exit 1
fi

echo "Cloning NVIDIA NRD into ${DEST_DIR}..."
if [[ ! -d "${DEST_DIR}/.git" ]]; then
    git clone --depth 1 --recursive "${NRD_URL}" "${DEST_DIR}"
fi

cmake -S "${DEST_DIR}" -B "${DEST_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNRD_STATIC_LIBRARY=ON \
    -DNRD_SHADERS_SPIRV=ON
cmake --build "${DEST_DIR}/build" --config Release -j "$(nproc)"

mkdir -p "${DEST_DIR}/lib"
find "${DEST_DIR}/build" -maxdepth 4 \( -name "libNRD.a" -o -name "libShaderMakeBlob.a" \) -exec cp -v {} "${DEST_DIR}/lib/" \;

echo ""
echo "NVIDIA NRD built successfully. Export before building the workspace:"
echo "  export NRD_SDK_DIR=\"${DEST_DIR}\""
