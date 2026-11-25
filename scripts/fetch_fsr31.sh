#!/usr/bin/env bash
# Fetches and builds the official AMD FidelityFX FSR 3.1 SDK (v1.1.4, MIT) for Linux Vulkan.
# Requires: git, curl, tar, g++, wine (for FidelityFX_SC.exe shader compiler).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${FSR31_SDK_DIR:-$ROOT_DIR/.fsr31-sdk}"
FFX_DIR="$SDK_ROOT/FidelityFX-SDK"
SHADERS_OUT="$SDK_ROOT/shaders"
LIB_OUT="$SDK_ROOT/lib"

mkdir -p "$SDK_ROOT" "$SHADERS_OUT" "$LIB_OUT"

if [ ! -d "$FFX_DIR/.git" ]; then
    echo "Cloning GPUOpen FidelityFX-SDK v1.1.4 (MIT)..."
    git clone --depth 1 --branch v1.1.4 \
        https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git \
        "$FFX_DIR"
fi

cat > "$SDK_ROOT/linux_compat.h" << 'EOF'
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <locale>
#include <codecvt>

#ifndef _countof
template <typename T, size_t N>
char (&_ffx_countof_helper(T (&)[N]))[N];
#define _countof(arr) sizeof(_ffx_countof_helper(arr))
#endif

inline int wcscpy_s(wchar_t* dst, size_t dst_count, const wchar_t* src) {
    if (!dst || !src || dst_count == 0) return 22;
    size_t i = 0;
    for (; i + 1 < dst_count && src[i] != L'\0'; ++i) dst[i] = src[i];
    dst[i] = L'\0';
    return 0;
}

template <size_t N>
inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) {
    return wcscpy_s(dst, N, src);
}

inline int strcpy_s(char* dst, size_t dst_count, const char* src) {
    if (!dst || !src || dst_count == 0) return 22;
    std::strncpy(dst, src, dst_count - 1);
    dst[dst_count - 1] = '\0';
    return 0;
}

template <size_t N>
inline int strcpy_s(char (&dst)[N], const char* src) {
    return strcpy_s(dst, N, src);
}

#ifndef sprintf_s
#define sprintf_s(buf, size, ...) std::snprintf((buf), (size), __VA_ARGS__)
#endif

#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))
#endif
EOF

SC_EXE="$FFX_DIR/sdk/tools/binary_store/FidelityFX_SC.exe"
SHADER_SRC="$FFX_DIR/sdk/src/backends/vk/shaders/fsr3upscaler"
SHADER_INC="$FFX_DIR/sdk/include/FidelityFX/gpu"

if ! command -v wine >/dev/null 2>&1; then
    echo "ERROR: 'wine' is required to run FidelityFX_SC.exe for shader permutation header generation." >&2
    exit 1
fi

echo "Compiling FSR 3.1 Vulkan GLSL shaders via FidelityFX_SC.exe..."
SHADERS=(
    "ffx_fsr3upscaler_prepare_inputs_pass.glsl:prepare_inputs"
    "ffx_fsr3upscaler_prepare_reactivity_pass.glsl:prepare_reactivity"
    "ffx_fsr3upscaler_shading_change_pass.glsl:shading_change"
    "ffx_fsr3upscaler_accumulate_pass.glsl:accumulate"
    "ffx_fsr3upscaler_rcas_pass.glsl:rcas"
    "ffx_fsr3upscaler_luma_pyramid_pass.glsl:luma_pyramid"
    "ffx_fsr3upscaler_autogenerate_reactivity_pass.glsl:generate_reactive"
    "ffx_fsr3upscaler_tcr_autogenerate_pass.glsl:tcr_autogenerate"
    "ffx_fsr3upscaler_shading_change_pyramid_pass.glsl:shading_change_pyramid"
    "ffx_fsr3upscaler_luma_instability_pass.glsl:luma_instability"
    "ffx_fsr3upscaler_debug_view_pass.glsl:debug_view"
)

for entry in "${SHADERS[@]}"; do
    file="${entry%%:*}"
    name="${entry##*:}"
    if [ ! -f "$SHADER_SRC/$file" ]; then
        continue
    fi
    WIN_INC="$(winepath -w "$SHADER_INC")"
    WIN_OUT="$(winepath -w "$SHADERS_OUT")"
    WIN_SRC="$(winepath -w "$SHADER_SRC/$file")"
    for variant in "" "_16bit" "_wave64" "_wave64_16bit"; do
        EXTRA_FLAGS=()
        if [[ "$variant" == *"16bit"* ]]; then
            EXTRA_FLAGS+=("-DFFX_HALF=1")
        fi
        if [[ "$variant" == *"wave64"* ]]; then
            EXTRA_FLAGS+=("-DFFX_WAVE=1")
        fi
        wine "$SC_EXE" -compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os \
            -DFFX_GLSL=1 "${EXTRA_FLAGS[@]}" \
            -I "$WIN_INC" \
            -output="$WIN_OUT" \
            -name="ffx_fsr3upscaler_${name}_pass${variant}" \
            "$WIN_SRC" >/dev/null
    done
done

echo "Compiling static library libffx_fsr3upscaler_vk.a..."
CXX_FLAGS=(
    -O2 -fno-tree-vectorize -fno-tree-slp-vectorize -std=c++17 -fshort-wchar
    -DFFX_GCC -DFFX_FSR3UPSCALER
    -include "$SDK_ROOT/linux_compat.h"
    "-I$FFX_DIR/sdk/include"
    "-I$FFX_DIR/sdk/src"
    "-I$FFX_DIR/sdk/src/shared"
    "-I$FFX_DIR/sdk/src/backends"
    "-I$FFX_DIR/sdk/src/backends/shared"
    "-I$SHADERS_OUT"
)

OBJS=()
SRCS=(
    "$FFX_DIR/sdk/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "$FFX_DIR/sdk/src/backends/vk/ffx_vk.cpp"
    "$FFX_DIR/sdk/src/backends/shared/ffx_shader_blobs.cpp"
    "$FFX_DIR/sdk/src/shared/ffx_assert.cpp"
    "$FFX_DIR/sdk/src/shared/ffx_message.cpp"
    "$FFX_DIR/sdk/src/shared/ffx_object_management.cpp"
    "$FFX_DIR/sdk/src/shared/ffx_breadcrumbs_list.cpp"
)

# Also include shaderblobs if present
if [ -f "$FFX_DIR/sdk/src/components/fsr3upscaler/ffx_fsr3upscaler_shaderblobs.cpp" ]; then
    SRCS+=("$FFX_DIR/sdk/src/components/fsr3upscaler/ffx_fsr3upscaler_shaderblobs.cpp")
fi

for src in "${SRCS[@]}"; do
    obj="$SDK_ROOT/$(basename "${src%.cpp}").o"
    g++ "${CXX_FLAGS[@]}" -c "$src" -o "$obj"
    OBJS+=("$obj")
done

rm -f "$LIB_OUT/libffx_fsr3upscaler_vk.a"
ar rcs "$LIB_OUT/libffx_fsr3upscaler_vk.a" "${OBJS[@]}"
echo "Done! Built $LIB_OUT/libffx_fsr3upscaler_vk.a"
