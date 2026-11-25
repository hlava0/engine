use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_fsr31)");
    println!("cargo:rerun-if-env-changed=FSR31_SDK_DIR");
    println!("cargo:rerun-if-env-changed=ENGINE_DISABLE_FSR31");
    println!("cargo:rerun-if-changed=ffi/fsr31_shim.h");
    println!("cargo:rerun-if-changed=ffi/fsr31_shim.cpp");

    if env::var("ENGINE_DISABLE_FSR31").as_deref() == Ok("1") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk_root = env::var("FSR31_SDK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../.fsr31-sdk"));
    let sdk_root = sdk_root.canonicalize().unwrap_or(sdk_root);
    let lib_dir = sdk_root.join("lib");
    let lib_file = lib_dir.join("libffx_fsr3upscaler_vk.a");
    let ffx_include = sdk_root.join("FidelityFX-SDK/sdk/include");
    let compat_h = sdk_root.join("linux_compat.h");

    if !lib_file.exists() || !ffx_include.exists() {
        println!(
            "cargo:warning=AMD FidelityFX FSR 3.1 SDK not found at '{}'; using built-in GLSL temporal upscaler.",
            sdk_root.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed={}", lib_file.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .flag("-fshort-wchar")
        .define("FFX_GCC", None)
        .define("FFX_FSR3UPSCALER", None)
        .include("ffi")
        .include(&ffx_include);
    if compat_h.exists() {
        build.flag("-include").flag(compat_h.to_string_lossy().as_ref());
    }
    build.file("ffi/fsr31_shim.cpp").compile("fsr31_shim");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ffx_fsr3upscaler_vk");
    println!("cargo:rustc-link-lib=vulkan");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo::rustc-cfg=has_fsr31");
}
