use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_nrd)");
    println!("cargo:rerun-if-env-changed=NRD_SDK_DIR");
    println!("cargo:rerun-if-env-changed=ENGINE_DISABLE_NRD");

    let disable_nrd = env::var("ENGINE_DISABLE_NRD")
        .ok()
        .is_some_and(|v| v == "1");
    if disable_nrd {
        println!(
            "cargo:warning=ENGINE_DISABLE_NRD=1; building with built-in GPL-3.0 SVGF denoiser fallback."
        );
        return;
    }

    let nrd_root = env::var("NRD_SDK_DIR").unwrap_or_else(|_| {
        "/home/viktor/ray_tracing_denoising_code/engine/thirdparty/nrd".to_string()
    });
    let nrd_include = PathBuf::from(&nrd_root).join("Include");
    let nrd_lib = PathBuf::from(&nrd_root).join("lib");
    let lib_nrd = nrd_lib.join("libNRD.a");

    if !nrd_include.exists() || !lib_nrd.exists() {
        println!(
            "cargo:warning=NVIDIA NRD SDK not found; building with built-in GPL-3.0 SVGF denoiser fallback."
        );
        return;
    }

    println!("cargo:rerun-if-changed=ffi/nrd_shim.h");
    println!("cargo:rerun-if-changed=ffi/nrd_shim.cpp");
    println!("cargo:rerun-if-changed={}", lib_nrd.display());
    println!(
        "cargo:rerun-if-changed={}",
        nrd_lib.join("libShaderMakeBlob.a").display()
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include("ffi")
        .include(&nrd_include)
        .file("ffi/nrd_shim.cpp")
        .compile("nrd_shim");

    println!("cargo:rustc-link-search=native={}", nrd_lib.display());
    println!("cargo:rustc-link-lib=static=NRD");
    println!("cargo:rustc-link-lib=static=ShaderMakeBlob");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo::rustc-cfg=has_nrd");
}
