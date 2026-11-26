use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=NRD_SDK_DIR");
    let nrd_root = env::var("NRD_SDK_DIR").unwrap_or_else(|_| {
        "/home/viktor/ray_tracing_denoising_code/engine/thirdparty/nrd".to_string()
    });
    let nrd_include = PathBuf::from(&nrd_root).join("Include");
    let nrd_lib = PathBuf::from(&nrd_root).join("lib");

    assert!(
        nrd_include.exists() && nrd_lib.exists(),
        "NVIDIA NRD SDK not found at '{nrd_root}'. Obtain NVIDIA NRD separately under the NVIDIA RTX SDKs License and set NRD_SDK_DIR to its root directory containing Include/ and lib/."
    );

    println!("cargo:rerun-if-changed=ffi/nrd_shim.h");
    println!("cargo:rerun-if-changed=ffi/nrd_shim.cpp");
    println!("cargo:rerun-if-changed={}", nrd_lib.join("libNRD.a").display());
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
}
