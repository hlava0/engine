use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest_dir.join("../../shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("shaders");

    fs::create_dir_all(&out_dir).expect("Failed to create OUT_DIR/shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    let mut compiled = Vec::new();
    if shader_dir.exists() {
        compile_dir(&shader_dir, &shader_dir, &out_dir, &mut compiled);
    }
    compiled.sort();

    let mut registry = String::from("pub fn shader_spv_bytes(name: &str) -> Option<&'static [u8]> {\n    match name {\n");
    for rel_str in &compiled {
        registry.push_str(&format!(
            "        {rel_str:?} => Some(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/shaders/{rel_str}.spv\"))),\n"
        ));
    }
    registry.push_str("        _ => None,\n    }\n}\n");

    fs::write(out_dir.join("registry.rs"), registry).expect("Failed to write shader registry.rs");
}

fn compile_dir(root: &Path, dir: &Path, out_dir: &Path, compiled: &mut Vec<String>) {
    let entries = fs::read_dir(dir).expect("Failed to read shader directory");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            compile_dir(root, &path, out_dir, compiled);
            continue;
        }

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let is_glsl = matches!(
            ext,
            "comp" | "vert" | "frag" | "rgen" | "rmiss" | "rchit" | "rahit" | "rint" | "rcall" | "task" | "mesh"
        );
        if !is_glsl {
            continue;
        }

        println!("cargo:rerun-if-changed={}", path.display());

        let rel = path.strip_prefix(root).expect("strip_prefix");
        let mut out_file = out_dir.join(rel);
        let mut os_name = out_file.as_os_str().to_os_string();
        os_name.push(".spv");
        out_file = PathBuf::from(os_name);

        if let Some(parent) = out_file.parent() {
            fs::create_dir_all(parent).expect("Failed to create output shader parent dir");
        }

        let status = Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("spirv1.6")
            .arg("-g")
            .arg("-o")
            .arg(&out_file)
            .arg(&path)
            .status()
            .expect("Failed to execute glslangValidator");

        if !status.success() {
            panic!("glslangValidator failed for {}", path.display());
        }

        let rel_str = rel.to_string_lossy().replace('\\', "/");
        compiled.push(rel_str);
    }
}
