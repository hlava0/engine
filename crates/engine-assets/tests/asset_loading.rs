use std::io::Cursor;

use ash::vk;
use engine_assets::{load_gltf_from_slice, load_texture_from_memory, upload_texture_to_gpu};
use engine_rhi::{BindlessConfig, BindlessTable, RhiConfig, RhiDevice};
use image::{ImageFormat, Rgba, RgbaImage};

/// Build a minimal valid binary GLB 2.0 containing a single triangle mesh,
/// a PBR metallic-roughness + emissive material, and a translated node.
fn build_minimal_glb() -> Vec<u8> {
    // 3 vertices (positions only, 3 * 12 = 36 bytes) + 3 u16 indices (6 bytes) -> padded to 44 bytes
    let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices: [u16; 3] = [0, 1, 2];

    let mut bin = Vec::new();
    for p in &positions {
        for &c in p {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let indices_offset = bin.len(); // 36
    for &idx in &indices {
        bin.extend_from_slice(&idx.to_le_bytes());
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{
    "name": "TranslatedTri",
    "mesh": 0,
    "translation": [2.0, 3.0, -4.0]
  }}],
  "materials": [{{
    "name": "CopperMat",
    "pbrMetallicRoughness": {{
      "baseColorFactor": [0.85, 0.45, 0.25, 1.0],
      "metallicFactor": 0.9,
      "roughnessFactor": 0.25
    }},
    "emissiveFactor": [0.1, 0.2, 0.3]
  }}],
  "meshes": [{{
    "name": "UnitTri",
    "primitives": [{{
      "attributes": {{ "POSITION": 0 }},
      "indices": 1,
      "material": 0
    }}]
  }}],
  "buffers": [{{ "byteLength": {} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {}, "byteLength": 6, "target": 34963 }}
  ],
  "accessors": [
    {{
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0.0, 0.0, 0.0],
      "max": [1.0, 1.0, 0.0]
    }},
    {{
      "bufferView": 1,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }}
  ]
}}"#,
        bin.len(),
        indices_offset
    );

    let mut json_bytes = json.into_bytes();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_length);

    // GLB header
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
    glb.extend_from_slice(&2u32.to_le_bytes()); // version 2
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    glb.extend_from_slice(&json_bytes);

    // BIN chunk
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
    glb.extend_from_slice(&bin);

    glb
}

#[test]
fn glb_in_memory_roundtrip() {
    let glb = build_minimal_glb();
    let scene = load_gltf_from_slice(&glb).expect("load_gltf_from_slice");

    assert_eq!(scene.meshes.len(), 1);
    let mesh = &scene.meshes[0];
    assert_eq!(mesh.name, "UnitTri");
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.indices, vec![0, 1, 2]);
    assert_eq!(mesh.vertices[0].pos, [0.0, 0.0, 0.0]);
    assert_eq!(mesh.vertices[1].pos, [1.0, 0.0, 0.0]);
    assert_eq!(mesh.vertices[2].pos, [0.0, 1.0, 0.0]);
    // Flat normal computed for CCW triangle in XY plane should be +Z
    for v in &mesh.vertices {
        assert!(
            (v.normal[2] - 1.0).abs() < 1e-5,
            "Unexpected normal: {:?}",
            v.normal
        );
    }
    assert!(mesh.bounding_sphere[3] > 0.5);

    assert_eq!(scene.materials.len(), 1);
    let mat = &scene.materials[0];
    assert_eq!(mat.name, "CopperMat");
    assert!((mat.albedo[0] - 0.85).abs() < 1e-5);
    assert!((mat.albedo[1] - 0.45).abs() < 1e-5);
    assert!((mat.albedo[2] - 0.25).abs() < 1e-5);
    assert!((mat.metallic - 0.9).abs() < 1e-5);
    assert!((mat.roughness - 0.25).abs() < 1e-5);
    assert!((mat.emissive[0] - 0.1).abs() < 1e-5);
    assert!((mat.emissive[1] - 0.2).abs() < 1e-5);
    assert!((mat.emissive[2] - 0.3).abs() < 1e-5);

    assert_eq!(scene.instances.len(), 1);
    let inst = &scene.instances[0];
    assert_eq!(inst.name, "TranslatedTri");
    assert_eq!(inst.mesh_index, 0);
    assert_eq!(inst.material_index, 0);
    // Column 3 of column-major matrix holds translation [2, 3, -4, 1]
    assert!((inst.transform[3][0] - 2.0).abs() < 1e-5);
    assert!((inst.transform[3][1] - 3.0).abs() < 1e-5);
    assert!((inst.transform[3][2] - (-4.0)).abs() < 1e-5);
    assert!((inst.transform[3][3] - 1.0).abs() < 1e-5);
}

#[test]
fn png_decode_and_gpu_bindless_upload() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1. Encode a 4x4 RGBA PNG in memory
    let mut img = RgbaImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, Rgba([(x * 60) as u8, (y * 60) as u8, 200, 255]));
        }
    }
    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
        .expect("encode 4x4 PNG");

    // 2. Decode via load_texture_from_memory
    let tex = load_texture_from_memory(&png_bytes).expect("load_texture_from_memory");
    assert_eq!(tex.width, 4);
    assert_eq!(tex.height, 4);
    assert_eq!(tex.rgba8.len(), 4 * 4 * 4);
    assert_eq!(&tex.rgba8[0..4], &[0, 0, 200, 255]);

    // 3. Upload to GPU via RhiDevice with validation enabled
    let mut rhi = RhiDevice::new(RhiConfig {
        app_name: "asset_texture_upload_test".to_string(),
        enable_validation: true,
        enable_sync_validation: true,
        ..Default::default()
    })
    .expect("RhiDevice::new");

    let bindless =
        BindlessTable::new(&rhi.device, BindlessConfig::default()).expect("BindlessTable");
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);
    let sampler = unsafe {
        rhi.device
            .create_sampler(&sampler_info, None)
            .expect("sampler")
    };

    let pool = rhi
        .create_command_pool(
            rhi.queue_families.graphics,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )
        .expect("command pool");
    let cmd = rhi.allocate_command_buffer(pool).expect("command buffer");

    unsafe {
        rhi.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin cmd");
    }

    let (gpu_image, staging_buf) =
        upload_texture_to_gpu(&mut rhi, cmd, &bindless, sampler, 0, &tex)
            .expect("upload_texture_to_gpu");

    unsafe {
        rhi.device.end_command_buffer(cmd).expect("end cmd");
    }
    rhi.submit_and_wait(rhi.graphics_queue, cmd)
        .expect("submit_and_wait");

    let dev = rhi.device.clone();
    staging_buf
        .destroy(&dev, rhi.allocator_mut())
        .expect("destroy staging");
    gpu_image
        .destroy(&dev, rhi.allocator_mut())
        .expect("destroy image");
    unsafe {
        dev.destroy_command_pool(pool, None);
        dev.destroy_sampler(sampler, None);
    }
    bindless.destroy(&dev);

    assert_eq!(
        rhi.validation_errors(),
        0,
        "Expected 0 Vulkan validation errors during texture upload"
    );
}
