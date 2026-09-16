use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use serde_json::{Value, json};
use wow_blp::convert::blp_to_image;
use wow_blp::parser::load_blp_from_buf;

use crate::ir::{BlendMode, Material, MaterialKind, Scene, TextureRef, Transform};
use crate::source::AssetSource;

pub fn export_scene(scene: &mut Scene, source: &dyn AssetSource, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    fs::create_dir_all(output.join("materials"))?;
    fs::create_dir_all(output.join("textures"))?;
    let warnings = export_textures(scene, source, output)?;
    scene.metadata.warnings.extend(warnings);
    export_materials(scene, output)?;
    export_gltf(scene, output)?;
    fs::write(
        output.join("scene.json"),
        serde_json::to_vec_pretty(&scene.metadata)?,
    )?;
    Ok(())
}

fn export_textures(scene: &Scene, source: &dyn AssetSource, output: &Path) -> Result<Vec<String>> {
    let mut textures = BTreeMap::<String, &TextureRef>::new();
    let mut warnings = Vec::new();
    for material in &scene.materials {
        for layer in &material.layers {
            if !layer.texture.wow_path.is_empty() {
                textures.insert(layer.texture.output_path.clone(), &layer.texture);
            }
            if let (Some(path), Some(data)) = (&layer.alpha_map, &layer.alpha_data) {
                let destination = output.join(path.replace('/', std::path::MAIN_SEPARATOR_STR));
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                let image = GrayImage::from_raw(64, 64, data.clone())
                    .context("invalid 64x64 terrain alpha map")?;
                image.save(destination)?;
            }
        }
        for texture in [&material.normal, &material.emissive].into_iter().flatten() {
            textures.insert(texture.output_path.clone(), texture);
        }
    }
    for (relative, texture) in textures {
        let destination = output.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
        if destination.is_file()
            && image::image_dimensions(&destination).is_ok_and(|size| size != (1, 1))
        {
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let converted = source.read(&texture.wow_path).and_then(|bytes| {
            let blp = load_blp_from_buf(&bytes)
                .with_context(|| format!("failed to decode {}", texture.wow_path))?;
            blp_to_image(&blp, 0).with_context(|| format!("failed to convert {}", texture.wow_path))
        });
        let image = match converted {
            Ok(image) => image,
            Err(error) => {
                warnings.push(format!(
                    "Texture {}: {error:#}; exported magenta placeholder",
                    texture.wow_path
                ));
                DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([255, 0, 255, 255])))
            }
        };
        image.save(&destination)?;
    }
    for material in &scene.materials {
        if matches!(material.kind, MaterialKind::Terrain) {
            export_baked_terrain_material(material, output)?;
        }
    }
    Ok(warnings)
}

fn export_baked_terrain_material(material: &Material, output: &Path) -> Result<()> {
    const SIZE: u32 = 512;
    const REPEAT: u32 = 8;
    let relative = terrain_baked_path(material);
    let destination = output.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
    if destination.is_file() {
        return Ok(());
    }
    let mut layers = Vec::new();
    for layer in &material.layers {
        if layer.texture.output_path.is_empty() {
            continue;
        }
        let path = output.join(
            layer
                .texture
                .output_path
                .replace('/', std::path::MAIN_SEPARATOR_STR),
        );
        layers.push((
            image::open(&path)
                .with_context(|| format!("failed to reopen terrain texture {}", path.display()))?
                .to_rgba8(),
            layer.alpha_data.as_deref(),
        ));
    }
    if layers.is_empty() {
        return Ok(());
    }
    let mut baked = RgbaImage::new(SIZE, SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut color = [0.0f32; 4];
            for (index, (texture, alpha)) in layers.iter().enumerate() {
                let tx = (x * REPEAT * texture.width() / SIZE) % texture.width();
                let ty = (y * REPEAT * texture.height() / SIZE) % texture.height();
                let sample = texture.get_pixel(tx, ty).0;
                let blend = if index == 0 {
                    1.0
                } else {
                    alpha
                        .and_then(|values| {
                            values.get(((y * 64 / SIZE) * 64 + x * 64 / SIZE) as usize)
                        })
                        .map_or(0.0, |value| f32::from(*value) / 255.0)
                };
                for channel in 0..4 {
                    color[channel] =
                        color[channel] * (1.0 - blend) + f32::from(sample[channel]) * blend;
                }
            }
            baked.put_pixel(
                x,
                y,
                Rgba([
                    color[0] as u8,
                    color[1] as u8,
                    color[2] as u8,
                    color[3] as u8,
                ]),
            );
        }
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    baked.save(destination)?;
    Ok(())
}

fn terrain_baked_path(material: &Material) -> String {
    format!("textures/terrain_baked/{}.png", material.name)
}

fn material_base_texture_path(material: &Material) -> Option<String> {
    if matches!(material.kind, MaterialKind::Terrain) {
        return Some(terrain_baked_path(material));
    }
    material
        .layers
        .first()
        .map(|layer| layer.texture.output_path.clone())
        .filter(|path| !path.is_empty())
}

fn export_materials(scene: &Scene, output: &Path) -> Result<()> {
    let mut aggregate = materialx_header();
    for material in &scene.materials {
        let body = material_body(material, true, "../");
        let mut document = materialx_header();
        document.push_str(&body);
        document.push_str("</materialx>\n");
        fs::write(
            output
                .join("materials")
                .join(format!("{}.mtlx", material.name)),
            document,
        )?;
        aggregate.push_str(&material_body(material, false, ""));
    }
    aggregate.push_str("</materialx>\n");
    fs::write(output.join("scene.mtlx"), aggregate)?;
    Ok(())
}

fn materialx_header() -> String {
    "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\" colorspace=\"lin_rec709\">\n".to_owned()
}

fn material_body(
    material: &Material,
    preserve_terrain_graph: bool,
    resource_prefix: &str,
) -> String {
    let name = xml(&material.name);
    let safe = safe_name(&material.name);
    let mut result = String::new();
    let has_texture = material
        .layers
        .first()
        .is_some_and(|layer| !layer.texture.output_path.is_empty());
    if has_texture {
        result.push_str(&format!("  <nodegraph name=\"NG_{safe}\">\n"));
        if matches!(material.kind, MaterialKind::Terrain) && preserve_terrain_graph {
            let mut previous = String::new();
            for (index, layer) in material.layers.iter().enumerate() {
                let image = format!("layer_{index}");
                result.push_str(&format!("    <image name=\"{image}\" type=\"color3\">\n"));
                result.push_str(&format!(
                    "      <input name=\"file\" type=\"filename\" value=\"{}{}\" colorspace=\"srgb_texture\" />\n",
                    resource_prefix,
                    xml(&layer.texture.output_path)
                ));
                result.push_str("    </image>\n");
                if index == 0 {
                    previous = image;
                    continue;
                }
                if let Some(alpha) = &layer.alpha_map {
                    let alpha_node = format!("alpha_{index}");
                    let mix_node = format!("mix_{index}");
                    result.push_str(&format!(
                        "    <image name=\"{alpha_node}\" type=\"float\">\n"
                    ));
                    result.push_str(&format!(
                        "      <input name=\"file\" type=\"filename\" value=\"{}{}\" colorspace=\"raw\" />\n",
                        resource_prefix,
                        xml(alpha)
                    ));
                    result.push_str("    </image>\n");
                    result.push_str(&format!("    <mix name=\"{mix_node}\" type=\"color3\">\n"));
                    result.push_str(&format!(
                        "      <input name=\"fg\" type=\"color3\" nodename=\"{image}\" />\n"
                    ));
                    result.push_str(&format!(
                        "      <input name=\"bg\" type=\"color3\" nodename=\"{previous}\" />\n"
                    ));
                    result.push_str(&format!(
                        "      <input name=\"mix\" type=\"float\" nodename=\"{alpha_node}\" />\n"
                    ));
                    result.push_str("    </mix>\n");
                    previous = mix_node;
                }
            }
            result.push_str(&format!(
                "    <output name=\"base_color_output\" type=\"color3\" nodename=\"{previous}\" />\n"
            ));
        } else {
            let texture = if matches!(material.kind, MaterialKind::Terrain) {
                terrain_baked_path(material)
            } else {
                material.layers[0].texture.output_path.clone()
            };
            result.push_str("    <image name=\"base_color\" type=\"color3\">\n");
            result.push_str(&format!(
                "      <input name=\"file\" type=\"filename\" value=\"{}{}\" colorspace=\"srgb_texture\" />\n",
                resource_prefix,
                xml(&texture)
            ));
            result.push_str("    </image>\n");
            result.push_str(
                "    <output name=\"base_color_output\" type=\"color3\" nodename=\"base_color\" />\n",
            );
        }
        result.push_str("  </nodegraph>\n");
    }
    result.push_str(&format!(
        "  <surfacematerial name=\"{name}\" type=\"material\">\n"
    ));
    result.push_str(&format!(
        "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"S_{safe}\" />\n"
    ));
    result.push_str("  </surfacematerial>\n");
    result.push_str(&format!(
        "  <open_pbr_surface name=\"S_{safe}\" type=\"surfaceshader\">\n"
    ));
    result.push_str("    <input name=\"base_metalness\" type=\"float\" value=\"0\" />\n");
    let roughness = if matches!(material.kind, MaterialKind::Liquid) {
        "0.08"
    } else {
        "0.8"
    };
    result.push_str(&format!(
        "    <input name=\"specular_roughness\" type=\"float\" value=\"{roughness}\" />\n"
    ));
    if has_texture {
        result.push_str(&format!(
            "    <input name=\"base_color\" type=\"color3\" nodegraph=\"NG_{safe}\" output=\"base_color_output\" />\n"
        ));
    } else if matches!(material.kind, MaterialKind::Liquid) {
        result.push_str(
            "    <input name=\"base_color\" type=\"color3\" value=\"0.03, 0.16, 0.22\" />\n",
        );
    } else {
        result.push_str(
            "    <input name=\"base_color\" type=\"color3\" value=\"0.5, 0.5, 0.5\" />\n",
        );
    }
    match material.blend {
        BlendMode::Mask | BlendMode::Blend | BlendMode::Additive | BlendMode::Modulate => {
            result.push_str("    <input name=\"geometry_opacity\" type=\"float\" value=\"1\" />\n");
        }
        BlendMode::Opaque => {}
    }
    if matches!(material.kind, MaterialKind::Liquid) {
        result
            .push_str("    <input name=\"transmission_weight\" type=\"float\" value=\"0.85\" />\n");
        result.push_str("    <input name=\"transmission_color\" type=\"color3\" value=\"0.55, 0.82, 0.92\" />\n");
    }
    if material.unlit {
        result.push_str("    <input name=\"emission_luminance\" type=\"float\" value=\"1\" />\n");
    }
    result.push_str("  </open_pbr_surface>\n");
    result
}

fn export_gltf(scene: &Scene, output: &Path) -> Result<()> {
    let mut binary = Vec::new();
    let mut views = Vec::<Value>::new();
    let mut accessors = Vec::<Value>::new();
    let mut meshes = Vec::<Value>::new();
    let mut nodes = Vec::<Value>::new();
    let mut scene_nodes = Vec::<usize>::new();

    for mesh in &scene.meshes {
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|vertex| vertex.position).collect();
        let normals: Vec<[f32; 3]> = mesh.vertices.iter().map(|vertex| vertex.normal).collect();
        let uv0: Vec<[f32; 2]> = mesh.vertices.iter().map(|vertex| vertex.uv0).collect();
        let colors: Vec<[u8; 4]> = mesh.vertices.iter().map(|vertex| vertex.color0).collect();
        let position = push_f32x3(&mut binary, &mut views, &mut accessors, &positions, true)?;
        let normal = push_f32x3(&mut binary, &mut views, &mut accessors, &normals, false)?;
        let uv = push_f32x2(&mut binary, &mut views, &mut accessors, &uv0)?;
        let color = push_u8x4(&mut binary, &mut views, &mut accessors, &colors)?;
        align4(&mut binary);
        let index_offset = binary.len();
        for index in &mesh.indices {
            binary.write_u32::<LittleEndian>(*index)?;
        }
        let index_view = views.len();
        views.push(json!({
            "buffer": 0,
            "byteOffset": index_offset,
            "byteLength": mesh.indices.len() * 4,
            "target": 34963
        }));
        let mut primitives = Vec::new();
        for primitive in &mesh.primitives {
            let accessor = accessors.len();
            accessors.push(json!({
                "bufferView": index_view,
                "byteOffset": primitive.first_index * 4,
                "componentType": 5125,
                "count": primitive.index_count,
                "type": "SCALAR"
            }));
            primitives.push(json!({
                "attributes": {
                    "POSITION": position,
                    "NORMAL": normal,
                    "TEXCOORD_0": uv,
                    "COLOR_0": color
                },
                "indices": accessor,
                "material": primitive.material,
                "mode": 4
            }));
        }
        meshes.push(json!({
            "name": mesh.name,
            "primitives": primitives,
            "extras": { "wow_source": mesh.source }
        }));
    }

    for node in &scene.nodes {
        let Some(mesh) = node.mesh else {
            continue;
        };
        let index = nodes.len();
        let mut value = node_json(&node.name, mesh, node.transform, None);
        if !node.children.is_empty() {
            value["children"] = json!(node.children);
        }
        nodes.push(value);
        scene_nodes.push(index);
    }
    let mut uses_instancing = false;
    for (mesh, transforms) in &scene.instances {
        if transforms.is_empty() {
            continue;
        }
        uses_instancing = true;
        let translations: Vec<[f32; 3]> =
            transforms.iter().map(|value| value.translation).collect();
        let rotations: Vec<[f32; 4]> = transforms.iter().map(|value| value.rotation).collect();
        let scales: Vec<[f32; 3]> = transforms.iter().map(|value| value.scale).collect();
        let translation = push_f32x3(
            &mut binary,
            &mut views,
            &mut accessors,
            &translations,
            false,
        )?;
        let rotation = push_f32x4(&mut binary, &mut views, &mut accessors, &rotations)?;
        let scale = push_f32x3(&mut binary, &mut views, &mut accessors, &scales, false)?;
        let extension = json!({
            "EXT_mesh_gpu_instancing": {
                "attributes": {
                    "TRANSLATION": translation,
                    "ROTATION": rotation,
                    "SCALE": scale
                }
            }
        });
        let index = nodes.len();
        nodes.push(node_json(
            &format!("{}_instances", scene.meshes[*mesh].name),
            *mesh,
            Transform::IDENTITY,
            Some(extension),
        ));
        scene_nodes.push(index);
    }

    let mut image_indices = BTreeMap::<String, usize>::new();
    let mut images = Vec::<Value>::new();
    let mut textures = Vec::<Value>::new();
    for material in &scene.materials {
        let Some(path) = material_base_texture_path(material) else {
            continue;
        };
        if image_indices.contains_key(&path) {
            continue;
        }
        let index = images.len();
        images.push(json!({ "uri": path }));
        textures.push(json!({ "source": index, "sampler": 0 }));
        image_indices.insert(path, index);
    }
    let materials: Vec<Value> = scene
        .materials
        .iter()
        .map(|material| {
            let alpha_mode = gltf_alpha_mode(material.blend);
            let mut value = json!({
                "name": material.name,
                "alphaMode": alpha_mode,
                "alphaCutoff": 0.5,
                "doubleSided": material.double_sided,
                "extras": {
                    "materialx": format!("materials/{}.mtlx", material.name),
                    "wow_source": material.source,
                    "wow_blend": format!("{:?}", material.blend).to_lowercase()
                }
            });
            if let Some(path) = material_base_texture_path(material)
                && let Some(texture) = image_indices.get(&path)
            {
                value["pbrMetallicRoughness"] = json!({
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                    "baseColorTexture": { "index": texture }
                });
            }
            value
        })
        .collect();
    let cameras: Vec<Value> = scene
        .cameras
        .iter()
        .map(|camera| {
            json!({
                "name": camera.name,
                "type": "perspective",
                "perspective": {
                    "yfov": camera.yfov,
                    "znear": camera.znear,
                    "zfar": camera.zfar
                }
            })
        })
        .collect();
    for (value, camera) in scene.cameras.iter().enumerate() {
        let index = nodes.len();
        nodes.push(json!({
            "name": format!("camera_{}", camera.name),
            "camera": value,
            "translation": camera.transform.translation,
            "rotation": camera.transform.rotation,
            "scale": camera.transform.scale
        }));
        scene_nodes.push(index);
    }
    let mut gltf = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("wow2strelka {}", env!("CARGO_PKG_VERSION"))
        },
        "buffers": [{
            "uri": "scene.bin",
            "byteLength": binary.len()
        }],
        "bufferViews": views,
        "accessors": accessors,
        "images": images,
        "textures": textures,
        "samplers": [{ "wrapS": 10497, "wrapT": 10497, "magFilter": 9729, "minFilter": 9987 }],
        "materials": materials,
        "meshes": meshes,
        "nodes": nodes,
        "cameras": cameras,
        "scenes": [{ "name": "WoW location", "nodes": scene_nodes }],
        "scene": 0
    });
    if uses_instancing {
        gltf["extensionsUsed"] = json!(["EXT_mesh_gpu_instancing"]);
    }
    fs::write(output.join("scene.bin"), binary)?;
    fs::write(output.join("scene.gltf"), serde_json::to_vec_pretty(&gltf)?)?;
    Ok(())
}

fn gltf_alpha_mode(blend: BlendMode) -> &'static str {
    match blend {
        BlendMode::Opaque => "OPAQUE",
        BlendMode::Mask => "MASK",
        BlendMode::Blend | BlendMode::Additive | BlendMode::Modulate => "BLEND",
    }
}

fn node_json(name: &str, mesh: usize, transform: Transform, extension: Option<Value>) -> Value {
    let mut node = json!({
        "name": name,
        "mesh": mesh,
        "translation": transform.translation,
        "rotation": transform.rotation,
        "scale": transform.scale
    });
    if let Some(extension) = extension {
        node["extensions"] = extension;
    }
    node
}

fn push_f32x3(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[f32; 3]],
    bounds: bool,
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        for component in value {
            binary.write_f32::<LittleEndian>(*component)?;
        }
    }
    let view = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 12,
        "target": 34962
    }));
    let accessor = accessors.len();
    let mut value = json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "VEC3"
    });
    if bounds && !values.is_empty() {
        let (min, max) = bounds3(values);
        value["min"] = json!(min);
        value["max"] = json!(max);
    }
    accessors.push(value);
    Ok(accessor)
}

fn push_f32x2(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[f32; 2]],
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        binary.write_f32::<LittleEndian>(value[0])?;
        binary.write_f32::<LittleEndian>(value[1])?;
    }
    let view = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 8,
        "target": 34962
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "VEC2"
    }));
    Ok(accessor)
}

fn push_f32x4(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[f32; 4]],
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        for component in value {
            binary.write_f32::<LittleEndian>(*component)?;
        }
    }
    let view = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 16,
        "target": 34962
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "VEC4"
    }));
    Ok(accessor)
}

fn push_u8x4(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[u8; 4]],
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        binary.extend_from_slice(value);
    }
    let view = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 4,
        "target": 34962
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5121,
        "normalized": true,
        "count": values.len(),
        "type": "VEC4"
    }));
    Ok(accessor)
}

fn bounds3(values: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for value in values {
        for axis in 0..3 {
            min[axis] = min[axis].min(value[axis]);
            max[axis] = max[axis].max(value[axis]);
        }
    }
    (min, max)
}

fn align4(binary: &mut Vec<u8>) {
    while !binary.len().is_multiple_of(4) {
        binary.push(0);
    }
}

fn safe_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::{bounds3, gltf_alpha_mode, xml};
    use crate::ir::BlendMode;

    #[test]
    fn computes_position_bounds() {
        let (min, max) = bounds3(&[[1.0, -2.0, 4.0], [-3.0, 5.0, 2.0]]);
        assert_eq!(min, [-3.0, -2.0, 2.0]);
        assert_eq!(max, [1.0, 5.0, 4.0]);
    }

    #[test]
    fn escapes_materialx_attributes() {
        assert_eq!(xml("a&\"b"), "a&amp;&quot;b");
    }

    #[test]
    fn maps_wow_blend_modes_to_gltf_alpha() {
        assert_eq!(gltf_alpha_mode(BlendMode::Opaque), "OPAQUE");
        assert_eq!(gltf_alpha_mode(BlendMode::Mask), "MASK");
        assert_eq!(gltf_alpha_mode(BlendMode::Additive), "BLEND");
    }
}
