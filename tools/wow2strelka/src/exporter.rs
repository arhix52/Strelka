use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use serde_json::{Value, json};
use wow_blp::convert::blp_to_image;
use wow_blp::parser::load_blp_from_buf;

use crate::ir::{
    AtmosphereMetadata, BlendMode, Material, MaterialKind, Scene, TextureRef, Transform,
};
use crate::source::AssetSource;

struct ExportLock {
    path: std::path::PathBuf,
}

impl ExportLock {
    fn acquire(output: &Path) -> Result<Self> {
        let path = output.join(".wow2strelka-exporting");
        fs::write(&path, std::process::id().to_string())?;
        Ok(Self { path })
    }
}

impl Drop for ExportLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn export_scene(scene: &mut Scene, source: &dyn AssetSource, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let _lock = ExportLock::acquire(output)?;
    fs::create_dir_all(output.join("materials"))?;
    fs::create_dir_all(output.join("textures"))?;
    let warnings = export_textures(scene, source, output)?;
    scene.metadata.warnings.extend(warnings);
    export_environment(scene, output)?;
    export_materials(scene, output)?;
    export_gltf(scene, output)?;
    fs::write(
        output.join("scene.json"),
        serde_json::to_vec_pretty(&scene.metadata)?,
    )?;
    Ok(())
}

fn export_environment(scene: &mut Scene, output: &Path) -> Result<()> {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 512;
    let relative = "textures/environment/wow_sky.png";
    let destination = output.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut sky = RgbaImage::new(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        let v = y as f32 / (HEIGHT - 1) as f32;
        for x in 0..WIDTH {
            let u = x as f32 / (WIDTH - 1) as f32;
            let color = if v < 0.5 {
                let t = (v * 2.0).powi(3);
                mix3([18.0, 46.0, 105.0], [175.0, 142.0, 92.0], t)
            } else {
                let t = ((v - 0.5) * 2.0).powf(0.7);
                mix3([175.0, 142.0, 92.0], [42.0, 28.0, 18.0], t)
            };
            let sun_distance = ((u - 0.72).powi(2) + (v - 0.30).powi(2)).sqrt();
            let sun = (1.0 - sun_distance / 0.022).clamp(0.0, 1.0).powi(2);
            sky.put_pixel(
                x,
                y,
                Rgba([
                    (color[0] + sun * 80.0).min(255.0) as u8,
                    (color[1] + sun * 90.0).min(255.0) as u8,
                    (color[2] + sun * 110.0).min(255.0) as u8,
                    255,
                ]),
            );
        }
    }
    sky.save(&destination)?;
    scene.metadata.environment.skybox = Some(relative.to_owned());
    scene.metadata.environment.ambient_color = Some([0.28, 0.34, 0.48, 1.0]);
    scene.metadata.atmosphere = Some(AtmosphereMetadata {
        color: [0.78, 0.62, 0.42],
        density: 0.00008,
        anisotropy: 0.25,
        height: 2000.0,
        source: "generated_barrens_preview".to_owned(),
    });
    let sidecar = json!({
        "environment": {
            "texture": relative,
            "backgroundTexture": relative,
            "intensity": 5000.0,
            "backgroundIntensity": 2500.0,
            "color": [0.82, 0.88, 1.0],
            "rotation": 0.0
        },
        "lights": [{
            "name": "wow_sun",
            "type": "distant",
            "orientation": [-45.0, 15.0, 0.0],
            "color": [1.0, 0.82, 0.58],
            "intensity": 50000.0,
            "halfAngle": 0.53,
            "visibleToCamera": false
        }],
        "atmosphere": {
            "color": [0.78, 0.62, 0.42],
            "density": 0.00008,
            "anisotropy": 0.25,
            "height": 2000.0
        }
    });
    fs::write(
        output.join("scene_light.json"),
        serde_json::to_vec_pretty(&sidecar)?,
    )?;
    Ok(())
}

fn mix3(left: [f32; 3], right: [f32; 3], amount: f32) -> [f32; 3] {
    [
        left[0] + (right[0] - left[0]) * amount,
        left[1] + (right[1] - left[1]) * amount,
        left[2] + (right[2] - left[2]) * amount,
    ]
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
    let has_graph = has_texture || material.normal.is_some() || material.emissive.is_some();
    if has_graph {
        result.push_str(&format!("  <nodegraph name=\"NG_{safe}\">\n"));
        if has_texture && matches!(material.kind, MaterialKind::Terrain) && preserve_terrain_graph {
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
        } else if has_texture {
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
        if let Some(normal) = &material.normal {
            result.push_str("    <image name=\"normal_map\" type=\"vector3\">\n");
            result.push_str(&format!(
                "      <input name=\"file\" type=\"filename\" value=\"{}{}\" colorspace=\"raw\" />\n",
                resource_prefix,
                xml(&normal.output_path)
            ));
            result.push_str("    </image>\n");
            result.push_str(
                "    <output name=\"normal_output\" type=\"vector3\" nodename=\"normal_map\" />\n",
            );
        }
        if let Some(emissive) = &material.emissive {
            result.push_str("    <image name=\"emission_map\" type=\"color3\">\n");
            result.push_str(&format!(
                "      <input name=\"file\" type=\"filename\" value=\"{}{}\" colorspace=\"srgb_texture\" />\n",
                resource_prefix,
                xml(&emissive.output_path)
            ));
            result.push_str("    </image>\n");
            result.push_str(
                "    <output name=\"emission_output\" type=\"color3\" nodename=\"emission_map\" />\n",
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
    if material.normal.is_some() {
        result.push_str(&format!(
            "    <input name=\"geometry_normal\" type=\"vector3\" nodegraph=\"NG_{safe}\" output=\"normal_output\" />\n"
        ));
    }
    if material.emissive.is_some() {
        result.push_str(&format!(
            "    <input name=\"emission_color\" type=\"color3\" nodegraph=\"NG_{safe}\" output=\"emission_output\" />\n"
        ));
        result.push_str("    <input name=\"emission_luminance\" type=\"float\" value=\"1\" />\n");
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
    if material.unlit && material.emissive.is_none() {
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
    let mut skins = Vec::<Value>::new();
    let mut animations = Vec::<Value>::new();
    let mut animation_groups = BTreeMap::<String, (Vec<Value>, Vec<Value>)>::new();
    let mut animation_sampler_maps = BTreeMap::<String, BTreeMap<(usize, usize), usize>>::new();
    let mut animation_accessors = BTreeMap::<(usize, usize, usize), (usize, usize)>::new();
    let mut inverse_bind_accessors = BTreeMap::<usize, usize>::new();
    let mut uses_instancing = false;

    for mesh in &scene.meshes {
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|vertex| vertex.position).collect();
        let normals: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|vertex| normalized3(vertex.normal))
            .collect();
        let uv0: Vec<[f32; 2]> = mesh.vertices.iter().map(|vertex| vertex.uv0).collect();
        let colors: Vec<[u8; 4]> = mesh.vertices.iter().map(|vertex| vertex.color0).collect();
        let position = push_f32x3(&mut binary, &mut views, &mut accessors, &positions, true)?;
        let normal = push_f32x3(&mut binary, &mut views, &mut accessors, &normals, false)?;
        let uv = push_f32x2(&mut binary, &mut views, &mut accessors, &uv0)?;
        let color = push_u8x4(&mut binary, &mut views, &mut accessors, &colors)?;
        let skin_attributes = if let Some(skin) = &mesh.skin {
            Some((
                push_u16x4(&mut binary, &mut views, &mut accessors, &skin.joints)?,
                push_f32x4(&mut binary, &mut views, &mut accessors, &skin.weights)?,
            ))
        } else {
            None
        };
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
            let mut attributes = json!({
                    "POSITION": position,
                    "NORMAL": normal,
                    "TEXCOORD_0": uv,
                    "COLOR_0": color
            });
            if let Some((joints, weights)) = skin_attributes {
                attributes["JOINTS_0"] = json!(joints);
                attributes["WEIGHTS_0"] = json!(weights);
            }
            primitives.push(json!({
                "attributes": attributes,
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
        let index = nodes.len();
        let mut value = node_json(&node.name, node.mesh, node.transform, None);
        if !node.children.is_empty() {
            value["children"] = json!(node.children);
        }
        nodes.push(value);
        scene_nodes.push(index);
    }
    for node_index in &scene.animated_nodes {
        let Some(node) = scene.nodes.get(*node_index) else {
            continue;
        };
        let Some(mesh_index) = node.mesh else {
            continue;
        };
        let Some(template) = scene.meshes[mesh_index].skin.as_ref() else {
            continue;
        };
        if let Some(object) = nodes[*node_index].as_object_mut() {
            object.remove("mesh");
        }
        let skinned_mesh_node = nodes.len();
        nodes.push(node_json(
            &format!("{}_skinned_mesh", node.name),
            Some(mesh_index),
            Transform::IDENTITY,
            None,
        ));
        nodes[skinned_mesh_node]["extras"] = json!({
            "strelka_shared_pose": true
        });
        if let Some(transforms) = scene.animated_instances.get(node_index)
            && !transforms.is_empty()
        {
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
            nodes[skinned_mesh_node]["extensions"] = json!({
                "EXT_mesh_gpu_instancing": {
                    "attributes": {
                        "TRANSLATION": translation,
                        "ROTATION": rotation,
                        "SCALE": scale
                    }
                }
            });
            uses_instancing = true;
        }
        let joint_base = nodes.len();
        for (bone_index, bone) in template.bones.iter().enumerate() {
            let children: Vec<usize> = template
                .bones
                .iter()
                .enumerate()
                .filter_map(|(child, candidate)| {
                    (candidate.parent == Some(bone_index)).then_some(joint_base + child)
                })
                .collect();
            let mut value = json!({
                "name": format!("{}_bone_{}", node.name, bone_index),
                "translation": bone.translation,
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0]
            });
            if !children.is_empty() {
                value["children"] = json!(children);
            }
            nodes.push(value);
        }
        let roots: Vec<usize> = template
            .bones
            .iter()
            .enumerate()
            .filter_map(|(index, bone)| bone.parent.is_none().then_some(joint_base + index))
            .collect();
        let mut placement_children = vec![skinned_mesh_node];
        placement_children.extend_from_slice(&roots);
        append_node_children(&mut nodes[*node_index], &placement_children);
        if let Some(equipment) = scene.equipment.get(node_index) {
            for item in equipment {
                let Some(attachment) = template
                    .attachments
                    .iter()
                    .find(|attachment| attachment.id == item.attachment_id)
                else {
                    continue;
                };
                let equipment_node = nodes.len();
                nodes.push(node_json(
                    &format!("item_{}_fdid_{}", item.item_id, item.model_file_id),
                    Some(item.mesh),
                    Transform {
                        translation: attachment.translation,
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    },
                    None,
                ));
                append_node_children(&mut nodes[joint_base + attachment.bone], &[equipment_node]);
            }
        }
        let inverse_bind = if let Some(accessor) = inverse_bind_accessors.get(&mesh_index) {
            *accessor
        } else {
            let values: Vec<[f32; 16]> = template
                .bones
                .iter()
                .map(|bone| bone.inverse_bind)
                .collect();
            let accessor = push_f32mat4(&mut binary, &mut views, &mut accessors, &values)?;
            inverse_bind_accessors.insert(mesh_index, accessor);
            accessor
        };
        let skin_index = skins.len();
        skins.push(json!({
            "name": format!("{}_skin", node.name),
            "inverseBindMatrices": inverse_bind,
            "joints": (0..template.bones.len()).map(|index| joint_base + index).collect::<Vec<_>>()
        }));
        nodes[skinned_mesh_node]["skin"] = json!(skin_index);
        for (clip_index, clip) in template.clips.iter().enumerate() {
            let clip_name = clip.name.split('_').next().unwrap_or(&clip.name).to_owned();
            let sampler_map = animation_sampler_maps.entry(clip_name.clone()).or_default();
            let (samplers, channels) = animation_groups.entry(clip_name).or_default();
            for (channel_index, channel) in clip.channels.iter().enumerate() {
                if channel.bone >= template.bones.len()
                    || channel.times.is_empty()
                    || channel.times.len() != channel.values.len()
                {
                    continue;
                }
                let path = match channel.path {
                    crate::ir::AnimationPath::Translation => "translation",
                    crate::ir::AnimationPath::Rotation => "rotation",
                    crate::ir::AnimationPath::Scale => "scale",
                };
                let accessor_key = (mesh_index, clip_index, channel_index);
                let (input, output) = if let Some(value) = animation_accessors.get(&accessor_key) {
                    *value
                } else {
                    let input = push_f32(&mut binary, &mut views, &mut accessors, &channel.times)?;
                    let output = match channel.path {
                        crate::ir::AnimationPath::Translation => {
                            let bind = template.bones[channel.bone].translation;
                            let values: Vec<[f32; 3]> = channel
                                .values
                                .iter()
                                .map(|value| {
                                    [bind[0] + value[0], bind[1] + value[1], bind[2] + value[2]]
                                })
                                .collect();
                            push_f32x3_data(&mut binary, &mut views, &mut accessors, &values)?
                        }
                        crate::ir::AnimationPath::Rotation => push_f32x4_data(
                            &mut binary,
                            &mut views,
                            &mut accessors,
                            &channel.values,
                        )?,
                        crate::ir::AnimationPath::Scale => {
                            let values: Vec<[f32; 3]> = channel
                                .values
                                .iter()
                                .map(|value| [value[0], value[1], value[2]])
                                .collect();
                            push_f32x3_data(&mut binary, &mut views, &mut accessors, &values)?
                        }
                    };
                    animation_accessors.insert(accessor_key, (input, output));
                    (input, output)
                };
                let sampler = if let Some(sampler) = sampler_map.get(&(input, output)) {
                    *sampler
                } else {
                    let sampler = samplers.len();
                    samplers.push(json!({
                        "input": input,
                        "output": output,
                        "interpolation": "LINEAR"
                    }));
                    sampler_map.insert((input, output), sampler);
                    sampler
                };
                channels.push(json!({
                    "sampler": sampler,
                    "target": {
                        "node": joint_base + channel.bone,
                        "path": path
                    }
                }));
            }
        }
    }
    animations.extend(
        animation_groups
            .into_iter()
            .filter(|(_, (_, channels))| !channels.is_empty())
            .map(|(name, (samplers, channels))| {
                json!({
                    "name": name,
                    "samplers": samplers,
                    "channels": channels
                })
            }),
    );
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
            Some(*mesh),
            Transform::IDENTITY,
            Some(extension),
        ));
        scene_nodes.push(index);
    }

    let mut image_indices = BTreeMap::<String, usize>::new();
    let mut images = Vec::<Value>::new();
    let mut textures = Vec::<Value>::new();
    for material in &scene.materials {
        let Some(path) = gltf_base_texture_path(material) else {
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
                "doubleSided": material.double_sided,
                "extras": {
                    "materialx": format!("materials/{}.mtlx", material.name),
                    "wow_source": material.source,
                    "wow_blend": format!("{:?}", material.blend).to_lowercase(),
                    "wow_shader_id": material.shader_id
                }
            });
            if alpha_mode == "MASK" {
                value["alphaCutoff"] = json!(0.5);
            }
            if let Some(path) = gltf_base_texture_path(material)
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
        "skins": skins,
        "animations": animations,
        "cameras": cameras,
        "scenes": [{ "name": "WoW location", "nodes": scene_nodes }],
        "scene": 0
    });
    if uses_instancing {
        gltf["extensionsUsed"] = json!(["EXT_mesh_gpu_instancing"]);
    }
    fs::write(output.join("scene.bin"), binary)?;
    fs::write(output.join("scene.gltf"), serde_json::to_vec(&gltf)?)?;
    Ok(())
}

fn gltf_alpha_mode(blend: BlendMode) -> &'static str {
    match blend {
        BlendMode::Opaque => "OPAQUE",
        BlendMode::Mask => "MASK",
        BlendMode::Blend | BlendMode::Additive | BlendMode::Modulate => "BLEND",
    }
}

fn gltf_base_texture_path(material: &Material) -> Option<String> {
    if matches!(material.blend, BlendMode::Opaque) {
        return None;
    }
    material_base_texture_path(material)
}

fn node_json(
    name: &str,
    mesh: Option<usize>,
    transform: Transform,
    extension: Option<Value>,
) -> Value {
    let mut node = json!({
        "name": name,
        "translation": transform.translation,
        "rotation": transform.rotation,
        "scale": transform.scale
    });
    if let Some(mesh) = mesh {
        node["mesh"] = json!(mesh);
    }
    if let Some(extension) = extension {
        node["extensions"] = extension;
    }
    node
}

fn append_node_children(node: &mut Value, children: &[usize]) {
    let mut all: Vec<usize> = node
        .get("children")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as usize)
                .collect()
        })
        .unwrap_or_default();
    all.extend_from_slice(children);
    if !all.is_empty() {
        node["children"] = json!(all);
    }
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

fn push_f32x3_data(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[f32; 3]],
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
        "byteLength": values.len() * 12
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "VEC3"
    }));
    Ok(accessor)
}

fn push_f32x4_data(
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
        "byteLength": values.len() * 16
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

fn push_u16x4(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[u16; 4]],
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        for component in value {
            binary.write_u16::<LittleEndian>(*component)?;
        }
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
        "componentType": 5123,
        "count": values.len(),
        "type": "VEC4"
    }));
    Ok(accessor)
}

fn push_f32(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[f32],
) -> Result<usize> {
    align4(binary);
    let offset = binary.len();
    for value in values {
        binary.write_f32::<LittleEndian>(*value)?;
    }
    let view = views.len();
    views.push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": values.len() * 4
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "SCALAR",
        "min": [values.iter().copied().fold(f32::INFINITY, f32::min)],
        "max": [values.iter().copied().fold(f32::NEG_INFINITY, f32::max)]
    }));
    Ok(accessor)
}

fn push_f32mat4(
    binary: &mut Vec<u8>,
    views: &mut Vec<Value>,
    accessors: &mut Vec<Value>,
    values: &[[f32; 16]],
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
        "byteLength": values.len() * 64
    }));
    let accessor = accessors.len();
    accessors.push(json!({
        "bufferView": view,
        "componentType": 5126,
        "count": values.len(),
        "type": "MAT4"
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

fn normalized3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length > 0.0 && length.is_finite() {
        [value[0] / length, value[1] / length, value[2] / length]
    } else {
        [0.0, 1.0, 0.0]
    }
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
    use super::{bounds3, export_gltf, gltf_alpha_mode, xml};
    use crate::ir::{
        BlendMode, BoneTemplate, Mesh, Node, Primitive, Scene, SkinTemplate, Transform, Vertex,
    };

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

    #[test]
    fn animated_nodes_use_skin_without_gpu_instancing() {
        let output = tempfile::tempdir().unwrap();
        let mut scene = Scene::default();
        scene.meshes.push(Mesh {
            name: "npc".to_owned(),
            source: "npc.m2".to_owned(),
            vertices: vec![Vertex {
                position: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
                uv0: [0.0; 2],
                color0: [255; 4],
            }],
            indices: vec![0],
            primitives: vec![Primitive {
                first_index: 0,
                index_count: 1,
                material: 0,
            }],
            skin: Some(SkinTemplate {
                joints: vec![[0; 4]],
                weights: vec![[1.0, 0.0, 0.0, 0.0]],
                bones: vec![BoneTemplate {
                    parent: None,
                    translation: [0.0; 3],
                    inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                }],
                clips: Vec::new(),
                attachments: Vec::new(),
            }),
        });
        scene.nodes.push(Node {
            name: "npc".to_owned(),
            mesh: Some(0),
            transform: Transform::IDENTITY,
            children: Vec::new(),
        });
        scene.animated_nodes.push(0);
        export_gltf(&scene, output.path()).unwrap();
        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.path().join("scene.gltf")).unwrap())
                .unwrap();
        assert_eq!(document["skins"].as_array().unwrap().len(), 1);
        assert!(document["nodes"][0].get("skin").is_none());
        assert_eq!(document["nodes"][1]["skin"], 0);
        assert!(document.get("extensionsUsed").is_none());
    }
}
