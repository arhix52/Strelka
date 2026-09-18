use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec3};
use image::{DynamicImage, RgbaImage, imageops};
use serde::Serialize;
use serde_json::{Value, json};

use crate::ir::{AnimationClip, MaterialKind, NpcMetadata, Scene, Transform};

#[derive(Debug)]
pub struct ValidationOptions {
    pub output: PathBuf,
    pub phases: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub strelka_cli: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub schema_version: u32,
    pub source_scene: String,
    pub spawn_count: usize,
    pub unique_variant_count: usize,
    pub phases: Vec<f32>,
    pub variants: Vec<VariantReport>,
}

#[derive(Debug, Serialize)]
pub struct VariantReport {
    pub key: String,
    pub name: String,
    pub display_id: u32,
    pub model_file_id: u32,
    pub model_path: String,
    pub texture_variations: Vec<u32>,
    pub equipment: [u32; 3],
    pub guids: Vec<u32>,
    pub clips: Vec<ClipReport>,
    pub issues: Vec<AuditIssue>,
    pub files: BTreeMap<String, String>,
    pub contact_sheet: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ClipReport {
    pub name: String,
    pub present: bool,
    pub fallback: bool,
    pub channel_count: usize,
    pub moves: bool,
}

#[derive(Debug, Serialize)]
pub struct AuditIssue {
    pub severity: &'static str,
    pub code: &'static str,
    pub message: String,
}

struct Variant<'a> {
    npc: &'a NpcMetadata,
    guids: Vec<u32>,
}

pub fn export_npc_validation(
    scene: &Scene,
    scene_output: &Path,
    options: &ValidationOptions,
) -> Result<ValidationReport> {
    let gltf_path = scene_output.join("scene.gltf");
    let gltf: Value = serde_json::from_slice(
        &fs::read(&gltf_path).with_context(|| format!("failed to read {}", gltf_path.display()))?,
    )?;
    let variants = unique_variants(&scene.metadata.npcs);
    fs::create_dir_all(&options.output)?;
    let mut report = ValidationReport {
        schema_version: 1,
        source_scene: gltf_path.display().to_string(),
        spawn_count: scene.metadata.npcs.len(),
        unique_variant_count: variants.len(),
        phases: options.phases.clone(),
        variants: Vec::with_capacity(variants.len()),
    };
    for (key, variant) in variants {
        let directory = options.output.join(&key);
        fs::create_dir_all(&directory)?;
        let mut variant_report = build_variant_report(scene, scene_output, &key, &variant)?;
        let fixture = extract_fixture(&gltf, variant.npc.node, &variant_report.name)?;
        let material_names = fixture_material_names(&fixture);
        write_materialx_sidecars(scene_output, &directory, &material_names)?;
        write_fixture_set(&directory, &fixture, &mut variant_report.files)?;
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&variant_report)?,
        )?;
        report.variants.push(variant_report);
    }
    if let Some(strelka_cli) = &options.strelka_cli {
        render_fixtures(strelka_cli, options, &mut report)?;
    }
    write_report(&options.output, &report)?;
    Ok(report)
}

fn unique_variants(npcs: &[NpcMetadata]) -> BTreeMap<String, Variant<'_>> {
    let mut result = BTreeMap::<String, Variant<'_>>::new();
    for npc in npcs {
        let identity = format!(
            "{}|{}|{:?}|{:?}",
            npc.display_id, npc.model_file_id, npc.texture_variations, npc.equipment
        );
        let key = format!(
            "{:05}_{}_{}",
            npc.display_id,
            safe_name(&npc.name),
            fnv1a(identity.as_bytes())
        );
        result
            .entry(key)
            .and_modify(|variant| variant.guids.push(npc.guid))
            .or_insert_with(|| Variant {
                npc,
                guids: vec![npc.guid],
            });
    }
    result
}

fn build_variant_report(
    scene: &Scene,
    scene_output: &Path,
    key: &str,
    variant: &Variant<'_>,
) -> Result<VariantReport> {
    let npc = variant.npc;
    let mesh_index = scene
        .nodes
        .get(npc.node)
        .and_then(|node| node.mesh)
        .context("NPC validation node has no mesh")?;
    let mesh = scene
        .meshes
        .get(mesh_index)
        .context("NPC validation mesh index is invalid")?;
    let mut material_indices: BTreeSet<usize> = mesh
        .primitives
        .iter()
        .map(|primitive| primitive.material)
        .collect();
    if let Some(equipment) = scene.equipment.get(&npc.node) {
        for item in equipment {
            if let Some(mesh) = scene.meshes.get(item.mesh) {
                material_indices.extend(mesh.primitives.iter().map(|primitive| primitive.material));
            }
        }
    }
    let mut issues = audit_materials(scene, scene_output, npc, &material_indices);
    let clips = audit_clips(
        mesh.skin.as_ref().map(|skin| skin.clips.as_slice()),
        &mut issues,
    );
    audit_equipment(scene, npc, &mut issues);
    Ok(VariantReport {
        key: key.to_owned(),
        name: npc.name.clone(),
        display_id: npc.display_id,
        model_file_id: npc.model_file_id,
        model_path: npc.model_path.clone(),
        texture_variations: npc.texture_variations.clone(),
        equipment: npc.equipment,
        guids: variant.guids.clone(),
        clips,
        issues,
        files: BTreeMap::new(),
        contact_sheet: None,
    })
}

fn audit_materials(
    scene: &Scene,
    scene_output: &Path,
    npc: &NpcMetadata,
    material_indices: &BTreeSet<usize>,
) -> Vec<AuditIssue> {
    let mut issues = Vec::new();
    let character = npc.model_path.starts_with("character/");
    let mut has_body = false;
    let mut has_hair = false;
    for index in material_indices {
        let Some(material) = scene.materials.get(*index) else {
            issues.push(AuditIssue {
                severity: "fail",
                code: "invalid_material_index",
                message: format!("material index {index} is out of range"),
            });
            continue;
        };
        if !matches!(material.kind, MaterialKind::M2) {
            continue;
        }
        let required = matches!(material.texture_type, Some(1..=8 | 11..=13));
        if material.texture_type == Some(1) && !material.layers.is_empty() {
            has_body = true;
        }
        if matches!(material.texture_type, Some(6..=8)) && !material.layers.is_empty() {
            has_hair = true;
        }
        if required && material.layers.is_empty() {
            issues.push(AuditIssue {
                severity: "fail",
                code: "missing_replaceable_texture",
                message: format!(
                    "{} has no texture for M2 texture type {:?}",
                    material.name, material.texture_type
                ),
            });
        }
        for layer in &material.layers {
            let path = scene_output.join(
                layer
                    .texture
                    .output_path
                    .replace('/', std::path::MAIN_SEPARATOR_STR),
            );
            if !path.is_file() {
                issues.push(AuditIssue {
                    severity: "fail",
                    code: "missing_texture_file",
                    message: format!("{} references missing {}", material.name, path.display()),
                });
            } else if image::image_dimensions(&path).is_ok_and(|size| size == (1, 1)) {
                issues.push(AuditIssue {
                    severity: "fail",
                    code: "placeholder_texture",
                    message: format!(
                        "{} uses a 1x1 placeholder {}",
                        material.name,
                        path.display()
                    ),
                });
            }
        }
    }
    if character && !has_body {
        issues.push(AuditIssue {
            severity: "fail",
            code: "missing_baked_body",
            message: "character model has no resolved Body texture".to_owned(),
        });
    }
    if character && !has_hair {
        issues.push(AuditIssue {
            severity: "warning",
            code: "missing_hair_texture",
            message: "character model has no resolved Hair/SkinExtra texture".to_owned(),
        });
    }
    issues
}

fn audit_clips(clips: Option<&[AnimationClip]>, issues: &mut Vec<AuditIssue>) -> Vec<ClipReport> {
    let mut result = Vec::new();
    for expected in ["Stand", "Walk", "Run"] {
        let clip =
            clips.and_then(|clips| clips.iter().find(|clip| clip.name.starts_with(expected)));
        let moves = clip.is_some_and(|clip| {
            clip.channels.iter().any(|channel| {
                channel.values.windows(2).any(|pair| {
                    pair[0]
                        .iter()
                        .zip(pair[1])
                        .any(|(left, right)| (*left - right).abs() > 1e-5)
                })
            })
        });
        if clip.is_none() || (expected != "Stand" && !moves) {
            issues.push(AuditIssue {
                severity: "fail",
                code: "missing_or_static_animation",
                message: format!("{expected} is missing or contains no changing channels"),
            });
        }
        if let Some(clip) = clip {
            for channel in &clip.channels {
                if channel.times.iter().any(|value| !value.is_finite())
                    || channel
                        .values
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                {
                    issues.push(AuditIssue {
                        severity: "fail",
                        code: "non_finite_animation",
                        message: format!("{} contains non-finite keys", clip.name),
                    });
                    break;
                }
            }
        }
        result.push(ClipReport {
            name: expected.to_owned(),
            present: clip.is_some(),
            fallback: clip.is_some_and(|clip| clip.fallback),
            channel_count: clip.map_or(0, |clip| clip.channels.len()),
            moves,
        });
    }
    result
}

fn audit_equipment(scene: &Scene, npc: &NpcMetadata, issues: &mut Vec<AuditIssue>) {
    let expected = npc.equipment.iter().filter(|item| **item != 0).count();
    let attached = scene.equipment.get(&npc.node).map_or(0, Vec::len);
    if attached < expected {
        issues.push(AuditIssue {
            severity: "warning",
            code: "missing_equipment_attachment",
            message: format!(
                "{expected} equipped item(s), but only {attached} attachment model(s) resolved"
            ),
        });
    }
    if let Some(mesh_index) = scene.nodes.get(npc.node).and_then(|node| node.mesh)
        && let Some(skin) = scene
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.skin.as_ref())
    {
        for item in scene.equipment.get(&npc.node).into_iter().flatten() {
            if !skin
                .attachments
                .iter()
                .any(|attachment| attachment.id == item.attachment_id)
            {
                issues.push(AuditIssue {
                    severity: "fail",
                    code: "missing_attachment_bone",
                    message: format!(
                        "item {} requests absent attachment {}",
                        item.item_id, item.attachment_id
                    ),
                });
            }
        }
    }
}

fn extract_fixture(source: &Value, root_node: usize, name: &str) -> Result<Value> {
    let source_nodes = array(source, "nodes")?;
    let mut reachable = BTreeSet::new();
    collect_nodes(source_nodes, root_node, &mut reachable)?;
    let node_map = index_map(reachable.iter().copied());
    let mesh_ids: BTreeSet<usize> = reachable
        .iter()
        .filter_map(|index| source_nodes[*index].get("mesh").and_then(Value::as_u64))
        .map(|index| index as usize)
        .collect();
    let skin_ids: BTreeSet<usize> = reachable
        .iter()
        .filter_map(|index| source_nodes[*index].get("skin").and_then(Value::as_u64))
        .map(|index| index as usize)
        .collect();
    let mesh_map = index_map(mesh_ids.iter().copied());
    let skin_map = index_map(skin_ids.iter().copied());
    let source_meshes = array(source, "meshes")?;
    let material_ids = mesh_material_ids(source_meshes, &mesh_ids);
    let material_map = index_map(material_ids.iter().copied());
    let animations = remap_animations(source, &reachable, &node_map)?;
    let mut accessor_ids = mesh_accessor_ids(source_meshes, &mesh_ids);
    for skin_id in &skin_ids {
        if let Some(accessor) = array(source, "skins")?[*skin_id]
            .get("inverseBindMatrices")
            .and_then(Value::as_u64)
        {
            accessor_ids.insert(accessor as usize);
        }
    }
    for animation in &animations {
        for sampler in array(animation, "samplers")? {
            accessor_ids.insert(sampler["input"].as_u64().unwrap_or(0) as usize);
            accessor_ids.insert(sampler["output"].as_u64().unwrap_or(0) as usize);
        }
    }
    let accessor_map = index_map(accessor_ids.iter().copied());
    let source_accessors = array(source, "accessors")?;
    let view_ids: BTreeSet<usize> = accessor_ids
        .iter()
        .filter_map(|index| {
            source_accessors[*index]
                .get("bufferView")
                .and_then(Value::as_u64)
        })
        .map(|index| index as usize)
        .collect();
    let view_map = index_map(view_ids.iter().copied());

    let mut nodes: Vec<Value> = reachable
        .iter()
        .map(|old| {
            let mut node = source_nodes[*old].clone();
            remap_array_field(&mut node, "children", &node_map);
            remap_scalar_field(&mut node, "mesh", &mesh_map);
            remap_scalar_field(&mut node, "skin", &skin_map);
            node
        })
        .collect();
    let fixture_root = *node_map
        .get(&root_node)
        .context("fixture root was not remapped")?;
    nodes[fixture_root]["translation"] = json!([0.0, 0.0, 0.0]);
    nodes[fixture_root]["rotation"] = json!([0.0, 0.0, 0.0, 1.0]);

    let meshes: Vec<Value> = mesh_ids
        .iter()
        .map(|old| {
            let mut mesh = source_meshes[*old].clone();
            for primitive in mesh["primitives"].as_array_mut().into_iter().flatten() {
                remap_scalar_field(primitive, "material", &material_map);
                remap_scalar_field(primitive, "indices", &accessor_map);
                if let Some(attributes) = primitive["attributes"].as_object_mut() {
                    for accessor in attributes.values_mut() {
                        if let Some(old) = accessor.as_u64()
                            && let Some(new) = accessor_map.get(&(old as usize))
                        {
                            *accessor = json!(new);
                        }
                    }
                }
            }
            mesh
        })
        .collect();
    let skins: Vec<Value> = skin_ids
        .iter()
        .map(|old| {
            let mut skin = array(source, "skins").unwrap()[*old].clone();
            remap_array_field(&mut skin, "joints", &node_map);
            remap_scalar_field(&mut skin, "skeleton", &node_map);
            remap_scalar_field(&mut skin, "inverseBindMatrices", &accessor_map);
            skin
        })
        .collect();
    let accessors: Vec<Value> = accessor_ids
        .iter()
        .map(|old| {
            let mut accessor = source_accessors[*old].clone();
            remap_scalar_field(&mut accessor, "bufferView", &view_map);
            accessor
        })
        .collect();
    let buffer_views: Vec<Value> = view_ids
        .iter()
        .map(|old| array(source, "bufferViews").unwrap()[*old].clone())
        .collect();
    let mut materials: Vec<Value> = material_ids
        .iter()
        .map(|old| array(source, "materials").unwrap()[*old].clone())
        .collect();
    let texture_ids = material_texture_ids(&materials);
    let texture_map = index_map(texture_ids.iter().copied());
    let source_textures = source
        .get("textures")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let image_ids: BTreeSet<usize> = texture_ids
        .iter()
        .filter_map(|index| source_textures.get(*index))
        .filter_map(|texture| texture.get("source").and_then(Value::as_u64))
        .map(|index| index as usize)
        .collect();
    let image_map = index_map(image_ids.iter().copied());
    for material in &mut materials {
        remap_material_textures(material, &texture_map);
        if let Some(path) = material
            .pointer_mut("/extras/materialx")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
        {
            material["extras"]["materialx"] = json!(format!("../../{path}"));
        }
    }
    let textures: Vec<Value> = texture_ids
        .iter()
        .filter_map(|old| source_textures.get(*old))
        .map(|source_texture| {
            let mut texture = source_texture.clone();
            remap_scalar_field(&mut texture, "source", &image_map);
            texture
        })
        .collect();
    let images: Vec<Value> = image_ids
        .iter()
        .filter_map(|old| source.get("images")?.as_array()?.get(*old))
        .map(|source_image| {
            let mut image = source_image.clone();
            if let Some(uri) = image.get("uri").and_then(Value::as_str) {
                image["uri"] = json!(format!("../../{uri}"));
            }
            image
        })
        .collect();
    let mut remapped_animations = animations;
    for animation in &mut remapped_animations {
        for sampler in animation["samplers"].as_array_mut().into_iter().flatten() {
            remap_scalar_field(sampler, "input", &accessor_map);
            remap_scalar_field(sampler, "output", &accessor_map);
        }
    }
    let (camera, camera_node) = validation_camera(source, source_meshes, &mesh_ids);
    let camera_node_index = nodes.len();
    nodes.push(camera_node);
    let has_images = !images.is_empty();
    let has_textures = !textures.is_empty();
    let mut fixture = json!({
        "asset": source["asset"].clone(),
        "buffers": [{
            "uri": "../../scene.bin",
            "byteLength": source["buffers"][0]["byteLength"]
        }],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "images": images,
        "textures": textures,
        "samplers": source.get("samplers").cloned().unwrap_or_else(|| json!([])),
        "materials": materials,
        "meshes": meshes,
        "nodes": nodes,
        "skins": skins,
        "animations": remapped_animations,
        "cameras": [camera],
        "scenes": [{
            "name": format!("{name} validation"),
            "nodes": [fixture_root, camera_node_index]
        }],
        "scene": 0
    });
    if let Some(extensions) = source.get("extensionsUsed") {
        fixture["extensionsUsed"] = extensions.clone();
    }
    if !has_images {
        fixture.as_object_mut().unwrap().remove("images");
    }
    if !has_textures {
        let object = fixture.as_object_mut().unwrap();
        object.remove("textures");
        object.remove("samplers");
    }
    Ok(fixture)
}

fn remap_animations(
    source: &Value,
    reachable: &BTreeSet<usize>,
    node_map: &BTreeMap<usize, usize>,
) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    for animation in source
        .get("animations")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        let source_samplers = array(animation, "samplers")?;
        let mut sampler_map = BTreeMap::new();
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for channel in array(animation, "channels")? {
            let Some(target) = channel.pointer("/target/node").and_then(Value::as_u64) else {
                continue;
            };
            if !reachable.contains(&(target as usize)) {
                continue;
            }
            let old_sampler = channel["sampler"].as_u64().unwrap_or(0) as usize;
            let sampler = *sampler_map.entry(old_sampler).or_insert_with(|| {
                let index = samplers.len();
                samplers.push(source_samplers[old_sampler].clone());
                index
            });
            let mut channel = channel.clone();
            channel["sampler"] = json!(sampler);
            channel["target"]["node"] = json!(node_map[&(target as usize)]);
            channels.push(channel);
        }
        if !channels.is_empty() {
            result.push(json!({
                "name": animation.get("name").cloned().unwrap_or_else(|| json!("clip")),
                "samplers": samplers,
                "channels": channels
            }));
        }
    }
    Ok(result)
}

fn write_fixture_set(
    directory: &Path,
    fixture: &Value,
    files: &mut BTreeMap<String, String>,
) -> Result<()> {
    fs::write(directory.join("scene.gltf"), serde_json::to_vec(fixture)?)?;
    files.insert("scene".to_owned(), "scene.gltf".to_owned());
    for clip in ["Stand", "Walk", "Run"] {
        let mut document = fixture.clone();
        let animations: Vec<Value> = fixture["animations"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|animation| animation["name"].as_str() == Some(clip))
            .cloned()
            .collect();
        document["animations"] = json!(animations);
        let file = format!("{}.gltf", clip.to_ascii_lowercase());
        fs::write(directory.join(&file), serde_json::to_vec(&document)?)?;
        files.insert(clip.to_ascii_lowercase(), file);
    }
    Ok(())
}

fn write_materialx_sidecars(
    scene_output: &Path,
    directory: &Path,
    material_names: &[String],
) -> Result<()> {
    let mut document =
        "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\" colorspace=\"lin_rec709\">\n"
            .to_owned();
    for name in material_names {
        let path = scene_output.join("materials").join(format!("{name}.mtlx"));
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(_) => continue,
        };
        let Some(start) = source
            .find(">\n")
            .and_then(|start| source[start + 2..].find(">\n").map(|next| start + next + 4))
        else {
            continue;
        };
        let end = source.rfind("</materialx>").unwrap_or(source.len());
        document.push_str(&source[start..end].replace("../textures/", "../../textures/"));
    }
    document.push_str("</materialx>\n");
    for stem in ["scene", "stand", "walk", "run"] {
        fs::write(directory.join(format!("{stem}.mtlx")), &document)?;
        let light = directory.join(format!("{stem}_light.json"));
        if light.is_file() {
            fs::remove_file(light)?;
        }
    }
    Ok(())
}

fn render_fixtures(
    strelka_cli: &Path,
    options: &ValidationOptions,
    report: &mut ValidationReport,
) -> Result<()> {
    for variant in &mut report.variants {
        let directory = options.output.join(&variant.key);
        let mut rendered = BTreeMap::<String, Vec<PathBuf>>::new();
        for clip in ["stand", "walk", "run"] {
            let scene = directory.join(format!("{clip}.gltf"));
            let prefix = directory.join(format!("{clip}.png"));
            let output = Command::new(strelka_cli)
                .arg(&scene)
                .arg("-o")
                .arg(directory.join(format!("{clip}_final.png")))
                .arg("-w")
                .arg(options.width.to_string())
                .arg("--height")
                .arg(options.height.to_string())
                .arg("--depth")
                .arg("1")
                .arg("--spp")
                .arg("1")
                .arg("--animation-frames")
                .arg(options.phases.len().to_string())
                .arg("--animation-fps")
                .arg(options.phases.len().to_string())
                .arg("--audit-frame-prefix")
                .arg(&prefix)
                .env("SPDLOG_LEVEL", "warn")
                .output()
                .with_context(|| format!("failed to launch {}", strelka_cli.display()))?;
            fs::write(
                directory.join(format!("{clip}.log")),
                [output.stdout.as_slice(), output.stderr.as_slice()].concat(),
            )?;
            if !output.status.success() {
                variant.issues.push(AuditIssue {
                    severity: "fail",
                    code: "render_failed",
                    message: format!("{clip} render exited with {}", output.status),
                });
                continue;
            }
            let paths: Vec<PathBuf> = (0..options.phases.len())
                .map(|index| directory.join(format!("{clip}-{:02}.png", index + 8)))
                .collect();
            if clip != "stand" && !images_move(&paths)? {
                variant.issues.push(AuditIssue {
                    severity: "fail",
                    code: "rendered_animation_static",
                    message: format!("{clip} phase images are effectively identical"),
                });
            }
            if let Some(message) = pose_instability(&paths)? {
                variant.issues.push(AuditIssue {
                    severity: "warning",
                    code: "pose_silhouette_instability",
                    message: format!("{clip}: {message}"),
                });
            }
            rendered.insert(clip.to_owned(), paths);
        }
        if !rendered.is_empty() {
            let contact = directory.join("contact_sheet.png");
            write_contact_sheet(options, &rendered, &contact)?;
            variant.contact_sheet = Some(format!("{}/contact_sheet.png", variant.key));
            if grayscale_score(&contact)? < 0.035 {
                variant.issues.push(AuditIssue {
                    severity: "warning",
                    code: "mostly_grayscale_render",
                    message: "contact sheet has unusually low color saturation".to_owned(),
                });
            }
        }
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(variant)?,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Silhouette {
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
}

fn pose_instability(paths: &[PathBuf]) -> Result<Option<String>> {
    let silhouettes: Vec<Silhouette> = paths
        .iter()
        .filter_map(|path| image::open(path).ok())
        .filter_map(|image| silhouette(&image.to_rgba8()))
        .collect();
    let Some(reference) = silhouettes.first().copied() else {
        return Ok(Some("no visible foreground silhouette".to_owned()));
    };
    let reference_area = reference.width * reference.height;
    let reference_aspect = reference.width / reference.height.max(1.0);
    for (phase, value) in silhouettes.iter().enumerate().skip(1) {
        let area_ratio = (value.width * value.height) / reference_area.max(1.0);
        let aspect_ratio = (value.width / value.height.max(1.0)) / reference_aspect.max(0.01);
        let center_distance = ((value.center_x - reference.center_x).powi(2)
            + (value.center_y - reference.center_y).powi(2))
        .sqrt();
        let reference_diagonal = (reference.width.powi(2) + reference.height.powi(2)).sqrt();
        if !(0.3..=3.0).contains(&area_ratio) {
            return Ok(Some(format!(
                "phase {phase} changes projected area by {area_ratio:.2}x"
            )));
        }
        if !(0.4..=2.5).contains(&aspect_ratio) {
            return Ok(Some(format!(
                "phase {phase} changes silhouette aspect by {aspect_ratio:.2}x"
            )));
        }
        if center_distance > reference_diagonal * 0.35 {
            return Ok(Some(format!(
                "phase {phase} shifts silhouette center by {center_distance:.1}px"
            )));
        }
    }
    Ok(None)
}

fn silhouette(image: &RgbaImage) -> Option<Silhouette> {
    let mut min_x = image.width();
    let mut min_y = image.height();
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut count = 0u64;
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    for (x, y, pixel) in image.enumerate_pixels() {
        let brightness = *pixel.0[..3].iter().max().unwrap_or(&0);
        if brightness < 12 || pixel.0[3] == 0 {
            continue;
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        count += 1;
        sum_x += u64::from(x);
        sum_y += u64::from(y);
    }
    (count != 0).then_some(Silhouette {
        width: (max_x.saturating_sub(min_x) + 1) as f32,
        height: (max_y.saturating_sub(min_y) + 1) as f32,
        center_x: sum_x as f32 / count as f32,
        center_y: sum_y as f32 / count as f32,
    })
}

fn images_move(paths: &[PathBuf]) -> Result<bool> {
    let images: Vec<RgbaImage> = paths
        .iter()
        .filter_map(|path| image::open(path).ok())
        .map(DynamicImage::into_rgba8)
        .collect();
    if images.len() < 2 {
        return Ok(false);
    }
    let base = &images[0];
    Ok(images[1..].iter().any(|image| {
        base.pixels()
            .zip(image.pixels())
            .map(|(left, right)| {
                left.0
                    .iter()
                    .zip(right.0)
                    .map(|(left, right)| left.abs_diff(right) as u64)
                    .sum::<u64>()
            })
            .sum::<u64>()
            > 100
    }))
}

fn write_contact_sheet(
    options: &ValidationOptions,
    rendered: &BTreeMap<String, Vec<PathBuf>>,
    path: &Path,
) -> Result<()> {
    let columns = options.phases.len().max(1) as u32;
    let rows = 3u32;
    let mut sheet = RgbaImage::new(options.width * columns, options.height * rows);
    for (row, clip) in ["stand", "walk", "run"].iter().enumerate() {
        for (column, image_path) in rendered.get(*clip).into_iter().flatten().enumerate() {
            if let Ok(image) = image::open(image_path) {
                imageops::replace(
                    &mut sheet,
                    &image.to_rgba8(),
                    (column as u32 * options.width) as i64,
                    (row as u32 * options.height) as i64,
                );
            }
        }
    }
    sheet.save(path)?;
    Ok(())
}

fn grayscale_score(path: &Path) -> Result<f32> {
    let image = image::open(path)?.to_rgb8();
    let mut saturation = 0.0f64;
    let mut samples = 0u64;
    for pixel in image.pixels() {
        let min = *pixel.0.iter().min().unwrap_or(&0) as f64;
        let max = *pixel.0.iter().max().unwrap_or(&0) as f64;
        if max < 12.0 {
            continue;
        }
        saturation += (max - min) / max;
        samples += 1;
    }
    Ok(if samples == 0 {
        0.0
    } else {
        (saturation / samples as f64) as f32
    })
}

fn write_report(output: &Path, report: &ValidationReport) -> Result<()> {
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(report)?,
    )?;
    let mut html = "<!doctype html><meta charset=\"utf-8\"><title>NPC validation</title><h1>NPC validation</h1>".to_owned();
    for variant in &report.variants {
        let failures = variant
            .issues
            .iter()
            .filter(|issue| issue.severity == "fail")
            .count();
        html.push_str(&format!(
            "<section><h2>{} — display {} — {} failure(s)</h2>",
            html_escape(&variant.name),
            variant.display_id,
            failures
        ));
        if let Some(contact) = &variant.contact_sheet {
            html.push_str(&format!(
                "<a href=\"{}\"><img loading=\"lazy\" width=\"768\" src=\"{}\"></a>",
                html_escape(contact),
                html_escape(contact)
            ));
        }
        html.push_str("<ul>");
        for issue in &variant.issues {
            html.push_str(&format!(
                "<li>{}: {} — {}</li>",
                issue.severity,
                issue.code,
                html_escape(&issue.message)
            ));
        }
        html.push_str("</ul></section>");
    }
    fs::write(output.join("report.html"), html)?;
    Ok(())
}

fn validation_camera(
    source: &Value,
    meshes: &[Value],
    mesh_ids: &BTreeSet<usize>,
) -> (Value, Value) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for mesh_id in mesh_ids {
        for primitive in meshes[*mesh_id]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let Some(position) = primitive
                .pointer("/attributes/POSITION")
                .and_then(Value::as_u64)
            else {
                continue;
            };
            let accessor = &source["accessors"][position as usize];
            if let (Some(low), Some(high)) = (vec3(accessor.get("min")), vec3(accessor.get("max")))
            {
                min = min.min(low);
                max = max.max(high);
            }
        }
    }
    if !min.is_finite() || !max.is_finite() {
        min = Vec3::new(-1.0, 0.0, -1.0);
        max = Vec3::new(1.0, 2.0, 1.0);
    }
    let center = (min + max) * 0.5;
    let extent = (max - min).max(Vec3::splat(0.1));
    let radius = extent.length().max(1.0);
    let eye = center + Vec3::new(radius * 0.85, radius * 0.25, radius * 1.25);
    let transform = look_at_transform(eye, center);
    (
        json!({
            "name": "validation",
            "type": "perspective",
            "perspective": {
                "yfov": 45.0f32.to_radians(),
                "znear": 0.01,
                "zfar": radius * 20.0
            }
        }),
        json!({
            "name": "camera_validation",
            "camera": 0,
            "translation": transform.translation,
            "rotation": transform.rotation,
            "scale": transform.scale
        }),
    )
}

fn look_at_transform(eye: Vec3, target: Vec3) -> Transform {
    let world = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
    Transform {
        translation: eye.to_array(),
        rotation: Quat::from_mat4(&world).normalize().to_array(),
        scale: [1.0; 3],
    }
}

fn collect_nodes(nodes: &[Value], index: usize, result: &mut BTreeSet<usize>) -> Result<()> {
    if index >= nodes.len() {
        bail!("node {index} is out of range");
    }
    if !result.insert(index) {
        return Ok(());
    }
    for child in nodes[index]["children"].as_array().into_iter().flatten() {
        if let Some(child) = child.as_u64() {
            collect_nodes(nodes, child as usize, result)?;
        }
    }
    Ok(())
}

fn mesh_material_ids(meshes: &[Value], mesh_ids: &BTreeSet<usize>) -> BTreeSet<usize> {
    mesh_ids
        .iter()
        .flat_map(|index| {
            meshes[*index]["primitives"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter_map(|primitive| primitive.get("material").and_then(Value::as_u64))
        .map(|index| index as usize)
        .collect()
}

fn mesh_accessor_ids(meshes: &[Value], mesh_ids: &BTreeSet<usize>) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    for mesh_id in mesh_ids {
        for primitive in meshes[*mesh_id]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(index) = primitive.get("indices").and_then(Value::as_u64) {
                result.insert(index as usize);
            }
            for accessor in primitive["attributes"]
                .as_object()
                .into_iter()
                .flat_map(|value| value.values())
            {
                if let Some(index) = accessor.as_u64() {
                    result.insert(index as usize);
                }
            }
        }
    }
    result
}

fn material_texture_ids(materials: &[Value]) -> BTreeSet<usize> {
    let pointers = [
        "/pbrMetallicRoughness/baseColorTexture/index",
        "/pbrMetallicRoughness/metallicRoughnessTexture/index",
        "/normalTexture/index",
        "/emissiveTexture/index",
        "/occlusionTexture/index",
    ];
    materials
        .iter()
        .flat_map(|material| {
            pointers
                .iter()
                .filter_map(|pointer| material.pointer(pointer))
        })
        .filter_map(Value::as_u64)
        .map(|index| index as usize)
        .collect()
}

fn remap_material_textures(material: &mut Value, map: &BTreeMap<usize, usize>) {
    let pointers = [
        "/pbrMetallicRoughness/baseColorTexture/index",
        "/pbrMetallicRoughness/metallicRoughnessTexture/index",
        "/normalTexture/index",
        "/emissiveTexture/index",
        "/occlusionTexture/index",
    ];
    for pointer in pointers {
        if let Some(value) = material.pointer_mut(pointer)
            && let Some(old) = value.as_u64()
            && let Some(new) = map.get(&(old as usize))
        {
            *value = json!(new);
        }
    }
}

fn fixture_material_names(fixture: &Value) -> Vec<String> {
    fixture["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|material| material.get("name").and_then(Value::as_str))
        .map(str::to_owned)
        .collect()
}

fn index_map(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    values
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect()
}

fn remap_scalar_field(value: &mut Value, field: &str, map: &BTreeMap<usize, usize>) {
    if let Some(old) = value.get(field).and_then(Value::as_u64)
        && let Some(new) = map.get(&(old as usize))
    {
        value[field] = json!(new);
    }
}

fn remap_array_field(value: &mut Value, field: &str, map: &BTreeMap<usize, usize>) {
    if let Some(values) = value.get_mut(field).and_then(Value::as_array_mut) {
        *values = values
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|old| map.get(&(old as usize)))
            .map(|new| json!(new))
            .collect();
    }
}

fn array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value]> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .with_context(|| format!("glTF field '{field}' is not an array"))
}

fn vec3(value: Option<&Value>) -> Option<Vec3> {
    let value = value?.as_array()?;
    Some(Vec3::new(
        value.first()?.as_f64()? as f32,
        value.get(1)?.as_f64()? as f32,
        value.get(2)?.as_f64()? as f32,
    ))
}

fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_owned()
}

fn fnv1a(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::{fnv1a, safe_name};

    #[test]
    fn validation_keys_are_stable_and_filesystem_safe() {
        assert_eq!(fnv1a(b"display|model"), fnv1a(b"display|model"));
        assert_eq!(safe_name("Horde Guard"), "horde_guard");
    }
}
