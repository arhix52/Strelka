use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Cursor, Seek, SeekFrom};

use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Mat3, Mat4, Quat, Vec3};
use wow_adt::{AlphaFormat, AlphaMap, ParsedAdt, RootAdt, parse_adt};
use wow_m2::chunks::M2Vertex;
use wow_m2::chunks::material::M2Material;
use wow_m2::header::M2Header;
use wow_m2::skin::{OldSkinHeader, SkinG};
use wow_m2::{SkinFile, parse_skin};
use wow_wmo::{ParsedWmo, parse_wmo};

use crate::ground_effect::{GroundEffectCatalog, scatter};
use crate::ir::{
    BlendMode, Camera, ColorSpace, FogMetadata, GroundEffectMetadata, LightMetadata,
    LiquidMetadata, Material, MaterialKind, MaterialMetadata, Mesh, Primitive, Scene,
    TerrainMetadata, TextureLayer, TextureRef, Transform, Vertex, WmoMetadata,
};
use crate::liquid::{adt_surfaces, wmo_surface};
use crate::source::{AssetSource, normalize};

const CHUNK_SIZE: f32 = 533.333_3 / 16.0;
pub(crate) const GRID_STEP: f32 = CHUNK_SIZE / 8.0;
const TILE_SIZE: f32 = 533.333_3;
const MAP_ZERO_POINT: f32 = 32.0 * TILE_SIZE;

struct ImportedM2 {
    vertices: Vec<Vertex>,
    materials: Vec<M2MaterialInfo>,
    texture_lookup: Vec<u16>,
    texture_names: Vec<String>,
}

struct M2MaterialInfo {
    flags: u16,
    blend_mode: u16,
}

pub struct Importer<'a> {
    source: &'a dyn AssetSource,
    scene: Scene,
    asset_meshes: HashMap<String, usize>,
    failed_assets: HashSet<String>,
    liquid_materials: HashMap<u16, usize>,
    ground_effects: Option<GroundEffectCatalog>,
    ground_stats: BTreeMap<u32, (HashSet<u32>, u32, usize)>,
    origin: Option<[f32; 3]>,
}

impl<'a> Importer<'a> {
    pub fn new(source: &'a dyn AssetSource) -> Self {
        Self {
            source,
            scene: Scene::default(),
            asset_meshes: HashMap::new(),
            failed_assets: HashSet::new(),
            liquid_materials: HashMap::new(),
            ground_effects: None,
            ground_stats: BTreeMap::new(),
            origin: None,
        }
    }

    pub fn import_tiles(mut self, map: &str, tiles: &[[u32; 2]]) -> Result<Scene> {
        match GroundEffectCatalog::load(self.source) {
            Ok(catalog) => self.ground_effects = Some(catalog),
            Err(error) => self
                .scene
                .metadata
                .warnings
                .push(format!("Ground effects unavailable: {error:#}")),
        }
        for tile in tiles {
            eprintln!("Importing ADT tile {},{}...", tile[0], tile[1]);
            self.import_tile(map, *tile)
                .with_context(|| format!("failed to import ADT tile {},{}", tile[0], tile[1]))?;
        }
        for (effect_id, (models, density, instance_count)) in &self.ground_stats {
            self.scene
                .metadata
                .ground_effects
                .push(GroundEffectMetadata {
                    effect_id: *effect_id,
                    model_file_ids: models.iter().copied().collect(),
                    density: *density,
                    instance_count: *instance_count,
                });
        }
        self.scene.metadata.source.origin = self.origin.unwrap_or([0.0; 3]);
        add_default_cameras(&mut self.scene);
        Ok(self.scene)
    }

    fn import_tile(&mut self, map: &str, tile: [u32; 2]) -> Result<()> {
        let base = format!("world/maps/{map}/{map}_{}_{}", tile[0], tile[1]);
        let bytes = self
            .source
            .read(&format!("{base}.adt"))
            .with_context(|| format!("unable to load ADT tile {},{}", tile[0], tile[1]))?;
        let mut cursor = Cursor::new(&bytes);
        let parsed = match parse_adt(&mut cursor) {
            Ok(parsed) => parsed,
            Err(_) => {
                let sanitized = sanitize_root_adt(bytes)?;
                parse_adt(&mut Cursor::new(sanitized))
                    .context("failed to parse root ADT after compatibility repair")?
            }
        };
        let mut root = match parsed {
            ParsedAdt::Root(root) => root,
            other => bail!("{} is {:?}, not a root ADT", base, other.file_type()),
        };
        let texture_path = format!("{base}_tex0.adt");
        if self.source.contains(&texture_path) {
            let texture_bytes =
                adapt_texture_fdid_chunks(self.source, self.source.read(&texture_path)?)
                    .with_context(|| format!("failed to adapt {texture_path}"))?;
            let mut cursor = Cursor::new(texture_bytes);
            match parse_adt(&mut cursor) {
                Ok(ParsedAdt::Tex0(texture)) => {
                    root.textures = texture.textures;
                    for source_chunk in texture.mcnk_textures {
                        if let Some(target_chunk) = root.mcnk_chunks.get_mut(source_chunk.index) {
                            target_chunk.layers = source_chunk.layers;
                            target_chunk.alpha = source_chunk.alpha_maps;
                        }
                    }
                }
                Ok(other) => self.scene.metadata.warnings.push(format!(
                    "{texture_path} parsed as {:?}, texture layers skipped",
                    other.file_type()
                )),
                Err(error) => self
                    .scene
                    .metadata
                    .warnings
                    .push(format!("{texture_path}: {error}")),
            }
        }
        if self.origin.is_none() {
            self.origin = root.mcnk_chunks.first().map(|chunk| {
                [
                    chunk.header.position[1],
                    chunk.header.position[2],
                    chunk.header.position[0],
                ]
            });
        }
        self.import_terrain(&root, tile)?;
        self.import_adt_liquids(&root, tile);
        self.import_ground_effects(&root, tile);
        self.import_placements(&root.models, &root.doodad_placements, false)?;
        self.import_wmo_placements(&root.wmos, &root.wmo_placements)?;

        let obj_path = format!("{base}_obj0.adt");
        if self.source.contains(&obj_path) {
            let mut cursor = Cursor::new(
                self.source
                    .read(&obj_path)
                    .with_context(|| format!("failed to read {obj_path}"))?,
            );
            match parse_adt(&mut cursor) {
                Ok(ParsedAdt::Obj0(objects)) => {
                    self.import_placements(&objects.models, &objects.doodad_placements, false)?;
                    self.import_wmo_placements(&objects.wmos, &objects.wmo_placements)?;
                }
                Ok(other) => self.scene.metadata.warnings.push(format!(
                    "{obj_path} parsed as {:?}, object placements skipped",
                    other.file_type()
                )),
                Err(error) => self
                    .scene
                    .metadata
                    .warnings
                    .push(format!("{obj_path}: {error}")),
            }
        }
        Ok(())
    }

    fn import_terrain(&mut self, root: &RootAdt, tile: [u32; 2]) -> Result<()> {
        let origin = self.origin.unwrap_or([0.0; 3]);
        for chunk in &root.mcnk_chunks {
            let Some(heights) = &chunk.heights else {
                continue;
            };
            if heights.heights.len() != 145 {
                continue;
            }
            let material = self.add_terrain_material(root, chunk, tile)?;
            let mut vertices = Vec::with_capacity(145);
            for row in 0..17 {
                let inner = row % 2 != 0;
                let logical_y = row as f32 * 0.5;
                let count = if inner { 8 } else { 9 };
                let row_start = (row / 2) * 17 + if inner { 9 } else { 0 };
                for x in 0..count {
                    let index = row_start + x;
                    let logical_x = x as f32 + if inner { 0.5 } else { 0.0 };
                    let normal = chunk
                        .normals
                        .as_ref()
                        .and_then(|normals| normals.normals.get(index))
                        .map(|normal| normal.to_normalized())
                        .unwrap_or([0.0, 1.0, 0.0]);
                    let color = chunk
                        .vertex_colors
                        .as_ref()
                        .and_then(|colors| colors.colors.get(index))
                        .map(|color| {
                            [
                                color.r.saturating_mul(2),
                                color.g.saturating_mul(2),
                                color.b.saturating_mul(2),
                                color.a,
                            ]
                        })
                        .unwrap_or([254, 254, 254, 255]);
                    let wow_x = chunk.header.position[1] - logical_x * GRID_STEP;
                    let wow_z = chunk.header.position[0] - logical_y * GRID_STEP;
                    let wow_y = chunk.header.position[2] + heights.heights[index];
                    vertices.push(Vertex {
                        position: [wow_x - origin[0], wow_y - origin[1], wow_z - origin[2]],
                        normal,
                        uv0: [logical_x / 8.0, logical_y / 8.0],
                        color0: color,
                    });
                }
            }
            let indices = terrain_indices();
            let mesh_name = format!(
                "terrain_{}_{}_{}_{}",
                tile[0], tile[1], chunk.header.index_x, chunk.header.index_y
            );
            let mesh = self.scene.meshes.len();
            self.scene.meshes.push(Mesh {
                name: mesh_name.clone(),
                source: format!("ADT {}_{}", tile[0], tile[1]),
                vertices,
                indices,
                primitives: vec![Primitive {
                    first_index: 0,
                    index_count: 8 * 8 * 12,
                    material,
                }],
            });
            self.scene.nodes.push(crate::ir::Node {
                name: mesh_name,
                mesh: Some(mesh),
                transform: Transform::IDENTITY,
                children: Vec::new(),
            });
            self.scene.metadata.terrain.push(TerrainMetadata {
                tile,
                chunk: [chunk.header.index_x, chunk.header.index_y],
                area_id: chunk.header.area_id,
                material: self.scene.materials[material].name.clone(),
                vertex_color_attribute: "COLOR_0".to_owned(),
                layer_count: self.scene.materials[material].layers.len(),
            });
        }
        Ok(())
    }

    fn import_adt_liquids(&mut self, root: &RootAdt, tile: [u32; 2]) {
        let origin = self.origin.unwrap_or([0.0; 3]);
        for surface in adt_surfaces(root, tile, origin) {
            let material = self.add_liquid_material(surface.liquid_type);
            let mesh = self.scene.meshes.len();
            self.scene.meshes.push(Mesh {
                name: surface.name.clone(),
                source: format!("ADT {}_{} liquid", tile[0], tile[1]),
                vertices: surface.vertices,
                indices: surface.indices,
                primitives: vec![Primitive {
                    first_index: 0,
                    index_count: 0,
                    material,
                }],
            });
            let index_count = self.scene.meshes[mesh].indices.len();
            self.scene.meshes[mesh].primitives[0].index_count = index_count;
            self.scene.nodes.push(crate::ir::Node {
                name: surface.name,
                mesh: Some(mesh),
                transform: Transform::IDENTITY,
                children: Vec::new(),
            });
            self.scene.metadata.liquids.push(LiquidMetadata {
                source: format!("ADT {}_{}", tile[0], tile[1]),
                tile: Some(tile),
                chunk: surface.chunk,
                layer: surface.layer,
                liquid_type: surface.liquid_type,
                fishable: surface.fishable,
                deep: surface.deep,
            });
        }
    }

    fn add_liquid_material(&mut self, liquid_type: u16) -> usize {
        if let Some(material) = self.liquid_materials.get(&liquid_type) {
            return *material;
        }
        let name = format!("liquid_type_{liquid_type}");
        let index = self.scene.materials.len();
        self.scene.materials.push(Material {
            name: name.clone(),
            source: format!("LiquidType {liquid_type}"),
            kind: MaterialKind::Liquid,
            layers: Vec::new(),
            normal: None,
            emissive: None,
            blend: BlendMode::Blend,
            double_sided: true,
            unlit: false,
        });
        self.scene.metadata.materials.insert(
            name.clone(),
            MaterialMetadata {
                document: format!("materials/{name}.mtlx"),
                source: format!("LiquidType {liquid_type}"),
                blend: BlendMode::Blend,
                double_sided: true,
                unlit: false,
            },
        );
        self.liquid_materials.insert(liquid_type, index);
        index
    }

    fn import_ground_effects(&mut self, root: &RootAdt, tile: [u32; 2]) {
        let Some(catalog) = self.ground_effects.as_ref() else {
            return;
        };
        let origin = self.origin.unwrap_or([0.0; 3]);
        let mut placements = Vec::new();
        for chunk in &root.mcnk_chunks {
            placements.extend(scatter(chunk, catalog, tile, origin));
        }
        for placement in placements {
            let Some(path) = self.source.path_for_fdid(placement.model_file_id) else {
                continue;
            };
            let key = normalize(&path);
            if self.failed_assets.contains(&key) {
                continue;
            }
            match self.import_m2(&path) {
                Ok(Some(mesh)) => {
                    self.scene
                        .instances
                        .entry(mesh)
                        .or_default()
                        .push(Transform {
                            translation: placement.position,
                            rotation: Quat::from_rotation_y(placement.yaw).to_array(),
                            scale: [placement.scale; 3],
                        });
                    let density = self
                        .ground_effects
                        .as_ref()
                        .and_then(|catalog| catalog.effect(placement.effect_id))
                        .map_or(0, |effect| effect.density);
                    let stat = self
                        .ground_stats
                        .entry(placement.effect_id)
                        .or_insert_with(|| (HashSet::new(), density, 0));
                    stat.0.insert(placement.model_file_id);
                    stat.2 += 1;
                }
                Ok(None) => {}
                Err(error) => {
                    self.failed_assets.insert(key);
                    self.scene
                        .metadata
                        .warnings
                        .push(format!("Ground effect M2 {path}: {error:#}"));
                }
            }
        }
    }

    fn add_terrain_material(
        &mut self,
        root: &RootAdt,
        chunk: &wow_adt::McnkChunk,
        tile: [u32; 2],
    ) -> Result<usize> {
        let name = format!(
            "terrain_{}_{}_{}_{}",
            tile[0], tile[1], chunk.header.index_x, chunk.header.index_y
        );
        let mut layers = Vec::new();
        if let Some(layer_chunk) = &chunk.layers {
            for (layer_index, layer) in layer_chunk.layers.iter().enumerate() {
                let Some(texture_path) = root.textures.get(layer.texture_id as usize) else {
                    continue;
                };
                let alpha_map = if layer_index == 0 || chunk.alpha.is_none() {
                    None
                } else {
                    Some(format!("textures/alpha/{}_layer_{}.png", name, layer_index))
                };
                let alpha_data = decode_alpha(chunk, layer_chunk, layer_index);
                layers.push(TextureLayer {
                    texture: texture_ref(texture_path, ColorSpace::Srgb),
                    alpha_map,
                    alpha_data,
                    effect_id: Some(layer.effect_id),
                    flags: layer.flags.value,
                });
            }
        }
        if layers.is_empty() {
            layers.push(TextureLayer {
                texture: TextureRef {
                    wow_path: String::new(),
                    output_path: String::new(),
                    color_space: ColorSpace::Srgb,
                },
                alpha_map: None,
                alpha_data: None,
                effect_id: None,
                flags: 0,
            });
        }
        let index = self.scene.materials.len();
        self.scene.materials.push(Material {
            name: name.clone(),
            source: format!("ADT {}_{}", tile[0], tile[1]),
            kind: MaterialKind::Terrain,
            layers,
            normal: None,
            emissive: None,
            blend: BlendMode::Opaque,
            double_sided: false,
            unlit: false,
        });
        self.scene.metadata.materials.insert(
            name.clone(),
            MaterialMetadata {
                document: format!("materials/{name}.mtlx"),
                source: format!("ADT {}_{}", tile[0], tile[1]),
                blend: BlendMode::Opaque,
                double_sided: false,
                unlit: false,
            },
        );
        Ok(index)
    }

    fn import_placements(
        &mut self,
        models: &[String],
        placements: &[wow_adt::DoodadPlacement],
        _inside_wmo: bool,
    ) -> Result<()> {
        for placement in placements {
            let resolved = if placement.uses_file_data_id() {
                self.source.path_for_fdid(placement.name_id)
            } else {
                models.get(placement.name_id as usize).cloned()
            };
            let Some(path) = resolved else {
                self.scene.metadata.warnings.push(format!(
                    "M2 placement {} references missing model {}",
                    placement.unique_id, placement.name_id
                ));
                continue;
            };
            let key = normalize(&path);
            if self.failed_assets.contains(&key) {
                continue;
            }
            match self.import_m2(&path) {
                Ok(Some(mesh)) => {
                    let transform = self.placement_transform(
                        placement.position,
                        placement.rotation,
                        placement.get_scale(),
                    );
                    self.scene
                        .instances
                        .entry(mesh)
                        .or_default()
                        .push(transform);
                }
                Ok(None) => {}
                Err(error) => {
                    self.failed_assets.insert(key);
                    self.scene
                        .metadata
                        .warnings
                        .push(format!("M2 {path}: {error:#}"));
                }
            }
        }
        Ok(())
    }

    fn import_wmo_placements(
        &mut self,
        wmos: &[String],
        placements: &[wow_adt::WmoPlacement],
    ) -> Result<()> {
        for placement in placements {
            let resolved = if placement.uses_file_data_id() {
                self.source.path_for_fdid(placement.name_id)
            } else {
                wmos.get(placement.name_id as usize).cloned()
            };
            let Some(path) = resolved else {
                self.scene.metadata.warnings.push(format!(
                    "WMO placement {} references missing object {}",
                    placement.unique_id, placement.name_id
                ));
                continue;
            };
            let key = normalize(&path);
            if self.failed_assets.contains(&key) {
                continue;
            }
            match self.import_wmo(&path, placement.doodad_set) {
                Ok(mesh) => {
                    let transform = self.placement_transform(
                        placement.position,
                        placement.rotation,
                        placement.get_scale(),
                    );
                    self.scene
                        .instances
                        .entry(mesh)
                        .or_default()
                        .push(transform);
                    self.scene.metadata.wmo.push(WmoMetadata {
                        source: path.clone(),
                        doodad_set: placement.doodad_set,
                        name_set: placement.name_set,
                    });
                }
                Err(error) => {
                    self.failed_assets.insert(key);
                    self.scene
                        .metadata
                        .warnings
                        .push(format!("WMO {path}: {error:#}"));
                }
            }
        }
        Ok(())
    }

    fn import_m2(&mut self, wow_path: &str) -> Result<Option<usize>> {
        let key = normalize(wow_path);
        if let Some(index) = self.asset_meshes.get(&key) {
            return Ok(Some(*index));
        }
        let bytes = self
            .source
            .read(&key)
            .with_context(|| format!("failed to read M2 {key}"))?;
        let adapted =
            adapt_m2_chunks(bytes).with_context(|| format!("failed to adapt chunked M2 {key}"))?;
        let model = parse_imported_m2(&adapted.model, adapted.chunked)
            .with_context(|| format!("failed to parse M2 {key}"))?;
        let skin_path = adapted
            .skin_fdids
            .first()
            .and_then(|fdid| self.source.path_for_fdid(*fdid))
            .unwrap_or_else(|| replace_extension(&key, "00.skin"));
        let skin_bytes = self
            .source
            .read(&skin_path)
            .with_context(|| format!("failed to read skin {skin_path}"))?;
        let skin = match parse_skin(&mut Cursor::new(&skin_bytes)) {
            Ok(skin) => skin,
            Err(full_error) => SkinG::<OldSkinHeader>::parse(&mut Cursor::new(&skin_bytes))
                .map(SkinFile::Old)
                .with_context(|| {
                    format!("failed to parse skin {skin_path}; auto parser failed: {full_error}")
                })?,
        };
        let mut indices: Vec<u32> = skin
            .get_resolved_indices()
            .into_iter()
            .map(u32::from)
            .collect();
        indices.truncate(indices.len() - indices.len() % 3);
        let (triangles, _) = indices.as_chunks::<3>();
        indices = triangles
            .iter()
            .filter(|triangle| {
                triangle
                    .iter()
                    .all(|index| *index < model.vertices.len() as u32)
            })
            .flatten()
            .copied()
            .collect();
        if model.vertices.is_empty() || indices.is_empty() {
            self.failed_assets.insert(key);
            return Ok(None);
        }
        let material_base = self.scene.materials.len();
        let texture_paths: Vec<String> = adapted
            .texture_fdids
            .iter()
            .map(|fdid| {
                self.source
                    .path_for_fdid(*fdid)
                    .unwrap_or_else(|| format!("fdid/{fdid}.blp"))
            })
            .collect();
        self.add_m2_materials(&key, &model, &skin, &texture_paths);
        let primitives = m2_primitives(&skin, material_base, model.materials.len(), indices.len());
        let mesh = self.scene.meshes.len();
        self.scene.meshes.push(Mesh {
            name: asset_name(&key),
            source: key.clone(),
            vertices: model.vertices,
            indices,
            primitives,
        });
        self.asset_meshes.insert(key, mesh);
        Ok(Some(mesh))
    }

    fn add_m2_materials(
        &mut self,
        path: &str,
        model: &ImportedM2,
        skin: &SkinFile,
        texture_paths: &[String],
    ) {
        if model.materials.is_empty() {
            self.add_object_material(
                path,
                0,
                MaterialKind::M2,
                None,
                BlendMode::Opaque,
                false,
                false,
            );
            return;
        }
        for (index, material) in model.materials.iter().enumerate() {
            let texture_combo = skin
                .batches()
                .iter()
                .find(|batch| batch.material_index as usize == index)
                .map(|batch| batch.texture_combo_index as usize)
                .unwrap_or(index);
            let texture_index = model
                .texture_lookup
                .get(texture_combo)
                .copied()
                .map(usize::from)
                .unwrap_or(texture_combo);
            let texture = model
                .texture_names
                .get(texture_index)
                .cloned()
                .filter(|texture| !texture.is_empty())
                .or_else(|| texture_paths.get(texture_index).cloned());
            let blend = match material.blend_mode {
                0 => BlendMode::Opaque,
                1 => BlendMode::Mask,
                2 => BlendMode::Blend,
                3 | 4 | 7 => BlendMode::Additive,
                _ => BlendMode::Modulate,
            };
            self.add_object_material(
                path,
                index,
                MaterialKind::M2,
                texture.as_deref(),
                blend,
                material.flags & 0x04 != 0,
                material.flags & 0x01 != 0,
            );
        }
    }

    fn import_wmo(&mut self, wow_path: &str, _doodad_set: u16) -> Result<usize> {
        let key = normalize(wow_path);
        if let Some(index) = self.asset_meshes.get(&key) {
            return Ok(*index);
        }
        let mut cursor = Cursor::new(self.source.read(&key)?);
        let root = match parse_wmo(&mut cursor)? {
            ParsedWmo::Root(root) => root,
            ParsedWmo::Group(_) => bail!("{key} is a WMO group, not a root"),
        };
        let mut groups = Vec::new();
        for index in 0..root.n_groups {
            let group_path = wmo_group_path(&key, index);
            if !self.source.contains(&group_path) {
                continue;
            }
            let group_bytes = self.source.read(&group_path)?;
            let liquid = wmo_surface(&group_bytes, &format!("{}_{}", asset_name(&key), index))
                .with_context(|| format!("failed to parse WMO liquid {group_path}"))?;
            let mut cursor = Cursor::new(group_bytes);
            if let ParsedWmo::Group(group) = parse_wmo(&mut cursor)? {
                groups.push((group, liquid));
            }
        }
        let material_base = self.scene.materials.len();
        if root.materials.is_empty() {
            self.add_object_material(
                &key,
                0,
                MaterialKind::Wmo,
                None,
                BlendMode::Opaque,
                false,
                false,
            );
        }
        for (index, material) in root.materials.iter().enumerate() {
            let texture = root
                .texture_offset_index_map
                .get(&material.texture_1)
                .and_then(|texture| root.textures.get(*texture as usize))
                .cloned()
                .or_else(|| self.source.path_for_fdid(material.texture_1));
            let blend = match material.blend_mode {
                0 => BlendMode::Opaque,
                1 => BlendMode::Mask,
                2 => BlendMode::Blend,
                3 | 4 => BlendMode::Additive,
                _ => BlendMode::Modulate,
            };
            self.add_object_material(
                &key,
                index,
                MaterialKind::Wmo,
                texture.as_deref(),
                blend,
                material.flags & 0x04 != 0,
                material.flags & 0x01 != 0,
            );
        }
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut primitives = Vec::new();
        for (group, liquid) in &groups {
            let vertex_base = vertices.len() as u32;
            let index_base = indices.len();
            for (index, position) in group.vertex_positions.iter().enumerate() {
                let normal = group.vertex_normals.get(index);
                let uv = group.texture_coords.get(index);
                let color = group.vertex_colors.get(index);
                vertices.push(Vertex {
                    position: [position.x, position.z, -position.y],
                    normal: normal
                        .map(|value| [value.x, value.z, -value.y])
                        .unwrap_or([0.0, 1.0, 0.0]),
                    uv0: uv.map(|value| [value.u, value.v]).unwrap_or([0.0; 2]),
                    color0: color
                        .map(|value| [value.r, value.g, value.b, value.a])
                        .unwrap_or([255; 4]),
                });
            }
            indices.extend(
                group
                    .vertex_indices
                    .iter()
                    .map(|index| vertex_base + u32::from(*index)),
            );
            for batch in &group.render_batches {
                let material_id = if batch.flags & 0x02 != 0 {
                    usize::try_from(batch.bounding_box_max[2]).unwrap_or(0)
                } else {
                    batch.material_id as usize
                };
                primitives.push(Primitive {
                    first_index: index_base + batch.start_index as usize,
                    index_count: batch.count as usize,
                    material: material_base
                        + material_id.min(root.materials.len().saturating_sub(1)),
                });
            }
            if let Some(surface) = liquid {
                let liquid_vertex_base = vertices.len() as u32;
                let liquid_index_base = indices.len();
                vertices.extend_from_slice(&surface.vertices);
                indices.extend(
                    surface
                        .indices
                        .iter()
                        .map(|index| liquid_vertex_base + index),
                );
                let liquid_material = self.add_liquid_material(surface.liquid_type);
                primitives.push(Primitive {
                    first_index: liquid_index_base,
                    index_count: surface.indices.len(),
                    material: liquid_material,
                });
                self.scene.metadata.liquids.push(LiquidMetadata {
                    source: key.clone(),
                    tile: None,
                    chunk: None,
                    layer: surface.layer,
                    liquid_type: surface.liquid_type,
                    fishable: false,
                    deep: false,
                });
            }
        }
        for light in &root.lights {
            self.scene.metadata.lights.push(LightMetadata {
                source: key.clone(),
                kind: format!("{:?}", light.light_type).to_lowercase(),
                position: [light.position[0], light.position[2], -light.position[1]],
                color: [
                    f32::from(light.color[2]) / 255.0,
                    f32::from(light.color[1]) / 255.0,
                    f32::from(light.color[0]) / 255.0,
                ],
                intensity: light.intensity,
                attenuation: [light.attenuation_start, light.attenuation_end],
            });
        }
        for fog in &root.fogs {
            self.scene.metadata.fog.push(FogMetadata {
                source: key.clone(),
                color: [
                    f32::from(fog.color_1[2]) / 255.0,
                    f32::from(fog.color_1[1]) / 255.0,
                    f32::from(fog.color_1[0]) / 255.0,
                    f32::from(fog.color_1[3]) / 255.0,
                ],
                start: fog.fog_end * fog.fog_start_multiplier,
                end: fog.fog_end,
            });
        }
        if self.scene.metadata.environment.ambient_color.is_none() {
            self.scene.metadata.environment.ambient_color = Some([
                f32::from(root.ambient_color[2]) / 255.0,
                f32::from(root.ambient_color[1]) / 255.0,
                f32::from(root.ambient_color[0]) / 255.0,
                f32::from(root.ambient_color[3]) / 255.0,
            ]);
            self.scene.metadata.environment.skybox = root.skybox.clone();
        }
        let mesh = self.scene.meshes.len();
        self.scene.meshes.push(Mesh {
            name: asset_name(&key),
            source: key.clone(),
            vertices,
            indices,
            primitives,
        });
        self.asset_meshes.insert(key, mesh);
        Ok(mesh)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_object_material(
        &mut self,
        source: &str,
        index: usize,
        kind: MaterialKind,
        texture: Option<&str>,
        blend: BlendMode,
        double_sided: bool,
        unlit: bool,
    ) {
        let name = format!("{}_mat_{index}", asset_name(source));
        let layers = texture
            .map(|path| {
                vec![TextureLayer {
                    texture: texture_ref(path, ColorSpace::Srgb),
                    alpha_map: None,
                    alpha_data: None,
                    effect_id: None,
                    flags: 0,
                }]
            })
            .unwrap_or_default();
        self.scene.materials.push(Material {
            name: name.clone(),
            source: source.to_owned(),
            kind,
            layers,
            normal: None,
            emissive: None,
            blend,
            double_sided,
            unlit,
        });
        self.scene.metadata.materials.insert(
            name.clone(),
            MaterialMetadata {
                document: format!("materials/{name}.mtlx"),
                source: source.to_owned(),
                blend,
                double_sided,
                unlit,
            },
        );
    }

    fn placement_transform(&self, position: [f32; 3], rotation: [f32; 3], scale: f32) -> Transform {
        let origin = self.origin.unwrap_or([0.0; 3]);
        let ax = rotation[0].to_radians();
        let ay = (rotation[1] - 90.0).to_radians();
        let az = rotation[2].to_radians();
        let (sa, ca) = ax.sin_cos();
        let (sb, cb) = ay.sin_cos();
        let (sc, cc) = az.sin_cos();
        let r00 = cb * cc + sa * sb * sc;
        let r01 = -cb * sc + sa * sb * cc;
        let r02 = ca * sb;
        let r10 = ca * sc;
        let r11 = ca * cc;
        let r12 = -sa;
        let r20 = -sb * cc + sa * cb * sc;
        let r21 = sb * sc + sa * cb * cc;
        let r22 = ca * cb;
        let rotation = Quat::from_mat3(&Mat3::from_cols(
            Vec3::new(-r00, r10, -r20),
            Vec3::new(-r01, r11, -r21),
            Vec3::new(-r02, r12, -r22),
        ));
        Transform {
            translation: [
                MAP_ZERO_POINT - position[0] - origin[0],
                position[1] - origin[1],
                MAP_ZERO_POINT - position[2] - origin[2],
            ],
            rotation: rotation.to_array(),
            scale: [scale; 3],
        }
    }
}

fn terrain_indices() -> Vec<u32> {
    let mut result = Vec::with_capacity(8 * 8 * 12);
    for y in 0..8usize {
        for x in 0..8usize {
            let top_left = y * 17 + x;
            let top_right = top_left + 1;
            let center = top_left + 9;
            let bottom_left = (y + 1) * 17 + x;
            let bottom_right = bottom_left + 1;
            result.extend([
                top_left as u32,
                center as u32,
                top_right as u32,
                top_right as u32,
                center as u32,
                bottom_right as u32,
                bottom_right as u32,
                center as u32,
                bottom_left as u32,
                bottom_left as u32,
                center as u32,
                top_left as u32,
            ]);
        }
    }
    result
}

fn adapt_texture_fdid_chunks(source: &dyn AssetSource, data: Vec<u8>) -> Result<Vec<u8>> {
    let mut offset = 0usize;
    let mut texture_paths = None;
    while offset + 8 <= data.len() {
        let magic = &data[offset..offset + 4];
        let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let end = offset
            .checked_add(8 + size)
            .context("ADT chunk offset overflow")?;
        if end > data.len() {
            bail!("ADT chunk extends past split texture file");
        }
        if magic == b"DIDM" {
            let mut paths = Vec::new();
            let (ids, remainder) = data[offset + 8..end].as_chunks::<4>();
            if !remainder.is_empty() {
                bail!("MDID chunk size is not divisible by four");
            }
            for bytes in ids {
                let fdid = u32::from_le_bytes(*bytes);
                paths.push(
                    source
                        .path_for_fdid(fdid)
                        .unwrap_or_else(|| format!("fdid/{fdid}.blp")),
                );
            }
            texture_paths = Some(paths);
            break;
        }
        offset = end;
    }
    let Some(texture_paths) = texture_paths else {
        return Ok(data);
    };
    let mut mtex = Vec::new();
    for path in texture_paths {
        mtex.extend_from_slice(path.as_bytes());
        mtex.push(0);
    }
    let mut result = Vec::with_capacity(data.len() + mtex.len());
    offset = 0;
    while offset + 8 <= data.len() {
        let magic = &data[offset..offset + 4];
        let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let end = offset
            .checked_add(8 + size)
            .context("ADT chunk offset overflow")?;
        if magic == b"DIDM" {
            result.extend_from_slice(b"XETM");
            result.extend_from_slice(&(mtex.len() as u32).to_le_bytes());
            result.extend_from_slice(&mtex);
        } else if magic != b"DIHM" {
            result.extend_from_slice(&data[offset..end]);
        }
        offset = end;
    }
    Ok(result)
}

fn sanitize_root_adt(data: Vec<u8>) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(data.len());
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let magic = &data[offset..offset + 4];
        let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let end = offset
            .checked_add(8 + size)
            .context("ADT chunk offset overflow")?;
        if end > data.len() {
            bail!("ADT chunk extends past root file");
        }
        let empty_mcse = magic == b"KNCM"
            && size >= 8
            && &data[end - 8..end - 4] == b"ESCM"
            && data[end - 4..end] == [0; 4];
        if empty_mcse {
            result.extend_from_slice(magic);
            result.extend_from_slice(&((size - 8) as u32).to_le_bytes());
            let mut payload = data[offset + 8..end - 8].to_vec();
            payload[0x00..0x04].fill(0);
            payload[0x0c..0x14].fill(0);
            payload[0x14..0x18].copy_from_slice(&136u32.to_le_bytes());
            payload[0x18..0x1c].copy_from_slice(&724u32.to_le_bytes());
            payload[0x1c..0x34].fill(0);
            payload[0x38..0x3c].fill(0);
            payload[0x50..0x68].fill(0);
            payload[0x74..0x80].fill(0);
            result.extend_from_slice(&payload);
        } else {
            result.extend_from_slice(&data[offset..end]);
        }
        offset = end;
    }
    if offset != data.len() {
        bail!("ADT root file has an incomplete chunk header");
    }
    Ok(result)
}

fn parse_imported_m2(data: &[u8], chunked: bool) -> Result<ImportedM2> {
    if chunked {
        return parse_minimal_m2(data);
    }
    let mut cursor = Cursor::new(data);
    match wow_m2::parse_m2(&mut cursor) {
        Ok(format) => {
            let model = format.model();
            Ok(ImportedM2 {
                vertices: model
                    .vertices
                    .iter()
                    .map(|vertex| Vertex {
                        position: [vertex.position.x, vertex.position.z, -vertex.position.y],
                        normal: [vertex.normal.x, vertex.normal.z, -vertex.normal.y],
                        uv0: [vertex.tex_coords.x, vertex.tex_coords.y],
                        color0: [255; 4],
                    })
                    .collect(),
                materials: model
                    .materials
                    .iter()
                    .map(|material| M2MaterialInfo {
                        flags: material.flags.bits(),
                        blend_mode: material.blend_mode.bits(),
                    })
                    .collect(),
                texture_lookup: model.raw_data.texture_lookup_table.clone(),
                texture_names: model
                    .textures
                    .iter()
                    .map(|texture| texture.filename.string.to_string_lossy())
                    .collect(),
            })
        }
        Err(full_error) => parse_minimal_m2(data)
            .with_context(|| format!("full M2 parser failed first: {full_error}")),
    }
}

fn parse_minimal_m2(data: &[u8]) -> Result<ImportedM2> {
    const MAX_VERTICES: u32 = 10_000_000;
    const MAX_MATERIALS: u32 = 65_536;
    let mut cursor = Cursor::new(data);
    let header = M2Header::parse(&mut cursor)?;
    if header.vertices.count > MAX_VERTICES {
        bail!("unreasonable M2 vertex count {}", header.vertices.count);
    }
    if header.render_flags.count > MAX_MATERIALS {
        bail!(
            "unreasonable M2 material count {}",
            header.render_flags.count
        );
    }
    cursor.seek(SeekFrom::Start(header.vertices.offset as u64))?;
    let mut vertices = Vec::with_capacity(header.vertices.count as usize);
    for _ in 0..header.vertices.count {
        let vertex = M2Vertex::parse_with_validation(
            &mut cursor,
            header.version,
            Some(header.bones.count),
            wow_m2::ValidationMode::Permissive,
        )?;
        vertices.push(Vertex {
            position: [vertex.position.x, vertex.position.z, -vertex.position.y],
            normal: [vertex.normal.x, vertex.normal.z, -vertex.normal.y],
            uv0: [vertex.tex_coords.x, vertex.tex_coords.y],
            color0: [255; 4],
        });
    }
    cursor.seek(SeekFrom::Start(header.render_flags.offset as u64))?;
    let mut materials = Vec::with_capacity(header.render_flags.count as usize);
    for _ in 0..header.render_flags.count {
        let material = M2Material::parse(&mut cursor, header.version)?;
        materials.push(M2MaterialInfo {
            flags: material.flags.bits(),
            blend_mode: material.blend_mode.bits(),
        });
    }
    cursor.seek(SeekFrom::Start(header.texture_lookup_table.offset as u64))?;
    let mut texture_lookup = Vec::with_capacity(header.texture_lookup_table.count as usize);
    for _ in 0..header.texture_lookup_table.count {
        texture_lookup.push(cursor.read_u16::<LittleEndian>()?);
    }
    Ok(ImportedM2 {
        vertices,
        materials,
        texture_lookup,
        texture_names: Vec::new(),
    })
}

struct AdaptedM2 {
    model: Vec<u8>,
    skin_fdids: Vec<u32>,
    texture_fdids: Vec<u32>,
    chunked: bool,
}

fn adapt_m2_chunks(data: Vec<u8>) -> Result<AdaptedM2> {
    if !data.starts_with(b"MD21") {
        return Ok(AdaptedM2 {
            model: data,
            skin_fdids: Vec::new(),
            texture_fdids: Vec::new(),
            chunked: false,
        });
    }
    let mut model = None;
    let mut skin_fdids = Vec::new();
    let mut texture_fdids = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let magic = &data[offset..offset + 4];
        let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
        let end = offset
            .checked_add(8 + size)
            .context("M2 chunk offset overflow")?;
        if end > data.len() {
            bail!("M2 chunk extends past file");
        }
        let payload = &data[offset + 8..end];
        match magic {
            b"MD21" => model = Some(payload.to_vec()),
            b"SFID" => skin_fdids = parse_fdid_chunk(payload)?,
            b"TXID" => texture_fdids = parse_fdid_chunk(payload)?,
            _ => {}
        }
        offset = end;
    }
    if offset != data.len() {
        bail!("M2 file has an incomplete chunk header");
    }
    let model = model.context("chunked M2 has no MD21 payload")?;
    if !model.starts_with(b"MD20") {
        bail!("MD21 payload does not contain an MD20 model");
    }
    Ok(AdaptedM2 {
        model,
        skin_fdids,
        texture_fdids,
        chunked: true,
    })
}

fn parse_fdid_chunk(data: &[u8]) -> Result<Vec<u32>> {
    let (ids, remainder) = data.as_chunks::<4>();
    if !remainder.is_empty() {
        bail!("M2 FileDataID chunk size is not divisible by four");
    }
    Ok(ids.iter().map(|bytes| u32::from_le_bytes(*bytes)).collect())
}

fn decode_alpha(
    chunk: &wow_adt::McnkChunk,
    layers: &wow_adt::MclyChunk,
    layer_index: usize,
) -> Option<Vec<u8>> {
    let alpha = chunk.alpha.as_ref()?;
    let layer = layers.layers.get(layer_index)?;
    let offset = layer.offset_in_mcal as usize;
    let end = layers
        .layers
        .iter()
        .skip(layer_index + 1)
        .map(|next| next.offset_in_mcal as usize)
        .find(|next| *next > offset)
        .unwrap_or(alpha.data.len());
    let data = alpha.data.get(offset..end)?.to_vec();
    let format = if layer.flags.alpha_map_compressed() {
        AlphaFormat::Compressed
    } else if data.len() >= AlphaMap::SIZE_UNCOMPRESSED_8BIT {
        AlphaFormat::Uncompressed4096
    } else {
        AlphaFormat::Uncompressed2048
    };
    AlphaMap::new(data, format).decompress().ok()
}

fn m2_primitives(
    skin: &SkinFile,
    material_base: usize,
    material_count: usize,
    index_count: usize,
) -> Vec<Primitive> {
    let mut result = Vec::new();
    for batch in skin.batches() {
        let Some(section) = skin.submeshes().get(batch.skin_section_index as usize) else {
            continue;
        };
        let first_index = section.triangle_start as usize;
        let section_index_count = section.triangle_count as usize;
        if section_index_count == 0 || first_index + section_index_count > index_count {
            continue;
        }
        result.push(Primitive {
            first_index,
            index_count: section_index_count,
            material: material_base
                + (batch.material_index as usize).min(material_count.saturating_sub(1)),
        });
    }
    if result.is_empty() {
        result.push(Primitive {
            first_index: 0,
            index_count,
            material: material_base,
        });
    }
    result
}

fn texture_ref(path: &str, color_space: ColorSpace) -> TextureRef {
    let normalized = normalize(path);
    TextureRef {
        wow_path: normalized.clone(),
        output_path: format!("textures/{}.png", replace_extension(&normalized, "")),
        color_space,
    }
}

fn replace_extension(path: &str, extension: &str) -> String {
    let stem = path.rsplit_once('.').map_or(path, |(stem, _)| stem);
    if extension.is_empty() {
        stem.to_owned()
    } else {
        format!("{stem}{extension}")
    }
}

fn wmo_group_path(path: &str, index: u32) -> String {
    let stem = path.rsplit_once('.').map_or(path, |(stem, _)| stem);
    format!("{stem}_{index:03}.wmo")
}

fn asset_name(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .rsplit_once('.')
        .map_or(path, |(stem, _)| stem)
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn add_default_cameras(scene: &mut Scene) {
    let Some((min, max)) = scene_world_bounds(scene) else {
        return;
    };
    let center = (min + max) * 0.5;
    let extent = max - min;
    let horizontal = extent.x.max(extent.z).max(10.0);
    let target = Vec3::new(center.x, min.y + extent.y * 0.15, center.z);
    let diagonal = extent.length().max(10.0);
    let close_eye = target + Vec3::new(horizontal * 0.24, horizontal * 0.16, horizontal * 0.20);
    let aerial_eye = target + Vec3::new(0.0, horizontal * 0.90, horizontal * 0.65);
    scene.cameras.push(Camera {
        name: "close".to_owned(),
        transform: look_at_transform(close_eye, target),
        yfov: 48.0f32.to_radians(),
        znear: 0.1,
        zfar: diagonal * 8.0,
    });
    scene.cameras.push(Camera {
        name: "aerial".to_owned(),
        transform: look_at_transform(aerial_eye, target),
        yfov: 50.0f32.to_radians(),
        znear: 0.1,
        zfar: diagonal * 8.0,
    });
}

fn scene_world_bounds(scene: &Scene) -> Option<(Vec3, Vec3)> {
    let mesh_bounds: Vec<Option<(Vec3, Vec3)>> = scene
        .meshes
        .iter()
        .map(|mesh| {
            let first = mesh.vertices.first()?;
            let mut min = Vec3::from_array(first.position);
            let mut max = min;
            for vertex in &mesh.vertices[1..] {
                let position = Vec3::from_array(vertex.position);
                min = min.min(position);
                max = max.max(position);
            }
            Some((min, max))
        })
        .collect();
    let mut result = None;
    for node in &scene.nodes {
        if let Some(mesh) = node.mesh
            && scene.meshes[mesh].source.starts_with("ADT ")
            && let Some(bounds) = mesh_bounds.get(mesh).copied().flatten()
        {
            extend_bounds(&mut result, bounds, node.transform);
        }
    }
    if let Some((terrain_min, terrain_max)) = result {
        let padding = (terrain_max.x - terrain_min.x).max(terrain_max.z - terrain_min.z) * 0.1;
        for (mesh, transforms) in &scene.instances {
            let Some(bounds) = mesh_bounds.get(*mesh).copied().flatten() else {
                continue;
            };
            for transform in transforms {
                let translation = Vec3::from_array(transform.translation);
                if translation.x >= terrain_min.x - padding
                    && translation.x <= terrain_max.x + padding
                    && translation.z >= terrain_min.z - padding
                    && translation.z <= terrain_max.z + padding
                {
                    extend_bounds(&mut result, bounds, *transform);
                }
            }
        }
    }
    result
}

fn extend_bounds(result: &mut Option<(Vec3, Vec3)>, bounds: (Vec3, Vec3), transform: Transform) {
    let matrix = Mat4::from_scale_rotation_translation(
        Vec3::from_array(transform.scale),
        Quat::from_array(transform.rotation),
        Vec3::from_array(transform.translation),
    );
    for x in [bounds.0.x, bounds.1.x] {
        for y in [bounds.0.y, bounds.1.y] {
            for z in [bounds.0.z, bounds.1.z] {
                let point = matrix.transform_point3(Vec3::new(x, y, z));
                match result {
                    Some((min, max)) => {
                        *min = min.min(point);
                        *max = max.max(point);
                    }
                    None => *result = Some((point, point)),
                }
            }
        }
    }
}

fn look_at_transform(eye: Vec3, target: Vec3) -> Transform {
    let world = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
    Transform {
        translation: eye.to_array(),
        rotation: Quat::from_mat4(&world).normalize().to_array(),
        scale: [1.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::{look_at_transform, terrain_indices};

    #[test]
    fn terrain_grid_has_four_triangles_per_cell() {
        let indices = terrain_indices();
        assert_eq!(indices.len(), 8 * 8 * 4 * 3);
        assert!(indices.iter().all(|index| *index < 145));
        assert_eq!(&indices[..12], &[0, 9, 1, 1, 9, 18, 18, 9, 17, 17, 9, 0]);
    }

    #[test]
    fn camera_transform_points_local_forward_at_target() {
        let eye = Vec3::new(4.0, 3.0, 8.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let transform = look_at_transform(eye, target);
        let forward = Quat::from_array(transform.rotation) * -Vec3::Z;
        assert!(forward.dot((target - eye).normalize()) > 0.999);
    }
}
