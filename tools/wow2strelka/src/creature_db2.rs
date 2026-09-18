use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::ac_db::CreatureSpawn;
use crate::source::AssetSource;

#[derive(Debug, Clone)]
pub struct ResolvedCreature {
    pub spawn: CreatureSpawn,
    pub resolved_display_id: u32,
    pub model_file_id: u32,
    pub model_path: String,
    pub scale: f32,
    pub texture_variations: Vec<u32>,
    pub remap_status: &'static str,
    pub equipment_models: Vec<ResolvedEquipment>,
}

#[derive(Debug, Clone)]
pub struct ResolvedEquipment {
    pub item_id: u32,
    pub slot: usize,
    pub attachment_id: u32,
    pub model_file_id: u32,
    pub model_path: String,
    pub texture_file_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct DisplayInfo {
    model_id: u32,
    extra_id: u32,
    scale: f32,
    texture_variations: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ModelData {
    file_data_id: u32,
    scale: f32,
}

#[derive(Debug, Clone, Default)]
struct ItemAssets {
    models: Vec<u32>,
    textures: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ExtraAppearance {
    body_texture: Option<u32>,
    hair_style: u8,
    hair_color: u8,
    facial_hair: u8,
}

pub struct CreatureCatalog {
    displays: HashMap<u32, DisplayInfo>,
    models: HashMap<u32, ModelData>,
    item_models: HashMap<u32, ItemAssets>,
    extra_appearances: HashMap<u32, ExtraAppearance>,
}

impl CreatureCatalog {
    pub fn load(source: &dyn AssetSource) -> Result<Self> {
        let mut displays = parse_display_info(&source.read_db2_csv("CreatureDisplayInfo")?)?;
        let texture_files = source.read_db2_csv("TextureFileData")?;
        let extra_appearances = parse_extra_appearances(
            &source.read_db2_csv("CreatureDisplayInfoExtra")?,
            &texture_files,
        )?;
        for display in displays.values_mut() {
            if let Some(texture) = extra_appearances
                .get(&display.extra_id)
                .and_then(|appearance| appearance.body_texture)
                && !display.texture_variations.contains(&texture)
            {
                display.texture_variations.insert(0, texture);
            }
        }
        let item_modified = source.read_db2_csv("ItemModifiedAppearance")?;
        let item_appearance = source.read_db2_csv("ItemAppearance")?;
        let item_display = source.read_db2_csv("ItemDisplayInfo")?;
        let model_files = source.read_db2_csv("ModelFileData")?;
        Ok(Self {
            displays,
            models: parse_model_data(&source.read_db2_csv("CreatureModelData")?)?,
            item_models: parse_item_models(
                &item_modified,
                &item_appearance,
                &item_display,
                &model_files,
                &texture_files,
            )?,
            extra_appearances,
        })
    }

    pub fn resolve(
        &self,
        source: &dyn AssetSource,
        spawn: CreatureSpawn,
    ) -> Result<Option<ResolvedCreature>> {
        let resolved = spawn
            .display_candidates
            .iter()
            .copied()
            .chain(std::iter::once((spawn.display_id, spawn.display_scale)))
            .find_map(|(display_id, template_scale)| {
                let display = self.displays.get(&display_id)?;
                let model = self.models.get(&display.model_id)?;
                let model_path = source.path_for_fdid(model.file_data_id)?;
                Some((display_id, template_scale, display, model, model_path))
            });
        let Some((resolved_display_id, template_scale, display, model, model_path)) = resolved
        else {
            return Ok(None);
        };
        let scale = template_scale * display.scale * model.scale;
        let mut equipment_models = Vec::new();
        for (slot, item_id) in spawn.equipment.iter().copied().enumerate() {
            if item_id == 0 {
                continue;
            }
            let attachment_id = match slot {
                0 => 1,
                1 => 2,
                _ => 1,
            };
            let Some(assets) = self.item_models.get(&item_id) else {
                continue;
            };
            for file_id in &assets.models {
                if let Some(model_path) = source.path_for_fdid(*file_id) {
                    equipment_models.push(ResolvedEquipment {
                        item_id,
                        slot,
                        attachment_id,
                        model_file_id: *file_id,
                        model_path,
                        texture_file_ids: assets.textures.clone(),
                    });
                }
            }
        }
        let mut texture_variations = display.texture_variations.clone();
        if let Some(appearance) = self.extra_appearances.get(&display.extra_id) {
            append_character_textures(source, &model_path, appearance, &mut texture_variations);
        }
        Ok(Some(ResolvedCreature {
            remap_status: if resolved_display_id == spawn.display_id {
                "exact_display_id"
            } else {
                "alternate_template_display"
            },
            spawn,
            resolved_display_id,
            model_file_id: model.file_data_id,
            model_path,
            scale: if scale.is_finite() && scale > 0.0 {
                scale
            } else {
                1.0
            },
            texture_variations,
            equipment_models,
        }))
    }
}

fn parse_extra_appearances(
    extra_data: &[u8],
    texture_data: &[u8],
) -> Result<HashMap<u32, ExtraAppearance>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(texture_data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let resource = index(&headers, &["materialresourcesid"])?;
    let file = index(&headers, &["filedataid"])?;
    let usage = index(&headers, &["usagetype"])?;
    let mut textures = HashMap::new();
    for row in reader.records() {
        let row = row?;
        if row.get(usage) != Some("0") {
            continue;
        }
        if let (Some(resource), Some(file)) = (
            row.get(resource)
                .and_then(|value| value.parse::<u32>().ok()),
            row.get(file).and_then(|value| value.parse::<u32>().ok()),
        ) {
            textures.entry(resource).or_insert(file);
        }
    }
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(extra_data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let id = index(&headers, &["id"])?;
    let resource = index(&headers, &["bakematerialresourcesid"])?;
    let hair_style = index(&headers, &["hairstyleid"])?;
    let hair_color = index(&headers, &["haircolorid"])?;
    let facial_hair = index(&headers, &["facialhairid"])?;
    let mut result = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let Some(id) = row.get(id).and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let material_resource = row
            .get(resource)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        result.insert(
            id,
            ExtraAppearance {
                body_texture: textures.get(&material_resource).copied(),
                hair_style: csv_u8(&row, hair_style),
                hair_color: csv_u8(&row, hair_color),
                facial_hair: csv_u8(&row, facial_hair),
            },
        );
    }
    Ok(result)
}

fn csv_u8(row: &csv::StringRecord, column: usize) -> u8 {
    row.get(column)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn append_character_textures(
    source: &dyn AssetSource,
    model_path: &str,
    appearance: &ExtraAppearance,
    textures: &mut Vec<u32>,
) {
    let components: Vec<&str> = model_path.split('/').collect();
    if components.len() < 3 || components[0] != "character" {
        return;
    }
    let root = format!("character/{}", components[1]);
    let candidate_groups = [
        vec![
            format!(
                "{root}/hair{:02}_{:02}.blp",
                appearance.hair_style, appearance.hair_color
            ),
            format!("{root}/hair00_{:02}.blp", appearance.hair_color),
            format!("{root}/hair00_00.blp"),
        ],
        vec![
            format!(
                "{root}/faciallowerhair{:02}_{:02}.blp",
                appearance.facial_hair, appearance.hair_color
            ),
            format!("{root}/faciallowerhair{:02}_00.blp", appearance.facial_hair),
            format!("{root}/faciallowerhair00_00.blp"),
        ],
        vec![
            format!(
                "{root}/facialupperhair{:02}_{:02}.blp",
                appearance.facial_hair, appearance.hair_color
            ),
            format!("{root}/facialupperhair{:02}_00.blp", appearance.facial_hair),
            format!("{root}/facialupperhair00_00.blp"),
        ],
    ];
    for candidates in candidate_groups {
        if let Some(file_id) = candidates
            .iter()
            .find_map(|path| source.fdid_for_path(path))
            && !textures.contains(&file_id)
        {
            textures.push(file_id);
        }
    }
}

fn parse_item_models(
    modified_data: &[u8],
    appearance_data: &[u8],
    display_data: &[u8],
    model_data: &[u8],
    texture_data: &[u8],
) -> Result<HashMap<u32, ItemAssets>> {
    let modified = parse_pairs(
        modified_data,
        &["itemid"],
        &["itemappearanceid"],
        Some(("itemappearancemodifierid", "0")),
    )?;
    let appearances = parse_pairs(appearance_data, &["id"], &["itemdisplayinfoid"], None)?;
    let display_resources = parse_multi_pairs(
        display_data,
        &["id"],
        &["modelresourcesid0", "modelresourcesid1"],
    )?;
    let display_materials = parse_multi_pairs(
        display_data,
        &["id"],
        &["modelmaterialresourcesid0", "modelmaterialresourcesid1"],
    )?;
    let resource_files = parse_multi_value_map(model_data, "modelresourcesid", "filedataid")?;
    let material_files = parse_multi_value_map(texture_data, "materialresourcesid", "filedataid")?;
    let mut result = HashMap::<u32, ItemAssets>::new();
    for (item_id, appearance_id) in modified {
        let Some(display_id) = appearances.get(&appearance_id) else {
            continue;
        };
        for resource in display_resources.get(display_id).into_iter().flatten() {
            result
                .entry(item_id)
                .or_default()
                .models
                .extend(resource_files.get(resource).into_iter().flatten().copied());
        }
        for resource in display_materials.get(display_id).into_iter().flatten() {
            result
                .entry(item_id)
                .or_default()
                .textures
                .extend(material_files.get(resource).into_iter().flatten().copied());
        }
    }
    for assets in result.values_mut() {
        let mut models = HashSet::new();
        let mut textures = HashSet::new();
        assets.models.retain(|file| models.insert(*file));
        assets.textures.retain(|file| textures.insert(*file));
    }
    Ok(result)
}

fn parse_pairs(
    data: &[u8],
    key_names: &[&str],
    value_names: &[&str],
    filter: Option<(&str, &str)>,
) -> Result<HashMap<u32, u32>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let key = index(&headers, key_names)?;
    let value = index(&headers, value_names)?;
    let filter = match filter {
        Some((name, expected)) => Some((index(&headers, &[name])?, expected)),
        None => None,
    };
    let mut result = HashMap::new();
    for row in reader.records() {
        let row = row?;
        if filter.is_some_and(|(column, expected)| row.get(column) != Some(expected)) {
            continue;
        }
        if let (Some(key), Some(value)) = (
            row.get(key).and_then(|value| value.parse().ok()),
            row.get(value).and_then(|value| value.parse().ok()),
        ) {
            result.entry(key).or_insert(value);
        }
    }
    Ok(result)
}

fn parse_multi_pairs(
    data: &[u8],
    key_names: &[&str],
    value_names: &[&str],
) -> Result<HashMap<u32, Vec<u32>>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let key = index(&headers, key_names)?;
    let values: Vec<usize> = value_names
        .iter()
        .filter_map(|name| headers.iter().position(|header| header == name))
        .collect();
    let mut result = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let Some(key) = row.get(key).and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        result.insert(
            key,
            values
                .iter()
                .filter_map(|column| row.get(*column))
                .filter_map(|value| value.parse::<u32>().ok())
                .filter(|value| *value != 0)
                .collect(),
        );
    }
    Ok(result)
}

fn parse_multi_value_map(
    data: &[u8],
    key_name: &str,
    value_name: &str,
) -> Result<HashMap<u32, Vec<u32>>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let key = index(&headers, &[key_name])?;
    let value = index(&headers, &[value_name])?;
    let mut result = HashMap::<u32, Vec<u32>>::new();
    for row in reader.records() {
        let row = row?;
        if let (Some(key), Some(value)) = (
            row.get(key).and_then(|value| value.parse::<u32>().ok()),
            row.get(value).and_then(|value| value.parse::<u32>().ok()),
        ) {
            result.entry(key).or_default().push(value);
        }
    }
    Ok(result)
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn index(headers: &[String], names: &[&str]) -> Result<usize> {
    headers
        .iter()
        .position(|header| names.iter().any(|name| header == name))
        .with_context(|| format!("missing DB2 column {}", names.join("/")))
}

fn parse_display_info(data: &[u8]) -> Result<HashMap<u32, DisplayInfo>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let id = index(&headers, &["id"])?;
    let model_id = index(&headers, &["modelid"])?;
    let extra_id = index(&headers, &["extendeddisplayinfoid"])?;
    let scale = index(&headers, &["creaturemodelscale"])?;
    let texture_columns: Vec<usize> = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| {
            header
                .starts_with("texturevariationfiledataid")
                .then_some(index)
        })
        .collect();
    let mut result = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let Some(display_id) = row.get(id).and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        result.insert(
            display_id,
            DisplayInfo {
                model_id: row
                    .get(model_id)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
                extra_id: row
                    .get(extra_id)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
                scale: row
                    .get(scale)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(1.0),
                texture_variations: texture_columns
                    .iter()
                    .filter_map(|column| row.get(*column))
                    .filter_map(|value| value.parse::<u32>().ok())
                    .filter(|value| *value != 0)
                    .collect(),
            },
        );
    }
    Ok(result)
}

fn parse_model_data(data: &[u8]) -> Result<HashMap<u32, ModelData>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalize).collect();
    let id = index(&headers, &["id"])?;
    let file_data_id = index(&headers, &["filedataid", "modelfiledataid"])?;
    let scale = index(&headers, &["modelscale"])?;
    let mut result = HashMap::new();
    for row in reader.records() {
        let row = row?;
        let Some(model_id) = row.get(id).and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        result.insert(
            model_id,
            ModelData {
                file_data_id: row
                    .get(file_data_id)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0),
                scale: row
                    .get(scale)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(1.0),
            },
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{parse_display_info, parse_model_data};

    #[test]
    fn parses_current_creature_tables() {
        let displays = parse_display_info(
            b"ID,ModelID,ExtendedDisplayInfoID,CreatureModelScale,TextureVariationFileDataID_0\n4,9,0,1.5,77\n",
        )
        .unwrap();
        let models = parse_model_data(b"ID,FileDataID,ModelScale\n9,123020,2\n").unwrap();
        assert_eq!(displays[&4].model_id, 9);
        assert_eq!(displays[&4].texture_variations, vec![77]);
        assert_eq!(models[&9].file_data_id, 123020);
    }
}
