use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use mysql::prelude::Queryable;
use mysql::{Opts, Pool, Row, params};

const TILE_SIZE: f32 = 533.333_3;

#[derive(Debug, Clone)]
pub struct CreatureSpawn {
    pub guid: u32,
    pub entry: u32,
    pub map: u16,
    pub zone_id: u16,
    pub area_id: u16,
    pub phase_mask: u32,
    pub equipment_id: i8,
    pub position: [f32; 3],
    pub orientation: f32,
    pub movement_type: u8,
    pub wander_distance: f32,
    pub name: String,
    pub subname: Option<String>,
    pub faction: u16,
    pub display_id: u32,
    pub display_scale: f32,
    pub display_candidates: Vec<(u32, f32)>,
    pub equipment: [u32; 3],
}

#[derive(Debug, Clone)]
struct TemplateModel {
    display_id: u32,
    scale: f32,
    probability: f32,
    index: u16,
}

pub struct AzerothCore {
    pool: Pool,
}

impl AzerothCore {
    pub fn connect(url: &str) -> Result<Self> {
        let options = Opts::from_url(url).context("invalid AzerothCore MySQL URL")?;
        let pool = Pool::new(options).context("failed to connect to AzerothCore world DB")?;
        Ok(Self { pool })
    }

    pub fn load_spawns(
        &self,
        map: u16,
        tiles: &[[u32; 2]],
        spawn_mask: u32,
        phase_mask: u32,
    ) -> Result<Vec<CreatureSpawn>> {
        let bounds = selected_world_bounds(tiles).context("NPC import requires selected tiles")?;
        let mut connection = self.pool.get_conn()?;
        let has_model_table = table_exists(&mut connection, "creature_template_model")?;
        let id_column = if column_exists(&mut connection, "creature", "id")? {
            "id"
        } else {
            "id1"
        };
        let sql = format!(
            "SELECT c.guid, c.{id_column} AS entry, c.map, \
             COALESCE(c.zoneId, 0) zone_id, COALESCE(c.areaId, 0) area_id, \
             c.spawnMask, COALESCE(c.phaseMask, 1) phase_mask, \
             COALESCE(c.equipment_id, 0) equipment_id, \
             c.position_x, c.position_y, c.position_z, c.orientation, \
             COALESCE(c.MovementType, 0) movement_type, \
             COALESCE(c.wander_distance, 0) wander_distance, \
             ct.name, ct.subname, COALESCE(ct.faction, 0) faction \
             FROM creature c JOIN creature_template ct ON ct.entry = c.{id_column} \
             WHERE c.map = :map AND (c.spawnMask & :spawn_mask) <> 0 \
             AND (:phase_mask = 0 OR (COALESCE(c.phaseMask, 1) & :phase_mask) <> 0) \
             AND c.position_x BETWEEN :min_x AND :max_x \
             AND c.position_y BETWEEN :min_y AND :max_y"
        );
        let rows: Vec<Row> = connection.exec(
            sql,
            params! {
                "map" => map,
                "spawn_mask" => spawn_mask,
                "phase_mask" => phase_mask,
                "min_x" => bounds[0],
                "max_x" => bounds[1],
                "min_y" => bounds[2],
                "max_y" => bounds[3],
            },
        )?;
        let mut spawns = Vec::new();
        for row in rows {
            let position_x = value::<f32>(&row, "position_x")?;
            let position_y = value::<f32>(&row, "position_y")?;
            if !tiles.contains(&adt_tile(position_x, position_y)) {
                continue;
            }
            spawns.push(CreatureSpawn {
                guid: value(&row, "guid")?,
                entry: value(&row, "entry")?,
                map: value(&row, "map")?,
                zone_id: value(&row, "zone_id")?,
                area_id: value(&row, "area_id")?,
                phase_mask: value(&row, "phase_mask")?,
                equipment_id: value(&row, "equipment_id")?,
                position: [position_x, value(&row, "position_z")?, position_y],
                orientation: value(&row, "orientation")?,
                movement_type: value(&row, "movement_type")?,
                wander_distance: value(&row, "wander_distance")?,
                name: value(&row, "name")?,
                subname: row
                    .get_opt::<Option<String>, _>("subname")
                    .transpose()
                    .context("invalid AzerothCore subname")?
                    .flatten(),
                faction: value(&row, "faction")?,
                display_id: 0,
                display_scale: 1.0,
                display_candidates: Vec::new(),
                equipment: [0; 3],
            });
        }
        let entries: HashSet<u32> = spawns.iter().map(|spawn| spawn.entry).collect();
        let models = if has_model_table {
            load_models(&mut connection, &entries)?
        } else {
            load_legacy_models(&mut connection, &entries)?
        };
        let equipment = load_equipment(&mut connection, &entries)?;
        for spawn in &mut spawns {
            if let Some(candidates) = models.get(&spawn.entry) {
                let mut candidates = candidates.clone();
                candidates.sort_by(|left, right| {
                    right
                        .probability
                        .total_cmp(&left.probability)
                        .then_with(|| left.index.cmp(&right.index))
                });
                spawn.display_candidates = candidates
                    .iter()
                    .map(|model| (model.display_id, model.scale))
                    .collect();
            }
            if let Some(model) = models
                .get(&spawn.entry)
                .and_then(|models| choose_model(models))
            {
                spawn.display_id = model.display_id;
                spawn.display_scale = model.scale;
            }
            if let Some(items) = equipment.get(&(spawn.entry, spawn.equipment_id)) {
                spawn.equipment = *items;
            }
        }
        Ok(spawns)
    }
}

fn value<T>(row: &Row, name: &str) -> Result<T>
where
    T: mysql::prelude::FromValue,
{
    row.get(name)
        .with_context(|| format!("AzerothCore query did not return '{name}'"))
}

fn table_exists(connection: &mut mysql::PooledConn, table: &str) -> Result<bool> {
    let count: Option<u64> = connection.exec_first(
        "SELECT COUNT(*) FROM information_schema.tables \
         WHERE table_schema = DATABASE() AND table_name = :table",
        params! { "table" => table },
    )?;
    Ok(count.unwrap_or(0) != 0)
}

fn column_exists(connection: &mut mysql::PooledConn, table: &str, column: &str) -> Result<bool> {
    let count: Option<u64> = connection.exec_first(
        "SELECT COUNT(*) FROM information_schema.columns \
         WHERE table_schema = DATABASE() AND table_name = :table AND column_name = :column",
        params! { "table" => table, "column" => column },
    )?;
    Ok(count.unwrap_or(0) != 0)
}

fn load_models(
    connection: &mut mysql::PooledConn,
    entries: &HashSet<u32>,
) -> Result<HashMap<u32, Vec<TemplateModel>>> {
    if entries.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT CreatureID creature_id, Idx model_index, CreatureDisplayID display_id, \
         COALESCE(DisplayScale, 1) display_scale, COALESCE(Probability, 1) probability \
         FROM creature_template_model WHERE CreatureID IN ({})",
        id_list(entries)
    );
    model_rows(connection.query(sql)?)
}

fn load_legacy_models(
    connection: &mut mysql::PooledConn,
    entries: &HashSet<u32>,
) -> Result<HashMap<u32, Vec<TemplateModel>>> {
    if entries.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT entry creature_id, 0 model_index, modelid1 display_id, \
         1 display_scale, 1 probability FROM creature_template \
         WHERE entry IN ({})",
        id_list(entries)
    );
    model_rows(connection.query(sql)?)
}

fn model_rows(rows: Vec<Row>) -> Result<HashMap<u32, Vec<TemplateModel>>> {
    let mut result = HashMap::<u32, Vec<TemplateModel>>::new();
    for row in rows {
        result
            .entry(value(&row, "creature_id")?)
            .or_default()
            .push(TemplateModel {
                display_id: value(&row, "display_id")?,
                scale: value(&row, "display_scale")?,
                probability: value(&row, "probability")?,
                index: value(&row, "model_index")?,
            });
    }
    Ok(result)
}

fn choose_model(models: &[TemplateModel]) -> Option<&TemplateModel> {
    models.iter().max_by(|left, right| {
        left.probability
            .total_cmp(&right.probability)
            .then_with(|| right.index.cmp(&left.index))
    })
}

fn load_equipment(
    connection: &mut mysql::PooledConn,
    entries: &HashSet<u32>,
) -> Result<HashMap<(u32, i8), [u32; 3]>> {
    if entries.is_empty() || !table_exists(connection, "creature_equip_template")? {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT CreatureID creature_id, ID equipment_id, ItemID1 item1, \
         ItemID2 item2, ItemID3 item3 FROM creature_equip_template \
         WHERE CreatureID IN ({})",
        id_list(entries)
    );
    let mut result = HashMap::new();
    for row in connection.query::<Row, _>(sql)? {
        result.insert(
            (value(&row, "creature_id")?, value(&row, "equipment_id")?),
            [
                value(&row, "item1")?,
                value(&row, "item2")?,
                value(&row, "item3")?,
            ],
        );
    }
    Ok(result)
}

fn id_list(entries: &HashSet<u32>) -> String {
    let mut entries: Vec<u32> = entries.iter().copied().collect();
    entries.sort_unstable();
    entries
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub fn adt_tile(position_x: f32, position_y: f32) -> [u32; 2] {
    [
        (32.0 - position_y / TILE_SIZE).floor().clamp(0.0, 63.0) as u32,
        (32.0 - position_x / TILE_SIZE).floor().clamp(0.0, 63.0) as u32,
    ]
}

fn selected_world_bounds(tiles: &[[u32; 2]]) -> Option<[f32; 4]> {
    let first = tiles.first()?;
    let first_x = (32.0 - first[1] as f32) * TILE_SIZE;
    let first_y = (32.0 - first[0] as f32) * TILE_SIZE;
    let mut bounds = [first_x - TILE_SIZE, first_x, first_y - TILE_SIZE, first_y];
    for tile in &tiles[1..] {
        let max_x = (32.0 - tile[1] as f32) * TILE_SIZE;
        let max_y = (32.0 - tile[0] as f32) * TILE_SIZE;
        bounds[0] = bounds[0].min(max_x - TILE_SIZE);
        bounds[1] = bounds[1].max(max_x);
        bounds[2] = bounds[2].min(max_y - TILE_SIZE);
        bounds[3] = bounds[3].max(max_y);
    }
    Some(bounds)
}

pub fn infer_map_id(map: &str) -> Result<u16> {
    match map {
        "azeroth" => Ok(0),
        "kalimdor" => Ok(1),
        other => bail!("cannot infer AzerothCore map ID for '{other}'; pass --ac-map"),
    }
}

#[cfg(test)]
mod tests {
    use super::{TemplateModel, adt_tile, choose_model, selected_world_bounds};

    #[test]
    fn crossroads_maps_to_kalimdor_36_32() {
        assert_eq!(adt_tile(-456.263, -2652.7), [36, 32]);
    }

    #[test]
    fn selected_bounds_cover_all_tiles() {
        let bounds = selected_world_bounds(&[[36, 32], [37, 33]]).unwrap();
        assert!(bounds[0] < -500.0 && bounds[1] >= 0.0);
        assert!(bounds[2] < -2600.0 && bounds[3] < -2100.0);
    }

    #[test]
    fn model_selection_prefers_probability_then_low_index() {
        let models = vec![
            TemplateModel {
                display_id: 1,
                scale: 1.0,
                probability: 0.5,
                index: 1,
            },
            TemplateModel {
                display_id: 2,
                scale: 1.0,
                probability: 0.5,
                index: 0,
            },
        ];
        assert_eq!(choose_model(&models).unwrap().display_id, 2);
    }
}
