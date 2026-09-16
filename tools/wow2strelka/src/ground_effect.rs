use std::collections::HashMap;

use anyhow::{Context, Result};
use wow_adt::McnkChunk;

use crate::source::AssetSource;

const FRILL_DENSITY: u32 = 16;
const MAX_DENSITY: u32 = 256;

pub struct GroundEffect {
    pub doodads: [Option<u32>; 4],
    pub density: u32,
}

pub struct GroundEffectCatalog {
    effects: HashMap<u32, GroundEffect>,
}

pub struct GroundPlacement {
    pub effect_id: u32,
    pub model_file_id: u32,
    pub position: [f32; 3],
    pub yaw: f32,
    pub scale: f32,
}

impl GroundEffectCatalog {
    pub fn load(source: &dyn AssetSource) -> Result<Self> {
        let doodad_csv = source.read_db2_csv("GroundEffectDoodad")?;
        let texture_csv = source.read_db2_csv("GroundEffectTexture")?;
        let doodads = parse_doodads(&doodad_csv)?;
        let effects = parse_effects(&texture_csv, &doodads)?;
        Ok(Self { effects })
    }

    pub fn effect(&self, id: u32) -> Option<&GroundEffect> {
        self.effects.get(&id)
    }
}

fn normalized_header(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn column(headers: &[String], names: &[&str]) -> Result<usize> {
    headers
        .iter()
        .position(|header| names.iter().any(|name| header == name))
        .with_context(|| format!("missing DB2 CSV column {}", names.join("/")))
}

fn parse_doodads(data: &[u8]) -> Result<HashMap<u32, u32>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalized_header).collect();
    let id = column(&headers, &["id", "internalid"])?;
    let model = column(&headers, &["modelfileid", "modelfiledataid", "modelid"])?;
    let internal = headers.iter().position(|header| header == "internalid");
    let mut result = HashMap::new();
    for record in reader.records() {
        let record = record?;
        let Some(model_id) = record
            .get(model)
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        if model_id == 0 {
            continue;
        }
        if let Some(row_id) = record.get(id).and_then(|value| value.parse::<u32>().ok()) {
            result.insert(row_id, model_id);
        }
        if let Some(internal_id) = internal
            .and_then(|index| record.get(index))
            .and_then(|value| value.parse::<u32>().ok())
        {
            result.insert(internal_id, model_id);
        }
    }
    Ok(result)
}

fn parse_effects(data: &[u8], doodads: &HashMap<u32, u32>) -> Result<HashMap<u32, GroundEffect>> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(data);
    let headers: Vec<String> = reader.headers()?.iter().map(normalized_header).collect();
    let id = column(&headers, &["id"])?;
    let density = column(&headers, &["density"])?;
    let slots = [
        column(&headers, &["doodadid0", "doodadid00"])?,
        column(&headers, &["doodadid1", "doodadid01"])?,
        column(&headers, &["doodadid2", "doodadid02"])?,
        column(&headers, &["doodadid3", "doodadid03"])?,
    ];
    let mut result = HashMap::new();
    for record in reader.records() {
        let record = record?;
        let Some(effect_id) = record.get(id).and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let mut models = [None; 4];
        for (index, column) in slots.iter().enumerate() {
            let doodad_id = record
                .get(*column)
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(u32::MAX);
            if doodad_id != u32::MAX {
                models[index] = doodads.get(&doodad_id).copied();
            }
        }
        if models.iter().all(Option::is_none) {
            continue;
        }
        result.insert(
            effect_id,
            GroundEffect {
                doodads: models,
                density: record
                    .get(density)
                    .and_then(|value| value.parse::<u32>().ok())
                    .unwrap_or(0),
            },
        );
    }
    Ok(result)
}

pub fn scatter(
    chunk: &McnkChunk,
    catalog: &GroundEffectCatalog,
    tile: [u32; 2],
    origin: [f32; 3],
) -> Vec<GroundPlacement> {
    let Some(heights) = &chunk.heights else {
        return Vec::new();
    };
    let Some(layers) = &chunk.layers else {
        return Vec::new();
    };
    if heights.heights.len() != 145 {
        return Vec::new();
    }
    let global_x = tile[0] * 16 + chunk.header.index_x;
    let global_y = tile[1] * 16 + chunk.header.index_y;
    let mut random = BlizzardRandomizer::new((global_y << 16) | (global_x & 0xffff));
    let cells: Vec<(usize, usize)> = (0..FRILL_DENSITY.min(MAX_DENSITY))
        .map(|_| {
            (
                (random.shuffle() & 7) as usize,
                (random.shuffle() & 7) as usize,
            )
        })
        .collect();
    let position = |index: usize, x: f32, y: f32| {
        [
            chunk.header.position[1] - x * super::importer::GRID_STEP - origin[0],
            chunk.header.position[2] + heights.heights[index] - origin[1],
            chunk.header.position[0] - y * super::importer::GRID_STEP - origin[2],
        ]
    };
    let mut result = Vec::new();
    for (list_index, (row, col)) in cells.into_iter().enumerate() {
        if chunk.header.is_no_effect_doodad(col, row)
            || chunk.header.is_hole_high_res(col, row)
            || (!chunk.header.flags.high_res_holes()
                && chunk.header.is_hole_low_res(col >> 1, row >> 1))
        {
            continue;
        }
        let layer = chunk.header.get_pred_texture(col, row) as usize;
        let Some(effect_id) = layers.layers.get(layer).map(|layer| layer.effect_id) else {
            continue;
        };
        let Some(effect) = catalog.effect(effect_id) else {
            continue;
        };
        let density = if effect.density == 0 {
            8
        } else {
            effect.density
        };
        let tl = position(row * 17 + col, col as f32, row as f32);
        let tr = position(row * 17 + col + 1, col as f32 + 1.0, row as f32);
        let ctr = position(row * 17 + 9 + col, col as f32 + 0.5, row as f32 + 0.5);
        let bl = position((row + 1) * 17 + col, col as f32, row as f32 + 1.0);
        let br = position((row + 1) * 17 + col + 1, col as f32 + 1.0, row as f32 + 1.0);
        for index in 0..density as usize {
            let rx = random.signed_unit();
            let ry = random.signed_unit();
            let Some(model_file_id) = effect.doodads[(index + list_index) & 3] else {
                continue;
            };
            let scale = random.signed_unit() * 0.1 + 1.0;
            let yaw = random.signed_unit() * std::f32::consts::PI;
            result.push(GroundPlacement {
                effect_id,
                model_file_id,
                position: fan_point((rx + 1.0) * 0.5, (ry + 1.0) * 0.5, tl, tr, bl, br, ctr),
                yaw,
                scale,
            });
        }
    }
    result
}

fn fan_point(
    fx: f32,
    fy: f32,
    tl: [f32; 3],
    tr: [f32; 3],
    bl: [f32; 3],
    br: [f32; 3],
    center: [f32; 3],
) -> [f32; 3] {
    let (u, v, w, a, b) = if fx + fy <= 1.0 {
        if fy <= fx {
            (2.0 * fy, 1.0 - fx - fy, fx - fy, tl, tr)
        } else {
            (2.0 * fx, fy - fx, 1.0 - fx - fy, bl, tl)
        }
    } else if fy >= fx {
        (2.0 * (1.0 - fy), fx + fy - 1.0, fy - fx, br, bl)
    } else {
        (2.0 * (1.0 - fx), fx - fy, fx + fy - 1.0, tr, br)
    };
    [
        u * center[0] + v * a[0] + w * b[0],
        u * center[1] + v * a[1] + w * b[1],
        u * center[2] + v * a[2] + w * b[2],
    ]
}

#[rustfmt::skip]
const GROUND_EFFECT_NOISE: [u8; 256] = [
    0x8e,0x14,0x27,0x99,0xfd,0xaa,0xc7,0x08,0xd5,0xe6,0x3e,0x1f,0xf6,0xbb,0x55,0xda,
    0x75,0xa0,0x4a,0x6a,0xe8,0xbd,0x97,0xff,0xde,0x9b,0xbc,0x9f,0x81,0x8a,0xa1,0x46,
    0x6e,0x0b,0xe3,0x63,0x76,0x7a,0x6c,0x5d,0x88,0xd3,0x69,0xca,0xc3,0x47,0xb9,0x25,
    0x83,0xab,0xa2,0x3f,0xa6,0x41,0x7c,0xba,0xe5,0xac,0x95,0x01,0x7e,0xcf,0x09,0xc1,
    0xd9,0x62,0x70,0x71,0x8d,0xdb,0x05,0x02,0x24,0x87,0xef,0x54,0xc6,0xd4,0x37,0x30,
    0xd0,0x1b,0xcb,0x7b,0xb8,0xe4,0xd8,0xec,0x49,0xce,0xad,0xdc,0x13,0xa9,0x94,0xc4,
    0x8f,0x39,0xae,0x0d,0x18,0x52,0xdd,0x0e,0x78,0xfa,0xf5,0x85,0x58,0xd2,0xaf,0x6d,
    0xa4,0xb2,0x53,0x3b,0x51,0xa5,0x50,0xbe,0xfc,0x2d,0xf4,0x11,0x48,0x98,0x16,0xf1,
    0x86,0xdf,0x3d,0x66,0x5e,0x44,0x2e,0x2f,0x36,0x07,0x6b,0x17,0x8b,0x29,0x4c,0xb6,
    0xe2,0x89,0x5f,0xe7,0xcd,0xa7,0x21,0xe1,0x4d,0xc9,0x65,0xed,0xfe,0xee,0x9c,0x23,
    0x33,0x7d,0xb7,0x04,0x9e,0x9a,0x2a,0x40,0xb3,0x10,0x5b,0xf3,0x82,0x77,0x1c,0x92,
    0x20,0x4e,0x1e,0x57,0x22,0x72,0x06,0x8c,0x67,0x2c,0x73,0xfb,0x59,0xc2,0x0a,0xbf,
    0x79,0x5c,0xf9,0x0c,0x28,0x1a,0x12,0x68,0x74,0x34,0x19,0x42,0xb1,0xc0,0x84,0xf8,
    0x38,0xf0,0x15,0x9d,0x60,0xf2,0x3a,0x6f,0xb4,0x90,0xeb,0x91,0x1d,0x7f,0x35,0x61,
    0x5a,0x32,0x03,0x56,0xa3,0xc5,0x2b,0x93,0x80,0x0f,0x4b,0x43,0xf7,0xa8,0xe0,0x3c,
    0x96,0xd1,0x64,0x26,0xd7,0x45,0xcc,0x4f,0xc8,0xb0,0xe9,0xb5,0x00,0xd6,0x31,0xea,
];

struct BlizzardRandomizer {
    source: u32,
    seed: u32,
}

impl BlizzardRandomizer {
    fn new(source: u32) -> Self {
        let seed = ((source % 0x2f) << 26)
            | ((source % 0x35) << 18)
            | ((source % 0x3b) << 10)
            | (4 * (source % 0x3d));
        Self { source, seed }
    }

    fn noise32(index: u8) -> u32 {
        let byte = |offset: u8| u32::from(GROUND_EFFECT_NOISE[index.wrapping_add(offset) as usize]);
        byte(0) | (byte(1) << 8) | (byte(2) << 16) | (byte(3) << 24)
    }

    fn shuffle(&mut self) -> u32 {
        let lane = |byte: u32, sub: i32, wrap: i32| -> u8 {
            let index = (byte & 0xff) as i32 - sub;
            if index < 0 {
                (index + wrap) as u8
            } else {
                index as u8
            }
        };
        let a = lane(self.seed, 0x1c, 0xf4);
        let b = lane(self.seed >> 8, 0x18, 0xec);
        let c = lane(self.seed >> 16, 0x0c, 0xd4);
        let d = lane(self.seed >> 24, 0x04, 0xbc);
        self.seed =
            u32::from(a) | (u32::from(b) << 8) | (u32::from(c) << 16) | (u32::from(d) << 24);
        self.source = self.source.wrapping_add(
            Self::noise32(a)
                ^ Self::noise32(d).rotate_left(1)
                ^ Self::noise32(c).rotate_left(2)
                ^ Self::noise32(b).rotate_left(3),
        );
        self.source
    }

    fn signed_unit(&mut self) -> f32 {
        let value = self.shuffle();
        let float = f32::from_bits((value & 0x007f_ffff) | 0x3f80_0000);
        if value as i32 >= 0 {
            float - 2.0
        } else {
            2.0 - float
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BlizzardRandomizer, GROUND_EFFECT_NOISE};

    #[test]
    fn noise_is_a_permutation() {
        let mut seen = [false; 256];
        for value in GROUND_EFFECT_NOISE {
            seen[value as usize] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn randomizer_is_deterministic() {
        let mut first = BlizzardRandomizer::new(0x0030_0020);
        let mut second = BlizzardRandomizer::new(0x0030_0020);
        for _ in 0..1000 {
            assert_eq!(first.shuffle(), second.shuffle());
        }
    }
}
