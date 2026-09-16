use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};
use wow_adt::chunks::mh2o::VertexDataArray;
use wow_adt::{LiquidType, McnkChunk, Mh2oEntry, Mh2oInstance, RootAdt};

use crate::ir::Vertex;

const GRID_STEP: f32 = (533.333_3 / 16.0) / 8.0;

pub struct LiquidSurface {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub liquid_type: u16,
    pub chunk: Option<[u32; 2]>,
    pub fishable: bool,
    pub deep: bool,
    pub layer: usize,
}

pub fn adt_surfaces(root: &RootAdt, tile: [u32; 2], origin: [f32; 3]) -> Vec<LiquidSurface> {
    let mut surfaces = Vec::new();
    for chunk in &root.mcnk_chunks {
        let entry = root.water_data.as_ref().and_then(|water| {
            water.get_entry(chunk.header.index_x as usize, chunk.header.index_y as usize)
        });
        if let Some(entry) = entry
            && entry.has_liquid()
        {
            for (layer, instance) in entry.instances.iter().enumerate() {
                if let Some(surface) = mh2o_surface(chunk, entry, instance, layer, tile, origin) {
                    surfaces.push(surface);
                }
            }
            continue;
        }
        if let Some(liquid) = &chunk.liquid {
            let mut vertices = Vec::with_capacity(81);
            for y in 0..9 {
                for x in 0..9 {
                    let vertex = &liquid.vertices[y * 9 + x];
                    vertices.push(Vertex {
                        position: [
                            chunk.header.position[1] - x as f32 * GRID_STEP - origin[0],
                            vertex.height - origin[1],
                            chunk.header.position[0] - y as f32 * GRID_STEP - origin[2],
                        ],
                        normal: [0.0, 1.0, 0.0],
                        uv0: [x as f32 / 8.0, y as f32 / 8.0],
                        color0: [255; 4],
                    });
                }
            }
            let indices = grid_indices(8, 8, |x, y| liquid.tile_flags[y * 8 + x] != 0x0f);
            if !indices.is_empty() {
                surfaces.push(LiquidSurface {
                    name: format!(
                        "liquid_{}_{}_{}_{}_legacy",
                        tile[0], tile[1], chunk.header.index_x, chunk.header.index_y
                    ),
                    vertices,
                    indices,
                    liquid_type: legacy_liquid_id(liquid.liquid_type),
                    chunk: Some([chunk.header.index_x, chunk.header.index_y]),
                    fishable: false,
                    deep: false,
                    layer: 0,
                });
            }
        }
    }
    surfaces
}

fn mh2o_surface(
    chunk: &McnkChunk,
    entry: &Mh2oEntry,
    instance: &Mh2oInstance,
    layer: usize,
    tile: [u32; 2],
    origin: [f32; 3],
) -> Option<LiquidSurface> {
    if !instance.validate_dimensions() {
        return None;
    }
    let width = instance.width as usize;
    let height = instance.height as usize;
    let x_offset = instance.x_offset as usize;
    let y_offset = instance.y_offset as usize;
    let vertex_data = entry.vertex_data.get(layer).and_then(Option::as_ref);
    let mut vertices = Vec::with_capacity((width + 1) * (height + 1));
    for y in 0..=height {
        for x in 0..=width {
            let gx = x_offset + x;
            let gy = y_offset + y;
            let water_height = mh2o_height(vertex_data, gx, gy, instance.min_height_level);
            vertices.push(Vertex {
                position: [
                    chunk.header.position[1] - gx as f32 * GRID_STEP - origin[0],
                    water_height - origin[1],
                    chunk.header.position[0] - gy as f32 * GRID_STEP - origin[2],
                ],
                normal: [0.0, 1.0, 0.0],
                uv0: [gx as f32 / 8.0, gy as f32 / 8.0],
                color0: [255; 4],
            });
        }
    }
    let bitmap = entry.exists_bitmaps.get(layer).copied().flatten();
    let indices = grid_indices(width, height, |x, y| {
        bitmap.is_none_or(|bits| bits & (1u64 << (y * width + x)) != 0)
    });
    if indices.is_empty() {
        return None;
    }
    let attrs = entry.attributes.as_ref();
    Some(LiquidSurface {
        name: format!(
            "liquid_{}_{}_{}_{}_{}",
            tile[0], tile[1], chunk.header.index_x, chunk.header.index_y, layer
        ),
        vertices,
        indices,
        liquid_type: instance.liquid_type,
        chunk: Some([chunk.header.index_x, chunk.header.index_y]),
        fishable: attrs.is_some_and(|value| value.fishable != 0),
        deep: attrs.is_some_and(|value| value.deep != 0),
        layer,
    })
}

fn mh2o_height(data: Option<&VertexDataArray>, x: usize, y: usize, fallback: f32) -> f32 {
    let index = y * 9 + x;
    match data {
        Some(VertexDataArray::HeightDepth(values)) => values[index]
            .map(|value| value.absolute_height(fallback))
            .unwrap_or(fallback),
        Some(VertexDataArray::HeightUv(values)) => values[index]
            .map(|value| value.absolute_height(fallback))
            .unwrap_or(fallback),
        Some(VertexDataArray::HeightUvDepth(values)) => values[index]
            .map(|value| value.absolute_height(fallback))
            .unwrap_or(fallback),
        Some(VertexDataArray::DepthOnly(_)) | None => fallback,
    }
}

fn legacy_liquid_id(liquid_type: LiquidType) -> u16 {
    match liquid_type {
        LiquidType::Water => 0,
        LiquidType::Ocean => 1,
        LiquidType::Magma => 2,
        LiquidType::Slime => 3,
    }
}

fn grid_indices(
    width: usize,
    height: usize,
    mut enabled: impl FnMut(usize, usize) -> bool,
) -> Vec<u32> {
    let stride = width + 1;
    let mut indices = Vec::with_capacity(width * height * 6);
    for y in 0..height {
        for x in 0..width {
            if !enabled(x, y) {
                continue;
            }
            let top_left = (y * stride + x) as u32;
            let top_right = top_left + 1;
            let bottom_left = ((y + 1) * stride + x) as u32;
            let bottom_right = bottom_left + 1;
            indices.extend([
                top_left,
                bottom_left,
                bottom_right,
                top_left,
                bottom_right,
                top_right,
            ]);
        }
    }
    indices
}

pub fn wmo_surface(data: &[u8], name: &str) -> Result<Option<LiquidSurface>> {
    let Some(offset) = data.windows(4).position(|window| window == b"QILM") else {
        return Ok(None);
    };
    if offset + 8 > data.len() {
        bail!("truncated WMO MLIQ header");
    }
    let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
    let payload = data
        .get(offset + 8..offset + 8 + size)
        .ok_or_else(|| anyhow::anyhow!("truncated WMO MLIQ payload"))?;
    if payload.len() < 30 {
        return Ok(None);
    }
    let mut reader = Cursor::new(payload);
    let x_vertices = reader.read_u32::<LittleEndian>()? as usize;
    let y_vertices = reader.read_u32::<LittleEndian>()? as usize;
    let x_tiles = reader.read_u32::<LittleEndian>()? as usize;
    let y_tiles = reader.read_u32::<LittleEndian>()? as usize;
    let base_x = reader.read_f32::<LittleEndian>()?;
    let base_y = reader.read_f32::<LittleEndian>()?;
    let _base_z = reader.read_f32::<LittleEndian>()?;
    let _material_id = reader.read_u16::<LittleEndian>()?;
    if x_vertices == 0
        || y_vertices == 0
        || x_tiles + 1 != x_vertices
        || y_tiles + 1 != y_vertices
        || x_vertices * y_vertices > 1_000_000
    {
        return Ok(None);
    }
    let mut vertices = Vec::with_capacity(x_vertices * y_vertices);
    for y in 0..y_vertices {
        for x in 0..x_vertices {
            let mut union = [0u8; 4];
            reader.read_exact(&mut union)?;
            let height = reader.read_f32::<LittleEndian>()?;
            vertices.push(Vertex {
                position: [
                    base_x + x as f32 * GRID_STEP,
                    height,
                    -(base_y + y as f32 * GRID_STEP),
                ],
                normal: [0.0, 1.0, 0.0],
                uv0: [
                    x as f32 / x_tiles.max(1) as f32,
                    y as f32 / y_tiles.max(1) as f32,
                ],
                color0: [255; 4],
            });
        }
    }
    let mut tiles = vec![0x0f; x_tiles * y_tiles];
    reader.read_exact(&mut tiles)?;
    let indices = grid_indices(x_tiles, y_tiles, |x, y| tiles[y * x_tiles + x] != 0x0f);
    if indices.is_empty() {
        return Ok(None);
    }
    Ok(Some(LiquidSurface {
        name: format!("liquid_{name}"),
        vertices,
        indices,
        liquid_type: 0,
        chunk: None,
        fishable: false,
        deep: false,
        layer: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::grid_indices;

    #[test]
    fn full_liquid_grid_has_two_triangles_per_cell() {
        assert_eq!(grid_indices(8, 8, |_, _| true).len(), 8 * 8 * 6);
    }

    #[test]
    fn liquid_mask_omits_disabled_cells() {
        assert_eq!(grid_indices(2, 2, |x, y| x == y).len(), 12);
    }
}
