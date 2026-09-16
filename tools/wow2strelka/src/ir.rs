use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub instances: BTreeMap<usize, Vec<Transform>>,
    pub nodes: Vec<Node>,
    pub cameras: Vec<Camera>,
    pub metadata: SceneMetadata,
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub source: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitive>,
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv0: [f32; 2],
    pub color0: [u8; 4],
}

#[derive(Debug)]
pub struct Primitive {
    pub first_index: usize,
    pub index_count: usize,
    pub material: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Material {
    pub name: String,
    pub source: String,
    pub kind: MaterialKind,
    pub layers: Vec<TextureLayer>,
    pub normal: Option<TextureRef>,
    pub emissive: Option<TextureRef>,
    pub blend: BlendMode,
    pub double_sided: bool,
    pub unlit: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialKind {
    Terrain,
    M2,
    Wmo,
    Liquid,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextureLayer {
    pub texture: TextureRef,
    pub alpha_map: Option<String>,
    #[serde(skip)]
    pub alpha_data: Option<Vec<u8>>,
    pub effect_id: Option<u32>,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextureRef {
    pub wow_path: String,
    pub output_path: String,
    pub color_space: ColorSpace,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    Opaque,
    Mask,
    Blend,
    Additive,
    Modulate,
}

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };
}

#[derive(Debug)]
pub struct Node {
    pub name: String,
    pub mesh: Option<usize>,
    pub transform: Transform,
    pub children: Vec<usize>,
}

#[derive(Debug)]
pub struct Camera {
    pub name: String,
    pub transform: Transform,
    pub yfov: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Debug, Default, Serialize)]
pub struct SceneMetadata {
    pub schema_version: u32,
    pub source: SourceMetadata,
    pub environment: EnvironmentMetadata,
    pub lights: Vec<LightMetadata>,
    pub fog: Vec<FogMetadata>,
    pub wmo: Vec<WmoMetadata>,
    pub materials: BTreeMap<String, MaterialMetadata>,
    pub terrain: Vec<TerrainMetadata>,
    pub liquids: Vec<LiquidMetadata>,
    pub ground_effects: Vec<GroundEffectMetadata>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct SourceMetadata {
    pub client: String,
    pub product: String,
    pub build: String,
    pub map: String,
    pub area: Option<String>,
    pub tiles: Vec<[u32; 2]>,
    pub coordinate_system: String,
    pub origin: [f32; 3],
}

#[derive(Debug, Default, Serialize)]
pub struct EnvironmentMetadata {
    pub ambient_color: Option<[f32; 4]>,
    pub skybox: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LightMetadata {
    pub source: String,
    pub kind: String,
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub attenuation: [f32; 2],
}

#[derive(Debug, Serialize)]
pub struct FogMetadata {
    pub source: String,
    pub color: [f32; 4],
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Serialize)]
pub struct WmoMetadata {
    pub source: String,
    pub doodad_set: u16,
    pub name_set: u16,
}

#[derive(Debug, Serialize)]
pub struct MaterialMetadata {
    pub document: String,
    pub source: String,
    pub blend: BlendMode,
    pub double_sided: bool,
    pub unlit: bool,
}

#[derive(Debug, Serialize)]
pub struct TerrainMetadata {
    pub tile: [u32; 2],
    pub chunk: [u32; 2],
    pub area_id: u32,
    pub material: String,
    pub vertex_color_attribute: String,
    pub layer_count: usize,
}

#[derive(Debug, Serialize)]
pub struct LiquidMetadata {
    pub source: String,
    pub tile: Option<[u32; 2]>,
    pub chunk: Option<[u32; 2]>,
    pub layer: usize,
    pub liquid_type: u16,
    pub fishable: bool,
    pub deep: bool,
}

#[derive(Debug, Serialize)]
pub struct GroundEffectMetadata {
    pub effect_id: u32,
    pub model_file_ids: Vec<u32>,
    pub density: u32,
    pub instance_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct AreaSelection {
    pub tiles: Vec<[u32; 2]>,
}
