mod exporter;
mod ground_effect;
mod importer;
mod ir;
mod liquid;
mod source;

use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::exporter::export_scene;
use crate::importer::Importer;
use crate::ir::AreaSelection;
use crate::source::{AssetSource, ClientSource};

#[derive(Debug, Parser)]
#[command(
    name = "wow2strelka",
    version,
    about = "Convert a WoW location into a normalized Strelka scene"
)]
struct Cli {
    /// WoW installation root containing .build.info, or a loose extracted tree.
    #[arg(long)]
    client: PathBuf,

    /// CASC product from .build.info.
    #[arg(long, default_value = "wow_classic")]
    product: String,

    /// WoW map directory name (aliases: eastern-kingdoms, ek).
    #[arg(long)]
    map: String,

    /// Logical area name stored in scene metadata.
    #[arg(long)]
    area: Option<String>,

    /// ADT coordinates, for example: --tiles 32,32 33,32 34,32
    #[arg(long, num_args = 1..)]
    tiles: Vec<Tile>,

    /// JSON area manifest with {"tiles": [[32,32], ...]}.
    #[arg(long)]
    area_file: Option<PathBuf>,

    /// CASC locale used when several localized root entries exist.
    #[arg(long, default_value = "enUS")]
    locale: String,

    /// Optional local community listfile (FileDataID;Path).
    #[arg(long)]
    listfile: Option<PathBuf>,

    /// Persistent CASC/listfile cache directory.
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Output scene directory.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct Tile([u32; 2]);

impl FromStr for Tile {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let Some((x, y)) = value.split_once(',') else {
            bail!("tile must use x,y syntax: '{value}'");
        };
        Ok(Self([
            x.parse()
                .with_context(|| format!("invalid tile x in '{value}'"))?,
            y.parse()
                .with_context(|| format!("invalid tile y in '{value}'"))?,
        ]))
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let map = normalize_map_name(&cli.map);
    let tiles = resolve_tiles(&cli)?;
    let cache = cli
        .cache
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("wow2strelka-cache"));
    let source = ClientSource::open(
        &cli.client,
        &cli.product,
        &cli.locale,
        cli.listfile.as_deref(),
        &cache,
    )?;
    let mut scene = Importer::new(&source).import_tiles(&map, &tiles)?;
    scene.metadata.schema_version = 1;
    scene.metadata.source.client = cli.client.display().to_string();
    scene.metadata.source.product = source.product();
    scene.metadata.source.build = source.build();
    scene.metadata.source.map = map;
    scene.metadata.source.area = cli.area;
    scene.metadata.source.tiles = tiles;
    scene.metadata.source.coordinate_system =
        "glTF right-handed, Y-up; WoW upper-left map space converted and origin-rebased".to_owned();
    export_scene(&mut scene, &source, &cli.output)?;
    println!(
        "Exported {} meshes, {} materials and {} instance transforms to {}",
        scene.meshes.len(),
        scene.materials.len(),
        scene.instances.values().map(Vec::len).sum::<usize>(),
        cli.output.display()
    );
    if !scene.metadata.warnings.is_empty() {
        println!(
            "Completed with {} recoverable asset warnings; see scene.json",
            scene.metadata.warnings.len()
        );
    }
    Ok(())
}

fn resolve_tiles(cli: &Cli) -> Result<Vec<[u32; 2]>> {
    if !cli.tiles.is_empty() {
        return Ok(cli.tiles.iter().map(|tile| tile.0).collect());
    }
    if let Some(path) = &cli.area_file {
        let data = fs::read(path)
            .with_context(|| format!("failed to read area manifest {}", path.display()))?;
        let selection: AreaSelection = serde_json::from_slice(&data)
            .with_context(|| format!("invalid area manifest {}", path.display()))?;
        if selection.tiles.is_empty() {
            bail!("area manifest contains no tiles");
        }
        return Ok(selection.tiles);
    }
    bail!(
        "--tiles or --area-file is required; --area names the scene but cannot uniquely resolve ADT tiles without AreaTable/UiMap data"
    )
}

fn normalize_map_name(map: &str) -> String {
    match map.to_ascii_lowercase().as_str() {
        "eastern-kingdoms" | "eastern_kingdoms" | "ek" => "azeroth".to_owned(),
        other => other.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Tile, normalize_map_name};
    use std::str::FromStr;

    #[test]
    fn parses_tile_pair() {
        assert_eq!(Tile::from_str("32,17").unwrap().0, [32, 17]);
    }

    #[test]
    fn maps_eastern_kingdoms_to_client_name() {
        assert_eq!(normalize_map_name("eastern-kingdoms"), "azeroth");
    }
}
