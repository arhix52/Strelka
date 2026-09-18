mod ac_db;
mod creature_db2;
mod exporter;
mod ground_effect;
mod importer;
mod ir;
mod liquid;
mod m2_animation;
mod npc_validation;
mod source;

use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::exporter::export_scene;
use crate::importer::Importer;
use crate::ir::{AreaSelection, AzerothCoreMetadata};
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

    /// Read NPC spawns from an AzerothCore world database.
    #[arg(long, env = "WOW2STRELKA_AC_URL", hide_env_values = true)]
    ac_url: Option<String>,

    /// AzerothCore map ID; inferred for Kalimdor and Eastern Kingdoms.
    #[arg(long)]
    ac_map: Option<u16>,

    /// AzerothCore spawn mask to include.
    #[arg(long, default_value_t = 1)]
    ac_spawn_mask: u32,

    /// AzerothCore phase mask; zero includes every phase.
    #[arg(long, default_value_t = 0)]
    ac_phase_mask: u32,

    /// Generate one isolated validation fixture per unique NPC variant.
    #[arg(long)]
    npc_validation_output: Option<PathBuf>,

    /// Optionally render NPC fixtures with this StrelkaCLI executable.
    #[arg(long)]
    npc_validation_strelka_cli: Option<PathBuf>,

    /// Normalized validation phases.
    #[arg(long, value_delimiter = ',', default_value = "0,0.25,0.5,0.75")]
    npc_validation_phases: Vec<f32>,

    /// NPC validation render width.
    #[arg(long, default_value_t = 256)]
    npc_validation_width: u32,

    /// NPC validation render height.
    #[arg(long, default_value_t = 256)]
    npc_validation_height: u32,
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
    let mut importer = Importer::new(&source);
    if let Some(url) = &cli.ac_url {
        let map_id = cli
            .ac_map
            .map(Ok)
            .unwrap_or_else(|| ac_db::infer_map_id(&map))?;
        let database = ac_db::AzerothCore::connect(url)?;
        let spawns = database.load_spawns(map_id, &tiles, cli.ac_spawn_mask, cli.ac_phase_mask)?;
        let catalog = creature_db2::CreatureCatalog::load(&source)?;
        let mut resolved = Vec::new();
        let mut skipped = 0usize;
        for spawn in spawns {
            match catalog.resolve(&source, spawn.clone())? {
                Some(npc) => resolved.push(npc),
                None => {
                    skipped += 1;
                    importer.push_warning(format!(
                        "NPC {} '{}' display {} cannot be resolved in client {}",
                        spawn.guid,
                        spawn.name,
                        spawn.display_id,
                        source.build()
                    ));
                }
            }
        }
        importer = importer.with_npcs(
            resolved,
            AzerothCoreMetadata {
                map_id,
                spawn_mask: cli.ac_spawn_mask,
                phase_mask: cli.ac_phase_mask,
                imported: 0,
                skipped,
            },
        );
    }
    let mut scene = importer.import_tiles(&map, &tiles)?;
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
    if let Some(relative) = &cli.npc_validation_output {
        if relative.is_absolute() || relative.components().count() != 1 {
            bail!("--npc-validation-output must be one directory name relative to --output");
        }
        let validation_output = cli.output.join(relative);
        let options = npc_validation::ValidationOptions {
            output: validation_output,
            phases: cli
                .npc_validation_phases
                .iter()
                .copied()
                .map(|phase| phase.clamp(0.0, 1.0))
                .collect(),
            width: cli.npc_validation_width.max(16),
            height: cli.npc_validation_height.max(16),
            strelka_cli: cli.npc_validation_strelka_cli.clone(),
        };
        let report = npc_validation::export_npc_validation(&scene, &cli.output, &options)?;
        println!(
            "Exported {} unique NPC validation fixtures to {}",
            report.unique_variant_count,
            options.output.display()
        );
    }
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
