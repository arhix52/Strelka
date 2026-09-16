# wow2strelka

`wow2strelka` converts a WoW map location into a normalized, origin-rebased
Strelka scene. It reads an installed CASC client directly, uses the
`warcraft-rs` format crates for ADT/WMO/M2/BLP, and keeps scene structure,
materials, and WoW lighting metadata separate.

## Output

```text
scene/
  scene.gltf       geometry, hierarchy, close/aerial cameras and instancing
  scene.bin        external glTF buffer
  scene.mtlx       Strelka-compatible MaterialX sidecar
  scene.json       lossless WoW-specific scene metadata
  materials/       one MaterialX/OpenPBR document per source material
  textures/        BLP textures converted to PNG
```

M2 and WMO meshes are deduplicated by normalized WoW path. Their placements are
written as `EXT_mesh_gpu_instancing` translation/rotation/scale accessors, so
500 copies of one tree remain one mesh and 500 transforms.

Terrain material documents under `materials/` retain the source layer order,
BLP resources, MCAL alpha maps, layer flags and `COLOR_0` vertex data. The root
`scene.mtlx` uses a 512x512 composited terrain preview because Strelka's current
MaterialX loader does not yet evaluate arbitrary terrain `mix` graphs. The full
splat graph and all layer metadata remain available in the per-material files.
M2/WMO alpha modes and double-sided flags are also represented in glTF so
Strelka's any-hit and opacity-micromap paths preserve foliage cutouts.

## Build

Rust 1.92 or newer is required by `warcraft-rs`:

```powershell
cd tools/wow2strelka
cargo build --release
```

On Windows, run this from a Visual Studio Developer PowerShell with the C++
build tools installed.

## Usage

Directly from a current Classic installation:

```powershell
wow2strelka `
  --client "C:\Program Files (x86)\World of Warcraft" `
  --product wow_classic `
  --map kalimdor `
  --area barrens `
  --tiles 32,32 33,32 34,32 `
  --output .\scenes\barrens
```

`eastern-kingdoms` and `ek` are aliases for the client map directory
`azeroth`. `--client` may also point to a loose extracted tree with paths such
as `world/maps/kalimdor/...`.

When tile selection is maintained separately, use an area manifest:

```json
{
  "tiles": [[32, 32], [33, 32], [34, 32]]
}
```

```powershell
wow2strelka `
  --client "C:\Program Files (x86)\World of Warcraft" `
  --map kalimdor `
  --area barrens `
  --area-file .\areas\barrens.json `
  --output .\scenes\barrens
```

An area name alone is not enough to select tiles reliably: WoW area boundaries
come from AreaTable/UiMap metadata and may cross ADT tiles. Until DB2 area
resolution is implemented, pass `--tiles` or `--area-file`.

The CASC backend downloads and caches the community listfile when needed.
`--listfile <path>` makes that input explicit and supports offline/reproducible
conversion.

## Current scope

- ADT heightfield geometry, normals, UVs, vertex colors, texture layers and
  alpha maps
- M2 static bind-pose geometry from the first skin profile
- WMO group geometry, materials, lights and fog
- ADT MH2O/MCLQ and WMO MLIQ liquid surfaces
- original DB2-driven ground-effect doodads with deterministic GPU instancing
- bounds-derived `close` and `aerial` perspective cameras
- BLP-to-PNG conversion
- mesh deduplication and GPU instance transforms

Animation, WMO doodad-set expansion, DB2-driven area lookup, shader
combiner emulation and automatic normal/emissive texture classification remain
explicit future stages. Unsupported individual assets are listed in
`scene.json` instead of aborting the whole location export.

GroundEffect DB2 CSV and CASC files missing from a partial installation are
downloaded from Wago and cached under the selected `--cache` directory.
