# SHARC on Metal

The Metal backend implements the public SHARC 1.8.3 behavior as an independent
Metal port. It does not include NVIDIA SDK source. The reference contract is the
[SHARC integration guide](https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md)
and the public 1.8.3 shader API.

## Pipeline contract

Every SHARC frame uses this order on both Metal 3 and Metal 4:

1. Clear all resources after allocation, scene/geometry/material changes, or a
   grid/layout change. Responsive mode also clears per-frame accumulation.
2. Trace one time-distributed pixel per `sharc_update_downscale` square. The
   default of 5 traces about 4% of the image.
3. Resolve every occupied entry, combine temporal history, reproject a newly
   selected adjacent LOD after camera movement, evict stale entries, and clear
   consumed atomic accumulation.
4. Trace the full image and query only resolved data. A successful eligible
   query terminates the path.

Explicit device/buffer barriers separate all four stages. Update and query are
different function-constant variants and never read radiance being written by
the same update pass.

Update paths are **segment-local**, which is the part of the integration
contract that is easiest to get wrong. The cache stores radiance per vertex and
carries the connecting throughput itself, through `sharcSetThroughput`, so an
update path resets its own throughput to one for every new segment. Letting the
path throughput accumulate on top weights each deposit twice, and -- because
Russian roulette then divides by the whole path's throughput rather than the
segment's -- writes compensation factors of thousands into cells that see a
handful of samples a frame. With cache resampling reading those cells back, the
error compounds every frame: on `iso_bathroom` the cache's per-frame fixed-point
sum grew from 20 to 32 occupied bits over 64 frames, wrapped the accumulator,
and doubled the image's mean.

The sparse update pass also runs a shorter bounce loop than the render pass.
Cache resampling answers an update path from the cache once it has
`sharc_propagation_depth` vertices behind it, so it cannot reach the render
depth, and the subsurface walk budget -- 64 iterations on any scene with skin or
soap in it -- is a tail the cache samples far too sparsely to be worth that many
empty dispatches at 4% occupancy. See `MetalRender::sharcUpdateIterations`.

The cache has equal-capacity private buffers with 4-byte compact hash entries,
32-byte atomic accumulation entries, and 32-byte fp16 resolved entries. The
default `2^22` entries use about 272 MiB. Update-path state is allocated only for
the sparse grid; at 1920×1080 and downscale 5 it uses about 13.3 MiB.

## Feature coverage

| SHARC behavior | Metal implementation / setting |
| --- | --- |
| Perspective screen-sized logarithmic hash grid, normal octants | `sharc_base_size` in pixels; compact 32-bit Metal key |
| Sparse update and independently propagated path segments | `sharc_update_downscale`, `sharc_propagation_depth` |
| Resolve, temporal accumulation, stale eviction | `sharc_accum_frames`, `sharc_stale_frames`, shared with OptiX |
| Query early-out with segment, path-footprint and receiver-bandwidth gates | `sharc_depth`, `sharc_roughness_threshold`, `sharc_metal_min_samples` |
| Bucket probing with an empty-slot early-out and collision-recovered history | 16-entry buckets, 2 empty slots, 8-slot resolve probe window |
| Cache resampling during update | `sharc_cache_resampling` |
| Material demodulation | `sharc_material_demodulation` |
| Separate emissive evaluation | `sharc_separate_emissive` |
| Directional luminance SH reconstruction | `sharc_directional` |
| Paired main/responsive entries and short responsive history | `sharc_metal_responsive`, `sharc_responsive_frames` |
| Adjacent-level history transfer on camera movement | `sharc_blend_adjacent_levels` |
| 32-frame fading history and accelerated convergence | `sharc_fade_acceleration` |
| Native fp16 persistent storage | Always enabled; update weights use the upstream fp32 baseline |
| Cache and hash-map diagnostics | `DebugMode::eSharc*` (shared with OptiX), `sharc_debug` modes, `SHARC stats` log lines |

The advanced Editor controls expose every optional SHARC behavior in this
table. Directional radiance uses NVIDIA's YCoCg luminance SH representation and
updates the directionality weight after the current BSDF direction is sampled.
It is off by default, as upstream's `SHARC_ENABLE_SH_ENCODING` is: it stops a
bright glossy sample being reused in an unrelated direction, and pays for that
with a first-order reconstruction of a signal a cell otherwise stores exactly.
It cannot represent delta transmission and is not a substitute for rejecting a
cache query on a receiver whose outgoing radiance is sharper than the cache can
represent.

## Diagnostics

The four cache views live in the shared `DebugMode` enum, so both backends
answer the same menu entry with the same picture and `render/pt/debug` means one
thing across the tree. Each answers at the first surface a camera ray meets,
replaces the pixel and ends the path, the way the upstream sample's debug branch
does; the background and any emitter along the way are kept out, because either
is orders of magnitude brighter than a debug colour.

- **Cache: voxel grid** is NVIDIA's `HashGridDebugColoredHash`. It reads no cache
  buffer, so it works with SHaRC switched off and is where `sharc_base_size`
  gets chosen before anything is allocated.
- **Cache: radiance** is the sample's own SHaRC view -- the radiance a query
  would return. It is the one diagnostic that keeps the render's exposure and
  tone curve, because it holds radiance and is only worth reading next to the
  render. Black means the voxel is missing or has not resolved.
- **Cache: occupancy** is NVIDIA's 7x7 table-entry grid: green once an entry has
  resolved radiance to answer with, amber while it is only reserved, black free.
- **Cache: bounce count** is blue for none, green for one, yellow for two, red
  for three or more, and is the direct measurement of what the cache buys. On
  the Cornell box at 512x384, 16 spp, depth 8, by fraction of the frame:

  | | red (three or more) | green (one) |
  | --- | --- | --- |
  | cache off | 0.363 | 0.210 |
  | cache on | 0.007 | 0.654 |

Alongside them, `Radiance cache (SHaRC) -> Metal -> Hash map diagnostics` is
about the map rather than the cache: `Query: hit / miss` is green where a lookup
is answered and red where it fails, `Cached sample count` ramps blue to white,
`Cached-key colors` paints resident cells by the key that found them,
`Hash bucket collisions` uses NVIDIA's blue-to-red probe-depth palette, and
`Counters in log` collects `SHARC stats` without drawing anything.

Everything but the radiance view bypasses exposure, tonemapping and MetalFX -- a
photographic exposure applied to a hash colour means nothing, and a denoiser fed
a per-voxel diagnostic smooths away the thing being diagnosed. `StrelkaCLI`
applies the same rule when it writes a PNG; it computes its own exposure there,
and applying the default of about 1/140 to a debug colour of 0.2 is what used to
make every one of these views come out black.

Occupancy shows the first `(width/8/16) * (height/8) * 16` entries of the table,
which is NVIDIA's layout and about 3000 cells; at the default 4M capacity that is
a 0.07% sample of it. Drop the entry count to 65K when reading that view.

`SHaRC capacity` exposes power-of-two sizes from 65K to 4M entries. Resolve
dispatches one thread per entry, but an empty entry costs a key read and
nothing else, so the measured cost tracks occupancy rather than capacity:
`iso_bathroom` renders within the noise floor at 262K and at 4M. What capacity
buys is headroom against failed insertions -- use the smallest size that keeps
them at zero, and read it off the diagnostics rather than rebuilding.

Changing SHARC enablement or capacity replaces its private buffers. The Metal 4
queue residency set is refreshed on that resource generation before submission;
this is required even when the image resolution and wavefront allocations did
not change.

The compact key is used because it is supported across the Metal GPU families
targeted by Strelka. Since all 32 compact-key bits carry spatial data, responsive
mode divides the configured table into disjoint persistent and responsive halves.
This preserves a collision-free companion namespace without requiring 64-bit
atomics; the configured capacity is therefore shared by the two classes.
`sharc_metal_responsive` marks the complete lighting signal as responsive because the
current Strelka light ABI has no per-light responsiveness tag. Light and
environment edits retain this short-history cache; material and geometry edits
still invalidate it.

`sharc_base_size` is the target perspective footprint in pixels for both
backends. Field of view and render height are folded into the world-space base
size each frame; changing either invalidates Metal's cache because its keys then
describe a different grid. Metal offsets the logarithmic exponent by 16 so its
unsigned compact LOD does not clamp ordinary indoor distances to the first level.
The old Metal-only `sharc_scene_scale` key is ignored.
The legacy `sharc_min_samples` key still sets both backends when explicitly
present; `sharc_metal_min_samples` overrides it for this Metal implementation.

Metal's compact default entry has no side, medium state, or angular
representation; its optional first-order SH mode still cannot reconstruct a
sharp lobe. `sharc_roughness_threshold` therefore also controls the minimum
roughness of every active reflective layer at the receiver. Specular and diffuse
transmission, plus fibre scattering, always continue tracing. The same rule is
applied to render queries and update-pass cache resampling; rejected receivers
do not consume entries, but their traced transport is still propagated to
earlier cacheable vertices. The default is 0.4: the 256-spp bathroom check
matches uncached path tracing at that value. Lower values recover more cache
hits but admit angular bias; higher values give up still more cache use in
glossy interiors.

## Reproducible scene evaluation

Build and test first:

```bash
cmake --build build/Release -j10
ctest --test-dir build/Release --output-on-failure
```

Run the A/B harness from a Python environment with NumPy:

```bash
python3 -m venv /tmp/strelka-sharc-venv
/tmp/strelka-sharc-venv/bin/pip install numpy
/tmp/strelka-sharc-venv/bin/python tools/sharc_ab.py \
    scenes/validation/cornell_box/cornell_box.toml \
    scenes/validation/mixed_materials/mixed_materials.toml \
    scenes/validation/metal_sphere/metal_sphere.toml \
    --spp 64 --reference-spp 512 \
    --out /tmp/strelka-sharc-ab
```

The two Editor scenes have 960x540 validation configs matching the reported
interactive workload:

```bash
/tmp/strelka-sharc-venv/bin/python tools/sharc_ab.py \
    scenes/validation/kids_room/kids_room.toml \
    scenes/validation/iso_bathroom/iso_bathroom.toml \
    --spp 64 --reference-spp 512 \
    --out /tmp/strelka-sharc-rooms
```

The harness retains the exact TOMLs, EXRs, and logs. It reports steady-state
median milliseconds per sample, speedup, relative error/RMSE/p95 against a
high-sample cache-off reference, and cache insert/query statistics. The scored
off/on renders have debug atomics disabled; the default `--debug 4` produces a
separate stats-only diagnostic render, so instrumentation does not contaminate
the timing or quality comparison. `speedup > 1` means SHARC is faster;
`quality > 1` means its low-sample image is closer to the reference than cache-off
at the same sample count. Record the Mac/GPU, resolution, depth, sampler, and all
SHARC settings with results.

Use `--sharc-capacity 262144` (and separate output directories) to sweep cache
sizes without editing the source scenes. The diagnostic output reports occupancy
and an effective hit rate whose denominator includes segment and footprint
rejections. `--sharc-update-downscale N` similarly isolates sparse-update cost.
Repeatable `--sharc-option KEY=VALUE` arguments apply feature ablations only to
the SHARC-on and diagnostic runs. For example:

```bash
/tmp/strelka-sharc-venv/bin/python tools/sharc_ab.py \
    scenes/validation/iso_bathroom/iso_bathroom.toml \
    --spp 64 --reference-spp 512 --sharc-capacity 262144 \
    --sharc-option sharc_directional=true \
    --out /tmp/strelka-sharc-iso-directional
```

Diagnostics also report accumulation clamps, rejected non-finite values, and
the maximum occupied fixed-point/sample-count bits. A nonzero clamp count or
31-32 occupied radiance bits indicates that `sharc_radiance_scale` should be
reduced before interpreting image-quality results.

### Measured

Apple M-series, Metal 4 wavefront tracer, Release, 64 spp against a 512 spp
cache-off reference, depth 8, Sobol, defaults otherwise. `quality` is the
cache-off relative error over the cache-on one, so above 1 means SHARC's
low-sample image is the closer of the two.

| scene | resolution | speedup | off rel | on rel | quality |
| --- | --- | --- | --- | --- | --- |
| `cornell_box` | 512x384 | 1.13 | 0.1163 | 0.0867 | 1.34 |
| `kids_room` | 960x540 | 1.00 | 0.2534 | 0.2081 | 1.22 |
| `iso_bathroom` | 960x540 | 1.00 | 0.1994 | 0.1648 | 1.21 |
| `mixed_materials` | 512x384 | 0.75 | 0.3121 | 0.2743 | 1.14 |
| `metal_sphere` | 512x384 | 0.79 | 0.3637 | 0.3094 | 1.18 |

The two 512x384 scenes render in under 3 ms/sample, where the sparse update's
fixed dispatch cost is a quarter of the frame and there is not enough tracing
left for its queries to pay for it. That is the expected shape of the tradeoff,
not a defect: on `iso_bathroom` the same update pass costs 4 ms and its queries
remove 4.6 ms of tracing, and the margin grows with path depth and scene cost.

The feature ladder is a correctness check rather than a quality one -- cache-on
and cache-off should agree to within the estimator's own noise. At 64 spp:
`07_alpha_clip` 0.5%, `08_alpha_blend` 2.0%, `11_emission` 0.8%,
`18_bounded_volume` 0.5%, `20_mirror_and_floor` 3.5%, `25_subsurface` 3.7%,
`28_hair` 4.7%, `19_env_and_light` 1.8% mean relative difference.

Use this scene matrix when changing cache behavior:

- `cornell_box`: long diffuse paths and the expected best case.
- `mixed_materials`: material demodulation and rough glossy paths.
- `metal_sphere` and feature test `20_mirror_and_floor`: query rejection on
  delta/specular chains.
- Feature tests `07_alpha_clip` and `11_emission`: pass-through throughput and
  separate emissive handling.
- Feature tests `18_bounded_volume` and `25_subsurface`: participating-media and
  subsurface propagation.
- `brainstem` and `28_hair`: complex geometry where sparse-update overhead and
  short eligible paths can erase the benefit.

## Diagnosing “SHARC does not help”

- No queries: `sharc_depth` is beyond the traced paths, or all candidate lobes
  fail the segment/footprint gate.
- Low hit rate: the cache is still warming, `sharc_metal_min_samples` is too high,
  voxels are too fine, entries go stale too quickly, or insert failures show
  insufficient capacity.
- Many collisions/failed insertions: increase `sharc_capacity`; also inspect
  whether `sharc_base_size` creates far more cells than the scene needs.
- High hit rate without speedup: paths are already short/cheap, update+resolve
  costs more than the terminated bounces, or max depth is too low for SHARC to
  remove meaningful work. Increase update downscale only after checking quality.
- Speedup with visible bias or light leaks: use smaller voxels (reduce the pixel
  size), raise the minimum sample count, and isolate material demodulation,
  separate emissive, and directional encoding one at a time. If those do not
  move the result, inspect the material at the query hit: the path-footprint
  test describes the lobes before that hit and does not by itself make a
  glossy/transmissive receiver safe for nondirectional outgoing radiance.
- Lag after lighting changes: enable responsive lighting only for that test and
  tune `sharc_responsive_frames`; it intentionally costs extra cache work.
- No benefit in animated geometry: Strelka clears SHARC on geometry refits for
  correctness. A dynamic-geometry reuse policy needs stable surface identities,
  which the current world-position key does not provide.

Measure performance in Release with Metal validation disabled. Use Metal API and
Shader Validation only for correctness runs; use an Xcode GPU capture or Metal
System Trace to attribute Update, Resolve, and Query costs.
