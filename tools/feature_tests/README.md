# Feature tests: Cycles vs Strelka

A ladder of one-feature-per-scene comparisons. The point is that when a render
looks wrong you can tell *which* of the three layers broke — the Blender
exporter, Strelka's glTF loader, or Strelka's shading — instead of guessing.

Verified against **Blender 5.2.0 LTS**.

## Method

Both renderers write **linear EXR**. Nothing goes through a tone curve:

- Cycles: view transform `Standard`, look `None`, exposure 0, gamma 1.
- Strelka: `tonemap.type = "none"`, and the photometric exposure is pinned to
  exactly 1.0 via `iso=100, fstop=1, shutter=1` — that is what makes
  `cm2_factor * iso / (shutter * fstop²) / 100` come out at 1.

Comparing tonemapped images would squash every highlight difference and
manufacture agreement that isn't there.

The world is black in both -- which takes driving the Background node
explicitly, because `World.use_nodes = False` is ignored on Blender 5.x. Lighting is a single rect area light, and both
sides derive it from **one** physical number: a Blender area light of power `P`
over area `A` emits radiance `P / (A·π)`, which is what goes into the sidecar.
Scene 12 is the exception — it has no sidecar on purpose, so the loader falls
back to `KHR_lights_punctual`.

## Running it

```bash
# 1. Build scenes + Cycles references (~15 s per scene, ~4 min total)
/Applications/Blender.app/Contents/MacOS/Blender -b \
    -P tools/feature_tests/build_features.py -- --out scenes/feature_tests

# 2. Render with Strelka (needs StrelkaCLI — see below)
tools/feature_tests/run_strelka.sh

# 3. Compare
/Applications/Blender.app/Contents/MacOS/Blender -b \
    -P tools/feature_tests/compare.py -- --out scenes/feature_tests
```

Useful flags on step 1: `--only 07` to rebuild one scene, `--no-render` to skip
Cycles entirely (fast, when you only want to inspect the exported glTF).

**StrelkaCLI is not on `arhix/wavefront`.** It lives on `bdpt_dev` as commit
`15f2d28`; cherry-pick it and rebuild into `build/Release` before step 2.

## Reading the results

`compare.py` prints `rel` (mean absolute error over mean reference radiance),
`rmse`, and `ratio` (mean Strelka / mean Cycles). It also writes
`<scene>_compare.png`: reference | Strelka | 8× difference.

**Read `00_calibration` first.** It is a 0.18 grey sphere under the key light
and nothing else. If its `ratio` is not ≈1.0, then light units or exposure
disagree, and every other row is re-measuring that same offset rather than
telling you anything about the feature it names. Fix 00 before reading further.

If a row reports `[v-flipped]`, the images matched only after a vertical flip —
tinyexr and Blender disagreed on scanline order. That is a real bug worth
fixing, but it is not a shading bug.

## The scenes

| Scene | Tests | Expected today |
|---|---|---|
| `00_calibration` | light units + exposure | 0.021 / 1.010 |
| `01_srgb_texture` | sRGB decode of base colour | 0.028 / 0.998 |
| `02_basecolor` | untextured albedo | 0.023 / 1.003 |
| `03_roughness` | dielectric roughness ramp | 0.024 / 1.015 |
| `04_metal` | conductor roughness ramp | 0.056 / 0.986 |
| `05_anisotropy` | `KHR_materials_anisotropy` | 0.063 / 1.006 |
| `06_normalmap` | normal map + tangents | 0.072 / 1.060 |
| `07_alpha_clip` | `alphaMode: MASK` | 0.022 / 1.015 |
| `08_alpha_blend` | `alphaMode: BLEND` | 0.051 / 1.044 |
| `09_glass_ior` | `KHR_materials_ior` / `_transmission` | 0.059 / 1.012 |
| `10_glass_absorption` | `KHR_materials_volume` | 0.038 / 1.012 |
| `11_emission` | `KHR_materials_emissive_strength` | 0.017 / 1.004 |
| `12_lights_punctual` | point / spot / sun via KHR | 0.032 / 1.006 |
| `13_uv2_vcol` | `TEXCOORD_1`, `COLOR_0` | 0.024 / 1.008 |

The right-hand column is `rel / ratio` as measured, not a prediction. Every scene
is CLOSE or OK. Read them against the noise floor, which is what two runs of the
*same* renderer differ by at these sample counts: 0.006 on `00_calibration` and
0.037 on the caustic scenes. A row two or three times its floor still has
something real in it; a row at its floor does not.

## What the exporter actually emits

`build_features.py` reads each exported `.gltf` back and prints its extensions,
vertex attributes and alpha modes, then writes `export_manifest.json`. This is
deliberate: it settles "did Blender drop it or did Strelka?" as a fact rather
than an assumption, and it will catch the exporter changing behaviour on a
future Blender release.

As of Blender 5.2 the exporter does emit `KHR_materials_ior`, `_transmission`,
`_volume`, `_specular`, `_anisotropy`, `_emissive_strength`,
`KHR_lights_punctual`, plus `TANGENT`, `TEXCOORD_1` and `COLOR_0`. So for most
of the table above the data is in the file and the gap is on Strelka's side.

Two things needed specific node setups rather than material properties, because
the 4.2+ exporter infers them from the node graph:

- `alphaMode: MASK` needs a `Math:GREATER_THAN` feeding the Alpha socket.
  Setting `material.blend_method = 'CLIP'` is silently ignored and yields BLEND.
- `KHR_materials_volume` needs a node group named `glTF Material Output` with a
  `Thickness` socket, alongside a Volume Absorption node.
- **Cycles dims caustics by default.** `blur_glossy = 1.0` widens glossy lobes
  after a diffuse bounce and `sample_clamp_indirect = 10.0` truncates the
  high-energy indirect samples a caustic consists of. Both alter the image, not
  merely its variance, and left on they made correct caustics look twice too
  bright. `reset_scene()` zeroes them; the cost is a noisier reference, which the
  noise floor measurement already accounts for.
- **A sun's strength must go through its Emission node, not `light.energy`.**
  With Cycles active the exporter reads the lamp's Emission node Strength and
  ignores `energy` outright, while Cycles renders with the product of the two.
  Setting only `energy` therefore renders one sun and exports a different one,
  with no warning. `set_sun_strength()` pins `energy` to 1.0 and drives the
  node. Point and spot lamps are unaffected — their branch falls back to
  `energy`, which is what the "no quadratic light falloff node" warning is.

## Not covered

Curves/hair, displacement, subsurface, sheen, shape keys and Sun & Sky are all
absent — either glTF cannot carry them or Strelka cannot render them, so a
comparison would only restate what is already known. Add them as those features
land.
