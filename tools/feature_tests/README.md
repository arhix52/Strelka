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
| `01_srgb_texture` | sRGB decode of base colour | 0.026 / 1.004 |
| `02_basecolor` | untextured albedo | 0.022 / 1.003 |
| `03_roughness` | dielectric roughness ramp | 0.024 / 1.015 |
| `04_metal` | conductor roughness ramp | 0.055 / 0.986 |
| `05_anisotropy` | `KHR_materials_anisotropy` | 0.061 / 1.007 |
| `06_normalmap` | normal map + tangents | 0.071 / 1.061 |
| `07_alpha_clip` | `alphaMode: MASK` | 0.025 / 1.018 |
| `08_alpha_blend` | `alphaMode: BLEND` | 0.024 / 1.016 |
| `09_glass_ior` | `KHR_materials_ior` / `_transmission` | 0.059 / 1.012 |
| `10_glass_absorption` | `KHR_materials_volume` | 0.038 / 1.012 |
| `11_emission` | `KHR_materials_emissive_strength` | 0.017 / 1.004 |
| `12_lights_punctual` | point / spot / sun via KHR | 0.031 / 1.002 |
| `13_uv2_vcol` | `TEXCOORD_1`, `COLOR_0` | 0.024 / 1.010 |
| `14_sheen` | `KHR_materials_sheen` roughness ramp | 0.053 / 1.016 |
| `15_clearcoat` | `KHR_materials_clearcoat` + IOR ramp | 0.038 / 0.995 |
| `16_iridescence` | `KHR_materials_iridescence` thickness ramp | 0.023 / 1.010 |
| `17_coated_glass` | transmission + clearcoat together | 0.067 / 0.994 |
| `18_bounded_volume` | `STRELKA_materials_medium` | 0.027 / 1.000 |

`18_bounded_volume` used to be the row that failed on purpose, first at 1.899 and
then at 1.215. Both numbers were real and only the first was Strelka's fault.

The 1.899 was shadow rays not attenuating through a bounded medium, and that is
fixed: they take a second traversal against `GEOMETRY_MASK_MEDIUM` and accumulate
optical depth across the boundaries they cross.

The 1.215 left over was the reference. `scene.cycles.volume_bounces` defaults to
**0** in Blender and this file set every other bounce limit but not that one --
and 0 does not mean "no volumes", it means single scattering. So the row was
comparing Strelka's multiply scattered medium against a reference that scatters
once. What makes that diagnosis rather than a guess is that the gap tracks the
albedo: rebuild the scene at a single-scattering albedo of 0.05 and the two agree
to within 1% down the whole box, at 1.0 the reference comes out 2.4x darker.
Setting `volume_bounces = MAX_DEPTH` takes the row to 1.000.

A reference is a measurement too, and this one had an unstated setting in it.

`17_coated_glass` is a regression guard rather than a feature test. Transmission
and a clearcoat on the same material is the one configuration that hid a
double-count: the separate specular lobe's selection weight is zeroed for a
transmissive material, because the transmission lobe runs its own Fresnel, while
its BRDF was still summed into `f_total`. With no other reflection lobe
selectable the specular term is never evaluated, which is why plain glass never
showed it -- add a coat and it rides along, divided by a pdf that does not
include it. It shipped as soap bubbles that glowed instead of being transparent.

Two more rows need reading with their per-sphere behaviour in hand, because the
whole-frame `ratio` hides what they are actually saying:

- **`14_sheen` is not an agreement check.** Cycles' Principled uses Zeltner et
  al.'s microflake sheen; Strelka implements Charlie with Ashikhmin visibility,
  because that is what the extension is specified against. Across the roughness
  ramp ours runs +25% at roughness 0.05 and −20% at roughness 1.0, crossing over
  around 0.6 — so the aggregate ratio of 1.016 is two errors cancelling, not two
  renderers agreeing. Treat the row as a regression guard: if it moves, something
  on our side changed.
- **`15_clearcoat` is an agreement check**, and it finds a real approximation.
  The IOR 1.0 sphere — where the coat's F0 is zero and the layer has to vanish —
  matches at 1.003, so the layering itself is right. From there the ratio falls
  to 0.94 by IOR 2.2: our coat takes energy out of the base for the way in and
  the way out, and never gives back what bounces between the coat's underside and
  the base. Cycles models that inter-reflection. The missing term is worth about
  6% at the strongest coat in the ramp.

Re-recorded after the sheen, subsurface, clearcoat-IOR, specular-colour,
thin-walled and iridescence work. Every row is at or better than the numbers it
replaces; `08_alpha_blend` moved the most, from 0.051 / 1.044, and that is *not*
attributed to any of it -- 32 commits touched shading between the original
recording and this one, two of which ("shadow rays take any hit" and "tabulate
256 Sobol dimensions") are the obvious candidates.

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

Curves/hair, displacement, shape keys and Sun & Sky are absent — either glTF
cannot carry them or Strelka cannot render them, so a comparison would only
restate what is already known.

Sheen, clearcoat, iridescence and bounded volumetrics are on the ladder as of
scenes 14 to 18. Subsurface
scattering, thin-film iridescence and bounded volumetrics are not, and are
covered only by unit tests in `tests/material/`, which pin the properties those
features exist for and the energy they are allowed to carry — a different
question from whether they agree with another renderer.

Subsurface is the one still missing, and the awkward one to add: Cycles' random
walk derives its scattering albedo from a diffuse colour through a fit, and
Strelka's extension carries the single-scattering albedo directly, so a scene
would have to invert that fit before the two could be compared at all.

Volume *emission* is not compared either, in the row that exists. Cycles adds it
with its own coefficient and Strelka adds it per free-flight event; the two
conventions do not line up, and `18_bounded_volume` sets emission to zero rather
than measure the mismatch as though it were an error.
