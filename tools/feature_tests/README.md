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

Hair has a second, narrower instrument beside the ladder, because a groom hides its
own errors -- see `28_hair` below for why:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/strand_probe.py
cd build/Release && for d in 1 2 3 4 8; do ./StrelkaCLI -c /tmp/strand_probe/strand_d$d.toml; done
cd ../.. && /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/strand_measure.py
```

Step 2 needs `StrelkaCLI` in `build/Release`, which a default build produces.
(This used to say the binary lived on another branch; it does not any more.)

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
| `15_clearcoat` | `KHR_materials_clearcoat` + IOR ramp | 0.037 / 1.002 |
| `16_iridescence` | `KHR_materials_iridescence` thickness ramp | 0.023 / 1.010 |
| `17_coated_glass` | transmission + clearcoat together | 0.067 / 0.994 |
| `18_bounded_volume` | `STRELKA_materials_medium` | 0.027 / 1.000 |
| `19_env_and_light` | an environment map *and* an area light | 0.026 / 1.000 |
| `20_mirror_and_floor` | a mirror filling the frame (denoiser guides) | 0.034 / 1.038 |
| `21_specular_color` | `KHR_materials_specular` tint ramp | 0.030 / 1.018 |
| `22_thin_walled` | Thin Wall roughness ramp (+ solid) | 0.094 / 0.966 |
| `23_diffuse_transmission` | `KHR_materials_diffuse_transmission` weight ramp | 0.012 / 1.000 |
| `24_orthographic` | ortho twin of `00_calibration` | 0.024 / 1.009 |
| `25_subsurface` | `STRELKA_materials_subsurface` (Van de Hulst recipe) | 0.056 / 1.009 |
| `26_dof` | thin-lens depth of field (`_camera.json`) | 0.024 / 1.015 |
| `27_ies` | IES point light via light sidecar | 0.047 / 1.032 |
| `28_hair` | close-up round linear Chiang groom (`STRELKA_materials_hair`) | 0.033 / 1.012 |

`19_env_and_light` is the only row with two kinds of light in it, and it is
there for one question: whether resampled importance sampling and plain
next-event estimation agree once `connectToLight` has to split its draw between
an environment and an analytic light. Every other row has one kind, where
resampling among candidates drawn from a single light is arithmetically a no-op.

The sky is the physical sky model with the sun disc turned *off*. Off because a
disc is a near-delta source inside an environment map: it converges slowly on
both sides and would make the row measure variance rather than bias. What is left
still spans an order of magnitude across the sphere, which is what environment
importance sampling is for.

The answer, at 512 spp: `ris_candidates = 1` gives 0.026 / 1.000 and
`ris_candidates = 8` gives 0.023 / 1.000, and the two Strelka images differ from
each other by 0.014 -- less than either differs from Cycles. See entry 6 of
docs/open-defects.md, which this row was built to settle and did.

`20_mirror_and_floor` is not a shading row -- every lobe in it is covered by 03
and 04 -- it is there so the denoiser's guide source can be measured on the case
it was written for. `render.guide_primary_hit` hands the denoiser the mirror's
own albedo, which is nothing; the walk hands it the world being reflected. On
this row the walk is 2.1x better at every sample count, which is what keeps it
the default. See entry 7 of docs/open-defects.md.

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
- **`15_clearcoat` is an agreement check.** The IOR 1.0 sphere — where the coat's
  F0 is zero and the layer has to vanish — matches at ~1.01, and the strong end
  of the ramp (IOR 2.2) is within 1% of Cycles. The underside geometric series in
  `clearcoat_base_scale` is what closed the previous 6% hole there; see
  `docs/open-defects.md` (Closed).

`25_subsurface` was recorded at 0.330 / 1.053 and read as a convention mismatch:
the Van de Hulst inversion is the right encoding for our extension but is not
Cycles' BaseColor→medium map, so the residual was written off as that. It was
not. The row is at 0.056 / 1.009 now, and getting there was four bugs, none of
which any other scene could see because no other scene runs a random walk.

- The walk never ran. `initSurfaceInteraction` copies the material into
  `MaterialParams` field by field, and `subsurface`, `subsurface_radius` and
  `subsurface_anisotropy` were not among them, so `si.subsurface` — the flag that
  gates the walk in `shade` — was whatever the stack held. Every subsurface
  material rendered as plain diffuse transmission, and the giveaway was that the
  mean free path did not change the image at all. That struct is now `= {}`, so
  the next field added to it reads as zero rather than as garbage.
- The albedo was applied twice. The diffuse-transmission lobe tints the ray on
  the way in, and the walk applies the medium's colour again at its first
  scattering event. A deep-red sphere came back at about a third of the light it
  should return. The entry tint is divided back out at the point the lobe is
  taken, which is what Cycles does at the same place.
- Channels were chosen uniformly. In a medium whose extinction differs threefold
  between channels the balance-heuristic weight can exceed one for whichever
  channel suited the sampled distance, and over a walk tens of steps long those
  compound into fireflies. The choice is now proportional to throughput times
  albedo, and the same distribution is passed to the weights — sampling from one
  density and weighting by another is how an unbiased estimator stops being one.
- Exit connections were dark twice over. The shadow ray leaving the medium was
  tagged with the medium it was leaving, so a dense extinction attenuated the
  whole distance to the light and nothing ever cancelled it; and the estimate was
  multiplied by the exit lobe's own density, `cos/pi`, when `connectLight` had
  already folded the cosine into the radiance it returns. The second one cost a
  factor of the cosine on the connection while MIS deducted the whole of it from
  the bounce ray, which is the 25–35% deficit that outlasted the other three.

What separates that from four guesses is `tools/feature_tests/sss_furnace.py`,
which builds the sphere alone under a uniform sky of radiance 1 and reads it back
with `sss_furnace_read.py`. A medium of single-scattering albedo 1 absorbs
nothing, so the sphere has to render as exactly 1 whatever its density — and it
does, 1.0004 at every mean free path from 0.06 to 1.0. Energy conservation is
therefore not what the remaining 5.6% is. The rest is the Van de Hulst fit
itself: driven at raw albedos, a thick sphere reads 6–8% above what the fit
predicts, because the fit describes a plane-parallel half-space and the test
subject is curved. That bias is the row's residual, and it is in the recipe
rather than in the walk.

`sss_probe.py` reads the row per sphere rather than per frame, and splits each
silhouette into its lit and shadowed halves — a walk that carries light the wrong
distance moves it between the halves while the total holds, which is a different
fault from losing it, and the whole-frame ratio cannot tell them apart.

`render.subsurface_iterations` defaults to 64 for the same reason. It is the point where the row
stops moving: 64 renders the same as 256 to within the comparison's noise and in
half the time, while 16 truncates enough of the walk's tail to lose about 2%.

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

Displacement and shape keys are absent — either glTF cannot carry them or
Strelka cannot render them, so a comparison would only restate what is already
known.

Sun & Sky is inside `19_env_and_light` as an environment map rather than as a
procedural sky: it is baked to an equirectangular EXR, which is the only form
Strelka takes.

Sheen, clearcoat, iridescence, bounded volumetrics, specular tint, thin-walled
glass, diffuse transmission, orthographic framing, subsurface, depth of field,
IES lights and Chiang hair are on the ladder as of scenes 14 to 28.

`21_specular_color` pins Specular IOR Level at 0.5 so Blender's exporter writes
`specularColorFactor` equal to the tint; level 1.0 would bake a factor of two
into the colour and the row would measure that encoding.

`22_thin_walled` is a thin roughness ramp at IOR 1.5 (0 → 0.45) plus a solid
control of the same IOR. A low-frequency striped card behind the row gives the
frosted spheres structure to blur. Blender's Thin Wall flag is not exported, so
the patcher writes `KHR_materials_volume.thicknessFactor = 0`, which is what
Strelka reads as a wall. Rough transmission follows Cycles / OpenPBR: a GGX
reflection of the view mirrored through the surface, with Kulla–Conty roughness
for the two interfaces. The ramp stops at 0.45 because above that Cycles'
multiscatter GGX and our single-scatter disagree on energy more than on blur.

`23_diffuse_transmission` builds a Mix(Principled, Translucent) for Cycles —
Principled 5.2 has no Diffuse Transmission socket — and the patcher writes
`KHR_materials_diffuse_transmission` with the same weights. A backlit panel is
required; front lighting alone looks like a darker diffuse.

`24_orthographic` is `00_calibration` under an orthographic camera whose vertical
extent matches the perspective framing. Drift here with a clean `00` means the
projection path, not light units.

`25_subsurface` is the albedo-convention bridge the ladder used to lack. Cycles
authors a diffuse subsurface colour on Base Color; `STRELKA_materials_subsurface`
carries the single-scattering albedo. The patcher inverts Van de Hulst's
approximation per channel (the same fit `tools/iso_bathroom/vray2strelka.py`
uses) and keeps the mean free path identical on both sides. Semi-infinite and
isotropic are both approximations — these spheres are neither — which is where
the last few per cent go; see the walk-through above for the four bugs the row
found on the way from 0.330 to 0.056, and for the furnace test that says the
remainder is the fit rather than the walk.

`26_dof` turns the thin lens on. glTF has no DOF, so the builder writes
`26_dof_camera.json` with the focus distance of the shared camera target, f/2.0,
and the 29 mm lens that a 45° vertical FOV on a 24 mm sensor implies. Cycles gets
the same numbers on `cam.dof`. Three grey spheres at different depths: the middle
one is sharp, the near and far ones measure the blur. The TOML still only carries
pose and FOV — DOF lives entirely in the camera sidecar.

`27_ies` is a point light whose angular distribution comes from a real LM-63
file: a Philips CDM-R111 35W/830 24° reflector, in
`tools/feature_tests/assets/`. Cycles samples it through a TexIES→Emission
chain; Strelka through the light sidecar's `ies` path. The GPU divides the
candela table by 177.83 lm/W, the D65 efficacy Cycles assumes for the same
conversion, so the row compares the angular distribution instead of two guesses
at a scale. No rect key: the IES light is the whole of the lighting.

It used to be a synthetic cosine^4 hotspot, and that made the row unable to
fail. A cos^4 point light and a bare point light differ by a smooth vignette, so
every way of mis-reading the table — dropping it, transposing the candela block,
folding a symmetric azimuth by repeating instead of mirroring, extrapolating
past the last tabulated angle — still rendered as a slightly vignetted point
light. A 24° beam does not: it paints a hard-edged pool with a dark surround,
and the parser has to get 19 vertical angles by 24 horizontal planes, values
wrapped eight to a line, and a `[LUMCAT]` line carrying a latin-1 degree sign,
all right to produce it. Recovering the beam angle from the loaded table gives
23.6°, which is the "24" in the lamp's own name — a check on the read that does
not depend on the render at all.

The lamp's `energy` is set explicitly to 1 W here, and that line is load-bearing.
Cycles renders the product of energy and whatever Emission strength the node
graph yields, so a new lamp's default 10 W multiplies a reference that otherwise
looks entirely reasonable. Left unset, it pushed this row to a ratio of 18, and
the factor of ten hid comfortably inside a fitted constant (π²/177.83, within
0.8% of the truth and derivable-looking) that made the row pass while leaving
every IES scene outside the suite ten times too bright. The honest constant
lands the row at 1.021 instead of 1.008; the remainder is Cycles' own
normalisation and interpolation of the table, and is worth more than a match.

That residue used to be two things, and the larger one is gone. Compared
radially against the reference, the row drifted from 0% on axis to 9% at 22° --
the steep flank of the beam, between the table's 20° and 25° rows, where the
beam falls 2.4x in one step. Strelka drew straight lines between the samples and
Cycles drew a Catmull-Rom through four of them, which reads 12.7% apart at the
midpoint of that interval. Strelka now draws the same cubic
(`common/ies_math.h`), and the radial curve is flat: 1.02-1.04 across the whole
floor instead of a hump.

What is left is a ~4% scale offset with no angular structure, and it predates
the interpolation work -- the old synthetic cos⁴ profile, where interpolation
barely mattered, sat at 1.028. It is not the table: on axis the direction lands
exactly on a tabulated angle, so both renderers read the same 3418.9 cd and
still differ. It sits with the ratios the rest of the ladder carries
(`06_normalmap` 1.061, `20_mirror_and_floor` 1.038) and has not been traced.

`28_hair` is one close-up particle groom on a grey scalp. There is deliberately
no bald control anymore: `00_calibration` already checks exposure and framing,
while a second sphere took half this row's pixels, made individual strands hard
to inspect, and once diluted a hair-only `rel 0.138` into a whole-frame `0.084`
that passed. The scene-specific camera puts the hair tips at roughly 155 px from
the image centre, so the groom occupies about 60% of the frame width.

The two renderers now receive the same geometry features explicitly. Cycles is
set to `cycles_curves.shape = "THICK"` and `subdivisions = 0`, matching the round
linear curves in the sidecar; its defaults are camera-facing `RIBBONS` subdivided
twice. `shape = 0` and `use_close_tip = true` are explicit too. Ribbons versus
thick happened to move this groom's mean by only 0.1%, and subdivisions 0 versus
2 by under 1% in every ring, but close aggregate numbers are not permission to
compare different primitives.

The radius convention is measured, not inferred. An isolated strand authored
with Blender diameter 0.04, viewed with a 1.5-unit orthographic frame at 512 px,
has a predicted diameter of 13.65 px; both Cycles and Strelka cover 14 px. That
rules out the visual impression that Strelka's curves are twice as thick.

**Do not debug hair on this scene.** A groom is optically dense, so most of what
leaves it has scattered off several strands, and two opposite errors in there cancel
in every number this harness prints -- for several passes the row read `0.042 / 1.001`
while its shell was 4.5% bright against a scalp 2.6% dark, and the mean said nothing.
Use `strand_probe.py` / `strand_measure.py` instead: one strand, no floor, same key
light and same sidecar path, so the lobe is measured with the transport removed. Read
two things off it -- the integrated cross-section against Cycles, which should agree
to about a percent, and that same number against `max_depth`, which *must* go flat
after the second bounce because a single convex fibre in an empty room has nothing to
scatter off twice. A row that keeps climbing means transmitted rays are re-entering
the strand they just left. That is exactly how the defect in entry 3 of
`docs/open-defects.md` was cornered, after the groom's aggregates had hidden it
through half a dozen hypotheses.

`curve_sidecar.hair_strand_radii` carries Blender's whole profile: its
`root_radius` / `tip_radius` properties are diameters, `shape` bends the
interpolation, and `use_close_tip` zeroes the last control point. The latter took
the outer ring from 1.049 to 1.041; it was real but not the remaining cause.
Those behaviours are pinned by the stdlib Python tests using values measured from
single-strand Cycles renders.

The scalp must stay on `scalp`, not `hair0`. Chiang's lobe is parameterised on a
cylinder's tangent and azimuth; putting it on the triangle emitter sphere made
each renderer invent a frame and was worth more than half the old disagreement.
The strands take `s.material` as a 1-based slot index, and `patch_hair` must
create `hair0` in the glTF because Blender drops materials no triangle
references. Without that, curves fall back to material 0 behind one warning.

Both sides use 2048 spp for this row. It reads `0.033 / 1.012`; in geometric rings
Strelka/reference is 0.989 on the scalp, 1.010 on the inner hair shell, 1.013 on the
outer shell, and 1.017 beyond the tips. Read those rings and not the whole-frame
mean: the mean is what hid a shell/scalp error of opposite signs for so long.

`rel` is spent on this row, in both directions. Cycles against itself at a second
seed scores **0.0506**, worse than Strelka's 0.0332 -- thousands of sub-pixel strands
at 2048 samples, where half a pixel of disagreement per strand costs more than any
shading term. The residual `1.012` mean is the ladder's own baseline, since
`00_calibration` reads `1.010`. Going further needs a converged reference and a metric
that tolerates sub-pixel placement, not another shading change.

Volume *emission* is not compared in `18_bounded_volume`. Cycles adds it with
its own coefficient and Strelka adds it per free-flight event; the two
conventions do not line up, and that row sets emission to zero rather than
measure the mismatch as though it were an error.
