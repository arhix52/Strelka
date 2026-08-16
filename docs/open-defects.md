# Open defects

Each entry is something measured and left unfixed, with the measurement that
found it and what has already been ruled out. They are written so that a reader
starting cold can act without repeating the elimination.

Most of what used to live here was found against the Isometric Bathroom /
Kids Bedroom V-Ray conversions. Anything that needs Chaos's closed formula or a
controlled V-Ray render to close -- `adv_base`, rect-light `directional`,
per-light diffuse/specular weights, `TexMulti` with empty slots, `BRDFCarPaint2`
flakes -- is gone from this file. Fitting those to one frame is a guess wearing
the clothes of a conversion; without V-Ray there is nothing to measure against.
The converter still drops them, and that is fine.

What remains is either a renderer defect that Cycles (or a Strelka A/B) can
answer, or an asset/converter note that does not need Chaos.

## Order of attack

| # | What | Where it lives | How we know it is done |
|---|---|---|---|
| ~~1~~ | ~~Clearcoat underside bounce~~ | done — see Closed | `15_clearcoat` 1.002 overall, IOR 2.2 within 1% |
| ~~2~~ | ~~Rough thin-wall blur~~ | done — see Closed | `22_thin_walled` 0.094 / 0.966 CLOSE |
| ~~3~~ | ~~Hair: strand width, then a fibre traced as a surface~~ | done — see Closed | `28_hair` 0.033 / 1.012, under the reference's own seed noise (0.051); isolated strand flat over depth |
| ~~4~~ | ~~Height-map mip aliasing~~ | done — see Closed | kids walls lod0 vs footprint ≤1%; default stays 0 |
| ~~5~~ | ~~OptiX ior-stack counters~~ | done — see Closed | bathroom reports 15 / 6 / 693 per sample at 256² depth 16; shading untouched |
| 6 | Two-sided different back face | material model / glTF | design first; one kids-bedroom material only |
| 7 | Bath water is a dish, not a volume | asset, not code | remodel in Blender; bath water R/G against Chaos PNG is a check, not a driver |
| ~~8~~ | ~~OptiX accumulates in tonemapped space~~ | done — see Closed | `00_calibration` bit-identical at `spp_per_launch` 1 and 512 |
| ~~9~~ | ~~OptiX had no atmospheric scattering at all~~ | done — see Closed | `fog_check.py` both directions within 1% of Cycles at matched depth |
| 10 | A volume vertex costs a bounce Cycles does not charge | `OptixRender.cu` raygen loop / `wavefront.metal`; both backends | the fog probe reads 1.00 at `max_depth` 2, not 3, against a `max_bounces` 2 reference |
| 11 | The iso bathroom's firefly tail | estimator, and an asset the machine does not have | 256 spp noise near the ladder's rows rather than 4x them |

6 is smaller. 7 is not a renderer bug. 10 is a convention to settle, not a bug to
find: it is measured, it is the same on both backends, and picking a side changes
every volumetric render.

Build the bathroom / kids bedroom when an entry asks for it:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom \
    --rebuild-rug --no-fog-volumes

cd build/Release
./StrelkaCLI ../../scenes/iso_bathroom/iso_bathroom.gltf -o /tmp/iso.png \
    -w 1024 --height 1024 --spp 512 --depth 16 --camera 0 --clamp 8 --tonemap aces
```

Kids bedroom:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Kids_Bedroom/Iso_Kids_Room.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out /tmp/kids --name kids_room \
    --no-fog-volumes
```

Patch means (linear luminance over a rectangle):

```bash
python3 tools/iso_bathroom/patch_mean.py /tmp/kids_ref.png /tmp/kids.png \
    -- 920 380 30 30 right_wall  660 260 40 60 curtain
```

Feature-test ladder (the default verifier for 1–3):

```bash
# see tools/feature_tests/README.md -- Cycles EXR refs, one feature per scene
```

---

## 6. One two-sided material has a different back face

Eight of nine `Mtl2Sided` materials in the kids bedroom are translucency only,
which `KHR_materials_diffuse_transmission` carries. Exactly one has a Back
sub-material linked. The renderer has nowhere to put a genuinely different back
shader; it is reported rather than approximated.

Lampshade / curtain residuals after the translucency fix are dominated by lighting
differences that used to be filed under V-Ray `directional` (removed from this
file), not by this entry.

**Fix**: design a back-face material slot (or reject and document). **Verify**:
that one kids-bedroom material, once a slot exists.

---

## 7. The bath looks full because the stack is wrong

Not a renderer bug. The bath water mesh is a 2 mm dish; what makes it look full
is an unbalanced ior stack applying water absorption to every segment after an
entry through the open surface. Repair the mesh and the accident goes with it.
Needs a real water volume upstream in the .blend.

**Verify after remodel**: bath water R/G via `bubble_profile.py` / patch means;
nested-dielectric counters should drop on that mesh.

---

## 10. A volume vertex costs a bounce Cycles does not charge

Both backends charge a scattering event inside a medium a unit of `max_depth` --
Metal's comment says why ("a medium with no depth budget of its own is a path
that wanders forever") and the OptiX port follows it. Cycles, with the same
number in `max_bounces`, gets one more vertex out of the path.

Measured on `tools/feature_tests/fog_check.py`, which is a box of fog, a low sun
and a grey floor, rendered twice -- once looking into the sun and once away from
it, because an inverted phase function is too dim in the first and far too bright
in the second and either alone reads as a brightness mistake. Cycles reference at
`max_bounces = volume_bounces = 2`, Strelka at 512 spp:

| `max_depth` | into the sun | away from it |
|---|---|---|
| 2 | 0.925 | 0.694 |
| **3** | **1.009** | **1.001** |
| 4 | 1.036 | 1.169 |

Both directions land on the reference at 3, and the offset is exactly one vertex
rather than a weight: the same probe with the medium removed reads 0.973 at
`max_depth` 2, so with no volume in the path the two conventions already agree.

This is why it is an entry and not a fix: the depth budget is the contract
between the two backends and every scene's TOML, so moving it re-times and
re-grades every volumetric render in the tree. What is not in doubt is the
transport -- at matched depth the fog agrees with Cycles to 1% in both
directions, which is the thing the probe was built to answer.

**Decide**: whether `max_depth` counts vertices or scattering events, on both
backends at once. **Verify**: the table above collapses to one column.

---

## 11. The iso bathroom carries a firefly tail, and this machine cannot finish the comparison

Reported as "even Metal is cleaner by 256 samples". Measured on OptiX at 1024²,
depth 16, 256 spp, graded against a 4096-spp render of the same backend on the
pixels where it is locally flat (`tools/parity/noise_check.py`): relative noise
**0.175**, against 0.084 for the ladder's noisiest row at twice the samples. It
is a real complaint.

What it is not: one broken mechanism. Sweeping `clamp_indirect` moves it smoothly,
with no knee anywhere --

| `clamp_indirect` | off | 128 | 64 | 32 | 16 | 8 | 4 |
|---|---|---|---|---|---|---|---|
| noise | 0.175 | 0.145 | 0.134 | 0.119 | 0.103 | 0.087 | 0.074 |
| mean / converged | 1.000 | 0.998 | 0.994 | 0.984 | 0.965 | 0.939 | 0.904 |

-- so samples above 128 carry 0.2% of the frame's energy and 15% of its variance,
and it keeps going all the way down. That is a broad heavy tail, which is what a
room of twelve transmissive materials, twelve bounded volumes and nine subsurface
walks lit by three small rect lights produces. Ablations rule out the obvious
suspects individually: `subsurface_iterations = 0` makes it slightly worse,
`ris_candidates = 8` moves it 1%, `estimator_mode = 1` (no NEE) makes it 2.6x
worse.

One real cause was found and fixed -- the Russian roulette differed from Metal's
in two ways that both manufacture fireflies (see Closed). It was worth 0.184 ->
0.175, which is to say it is not the answer here.

**Why this entry cannot be closed on Linux.** The scene's environment is
`abandoned_hall_01_4k.exr` at a `/Users/ikryukov/...` path, and there is no copy
on this machine, so every measurement above was taken on a bathroom lit by three
rect lights and no environment -- which is not the render the report is about. A
stand-in environment at the same intensity puts the noise at 0.139. There is no
Metal here to compare against and no `.blend` to build a Cycles reference from.

**Verify, on a machine that has the assets**: the same `noise_check.py` split run
on both backends at 256 spp against a converged render of each, so the claim
becomes a number rather than a look. If OptiX is genuinely the noisier one, the
next thing to read is `connectToLight`'s solid-angle pdf against Metal's on a
grazing rect light, which is the one estimator this scene leans on hardest and
the ladder's single-light rows are too easy to expose.

---

## Closed (kept for the measurement, not the work)

### OptiX rendered the pine forest with no atmosphere at all

**Fixed.** `atmosphere` -- the sidecar block carrying a homogeneous haze, which
`oka::Scene` has loaded all along -- was read by `MetalFrameUniforms.mm` and by
nothing on the OptiX side. Not degraded, not approximated: `grep -rl atmosphere
src/` returned Metal, the scene container and the loader, and no CUDA at all. The
pine forest rendered with a hard, dark treeline where the reference has haze.

Ported from `src/shaders/metal/fog.h` as `src/shaders/optix/fog.h` -- a slab
rather than a bounded volume, for the reason that file gives -- with three
integration points: a free-flight draw against the segment in
`__closesthit__radiance`, the same draw in `__miss__ms` for the ray that was on
its way to the sky, and the haze's transmittance folded into `traceOcclusion` so
a shadow ray is not a hole in the fog. `__miss__ms` moved into the closest-hit
translation unit to get there: a scattering event needs next-event estimation, and
OptiX modules do not share device functions.

Validated three ways rather than by looking at it:

* `tools/feature_tests/fog_check.py`, the controlled probe -- 1.009 into the sun
  and 1.001 away from it against Cycles, at the matched depth entry 10 explains.
  Both directions matters: an inverted phase function passes one and fails the
  other, and at g = 0.8 the forward lobe is 730x the backward one.
* The pine forest against the Metal render of the same TOML, by horizontal band:
  OptiX / Metal is 0.999 in the sky and 1.006 in the treeline, which is where the
  haze does its work. Against the Cycles render of the original, OptiX reads 0.696
  and 0.683 in those bands and Metal 0.697 and 0.679 -- the two backends agree
  with each other far more tightly than either agrees with Cycles, which is the
  shape a correct port has.
* The original `.blend` in Blender 5.2, which is where the sidecar's numbers were
  checked rather than assumed. The fog box is 208.82 x 208.82 x 29.15 with a
  world-space ceiling at y = 14.576, and Cycles' Volume Scatter has no absorption
  -- its extinction is Density x Color and every event scatters. So density
  0.004 x colour 0.8 = the sidecar's 0.0032 with albedo 1, and `height` 14.576 is
  the box's ceiling and not, as it looks, half of its 29.15. `collect_atmosphere`
  in `export_scene.py` is right.

What the pine forest still does not agree with Cycles about is a whole-image
factor: 0.696 in the sky and 0.683 in the treeline, and Metal reads 0.697 and
0.679 there. Both backends, equally, so it is the environment bake or the exposure
block rather than the renderer. In the dark understory OptiX is the closer of the
two (0.815 against Metal's 0.611).

### Russian roulette differed from Metal's, in the direction that makes fireflies

**Fixed.** OptiX rolled `p = clamp(luminance(throughput), 0.05, 0.95)`; Metal
rolls `q = min(max_component(throughput), 1)`. Two differences, both paying in
variance:

* The 0.95 ceiling killed strong paths. A throughput at or above one -- most of
  what survives four bounces in a bright interior -- was killed 5% of the time
  anyway and the survivors scaled by 1/0.95. Unbiased, pure added variance, and
  charged again at every bounce: over the twelve remaining at depth 16, 46% of
  such paths die and the survivors come back carrying 1.85x.
* Luminance is the wrong norm for a coloured path. A throughput of (0, 0, 5) --
  what blue-tinted glass leaves -- has luminance 0.36, so it was killed 64% of
  the time and the survivors multiplied by 2.8 while carrying five units of blue.
  That is where the bathroom's *coloured* speckle came from.

Worth 0.184 -> 0.175 on the bathroom with the converged mean unmoved at 0.2169,
and nothing on the feature ladder, which has no path long enough to reach the
roulette with a coloured throughput. Kept from the old form: the 0.05 floor, which
Metal does not have, because without it a path whose largest channel is 1e-6
survives one time in a million carrying 1e6.

### "OptiX is noisier per sample than Metal" was a one-row film offset

**Fixed, and it was never noise.** The ladder's `rel` ran about 1.3x the recorded
column on every row at once, which reads as variance and was written up as
variance. It was the camera: Metal builds the film position as
`height - (y + jitter)` and OptiX flipped the integer before adding the jitter,
`(height - y) + jitter` -- the same band one row out, with row 0 sampling off the
film entirely. A one-pixel displacement puts a full-contrast residual on every
silhouette and nothing anywhere else.

What separates the two hypotheses is *where* the residual lives, and no per-scene
`rel` can say. `tools/parity/noise_check.py` grades against a converged render of
the same backend and splits the residual by a 5x5 box -- what a blur destroys is
per-pixel noise, what it keeps is bias -- and can restrict that to pixels where
the reference is locally flat. Restricted to the three quarters of the frame with
no edge under them, the two backends read **1.02x**; whole-frame, 2.36x. All of
the excess sat on the edges.

Fitting the sub-pixel shift between the two backends' renders then names it
outright: dy = +1.00 against Metal on every scene with an edge in it, dx = 0, and
Metal aligned with Cycles. With it fixed, 27 of 29 rows reproduce Metal's recorded
`rel` *and* `ratio` to three digits, and per-sample noise is 0.98x over the ladder.

`08_alpha_blend` was the one row that really was noise, at 1.57x, and a different
cause: the coverage draw came from a standalone radical inverse instead of the
Sobol table, so it was stratified across a pixel's samples but not *jointly* with
the pixel jitter, which is what a continuous coverage test needs. `07_alpha_clip`
never showed it because MASK resolves to 0 or 1. Full write-up, both fixes and the
three pass-bar rows that do not match the artefact they were taken from:
`tools/parity/ladder_optix_aligned.txt`.

### OptiX accumulated in tonemapped space, so the ladder measured the accumulator

**Fixed.** `accumulate()` folds the launch into the history linearly, in
radiance, weighted `m/(n+m)` -- byte for byte what `wavefrontResolve` does on
Metal. The verify condition below is met: `00_calibration` at 512 total samples
renders **bit-identical** at `spp_per_launch = 1` and at `spp_per_launch = 512`
(`rel` between the two images is 0.00000, both 0.0209 / 1.0098 against Cycles),
so the launch split no longer names two different measurements. The evidence
that found it is kept below.

`accumulate()` in `src/shaders/optix/OptixRender.cu` used to blend the new sample into the
history as `inverseTonemap(lerp(tonemap(prev), tonemap(new), a))`, and
`inverseTonemap` (`postprocessing/Utils.h`) is `c / (exposure - c * exposure)`,
which diverges as its argument approaches 1. Every launch pays that round trip, so
the bias compounds with the number of launches rather than the number of samples.

`RenderConfig::sppPerLaunch` defaults to 1 and **every ladder toml ships
`spp_per_launch = 1`**, which is the worst case: 512 spp is 512 round trips.
Measured on `00_calibration` -- a 0.18 grey sphere, tone curve off, exposure pinned
to exactly 1.0 -- at a fixed 512 total samples, varying only the split:

| `spp_per_launch` | ratio | rel |
|---|---|---|
| 1 | 0.9728 | 0.0286 |
| 8 | 1.0011 | 0.0462 |
| 64 | 1.0023 | 0.0265 |
| 512 (one launch, accumulator never runs) | **1.0023** | **0.0195** |

So the backend's light units and exposure are right: at one launch the row reads
0.0195 / 1.0023 against a recorded 0.021 / 1.010, i.e. slightly *better* than the
number Metal set. What the ladder was reporting was the accumulator.

This is worth stating because a first pass at this entry read the same evidence
the other way. The deficit is flat across brightness bands (0.952, 0.969, 0.961,
0.961, 0.967 by decile) and identical at 64 and 512 spp, and the photometric block
matches `MetalFrameUniforms` line for line -- which correctly rules out variance,
exposure and the BRDF, and looks exactly like a per-light-type scale, because
scenes of different brightness are compressed by different amounts. Splitting the
sample budget instead of the sample count is what separates the two hypotheses,
and nothing in the per-scene table does.

It also inflates how bad other defects look, and by a lot: the environment
auto-scale error reads 8.7x through this accumulator and 500x without it.

One thing the fix does not carry over: the 0.0195 / 1.0023 in the table above was
this backend beating the number Metal set, and it is not what the row reads now.
It was measured against a film position one row off, which biases the mean as well
as the edges -- `00_calibration` reads 0.021 / 1.010 today, which is Metal's number
exactly. See `tools/parity/ladder_optix_aligned.txt`.

---

### Nested dielectrics: both backends count losses now

`ior_stack_pop` matches on the material being left, on both backends, and
`tests/material/test_ior_stack.cpp` pins it. What was missing was the
measurement: Metal reported three counters -- pushes onto a full stack, pops
that matched nothing, and paths that reached the environment still inside a
medium -- and OptiX called push and pop without asking either question, so an
asset could lose paths on one backend and be silent on the other.

OptiX now raises the same three, into a three-word device buffer zeroed per
launch and read back once per scene at the frame's existing synchronisation
point. The report is Metal's `reportIorStackStats`, word for word. On the
unpatched bathroom at 256² / depth 16 it says 15 / 6 / 693 per sample; Metal at
1024² / depth 16 with the converter's holes capped says 0 / 7 / 2, and without
0 / 19 / 2. The third counter is the largest here because it is the one no exit
event can catch -- the ray left through the hole -- and because this render had
neither the capping nor the resolution the Metal numbers were taken at.

Nothing about shading moved: the counters are pure observation, guarded on the
buffer pointer being non-null, and the ladder is bit-identical either side of
them.

What still cannot be fixed in the renderer: a ray leaving an open mesh goes out
through a hole with no exit event. The converter caps flat, small loops
(`Brush_Fibers`); it correctly leaves foam and water open -- capping the bath
water is arithmetically the best patch mean and plainly wrong on screen, because
the water is a 2 mm dish. See entry 7.

### Height-map mip aliasing

**Measured; default stays off.** The conversion bug (height / mix masks forced
into `normalTexture`) was already fixed by `TextureBaker.bake_height_normal`.
What remained was whether mip-0 sampling of those baked maps still aliases
enough to justify flipping `render/pt/textureLod` / `textureLodMode` on by
default.

Kids bedroom, camera 0, 1280², ACES, clamp 8, sobol — lod 0 vs ray-footprint
LOD (`texture_lod = true`):

| patch | 256 spp mean Δ | 16 spp high-freq Δ |
|---|---|---|
| wall_left (300,300 60×60) | −1% | −0.6% |
| wall_right (820,300 60×60) | −1% | −0.7% |
| right_wall (1150,475 40×40) | 0% | −0.1% |

Plaster normals after the bake sit at scale ~0.001, so there is almost nothing
left to alias. The wall mean gap vs the Chaos PNG is lighting / conversion, not
mip selection — footprint LOD does not close it. The shader comment already
says the right thing: at high spp supersampling hides minification, and an
offline render should leave mip 0 alone so the estimate converges to the
unfiltered texture. Interactive 1 spp can opt in via TOML / settings.

### Hair Chiang lobe

**Fixed.** Curves used to shade as a rough dielectric cylinder at IOR 1.55 --
silhouette right, lobe dark. `MATERIAL_TYPE_HAIR` now runs Chiang et al. 2016
(R / TT / TRT / TRRT+) on the curve shade path, ported from Cycles' Principled
Hair. `STRELKA_materials_hair` marks the material; pigment colour is Direct
Coloring reflectance, roughness is longitudinal, `radialRoughness` / `coat` ride
in the extension. Non-hair scenes stay on the triangle kernels
(`kFeatureCurves`).

### Hair strands were exported twice too thick

**Fixed, and it was not the lobe.** `28_hair` first read `0.084 / 0.977 CLOSE`,
but that is the *whole frame* of a two-subject scene: the bald control measured
`0.029 / 1.007` and the hair sphere on its own `0.138 / 0.948`, past the
ladder's own `TOL_CLOSE = 0.10`. The control that stops a framing shift from
looking like a lobe win is also what diluted the lobe's error under the cutoff.
Reading a scene with a control subject by its mean is the trap here.

The cause was geometry. Blender's particle properties `root_radius` /
`tip_radius` are *diameters* -- the UI labels them "Diameter Root" / "Diameter
Tip" -- and Cycles halves them when it builds the curve. Both sidecar exporters
passed them through as radii, so every strand rendered at twice its reference
thickness. That is not only a silhouette error: it doubles the chord a ray
crosses inside a strand, so the Chiang lobe absorbed over twice the path and the
groom shifted toward the pigment's dominant channel. Halving them on the hair
half:

| variant | rel | ratio | R | G | B |
|---|---|---|---|---|---|
| radius as exported | 0.138 | 0.967 | 1.003 | 0.962 | 0.932 |
| radius halved | 0.112 | 0.977 | 0.987 | 0.974 | 0.967 |

The R↔B spread collapses from 0.071 to 0.020, and against a converged reference
it is 0.008 -- the "warm tilt" was the radius, not absorption. Chasing it in
`sigma_from_reflectance` would have been a fit to a geometry bug; the Fresnel
convention, the absorption fit and the roughness→variance remap were each
checked against Cycles and each match. Fixed in `curve_sidecar.hair_radii`,
which `build_features.py` and `vray2strelka.py` both call.

The conversion now has a test on each end of it, because a bug that presents as
shading and lives in geometry is one nobody should have to find twice. The unit
suite cannot reach the exporter -- it is Python that Blender runs -- so
`tools/iso_bathroom/test_curve_sidecar.py` covers `hair_radii` and parses the
written bytes back independently of the writer, and `ctest` runs it beside the
C++ binary as `curve_sidecar_writer`. `tests/sceneloader/test_curve_sidecar.cpp`
takes the reader: that radii pass through unscaled and a taper stays monotonic,
both bases, `mSegmentsPerStrand` for uniform and mixed sets, material binding and
its fallback, the instance transform, multiple sets' payload offsets, the
`<stem>_curves.bin` lookup, and the five malformed files that must be refused
rather than handed to a driver.

The second half was the reference. Cycles at 256 spp carries `rel 0.074` of its
own variance on the hair sphere, measured against the same scene at 2048 --
most of what the row was reporting as disagreement. A groom is the noisiest
subject in the ladder, so `SCENE_SAMPLES` converges this one reference; it costs
a minute.

With both: hair half `0.098 / 0.970`, bald control `0.023 / 1.007`, whole frame
`0.061 / 0.988`. That still looked wrong on screen, and it was.

### The groom's scalp was shaded with the hair lobe

**Fixed, and it was not the lobe either.** With the radii right, the row still
disagreed in a way a mean hides: measured in rings about the projected scalp
centre, the scalp disc read `0.809` of the reference while the shell of strands
read `1.066` -- too dark in the middle, too bright outside, which is a
redistribution and not a level. On screen it was a bright sharp-edged lens under
the ball, and the floor's darkest row sat 23 px lower than the reference's while
the bald control's shadow landed within a pixel of it.

The scene shaded the fur ball's *emitter mesh* with `hair0`, against its own
docstring ("a short Chiang groom on a grey scalp"). Chiang's model is
parameterised on a cylinder: a longitudinal angle along a tangent and an azimuth
around a circular cross-section. A triangle sphere has neither, so each renderer
invents a tangent frame and the row compares the inventions. Giving the scalp the
grey dielectric the bald control already wears:

| | scalp disc | strand shell | hair half rel | hair half ratio |
|---|---|---|---|---|
| hair lobe on the scalp mesh | 0.809 | 1.066 | 0.098 | 0.969 |
| grey scalp | 0.960 | 1.049 | 0.046 | 1.006 |

At that stage the old two-subject framing read `0.028 / 1.007`, which moved it
from the ladder's worst entry into the middle; the lens and displaced shadow
were both gone. The later close-up framing is reported below.

Two things this turned up on the way. The exporter drops materials no triangle
references, and the curve sidecar binds by name, so the first attempt at this fix
silently rendered the strands as stage grey behind one warning -- `patch_hair`
now creates the material rather than relying on a mesh to keep it alive. And
`STRELKA_SCENE_SPP` converges our side too: Strelka's own variance at 512 spp was
`rel 0.019` of the 0.046, and at 2048 the mean ratio does not move at all.

### The strand profile, and what the tips were actually worth

`use_close_tip` was the next thing found and it was real but small. Blender
defaults it on, and Cycles then forces the *last* control point to radius zero so
a strand ends in a point; both sidecar exporters wrote the tip radius there, so
our strands ended blunt. Measured on one strand of constant radius against an
orthographic camera, the taper occupies exactly the last segment and removes 4% of
a uniform strand's silhouette -- about 3% of this groom's projected area, all of
it in the outer ring.

Fixing it moved the outer ring from `1.049` to `1.041` and the row from `0.0280`
to `0.0272`. Right direction, right place, an eighth of the size needed. Worth
recording as a negative result: the shell excess is not the tips.

The same pass replaced the exporters' two-radius interpolation with
`hair_strand_radii`, which computes what Cycles renders along the whole strand --
diameters, `shape`, and the closed tip together. `shape` was measured too rather
than read out of Cycles' source: at -0.5, 0 and +0.5 the profile
`r(t) = (1-t)**p * (root - tip) + tip` holds to under a percent, with
`p = 1 + shape` below zero and `1 / (1 - shape)` above. Both exporters ignored
`shape` entirely before this, which was silently correct only because every groom
in the tree leaves it at 0.

Subdivision was tested next and was not it. Cycles' default
`cycles_curves.subdivisions = 2` versus 0 moves the inner ring by 1% and the
outer ring by 0.1%; Strelka's shell remains about 4% bright either way. The
feature scene now sets 0 explicitly because the sidecar carries linear curves,
but matching the feature does not close the residual.

### `28_hair` now compares the same primitive, close up

The bald sphere is gone. Its job was exposure / framing control, which
`00_calibration` already owns; here it occupied half the frame and twice hid a
hair-only error inside a passing whole-frame mean. A scene-specific camera moved
from 4.6 to 3.0 units and centres the groom, putting the tips about 155 px from
the image centre.

Cycles is explicitly `THICK` and unsubdivided, matching Strelka's round linear
curves. Its defaults are camera-facing `RIBBONS` subdivided twice. The old
RIBBONS/THICK probe changed the mean only 0.1%, but that is not a reason to call
two different primitives the same feature. An isolated geometry probe settles
the remaining visual suspicion: Blender diameter 0.04 in a 1.5-unit orthographic
frame predicts 13.65 px, and both renderers cover 14 px. Strelka's curve is not
twice as thick.

It does look heavier because its contrast is higher. In the close-up outer ring,
moderate-threshold occupied pixels are 1.11x the reference and mean radiance is
1.045x; at progressively high thresholds the coverage ratio rises because the
bright strand tail is stronger, not because the geometric support widens. With
the control removed, the row is `0.042 / 1.001`; rings read scalp 0.974, inner
hair 1.051, outer hair 1.045, and beyond tips 0.992. So it was a shading residual
on matched geometry, not a sidecar width convention -- and the entry below is what
it turned out to be.

An earlier version of this entry blamed the residual on bounce accounting, saying
Cycles caps per category while we cap total path length. That was wrong and the
measurement says so: `build_features.py` sets every Cycles category to
`MAX_DEPTH`, and Strelka's scalp ratio saturates at depth 16 (`0.844`, unchanged
at 32 and 64), so depth was never what was holding it.

**The kids bedroom needs re-exporting**: its monster and spider grooms were built
at double thickness too, on top of wanting the `STRELKA_materials_hair`
extension the converter now writes.

### A fibre was being traced as if it were a surface

Two defects, in opposite directions, in the path tracer rather than in the lobe or
the sidecar. Together they are the whole of the residual above, and the reason it
survived so many passes is that they cancel: every aggregate metric on the groom
read close while both were live. `28_hair` sat at `0.042 / 1.001` -- a mean within
a tenth of a percent -- over a shell 4.5% bright and a scalp 2.6% dark.

What separated them is `tools/feature_tests/strand_probe.py`: one strand, black
background, no floor, same key light, same sidecar path. Read the integrated
cross-section as a function of `max_depth` (`strand_measure.py`):

| max_depth | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| before | 0.538 | 0.550 | 1.001 | 1.113 | **1.293** |
| after | 0.943 | 0.997 | 0.997 | 0.997 | 0.997 |

One convex fibre in an otherwise empty room has nowhere to send light that could
come back, so the row *must* go flat after the second bounce. The old one never
stopped climbing, and the whole light-facing half of the cross-section was exactly
zero until depth 3.

**Direct light could not reach the far side of a strand.** `connectLight` and
`connectEnvLight` tested `dot(shading_normal, L) > 0` and folded
`saturate(dot(shading_normal, L))` into the radiance, and `wavefrontShade` rejected
any connection whose hemisphere disagreed with `front_face`. Those are surface
tests. Chiang's TT and TRT terms describe light that entered one side of the fibre
and left the other, and TT alone is about four fifths of a bright strand's albedo:
integrating the lobe over the sphere gives R 0.047, **TT 0.799**, TRT 0.033,
TRRT+ 0.001. So next-event estimation was discarding the dominant lobe, which then
had to be found by chance -- 46% of an isolated strand's light was still missing at
two bounces, and what did arrive came as noise. Even with the hemisphere opened,
the shadow ray had to start beyond the strand, or the fibre occludes itself.

**Transmitted bounces re-entered the strand they had just left.** The lobe is a
whole-fibre model: its `T` factor is `exp(-sigma * chord)` over the path *inside*
the strand, so a direction leaving the far side has already paid for the crossing.
The transmission branch offset the next ray along `-faceNg`, into the fibre, and it
hit the far wall and bought a second whole-fibre event -- and a third, and a fourth,
which is the 1.00 -> 1.29 climb above. The same branch also pushed the strand onto
the IOR stack as a medium it never exited.

The fix keeps the pbrt convention that `hair_chiang_eval` divides by `|n.wi|` and
the renderer multiplies it back, so it has to be the same `|n.wi|` on both sides and
never a clamp to zero: `shadingCosine` and `lightReachesShadingPoint` in
`shading_common.h` (both identities for every non-hair material, which is why all 28
other ladder rows re-render unchanged to four digits), and `fibreExitOrigin` in
`wavefront.metal`, which walks the chord `-2r(n.u)` across the fibre and starts the
ray at the exit. The radius it needs was already in hand: `fetchCurve` computes the
distance from the hit to the axis to build the radial normal.

What it bought, on the groom: rings scalp `0.974 -> 0.989`, inner hair
`1.051 -> 1.010`, outer hair `1.045 -> 1.013`, beyond tips `0.992 -> 1.017`. The row
goes `0.042 / 1.001` to `0.033 / 1.012`. High-frequency energy in the shell -- what
made the strands look heavy -- was 1.85x the reference and is now 1.04x, so the
appearance the eye was objecting to is gone. On the isolated strand the lobe now
agrees with Cycles to 0.3% integrated, with the profile centroid inside 0.02 px and
the same rms spread to three digits.

`rel` is finished as a measurement here, in both directions. Cycles against itself
at another seed on this scene is **0.0506**, above Strelka's 0.0332: thousands of
sub-pixel strands at 2048 samples, where a half-pixel disagreement per strand costs
more than any shading term. And the remaining `1.012` mean is the ladder's own
baseline -- `00_calibration` reads `1.010`. To go further this row needs a converged
reference and a metric that tolerates sub-pixel placement, not another fix.

The property the material side of this depended on is now asserted:
`tests/material/test_hair_chiang.cpp` checks that most of the lobe's sampled energy
leaves below the shading normal and that `eval` returns a positive pdf and a nonzero
value for every direction on the far side. That fails if anyone clamps the hair lobe
to a hemisphere again. The transport side cannot be a unit test -- the depth ladder
above is its instrument.

### Rough thin-wall blur

**Fixed.** Smooth thin walls stay a delta at `-V`. Rough ones are a GGX
reflection of the view mirrored through the surface, with Kulla–Conty
transmission roughness (`α √(3.4 (η-1)(η-0.5)²/η³)`), matching Cycles /
OpenPBR. Fresnel for the reflect/transmit split is taken at the shading
normal once, the same way Cycles bakes weights into its two closures; a
microfacet F made the coin flip track the reflection distribution instead.

`scenes/feature_tests/22_thin_walled` is now a roughness ramp (0 → 0.45) plus a
solid control, with a striped card behind so the blur is visible. Measured
0.094 / 0.966 CLOSE. Above ~0.45 Cycles' multiscatter GGX and our single-
scatter disagree on energy more than on blur, so the ramp stops there. Smooth
(roughness 0) is unchanged -- soap bubbles stay bit-identical.

### Clearcoat underside bounce

**Fixed.** `clearcoat_base_scale` now returns the single-scatter
`(1-F_L)(1-F_V)` times the per-channel geometric series against
`½(F_L+F_V)`, capped at `1-F_ms` so a base whose albedo we have under-counted
(specular sits under the coat too) cannot climb past one.

The two earlier attempts failed for named reasons: the *internal* hemispherical
average (~0.6 with TIR) is the wrong Fresnel for a model that never refracts L
and V into the coat -- it put a white ceramic at 2.43 directional albedo; the
*external* hemispherical average is larger than F at normal incidence, so
dividing `(1-F0)²` by `(1-F_avg ρ)` put the same ceramic at 1.02. Using the
directional pair keeps the identity `F + (1-F)² ρ / (1-F ρ) ≤ 1` where
`F_L ≈ F_V`.

`scenes/feature_tests/15_clearcoat`: overall ratio 0.995 → 1.002; the IOR 2.2
end of the ramp, which was the 6% hole, is within 1% of Cycles. Grazing
directions where an *uncoated* dielectric already exceeds 1 (additive
diffuse + specular floor) are excluded from the absolute bound in
`test_clearcoat.cpp`; the coat is required not to make that overshoot worse.

### RIS vs NEE

**Does not reproduce.** An earlier cross-estimator disagreement (NEE 4096 vs RIS
4096 at ~0.049) re-measured at 0.0126 -- five times smaller than the noise either
estimator has left at 1024 spp. Likely closed by the thin-wall `eval` fix
(`93421cf`). `19_env_and_light` and a bathroom lighting ablation both exonerate
the env/analytic split. RIS costs ~40% more time per sample; default stays at 1
for that reason, not because the answer moves.

### Denoiser

**Fixed.** Headless `denoise = true` now writes the denoised texture. Floor at
~0.105 rel RMSE on the bathroom; past ~200 spp plain accumulation wins.
`guide_primary_hit` stays a switch (bathroom +6% at 16 spp, mirror scene −110%).
`20_mirror_and_floor` is the rung. `denoise_firefly_clamp` is a config key.
