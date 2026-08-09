# Open defects

Each entry is something measured and left unfixed, with the measurement that
found it and what has already been ruled out. They are written so that a reader
starting cold can act without repeating the elimination.

Everything below reproduces from the Isometric Bathroom conversion. Build the
scene once:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom \
    --rebuild-rug --no-fog-volumes

cd build/Release
./StrelkaCLI ../../scenes/iso_bathroom/iso_bathroom.gltf -o /tmp/iso.png \
    -w 1024 --height 1024 --spp 512 --depth 16 --camera 0 --clamp 8 --tonemap aces
```

The reference render is
<https://documentation.chaos.com/download/attachments/117637916/Iso_Bathroom_GPU.png>.
Crop the same region from both with `sips`, which does not touch the bytes:

```bash
sips -c 150 190 --cropOffset 330 300 /tmp/iso.png --out /tmp/crop.png
```

Entry 1 needs the export *with* its fog volumes, so drop `--no-fog-volumes` and
send it somewhere else -- the checked-in scene is the one without.

`tools/iso_bathroom/bubble_profile.py` reads back a rendered PNG as mean
luminance per radial bin across one of the bubbles, against the wall just
outside it. A defect with a radius is far easier to name than one with only a
colour.

---

## 1. The transmissive meshes in this scene are open, and that is what colours the water

**Symptom.** With the bathtub's `EnvironmentFog` gizmo present, the bath water
loses its cyan everywhere except a ring at the tub's rim. Measured as the
red-to-green ratio over the water, against the reference's 0.871:

| Configuration | R/G |
|---|---|
| gizmo present | 0.968 |
| gizmo present, density 1e-6 and no emission | 0.967 |
| gizmo removed from the same export | 0.777 |

**What it is.** `Water_Bathtub` is an open mesh: 3712 triangles carrying 288
boundary edges. `Bubbles_Mtl`'s mesh, which is the foam in the tub, carries 672
across 9454. A ray entering an open transmissive mesh pushes the IOR stack and
never finds the exit that would pop it, so every segment it travels afterwards --
anywhere in the room -- is attenuated as though it were still inside the water.

That is where the colour comes from. The water is a 2 mm slab (world Y 0.14487
to 0.14687) with an attenuation distance of 0.1, so its own thickness is worth
1.2% of the red channel; it renders at 0.777 against a wall at 1.0. The number
the gizmo is being measured against was never the water's absorption.

It also explains the direction of the error. Removing the gizmo gives 0.777 and
the reference is 0.871, so the case treated as correct is the one that
*over*-absorbs.

### Ruled out, with the test that ruled it out

Rendered from one export, with the two gizmo nodes detached from the scene graph
for the "removed" row, so the two differ by nothing else. The water's absorption
is set 20x stronger for these (attenuationColor 0.05, attenuationDistance 0.05)
to put the effect well clear of the noise; the ratios below are that scene.

| Suspect | Test | Result |
|---|---|---|
| The medium itself | density 1e-6, emission 0 | R/G 0.967 against 0.968 at full density -- the boundary alone does it |
| The pass-through budget | `PATH_PASSTHROUGH_MAX` 32 -> 255 | byte-identical output |
| The bounce budget | `--depth 16` -> `--depth 64` | 0.918 -> 0.911 |
| Primary visibility | water given emission 20 | 237.32 against 237.35 -- both renders see the same surface |
| Segment absorption at a scattering vertex | fixed, see below | the ladder and this scene both unmoved |
| **The IOR stack** | water marked `thin_walled`, which is the one path that never pushes it | **0.431 / 0.918 becomes 0.988 / 0.993 -- the whole difference collapses** |

The last row is the finding: with nothing pushed onto the stack the gizmo makes
no difference at all. What the gizmo changes is how long an unbalanced path
survives, not what any surface does.

One real gap was found on the way and is fixed rather than listed: the absorption
over a segment was applied in the surface branch and the boundary-crossing branch
of `shade`, but not at a volume scattering vertex, which returns from its own
branch before either. It is hoisted to the top of the kernel now. It is worth
nothing in this scene -- the fog gizmo could be neutered to 1e-6 density and the
symptom did not move -- and it is still wrong to skip.

### What to do about it

Two candidates, and the measurement to choose between them is the same:

- close the meshes in `tools/iso_bathroom/vray2strelka.py`, which is where the
  asset is already being repaired for other reasons;
- or make an unmatched exit recoverable in `ior_stack.h`, which is the general
  fix and the riskier one -- `ior_stack_pop` searches by priority and silently
  succeeds when it finds nothing, so there is no signal today that a path is
  lost.

Either way the check is the same: with the water closed, the gizmo should stop
mattering, and R/G should move toward 0.871 rather than away from it.

`tools/iso_bathroom/vray2strelka.py --no-fog-volumes` is the lever meanwhile, and
is what the current export uses.

---

## 2. A pass-through costs a bounce, whatever the comments say

Three places in `src/shaders/metal/wavefront.metal` deliberately do not advance
the path's `depth`: cutout geometry, a medium boundary crossing, and a
subsurface walk step. Each says why -- a hedge of cutout leaves would otherwise
exhaust `maxDepth` before any of its transport happened, and a volume the light
crosses twice would cost two bounces.

The budget they are avoiding is not the one that ends the path. `MetalRender.mm`
drives the wavefront as `for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)`,
one extend/shade pair per iteration, and a path that spends an iteration passing
through something has spent it whether or not `depth` moved. `depth` gates NEE
weighting, clamping and Russian roulette; it does not gate the loop.

Not measured as a cost anywhere yet -- raising `--depth` from 16 to 64 on the
bathtub above moved R/G by 0.007, so whatever that scene is limited by, it is not
this. It is recorded because the comments state the opposite, and the next person
to trust them will be debugging a canopy that goes black at a `maxDepth` that
looks generous.

---

## 3. A rough thin-walled surface transmits as a delta but is weighted as glossy

`standard_pbr_sample` sends thin-walled transmission straight through -- `wi` is
exactly `-wo` -- at every roughness, because a thin wall has no interior to
refract across. The pdf and the event type do not agree with that. Measured over
50k samples per roughness, on a thin-walled dielectric at IOR 1.6:

| Roughness | max distance from `wi` to `-wo` | mean returned pdf | event |
|---|---|---|---|
| 0.0 | 0 | 0.945 | `SPECULAR_TRANSMISSION` |
| 0.1 | 0 | 48.9 | `GLOSSY_TRANSMISSION` |
| 0.3 | 0 | 0.619 | `GLOSSY_TRANSMISSION` |
| 0.6 | 0 | 0.045 | `GLOSSY_TRANSMISSION` |

So a frosted thin sheet passes light as a perfect mirror-through while telling
MIS it sampled a spread lobe, and `standard_pbr_eval` compounds it: it builds
the half vector as `normalize(V + eta * wi)`, a refraction that never happened,
and evaluates a BTDF over directions the sampler cannot produce. A light seen
through such a sheet is therefore weighted against a density that describes a
different surface.

The soap bubbles are at roughness 0 and are not affected -- the smooth row above
is self-consistent. What this costs has not been measured because no scene in
the tree has a rough thin-walled material; `tools/feature_tests/` would need a
new rung before the fix could be checked against Cycles, and inventing the
weighting without that is how the clearcoat term below got rejected twice.

---

## 4. The clearcoat does not return what bounces under it

`scenes/feature_tests/15_clearcoat` runs 6% dark against Cycles at the strong end
of its IOR ramp and matches exactly at IOR 1.0 — the signature of a missing term
scaling with the coat's reflectance. Cycles models the light that goes through
the coat, off the base, and back down off the coat's underside.

Two formulations were tried and both produced a material brighter than the light
falling on it, measured by `tests/material/test_clearcoat.cpp`: summed against
the coat's internal hemispherical reflectance a glazed white ceramic reached 2.43
directional albedo, and against the external average 1.02. The round trip carries
a 1/eta² radiance compression that does not separate cleanly from the reflectance
when what sits under the coat is a full BSDF rather than a Lambertian.

A documented 6% beats an energy violation, so it stays out until it can be
derived rather than fitted.

---

## 5. V-Ray colour correction drops `adv_base`

`tools/iso_bathroom/vray2strelka.py`'s `bake_color_correction` implements
brightness, contrast, the advanced lightness curve and the hue tint, but not
`adv_base`. The wood materials set it to 2.0, and without it the contrast of 2.4
pivots around 0.5 and crushes the dark end to black — which is what makes the
window frame read as too dark and too contrasty against the reference.

V-Ray's formula for the advanced lightness mode is not documented anywhere this
conversion could check, and fitting one by eye would be a guess wearing the
clothes of a conversion.
