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

## 1. Nested dielectrics cannot tell two objects apart

**Fixed for this scene, open in general.** The open-mesh half is closed and the
identification half is closed; what is left is that neither has a defence.

`ior_stack_pop` now matches on the material being left and falls back to the
priority. Priority alone could not identify anything: glTF has no way to author
it and the loader gives one value to everything transmissive, so all twelve
refracting materials in the bathroom sit at 10 and leaving the shower glass
popped whichever of them was topmost. `tests/material/test_ior_stack.cpp` pins
it. It is worth nothing in this frame -- byte-identical output -- because the
scene rarely has two of them nested at once, which is exactly why it went
unnoticed.

The converter now fills the boundary loops of any refracting mesh, because a ray
that leaves an open one through the hole never crosses a surface and so never
generates the exit event a recovery would have to hang on. There is nothing the
renderer can do about it. What was open:

| Mesh | Boundary edges |
|---|---|
| `Brush_Fibers` | 2912 |
| `Bubbles` (the foam) | 196 |
| `Water_Bathtub` | 96 |
| `Water_Shower` | 64 |

That takes the bath water's red-to-green ratio from 0.776 to 0.823 against the
reference's 0.837 over the same rectangle -- the error falls by 4.4x -- and moves
no other patch in the frame by more than 0.001.

It also corrects a prediction this entry used to make. The water is a 2 mm slab
and I reasoned from its thickness that closing it could be worth about 1% of the
red channel. It is worth 6%, because the colour never came from one crossing:
the slab is thin, nearly parallel-sided and sits over a reflective tub, so a
path crosses it many times.

**What is still open** is that both failures were silent. `ior_stack_push` does
nothing when the stack is full at four entries, and `ior_stack_pop` returns
success having found nothing. A scene can lose paths to either without a single
warning, and this one did for as long as it has existed. Somewhere to put a
counter -- a debug view of stack depth, or a once-per-frame tally of unmatched
pops -- would turn the next instance into a measurement instead of a hunt.

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
`adv_base`. It is the largest material difference left against the reference:
the window frame, the sill, the stool and the vanity all read too dark and too
contrasty, and the shower's pastel checkerboard comes out saturated maroon and
olive where the reference is cream and pale green.

The two materials that set it, read out of the blend:

| Material | `lightness_mode` | `adv_base` | `adv_contrast` | `adv_brightness` | `adv_offset` |
|---|---|---|---|---|---|
| `Wood_Light_Mtl` | 1 | 2.0 | 2.4 | 1.5 | -0.20 |
| `Pot_02_Ceramic_Mtl` | 1 | 5.0 | 1.0 | 1.3 | -0.15 |

What that second row rules out, and it is the useful part: **`adv_base` is not
only a pivot for the contrast.** The pot sets contrast to exactly 1.0, where a
pivot cannot do anything at all, and still authors a base of 5.0. So the term
enters somewhere the contrast does not reach.

It also rules out the two readings that are easiest to reach for. A plain
multiplier gives the pot `5 * in * 1.3 - 0.15`, which is past 1.0 for anything
above 0.18 and would render it flat white; a plain divisor gives
`in / 5 * 1.3 - 0.15`, which is negative below 0.58 and would render it black.
The pot is neither. Whatever `adv_base` is, it roughly preserves mid grey at 5.0
while doing nothing visible at 1.0 — which is the shape of a base for a
logarithm or a gamma, not of a gain.

Still not fitted. The constraint above narrows the search rather than closing
it, and a curve fitted to two materials by eye is a guess wearing the clothes of
a conversion.

---

## 6. The room is darker than the reference and the backdrop is brighter

Not exposure: an exposure error scales the whole frame, and this moves the two
halves of it in opposite directions. Mean luminance over patches of the 1024²
render against the same patches of the reference:

| Patch | Reference | Strelka | |
|---|---|---|---|
| backdrop, outside the room | 0.703 | 0.764 | +9% |
| outer wall | 0.418 | 0.398 | -5% |
| interior white wall | 0.674 | 0.630 | -6% |
| floor tile | 0.846 | 0.725 | -14% |
| tiled wall | 0.773 | 0.648 | -16% |
| tub rim | 0.676 | 0.525 | -22% |

The backdrop is lit by the dome light alone and everything else is lit through a
window and by two rect lights plus whatever bounces. So the shape of it is that
the environment carries too much of the frame and the room's own light too
little, or that too much is lost per bounce inside a closed room.

Ruled out: indirect clamping (`--clamp 8` against none moves the floor tile by
1.5% and nothing else by more than that) and path depth (16 against 32 is
identical to four decimal places). Entry 5 accounts for part of the tiles and
none of the plaster.

Worth checking next, in this order: the sidecar's radiance conversion for the two
rect lights against V-Ray's `intensity` units, the dome light's 0.3 multiplier,
and whether the room's white plaster is being converted with an albedo low enough
to cost this much over the several bounces an enclosed room needs.

A note on what this is *not*: the lamp globe above the mirror reads as a dull
grey ball here and a bright white one in the reference, which looks like a
missing light and is not. The blend has exactly four emitters -- a dome, two
`VRayRectLight`s and one mesh light on `Light_Plane` -- and all four are
exported. `Lamp_Bulb` is `Glass_Clear_Mtl`, `Lamp_Plafond` is a plain diffuse
shade, and neither emits in V-Ray either. The globe is dim here for the same
reason the rest of the room is.
