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

`--close-transmissive` joins the boundary loops of refracting meshes, which is
entry 1's lever and is off by default for the reason given there. The fog volumes
are the other one: dropping `--no-fog-volumes` exports them, and they cost a
boundary crossing per scattering event, so they are a good way to stress the
medium code.

`tools/iso_bathroom/bubble_profile.py` reads back a rendered PNG as mean
luminance per radial bin across one of the bubbles, against the wall just
outside it. A defect with a radius is far easier to name than one with only a
colour.

---

## 1. Nested dielectrics cannot tell two objects apart

**Half fixed.** The renderer can now tell two nested objects apart; it still
cannot survive a mesh with a hole in it, and neither failure says a word.

`ior_stack_pop` now matches on the material being left and falls back to the
priority. Priority alone could not identify anything: glTF has no way to author
it and the loader gives one value to everything transmissive, so all twelve
refracting materials in the bathroom sit at 10 and leaving the shower glass
popped whichever of them was topmost. `tests/material/test_ior_stack.cpp` pins
it. It is worth nothing in this frame -- byte-identical output -- because the
scene rarely has two of them nested at once, which is exactly why it went
unnoticed.

The other half cannot be fixed in the renderer at all: a ray leaving an open mesh
goes out through the hole without crossing a surface, so there is no exit event
to hang a recovery on. It has to be repaired where the geometry is, and that is
where this stands. What is open in this scene:

| Mesh | Boundary edges | Shape |
|---|---|---|
| `Brush_Fibers` | 2912 | open-ended strips |
| `Bubbles` (the foam) | 196 | a cluster of open shells |
| `Water_Bathtub` | 96 | one loop, flat, at the top of a 2 mm dish |
| `Water_Shower` | 64 | one flat loop |

`--close-transmissive` joins those loops and is **off by default**, because the
repair is only sound where the loops were meant to be joined. Measured both ways:

| | bath water R/G |
|---|---|
| left open, as shipped | 0.776 |
| water capped, foam capped | 0.823 |
| foam bridged, water left open | 0.709 |
| reference | 0.837 |

So closing the water is right -- it takes the error down by 4.4x -- and closing
the foam is not: bridging a cluster of open shells draws a bright streak across
the tub, and capping the fibre strips is no better. Partial closure is worse than
none, which is the row at 0.709.

What would make it shippable is a per-mesh decision rather than a per-scene flag:
cap a boundary loop that is planar and closes a dish, bridge a pair of loops that
face each other, and leave anything else alone and say so. The measurement to
aim at is already here.

**What is still open** is that both failures were silent. `ior_stack_push` does
nothing when the stack is full at four entries, and `ior_stack_pop` returns
success having found nothing. A scene can lose paths to either without a single
warning, and this one did for as long as it has existed. Somewhere to put a
counter -- a debug view of stack depth, or a once-per-frame tally of unmatched
pops -- would turn the next instance into a measurement instead of a hunt.

---

## 2. A thin wall does not blur what is behind it

The consistency half of this is fixed: thin-walled transmission returns exactly
`-V` at every roughness, and now reports itself as the delta it is rather than as
a spread lobe with a microfacet density behind it. `eval` returns zero for it
instead of building a half vector for a refraction that never happened. Measured
before the fix, over 50k samples: at roughness 0.1 every transmitted sample
landed on one direction carrying a pdf of 48.9.

What is left is that a frosted thin sheet ought to blur what is behind it and
does not. Modelling that means refracting through the microfacet and back at the
second interface, which is a small amount of code and no way at all to check it:
there is no rough thin-walled material anywhere in the tree, so
`tools/feature_tests/` has nothing to build a rung from. A lobe written against
no measurement is how the clearcoat term below was rejected twice.

The soap bubbles are at roughness 0 and are unaffected either way.

---

## 3. The clearcoat does not return what bounces under it

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

## 4. V-Ray colour correction drops `adv_base`

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

## 5. The rect lights lose their directionality, and the room loses the light

**Symptom.** Not exposure: an exposure error scales the whole frame, and this
moves the two halves of it in opposite directions. Mean luminance over patches of
the 1024² render against the same patches of the reference:

| Patch | Reference | Strelka | |
|---|---|---|---|
| backdrop, outside the room | 0.703 | 0.764 | +9% |
| outer wall | 0.418 | 0.396 | -5% |
| interior white wall | 0.674 | 0.627 | -7% |
| floor tile | 0.846 | 0.724 | -14% |
| tiled wall | 0.773 | 0.643 | -17% |
| tub rim | 0.676 | 0.519 | -23% |

**What it is.** Both `VRayRectLight`s carry a `directional` parameter that the
conversion drops:

| Light | `intensity` | `directional` | other |
|---|---|---|---|
| `VRayRectLight_Side` | 35.0 | 0.1 | |
| `VRayRectLight_Window` | 10.0 | 0.5 | `invisible 1`, `specular_contribution 0.5` |

V-Ray's `directional` narrows the emission lobe: at 0 the light is an ordinary
Lambertian rectangle, and toward 1 it concentrates along its normal. Both of
these point into the room. Exported as plain Lambertian rectangles they spread
their power over the whole hemisphere instead, so less of it reaches the room and
more of it leaves through the open sides of what is, after all, a cutaway
diorama -- and the backdrop behind it is lit by the difference.

One cause, both signs. That is the part worth having: the room being dark and the
backdrop being bright are not two problems.

### Ruled out

| Suspect | Test | Result |
|---|---|---|
| Indirect clamping | `--clamp 8` against none | floor tile +1.5%, nothing else past that |
| Path depth | `--depth 16` against 32 | identical to four decimals |
| Light magnitude | rect intensity x1, x1.5, x2, x3 | no single gain reconciles them, see below |

The gain sweep is what rules out a units error and points at the distribution.
Scaling both rect lights together:

| Patch | Reference | x1 | x1.5 | x2 | x3 |
|---|---|---|---|---|---|
| backdrop | 0.703 | 0.764 | 0.829 | 0.867 | 0.907 |
| floor tile | 0.846 | 0.724 | 0.783 | 0.822 | 0.866 |
| tiled wall | 0.773 | 0.643 | 0.699 | 0.733 | 0.771 |
| tub rim | 0.676 | 0.519 | 0.547 | 0.566 | 0.591 |

By x3 the tiled wall has arrived, the floor has overshot, the backdrop is 29%
over, and the tub rim is still 13% under. A brightness that is wrong by a
different factor in every part of the frame is not a brightness.

### Why it is not fixed here

Implementing `directional` means knowing V-Ray's falloff, and Chaos documents
what the slider does rather than the function behind it. Fitting a cosine power
to this one frame would be the same mistake as entry 4 -- worse, because the
lights are what everything else in the scene is measured against, so a fitted
light would quietly absorb every other error in the conversion.

What would close it without a guess: one V-Ray render of a rectangle at
`directional` 0, 0.25, 0.5, 0.75 and 1 against a flat wall. The falloff can be
read straight off that, and the sidecar already has somewhere to put it.

### A note on what this is *not*

The lamp globe above the mirror reads as a dull grey ball here and a bright white
one in the reference, which looks like a missing light and is not. The blend has
exactly four emitters -- a dome, the two `VRayRectLight`s and one mesh light on
`Light_Plane` -- and all four are exported. `Lamp_Bulb` is `Glass_Clear_Mtl`,
`Lamp_Plafond` is a plain diffuse shade, and neither emits in V-Ray either. The
globe is dim for the same reason the rest of the room is.

---

## 6. RIS and plain next-event estimation converge to different images

Turning up `render.ris_candidates` on the bathroom does not reduce variance -- it
moves the answer. Both estimators converge cleanly toward *their own* result, and
the two results are not the same. Relative RMSE, 1024 spp against a 4096 spp
reference of the matching estimator:

| | whole frame | floor tile |
|---|---|---|
| NEE 1024 against NEE 4096 | 0.0435 | 0.0319 |
| RIS 1024 against RIS 4096 | 0.0396 | 0.0270 |
| **NEE 4096 against RIS 4096** | **0.0490** | **0.0698** |

The third row is the finding. Two converged renders of the same scene differ by
more than either differs from its own half-converged version, and on the floor by
five times the noise left at 4096 spp. One of them is wrong.

The ladder cannot say which. RIS and NEE agree *exactly* where it can test them:
`00_calibration` and `02_basecolor` come out at 1.0098 and 1.0028 against Cycles
either way, and so does `12_lights_punctual` at 1.0020 with three lights in it.
That is not luck -- resampling among candidates is a no-op when the candidates
are drawn from one light, and evidently faithful with three punctual ones. What
the bathroom has and no rung does is an environment map *and* analytic lights at
once, which is where `connectToLight` splits its draw between the two strategies.

So the next step is a rung, not a debugger: one scene with an environment and a
rect light together, which the ladder wants for its own sake.

It also costs 60% more time per sample here, so there is no reason to raise it
until this is settled. The default of 1 is plain NEE and is what every measured
row in the ladder was recorded with.

---

## 7. The denoiser makes the image worse at every sample count

Not a question of giving it enough samples first. Relative RMSE against a 4096
spp reference, denoiser off against on, same spp:

| spp | off | on | |
|---|---|---|---|
| 16 | 0.5085 | 0.5317 | 1.05x worse |
| 64 | 0.2446 | 0.2822 | 1.15x |
| 256 | 0.1042 | 0.1723 | 1.65x |
| 1024 | 0.0435 | 0.1437 | 3.30x |

At 16 spp a guided denoiser should be transformative and it is slightly negative;
by 1024 it is destroying most of what the samples bought. The architecture around
it is not the problem -- it is handed the accumulated estimate rather than one
sample, and its guides come from a canonical extra sample, so "accumulate first,
then denoise" is already what happens.

It is spatially selective, and that is the lead. The backdrop is untouched
(0.0127 against 0.0128 at 1024 spp); the floor tiles go 0.0319 to 0.1272. Flat
matte surfaces survive, tiled and glossy ones do not.

The guides say why. Rendered at `render.debug` 3, 5 and 6 -- diffuse albedo,
normal, roughness -- all three are salt and pepper over exactly the floor and the
tiled walls, and clean over the plaster, the backdrop, the tub and the plant. The
roughness view is the clearest: those surfaces come back as a per-pixel mix of
black and white, i.e. the guide vertex lands on a near-mirror for one pixel and
on something rough for the next.

That is the guide walk. `shade` takes guides from "the first surface that can
actually be described", walking past anything with `roughness <= 0.05` so that a
mirror does not hand the denoiser a featureless black albedo where a reflected
world is. The bathroom's ceramics sit close enough to that threshold that the
decision flips from pixel to pixel, and what the denoiser then demodulates
against is the albedo of whatever each pixel's reflection happened to land on.

Not fixed because the fix is a design choice with no measurement behind it yet.
Hysteresis on the threshold, a roughness taken from the material rather than the
textured value, or simply taking the guide at the primary hit whenever the camera
is static are all plausible and all differ on the case the walk exists for, which
is a mirror. The measurement to aim at is the table above: the denoiser has to
beat 0.0435 at 1024 spp before it is worth turning on.

Fixed on the way, because it made the above impossible to see: the AOV debug
views could not show what the denoiser receives. Looking at a guide requires
`debug != 0`, `debug != 0` disables denoising, and the canonical guide sample was
tied to denoising being on -- so every guide view rendered its guides the other
way, every sample overwriting the last into a buffer that is assigned rather than
accumulated. The views now assemble guides the same way whether they are being
consumed or looked at. It did not change what these three views show, which is
how I know the speckle is real.
