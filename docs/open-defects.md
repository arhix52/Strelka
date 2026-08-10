# Open defects

Each entry is something measured and left unfixed, with the measurement that
found it and what has already been ruled out. They are written so that a reader
starting cold can act without repeating the elimination.

Most of what is below reproduces from the Isometric Bathroom conversion; entry 8,
and the second half of entry 5, come from the Isometric Kids Bedroom, which the
same converter reads:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Kids_Bedroom/Iso_Kids_Room.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out /tmp/kids --name kids_room \
    --no-fog-volumes
```

Its reference is
<https://documentation.chaos.com/download/attachments/117637916/Render_Camera_1280_0x008_15m.png>,
at 1280 square.

Build the bathroom once:

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

`tools/iso_bathroom/patch_mean.py` is the other measurement in this file: mean
*linear* luminance over a rectangle of a rendered PNG, across several images at
once, so every patch row below can be reproduced by reading the command rather
than by trusting the number. Linear because the images are display-encoded, and a
mean of sRGB bytes weights the dark half of the range far too heavily -- which is
why the tables that predate it are not comparable with the ones that follow it.

```bash
python3 tools/iso_bathroom/patch_mean.py /tmp/kids_ref.png /tmp/kids.png \
    -- 920 380 30 30 right_wall  660 260 40 60 curtain
```

---

## 1. Nested dielectrics cannot tell two objects apart

**Half fixed, and no longer silent.** The renderer can now tell two nested
objects apart, and it now counts what it loses; it still cannot survive a mesh
with a hole in it, and that part cannot be fixed here.

`ior_stack_pop` now matches on the material being left and falls back to the
priority. Priority alone could not identify anything: glTF has no way to author
it and the loader gives one value to everything transmissive, so all twelve
refracting materials in the bathroom sit at 10 and leaving the shower glass
popped whichever of them was topmost. `tests/material/test_ior_stack.cpp` pins
it. It is worth nothing in this frame -- byte-identical output -- because the
scene rarely has two of them nested at once, which is exactly why it went
unnoticed.

The other half cannot be fixed in the renderer at all: a ray leaving an open mesh
goes out through the hole without crossing a surface, so there is no exit event to
hang a recovery on. The converter repairs what it safely can, per boundary loop:

| Mesh | Loops | Span, as a fraction of the object | Out of plane | Verdict |
|---|---|---|---|---|
| `Brush_Fibers` | 364 | 0.013 - 0.014 | 1e-5 | capped, 2912 edges |
| `Bubbles` (foam) | 2 | 0.29, 0.41 | 7e-4 | left open |
| `Water_Shower` | 1 | 0.56 | 2e-3 | left open |
| `Water_Bathtub` | 1 | 1.00 | 0 | left open |

A loop is capped when it is flat *and* small against the object it belongs to --
a hole rather than a feature. The two clusters in that table are an order of
magnitude apart, so the threshold sits in the gap rather than on a case. That
takes 2912 of the scene's 3268 open edges, which is the largest single unbalance
in it, and moves no patch of the frame by more than 0.003.

### What refuses to be repaired, and why that is the finding

Every wider rule I tried made the render worse while making the number better,
which is worth writing down because the number is the obvious thing to optimise:

| Repair | bath water R/G | Render |
|---|---|---|
| none | 0.776 | correct |
| holes only, as above | 0.776 | correct |
| foam capped as well | 0.731 | green streaks across the tub |
| foam bridged instead | 0.709 | one bright streak |
| every flat loop capped, water included | **0.823** | a pale sheet across the tub, duck half under it |
| reference | 0.837 | |

The last row but one is the trap. Capping the bath water is arithmetically the
best result available and it is plainly wrong on screen, because the water is a
2 mm dish and not a tub full of water: its one boundary loop spans the whole
object, so a cap does not close a volume, it lays a second sheet over one.

What makes the bath look full today is the defect itself. An unbalanced stack
applies the water's absorption to every segment the path travels afterwards,
including the inside of the tub -- so the tub reads as full of water because the
renderer believes the ray still is. The asset has no water volume, and nothing
here can invent one. Repair the mesh and the accident goes with it.

So the bath is right for the wrong reason, and will stay that way until the water
is modelled as a volume upstream. That is a note for whoever next opens the
.blend, not something the converter should paper over.

### The silence is fixed: they are counted now

`ior_stack_push` still does nothing when the stack is full and `ior_stack_pop`
still returns success having found nothing, because both are the right thing to
do -- but neither is silent any more. Three counters, in a shared buffer `shade`
and `miss` write and the host reads back once per scene:

| | what it means |
|---|---|
| pushes onto a full stack | four nested dielectrics; wants a deeper stack |
| pops that matched nothing | a ray *leaving* something it never entered |
| paths that reached the environment still inside a medium | a ray that *entered* something it never left |

The third one is the one this entry said could not be caught. It cannot be caught
at an exit event -- a ray leaving through a hole crosses no surface, so there is
no event -- but it can be caught where the path ends, and a path that reaches
infinity while its stack still holds glass has plainly lost track.

Per sample, at 1024², depth 16:

| Bathroom | full-stack pushes | unmatched pops | escaped inside |
|---|---|---|---|
| as shipped (holes capped) | 0 | 7 | 2 |
| `--no-close-transmissive` | 0 | 19 | 2 |

Which is the measurement this entry was missing, and it says two things. The
repair is real and does what the edge count claimed -- roughly a third of the
unmatched pops survive it. And the residual is a handful of paths in a million,
which is why capping the holes moved no patch of the frame by more than 0.003:
the defect was always small, and the reason it took a hunt to find was that
nothing counted it.

The counters cost one atomic each, only on the branch where something has already
gone wrong, plus a search over at most four entries on a path that has already
established it is a transmission through a solid. `ior_stack_full` and
`ior_stack_can_pop` are predicates rather than return values because the callers
are in two renderers and a changed signature is a changed OptiX payload.

**Still open**: the OptiX path calls push and pop without asking either question,
so this measurement exists on Metal only.

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

### Confirmed on a second scene -- but most of what was measured there was not this

**The Isometric Kids Bedroom numbers this entry used to carry were wrong, and
they were wrong in this entry's favour.** Its walls were not dark because the
rect lights spread their power over a hemisphere. They were dark because a
greyscale height map was wired into `normalTexture`, and a greyscale below 0.5
decodes to a shading normal that points *into* the surface. See "A height map is
not a normal map" under entry 8: the right wall did not measure -58%, it measured
0.0000 -- not dark, black -- and it now measures within 9% of the reference.

What is left of this defect there, mean linear luminance over patches of the
1280² render (`tools/iso_bathroom/patch_mean.py`; not comparable with the
sRGB-byte means this table used to hold):

| Patch | Reference | Strelka | |
|---|---|---|---|
| right wall | 0.2713 | 0.2951 | +9% |
| back wall | 0.0854 | 0.0640 | -25% |
| left wall | 0.1355 | 0.4449 | **+228%** |
| floor | 0.0265 | 0.0405 | +53% |
| outer wall | 0.2344 | 0.0796 | **-66%** |
| backdrop, outside the room | 0.0469 | 0.1278 | **+173%** |

The backdrop and the outer wall still say what this entry says: light that should
be inside the room is outside it. The interior no longer does -- one wall is
over, one is under, one is close -- so this scene is no longer evidence that the
room as a whole is starved, and the "both signs, one cause" claim now rests on
the bathroom alone.

The left wall at more than three times the reference is the largest single thing
wrong with this frame, and it has no explanation here. It is the same material as
the right wall, which is within 9%.

The parameters are unchanged and still not carried:

| Light | `intensity` | `directional` |
|---|---|---|
| `VRayRectLight_Window_02` | 1.5 | 0.95 |
| `VRayRectLight_Window_01` | 1.5 | 0.90 |
| `VRayRectLight_CorridorLight_Fill` | 1.5 | 0.70 |
| `VRayRectLight_Main` | 0.8 | 0.65 |
| `VRayRectLight_Laptop` | 10.0 | 0.20 |

Against the bathroom's 0.1 and 0.5. Two windows at 0.9 and 0.95 are nearly
searchlights aimed into the room; exported as Lambertian rectangles they spread
that over a hemisphere, which is still the best account of a backdrop at nearly
three times the reference with the room's own outer wall at two thirds of it.

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

**Does not reproduce.** This entry recorded that turning up
`render.ris_candidates` moved the answer rather than reducing variance, on the
strength of one measurement:

| | whole frame | floor tile |
|---|---|---|
| NEE 1024 against NEE 4096 | 0.0435 | 0.0319 |
| RIS 1024 against RIS 4096 | 0.0396 | 0.0270 |
| **NEE 4096 against RIS 4096** | **0.0490** | **0.0698** |

The third row was the finding: two converged renders differing by more than
either differs from its own half-converged version. The same three rows, taken
again on the same scene, at both 1024² and 512², depth 12:

| | 1024² | 512² |
|---|---|---|
| NEE 1024 against NEE 4096 | 0.0583 | 0.0588 |
| RIS 1024 against RIS 4096 | 0.0561 | 0.0567 |
| **NEE 4096 against RIS 4096** | **0.0126** | **0.0121** |

The two self-convergence rows are close to what they were. The cross-estimator
row is a quarter of it, and it is now *five times smaller* than the noise either
estimator has left at 1024 spp -- the opposite conclusion, at the same resolution
the original was taken at, and the same at half of it. Something
between the two measurements closed it, and the most likely candidate is
`93421cf`, which stopped `eval` building a half vector for a refraction that
never happened on a thin wall -- the bathroom is full of thin walls, and RIS
picks its survivor by `luminance(f)`, so a wrong `f` moves which candidate wins
as well as what it is worth.

### The rung this entry asked for exists, and it exonerates the split

The hypothesis was that the disagreement lived in `connectToLight`'s split
between an environment and the analytic lights, which no ladder row had.
`scenes/feature_tests/19_env_and_light` now does: a physical sky baked to an
equirectangular map, plus the ladder's usual area light, over three roughnesses.
Against Cycles it is 0.026 / 1.000 at `ris_candidates = 1` and 0.023 / 1.000 at
8, and the two Strelka images differ from each other by 0.014 -- less than either
differs from the reference.

An ablation on the bathroom says the same thing from the other end. NEE against
RIS at 1024 spp, 512², with the sidecar's lights and its environment removed in
turn -- mean absolute difference over mean radiance, the metric `compare.py`
prints as `rel`, not the relative RMSE of the table above:

| Lighting | NEE vs RIS |
|---|---|
| environment + three rect lights | 0.0178 |
| rect lights only | 0.0223 |
| environment only | 0.0142 |

The disagreement does not need the environment at all -- it is *largest* with the
environment removed. Whatever the residual is, it is not the split.

**What is left** is a small, ordinary estimator difference at the level of the
remaining noise, and no evidence that either is wrong. RIS still costs about 40%
more time per sample on this scene (14.1 s against 20.0 s for 1024 spp at 512²),
which is the honest reason to leave the default at 1; it is no longer that the
answer moves.

---

## 7. The denoiser floors out, and the guide source is a switch rather than a default

**Fixed first, because it invalidated the previous version of this entry:** the
denoised frame never reached the file. It lands in a texture, the display path
consumed it, and StrelkaCLI writes from the buffer -- so a headless
`denoise = true` wrote the estimate the denoiser had been *handed*, while still
paying for it: a canonical guide sample it does not accumulate, frame jitter, and
the firefly clamp. That is the whole of what this entry previously reported as a
denoiser that damages the image. It does not.

Relative RMSE against a 4096 spp reference, with the output actually connected:

| spp | denoiser off | on, guide walk | on, guides at the primary hit |
|---|---|---|---|
| 4 | 1.0860 | 0.2747 | **0.2128** |
| 8 | 0.7615 | 0.2345 | **0.1855** |
| 16 | 0.5085 | 0.1709 | **0.1492** |
| 64 | 0.2446 | 0.1248 | 0.1248 |
| 256 | 0.1042 | 0.1101 | 0.1136 |
| 1024 | 0.0435 | 0.1056 | 0.1091 |

Two things fall out of that.

**The denoiser has a floor at about 0.105 and cannot go under it.** Between 256
and 1024 spp its input improves by 2.4x and its output does not move. So it is
worth 5x at 4 spp, 2x at 64, and nothing past roughly 200 -- accumulating and
then denoising cannot beat accumulating, because what limits the result is the
reconstruction and not the samples. For a still at 1024 spp, plain accumulation
is 2.4x better than the best the denoiser can produce.

**`render.guide_primary_hit` is worth about a fifth on this scene**: -23% at 4
spp, -21% at 8, -13% at 16, nothing at 64, and +3% at 256 and above, which is
inside the floor. Taking the guides at the camera-visible surface removes the
flicker described below. On a scene with a large mirror it costs 110% instead --
see the rung below, which is why it is not the default.

The flicker, for the record. Guides are otherwise taken from "the first surface
that can actually be described", walking past anything with `roughness <= 0.05`
so a mirror does not hand over a featureless black albedo where a reflected world
is. The bathroom's ceramics sit close enough to that threshold that the decision
flips per pixel: rendered at `render.debug` 3, 5 and 6, the albedo, normal and
roughness guides are all salt and pepper over the floor and the tiled walls, and
clean everywhere else. With `guide_primary_hit` they are clean everywhere except
the mirror and the chrome, which is correct -- those have no diffuse albedo.

**Settled: the walk stays the default.** The rung this entry asked for is built
-- `scenes/feature_tests/20_mirror_and_floor`, a mirror filling the background
above a textured rough floor -- and it answers the question in the direction that
keeps the walk. Relative RMSE against a 4096 spp render of the same scene:

| spp | denoiser off | on, guide walk | on, guides at the primary hit |
|---|---|---|---|
| 4 | 0.1299 | **0.2252** | 0.4132 |
| 8 | 0.0910 | **0.1931** | 0.3969 |
| 16 | 0.0688 | **0.1868** | 0.3878 |
| 64 | 0.0327 | **0.1826** | 0.3829 |

The walk is better by a factor of 2.1, at every sample count. Against that, the
bathroom -- re-measured at 16 spp with the same reference discipline -- prefers
the primary hit by 6%. A feature that buys 6% where it wins and costs 110% where
it loses is not a default; it is a switch, which is what it is.

The same table says something else this entry did not: **on a scene that is
mostly mirror the denoiser is a net loss at every sample count**, including 4.
There is nothing for a reconstruction filter to average over a specular image,
and the floor it hits here (0.18) is well above the one the bathroom showed
(0.105). The floor is a property of the scene, not of the denoiser.

### The firefly clamp was measuring itself

Found while building that rung, and worth its own line. `denoiseFireflyClamp` is
8 in exposed units, and on a scene with a mirror pointed at a light that is not a
conditioning term -- it is a truncation. With it on, the denoised mean came out
21% below the reference, every value above 8 was gone, and the error at 4 spp was
0.5566 instead of 0.2252. Both guide sources were hit equally, so the comparison
above survived it, but the absolute numbers did not mean what they appeared to.

It is now a config key (`render.denoise_firefly_clamp`, still 8 by default) so
the harness can ask the question. On the bathroom, which has no such highlight,
turning it off is worth 2%.

There is also a smaller thing this uncovered and fixed: the AOV debug views could
not show what the denoiser receives. Looking at a guide requires `debug != 0`,
which disables denoising, which is what the canonical guide sample was tied to --
so every guide view assembled its guides the other way, every sample overwriting
the last into a buffer that is assigned and not accumulated. The views now build
them the same way whether they are consumed or looked at.

---

## 8. What the Isometric Kids Bedroom still needs

The conversion reaches the end and the room reads. What it cannot carry, in
descending order of how much of the frame it costs.

Two of the things below are now done -- the hair and the two-sided materials --
and a third, filed here as a line item, turned out to be the largest defect in
the scene: a height map on a normal-map socket, which had turned the walls
black. Entry 5's account of this scene was built on measurements taken with that
bug in, and has been corrected.

### Hair

**Geometry done; the lobe is what is left.** The strands reach the renderer, are
traversed as curves rather than as triangles, and shade with the pigment colour
V-Ray authored. What they do not have is a hair BSDF, and that is now a
measurement rather than a guess.

The route in was the one this entry proposed: a binary sidecar beside the glTF,
`<stem>_curves.bin`, the way the analytic lights already ride in
`<stem>_light.json`. Format in `src/sceneloader/curve_sidecar.h`, writer in
`tools/iso_bathroom/curve_sidecar.py`. Binary and not JSON because the payload is
8.4 million control points -- 139 MB packed, and around 400 MB as text.

`MetalRender` now builds one curve BLAS per set through
`AccelerationStructureCurveGeometryDescriptor`, and `shade` rebuilds the hit from
the same buffers the structure was built from: segment index plus the parameter
along it gives the axis point, the tangent and the radial normal. Where along the
strand a hit landed comes out of the segment index alone -- a particle system
gives every strand the same segment count, so `segment % segmentsPerStrand` is
the strand's own coordinate and no per-point attribute has to be stored.

Two things had to be a compiled variant rather than a branch. `curve_data` is an
intersector *tag*, so a curve-capable traversal is a different type, not a
different code path: `kFeatureCurves` selects `wavefrontExtendStaticCurve` and
its three siblings, and a scene with no hair traverses exactly the kernels it
did before. And the metallib moved from `-std=metal3.0` to `3.1`, which is where
`curve_data`, `geometry_type::curve` and `curve_parameter` first exist.

That last one is the only thing here that touches a scene without hair, and it
touches it by 66 pixels in 160 000 at 32 spp -- relative RMS 5.8e-4, mean
absolute 1.2e-6. Built with `STRELKA_METAL_STRICT_FP=ON` the bathroom and the
bedroom are **bit-identical** before and after the whole change, which is what
that switch is for: the difference is multiply-add contraction landing
differently in a kernel whose registers moved, not a change in what is computed.

#### What it costs and what it buys

| | without the grooms | with them |
|---|---|---|
| frame, 128 spp at 1280² | 3.7, 4.0 s | 5.0, 5.2 s (+30%, interleaved) |
| acceleration structures | 0.17 GB | 1.11 GB |
| curve buffers | -- | 0.17 GB (8.4 M control points, 7.5 M segments) |
| device total | 0.60 GB | 1.63 GB |

938 644 strands, and they change 2170 pixels of a 1.6 M pixel frame. Both grooms
are small in this shot -- the monster is forty pixels across and the spider
twelve -- so this is a fair statement of what hair costs when it is *not* the
subject, not of what it costs in general.

Mean linear luminance over the patch each groom occupies, both sides measured
with the normal-map fix below already in (`tools/iso_bathroom/patch_mean.py`):

| Patch | Reference | Bald | With strands |
|---|---|---|---|
| monster | 0.0707 | 0.0663 (-6%) | 0.0407 (**-42%**) |
| spider | 0.0721 | 0.1169 (+62%) | 0.0565 (-22%) |

**Adding correct geometry makes the monster's number worse**, and that is the
finding rather than an argument against it. A bald ball in the fur's own blue
lands within 6% of a furry one by coincidence -- the same patch mean, a different
object. What the strands change is the silhouette, which the reference has and a
sphere does not, and a patch mean cannot see. What they get wrong is how much
light comes back out, which is the lobe.

The spider moves the other way, +62% to -22%, for the same reason with a
different sign: it was a pale smooth body where the reference has dark fuzz.

#### The lobe, which is now the whole of what is missing

The strands shade as rough dielectric cylinders at IOR 1.55, with the colour
derived rather than fitted: melanin and pheomelanin are pigment concentrations in
Chiang et al. 2016's model, which is what V-Ray Hair Next exposes, so
`exp(-sigma_a)` is what survives them and `dye_color` multiplies it. The monster's
0.05 melanin and 0.5 pheomelanin over a blue dye give (0.185, 0.184, 0.620), and
the spider's zero pigment leaves its grey dye alone.

What a cylinder cannot do is what makes fur bright: light entering a strand,
refracting, and leaving through a neighbour. The dropped terms are the whole
reason V-Ray's plugin has them -- primary, secondary and transmission lobes, the
`highlight_shift` that offsets the two specular bands, `primary_glossiness_boost`
-- and their absence has one sign, which is the 42% above. That is a rung the
ladder can hold: a groom, one light, and a Cycles reference.

#### Smaller things the export settles

- Strand resolution is `2**display_step` segments, pushed to the render setting
  and capped by `--hair-max-step` (default 3, so 8 segments). Both systems author
  fewer guide segments than that, so nothing is lost. Halving it moves the spider
  by a third of its error in the direction of the reference, which is not an
  improvement -- only a straighter strand catching light differently, and one
  more reason to believe the residual is the lobe.
- Child count is the render count, not the viewport's. Blender caches both, and
  the depsgraph hands over whichever was last evaluated -- the viewport numbers
  are an eighth of the render ones here, so reading them exports a thinner groom
  that looks like a converter that half-worked.
- The material comes from the particle system's slot *index* read off the
  original object. `material_slot` on the evaluated copy answers "Default
  Material" for every system in this file, which put both grooms on a material
  that does not exist.
- `Spider_Hair_Mtl` authors 2.5% transparency. Writing that as a BLEND material
  puts the entire scene on the cutout shadow traversal for a difference nothing
  can see, so the conversion has a floor at 5%.

### Two-sided materials

**Done, except for one material out of nine.** `Mtl2Sided` is two things at once,
and the entry was written as though the exotic one were the common case. It is
not: of the nine such materials here -- the curtains, the lampshade, the paper
plane, the notebook pages, both ping-pong balls, the ship's sails, the sticky
notes -- exactly one has a Back sub-material linked at all. The other eight are
translucency and nothing else, which `KHR_materials_diffuse_transmission` carries
exactly.

The factor is V-Ray's `translucency`. The *colour* is what needed care: Strelka's
lobe is `diffuse_transmission_color / pi` on its own, not the albedo times
anything, so the factor colour is the whole transmitted tint. Four of the nine
are textured, and for those the plugin's constant is the untouched default 0.5 --
V-Ray ignores it when a map is plugged in. Writing it made a backlit curtain
transmit mid grey. The mean of the map, taken off a 16x16 copy, is what a sheet
of it transmits, and it is one number rather than a texture slot the material
struct does not have.

Mean linear luminance, 1280² against the reference:

| Patch | Reference | Before | After |
|---|---|---|---|
| lampshade | 0.8727 | 0.2513 (-71%) | 0.4258 (**-51%**) |
| curtain, lit edge | 0.1122 | 0.3102 (+176%) | 0.2590 (+131%) |
| curtain, shaded | 0.1391 | 0.1756 (+26%) | 0.1608 (+16%) |

Every patch moves toward the reference and none of them arrives, which is the
same story as everywhere else in this scene: the curtains hang directly in front
of the two window rect lights that entry 5 is about, so what is left of their
error is that entry's, not this one's.

Finding the front sub-material needed the links rather than the node order. The
V-Ray nodes load as `NodeUndefined`, so their sockets are dead for evaluation --
but the links survive in the tree, and they are the only thing that distinguishes
front from back. `Paper_Notepad_Mtl` has two `BRDFVRayMtl` nodes and the front is
not the first of them, so reading "the highest-priority plugin anywhere in the
tree", which is what the converter did, was a coin flip on that one material.

**What is still open** is that one material: a genuinely different shader on the
back face, which the renderer has nowhere to put. It is reported rather than
approximated -- picking a side would be a converter deciding something it cannot
know.

### A height map is not a normal map, and it was not harmless

**Fixed, and it was the largest thing wrong with this scene.** The previous
version of this entry filed it as a line item and said it was "harmless here only
because the authored amount is 0.001". That was wrong twice over.

V-Ray's `bump_type` 0 means the map on the bump socket is a *height field*; the
converter wired it into a tangent-space normal map regardless. Seven of this
scene's nine bump maps are height fields, and the walls' is the greyscale mix
mask -- the same image that drives their colour.

The amount does not save it, because glTF's `normalTexture.scale` multiplies x
and y and leaves z alone (`shading_common.h:442`, which is what the spec says).
A mask texel of 0 decodes to (-1, -1, -1); scaling xy by 0.001 leaves
(0, 0, -1), and a shading normal pointing into the surface is a surface that
faces nothing. The mask is mostly dark, so the plaster walls were mostly black.

| Patch | Reference | Before | After |
|---|---|---|---|
| right wall | 0.2713 | **0.0000** (-100%) | 0.2951 (+9%) |
| back wall | 0.0854 | 0.0011 (-99%) | 0.0640 (-25%) |
| monster | 0.0707 | 0.1551 | 0.0407 |

The right wall did not measure "dark". It measured zero.

The conversion is the textbook one and needs no fitting: a height map spans the
uv range in `width` texels, so `dh/du` is the central difference times the width
times the map's repeat, and the normal is `(-amount * dh/du, -amount * dh/dv, 1)`
normalised, with the node strength then 1 because the amount is already in the
map. `TextureBaker.bake_height_normal`.

**Left open by it**, because it is a different question: those baked maps are
sampled at mip 0 by default (`render/pt/textureLodMode` 0), and a mask used as a
height field has one-texel edges, so the sparse 45-degree tilts at those edges
alias. V-Ray never sees them -- it differentiates the texture over the ray's
actual footprint, which is many texels wide at this distance. Nothing here
measures what that costs.

### Smaller, and each is a line rather than a project

- ~~`Leather_Nrm_Bump.tx` is an OIIO tiled texture Blender cannot read.~~ Stale:
  a `.tx` is a tiled TIFF, and Blender reads this one by content rather than by
  extension. It arrives at 1595x1537 with a mean of (0.500, 0.499, 0.980), which
  is what a tangent-space normal map looks like, and the exporter writes it out
  as `Leather_Nrm_Bump.tx.png`. Whatever was missing was fixed by the texture
  path work, not by anything aimed at this.
- `TexMulti` picks one of N textures by object ID and the list of N is not in the
  .blend at all -- five empty slots, nothing linked. The coloured pencils it
  drives come out at the plugin's default grey. Their object names say which
  colour each was meant to be; reading them would be a guess wearing the clothes
  of a conversion.
- Per-light `diffuse_contribution` / `specular_contribution` and include/exclude
  lists have no equivalent in Strelka. This scene's spot names `Terrain` and its
  dome asks for 0.8 diffuse and 1.5 specular.
- `BRDFCarPaint2`'s flake layer. Flakes are a spatially varying normal, not a
  colour, so the flatten in `convert_layered` takes the base colour and the coat
  gloss and reports the rest.
- The plaster walls are the right colour family and the wrong value: with the
  normals fixed they read a lighter, more yellow green than the reference's
  olive. Their base colour is a baked `TexMix`, so this is a question about that
  bake rather than about lighting -- and it is now visible, which it was not
  while the wall was black.
