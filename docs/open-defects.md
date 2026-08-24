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
| 12 | Metal's radiance cache has no resolve pass | `src/shaders/metal/sharc.h` + a new kernel | Metal's cache survives a camera movement, as OptiX's now does |
| 13 | OpenPBR's vendored BSDF has never been through nvcc | `third_party/openpbr_bsdf`, `src/shaders/optix` | an OptiX module that calls `openpbr_prepare` compiles, and the ladder is unmoved |
| 14 | The subsurface walk does not reproduce run to run | `wavefront.metal` medium path, Metal | two runs of one binary on `25_subsurface` are bit-identical |
| ~~15~~ | ~~Diffuse summed with specular instead of layered under it~~ | done — see Closed | `00_calibration` 0.012 / 1.008, its three regions within 0.4% of each other |
| 16 | Subsurface is 3% dark since the entry was corrected | the entry weight, `wavefront.metal` / `OptixRender_closest_hit.cu` | the four subsurface rows back at ratio ~1.00; the slab is already there |

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

## 12. Metal's radiance cache is still the one-frame version

**OptiX-only fix, deliberately, and this is the hand-off.** The two backends'
caches were a port of each other -- `src/shaders/optix/sharc.h` says so, and says
they must stay one -- and they no longer are.

**What OptiX gained.** An entry is now in two halves, after SHARC v1.8.3: `accum`
is what a voxel gathered *this frame* and is written by atomics from the shading
path, `resolved` is what it has concluded *across* frames and is the only half a
path reads. A resolve pass (`src/shaders/optix/sharc_resolve.cu`) runs once per
frame between launches and merges the first into the second under a bounded
temporal window, ages entries nobody visited, and hands back the slots of ones
that have gone stale. The entry grew from 20 bytes to 32.

**Why it mattered, and the measurement that says so.** The old host code cleared
the whole table whenever accumulation restarted. Accumulation restarts on every
camera movement, so in the editor the table was wiped several times a second and
never held more than one frame of anything.

With the resolve pass the cache demonstrably shortens paths, which it could not
be shown to do before. `DebugMode::eSharcBounces` -- the bounce-count heatmap,
green for one indirect bounce and red for two or more -- is the direct
measurement:

| scene | red (deep) off -> on | green (one bounce) off -> on |
|---|---|---|
| cornell_box, 512x384, 64 spp, depth 8 | 0.466 -> 0.068 | 0.312 -> 0.624 |
| pine_scene, 1280x720, 120 spp, depth 8 | 0.488 -> 0.349 | 0.537 -> 0.628 |

and it costs nothing in accuracy: ratio **0.9997** against the same render with
the cache off on pine at depth 8 and 16, 0.9925 on the Cornell box.

**It still does not make the pine forest faster, and that is not a contradiction.**
400 spp at 1280x720, `texture_downscale` 2, three runs each, wall clock:

| depth | cache off | cache on |
|---|---|---|
| 4 | 5.5 / 5.7 / 5.6 s | 5.7 / 6.0 / 6.1 s |
| 8 | 6.2 / 5.9 / 5.7 s | 5.8 / 5.8 / 5.8 s |
| 16 | 5.7 / 5.7 / 5.7 s | 5.8 / 5.8 / 5.8 s |

Neutral at depth 8 and 16, and about 5% *worse* at depth 4, where there is no
path left to cut and the cache is pure overhead. This is the same conclusion
`docs/open-perf.md` already recorded, and the reason is in that file rather than
in this one: pine is not bound by path length. Its stall profile is
`long_scoreboard` 38.1 cycles per issued instruction with the SM at 6-12%, so
removing traversal work removes something the frame was not waiting on.

**Do not benchmark this on short runs.** An idle 4090 sits at 270 MHz against a
3105 MHz maximum, and a 60-spp render is under a second of GPU work -- not long
enough for the clocks to come up. The same configuration measured 8.8 to 14.4
ms/sample run to run that way, which is wide enough to "show" any result wanted.
Every number above is 400 spp with a warm-up run discarded.

**What Metal has to gain to be a port again**, in the order it matters:

1. The split entry and the resolve dispatch. Metal is a wavefront, so this is
   one more kernel in the per-frame sequence rather than a call after the
   launch -- but the arithmetic is already shared and already tested:
   `oka::sharc::resolveEntry` in `src/shaders/optix/sharc_grid.h` is a pure
   function of one entry, compiles on the host, and has the cases in
   `tests/render/test_sharc_grid.cpp`. Metal needs to call it, not rewrite it.
2. Stop clearing on accumulation restart. This is the whole point; without (1)
   it cannot be done, because nothing else ages the entries the camera left.
3. The four debug views. `DebugMode::eSharcGrid`, `eSharcRadiance`,
   `eSharcOccupancy` and `eSharcBounces` are already in Metal's `ShaderTypes.h`
   so the two enums stay one numbering and the editor keeps one menu; Metal
   currently renders normally when one is selected. `src/shaders/optix/sharc.h`
   (`sharcDebugColour`, `sharcDebugOccupancy`) and the raygen's heatmap are the
   reference.
4. Rounding in `encode`. Older, and still not done on Metal: it truncates where
   OptiX rounds, which is a quarter of a percent of systematic darkening for as
   long as the cache is on. Measured -- `00_calibration` and `02_basecolor` both
   read 0.997 truncating and 1.000 rounding.
5. The 64-bit key, and then reprojection and responsive lighting on top of it.
   The key layout, the adjacent-level walk and the blend are all in
   `sharc_grid.h` and all tested on the host, so this is the same kind of port
   as (1): call the arithmetic, do not rewrite it. Metal's entry is still the
   20-byte one, so this is where the two backends are furthest apart.

**Reprojection and responsive lighting are on OptiX too**, and both needed the
key to stop being a checksum first. It is now the SDK's layout -- 17 bits per
axis, 9 for the level, 3 for the normal bucket, one flag -- which costs four
bytes an entry (32 -> 40) and buys two things a hash cannot give:

* **Adjacent-level reprojection.** The level under a point follows its distance
  to the eye, so moving the eye re-quantises a world that has not moved: the
  point lands in a different voxel and its new entry starts from nothing while
  everything the cache knew sits one level away, waiting to be evicted unread.
  The resolve pass now decodes an entry's voxel, works out which way the level
  moved, and blends the neighbour in. Port of `SharcGetAdjacentLevelHashKey` and
  the `SHARC_BLEND_ADJACENT_LEVELS` arm of `SharcResolveEntry`.
* **Responsive lighting.** A light marked `responsive` (light panel, or
  `"responsive": true` in the `<stem>_light.json` sidecar) has its contribution
  cached in a second entry per voxel on a much shorter window, so it can change
  faster than the rest of the cache follows.

**Where the responsive port departs from the SDK, and why.** The SDK splits by
*component* at each vertex -- direct lighting from a responsive light goes to the
responsive entry, everything else to the main one -- and keeps the two entries
adjacent so a six-bit index offset can be packed into the path state it carries
between vertices. This backend has no such state: the deposit happens once, at
the end of the path, from a per-pixel record. So:

* The split is made on the path. The record accumulates how much of what the
  path gathered came from responsive lights, and the deposit subtracts it. Still
  an additive split, which is what makes a reader's `main + responsive` correct;
  splitting by *path* instead would make the two means overlap and double-count.
* Both entries are claimed together and deposited into together, including when
  the responsive half receives zero, so the two means are over the same paths.
* Consequently the responsive entry never needs its neighbour's sample count,
  and the SDK's rule that Resolve must not clear accumulation under responsive
  lighting -- with the host clearing it before every update instead -- does not
  apply. One pass fewer.
* The responsive entry is hashed independently rather than kept adjacent, so a
  query costs a second probe. In exchange the two do not compete for the same
  eight slots, and no index offsets need packing.

**Two defects found by looking at the isometric bathroom in the editor**, both
reported as "the mirror got worse" and "the cache buys nothing", and both real:

* **A surface reached by a delta bounce was being answered for by the cache.**
  `prd->depth` counts segments, not scattering events, so a mirror at depth 0
  put what it reflects at depth 1 -- where the gate let the cache answer. What
  the viewer saw *in* the mirror, and anything behind the shower glass (a delta
  transmission is specular too), was replaced by a voxel average. On the bath
  water this is unmistakable once seen: rectangular blocks of constant colour
  through the surface. The gate now also requires `!prd->specularBounce`. This
  is the SDK's own rule and its own listed mistake -- a replaced primary surface
  is still a primary surface -- and it was simply not implemented.

* **The cache put a floor under the render's convergence.** Its error is one
  value per voxel, held across a temporal window, so it is correlated in space
  *and* time and does not average away with samples. Plain path tracing keeps
  converging past it. Measured on iso_bathroom at 960x540, depth 8, against a
  4096-spp reference of the same backend, taking the residual that survives a
  9-pixel blur so the number is structured error rather than noise:

  | spp | no cache | cache | + mirror fix | + read cutoff |
  |---|---|---|---|---|
  | 16 | 0.154 | 0.098 | 0.108 | 0.108 |
  | 64 | 0.063 | 0.039 | 0.042 | 0.042 |
  | 256 | 0.022 | 0.019 | 0.020 | 0.019 |
  | 1024 | **0.0064** | **0.0127** | 0.0123 | **0.0064** |

  Below the crossover the cache is the better image as well as the faster one;
  above it the cache is the only thing between the render and convergence. So
  reads now stop after `render/pt/sharcReadFrames` accumulated samples (default
  128) while deposits carry on, which leaves the table warm for the next camera
  movement. At 1024 spp that restores the no-cache structured error exactly, and
  it changes nothing at 16 and 64 spp.

  This is not a defect in the port -- it is what a world-space cache is. SHaRC
  is a real-time technique, aimed at one sample per pixel per frame with a
  denoiser downstream; a renderer that accumulates a still has a regime where it
  helps and a regime where it does not, and the cutoff is where they meet.

* **The cutoff then disabled the cache in the one place it matters most.** With
  accumulation off the renderer reports `mSubframeIndex` as the whole sample
  budget rather than zero -- deliberately, because a counter that reset every
  frame hung StrelkaCLI on the single-hit debug views -- so `subframe_index` is
  256 from the second frame on and `subframe_index < 128` was never true. The
  cache was never read. The limit now applies only while the film is
  accumulating, which is also the only situation its reasoning covers: with
  accumulation off every frame is a fresh one-sample image, nothing is
  converging past the cache, and the cache is the entire reason the frame is not
  black.

**How large the effect is, and where it is not.** The reference is NVIDIA's own
banner: one sample per pixel, black room without the cache, lit room with it.
That is not what this scene does, and the reason is the scene rather than the
port.

iso_bathroom, 960x540, depth 8, one sample against a warm cache, graded against
a converged reference at the same pose:

| | rel |
|---|---|
| one sample, no cache | 0.9587 |
| one sample, warm cache | 0.9056 |

Six percent. Both images are essentially noise at one sample.

**Why it is only that much is still open**, and three explanations have been
measured and rejected, so nobody need try them again:

* *The cached values are undersampled.* No -- raising the samples a voxel must
  carry before a path will trust it makes the image **worse** (rel 0.9056 at 8,
  0.9174 at 64, 0.9489 at 512), because fewer voxels qualify. The cache is
  already giving what it has.
* *The scene's light is direct, so next-event estimation gets there first and
  leaves the cache nothing.* No -- with `estimator_mode = 1`, which removes
  next-event estimation entirely and forces every path to find the light by
  sampling, the cache is worth 19% against 17% with it. Barely moved.
* *The update paths pollute the image.* One path in eight never reads and traces
  to the end, and unlike the SDK -- whose update pass is a separate launch that
  never touches the film -- ours are part of the render. Raising the share to one
  in sixty-four moved 17% to 18%.

What is established is the size: about 18% less error at 16-32 samples, 36% less
structured error, and 12% off the frame. Those are worth having and they are not
the banner.

Where the cache does pay on this scene: 21% less error and 36% less structured
error at 16 samples, 18% at 32, and 12% off the frame time. Those are worth
having and they are not an order of magnitude.

> An earlier revision of this file claimed 88% here. That figure was wrong: the
> harness that produced it reset the film after the warm-up but left the loop's
> own bound counting from the restart, so the "one sample" render with the cache
> was really 201 samples and the one without it really was one. Corrected above.

**The eligibility test decides how much of this is reachable at all.** The port
originally gated reads on `roughness > 0.3` at the surface being *hit*, which is
the wrong surface -- what matters is the lobe that launched the segment -- and
far too blunt. Share of paths the cache terminates after a single bounce, one
sample, warm cache:

| | 1 bounce | 3+ bounces |
|---|---|---|
| no cache | 44.9% | 37.7% |
| `roughness > 0.3` on the hit | 54.9% | 26.3% |
| no gate at all | 75.9% | 13.8% |
| SDK segment-length + footprint test | 57.7% | 30.7% |

The gate is what keeps a reflection sharp, so removing it is not on the table --
that is what put voxel blocks through the bath water. The SDK's test
(`oka::sharc::mayReadCache`, AGENTS.md 14.8) asks the right question instead: is
the segment longer than a voxel's diagonal, and had the lobe that launched it
already spread wider than a voxel by the time it arrived. It recovers a little
of the gap at the same image quality, and the rest of the gap is paths that
genuinely must not read.

To see the cache in the regime it is built for, turn off *Accumulate while
still* so every frame is one sample again -- and expect the size of effect above
rather than the banner's.

**Three more things a correctness pass over the cache turned up**, all fixed:

* **The segment length was measured from the wrong end.** The eligibility test
  used `optixGetRayTmax()`, which is the distance from wherever the ray was last
  *restarted* -- a cutout, a medium boundary -- not from the vertex that
  scattered. `prd->misDistance` already carries the difference, for the MIS
  weight, for the same reason. Under-measuring fails both halves of the test, so
  a scene with alpha cutouts or glass in front of things refused the cache on
  paths that qualify. No measurable change on the bathroom; it will matter on the
  pine forest.
* **A scattering event inside a medium inherited the lobe that entered it.**
  Entering water or glass is a delta transmission, roughness zero, so every
  vertex inside the medium was refused. A phase function, and the cosine lobe off
  a subsurface exit, are as broad as a lobe gets and are now recorded as such.
  Worth 16.8% -> 17.2% on the bathroom.
* **The overflow guard had no headroom for its own race.** `kMaxCount` sat
  exactly at the arithmetic bound, and the guard is a plain read followed by four
  atomic adds -- so every thread already past the read still deposits, which on
  this hardware is a hundred thousand or so. Halved, which costs nothing: the
  accumulator is cleared every frame anyway.

**And it does pay for itself, where the regime is right.** iso_bathroom,
1920x1080, depth 8, 1500 spp, warm clocks, two runs each: **12.6 / 12.7 s
without the cache against 11.1 / 11.2 s with it**, i.e. 12%. In the editor that
is 12% off every frame you are moving the camera through, since accumulation
restarts on each movement and the render is therefore always below the cutoff.

Two more things that came out of measuring the responsive port, both worth
keeping in mind before changing any of this:

* **The two halves must have the same lifetime.** The SDK uses its responsive
  frame count for the eviction threshold as well as the window; here that read
  0.68% dark (ratio 0.9932 where the split should be exact), because a voxel
  visited intermittently lost its responsive half while the main half survived,
  and every read of it then returned `total - responsive` alone. The window is
  short; the lifetime is the main one's.
* **A find must not stop at the first empty slot.** That is the ordinary
  open-addressing shortcut and it was correct until eviction existed -- eviction
  frees a slot in the middle of a probe run, and everything the run reaches past
  that hole becomes invisible. For the main lookup that is a hit-rate loss; for
  the responsive half it is the same 0.68% bias, because failing to find one half
  of an additive split does not return "not cached". Both fixed: ratio 0.9999.

**What is still not ported, on either backend**, so nobody goes looking: the
resolve pass's linear-probe re-find, which needs an entry's neighbours and so
cannot live in a pure function of one entry; SH-directional encoding
(`SHARC_ENABLE_SH_ENCODING`) and material demodulation, which are SDK
compile-time options neither backend sets; and cache resampling during update
(`SHARC_ENABLE_CACHE_RESAMPLING`), which belongs to the SDK's separate update
pass and has no counterpart in a single-launch integrator.

---

## 13. Adobe's OpenPBR carries 191 functions with no execution space, and nvcc has not seen them

`third_party/openpbr_bsdf` is Adobe's OpenPBR 1.1.1, vendored unmodified. It
advertises a CUDA backend (`interop/openpbr_interop_cuda.h`, selected on
`__CUDACC__`), and the Metal and host C++ backends are both confirmed working:
`wavefront.metal`'s flags compile it with zero errors and zero warnings, and
`tests/material/test_openpbr_*.cpp` run it on the host through
`openpbr/openpbr_bridge.h`.

What is not confirmed is CUDA, and there is a specific reason to doubt it rather
than assume it. Counted in the tree as vendored:

    191   function definitions with no linkage or execution-space macro at all,
          e.g. `float openpbr_average_fresnel(const float eta) {...}` at file
          scope (impl/openpbr_lobe_utils.h:211)
     87   function definitions carrying OPENPBR_INLINE_FUNCTION, which the CUDA
          interop layer defines as `__device__ inline`

A bare definition is a `__host__` function to nvcc, and calling one from
`__device__` code is an error, not a warning. The 191 are called from the 87. If
that reading is right, `openpbr.h` cannot be compiled by nvcc as shipped, and the
OptiX half of OpenPBR support needs one of:

  - NVRTC with `-default-device`, which changes how `src/shaders/CMakeLists.txt`
    builds the OptiX IR (it uses `cuda_wrap_srcs`, i.e. nvcc), or
  - a build-time transform that prefixes those definitions with
    `__device__ inline`, or
  - an upstream fix; the repository is active (last commit 2026-08-11).

The same property has already bitten the host build, which is what makes it worth
writing down rather than guessing at: two host translation units that both
include `openpbr.h` fail to link with ~200 duplicate symbols. That is handled --
`openpbr/openpbr_shim.h` puts the library in an anonymous namespace on the host
only -- but the fix does not transfer, because a shader module wants those
definitions to keep external linkage.

This is unresolved only because there is no CUDA on this machine: `nvcc` is
absent, `hix.local` does not resolve from this network, and the Slurm session at
`login-bia.nvidia.com` needs an MFA login that has expired. It is a ten-minute
question for anyone with a CUDA toolkit -- compile one `.cu` that includes
`<strelka/material/openpbr/openpbr_bridge.h>` and calls `openpbr_prepare_at`.

## 14. Two runs of one binary do not agree, when OpenPBR and subsurface are both on

Rewritten after re-measuring it. The previous entry named the wrong witness and
the wrong subsystem, and its advice -- grade `25_subsurface` on a threshold
rather than on equality -- cost sharpness for nothing.

**`25_subsurface` is deterministic.** Seven runs on HEAD and five on a fresh
build of `e5684bd`, the very commit the old entry cited, are all bit-identical --
`ac667285835f3a04904db0d15b224b12` on *both* builds. So neither the row nor the
OpenPBR work moves it, and an exact-match regression on it is valid.

**The real witness is the Open Chess Set with OpenPBR.** 960x540, 512 samples:
eight of ten runs differ from the first. What differs is astonishingly small and
fixed -- exactly two pixels ever change, always the same two, (329,482) and
(336,494), each toggling between two values 1.53e-02 and 3.12e-02 apart. That is
0.0002% of the components: two paths out of 265 million path-samples, each
binary rather than drifting.

Reproduce it far more strongly with constant parameters and no MaterialX at all:
the chess `.glb`, an `_openpbr.json` naming both materials with
`subsurface_weight` 1 and `subsurface_radius` 0.02, and the `.mtlx` moved aside.
Six runs, six different images.

### What it takes, and what it does not

It needs **OpenPBR and subsurface together**. Either alone is stable:

| configuration | runs | distinct |
|---|---|---|
| glTF subsurface, no OpenPBR (`25_subsurface`) | 12 | 1 |
| glTF subsurface, no OpenPBR (chess, `.mtlx` moved aside) | 10 | 1 |
| OpenPBR, no subsurface (sidecar, constant parameters) | 6 | 1 |
| OpenPBR **and** subsurface (sidecar, radius 0.02) | 6 | **6** |

Ruled out, each by building it and running the six:

- **The walk length.** `subsurface_iterations = 0` still diverges.
- **MaterialX textures.** The constant-parameter sidecar diverges harder than
  the textured document does.
- **Launch batching.** `spp_per_launch = 512`, one launch for the whole render,
  still diverges.
- **`openpbr_interior_volume`.** Replacing the OpenPBR medium's extinction,
  albedo and anisotropy with constants: still diverges.
- **The medium-properties branch** as a whole, routed through the glTF path.
- **The entry event.** Entering on `DIFFUSE_TRANSMISSION` like the glTF path
  instead of `TRANSMISSION`: still diverges.
- **The missing miss/shade barrier**, the one place in the encoder where an
  absent barrier is argued for in a comment. Making it unconditional: still
  diverges.
- **Out-of-range energy-table reads.** `OPENPBR_ASSERT` expands to nothing on
  Metal, so the library's "index must be clamped" preconditions are unchecked
  there and a violation would be an out-of-bounds constant read -- which would
  explain a value that varies between runs. It does not happen: 44,800 calls of
  `openpbr_prepare`/`sample`/`eval`/`pdf` through the bridge, compiled with
  asserts live, across degenerate roughness, anisotropy, IOR and grazing angles,
  trip nothing.

### Where that leaves it

Neither feature's own code explains it and no shared buffer has been caught. The
remaining reading is that the combination changes the shade kernel's register
pressure and occupancy enough to expose a race that neither variant's scheduling
reaches on its own -- which would make it a pre-existing defect in shared code
that OpenPBR only reveals. That is a hypothesis, not a finding: it has not been
localised to a buffer or a stage.

Worth what it costs, in the meantime: the effect is two pixels out of half a
million, so it changes no grade and no image anyone looks at. What it costs is
the method -- an exact-match regression on a scene with both features on will
report a change that did not happen. Grade those on `rel` against a threshold.
Every other scene, including `25_subsurface`, can be graded on equality.

## 16. Subsurface transmits far too much straight through a body

`25_subsurface` and `29_subsurface_skin` both **pass with subsurface switched
off** -- strip the extension, render the same base colours as plain Lambertian
diffuse, and the control grades rel 0.060 against the Cycles subsurface
reference, against 0.057 for the real walk. Both are optically thick, and the van
de Hulst mapping is defined to make a thick medium reproduce a chosen diffuse
albedo, so those rows measure the mapping and not the transport.

`30_subsurface_translucent` and `31_subsurface_absorbing`
(`tools/feature_tests/sss_regimes.py`) are the rows that do measure it, and they
fail: **0.0966 / 1.027** and **0.2294 / 1.112**.

### The disagreement is in the paths that never scatter

`31_subsurface_absorbing` is the sharp instrument. Same body and mean free path
as row 30, but a base colour of 0.05, so a path either crosses the body without a
single collision or is absorbed. It splits the frame cleanly:

| | ratio |
|---|---|
| lit half | **1.030** |
| shadowed half | **3.304** |

What comes back out of the lit side is right. What crosses to the far side is
three times too much. On row 30's scattering medium the same split reads 0.92
lit and 1.34 shadowed, and the error grows as the body gets optically thinner:
at mean free paths of 0.05, 0.15, 0.40 and 0.80 against a sphere of radius 0.48
the frame ratio runs 0.997, 1.007, 1.048, 1.093 and the shadowed half 1.06, 1.10,
1.26, 1.49.

`subsurface_iterations` places it exactly. At **zero** -- no scattering event
permitted at all -- the shadowed half of the thinnest row already reads 1.445,
and allowing the full 64 steps only takes it to 1.485. The whole excess is in the
zero-collision path: enter, cross, leave.

### Ruled out, each by measurement

- **The colour to medium mapping.** The thick rows are flat per channel: 25 reads
  R 0.979 / G 0.998 / B 1.015 and 29 reads 0.989 / 1.005 / 1.002. Cycles uses van
  de Hulst here too -- confirmed in `intern/cycles/kernel/integrator/subsurface_random_walk.h`.
- **A radius or extinction convention.** No single scale reconciles the two
  halves. On the absorbing row, scaling the radius by 0.3 brings the shadowed
  half from 3.30 to 1.11 and simultaneously drives the lit half from 1.03 to
  0.54. Cycles computes `*sigma_t = reciprocal(max(radius, 1e-16))` -- the same
  reciprocal we do.
- **Next-event estimation and MIS.** `render.estimator_mode` 0 and 1 agree to
  three digits on the failing row: 1.093 against 1.090, with the shadowed half at
  1.487 and 1.483. Two independent unbiased estimators agreeing means the
  transported quantity is what differs.
- **Walk length.** See the `subsurface_iterations` sweep above; and
  `29_subsurface_skin` shows 64 and 256 steps give the same answer.
- **Cycles rejecting zero-scatter paths.** It does not: a path that reaches the
  boundary with no collisions exits and contributes, weighted
  `transmittance / dot(channel_pdf, pdf)`, which is the weight
  `sssBoundaryWeight()` computes.
- **Cycles' Dwivedi guiding.** Present -- it measures the opposite interface on
  the first bounce and biases later scattering toward it -- but it is
  variance reduction under MIS, so it cannot move the mean.

### Line by line against Cycles

`intern/cycles/kernel/integrator/subsurface_random_walk.h` and `subsurface.h`,
read against `wavefront.metal`'s medium path and `src/shaders/metal/subsurface.h`.

**Identical, and now confirmed rather than assumed:**

| quantity | Cycles | here |
|---|---|---|
| colour to single-scattering albedo | `subsurface_random_walk_remap()`, van de Hulst | `patch_subsurface`'s inversion of the same fit |
| extinction | `*sigma_t = reciprocal(max(radius, 1e-16))` | `sssSigmaT()` |
| channel choice | `volume_sample_channel(alpha, throughput, ...)` | `sssChannelPdf(throughput, albedo)` |
| free flight | `t = -logf(1.0f - randt) / sample_sigma_t` | `sssSampleDistance()` |
| weight at a scattering event | `sigma_s * transmittance / dot(channel_pdf, pdf)` | `sssScatterWeight()` |
| weight at the boundary | `transmittance / dot(channel_pdf, pdf)` | `sssBoundaryWeight()` |
| entry direction | cosine about `-N` from the BSSRDF closure | the diffuse-transmission lobe |
| exit | `bsdf_diffuse_setup(sd, N, weight)` | cosine lobe about the exit normal |
| a zero-collision path | exits and contributes | the same |

The mapping check is worth stating numerically because it had never been done:
on `31_subsurface_absorbing`'s base colour of 0.05, our inversion produces
0.218484 and Cycles' `subsurface_random_walk_remap` produces 0.218489.

**Present in Cycles and absent here:**

1. **An albedo floor.** `subsurface_random_walk_coefficients()` ends with

       const float min_alpha = 0.2f;
       if (alpha[i] < min_alpha) { throughput[i] *= alpha[i] / min_alpha; alpha[i] = min_alpha; }

   The walk is run at a higher albedo than authored and the entry throughput is
   scaled by the ratio. Note this is exact for one scattering event and not for
   n, so it is a deliberate approximation on Cycles' side, and it changes the
   weight a zero-collision path carries from 1 to `alpha/0.2`. It does **not**
   explain this row -- 0.2185 is above the floor, so it does not fire -- but any
   darker medium will diverge for this reason alone.

2. **Dwivedi guided sampling.** `guided_fraction`, `sample_phase_dwivedi()`,
   `forward_stretching` / `backward_stretching` applied to `sample_sigma_t`, and
   a three-way MIS between classic, forward-guided and backward-guided across
   three channels. Variance reduction under MIS, so it cannot move the mean, and
   its absence is a noise cost rather than a bias.

3. **The opposite-interface probe.** On the first bounce Cycles traces
   `ray.tmax = max(t, 10.0f / reduce_min(sigma_t))` -- further than the sampled
   distance -- records `opposite_distance`, and only then decides
   `hit = ray.tmax < t`. The extra reach exists to feed the guiding above; the
   decision itself is the same test this kernel makes.

4. **`*throughput = safe_divide_color(*throughput, albedo)`** at walk entry. We
   have the equivalent in `sssEntryTint`, divided out at the same place.

### The slab, and what it says instead

`tools/feature_tests/sss_slab.py` builds five flat plates of known thickness in
the same nearly-absorbing medium, lit from above with the top face held at a
fixed height so every row receives identical irradiance, and viewed from below.
Everything the measurement does not control -- the entry and exit lobes, the
light, the solid angle, the camera -- is common to all five, so it cancels in the
ratio of two thicknesses and

    T(d1) / T(d2) = exp(-sigma_t * (d1 - d2))

is exact. **This grades the feature with no reference at all**, which is what the
sphere could not do.

At a mean free path of 0.2, so a nominal `sigma_t` of 5:

| pair | analytic | Cycles | here | sigma Cycles | sigma here |
|---|---|---|---|---|---|
| 0.05 -> 0.10 | 0.7788 | 0.7919 | 0.7037 | 4.67 | **7.03** |
| 0.10 -> 0.20 | 0.6065 | 0.6228 | 0.5185 | 4.74 | **6.57** |
| 0.20 -> 0.40 | 0.3679 | 0.3819 | 0.2922 | 4.81 | **6.15** |
| 0.40 -> 0.80 | 0.1353 | 0.1418 | 0.0962 | 4.88 | **5.85** |

Cycles brackets Beer-Lambert from below by 2 to 7%, which is what the small
scattering component at this albedo should do, and converges toward it as the
slab thickens. **We attenuate too much**, by 17 to 40%, and our effective
extinction is not even constant -- it falls with thickness, so the transmitted
profile is not an exponential at all.

`subsurface_iterations = 0` reproduces the slab numbers to four digits, so no
scattering event is involved: the defect is entirely in the free flight and the
boundary weight.

This **inverts** the reading the sphere gave. On `31_subsurface_absorbing` the
shadowed half is 3.3 times too bright; on a slab the transmitted light is too
dark. Both can be true -- the sphere's shadowed half is fed by paths around the
limb rather than through the body -- but it means "carries light too far" was the
wrong description, and any fix has to be graded on the slab, where the path
length is a number rather than a distribution.

### Cause: the entry direction

Not the free flight, not the ray bound, not the exit weight. The sampler was
eliminated first -- `pcg`, `sobol` and `hybrid` give the same effective
extinction to two decimals, and `halton` to within 1% -- so the exponential draw
is sound. What is left is the *length* of the path being attenuated, and that is
set by the direction the walk is entered on.

This walk enters along the glTF diffuse-transmission lobe: a cosine hemisphere
about `-N`. A path entering at angle `theta` then crosses a slab of thickness `d`
along `d / cos(theta)`, so the transmitted fraction is not `exp(-tau)` but

    T(tau) = 2 * E3(tau) = 2 * integral_0^1 mu * exp(-tau / mu) dmu

which decays faster and whose logarithmic slope falls with thickness. That is the
measured shape, and it is not close:

| pair | pure exponential | cosine entry | measured here | Cycles |
|---|---|---|---|---|
| 0.10 -> 0.20 | 5.00 | 7.03 | **6.57** | 4.74 |
| 0.20 -> 0.40 | 5.00 | 6.46 | **6.15** | 4.81 |
| 0.40 -> 0.80 | 5.00 | 5.97 | **5.85** | 4.88 |

Ours tracks the cosine prediction and sits a few percent under it, which is the
in-scattering this albedo still has. Cycles tracks the pure exponential and sits
a few percent under *that*, for the same reason.

**Cycles does not enter diffusely.** `subsurface_entry_bounce()` in
`intern/cycles/kernel/integrator/subsurface.h` refracts through a rough
dielectric interface:

    const float3 local_H = microfacet_ggx_sample_vndf(local_I, alpha, alpha, rand_bsdf);
    const float3 H = to_global(local_H, X, Y, Z);
    const float cos_HI = dot(H, sd->wi);
    const float arg = 1.0f - (sqr(neta) * (1.0f - sqr(cos_HI)));
    const float dnp = max(sqrtf(arg), 1e-7f);
    const float nK = (neta * cos_HI) - dnp;
    *wo = -(neta * sd->wi) + (nK * H);

Snell about a GGX microfacet normal, with `neta = 1 / bssrdf->ior`. At a low
interface roughness that is a narrow cone about the refracted view direction, so
the path length through a slab is close to `d` and the profile is exponential.
A cosine hemisphere is used only by `CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID`, and
even there on only half the draws -- the other half refracts.

This single difference explains both symptoms that looked contradictory. A cosine
entry takes longer chords, which is the slab reading too dark, and spreads
laterally far more, which is `31_subsurface_absorbing`'s shadowed half reading
3.3x too bright: that half is fed by paths that wrap the limb rather than cross
the body.

### Fixed, and what it left behind

`subsurface_entry_direction()` in `material/microfacet.h` refracts Snell about the
shading normal at a fixed index of 1.4, and both backends take it at the point
they enter the medium. The lobe still decides *whether* the walk is entered and
still supplies the weight -- cosine-sampled against a cosine density, so the two
cancel and one is left after the tint, which is the weight Cycles' entry carries
too. Only the direction changes.

Smooth rather than through a GGX microfacet. Cycles refracts about a sampled half
vector, but refraction alone compresses the cone -- at 1.4 even a grazing ray
bends to 45.6 degrees -- and the slab says a smooth interface reproduces its
exponential. 1.4 is not a parameter: the extension does not carry one, and it is
Cycles' skin default.

**The slab is now correct**, and it is graded against algebra rather than against
the reference:

| pair | analytic | Cycles | here | sigma Cycles | sigma here |
|---|---|---|---|---|---|
| 0.05 -> 0.10 | 0.7788 | 0.7919 | 0.7930 | 4.67 | 4.64 |
| 0.10 -> 0.20 | 0.6065 | 0.6228 | 0.6242 | 4.74 | 4.71 |
| 0.20 -> 0.40 | 0.3679 | 0.3819 | 0.3839 | 4.81 | 4.79 |
| 0.40 -> 0.80 | 0.1353 | 0.1418 | 0.1431 | 4.88 | 4.86 |

Both track the pure exponential from below by the in-scattering this albedo still
has, and they agree with each other to 0.03 in every row. In absolute terms the
slab reads 1.002 to 1.020 against Cycles where it read 0.852 to 0.327.

The sphere rows improve where they measure transport and go slightly dark
everywhere:

| row | before | after |
|---|---|---|
| `25_subsurface` | 0.0479 / 0.997 | 0.0501 / 0.976 |
| `29_subsurface_skin` | 0.0496 / 0.996 | 0.0531 / 0.972 |
| `30_subsurface_translucent` | 0.0966 / 1.027 | 0.0780 / 0.969 |
| `31_subsurface_absorbing` | 0.2294 / 1.112 | 0.1337 / 0.896 |

31's `rel` nearly halves and 30's falls by a fifth, which is the transport being
right. What is left is a deficit of about 2.4 to 3% on the two thick rows -- and
those are the rows that cannot tell subsurface from diffuse, so they were reading
0.997 with a wrong entry, which is the cancellation pattern of entry 15 again.

2.8% is the normal-incidence Fresnel reflectance of a 1.4 interface, `((1-1.4) /
(1+1.4))^2`. That is a coincidence worth testing rather than a conclusion: the
entry weight is one on both sides, so neither renderer is taking a Fresnel share
out, and if the number is real it is on the exit rather than the entry. Nothing
here has measured it yet.

Grade the remainder on the four subsurface rows returning to ratio ~1.00 without
moving `sss_slab_read.py`'s `sigma here` column off Cycles'.

## Closed (kept for the measurement, not the work)

### The diffuse lobe was summed with the specular one instead of layered under it

`tools/feature_tests/README.md` recorded `00_calibration` at 0.021 / 1.010; on
HEAD it renders 0.0337 / 1.033, and every other row moves with it. Bisected, and
then the bisect turned out to be measuring the removal of a second error rather
than the arrival of a first.

### The recorded 1.010 was two errors cancelling

`git bisect run` over `cf9c951..3f00e2f`, grading `00_calibration`'s frame mean
against its checked-in reference, names `e17db4b` ("Fix the MIS estimate, the
light densities, and the IES path"): 1.0098 before, 1.0332 after. Running the
estimator A/B on both builds says what actually happened:

| build | estimator | frame | sphere | back wall |
|---|---|---|---|---|
| `e17db4b^` | NEE + MIS | 1.0098 | **0.9554** | 1.0045 |
| `e17db4b^` | BSDF only | 1.0331 | 1.0087 | 1.0357 |
| HEAD | either | 1.0332 | 1.0093 | 1.0363 |

Before that commit the two estimators disagreed by 3%, which is the disagreement
it set out to close and did. What it closed was a next-event deficit on the
sphere -- 0.955 to 1.009 -- and the frame mean had been averaging that deficit
against a stage that was already 3.5% bright. `e17db4b` is not a regression: it
removed the half that was cancelling, and the recorded 1.010 was never a correct
render of anything.

### What is left is the stage, and it predates all of this

The stage reads 1.0357 under BSDF sampling alone on the *old* build, so the
excess is older than the commit the bisect named. Two things localise it:

- Only surfaces with a specular lobe carry it. `00_calibration`'s sphere is
  authored `KHR_materials_specular: {specularFactor: 0}` and matches at 1.009;
  the stage takes the default and reads 1.036 on the wall, 1.025 on the floor.
  The same split holds on every row: `02_basecolor` spheres 1.004 and
  `04_metal` spheres 1.008 against their stages at 1.02-1.03, while
  `03_roughness`, whose subjects are themselves rough dielectrics, runs 1.036 on
  the spheres too.
- It is per bounce. Splitting each surface by how much light it receives, the
  directly lit fifth reads 1.005 on the sphere and 1.037 on the floor, and the
  dimmest fifth -- which is almost entirely inter-reflection -- reads 1.044 to
  1.058 everywhere.

### The cause, measured on the host

`tests/material/test_standard_pbr_furnace.cpp` integrates `bsdf * cos` over the
hemisphere for a white base with no metal. With the specular weight at zero the
directional albedo is exactly 1.000 head on. With it at one:

| roughness | cos(V) 1.0 | 0.7 | 0.3 |
|---|---|---|---|
| 0.85 | 1.042 | 1.047 | 1.069 |
| 0.50 | 1.074 | 1.076 | 1.129 |
| 0.20 | 1.080 | 1.082 | 1.227 |

`f_diffuse` is `albedo/pi` scaled only by the metallic, transmission and
diffuse-transmission complements. Nothing takes out what the specular lobe
reflects, so the two are summed. The coat has had this fixed for a while --
`clearcoat_base_scale()` takes `(1-F_L)(1-F_V)` out of the base and gives the
interreflection series back -- and the specular layer never got the same
treatment.

The material library itself is *not* what `e17db4b` changed: compiling
`bsdf_eval` and `bsdf_sample` from `e17db4b^` and from HEAD against the same
opaque dielectric gives identical sums to seven digits.

### The fix, and the one that did not work

Mirroring `clearcoat_base_scale()` with the specular lobe's `F0` was tried and
reverted. Schlick with `F0 = 0` is not zero -- its `(1-F0)(1-cos)^5` tail reaches
one at grazing -- so any `1 - F` complement takes energy from a material that has
no specular lobe at all: the furnace's `specular = 0` row falls from 1.000 to
0.976. The specular weight cannot be recovered from `F0` after `gltf_f0()` has
folded it in.

What it wants is the specular layer's *directional albedo*, and that turned out
not to need a fit. Integrating the lobe numerically over `F0` shows it is exactly

    E = (A * F0 + B) * (1 + F0 * t),    t = ggx_energy_term(roughness, NdotV)

-- the split-sum form times the factor `ggx_energy_compensation()` already
applies -- and that `A`, the coefficient of `F0`, is the single-scatter white
albedo `1 / (1 + t)`. At roughness 0.85 head on the measured `A` is 0.4855
against `1/(1+t)` = 0.4819, and the resulting albedo is 0.0419 against a measured
0.0422. `ggx_specular_albedo()` in `microfacet.h` is those two lines.

`B` is dropped deliberately: it is what the lobe reflects at `F0 = 0`, the
Schlick tail, and subtracting it is what broke the first attempt. It is zero at
normal incidence and is the grazing residual pinned separately in the furnace
test.

`specular_base_scale()` then scales the diffuse lobe by `1 - E`, and the
diffuse-transmission lobe by the same factor -- it sits under the same interface,
and scaling only the reflected half made a canopy grow brighter as it became more
translucent, which `test_diffuse_transmission.cpp` caught immediately.

**Every row moved, and all of them toward the reference:**

| row | before | after |
|---|---|---|
| `00_calibration` | 0.0337 / 1.033 | **0.0123 / 1.008** |
| `01_srgb_texture` | 0.0365 / 1.032 | 0.0213 / 1.014 |
| `02_basecolor` | 0.0291 / 1.026 | 0.0137 / 1.006 |
| `03_roughness` | 0.0380 / 1.035 | 0.0174 / 1.008 |
| `04_metal` | 0.0470 / 1.025 | 0.0374 / 1.007 |
| `05_anisotropy` | 0.0656 / 1.022 | 0.0588 / 1.004 |
| `06_normalmap` | 0.0922 / 1.089 | 0.0645 / 1.053 |
| `07_alpha_clip` | 0.0365 / 1.029 | 0.0192 / 1.002 |
| `08_alpha_blend` | 0.0345 / 1.030 | 0.0169 / 1.000 |
| `09_glass_ior` | 0.0707 / 1.031 | 0.0550 / 1.006 |
| `10_glass_absorption` | 0.0506 / 1.032 | 0.0334 / 1.007 |
| `11_emission` | 0.0174 / 1.008 | 0.0150 / 1.001 |
| `12_lights_punctual` | 0.0407 / 1.024 | 0.0253 / 0.996 |
| `13_uv2_vcol` | 0.0387 / 1.035 | 0.0184 / 1.009 |
| `25_subsurface` | 0.0568 / 1.014 | 0.0479 / 0.997 |
| `29_subsurface_skin` | 0.0591 / 1.013 | 0.0496 / 0.996 |
| `30_subsurface_translucent` | 0.1018 / 1.044 | 0.0966 / 1.027 |

`00_calibration` is now better than the 0.021 / 1.010 it was originally recorded
at, and -- the point of the whole entry -- its three regions agree with each
other: sphere 1.0037, back wall 1.0029, floor 1.0006, where they used to be
1.009 / 1.036 / 1.025. Five rows grade OK rather than CLOSE.

The material library is shared, so the OptiX backend takes the same change with
no edit of its own.

Still open, and now visible rather than buried under this: at grazing views a
white base climbs above unity even with the specular lobe switched off -- 1.016
at roughness 0.85, 1.161 at 0.20. That is single-scattering GGX/Smith masking
plus the Schlick tail, it is a different defect, and the furnace test pins it so
a fix to one cannot be credited with the other.

### A trap for whoever picks it up

Comparing an old build against HEAD on any scene with more than about four
instances does not work until the acceleration-structure fix in the Closed
section is cherry-picked onto it. Without it the old build is missing geometry --
on `03_roughness` the roughness-0.5 sphere renders at 0.44 of the reference
because it is absent -- and the missing bounce light moves the walls and floor
in the same direction as the thing being measured. `00_calibration` has four
instances and is clean, which is why the bisect used it and nothing else.


### The subsurface exit vertex shaded with the flat normal, and drew the tessellation

**Fixed.** `29_subsurface_skin` showed horizontal latitude banding on every
sphere -- plainly in the render and unmistakably in the 8x difference, where it
was the only structure in an otherwise noise-only image. The rings are the UV
sphere's 32 `ring_count` bands.

The exit from a subsurface walk took the *geometric* normal for four things: the
cosine lobe it leaves on, the shading normal it hands the light connection, that
connection's cosine, and the ray offsets. Only the last of those wants it. Every
other shading vertex in both kernels uses the interpolated normal, and that
normal was already in scope at the exit.

**Why it only shows on a dense medium, and the method mistake that hid it.**
Measured first on `25_subsurface`, at 2048 samples, the fix moved the image by
rel **0.0032** -- below the noise floor -- and was written off. That was the wrong
row. At a mean free path of 0.05 against a radius of 0.48 the walk exits over
roughly a tenth of the radius, which averages across many triangles and hides the
flat normal. At `29_subsurface_skin`'s 0.005 it exits within one triangle of where
it entered, and there is nothing to average. Same code, same tessellation, and one
row cannot see what the other shows at a glance.

The aggregate barely moves even on the row that shows it -- 0.0599 to 0.0591,
with the two images differing by rel 0.0085 -- because banding is structure at
low amplitude and `rel` is amplitude. The contact sheet is what graded this, not
the number.

Fixed in `wavefront.metal` and, behaviour-for-behaviour, in
`OptixRender_closest_hit.cu`. **The OptiX half is unverified**: there is no CUDA
on this machine (see entry 13). It is the same four-line change against the same
already-in-scope `surfaceHit.normal`.

Adobe ships `openpbr_volume_faceting_correction` in `third_party/openpbr_bsdf`
for a related but different problem -- the *distribution* of exit points follows
the geometry even when the normal does not. Not needed here, because this exit
applies its own explicit cosine lobe rather than relying on that distribution.


### Bottom-level structures were read while they were still being built

**Fixed.** Two of the five spheres in `25_subsurface` were missing -- not dark,
absent: the `render.debug = 1` normal buffer showed the wall and floor behind
them, so they were not in traversal at all. The row was at 0.2166 / 0.880 against
a README that records 0.056 / 1.009.

It is not a shading bug and not a loader bug. Giving all five spheres sphere 3's
material changed nothing; each sphere rendered correctly when it was the only one
in the scene; and the CPU-side bookkeeping is **byte-identical** between a run
that loses two structures and one that loses none -- same groups, same BLAS
indices, same geometry entries, same instance masks, same `gpuResourceID`s.

What named it was `STRELKA_AS_GROUP`, the batch size for bottom-level builds.
Which structures went missing was fixed for a given batch size and moved when the
batch size did: 1 lost two spheres, 2 and 3 lost none, 4 lost three, 6 lost one,
7 and 8 lost none. A knob that only decides how many builds share a command
buffer cannot change an image, so the fault was in the batching rather than in
anything the batches carry.

`Metal3AsPath` committed each batch to the queue with no dependency between them.
Metal schedules command buffers in commit order but does not make one wait for
the previous to *complete*, so the top-level build -- which reads every bottom
level its instance buffer names -- could run against structures still being
written. Confirmed by making each flush `waitUntilCompleted()`, which fixed it.

The fix chains the acceleration-structure command buffers on the shared event the
path already keeps, so the wait is on the GPU rather than a CPU round trip per
structure. The image no longer depends on the batch size, at any value from 1
to 8.

The Metal 4 path does not have this: `Metal4Context::beginImmediate()` reuses one
command buffer and `submitAndWait()` blocks, so its builds are already serial.

**It costs build time, and that is the trade.** On the pine forest (259
structures), the BLAS phase goes from a median of 1818 ms to 2236 ms and the
finish phase from 275 ms to 395 ms -- about +23% on that stage, some hundreds of
milliseconds of a multi-second load. The bottom levels are independent of each
other, so in principle only the top level needs to wait and the rest could still
overlap. That is not what this does, because a shared event carries one value:
if two builds may complete out of order, waiting on the highest value does not
imply the lower ones are done, and the cheap version would be unsound in exactly
the way the bug was. Recovering the time wants a fence per structure or a
separate event, and is worth doing only if load time on heavy scenes becomes the
complaint.

**Six rows were affected, not one.** Graded against Cycles, unfixed then fixed:

| row | rel before | rel after | ratio before | ratio after |
|---|---|---|---|---|
| `06_normalmap` | 0.1502 | 0.0922 | 0.898 | 1.089 |
| `09_glass_ior` | 0.1210 | 0.0707 | 1.026 | 1.031 |
| `10_glass_absorption` | 0.1114 | 0.0506 | 1.045 | 1.032 |
| `11_emission` | 0.1777 | 0.0174 | 0.826 | 1.008 |
| `13_uv2_vcol` | 0.1271 | 0.0387 | 1.050 | 1.035 |
| `25_subsurface` | 0.2166 | 0.0570 | 0.880 | 1.015 |

The other nine rows are bit-identical across the two builds, which is what a
synchronisation fix should look like: it changes the scenes where the hazard bit
and nothing else. Entry 15 above is why the "after" column still sits ~3% high.

The reason this surfaced as a subsurface defect is that `25_subsurface` was the
row hurt worst, and the walk was the obvious suspect. It was not the walk. Two
measurements ruled the walk out before the geometry was suspected: the missing
spheres did not move at all between `subsurface_iterations` of 16, 64 and 256 --
a truncated walk would have -- and they read exactly the backdrop behind them.

### Subsurface at skin mean free paths does not need a longer walk

**Measured, no change needed.** `tools/feature_tests/sss_regimes.py` adds row
`29_subsurface_skin`, which is `25_subsurface` at Blender's own skin preset -- Subsurface Radius (1, 0.2, 0.1)
at Subsurface Scale 0.005 -- which against the same 0.48 sphere is between one
hundred and one thousand mean free paths, and a single-scattering albedo that
rounds to one in red. The concern was that Russian roulette does not fire in a
medium that absorbs nothing, leaving the step ceiling as the only thing that ends
a walk, and every walk it ends as energy discarded.

It does not happen. The skin row grades 0.0599 / 1.014, against 0.0570 / 1.015
for the coarse row -- the same, inside the ladder's own 3% offset. Over
`subsurface_iterations`:

| iterations | rel | ratio |
|---|---|---|
| 8 | 0.0787 | 0.974 |
| 16 | 0.0682 | 0.993 |
| 64 | 0.0599 | 1.014 |
| 256 | 0.0598 | 1.020 |

64 and 256 are the same answer, so the default is not truncating. The reasoning
that predicted trouble was wrong about which length matters: a path enters the
medium about one mean free path deep and escapes from there, so the number of
scattering events is set by the depth of penetration and not by how many free
paths span the body. A thousand-free-path sphere and a ten-free-path sphere ask
the walk for the same number of steps.


### Subsurface free flights drawn against infinity

**Fixed.** A five-second `MTL4CommandQueueErrorTimeout` killed the Metal 4 queue
on the Open Chess Set with its MaterialX materials -- in the editor after a few
seconds of accumulation, and in the CLI 4 runs out of 4 at 4096 samples
(960x540, depth 8), at samples 111, 2066, 2815 and once near the end. Every
reproduction blamed the same stage: `last_completed=shade suspected=shadow`.

The walk drew its free flight with `sssSampleDistance(sigmaT, channelPdf, 1e16f,
...)`. The draw bounds the ray, so "no surface within it" is the ordinary way a
walk scatters -- which makes a flight longer than the medium indistinguishable
from one that stayed inside, and the event is taken at that distance anyway. For
a closed mesh the boundary wins and it never shows. For the paths that leak out
through a hole -- this scene reports 4677 of them per sample in the
nested-dielectrics warning -- nothing stops the ray, and the walk lands a scatter
event as far out as `-log(1e-7)/sigma_t` allows: with sigma_t around 1 that is 16
units in a scene half a unit across. The light connection made from there is a
shadow ray tens of times longer than the scene, and that is what puts Metal's
traversal into the pathological mode this tree has hit before (see the
triangle-only intersector note in `wavefront.metal`) and holds the dispatch past
the GPU watchdog.

The ceiling is now `uniforms.sceneExtent`, the diagonal of `Scene::worldBounds()`:
a free flight longer than the scene cannot have stayed inside a bounded medium,
so `sssSampleDistance` declines it and the ray runs to its boundary instead.

Measured, at 512 samples on the failing camera: two runs of the fixed build
differ by `rel` 0.00004 (the run-to-run floor, defect #14), and fixed against
unfixed by 0.00006, with the means equal to six decimals. The image does not
move; 4096-sample runs that failed 4 out of 4 now pass. `Scene::worldBounds()` is
covered by `tests/scene/test_world_bounds.cpp`, including that a transform change
invalidates its cache.

Still unbounded, deliberately: `fogSampleDistance` keeps its `1e16f`, because fog
*is* unbounded -- it fills the world rather than a mesh, so a long flight in it
is legitimate. If a fog scene ever reproduces this, the bound to reach for is the
same one.

### Diagnostics this cost

**Fixed.** Three of them, each of which had been hiding the fault:

- The failure report printed the **first** failure it saw. A GPU reset delivers
  failures in queue order, not causal order, so it named a discarded bystander
  (`kIOGPUCommandBufferCallbackErrorInnocentVictim`) and buried the cause.
  Ranked now by `metal4FailureRank()`, which is what made the queue timeout
  visible at all.
- `STRELKA_STAGES=1` was read in `runBenchmark()` alone, so following the failure
  path's own advice on the editor changed nothing and it printed "stage diagnosis
  disabled" a second time. It applies to interactive runs now, and it is what
  named the shadow stage.
- On a device error the editor dumps the viewport as a paste-ready CLI `.toml`.
  A chunk index says what the renderer was doing, not what it was looking at, and
  the sampler in that dump (`hybrid`, not `sobol`) was part of reproducing it.


### IES tables were interpolated with straight lines, Cycles uses a cubic

**Fixed.** With the parser reading real files and the ladder row carrying a real
beam, the row still sat at ratio 1.050 / rel 0.057 against the reference, and the
error had a shape: 0% on axis, rising to 9% at 22 degrees and falling again.
That is the steep flank of the Philips reflector's beam, between the table's 20
and 25 degree rows, where the intensity drops 2.4x in one 5 degree step.

Both renderers were reading the same four numbers and drawing different curves
through them. Cycles interpolates with Catmull-Rom over four samples per axis
(intern/cycles/kernel/util/ies.h); Strelka drew straight lines, which cut the
corner of a convex curve and read 12.7% high at the midpoint of that interval.

Strelka now runs the same cubic, with the same endpoint fallbacks and the same
wrap behaviour at the poles and at the azimuth seam, from one header both
backends and the host loader share (`common/ies_math.h`). Two smaller things
came with it, both also Cycles' behaviour and both previously wrong here:

* the azimuth table is unfolded to the full turn **at load time** rather than
  folded at every lookup, so the cubic has real neighbours at the seam instead
  of reflected guesses, and a table that omits its 360 degree duplicate gets one;
* a direction outside the tabulated range returns zero rather than the clamped
  edge value, which is what the file actually says about it.

Measured on the row: ratio 1.050 -> 1.032, rel 0.057 -> 0.047, and the radial
error curve went from a hump to flat (1.02-1.04 everywhere). What remains is a
scale offset with no angular structure that predates this work -- the old
synthetic cos^4 profile, where interpolation barely mattered, sat at 1.028 -- and
is not the table: on axis the lookup lands exactly on a tabulated angle, so both
renderers read the same 3418.9 cd and still differ. Untraced, and in the band the
rest of the ladder occupies.

Tests: the cubic against its closed form and against straight lines on data that
is straight; the 12.7% gap at the steep step; non-negativity (a cubic through
non-negative samples can dip below zero); zero outside the table; and continuity
across the azimuth seam.

### The IES reader could not read most of an IES file

**Fixed.** The audit above tested the *evaluation* of a photometric table
against hand-built profiles. That left the parser untested, and the only case in
the suite wrote the smallest file that could possibly parse: a version line,
`TILT=NONE`, three angles, three numbers. A real LM-63 file is not that, and four
of the paths a real one takes were broken. Each fails the same way -- the profile
is rejected or misread, and the luminaire silently becomes an isotropic point
light with whatever intensity the sidecar gave it, which does not look like a
failure. It looks like a plain lamp.

1. **`TILT=<filename>`**, the third legal specification, was treated as
   `TILT=INCLUDE`: the parser consumed the photometric header as a tilt block --
   lamp count read as a pair count -- and threw the file away.
2. **A keyword line whose text begins with those four letters** ("[TESTLAB]
   TILTON Photometrics") was mistaken for the TILT line, with the same result.
   The reader scanned for a *token* starting with "TILT" rather than for the
   line.
3. **A UTF-8 BOM** in front of a file with no version line made the TILT line
   invisible to the line filter, so the file had no TILT at all.
4. **`tilt=none` in lower case, or `TILT = NONE` spaced out**, were not
   recognised either.

The reader now splits the file at the TILT line -- everything above is free text,
everything below is numbers, which is what the format is built around -- instead
of guessing line by line from the first character. Two more things came out of
looking at it: a non-type-C file (photometric types A and B tabulate a different
pair of angles) is now loaded with a warning rather than silently read as type C,
and a descending angle table is refused rather than fed to a binary search that
would return a plausible wrong number for every direction.

Tests: `tests/sceneloader/test_ies_loader.cpp` -- a realistic LM-63-2002 file
with keywords, a candela multiplier, a ballast factor and wrapped values; all
three TILT forms; the header quirks above; the candela layout (horizontal plane
outermost, which if transposed rotates the beam into a different plane); and the
refusals. Each was checked by re-introducing the defect it guards.

### The two lumens-per-watt constants are not a mistake

**Not a defect; documented and pinned.** The audit reported three
candela-to-watt assumptions and recommended collapsing them. Looking closer,
`LIGHT_UNIT_INTENSITY` does not convert anything -- it is radiant intensity
already, and the glTF loader divides before handing the value over -- so there
are two, and they are two on purpose:

  * `kLuminousEfficacyD65` (177.83) converts a **measurement**. An IES file is
    candela from a real luminaire with an unknown spectrum, and an illuminant has
    to be assumed. Cycles assumes D65 for the same conversion.
  * 683 lm/W in `gltfloader.cpp` undoes a **bookkeeping step**. Blender's glTF
    exporter writes candela as `watts * 683 / (4 pi)`, so recovering the watts
    the artist typed means dividing by 683 -- whatever the lamp's spectrum is.

Collapsing them breaks agreement with one reference or the other: with Cycles on
IES profiles, or with Blender on a round-tripped lamp, which
`tests/scene/test_light_units.cpp` already pins as a shipped field regression.
Both constants now live in one place with that reasoning next to them
(`scene/light_desc.h`; the Metal path used to hard-code its own copy of 1/177.83),
and a test states the 3.84x gap so it stops being reported as new.

### A second audit pass over the lights and the environment

**Fixed.** The MIS audit above looked at how the two halves of the estimate are
combined. This pass looked at what they are combining: the environment's
distribution and the analytic light types.

What the environment turned out to have right, measured rather than assumed:
the reported density integrates to 1.000 over the texels' true solid angles
(exactly, not by Monte Carlo -- uniform sky, a theta gradient, and a sun at the
horizon and at the zenith); a jittered texel sample lands back in its own texel
for all but 0.001-0.02% of draws, and those are float rounding at the texel
boundary; the OptiX point texture really is unnormalised-coordinate point-filtered
under a `tex2D(x + 0.5, y + 0.5)`; and both backends build one alias table from
one host builder.

What was wrong:

1. **Metal wrapped the environment in v.** `address::repeat` on both axes, where
   an equirectangular map wraps in azimuth only. The bilinear tap in the first
   row blended the zenith with the last row -- the nadir. OptiX has always
   clamped v. Narrow (a row out of 512) but visible on a background that looks
   up, and a backend disagreement in the one place they should agree texel for
   texel.

2. **One NaN texel took the whole HDRI with it.** `buildIblAliasTable` clamped
   negatives with `std::max` and let NaN through, so `totalPower` went NaN,
   `envPdfScale` went NaN, and every density -- sampling and MIS weight alike --
   followed.

3. **IES tables were extrapolated past their own data.** Neither the CPU loader
   nor either shader clamped the interpolation weight, so a direction outside the
   tabulated range continued the last interval. Measured on a table that falls to
   zero at 90 degrees: -1800 cd at 180, a negative light that only `emitsLight()`
   kept out of the frame. On a table that rises to its last entry -- the case
   nothing rejects -- +2800 cd at 180, a luminaire shining nearly three times its
   own peak straight up.

4. **Quadrant-symmetric IES profiles were unfolded with a repeat, not a mirror.**
   `fmod(azimuth, 90)` sends 90 to 0 and 100 to 10, where mirroring sends them to
   90 and 80. A luminaire bright along one axis and dark along the other came out
   rotated by a quadrant: the dark axis read as the bright one.

And four copies of things that should have been one, which is how all of the
above happened in the first place:

| what | was | now |
|---|---|---|
| equirectangular uv, luminance, texel pdf | `common/env_light.h` + `metal/env_light_metal.h` | `common/env_map_math.h` |
| alias-table draw | shared header + a Metal transcription missing its guards | `common/env_alias_sampling.h`, moved so Metal can reach it |
| spherical-rectangle sampling | `common/lights.h` + `metal/lights_metal.h` + `scene/rect_light_sampling.h` | `common/rect_sampling.h` |
| IES fold and clamp | `iesloader.cpp` + both light headers | `common/ies_math.h` |

The rect case is worth stating on its own: `tests/scene/test_rect_light_sampling.cpp`
was the only test of the spherical-rectangle construction in the tree, and it
exercised the host copy -- the one no GPU runs. It now covers the shipping code.

Every validation scene renders bit for bit identically across this pass
(`cornell_box`, `metal_sphere`, `mixed_materials`, `kids_room`, all at 32 spp),
which is what says the four unifications were refactors and nothing else.

Tests: `tests/render/test_env_map_math.cpp`, `tests/render/test_ies_math.cpp`, a
NaN case in `test_ibl_alias_table.cpp`, and the existing rect and alias suites
now pointed at the shared code. Each was checked by re-introducing the defect it
guards.

Not fixed, because it is a calibration decision rather than a code one: the
renderer carries three different candela-to-watt assumptions -- `/683` for glTF
`KHR_lights_punctual`, `/177.83` (D65) for IES profiles, and 1:1 for a sidecar
light authored with `unit = "intensity"`. No scene in the repository uses either
of the first or the third (all sidecar lights are `radiance` or `power`), so
nothing currently renders differently for it; a scene that mixed a punctual light
with an IES one would be off by 3.84x between them.


### The MIS estimate was audited end to end, and seven things came out of it

**Fixed.** An audit of both backends against the two properties multiple
importance sampling rests on -- the halves agree on the density of every
direction, and each half divides by the density its own sampler drew from --
found seven defects. They are grouped here because six of them are the same
mistake in different clothes: something that belongs to one half of the estimate
was decided by a draw belonging to the other, or a density was shared by name
and not by code.

The measurement that closes them all is the estimator A/B, which the
`render/validate/estimatorMode` switch exists for: NEE + MIS against BSDF
sampling alone, two independent unbiased estimators that must agree at
convergence. On `cornell_box` at 2048 spp, 512x384:

| | mean, NEE + MIS | mean, BSDF only | disagreement |
|---|---|---|---|
| before | 4.6444 | 4.6845 | 0.86% |
| after | 4.6914 | 4.6845 | 0.15% |

The BSDF-only column is identical to the last digit across the change, which is
what says the next-event half moved and the transport did not.

What was found:

1. **The sphere light's density was the constant 1/(4pi)** while the sampler drew
   a point uniformly over the sphere's *area*. Neither a solid-angle nor an area
   density. Both halves used the same wrong number, so the weights still summed
   to one and nothing looked inconsistent -- the light was simply wrong, by
   `d^2 / (r^2 cos)`. Against the analytic irradiance of a uniformly emitting
   sphere: 111x too bright at r = 0.5, d = 4, and worse with distance. A point
   light given a soft radius took the same path and jumped by 4 pi the moment
   its radius crossed the softness threshold.

2. **`standard_pbr_eval()` did not describe reflections off a transmissive
   surface.** `standard_pbr_sample()` produces them from inside the transmission
   lobe, through its Fresnel coin flip; eval left the term out of both f and the
   pdf. On glass, *every* non-delta reflection the sampler produced was reported
   by eval as pdf 0 -- 8194 of 8194 at roughness 0.15 -- so next-event estimation
   could not see a rough glass reflection at all, while the light hit still
   deducted a MIS share for it.

3. **Refraction: two mistakes in the change of variables**, and this one closes
   the "sample and eval genuinely disagree" note that
   `test_sample_eval_consistency.cpp` carried as a known bug. The half vector was
   rebuilt as `normalize(V + eta * wi)` where Walter et al. 2007 build it from
   `eta_i * V + eta_t * wi` -- about 0.2 rad away from the one the sampler bent
   around -- and the density used `ggx_vndf_pdf()`, which is already divided by
   the `4 * VdotH` that turns a half-vector density into a reflected-direction
   one. Integrated over the sphere the reported pdf came to 0.24 where the
   sampler produces a non-delta event 0.96 of the time.

4. **Next-event estimation was gated on the BSDF sample's event type.** A
   material carrying both a delta lobe and a smooth one -- a clearcoat over a
   diffuse base, and glTF's default coat roughness is 0 -- delivered the smooth
   lobe's direct light only on the draws where the delta lobe lost the lobe
   selection. Measured as the fraction of draws flagged specular: 52% on a 0.18
   grey base under a smooth coat, 73% on a dark lacquered paint. A vertex whose
   sample came back `BSDF_EVENT_ABSORB` lost its direct lighting outright.

5. **Metal decided `PATH_FLAG_NEE_DONE` from whether the connection produced a
   shadow ray**, in all three of its volume paths. OptiX documents this exact
   trap and avoids it; the flag has to record what was *available* at the vertex.
   Deciding it from the outcome hands the bounce ray the whole contribution on
   exactly the draws where the connection came back empty. Analytically that is
   `1.5 - 0.5 * emitterFraction` times the correct answer, which
   `test_nee_pairing.cpp` now both measures and states.

6. **Metal never sampled `LIGHT_TYPE_DOME`.** A missing switch case is not a
   compile error: the sample stayed zero-initialised, the facing test rejected
   pdf 0, and a dome light was silently black on that backend only. OptiX had the
   same hole and it was fixed there alone.

7. **Smaller divergences between the backends**, each of which breaks the
   agreement the halves need: `render/pt/misHeuristic` was read by OptiX and
   ignored by Metal, whose kernels called the balance form unconditionally; the
   distant light's cone density was written `1/(2pi(1-cos a))` on one side and
   `1/(4pi sin^2(a/2))` on the other (equal on paper, 0.2% apart at the sun's
   half angle and 4.9% at 0.001 rad); Metal's local-light connections left from
   the raw hit position with a fixed 1 mm ray tMin standing in for the face
   offset the bounce ray uses; Metal's facing test admitted directions above
   1e-3 while its light hit admitted everything above 0; a soft point or spot was
   exempted from the delta rule and weighed against a BSDF strategy that
   `visibilityMask = 0` had switched off; and `misWeightBalance(0, 0)` was a NaN
   in the pixel.

Where the shared code now lives: `src/shaders/common/light_pdf.h` (densities and
heuristics), `src/shaders/common/nee_pairing.h` (which directions the halves
share, and when a vertex owes a deduction -- moved out of `shaders/optix/` so
Metal includes it too), `bsdf_has_smooth_lobe()` in the material library, and
`pbr_reflection_terms()` in `standard_pbr.h`, which is now the single place the
reflection hemisphere is evaluated rather than the fourth copy of it.

Tests: `tests/render/test_light_pdf.cpp`, `tests/material/test_smooth_lobe.cpp`,
and new cases in `test_nee_pairing.cpp` and `test_sample_eval_consistency.cpp`.
Each was checked by re-introducing the defect it guards and confirming it fails.

### Rendered images that moved

`kids_room` drops 13% in mean radiance -- it has two 6.5 mm sphere lights, which
is defect 1 at its worst. `mixed_materials` moves 0.4% in the mean with local
differences up to 62 on its glass. `cornell_box` and `metal_sphere` gain about
1%, which is the direct lighting defects 4 and 7 were losing.


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
