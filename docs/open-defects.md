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
| ~~13~~ | ~~OpenPBR's vendored BSDF has never been through nvcc~~ | done — see 13 | closest-hit OPTIXIR builds clean; `scenes/validation/openpbr` renders on OptiX |
| 14 | The subsurface walk does not reproduce run to run | `wavefront.metal` medium path, Metal | two runs of one binary on `25_subsurface` are bit-identical |
| ~~15~~ | ~~Diffuse summed with specular instead of layered under it~~ | done — see Closed | `00_calibration` 0.012 / 1.008, its three regions within 0.4% of each other |
| ~~16~~ | ~~The walk was entered diffusely where Cycles refracts~~ | done — see Closed | four subsurface rows at 0.994-1.003; slab within 5% of Cycles |

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

## 13. Adobe's OpenPBR does not compile under nvcc as shipped — done

Measured on CUDA 13.3 / OptiX 9.1. The prediction this entry recorded was right,
and it was not the only thing wrong: **openpbr's CUDA backend had never been
compiled by anyone**. Three independent errors, in the order nvcc finds them.

**1. Execution space.** 211 definitions carry no execution-space macro at all
(the count is higher than the 191 estimated here, because 46 spread the return
type and the name over two lines and the original count missed them), e.g.
`float openpbr_average_fresnel(const float eta) {...}` at file scope in
`impl/openpbr_lobe_utils.h`. Each is a `__host__` function to nvcc, and the 87
carrying `OPENPBR_INLINE_FUNCTION` (`__device__ inline`) call them.

**2. Vector types.** `interop/openpbr_interop_cuda.h` aliases `vec3` to CUDA's
`float3`, which is a bare aggregate: no three-argument constructor, no
`operator[]`, no `.rgb`, no comparison operators, no GLSL-named componentwise
math. openpbr is written against GLM and MSL vectors and uses all of those. The
alias cannot work for any renderer, not just this one.

**3. Lookup tables.** The eight tables are declared `OPENPBR_CONSTEXPR_GLOBAL`,
which that layer spells `static inline constexpr` — a host global. Device code
cannot read one.

Answered without editing the submodule, which is still byte-identical to
upstream `9edf806`:

  - `tools/openpbr_device_headers.py` rewrites (1) and (3) into
    `${CMAKE_BINARY_DIR}/src/shaders/openpbr_device`, put *ahead* of the
    submodule on the OPTIXIR include path. It prints its patch count, so a
    submodule update that changes a form it matches shows up in the build log
    before it shows up as a `__host__`-from-`__device__` error.
  - `src/material/include/strelka/material/openpbr/openpbr_cuda_vec.h` answers
    (2) through `OPENPBR_USE_CUSTOM_VEC_TYPES`, which the interop layer offers
    for exactly this. It is included by `openpbr_shim.h` on the CUDA branch only.
  - `--expt-relaxed-constexpr` in `OPTIX_NVCC_FLAGS` covers the ~30 helpers
    spelled `OPENPBR_CONSTEXPR_FUNCTION`, i.e. `static constexpr`. This is
    nvcc's own suggested answer for that diagnostic, and the alternative is
    thirty more rewrite rules for one-line function bodies.

One real bug in Strelka's own code fell out of it: `openpbr_bridge.h` declared
the sample's out-parameter `float3 wi`, which binds to openpbr's `vec3&` on
Metal and the host only because `vec3` is a typedef there. It is now spelled
`vec3`, which is correct on all three.

`OptixRender_closest_hit.cu` compiles to a 3.7 MB OPTIXIR module with zero
errors and zero warnings, and `scenes/validation/openpbr` renders.

The host-side duplicate-symbol problem this entry also described is unrelated
and still handled the same way: `openpbr_shim.h` puts the library in an
anonymous namespace on the host, and does not on either GPU, where a shader
module is one translation unit and the definitions must keep external linkage.

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

Closed investigations are intentionally kept in Git history rather than this
working list. Use `git show b720fc5^:docs/open-defects.md` when old measurements
or discarded hypotheses are needed.
