# Open performance items

The same discipline as `open-defects.md`, for cost rather than correctness: each
entry is something measured, with the measurement that found it and what has
already been ruled out, so a reader starting cold can act without repeating the
elimination. What is closed is here too, because the *why* is what the next
person needs and it is the same why either way.

## pine_scene is a different launch, 2026-09-11

Everything above was measured on the two rooms, and pine does not share their
limiter. One launch of 16 samples at 1280x720, depth 4:

| | pine_scene | kids_room |
|---|---|---|
| DRAM per launch | **90.6 GB** | 9.2 GB |
| DRAM throughput | **62.5 %** | 5.7 % |
| `long_scoreboard`, cycles per issued instruction | **27.1** | 3.4 |
| `no_instruction` | 20.2 | 32.8 |
| L2 hit / L1 hit | 83.5 % / 40.2 % | 98.2 % / 45.7 % |
| SM throughput | 7.3 % | 9.2 % |
| active lanes of 32 | 14.0 | 15.2 |

The rooms are issue-starved; **pine waits on memory**, at 62 % of the card's
DRAM peak. Nothing in this file's instruction-footprint work applies to it, and
its numbers should not be read as a smaller version of kids_room's.

Where the traffic is, same launch:

| | |
|---|---|
| global loads | 53.0 GB |
| local loads / stores | 31.4 / 32.7 GB |
| texture | 7.4 GB |
| DRAM read / write | 71.7 / 18.9 GB |

And 76 % of the closest hit's *global* sector requests are scalar 32-bit LDGs,
whose top lines are three groups of eight loads at offsets 0x0..0x1c from three
different bases: the three 32-byte `Scene::Vertex` records of the triangle
fetch, one dword at a time.

Two things measured against that and neither kept:

- **Loading each vertex as two 16-byte vectors** instead of eight dwords.
  `Vertex` is float3 plus uints, so its alignment is 4 and nvcc emits the eight;
  the buffer is cudaMalloc'd on a 32-byte stride, so a `float4` pair is legal.
  Four times fewer requests for the same bytes, and pine does not move: 109.9
  against 109.2 ms, kids_room unchanged. The sectors are what costs, not the
  requests that ask for them.
- **Fetching only the uv for the coverage test**, and the three positions only
  when the test decides to pass through -- 36 bytes of vertex against 96, with
  no normal, tangent or colour unpacking and no attribute transforms. 106.2
  against 106.4 ms. Also not kept.

  A trap found on the way: `optixGetTriangleVertexData()` would be cheaper still
  -- traversal has just read that triangle -- but it requires
  `OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS` on every GAS, and without it the
  launch takes an illegal address rather than failing at build time. The flag
  means keeping the vertex buffer a second time, which is 1.5 GB in this scene.

### Opacity micromaps: the hardware has them, this renderer could not use them

Before writing an any-hit program, the feature that already exists for this:
`render/pt/opacityMicromaps`, off by default, with `opacity_micromap_policy.h`
behind it. It is the hardware version of the same idea -- traversal resolves the
regions of a cutout that are wholly inside or wholly outside, and calls the
shader only where the answer varies -- and it changes no sampling.

**The card has the engine.** `OPTIX_DEVICE_PROPERTY_RTCORE_VERSION` reads 30 on
this machine, which is Ada's third-generation RT core; the opacity micromap
engine arrived with it, and OptiX emulates it in software on anything older.
The renderer logs the number now, because "is this hardware or emulation" is the
first question any measurement of this feature has to answer.

**But no pipeline here ever declared it.**
`OptixPipelineCompileOptions::allowOpacityMicromaps` was never assigned, and the
struct is zero-initialised, so a pipeline that traverses a structure carrying
micromaps is not told they exist. Every micromap this renderer has ever built
was ignored. It is set from the same setting now.

With that fixed, `07_alpha_clip` builds micromaps (level 4, 128 of 512
microtriangles resolved on the cutout card) and renders to the same mean to six
digits, with a largest per-pixel difference of 9.1e-4 at 512 spp -- noise, not
bias, and the expected kind: a microtriangle resolved transparent reports no hit
at all, so the path does not spend the pass-through counter that seeds the
stochastic test, and the sequence shifts.

**It does not pay on pine.** 110.4 ms against 106.3 with it off, repeatably --
4 % for declaring the feature and building the structures, with nothing to show
for it, because the classifier resolves *nothing* on this asset: 144 meshes,
every one "opacity micromap resolves nothing, skipped". A foliage card is two
triangles across a whole needle atlas, so at the level-4 cap a microtriangle's
uv footprint covers ~8000 texels and the classifier's 4096-texel bound declares
it unknown without looking. Raising the two caps -- level 4 -> 6, texels
4096 -> 65536 -- does not change the verdict either: the meshes still resolve
nothing and the frame is 108.9 ms, still behind the 106.3 of having the feature
off. Both probes are reverted; the pipeline flag and the RT core log are not.

**What the cutouts cost, and what is left of it.** Compiling the coverage test
out entirely (`hasCutout` forced false -- wrong image, measured for the bound)
is 93.3 ms against 106.4: the feature is 13 % of pine's frame. Moving the test
above `initSurfaceInteraction()`, so that a needle the path slips through no
longer resolves base colour, metallic-roughness, emission and the normal map
first, takes 3 % of that (109.2 -> 106.1). The other 10 % is not shading at all:
it is the traversal restart every pass-through costs, one full trace per needle,
with the segment bookkeeping and the continuation-stack spill that go with it.
An `__anyhit__radiance` that ignores the intersection is what removes that --
the radiance hit group has no any-hit program today, and cutout instances
already omit `OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT` for the shadow ray's sake. It
needs a per-intersection random rather than the per-vertex `passthrough`
counter, so it would change the noise pattern (not the estimate), and Metal
keeps the loop until someone ports it.

## The profile after the traffic work, 2026-09-11

`--set full` on the render launch, kids_room, one launch of 16 samples at
1280x720 depth 4. Run as root, and with `-k optixLaunch`: that is the name the
render launch carries, and the accel builds are hundreds of launches ahead of it.

| | |
|---|---|
| duration | 135.2 ms |
| SM throughput | 9.2 % |
| memory throughput / DRAM | 32.3 % / **5.7 %** |
| L1 hit / L2 hit | 45.7 % / 98.2 % |
| IPC (of 4) | 0.39 |
| active warps / scheduler | 3.98 |
| **eligible warps / scheduler** | **0.11** |
| cycles with no eligible warp | 90.1 % |
| **active lanes of 32** | **13.8** |
| registers / occupancy | 255 / 33.1 % |

Warp stall, cycles per issued instruction (40.11 total):

| | |
|---|---|
| `no_instruction` | **32.79** |
| `long_scoreboard` | 3.40 |
| `wait` | 1.66 |
| `selected` | 1.00 |
| `branch_resolving` | 0.40 |
| everything else, each | < 0.25 |

Nothing is waiting on memory any more: DRAM is at 5.7 %, L2 hits 98 %, and
`long_scoreboard` is 3.4 cycles against 16.8 in the September profile. **The
launch is issue-starved: 82 % of its stall is `no_instruction`.**

### `imc_miss` does not mean what this file used the last time

An earlier entry ruled out instruction fetch on the grounds that `imc_miss` was
zero. That reasoning is wrong: IMC is the *immediate constant* cache, not the
instruction cache, so `imc_miss` says nothing about instruction fetch. It is
0.16 cycles here and it was never the metric to read. `no_instruction` is the
one, and it is the whole frame.

### Where the samples are, per program

PC samples over the same launch, 15.09 million of them, grouped by the OptiX
program they landed in (the report's source page, aggregated by kernel name):

| program | share | `no_instruction` |
|---|---|---|
| `__closesthit__radiance` | **60.3 %** | 89.4 % |
| `__intersection__light` | **25.3 %** | 91.7 % |
| `__raygen__rg` | 7.9 % | 85.0 % |
| `__closesthit__analytic_light` | 2.5 % | 93.0 % |
| everything else, summed | 4.0 % | -- |

Every program is ~90 % `no_instruction`, which is what a 189 560-instruction
module with a reordered warp jumping between programs looks like. The two
levers this leaves are the size of the code on the hot path and the 13.8 lanes.

### A quarter of the frame is the analytic-light intersector

`__intersection__light` is a custom-primitive program, and analytic lights are
in the same structure as the geometry -- so **every ray runs it, shadow rays
included**: `RAY_MASK_SHADOW` carries `GEOMETRY_MASK_LIGHT` and
`GEOMETRY_MASK_LIGHT_HIDDEN` on purpose, so that an emitter stops a shadow ray
the way geometry does. Metal's `ShaderTypes.h` has the identical mask, so this
is a shared decision rather than an OptiX detail.

Measured what it costs, by taking the two light bits out of the shadow mask:
kids_room 90.6 -> **75.6** ms per launch, best of three, 16 samples at 1280x720
depth 4. The ladder does not move a digit on any of its 33 rows -- it has no
scene where one emitter stands between a surface and another emitter -- but
kids_room does: 15.1 % of its pixels change by more than 1e-3 and the image
gains 0.23 % of its mean.

**The reference says the cheaper answer is the right one.**
`tools/feature_tests/light_occlusion_probe.py` builds the smallest scene that can
tell -- a grey floor, a big rect light above it, a small rect light hung directly
between the two -- and renders it in Cycles. Across the floor under the blocker,
Cycles reads 0.311 / 0.385 / 0.464 / 0.518 / **0.540** and Strelka read 0.316 /
0.363 / 0.355 / 0.301 / **0.254**: Cycles draws no silhouette there and Strelka
drew one. So `RAY_MASK_SHADOW` is the geometry bits alone on both backends now,
and the same probe reads 0.316 / 0.394 / 0.464 / 0.513 / 0.535 against the
reference.

Mesh emitters are unchanged -- they are triangles, they keep the geometry bit,
and Cycles blocks with them too.

| ms per launch, 16 samples, best of three | before | after | |
|---|---|---|---|
| kids_room | 90.6 | **76.1** | -16% |
| iso_bathroom | 47.7 | **41.7** | -13% |
| pine_scene | 113.8 | **109.5** | -4% |

The ladder moves one row and it moves toward Cycles: `19_env_and_light` 0.064 /
0.986 -> 0.063 / **0.992**. The other 32 are identical to the digit, which is
also why the ladder could not have found this on its own -- no row has one
emitter standing between a surface and another.

PC samples after, same launch: 15.09 M -> 12.22 M, with
`__closesthit__radiance` 60.3 -> 68.7 % of them and `__intersection__light`
25.3 -> **15.1 %**. Secondary and camera rays still test the light surfaces,
which is what the MIS estimate needs them for.

The probe found a second thing, unrelated to this and not yet chased: the
blocker is visible to the camera in Strelka at its full radiance (3.248) where
Cycles shows the wall behind it (0.105). That is a camera-visibility question
about a rect light seen from a grazing angle, not a shadow question -- it
predates this change, since nothing below the primary mask moved.

## Scene-content bound values, 2026-09-11

The OptiX pipeline compiled four scene-level facts as runtime tests that Metal
has compiled as function constants since the wavefront landed: whether the scene
has an OpenPBR material, curve geometry, a cutout material, or a medium of
either kind. `PipelineSpec` gained `hasOpenPBR` / `hasCurves` / `hasCutout` /
`hasSubsurface`, each bound the same way the existing eighteen are, and the
closest hit gates on them (`kFcOpenPBR`, `kFcCurves`, `kFcAlpha`,
`kFcSubsurface` are the Metal spellings of the same four).

OpenPBR is the one that mattered, and it could not have been folded before: the
test was `params.openpbrParams != nullptr`, a pointer read at runtime, so a
module compiled for a scene with no OpenPBR material still carried the whole
Adobe lobe stack plus the two locals it needs -- `OpenPBRParams` is 272 bytes,
`OpenPBR_PreparedBsdf` is 720, and the prepared BSDF is built before next-event
estimation and read after it, so it crosses the shadow traversal and is charged
to the continuation stack.

Measured on the continuation-stack figure the renderer already logs at debug
level, plus OptiX's own pipeline statistics -- which needed a way to raise the
context log callback past errors, so `STRELKA_OPTIX_LOG_LEVEL` is now a knob
(4 prints the statistics; validation mode also prints them but compiles at
optLevel 0, so its numbers are not the ones the frame runs).

| | before | after |
|---|---|---|
| continuation stack, kids_room and iso_bathroom | 2272 B | **672 B** |
| instructions in entry functions, kids_room | 222 168 | 189 562 |
| continuation stack, an all-OpenPBR scene | -- | 2080 B |

So ~1600 of the 2272 bytes were OpenPBR, in scenes that have none of it; the
remaining three constants are worth the rest. The last row is the control: with
`material_model = "openpbr"` the same pipeline still compiles the lobe stack and
pays for it, which is what says the drop is specialisation and not deletion.

1280x720, `depth` 4, 16 samples per launch, best of three, ms per launch:

| | before | after | |
|---|---|---|---|
| kids_room | 132.2 | 124.5 | -5.8% |
| iso_bathroom | 70.6 | 64.7 | -8.4% |
| pine_scene | 151.6 | 138.4 | -8.7% |

The ladder is unchanged at 64 spp across all 33 rows, including the four that
grade the gated paths: `07_alpha_clip` 1.012, `08_alpha_blend` 1.007,
`18_bounded_volume` 1.003, `25_subsurface` 1.003, `28_hair` 1.011.

### What the counters say, and what 255 registers is not

Nsight Compute needs root on this box (`ERR_NVGPUCTRPERM`); run as root, with
`-k optixLaunch` -- the render launch is reported under that name, and without
the filter the profiler lands on `optixAccelBuild`, of which kids_room issues
hundreds. kids_room, one launch of 16 samples at 1280x720, depth 4:

| | before | after |
|---|---|---|
| registers / thread | 255 | 255 |
| achieved occupancy | 33.16 % | 33.16 % |
| local loads | 36.51 GB | **29.04 GB** |
| local stores | 41.32 GB | **29.59 GB** |
| DRAM | 32.28 GB | **16.98 GB** |
| launch | 187.28 ms | **177.81 ms** |

So the whole of this is local-memory traffic and DRAM. Neither the register
count nor the occupancy moved, and **`launch__registers_per_thread` = 255 is not
a measurement of our shader**:

- `debug = 1` -- which shades the primary hit and returns, with the bound value
  compiling the rest of the closest hit away -- also reports 255.
- `STRELKA_OPTIX_MAX_REGISTERS=96` also reports 255, while achieved occupancy
  goes 33.16 % -> **41.42 %** and local stores go 29.6 -> 34.3 GB.

255 is the envelope OptiX compiles its launch kernel with; what actually varies
is occupancy and what spills. Reading the 255 in the profile below as "our
shading needs 255 registers, and that is why occupancy is 32 %" is the wrong
chain -- it would be 255 for a kernel that shades nothing. Occupancy is the
number to watch, and the register ceiling is still the only direct lever on it.

That ceiling is now roughly free rather than a loss: at 96, and best of three at
16 samples per launch, kids_room 123.6 against 124.5 ms, iso_bathroom 64.4
against 64.7, pine_scene 134.3 against 138.4. Still a diagnostic -- the
differences are inside the run-to-run spread and chess_set was the scene that
lost 25 % to it -- but the sign is no longer against it.

### The estimate runs last now, and `si` dies before the shadow ray

The 672 bytes above were not the surface's own state being carried between
bounces -- they were state carried across a *traversal*. Compiling next-event
estimation out (`estimator_mode = 1`) took the continuation stack from 672 to
**160** bytes, which is PerRayData and nothing else: everything above that is
live only because `__closesthit__radiance` read `si` again after the shadow ray
returned.

It read it for one reason: the BSDF sample and the whole of the next segment sat
below the estimate. Both halves need `si`, but only one of them needs it after a
ray. `sampleNextBounce()` now runs first and returns its decisions as a value --
the throughput factor, the medium it entered, whether the path stops -- and the
estimate runs after it, with the committing writes after that. So the estimate
still sees the throughput and the medium of the vertex it is estimating, and
`si` is dead by the time the ray goes out.

Two properties keep this from changing any image. `random<Dim>()` is a pure
function of (sampleIdx, dimension, seed, depth) and does not advance the
sampler, so drawing the BSDF dimensions before the light ones changes no draw.
And the estimate was never gated on the event that came back -- `didNee` is
decided by the material, above both halves -- which is the property
`neeRunsAtVertex()` was introduced to keep, and the reason this reordering is
available at all. The ladder is identical to the digit on all 33 rows, and an
all-OpenPBR Cornell box renders to the same mean it did before.

kids_room, one launch of 16 samples at 1280x720, depth 4, through both changes:

| | baseline | + bound values | + estimate last |
|---|---|---|---|
| continuation stack | 2272 B | 672 B | **288 B** |
| local loads | 36.51 GB | 29.04 GB | **25.14 GB** |
| local stores | 41.32 GB | 29.59 GB | **25.38 GB** |
| DRAM | 32.28 GB | 16.98 GB | **9.16 GB** |
| registers / occupancy | 255 / 33.16 % | 255 / 33.16 % | 255 / 33.11 % |

ms per launch, 16 samples, best of three, the renderer's own timer:

| | baseline | + bound values | + estimate last | |
|---|---|---|---|---|
| kids_room | 132.2 | 124.5 | **94.4** | -29% |
| iso_bathroom | 70.6 | 64.7 | **47.7** | -32% |
| pine_scene | 151.6 | 138.4 | **113.8** | -25% |

Occupancy did not move through any of it. The whole of a third off the frame is
local-memory traffic and the DRAM behind it, which is worth stating plainly
because the counters name the opposite culprit: 255 registers and 33% occupancy
look like the limiter and were not.

### Where the remaining local traffic is, and why the next cut bought nothing

Split by ablation, kids_room, one launch of 16 samples at 1280x720 depth 4.
`estimator_mode = 1` removes next-event estimation and with it every shadow ray:
local loads 25.14 -> 14.26 GB, stores 25.38 -> 17.47. So a little under half the
traffic is the shadow ray and the state that crosses it, and the rest is the
path loop -- PerRayData is 136 bytes in local memory, reached through the
payload pointer by both programs, and every field touch is an LDL or an STL.

The continuation stack is 288 bytes now, of which the raygen chain is 160
(`estimator_mode = 1` measures it directly) and the shading program's frame
across the shadow ray is the other 128.

Committing the bounce *before* the estimate rather than after -- passing the
throughput and the medium the vertex arrived with by value, so that nothing the
bounce decided rides the shadow ray -- cuts the spill again: loads 25.14 ->
22.14 GB, stores 25.38 -> 22.93, DRAM 9.16 -> 7.47. **And it is worth no time at
all**: 135.4 -> 135.0 ms under the profiler, and the renderer's own timer cannot
separate it from run-to-run spread.

That is the useful negative result. 7.47 GB over a 135 ms launch is 55 GB/s,
which is nothing for this card, and the local traffic that is left lands in L1
and L2 rather than DRAM. The launch stopped being traffic-bound somewhere
between the first cut and this one, so the next byte of continuation stack is
not worth chasing for its own sake -- it needs a new profile first.

### Two things tried against the new limiter, and what they measured

**Specialising `__intersection__light` on the shapes the scene has.** A bit per
LightType in `Params`, bound into the pipeline, so a scene of point lights
compiles no rectangle or disc test. It works -- 189 560 -> 187 698 instructions
in the entry functions -- and it is worth **nothing**: kids_room 76.1 -> 75.5,
iso_bathroom 41.7 -> 45.0, pine_scene 109.5 -> 111.1, all inside the spread.
Reverted. The program is 15 % of the samples because every ray runs it, not
because it is large, and 1 % of the module is not what those samples are waiting
for.

**How much the glTF lobe stack is worth at all.** Compiling the sheen,
clearcoat, iridescence and transmission lobes out of `standard_pbr.h` outright
-- wrong images, measured for the bound only -- takes kids_room 76.1 -> 67.7 ms,
**-11 %**, while removing only 2 % of the module (189 560 -> 185 449
instructions). So the cost is the instructions the vertex *executes*, not the
ones the module carries, and per-scene specialisation can only claim the part a
scene does not use. kids_room uses all four (22 sheen materials, 10
transmission, 2 iridescence, 2 clearcoat), iso_bathroom all four, and
pine_scene none of sheen, clearcoat or iridescence -- so a `lobeMask` bound
value would pay on pine and very little on the rooms.

The more promising reading of the same measurement: `pbr_lobe_weights()` and the
combined-pdf machinery run in `bsdf_has_smooth_lobe`, in `bsdf_sample`, in
`bsdf_eval` and in `bsdf_pdf` -- three or four times per vertex over identical
inputs. OpenPBR already prepares its lobe stack once per vertex
(`openpbr_prepare_at`); the glTF model does not, and that is the same 11 % of
executed instructions seen from the side where the image stays correct.

### Preparing the glTF lobe stack once per vertex

`standard_pbr_sample()` and `standard_pbr_eval()` opened with the same twenty
lines: flip the frame for an opaque back hit, take NdotV, build the tangent
frame, split the roughness into ax/ay, weigh the five lobes, apportion them onto
the 23-bit categorical lattice, and compute F0. A vertex ran that prologue in
`bsdf_has_smooth_lobe()` before next-event estimation, in `bsdf_eval()` for the
connection, in `bsdf_sample()` for the bounce -- and once more inside
`pbr_finish_continuous_sample()`, which evaluated through the plain entry point
and so prepared the vertex a second time inside the draw it was finishing. Four
times, over inputs that cannot change between them.

`pbr_prepare()` now returns that state as a `PbrPrepared`, and the closest hit
builds one per vertex beside the `openpbr_prepare_at()` it has always built.
The old entry points remain and prepare on the spot, so Metal and the tests are
where they were; `pbr_prepare_for()` returns a zeroed struct for a material with
no glTF lobe stack, so a hair vertex does not pay for a cache it cannot read.

It is a cache and the images say so: all 33 ladder rows are identical to the
digit, and `unit_tests` passes.

| | before | after |
|---|---|---|
| kids_room, ms per launch (best of five) | 76.1 | **70.8** |
| iso_bathroom | 41.7 | **39.7** |
| pine_scene | 109.5 | 109.2 |
| instructions executed, kids_room launch | 15.85 G | **13.30 G** |
| `no_instruction`, cycles per issued instruction | 32.79 | **27.56** |
| active lanes of 32 | 13.80 | **15.20** |
| duration under the profiler | 135.2 ms | **103.4 ms** |

The last four columns carry the light-mask change above as well -- they are both
in this build -- but the ms rows are the two measured separately.

Still open, in the order they look worth doing:

- **Metal prepares nothing.** `wavefront.metal` calls the entry points that
  prepare on the spot, so the same four preparations per vertex are still there.
  The behaviour is identical either way, which is why this is a performance item
  and not a parity one.
- **An OpenPBR pipeline is still 1760 bytes of continuation stack** against 288
  for the glTF one, and the reordering only took 320 of it. `OpenPBRParams` is
  272 bytes and `OpenPBR_PreparedBsdf` is 720; which of them is still crossing a
  ray, and why, has not been established.
- **`allOpenPBR`**, Metal's `kFcAllOpenPBR`: a scene where every material is
  OpenPBR compiles the glTF lobe stack for nothing. Metal additionally requires
  no curves, because hair keeps its own BSDF.
- **`SurfaceInteraction` is 268 bytes.** It no longer crosses the shadow ray,
  but it is still built in full at every vertex; the sheen, iridescence,
  clearcoat, subsurface and diffuse-transmission fields -- about 84 bytes -- are
  read only by lobes most scenes do not have.

## The sampler, 2026-09-08

The profile below put `random.h:278` -- `X ^= sb_matrix[lowestSetBit(bits)][dim]`
-- at the top of every scene's PC samples, 12.8% to 26.7% of them. It is gone,
along with the 32 KB table it read. Editor defaults, 1280x720 `max_depth` 4,
interleaved A/B over four rounds: **iso_bathroom 5.93 -> 5.10 ms (-13.9%),
kids_room 9.83 -> 9.35 ms (-4.8%)**.

### What the table cost, before deciding what to do about it

Replacing the table read with arithmetic and changing nothing else: iso_bathroom
5.90 -> 4.80, kids_room 9.90 -> 7.50, pine 12.65 -> 11.00, bar 9.60 -> 8.70. The
Owen scramble around it is free -- that variant is no slower than plain PCG,
which has neither table nor scramble. So the sampler's whole cost was the gather.

Two things measured on the way, both worth not repeating:

* **Moving the pixel from the Sobol' index into the scramble seed, to match
  Metal, is 7-9% slower.** The index is Owen-scrambled with that seed, so a
  per-pixel seed leaves the walk exactly as divergent as a per-pixel index did
  and makes the value scramble divergent too, which it was not.
* **A nibble-radix fold of the table** -- eight fixed XORs against a 128 KB table
  whose entries are the XOR of the direction numbers a nibble names, bit-exact
  with the walk -- is a win on one scene and a loss on two: iso_bathroom -9.3%,
  kids_room -0.5%, pine +0.4%, bar +2.1%. Trading sixteen loads of 32 KB for
  eight of 128 KB does not pay.

### What replaced it

Two dimensions, and padding. A draw used to ask for dimension
`Dim + depth * eNUM_DIMENSIONS`, which reaches 117 on a depth-8 path and is the
only reason 256 dimensions had to be tabulated. A path is now a list of 21
*decisions*; every decision draws from the same two-dimensional sequence and is
told apart by its Owen scramble seed. Cycles is built this way for the same
reason. What is paired is what is genuinely 2D and consumed together -- pixel
offset, lens point, light point, BSDF direction -- so those keep the
(0,2)-sequence's joint stratification; the categorical draws stay 1D.

Both dimensions have a closed form, checked against the columns they replace on
all 32 rows each:

* dimension 0's direction numbers are `1 << (31 - k)`, so the walk is a bit
  reversal;
* dimension 1's are `v0 = 1 << 31, vk = vk-1 ^ (vk-1 >> 1)` -- Pascal's triangle
  mod 2, which is the fifth Kronecker power of `[[1,0],[1,1]]` and therefore
  decomposes into five masked shift-XOR stages. Verified exhaustively on the 32
  basis vectors, which by linearity settles every index.

Two draws, compiled standalone and disassembled: **112 SASS instructions, 5 LDG,
5 BRA, 2 FLO before; 104, 3, 1, 0 after** -- and the three remaining LDG are the
probe's own parameter load and stores. The data-dependent loop and the `__ffs`
chain are gone with the table.

### The bug this grew, and the test that now catches it

Padding by scrambling one value with per-decision seeds is wrong, and wrong in
exactly the way the note above the old table describes. An Owen scramble is a
bijection, so two decisions plotted against each other trace a curve rather than
covering the square -- each uniform alone, the pair degenerate. Written that way
it converged **kids_room 8.2% dark**, which is what a light selection taking two
coordinates off a diagonal looks like. Padding has to shuffle the *index* per
decision as well; a nested scramble maps dyadic blocks to dyadic blocks, so
nothing the Z-curve argument depends on is lost.

`tests/render/test_sobol_matrix.cpp` grades a pair of draws by how much of a
16x16 grid they cover over 4096 indices. Inequality was not enough and would not
have caught this: the degenerate pair differs at every index. Coverage is 0.0625
without the index shuffle -- sixteen cells of 256, a diagonal -- against a bar of
0.95.

After the fix the two samplers agree where it matters: at 1024 spp the mean
differs by 2e-4 relative on both scenes, and the blur-surviving residual fell
6-8x.

### Quality

FLIP against each sampler's own converged 1024 spp, editor defaults:

| | 1 spp | 2 spp | 4 spp | 16 spp |
|---|---|---|---|---|
| iso_bathroom, tabulated | 0.6788 | 0.5681 | 0.4837 | 0.3431 |
| iso_bathroom, padded | 0.7031 | 0.5731 | 0.4793 | 0.3435 |
| kids_room, tabulated | 0.4685 | 0.3596 | 0.2730 | 0.1633 |
| kids_room, padded | 0.4592 | 0.3497 | 0.2644 | 0.1699 |

A wash: a few percent each way, no systematic loss from giving up stratification
between decisions. That was the risk and it did not materialise.

### The Z-curve, and what its scramble is worth

`initSampler` hands each pixel a block of one global sequence and orders the
pixels along a Morton curve, so a 2x2 quad holds the four child blocks of one
parent and neighbouring errors anti-correlate. That is Ahmed and Wonka's
ZSampler (SIGGRAPH Asia 2020) and it was already the construction here; what was
missing was the hierarchical scramble that removes the curve's regularity.
Added, in both the cheap form (a two-bit XOR per level, four of the twenty-four
permutations) and the full one (all twenty-four).

Both measure as **nothing**: time within noise, FLIP identical to the fourth
decimal, SSIM to 1e-4, and the lag-1 autocorrelation of the high-passed residual
unmoved. The full version is kept because it is the construction the paper
describes and it is free, not because it was shown to help. What was not tested
is whether it removes a *visible* Z-curve pattern -- no summary statistic sees a
structured artefact the eye does.

The anti-correlation the blocking is supposed to produce is there and is small:
lag-1 on iso_bathroom is -0.065 against -0.050 for white noise through the same
high-pass, so the mechanism operates but does not dominate.

### Blue noise, and headless against the editor

`render/pt/samplerType` was read by the Metal backend alone -- OptiX ignored it
and always ran plain Sobol', so the editor's default of 4 silently rendered
something else there. OptiX now implements 2, 3 and 4, and says so once when
handed a value it does not (0 and 1 are Metal's). Ahmed and Wonka's toroidal-shift
mask is ported with it, sharing Metal's 128x128 void-and-cluster tile so the two
backends shift identically.

`hasBlueNoise` is a bound value, and that is not decoration: as a runtime test
the branch cost 3-9% *with the mask off*, on bit-identical output, because it
inlines into every one of a path's draws. Written as an `if/else` it was worse
still -- both sides inlined their own copy of the draw and two draws went from
112 SASS instructions to 224. The mask now picks the draw's arguments rather than
which draw to make.

`RenderConfig::samplerType` and `blueNoiseSwitchSpp` were 2 and 16 against the
editor's 4 and 4, so one config rendered two different images depending on which
application opened it. They are the editor's values now.

## The 2026-09-07 profile, four scenes

A full pass over `iso_bathroom` (its perspective camera, index 1), `kids_room`,
`pine_scene` and `bar` at 1280x720, `max_depth` 4, one sample per launch,
following `~/optix_profiling_methodology.md`. Configs, scripts and the
`.ncu-rep` reports are in `~/strelka_profile`; `sweep.py` drives the ablations
off the renderer's own GPU timer, `gaps.py` reads Nsight Systems, `pcsamp.py`
and `pcsamp_lines.py` read the reports.

Two things about the tooling that this file previously recorded the other way
round, both re-checked on Nsight Compute 2026.2.1:

- **`--set full` and PC sampling do work on an OptiX launch.** The note further
  down that a section cannot be opened on one is out of date. Source
  correlation works too, with `--print-source cuda,sass` and
  `STRELKA_OPTIX_LINEINFO=1`.
- **Counters need root here.** There is no `NVreg_RestrictProfilingToAdminUsers`
  entry on this box, so even `ncu --query-metrics` fails with ERR_NVGPUCTRPERM
  as the user. `--query-metrics` also lists hardware counters only: the
  `launch__*` family is always collectable and must not be filtered against it.

### Step 1 -- one scene was not GPU-bound at all

| | iso_bathroom | kids_room | pine | bar |
|---|---|---|---|---|
| launch, median ms | 6.06 | 9.79 | 13.12 | 9.53 |
| gap between launches, median ms | 0.028 | 0.026 | **2.324** | 0.027 |
| GPU busy, % of render span | 98.9 | 99.4 | **85.2** | 99.2 |

pine's gap is the same 2.3 ms at 320x180 as at 1280x720, and under 1 ms of
CUDA API calls accounts for all 31 of them together, so it was host compute.
It is `Scene::worldBounds()`, called once per frame by
`updateEmitterSelectionProbabilities()` for the scene extent that splits
emitter selection. `ensureInstanceWorldBounds()` was already generation-cached;
the reduction of that array to one box was not, and pine has **1 124 123
instances** -- 31 MB of `MeshBounds` streamed per frame at 13.6 GB/s, which is
what a scalar min/max walk with a branch per element costs. Caching the
reduction on the same generation counter: gap 2.324 -> 0.026 ms, GPU busy
85.2% -> 99.8%. `tests/scene/test_world_bounds.cpp` covers the transform-change
invalidation already and now covers the instance-count one, which is the case
the generation counter alone does not see.

No other scene has this: the next largest is bar at 1342 instances.

### Step 2 -- the limiter, and it is not the same on all four

| | iso_bathroom | kids_room | pine | bar |
|---|---|---|---|---|
| SM throughput % | 10.4 | 9.2 | 9.1 | 8.9 |
| memory throughput % | 32.7 | 29.1 | **54.1** | 32.8 |
| DRAM throughput % | 24.8 | 18.9 | **54.1** | 28.3 |
| L1 hit % | 55.7 | 56.3 | 49.7 | 53.9 |
| L2 hit % | 96.7 | 96.1 | **83.6** | 92.4 |
| DRAM GB per launch | 1.51 | 1.94 | **6.97** | 2.88 |
| effective GB/s | 244 | 186 | **532** | 279 |
| registers / thread | 255 | 255 | 255 | 255 |
| achieved occupancy % | 31.6 | 32.4 | 32.8 | 32.0 |
| active warps / scheduler | 3.91 | 3.95 | 3.96 | 3.91 |
| **eligible warps / scheduler** | **0.16** | **0.12** | **0.11** | **0.12** |
| cycles with no eligible warp % | 86.2 | 89.1 | 89.9 | 88.9 |
| active lanes of 32 | 18.7 | 13.7 | 12.3 | 14.2 |
| IPC (of 4) | 0.53 | 0.43 | 0.40 | 0.44 |

Read the eligible-warp row first, because it is the whole diagnosis for three
of the four scenes. At 255 registers a scheduler gets four warps, and for
86-90% of cycles not one of them can issue. That is what the `no_instruction`
stall below is: not an instruction cache problem -- `imc_miss` is 0.00 on all
four -- but a scheduler with nothing to run whenever its handful of warps are
all waiting. Half the lanes in those warps are idle on top of it.

pine is the exception and is genuinely bandwidth-bound: 532 GB/s is over half
of the card's peak, L2 misses three times as often as the rooms do, and its
7 GB per launch is the geometry working set (1.54 GB of vertices and 1.44 GB of
BLAS against 72 MB of L2).

Warp stall, cycles per issued instruction:

| | iso_bathroom | kids_room | pine | bar |
|---|---|---|---|---|
| `no_instruction` | 17.0 | **26.8** | 16.2 | **22.4** |
| `long_scoreboard` | 4.9 | 4.3 | **16.8** | 7.1 |
| `wait` | 2.5 | 2.2 | 2.6 | 2.5 |
| everything else, summed | 3.0 | 2.5 | 3.1 | 2.9 |
| `imc_miss` | 0.00 | 0.00 | 0.00 | 0.00 |
| `tex_throttle` | 0.00 | 0.00 | 0.00 | 0.00 |

### Step 4 -- ablations

**Primary rays are not the cost.** `debug = 1` shades the first hit and stops,
which is ablation A; it issues exactly one launch, so it has to be profiled
with `--launch-skip 0`.

| | iso_bathroom | kids_room | pine | bar |
|---|---|---|---|---|
| primary-only launch, ms | 0.27 | 1.58 | 1.09 | 0.39 |
| share of the full frame | 4.4% | 15% | 8.3% | 3.7% |
| active lanes of 32, primary only | 29.9 | 27.5 | 26.2 | 29.5 |
| `no_instruction`, primary only | 0.8 | 0.7 | 0.5 | 0.6 |
| `long_scoreboard`, primary only | 8.5 | 9.3 | **40.3** | 11.0 |

Coherent traversal runs with nearly every lane live and no issue starvation at
all; it waits on memory and nothing else. Everything in the first table --
the idle lanes, the 90% of cycles with no eligible warp -- appears only when
shading, next-event estimation and incoherent bounces are added. The frame is
the megakernel's shading code, not the BVH.

**Textures are not the cost either.** Halving and quartering every texture
(`texture_downscale` 1 / 2 / 4) moves iso_bathroom 5.90 / 6.30 / 6.20 ms and
pine 12.35 / 12.50 / 12.85 -- within the timer's 0.05 ms quantum of nothing,
and if anything the smaller textures are slower. `tex_throttle` is 0.00.

**Depth**, ms/sample, depth 1 / 2 / 4 / 8 / 16: iso_bathroom 1.20 / 3.50 /
5.90 / 8.00 / 8.90; pine 4.90 / 8.95 / 12.45 / 13.10 / 13.00. pine is flat past
the fourth bounce -- Russian roulette is already killing those paths -- while
iso_bathroom is still paying for them at 8.

**Resolution** scales linearly from 640x360 up (iso_bathroom 11.7 / 6.4 / 5.5
ns per pixel at 640x360 / 1280x720 / 1920x1080; pine 18.7 / 13.7 / 12.6), so
there is no working-set cliff between those sizes.

**Capping registers still costs more than it buys**, and now on all four:
default 255 against 128 / 96 / 64 gives iso_bathroom 5.90 / 5.90 / 6.50 / 8.15
and pine 12.60 / 13.10 / 12.90 / 16.00. `STRELKA_OPTIX_MAX_REGISTERS` stays a
diagnostic.

### Where the samples are

`__closesthit__radiance` is 84-88% of PC samples on every scene, and it is
`no_instruction` for 57-77% of them. `__intersection__light` is the only other
program worth a line, and only where there are analytic emitters with a
surface: 19.3% of kids_room's samples, 5.4% of iso_bathroom's, 91% and 87%
`no_instruction`.

By source line, merged across programs, share of attributed samples:

| line | iso_bathroom | kids_room | pine | bar | dominant stall |
|---|---|---|---|---|---|
| `random.h:278` `X ^= sb_matrix[lowestSetBit(bits)][dim]` | **22.7%** | **15.6%** | **26.7%** | **12.8%** | `long_scoreboard` ~60% |
| `random.h:261` / `:276` / `:309` (the rest of the Sobol' walk) | 9.8% | 7.3% | 10.9% | 6.5% | `short_scoreboard`, `wait` |
| `vec_math.h:559` `dot` | 5.1% | 5.3% | 4.6% | 4.7% | `no_instruction` |
| `microfacet.h:222/226/234` (the saturating product/sum guards) | 6.0% | 4.9% | 5.5% | **9.0%** | `no_instruction` ~90% |
| `material_math.h:321/331/569/578` (two-float add, `frexpf`/`ldexpf`) | 1.0% | **7.2%** | -- | 4.7% | `no_instruction` ~92% |
| `standard_pbr.h:392/396` (the lobe-weight loop) | 1.4% | 1.4% | 1.9% | 3.3% | `no_instruction` ~90% |

Sample rank is where warps wait, not what the frame costs -- this file has made
that mistake twice and the ranking is a pointer, not a bill. With that said,
**the Sobol' sampler is a third of iso_bathroom's and pine's samples and the
top line on all four scenes**, and unlike the `no_instruction` lines its stall
is a real dependent load: `sb_matrix` is a 32 KB `__device__` table, the row
index is warp-uniform but `dim` is not, so a warp scatters 32 reads across one
1 KB row and then waits for them with three other warps to hide behind.

This is the item the "Taking connectToLight apart" section below left open,
and the profile says it is now the largest single line in the launch rather
than one contributor among several. The two things named there are still the
two things to try: draw fewer numbers (four of a connection's five draws are
categorical decisions that need far fewer than 128 bits between them), or a
Gray-code Sobol' that advances by one XOR instead of walking the set bits of
the index. Both remove table reads rather than making them faster, which is
the right shape when there are four warps to hide the latency behind.

The second cluster is different in kind: `microfacet.h`'s overflow guards,
`material_math.h`'s two-float arithmetic and `standard_pbr.h`'s lobe loop are
all cheap code whose samples are 90% `no_instruction`. They are not slow; they
are what the scheduler happens to be stalled on because there is nothing to
issue. Deleting arithmetic there will not pay -- the same reasoning that made
replacing the sheen albedo lookup with a constant a 13% *regression*.

### Scene notes found while setting this up

- **`bar_max.gltf` renders almost black; `bar_max.glb` is correct.** Same
  camera, same config: the `.gltf` gives a frame with 1.6% of pixels above
  1e-6, the `.glb` 96%. `docs/bar-scene-port.md` uses the `.glb` throughout.
  Untriaged.
- **iso_bathroom's environment map cannot be loaded.** Its light sidecar names
  `/Users/ikryukov/Isometric_Bathroom_Scene/Assets/abandoned_hall_01_4k.exr`,
  which is not in `~/strelka_assets`, so the scene profiles with its three
  analytic lights and no dome. Its numbers are comparable across this file's
  runs but not to a run that has the map.
- **pine's host memory is 30.8 GB** for a 5.6 GB device scene, all of it the
  1.12 M instances and their bounds. Not a bottleneck at 62 GB of RAM, but it
  is what makes any O(instances) host pass expensive.

## Where this stands

**The table below this one is history.** Between it and now, the analytic
lights were rewritten to intersect and sample exactly (`495e2a7` and the batch
around it), and that work put back several times what the optimisation work had
taken out. Measured 2026-09-05 at 1920x1080 `max_depth` 8, the configuration
this file targets:

| | iso_bathroom | kids_room | pine_scene | chess_set |
|---|---|---|---|---|
| recorded here (2026-08-21) | 8.10 | 8.60 | 18.30 | -- |
| after the light-correctness batch | 39.3 | 216.6 | 40.5 | 12.1 |
| after the four fixes below | 31.3 | 47.4 | 30.7 | 11.6 |
| with the lights in the BVH (5) | 19.5 | 30.0 | 30.8 | 11.7 |
| with the quick factorisation (6) | 18.9 | 28.2 | 27.3 | 12.5 |

kids_room was 24x its recorded number and is now 3.5x; the residue is real work
(exact intersection, exact densities) and the shadow rays that carry it. At
1280x720 `max_depth` 4, where the entries below were found: kids_room 86.7 ->
11.2, iso_bathroom 15.0 -> 6.5, pine_scene 17.8 -> 12.5, chess_set 5.7 -> 5.5.
The ladder is unchanged across all four, all 33 rows.

What the six were, in the order they were found:

1. **The representability check ran per ray, per light.**
   `analyticEllipsoidIsRepresentable()` builds a scaled affine basis and probes
   the area density along three object axes to certify that the sampler and the
   intersector can both represent a light's transform. It is a property of the
   transform. `Scene::setLight` already ran it and already refused to enable a
   light that fails; the device ran it again for every ray-light pair. Hoisting
   it: kids_room 86.7 -> 40.0. A single sphere light cost 24.6 ms of that.
2. **Both scans over the light table were exact.** `findAnalyticAreaLightHit`
   per ray segment, `analyticLightsOccludeSegment` before traversal on every
   next-event connection. A bounding ball in front of them: 40.0 -> 22.4.
   Superseded by (5), which deleted both scans and the ball with them -- and
   that was the right end for it: the ball had a test of the usual shape over
   twenty thousand random rays, it passed, and the ball still disagreed with the
   exact intersector on 333 of kids_room's 921 600 pixels. Removing it restored
   them. Nothing has explained which rays it dropped; what is known is that a
   pre-test that is one epsilon wrong loses light silently, which is the reason
   (5)'s AABB is graded against the exact intersector's own hits.
3. **The Sobol' sampler read its direction numbers the wrong way round, and
   read too many of them.** Found by PC sampling, which is the first thing in
   this file located by asking the hardware *where* rather than *what*: one
   line held 54% of the launch's `long_scoreboard` samples. Two changes --
   storing the table `[bit][dimension]` so a warp reads consecutive words, and
   walking only the set bits of the sample index -- took kids_room 22.4 -> 19.6
   -> 19.0 and pine_scene 17.6 -> 13.9. Values are unchanged, which
   `tests/render/test_sobol_matrix.cpp` now pins; the table had no test before.
4. **A rect or disc light rebuilt its own area density per draw.** Packed into
   `pad0`, which those types write and never read.
5. **The lights were not in the acceleration structure at all.** Every analytic
   emitter was found by walking the light table: once in the raygen program to
   bound the traversal, once in the miss program to shade what the bound let
   through, and once per shadow ray before traversal. Three O(lights) walks per
   segment, in the two places a path spends its whole life. They are now custom
   primitives in the same structure as the geometry -- one AABB per light, split
   into a camera-visible GAS and a camera-hidden one so that an instance mask
   makes the visibility decision the walks used to make per ray, and
   `__intersection__light` runs the same exact intersector on the one light the
   box selected. kids_room 19.0 -> 12.1 and iso_bathroom 12.1 -> 6.8 at
   1280x720 depth 4; pine_scene and chess_set are unmoved, which is the control
   -- neither has an analytic emitter with a surface. Verified bit-exact against
   the walks in three steps (structure, then the radiance path, then the shadow
   path), which is also how the ball in (2) was caught.
6. **The affine factorisation ran per ray for lights that could not need it.**
   `scaledAffineBasis()` equilibrates a light's axes by powers of two before
   forming the determinant and the adjugate -- twelve `frexp` and twelve
   `ldexp` -- and it exists because the determinant is cubic in the axis scale:
   a light with axes of 1e13 overflows a float without it, which is what the
   1e13 and 2e18 lights in `test_light_pdf.cpp` are. Inside the range a scene
   authored in metres uses, it buys nothing. A quick path for axis scales in
   [1e-10, 1e10], with the same conditioning test deciding and the exact path
   still there for everything outside, plus carrying the adjugate in the basis
   rather than reforming it in every solve: kids_room 12.0 -> 11.2,
   iso_bathroom 6.8 -> 6.5, pine_scene 13.0 -> 12.5.
   `tests/render/test_procedural_analytic_lights.cpp` drives axis scales across
   both bounds and requires the solve to invert the axes it was handed on
   either side.

### What the profile says now, and what it does not

Re-measured 2026-09-05 after (5), one steady-state launch at 1280x720 depth 4:

| | kids_room | iso_bathroom | pine_scene |
|---|---|---|---|
| stall `no_instruction` | 30.0 | 18.1 | 16.3 |
| stall `long_scoreboard` | 4.6 | 5.5 | 18.1 |
| instruction cache hit | 99.83 % | 99.79 % | 99.89 % |
| `imc_miss` | 0.00 | 0.00 | 0.00 |
| DRAM | 15.2 % | 21.6 % | 53.9 % |
| SM throughput | 8.7 % | 9.6 % | 8.9 % |
| registers / occupancy | 255 / 32.6 % | 255 / 32.2 % | 255 / 32.8 % |
| active lanes of 32 | 13.2 | 18.6 | 12.2 |

Two different scenes: kids_room and iso_bathroom wait on instruction issue,
pine_scene waits on memory. Nothing here is an instruction-cache problem, on
the same evidence as the section below.

Where the frame goes on kids_room, by removing pieces and measuring (12.0
ms/sample at the time):

| | ms | share |
|---|---|---|
| `connectToLight` -- selection, exact sampling, densities | 4.9 | 41 % |
| the rest of the next-event scaffolding (MIS, the RIS loop) | 1.6 | 13 % |
| `bsdf_eval` inside next-event estimation | 0.9 | 7.5 % |
| tracing the shadow ray | 0.9 | 7.5 % |
| everything else (BSDF sampling, traversal, shading) | 3.7 | 31 % |

Read that against the PC samples, which rank differently: the top line by
samples is the sheen albedo table (`standard_pbr.h:278` -- 11.5 % of kids_room's
samples, 20.5 % of iso_bathroom's, 24.1 % of pine_scene's, and most of the
frame's `long_scoreboard`). Replacing the lookup with a constant makes kids_room
*slower*, 13.6 against 12.0. Sample rank is where warps wait, not what the frame
costs; this file has now made that mistake twice.

Three things measured and not worth doing:

* **A cache of the factored basis in a buffer beside the light table.** The
  natural form of (6): factor each light once on the host, index it by light id
  on the device. It is slower -- kids_room 14.3 against 12.0 -- because
  `ScaledAffineBasis` is 88 bytes and the light id is divergent, so every
  connection and every intersection pays an uncoalesced read of it instead of
  arithmetic that was already in registers. The quick path in (6) is what
  remained after this was thrown away.
* **Widening the shader-reorder hint.** `optixReorder` is worth a great deal
  already: turning it off costs kids_room 12.0 -> 15.6, iso_bathroom 6.8 -> 9.2,
  pine_scene 13.0 -> 22.9. Adding the medium flag and the bounce depth to the
  hint, 5 bits to 8, makes all three worse (12.5 / 7.2 / 13.4): the groups get
  smaller than the coherence they buy.
* **Capping registers.** `STRELKA_OPTIX_MAX_REGISTERS` 128 is worth 1.7 % on
  kids_room, 96 nothing, 72 costs 22 %. Spill is not the problem either --
  `LDL`/`STL` are 2.6-3.5 % of PC samples despite 3.2 GB of local loads and
  2.2 GB of local stores per launch.

### Taking connectToLight apart

After (6), kids_room's frame was 10.8 ms/sample and `connectToLight` was 3.6 ms
of it (33%; iso_bathroom 0.7 of 6.3). Split the same way, by removing one piece
at a time and measuring:

| | kids_room |
|---|---|
| the spherical-rectangle draw (against uniform area sampling) | 0.4 ms |
| `emittedLightRadiance` -- falloff, IES, cone | 0.3 ms |
| stating the sample's density twice | 0.4 ms |
| the Sobol' sampler's five draws | ~2.5 ms |

The third of those is fixed: `connectLight` asked `getLightPdf()` for the
density of the sample it had just taken, and that function re-derives it from
the point and the vertex -- a second `fillLightData()`, which for a sphere light
is an entire ellipsoid intersection, and a second `rectSolidAngle()`. The
sampler already had both. Now it carries the solid angle it drew over in
`LightSampleData` and the caller states the density from what it has: 10.8 ->
10.6 ms/sample on kids_room, and the frame moves by 1.5e-7 relative, which says
the two were computing the same number.

The first two are not worth anything -- the spherical rectangle is what keeps a
rect light quiet, and 0.4 ms is a fair price for it.

The fourth is the open one, and it resisted two attempts:

* **Caching the dimension-independent half of the Owen scramble in
  `SamplerState`.** `hash(seed + depth)` and the scrambled sample index are the
  same for every draw a vertex makes, and a connection makes five. Hoisting them
  is worth 23% of the frame when measured with a cheap LCG standing in for the
  sampler -- and *costs* 4% when done for real, because `SamplerState` lives in
  `PerRayData`, three more words took it from 136 to 152 bytes, and the
  continuation stack is charged per thread. 11.2 against 10.8.
* **The same cache in registers**, built at the top of `connectToLight` and
  passed down instead of stored. Also slower, 11.3: at 255 registers per thread
  there is nothing to spend.

What is left to try is reducing the *number* of draws -- four of the five are
categorical decisions (environment or local, mesh or analytic, which bucket,
which alias) that between them need far fewer than 128 bits -- or a Gray-code
Sobol' that advances by one XOR instead of walking the set bits of the index.

### The stall is not what this file used to say, and not what it looks like

The counters have inverted since the table below was taken. iso_bathroom, one
steady-state launch at 1280x720 depth 4, before today's fixes:

| | recorded (2026-08-21) | 2026-09-05 |
|---|---|---|
| DRAM throughput | 44.6 % | **9.5 %** |
| L2 hit rate | 93.1 % | 98.4 % |
| stall `long_scoreboard` | 9.7 | **2.2** |
| stall `no_instruction` | 13.5 | **36.1** |
| SM throughput | 11.8 % | 7.4 % |

`no_instruction` reads as "the instruction cache is thrashing". It is not:

- **`imc_miss` is 55 PC samples out of 1 864 240.** Instruction fetch misses
  effectively do not happen.
- **Making the module 35% smaller changed nothing.** Compiling the OpenPBR path
  out statically took the module from 12 MB to 7.8 MB and the frame from 22.2
  to 22.2 ms.
- **Moving cold code out of line changed nothing.** `__noinline__` on the whole
  cold ellipsoid chain: 22.2 -> 22.2. An OptiX direct callable for the same
  code, program group and SBT record and all: 22.2 -> 22.3, and the
  continuation stack grew 2400 -> 2432 B. Both are reverted; the machinery is
  not worth carrying for nothing.

What is left is the shape of the launch: 255 registers per thread, 32%
occupancy, 15.5 of 32 lanes active. Two or three warps per scheduler, half
their lanes idle, and nothing else to issue the moment one of them waits.
`no_instruction` is what that looks like from the counter's side.

The register ceiling is the direct lever on it, and **its sign has flipped**
since the note further down this file recorded that capping at 96 costs 11%.
After the four fixes, at 1280x720 depth 4:

| cap | iso_bathroom | kids_room | chess_set |
|---|---|---|---|
| 255 (default) | 12.1 | 19.5 | 5.2 |
| 96 | 11.7 | 18.9 | 6.5 |
| 64 | 12.2 | 18.2 | 9.0 |

It buys 3-7% on the rooms and costs chess_set 25-73%, which is the scene whose
DRAM throughput is 65%: more warps help a launch waiting on latency and hurt
one already saturating bandwidth. `STRELKA_OPTIX_MAX_REGISTERS` stays a
diagnostic rather than becoming a default, and picking it per scene from the
first frames' counters is the version of this that would pay.

Nsight Compute cannot open a section on an OptiX launch, and on AD102 it cannot
PC-sample `no_instruction` at all -- the metric does not exist for the sampler,
which is consistent with a warp that has no instruction having no address to
attribute. The reasons it *can* sample are named explicitly in
`/tmp/ncu_pcsamp.sh`; `-lineinfo` is now on in the OPTIXIR and
`STRELKA_OPTIX_LINEINFO` turns on the module debug level that keeps the line
table, both measured free.

---

## Where this stood in August


The target is the interactive render: 1920x1080, one sample per launch,
`max_depth` 8 -- what the editor asks for on every frame. ms/sample, best of
three:

| | iso_bathroom | kids_room | pine_scene |
|---|---|---|---|
| before | 10.60 | 11.90 | 23.90 |
| final | **8.10** | **8.60** | **18.30** |
| | **-24%** | **-28%** | **-23%** |

Five things were built and measured. Four landed; the fifth is written up under
"Payload registers" below because the negative result is the useful part. The
ladder (`tools/parity/run_ladder.py`, 29 rows) is unchanged to the digit across
all of them.

Per step, at 1280x720 depth 4 where the earlier work was measured:

| | iso | kids | pine |
|---|---|---|---|
| before | 4.40 | 5.00 | 11.20 |
| no diffuse/specular split AOV | 4.00 | 4.70 | 10.90 |
| + pipeline specialisation | 3.30 | 3.60 | 9.20 |
| + PerRayData 208 -> 176 B | 3.20 | 3.60 | 9.00 |
| + PerRayData 176 -> 160 B | 3.20 | 3.60 | 8.60 |
| + PerRayData 160 -> 136 B | 3.20 | 3.60 | 8.50 |

The three PerRayData steps in order: packing the counters and flags into one
bit-field word and dropping two fields that duplicated the sampler and the
launch index; then moving the radiance cache's per-path bookkeeping into its own
per-pixel buffer, the way Metal has always held it. Note where they land -- the
two room scenes stopped responding to struct size once they became
instruction-cache bound, and only pine, which is bound by DRAM, kept paying out.

What the whole set did to the iso bathroom's counters at 1280x720: DRAM
throughput 44.6% -> 20.8%, local loads 2.21 -> 1.26 GB per launch, local stores
2.15 -> 1.33 GB, global stores 920 -> 723 MB, `long_scoreboard` 9.7 -> 5.7 cycles
per issued instruction, `no_instruction` 13.5 -> 11.2.

**A specialised pipeline does not render bit-identically, and that is expected.**
The renderer is deterministic -- two runs of one binary are bit-identical, checked
-- but a module compiled against different constants folds them differently and
contracts different multiply-adds, and a path tracer amplifies that: one random
number landing on the other side of a threshold is a different path. Measured
before/after at 32 spp: signed mean difference 0.005% of the image mean on
pine_scene, 0.0002% on the two rooms, with 11% of pine's pixels moving by more
than 1e-3 -- scattered, zero-mean, and gone by the 128 spp the ladder grades at.
Grade this backend against the ladder and against converged means, not against a
hash.

The one knob that did not move is the register ceiling
(`STRELKA_OPTIX_MAX_REGISTERS`, and the measurement is in the comment beside it
in `OptixRender.cpp`): capping at 96 raises occupancy from 32.9% to 41.0% and
costs 11%, because the launch is memory bound rather than latency starved.

### Payload registers: measured, and it loses

Worth stating plainly because the opposite is the natural assumption, and it was
mine: **an OptiX payload value is not a register across a traversal.** It is
preserved in the continuation stack, and it costs more there than a struct field
does.

Built in full -- the hot half of the path state (the two colours, the next ray,
the pdf, the distance and the packed counters) moved into fifteen payload words,
`numPayloadValues` 2 -> 17, the shading programs split into a body and a wrapper
so the write-back happens once instead of at a dozen early returns. PerRayData
went from 136 bytes to 72. The continuation stack went *up*, 480 -> 544 bytes,
and the frame got slower: +2.5% iso, +3.4% kids, +7.7% pine.

The mechanism, isolated afterwards in one line: raising `numPayloadValues` from
2 to 17 and **not using the extra words at all** takes the continuation stack
from 480 to 608 bytes. That is ~8.5 bytes of stack per payload word -- more than
the 4 bytes the same value costs as a struct member, because it is saved and
restored around both `optixTraverse` and `optixInvoke`.

So the split does not exist to be re-attempted. `STRELKA_PAYLOAD_COUNT` stays at
2 and carries this note.

## OpenPBR: free when off, and what it costs when on

The OpenPBR material path is behind a function constant (`kFcOpenPBR`,
`WavefrontFeatures::kOpenPBR`), on the argument that a scene without an OpenPBR
material must compile a kernel that does not contain it. That argument was made
from the instruction-cache findings below; this is the measurement of it.

**Not the RTX 4090 numbers above.** Those are OptiX. This is an M4 Pro on Metal,
so the only comparison that means anything here is HEAD against HEAD-plus-the-
feature on the same machine in the same session.

1920x1080, `max_depth` 8, sobol, one sample per launch, median ms/sample over
samples 8..24 of a 24-sample render, best of three:

| | iso_bathroom | kids_room |
|---|---|---|
| e5684bd (before the feature) | 83.4 | 122.6 |
| with the feature, bit **off** | 80.6 | 122.5 |
| with the feature, bit **on** | 177.7 | 199.3 |

**Off costs nothing measurable.** -3.4% and -0.1% against a run-to-run spread of
about 3% (three consecutive runs of one binary on iso_bathroom: 87.6 / 84.7 /
85.8 for HEAD, 84.6 / 85.3 / 82.8 for the feature build). The honest statement is
not "free" but "below what this measurement can resolve" -- and the images are
bit-identical on four of five ladder rows, with the fifth differing by half a
half-float ULP, so there is no mechanism for it to be otherwise.

Worth stating what "off" does *not* mean: the metallib still carries the whole
OpenPBR implementation, about 264 KB of it, most of that lookup tables. That is
data rather than instructions, which is why it does not show up here -- the
tables are only read by a kernel specialised with the constant on.

**On costs 2.1x and 1.6x.** That figure is `render/material/model = openpbr`,
which routes *every* material in the scene through the OpenPBR BSDF -- a switch
that exists so one asset can be rendered both ways and compared, not a production
mode. A scene where only some materials are OpenPBR pays in proportion. The two
rooms differ because kids_room spends more of its frame in traversal, so a more
expensive shade kernel dilutes.

No attempt has been made to reduce it. The obvious lever is Adobe's own
specialisation constants -- `EnableSheenAndCoat`, `EnableDispersion`,
`EnableTranslucency`, `EnableMetallic` -- which are wired to a macro that
currently answers `true` unconditionally. Packing a scene-wide scan of which
lobes any OpenPBR material actually uses into four more function constants is
the same trick this file already records as worth 15-23% of the frame.

## The measurement everything below is read against

RTX 4090, OptiX 9.1, Release, sobol, one sample per launch. Scenes are the three
in `~/strelka_assets`. Timings are the median `ms/sample` over samples 8..24 of a
24-sample render; counters come from one steady-state `optixLaunch`.

Two resolutions appear here. **1920x1080 at `max_depth` 8 is the target** -- it
is what the editor asks for on every frame, and it is what the numbers in "Where
this stands" are. The table just below, and the entries that were found against
it, are at 1280x720 `max_depth` 4, which is where this started; the ordering of
the findings is the same at both, but the two room scenes cross from
memory-bound to instruction-cache-bound somewhere between them, so read the
stall columns rather than transplanting a conclusion.

Nsight Compute cannot open a *section* on an OptiX launch -- it is a "cmdlist
workload" and every `--section` / `--set` request silently profiles nothing.
Metrics have to be named explicitly, and the launch selected by ordinal
(`-s <n> -c 1`), because `-k optixLaunch` does not match it either. Enumerate
ordinals with `--metrics gpu__time_duration.sum -c 1200 --csv` and look for the
`optixLaunch` rows; on scenes with many acceleration builds the ordinal moves
between runs, so re-enumerate rather than reusing a number.

| | iso_bathroom | kids_room | pine_scene |
|---|---|---|---|
| ms/sample | 4.5 | 5.1 | 11.2 |
| SM throughput | 11.8 % | 10.8 % | 6.5 % |
| DRAM throughput | 44.6 % | 48.8 % | 65.3 % |
| L2 hit rate | 93.1 % | 91.8 % | 81.4 % |
| DRAM read / write per launch | 0.66 / 1.34 GB | 0.99 / 1.46 GB | 5.09 / 2.15 GB |
| local ld / st per launch | 2.21 / 2.15 GB | 2.48 / 2.35 GB | 2.85 / 2.55 GB |
| global ld / st per launch | 3.23 / 0.92 GB | 3.58 / 0.96 GB | 5.84 / 1.00 GB |
| registers / thread | 255 | 255 | 255 |
| achieved occupancy | 29.9 % | 31.6 % | 32.8 % |
| active lanes per instruction (of 32) | 21.7 | 19.4 | 15.7 |
| global store sectors / request | 19.1 | -- | 13.2--15.6 |
| stall `no_instruction` | 13.5 | 12.5 | 11.9 |
| stall `long_scoreboard` | 9.7 | 12.1 | 38.1 |

Those are the *before* numbers, kept because every entry below was found against
them. None of the three is compute bound. What the entries share is that the SM
sits at 6--12 % while every issued instruction waits 28 (iso, kids) to 55 (pine)
cycles.

Two mechanisms behind that, both now measured rather than inferred:

- **`no_instruction` is the instruction cache**, and on the room scenes it was
  the single largest stall -- roughly half of all of it. The mega-kernel carried
  every feature whether or not the scene used one. Compiling the launch
  parameters that select features in as constants
  (`OptixModuleCompileBoundValueEntry`, see `OptiXRender::PipelineSpec`) was
  worth 15--23% of the frame on its own. The first compile of a given
  specialisation costs about two seconds; OptiX's own disk cache makes every one
  after that 4 ms, including across processes.
- **`PerRayData` sizes the continuation stack, byte for byte** -- 208 B of
  struct gave a 576 B stack, 464 gave 832. That stack is per thread and it is
  local memory, so the struct sets the launch's whole local working set. Adding
  256 bytes of ballast the path never reads measured 39--61% slower; removing
  the `= {}` with the ballast still there gave none of it back, so it is the
  size and not the zeroing.

## Ruled out, so nobody re-measures them

- **Hair / curves.** `kids_room` 5.1 ms/sample, `kids_nohair` 4.9. The groom is
  4 % of the frame.
- **Texture resolution.** `pine_scene` with `texture_downscale` 1 vs 4:
  11.2 vs 11.0 ms/sample. Its 4.46 GB of textures are not what the DRAM read
  traffic is.
- **`texture_lod`, `sharc`, `opacity_micromaps`** on pine: 11.0, 11.0, 11.2 --
  all inside run-to-run noise. `sharc` has since been re-measured against a cache
  that actually persists between frames (it used to be cleared on every
  accumulation restart, so it never held more than one frame). It now cuts the
  scene's deep-bounce heatmap by 29% and still does not move the frame time:
  400 spp, depth 8, 5.7-6.2 s off against 5.8 s on. That is consistent with the
  rest of this file rather than surprising -- pine stalls on `long_scoreboard`
  at 38.1 cycles with the SM at 6-12%, so removing traversal work removes
  something the frame was not waiting on. See `docs/open-defects.md` entry 12 for
  the full table, and for the warning about measuring any of this on runs too
  short for the GPU clocks to come up. Micromaps in particular build nothing there: all
  144 meshes log `opacity micromap resolves nothing, skipped`, which is a
  separate question (does that scene have alpha cutouts at all?) rather than a
  cost.
- **`sort_rays`.** It is the Metal wavefront's key. OptiX reorders
  unconditionally (`params.enableShaderReorder`, gated only on hardware support
  and on the `render/pt/shaderReorder` kill switch), so the flag measures
  nothing on this backend.

---

## 1. The last 36 bytes of PerRayData are the nested-dielectric stack

**Measured, and left alone because the return does not justify a tri-platform
change.** PerRayData is 136 bytes. The largest single item left in it is
`IorStack` at 36 -- four entries of `{float ior; uint32_t packed}` plus a top
index -- carried by every path in every scene whether or not it ever enters a
dielectric.

It can be 20 bytes. Both stored fields are pure functions of the material index:
`si.ior` is `MaterialParams::ior` copied straight through (`bsdf.h`, three sites,
no texture) and the priority likewise. An entry could be the material index
alone, with the accessors reading ior and priority back from the material table
the shading path already has in hand.

What stops it is the ratio. `ior_stack.h` is one of the genuinely tri-platform
headers, so `ior_stack_current_ior`, `_pop` and `_peek_after_pop` would all take
the material table, and every Metal call site would have to change -- on a
machine that cannot build Metal. The failure mode is at least loud (CI builds
macOS), but the payoff is small: 16 bytes, and the slope below puts that at
0.3% / 0.9% / 1.5% on the three scenes.

**The slope, measured at the target configuration** by adding 64 bytes of
ballast the path never reads: +1.2% iso, +3.4% kids, +6.0% pine. The cost of a
byte is real but it is convex and we are on the flat part -- and the two room
scenes have stopped responding to struct size at all, because they are now bound
by the instruction cache rather than by memory (`no_instruction` 13.6 and 12.6
against `long_scoreboard` 5.3 and 6.2). Only pine still pays out, and pine's real
problem is item 2.

There is also one free-looking 4 bytes: `float2 pixelSample` wants eight-byte
alignment, so the struct's 132 bytes of members round up to 136. Two floats
instead would recover it, at 0.1--0.4%, which is not worth turning a natural
float2 into a pair.

**Already ruled out:** payload registers (see "Payload registers" above -- they
cost more stack, not less); the zero-initialisation; and a lower register
ceiling, which trades the spills back and loses.

## 2. pine_scene is 50.8 M unique triangles that should have been instances

**Measured.** `long_scoreboard` is 38.1 cycles per issued instruction, three
times either room scene, and DRAM read is 5.09 GB per launch at 65 % of peak
with the L2 hit rate down at 81 %. This one is not the shader.

Reading the glTF settles where it comes from:

```
meshes: 292   nodes with mesh: 433   unique meshes referenced: 275
top reuse: every entry x2, nothing higher
triangles (unique meshes): 50 788 404
```

`EXT_mesh_gpu_instancing` is in `extensionsUsed`, and effectively unused: no
mesh is referenced more than twice. A forest of individually-unique trees is
1.54 GB of vertices, 0.56 GB of indices and 1.44 GB of BLAS that has to stream
through L2 on every bounce.

**Already ruled out:** textures (see above). The acceleration structures are
built with the right flags -- `accel_build_policy.h` gives static geometry
`PREFER_FAST_TRACE | ALLOW_COMPACTION`, and `PREFER_FAST_BUILD` is confined to
skinned meshes, which this scene has none of.

**What to do.** Deduplicate identical trees and ferns into instances; that cuts
the vertex and BLAS footprint and is the only change that moves the L2 hit rate.
Quantising vertex attributes (oct32 normal, half2 uv) would take `Scene::Vertex`
from 32 B to 20 B, but that struct is pinned at 32 B by the Metal kernels, which
read attributes at hardcoded byte offsets -- `tests/scene/test_vertex_packing.cpp`
guards it -- so it is a both-backends change, not a pine fix. LOD for distant
trees is the third lever.

## 3. Next-event estimation is 38 % of the frame, everywhere

**Measured.** `estimator_mode = 1` (BSDF sampling only) against the default
NEE + MIS, same depth:

| | NEE + MIS | BSDF only | NEE's share |
|---|---|---|---|
| iso_bathroom | 4.5 | 2.8 | 1.7 ms, 38 % |
| kids_room | 5.1 | 3.1 | 2.0 ms, 39 % |
| pine_scene | 11.0 | 7.7 | 3.3 ms, 30 % |

That is a fair price for the variance it removes -- the entry is here because
the number is stable across three very different scenes, which makes it the
largest single *feature* cost in the renderer and therefore where a better
estimator would pay.

`risCandidates` already exists and defaults to 1; `docs/open-defects.md`
("RIS vs NEE") records that RIS costs ~40 % more per sample and does not move
the answer, which is why. Reservoir reuse across pixels and frames is the
version of that idea which pays for itself, and nothing here has measured it.

## 4. Scene load spends 124--144 ms in acceleration builds

**Measured**, from the nsys CUDA kernel summary (this is startup, not a
per-frame cost -- it does not appear in the `ms/sample` numbers above):

| | `optixAccelBuild` | instances | `optixAccelCompact` | peak scratch |
|---|---|---|---|---|
| iso_bathroom | 26 ms | 121 | 0.7 ms | -- |
| kids_room | 124 ms | 394 | 5.4 ms | 2.74 GB |
| pine_scene | 144 ms | 302 | 5.0 ms | -- |

One build and one compaction per mesh, with the largest single build at 40 ms
(kids) and 14 ms (pine). `optixAccelBuild` takes an array of build inputs;
batching meshes into fewer calls, and compacting in fewer passes, is what turns
this into editor open-latency rather than a wait. kids_room's 2.74 GB of accel
scratch is the same shape of problem seen from the memory side -- it is
transient, and it is the largest single allocation the scene makes.

Sliced upload (`MetalScenePreparation` / `OptixScenePreparation`) already hides
some of this behind a first frame; the entry is about the total, which it does
not reduce.

---

## Asset note: the iso bathroom renders without its dome light

`iso_bathroom_light.json` names
`/Users/ikryukov/Isometric_Bathroom_Scene/Assets/abandoned_hall_01_4k.exr`, an
absolute macOS path that does not resolve here, and the load fails with
`code(-7)`. The scene still renders, lit only by its local lights. Every
iso_bathroom number in this file was taken in that state -- which is a fair
measurement of *that* scene, but it is not the scene the light sidecar
describes, and the environment lookup in `__miss__ms` is doing less work than it
would with a real map.
