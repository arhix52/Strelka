# Open performance items

The same discipline as `open-defects.md`, for cost rather than correctness: each
entry is something measured, with the measurement that found it and what has
already been ruled out, so a reader starting cold can act without repeating the
elimination. What is closed is here too, because the *why* is what the next
person needs and it is the same why either way.

## Where this stands

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
