# Open performance items

Keep only actionable measurements here. Closed investigations belong in Git
history; use `git show 0e03a4b^:docs/open-perf.md` for their full records.

## Measurement contract

- Measure Release builds with validation and audit counters disabled.
- Record scene, resolution, depth, samples per launch, device and median launch time.
- Use Xcode Metal capture/counters on Apple GPUs and Nsight Systems/Compute on NVIDIA.
- Validate stochastic changes by image statistics and noise, not exact pixels.

## Current priorities

### Pine geometry and alpha traversal

Pine is memory-latency bound rather than instruction-fetch bound. At 1280x720,
depth 4 and 16 samples per launch, the recorded OptiX launch consumed 90.6 GB
of DRAM traffic, reached 62.5% of peak DRAM throughput and spent 27.1 cycles per
issued instruction on `long_scoreboard`. Its 275 referenced unique meshes contain
50.8 M unique triangles; no mesh is reused more than twice.

Hardware opacity micromaps at level 4 reduced a 1080p/24-spp launch from 16.470
to 16.020 ms and removed about 117 M executed instructions, but added roughly
67 seconds of classification and 76 MiB of maps. Keep them opt-in until their
build is cached or amortised. Software alpha microgeometry is rejected: adaptive
subdivision through level 3 reduced unknown coverage only 90.91% to 90.18% while
growing surviving geometry 1.63x.

Next useful experiments must remove a dependent traversal/fetch or improve
global ray coherence. Merely repacking an L2-resident alpha record, changing
vertex load width or sorting 64 rays within a threadgroup did not move time.

### Metal wavefront traversal

Continue measuring `wavefrontExtendStatic` and `wavefrontShadowStatic` separately.
The static primitive-BLAS path, compact static acceleration structures and the
external 12-byte cutout record are the baseline. Prefer reductions in returned
alpha candidates, live state and dependent geometry fetches over extra local
sorting passes.

### OpenPBR live state

The specialised OptiX glTF pipeline reduced continuation stack use from 2272 to
288 bytes in scenes without OpenPBR. An OpenPBR pipeline still uses about 1760
bytes. Establish which parts of the 272-byte `OpenPBRParams` and 720-byte
prepared BSDF cross a traversal, then split only the live subset.

`SurfaceInteraction` is 268 bytes. About 84 bytes are fields for sheen,
iridescence, clearcoat, subsurface and diffuse transmission. Specialise or split
these only after a profile shows that their lifetime, rather than execution of
the active lobes, is limiting the target scene.

### Next-event estimation

NEE plus MIS costs 30-39% of the measured frames, but pays for itself in
variance. The recorded depth-matched costs were 1.7 ms for iso_bathroom, 2.0 ms
for kids_room and 3.3 ms for pine. Raising independent RIS candidates was slower;
future work should test spatial or temporal reservoir reuse rather than more
independent candidates.

### Scene-open latency

Recorded OptiX acceleration builds cost 26 ms for iso_bathroom, 124 ms for
kids_room and 144 ms for pine. kids_room reached 2.74 GB of transient scratch.
Batching build inputs, reusing scratch and compacting in fewer passes are the
remaining engine-side opportunities. Sliced scene preparation improves time to
first frame but does not reduce total work.

## Current reference measurements

The latest retained OptiX comparison uses 1280x720, depth 4 and 16 samples per
launch:

| change | kids_room | iso_bathroom | pine |
|---|---:|---:|---:|
| scene-bound feature specialisation | -5.8% | -8.4% | -8.7% |
| move next-bounce preparation before NEE | -29% total | -32% total | -25% total |
| prepare the glTF lobe stack once | 76.1 to 70.8 ms | 41.7 to 39.7 ms | 109.5 to 109.2 ms |

The rooms are issue-starved; pine waits on memory. Do not infer that an
optimisation measured on one limiter transfers to the other.

## Rejected experiments

| Experiment | Result |
|---|---|
| Load each 32-byte vertex as two vectors | Fewer requests, no pine speedup |
| Fetch only UV before a cutout decision | Less vertex work, no speedup |
| Duplicate exact 12-byte cutout UV on OptiX | Bit-identical, time flat |
| Remove hit position and reconstruct it | Slower and changed trajectories |
| Store only tangent handedness | Bit-identical, time flat |
| Integer glTF lobe CDF | Slower |
| Split medium state from OptiX per-ray data | Less traffic, no useful time change |
| Cap opacity micromaps at level 3 | Less memory, lost half the speedup |
| Discard sparse opacity maps | Less memory, slower |
| Specialise analytic-light shapes | Smaller module, no stable speedup |
| Limit OptiX registers to 96 | Scene-dependent; keep as diagnostic only |
| Per-pixel external IOR stack | More memory and slower chess_set |
| Downscale pine textures | No meaningful change |
| SHaRC on pine | Less deep-bounce traversal, frame time unchanged |
| Local octant shadow-ray sort | Slight regression at depth 1 and 8 |
| Software alpha subdivision/contours | Geometry growth overwhelms candidate reduction |

## Known measurement caveat

The iso_bathroom light sidecar names an unavailable absolute-path environment
map. Existing iso_bathroom numbers therefore cover its analytic lights without
the intended dome light.

## 2026-09-14: compact OptiX alpha primitive inputs

Target profile: RTX 4090, OptiX 9.1, CUDA 13.4, 1920x1080, depth 8, Sobol,
one sample per `optixLaunch`. Nsight Systems medians are over 24 launches;
Nsight Compute profiles launch 9 after eight warmups with line information.

Pine's alpha any-hit used three index reads and three random 32-byte vertex
records to recover 12 bytes of packed UV. A dense, exact 12-byte UV record for
each cutout triangle removes both indirections. Only 1,229,459 of the scene's
roughly 50 million triangles need it: 14.8 MB and 8 ms of scene-build time.

| pine | disabled | compact | change |
|---|---:|---:|---:|
| Nsight Systems median | 16.503 ms | 16.267 ms | -1.43% |
| NCU duration | 17.277 ms | 17.080 ms | -1.14% |
| DRAM read | 8.295 GB | 8.170 GB | -1.50% |
| DRAM write | 2.132 GB | 2.101 GB | -1.46% |
| instructions | 1.813 G | 1.792 G | -1.18% |
| long-scoreboard cycles / issue | 26.75 | 26.32 | -1.60% |
| local load / store sectors | 129.1 M / 94.7 M | 129.3 M / 94.8 M | unchanged |

The 24-spp output is bit-identical. In PC sampling, radiance any-hit falls from
12.4% to 10.9% and occlusion any-hit from 3.6% to 3.0%; the old random vertex
fetch leaves the top source lines. `STRELKA_NO_PRIMITIVE_ALPHA_DATA=1` is the
same control switch on Metal and OptiX.

The launch remains latency-bound: 61% DRAM versus 8% SM throughput, 91.6% of
scheduler cycles with no eligible warp, 255 registers, and unchanged local
traffic. The remaining alpha texture fetch is 4.1% of attributed source samples
and 100% long-scoreboard. OMM removes 32--38% of the any-hit PC samples and
about 3.4% of launch time, but its current 67-second preprocessing cost makes
serialization/caching that build, rather than enabling it by default, the next
OMM problem.

A broad CUDA port of Metal's finite-input assumptions was rejected: it removed
9.3% of instructions but increased DRAM writes 37.7%, local traffic about 32%,
long-scoreboard 26.75 -> 39.14, and frame time 4.8%. Narrow bound-value probes
for all-BLEND and unit base alpha were also neutral or slower; do not combine
scene facts merely because they are true.
