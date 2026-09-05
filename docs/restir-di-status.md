# ReSTIR DI memory/pass status

Phase: BASIC bias correction complete; OFF preserves the prior estimator. Procedural lights and the one-final-shadow rule are unchanged.

Measurement: Apple M4 Pro, Release, 1920x1080, depth 4, 1 spp/frame; 8 warm-up + 32 measured frames; mean per launch, median 5 launches. MTL validation/audit disabled for timing.

## Buffer ledger

| buffer | B/px | fields | writer -> readers | lifetime | elsewhere / decision |
|---|---:|---|---|---|---|
| reservoir x2 | 64 | type+stable light ID, 3 exact sample payload words, weightSum, target, M, age/valid | shade/fused -> temporal/fused | two frames | world position/normal/radiance/PDF reconstructed; 48 -> 32 B each |
| surface history x2 | 40 | geometry normal, view depth, material ID+valid | shade -> temporal/fused | two frames | AOV is optional and not a stable ReSTIR source; world position removed; 32 -> 20 B each |
| shading point | 104 | position, Ns/T/sign, ray, color, throughput, UV, IOR/LOD/radius, medium/sample flags | shade -> fused | one frame | Ng/material come from history; B reconstructed from Ns/T/sign; 136 -> 104 B |

Cached option A (104 B) is retained: it removed duplicate Ng/material/bitangent and produced byte-identical output. Option B needs retained instance/geometry/primitive/barycentric IDs plus triangle/curve vertex and material fetches after the hit buffer becomes immutable initial-reservoir storage; no correctness-preserving smaller variant avoided those random reads, so it was not shipped.

Total ReSTIR storage: 296 -> 208 B/px, 585.4 -> 411.3 MiB at 1080p. Reservoir target met: 96 -> 64 B/px. ReSTIR OFF allocates 0 ReSTIR bytes.

## Pipeline

First-hit shade still performs initial RIS + temporal reuse. It writes the immutable initial reservoir into the dead 32-B hit record and appends only valid first non-delta hits to the reused miss queue. One indirect `wavefrontRestirSpatialFinal` reads that queue, merges neighbors in registers, writes history, reconstructs the selected sample, and enqueues at most one shadow ray.

ReSTIR dispatches/frame: spatial+final 2 -> fused 1; total depth-4 dispatches 32 -> 31. The spatial intermediate buffer/pass and separate final PSO are gone. Active-queue smoke: 76,018 active / 76,032 dispatched versus 76,800 pixels; misses do no reuse work.

## GPU timing (before -> after, ms)

| scene | NEE | c1/T/S2 | c2/T/S2 | c2 overhead reduction |
|---|---:|---:|---:|---:|
| Uniform grid | 65.06 -> 61.94 | 95.10 -> 86.99 | 104.00 -> 95.27 | 14.4% |
| Distributed lights | 74.52 -> 69.43 | 88.20 -> 75.94 | 97.87 -> 81.54 | 48.1% |
| Occluded lights | 27.14 -> 22.55 | 30.88 -> 24.38 | 31.14 -> 24.62 | 48.2% |
| 512 moving lights | 51.36 -> 53.30 | 74.65 -> 62.56 | n/a -> 64.18 | 60.3% (c1) |

No primary scene regressed >3% versus its ReSTIR baseline. The 25% overhead goal is met on distributed, occluded, and moving, not uniform: its remaining cost is selected-sample BSDF/material reconstruction plus the second candidate.

Metal 4 exposes whole-command-buffer GPU intervals, and temporal is fused into shade, so splitting first-hit/initial/temporal timestamps would perturb the pipeline. Controlled attribution accounts for 100% of current c2-NEE delta: c1 reuse/fused/shadow + candidate 2 = 25.04+8.28 ms uniform, 6.52+5.59 distributed, 1.83+0.24 occluded, 9.26+1.62 moving.

## Quality and invariants

Equal-time linear EXR against one 512-frame NEE reference; budget = 128 NEE frames. Values are frames / rMSE / mean luminance ratio:

| scene | NEE | c1/T/S2 | c2/T/S2 | crossover |
|---|---:|---:|---:|---|
| Uniform | 128 / .0811 / .9998 | 91 / .1109 / 1.0594 | 83 / .1040 / 1.0248 | none |
| Distributed | 128 / .5556 / .9999 | 117 / .5225 / 1.0031 | 108 / .5506 / 1.0019 | c1; c2 marginal |
| Occluded | 128 / .1612 / .9987 | 118 / .2692 / 1.1503 | 117 / .1848 / 1.0151 | none |

Optimized vs pre-fused mean ratio is 0.99999997, so these changes introduce no systematic shift; equal-time c2 stays within 2.5% of its reference mean. Moving uses equal frame count, not equal time: existing light-change rejection yields 0 temporal merges, so reuse-induced temporal error is zero; GPU timings are reported separately above.

Audit: candidates/reuse queries 0; first-bounce NEE 0 with ReSTIR; final rays <= eligible; guides add 0 traversal; static BLAS/TLAS build/refit 0; moving 32 frames gives build 0/0, refit/upload/mapping 32/32/32 (max 1/frame); NEE ReSTIR allocation 0; history ping-pong observed via nonzero static temporal merges.

Validation: Debug/Release Metal compile; actual MTLDevice static/active/moving runs; focused 31 cases/72,689 assertions; full Debug ctest 3/3, 943 cases and 67,159,707 assertions. Fused vs pre-fused image rMSE 4.8e-6, mean ratio 0.99999997; cached compaction and redundant-write removal are byte-identical.

Commits: `7b0292e` Pack ReSTIR reservoir and history; `d182945` Fuse spatial and final ReSTIR passes; `ef58741` Add ReSTIR basic bias correction. Dominant bottleneck: 104-B same-frame shading cache and selected-sample BSDF/material evaluation.

## Canonical OFF/BASIC validation (2026-09-05)

Both production modes clamp imported temporal M to `maxHistoryLength * currentM`; candidate generation, rejection, RNG, accepted sources, dispatches and visibility work are identical. OFF finalizes with `1/M`. BASIC follows RTXDI 3.1 (`f12037f`): `pi / sum(Mi*pi)`, with source targets evaluated on their own current/previous surfaces and no visibility rays.

The old .5225/.2692 rMSE values are valid linear HDR. The newer .0711/.0553 values are invalid: that command omitted `--tonemap none`, clipped EXR to 1, then labelled it linear. The canonical harness now requires linear float EXR, all-pixel ROI and one reference per scene.

Target surface ledger: position, ray, Ng/Ns/frame, material/UV/LOD, throughput, geometry/instance/primitive identity and sample/medium. A 64-B union stores an exact evaluated core closure when possible, otherwise identity+barycentrics for exact geometry/material refetch. Two records replace the old 2x104-B cache: BASIC 232 B/px (458.8 MiB), OFF 208 B/px, NEE 0.

1080p, depth 4, three disjoint 128-frame Sobol windows, 512-frame NEE reference:

| scene | NEE mean/rMSE | OFF c1 | BASIC c1 | BASIC c2 | GPU ms NEE/OFF/B1/B2 |
|---|---:|---:|---:|---:|---:|
| Uniform | 1.0000/.0811 | 1.0133/.0821 | 1.0135/.0821 | 1.0063/.0811 | 73.1/102.4/116.1/126.4 |
| Distributed | 1.0001/.6107 | 1.0048/.5475 | 1.0049/.5475 | 1.0027/.5463 | 74.2/81.4/84.8/93.2 |
| Occluded | .9998/.1606 | 1.0528/.1837 | 1.0992/.2171 | 1.0441/.1737 | 23.5/25.4/25.6/25.9 |

OFF/BASIC Debug work audit: 0 candidate/reuse queries, identical accepted merges and dispatches, final rays <= eligible. Compact vs 104-B BASIC: Uniform rMSE 1.3e-6; mixed diffuse/glossy/transmission rMSE 9.6e-9; mean ratio 1.0000000.

OPEN: BASIC does not reduce c1 mean bias on the unclipped Occluded reference and Uniform overhead is +13.7 ms, not <=9 ms. No heuristic or RAY_TRACED correction was added.
