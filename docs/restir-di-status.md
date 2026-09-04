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

## Static reuse bias correction (2026-09-05)

`restirBiasCorrection = off | basic`. OFF is RTXDI OFF: source targets affect merge weights, but normalization evaluates only the selected sample at the current surface. BASIC evaluates it at current plus every compatible temporal/spatial source and uses `W = wSum*pSelectedSource/(pCurrent*sum(Mi*pi))`; RAY_TRACED is not implemented.
Target audit: initial, temporal, spatial-source and final all call `evaluateRestirConnection`; reconstructed position, Ng/Ns, textured material/UV/LOD, BSDF, light radiance/PDF, sidedness and MIS are identical. Diffuse, glossy and transmission smoke passed; complementary BSDF-hit MIS was unchanged. Light/material/geometry edits invalidate history.
The 20-B history cannot rebuild an exact textured BSDF. BASIC ping-pongs the existing 104-B exact shading record: 312 B/px, 617.0 MiB at 1080p (+104 B/px, +205.7 MiB); OFF stays 208 B/px. No visibility is evaluated during reuse.

Three fixed 128-frame Sobol windows, 320x240, long NEE reference. Cells are c1 | c2; merges are accepted T/S per frame (k), then mean effective M:

| scene/stage | mean ratio / rMSE | T/S k | M |
|---|---|---:|---:|
| Uniform I | 1.0000/.0810 \| .9999/.0806 | 0/0 \| 0/0 | 1.00 \| 2.00 |
| Uniform T | 1.0162/.0888 \| 1.0060/.0857 | 31.0/0 \| 33.0/0 | 7.88 \| 16.16 |
| Uniform S2 | 1.0729/.1101 \| 1.0351/.0880 | 0/31.2 \| 0/47.8 | 2.08 \| 4.80 |
| Uniform T/S2 | 1.0210/.0851 \| 1.0079/.0821 | 34.6/70.2 \| 35.0/71.8 | 51.26 \| 102.63 |
| Distributed I | .9987/.0955 \| .9987/.0955 | 0/0 \| 0/0 | 1.00 \| 2.00 |
| Distributed T | .9987/.0955 \| .9987/.0955 | 44.4/0 \| 44.4/0 | 8.48 \| 16.95 |
| Distributed S2 | .9987/.0955 \| .9987/.0955 | 0/93.8 \| 0/93.8 | 2.98 \| 5.95 |
| Distributed T/S2 | .9987/.0955 \| .9987/.0955 | 44.4/93.8 \| 44.4/93.8 | 52.00 \| 104.00 |
| Occluded I | .9994/.0944 \| 1.0010/.0837 | 0/0 \| 0/0 | 1.00 \| 2.00 |
| Occluded T | .9996/.0945 \| 1.0011/.0839 | 1.04/0 \| 1.21/0 | 4.00 \| 7.57 |
| Occluded S2 | .9998/.0915 \| 1.0012/.0823 | 0/.71 \| 0/1.11 | 1.44 \| 3.02 |
| Occluded T/S2 | 1.0000/.0881 \| 1.0013/.0811 | 1.16/1.31 \| 1.16/1.31 | 23.75 \| 42.80 |

First stable shift is temporal on Uniform (+1.6% c1), then spatial (+7.3%); Initial stays within 0.1%. BASIC also caps temporal history to `maxAge*currentM`, instead of importing spatially-expanded M.

1080p Release, three equal-frame windows: Uniform OFF→BASIC c1 `1.0319/.0870→1.0135/.0821`, c2 BASIC `1.0064/.0811`; Distributed `.9992/.0732→.9992/.0732`, c2 `.9992/.0732`; Occluded `.9996/.0562→1.0002/.0563`, c2 `1.0001/.0543`. GPU ms OFF/BASIC1/BASIC2: Uniform 81.3/98.9/106.9, Distributed 70.6/76.3/81.1, Occluded 23.0/23.2/23.3.
Equal NEE-time mean/rMSE: Uniform `1.0594/.1109→1.0346/.1113` (c2 `1.0150/.1109`); Distributed `.9943/.0711→.9939/.0736` (c2 `.9935/.0760`); Occluded `.9976/.0553→.9976/.0551` (c2 `.9985/.0511`). Audit: candidate/reuse queries 0; final visibility <= eligible; full Debug ctest 3/3 PASS.
