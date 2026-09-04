# ReSTIR DI status

Current phase: validation complete

Completed:
- Reference NEE remains the default and is unchanged when ReSTIR is disabled.
- Initial RIS runs at the first non-delta camera surface; secondary bounces use one-sample NEE.
- Analytic area/punctual/distant/dome, environment and instanced emissive triangles reconnect from stable samples.
- Proposal PDF is emitter-class PMF x light-selection PMF x conditional solid-angle PDF.
- Target is luminance of unoccluded `Li * cos * BSDF * MIS`; candidate weight is target / proposal PDF.
- Weighted replacement counts every candidate in `M`; normalization is `weightSum / (M * selectedTarget)`.
- Double-buffered temporal reuse reprojects existing motion data and reevaluates the current target.
- History rejects cuts/resets, resolution/light topology changes, invalid depth/normal/material/geometry and excessive age.
- Spatial reuse uses four deterministic frame-rotated neighbors by default and reevaluates each sample.
- Visibility is never reused; exactly one final shadow ray is emitted per valid ReSTIR shading point.
- Actual Apple M4 Pro execution covered static area, many-light, emissive, HDR environment, glossy and moving-camera scenes.
- Moving-camera audit: 100% nonzero motion after orbit, with no audit failures.

Estimator summary:
- Initial update: `w = target / proposalPdf`; replacement probability is `w / weightSum`.
- Reuse update: `w = source.weightSum * currentTarget / source.target`, carrying source `M`.
- Final RGB: current unoccluded integrand x reservoir normalization, followed by one visibility query.

Commits:
- `c8e9ded` ReSTIR reservoir and initial RIS
- `c710fb6` Add ReSTIR temporal reuse
- `0a0e748` Add ReSTIR spatial reuse
- `4823234` Integrate ReSTIR DI into Metal wavefront pipeline
- Validate and optimize ReSTIR DI (this commit)

Tests:
- `cd build/Debug && ctest --output-on-failure`
- `./build/Debug/unit_tests -tc="*ReSTIR*"` (8 cases, 28 assertions)
- `./build/Debug/unit_tests -tc="*Metal wavefront buffer*"`
- Metal shader compilation: Debug and Release builds pass.

Benchmark: Apple M4 Pro, Release, 320x240, depth 4, validation off, 32 spp unless noted.

| Scene | NEE ms/frame | ReSTIR ms/frame | ReSTIR rel. L1 | mean ratio |
|---|---:|---:|---:|---:|
| Cornell area | 2.1 | 3.7 | 0.1040 vs NEE-512 | 1.000 |
| 512 analytic lights | 47.1 | 48.8 | 0.0847 vs NEE-256 | 1.000 |
| emissive mesh | - | - | 0.0203 vs NEE-256 | 1.000 |
| HDR env + area | - | - | 0.0537 vs NEE-512 | 0.987 |
| glossy | - | - | 0.0391 vs NEE-512 | 0.992 |

GPU time: area initial/final overhead 1.3 ms, temporal 0.2 ms, spatial 0.1 ms; full frame 3.7 ms.
Equal-time area quality: NEE 55 spp rel. L1 0.0801; ReSTIR 32 spp 0.1031.
Shadow rays: at most 1 per first-surface ReSTIR point, plus ordinary secondary NEE rays.
Reservoir memory: 96 B/pixel double-buffered; history/work adds 200 B/pixel (296 B/pixel total).
GPU counters: max threads/threadgroup 1024; register/occupancy/traffic counters require Xcode trace inspection.

Limitations:
- Existing power-weighted sampler is already strong; no quality/time crossover was measured through 512 analytic lights.
- Per-pass counter/traffic/occupancy inspection is in `/tmp/restir-di.gputrace`; CLI reports aggregate/toggle deltas.
- Open: Xcode UI counter extraction from the captured trace; no V1 correctness or execution blocker.

MTLDevice test:
- Verified on Apple M4 Pro, including `.gputrace` capture.
- `./build/Release/StrelkaCLI --config scenes/validation/cornell_box/cornell_box.toml --width 320 --height 240 --spp 32 --depth 4 --restir-di --profile-stages`
