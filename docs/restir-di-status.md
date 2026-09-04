# ReSTIR DI status

Current phase: performance/correctness triage complete. Classification: **experimental**, default **OFF**.

Completed:
- CTest has three macOS tests in both `ba00857` and HEAD. The earlier fourth entry was a stale, never-committed
  `sampling_audit` registration in `build/Sanitize`; no tracked test was removed or disabled by ReSTIR.
- ReSTIR target is luminance of unoccluded `Li * BSDF * light-side MIS`; candidate weight is `target / q`.
- `q` is the emitter-class PMF times light/mesh/triangle PMF times the conditional solid-angle PDF.
- BSDF hits of analytic/emissive lights and environment/distant misses use the complementary MIS weight with the same `q`.
- `PATH_FLAG_NEE_DONE` gates the complement. One selected ReSTIR sample emits one final visibility ray. No double count,
  stale target reuse, or effective-PDF defect was found.
- Actual Metal execution, shader compilation, and Metal System Trace capture passed on Apple M4 Pro.

Method: Release, Apple M4 Pro, 1 spp/frame, 32 warm-up + 128 measured frames, depth 4, validation off.
The 1080p timing sweep paired NEE and ReSTIR in one process; NEE used one candidate. Moving camera used a small
continuous orbit. Image rows compare equal time: NEE 168 spp versus ReSTIR 128 spp against high-spp NEE.

| Scope / configuration | NEE | ReSTIR | Result |
|---|---:|---:|---|
| 1080p static, c1, T0/S0 | 51.51 ms | 59.92 ms | 1.163x |
| 1080p static, c1, T1/S0 | 49.02 ms | 62.25 ms | 1.270x |
| 1080p static, c1, T1/S2 | 48.97 ms | 64.13 ms | 1.310x |
| 1080p static, c1, T1/S4 | 50.13 ms | 66.31 ms | 1.323x |
| 1080p static sweep, c2 | 49.11-50.57 ms | 64.55-70.23 ms | 1.284-1.413x |
| 1080p static sweep, c4 | 48.00-50.83 ms | 74.67-79.69 ms | 1.475-1.617x |
| 1080p static sweep, c8 | 47.89-49.53 ms | 85.35-95.54 ms | 1.782-1.945x |
| 1080p moving, c1/c2/c4/c8, T1/S2 | 49-51 ms | 64.35/69.19/81.02/97.33 ms | 1.30-1.90x |
| Cornell equal-time rel. L1 | 0.0500 | 0.0577 | NEE wins |
| 512 lights equal-time rel. L1 | 0.0235 | 0.0341 | NEE wins |
| Occluded interior, 512 lights, rel. L1 | 0.1338 | 0.2153 | NEE wins |
| Emissive mesh equal-time rel. L1 | 0.0047 | 0.0109 | NEE wins |
| HDR environment + area, rel. L1 | 0.0397 | 0.0468 | NEE wins |

Pass cost at 1080p, derived from paired feature toggles: base initial/copy/final `+8.41 ms`; temporal `+4.82 ms`;
spatial 2/4 neighbors `+2.93/+5.35 ms`. Extra initial candidates c2/c4/c8 add `+5.88/+15.63/+29.05 ms`.
Metal 4 exposes total command-buffer GPU time here, so these are toggle deltas, not timestamp samples.

The 47 ms many-light cost is shared: `extend` tests every analytic light surface for every path and `shadow` tests
every analytic light surface for every shadow ray. At 320x240/512 lights NEE is 48.98 ms; c1 T0/S0 is 49.26 ms and
c1 T1/S2 is 51.18 ms. ReSTIR retains one shadow ray and adds passes, so it cannot remove this O(pixels * lights) work.

Memory at 1920x1080: existing wavefront buffers 646.2 MiB (~327 B/pixel); incremental ReSTIR 585.4 MiB
(296 B/pixel): reservoirs 189.8 MiB (96 B/pixel), temporal G-buffer 126.6 MiB (64 B/pixel), spatial/work
268.9 MiB (136 B/pixel). Current allocation is unconditional even while ReSTIR is disabled.

Commands:
- Tests: `cmake --build build/Debug -j8 && cd build/Debug && ctest --output-on-failure`
- Enable: `./build/Release/StrelkaCLI -c scenes/validation/cornell_box/cornell_box.toml --spp 128 --restir-di --restir-candidates 1 --restir-temporal=true --restir-spatial=false`

Commits: `c8e9ded`, `c710fb6`, `0a0e748`, `4823234`, `73fd669`; triage commit follows.

Open: no measured quality/time crossover through 512 lights. Register/occupancy counters need interactive Xcode trace
inspection; this does not block the measured conclusion.
