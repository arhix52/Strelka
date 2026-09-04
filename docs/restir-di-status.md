# ReSTIR DI status

Current phase: analytic-light ray intersection scaling complete. ReSTIR remains experimental and default OFF.

The bottleneck was independent of reservoir sampling: Metal `extend` and `shadow` each linearly scanned the analytic
light array per ray. Area-light proxy meshes now live in the existing BLAS/TLAS. Point/spot/projector and
distant/dome lights have mask 0 and are never ray surfaces. Hardware closest-hit preserves the light-table ID;
shade re-evaluates that selected light's exact transform, normal, sidedness, emission, and MIS PDF.

Shadow rays use the same TLAS with the sampled open segment `tMax`. No light ID is excluded: the offset target avoids
endpoint self-hits while the near side of a sphere and every triangle of a compound emitter remain occluders.

Release, Apple M4 Pro, depth 4, 1 spp, 8 warm-up + 32 measured frames, validation off:

| lights | 320x240 NEE before | 320x240 NEE after | 320x240 ReSTIR | 1080p NEE after | 1080p ReSTIR |
|---:|---:|---:|---:|---:|---:|
| 1 | — | 2.02 ms | 2.70 ms | 50.99 ms | 64.74 ms |
| 32 | — | 2.31 ms | 3.03 ms | 49.14 ms | 65.19 ms |
| 128 | — | 2.15 ms | 2.93 ms | 49.40 ms | 64.36 ms |
| 512 | 48.98 ms | 2.20 ms | 3.02 ms | 51.98 ms | 67.68 ms |
| 2048 | — | 3.12 ms | 4.12 ms | 66.77 ms | 83.13 ms |

Five-run validation medians, 1920x1080, Apple M4 Pro (AS memory is BLAS + TLAS allocation):

| lights / motion | BLAS build | TLAS build/refit | GPU render | AS memory |
|---|---:|---:|---:|---:|
| 512 static | 4.35 ms | 1.33 ms / — | 50.43 ms | 0.008 + 0.101 MiB |
| 2048 static | 3.92 ms | 1.23 ms / — | 64.83 ms | 0.008 + 0.399 MiB |
| 512 moving | 4.35 ms | 1.33 ms / 0.194 ms | 49.33 ms | 0.008 + 0.101 MiB |

ReSTIR is c1/T/S2. At 512 lights its incremental passes cost 0.82 ms at 320p and 15.70 ms at 1080p. NEE's total
change from 1 to 512 lights is +0.18 ms at 320p and +0.99 ms at 1080p; traversal is no longer linear. Metal 4 exposes
whole command-buffer GPU time in the harness. Exact extend/shadow split requires interactive Shader Timeline; the
automated `STRELKA_STAGES=1` path provides stage/ray breadcrumbs but intentionally avoids intrusive MTL4 timestamps.

Tests: Debug CTest, Metal shader compilation, Cornell MTLDevice display regression, nearest-area hit, punctual/infinite
non-surface policy, open shadow segment, selected-light exclusion, transformed area math. Cornell 256 spp NEE vs
BSDF-only: mean luminance 5.5626 vs 5.4665 (-1.73%, within current low-spp estimator noise).

Reservoir memory is unchanged: 96 B/pixel; incremental ReSTIR buffers 296 B/pixel. Reference NEE remains available.

Commands:
- Scaling: `python3 tools/benchmark_analytic_light_scaling.py --frames 32`
- Pass trace: `STRELKA_STAGES=1 STRELKA_BENCH=128 STRELKA_BENCH_W=1920 STRELKA_BENCH_H=1080 ./build/Release/StrelkaEditor -s <scene.glb>`
- Enable ReSTIR: `./build/Release/StrelkaCLI -c <scene.toml> --restir-di --restir-candidates 1 --restir-temporal=true --restir-spatial=true --restir-neighbors 2`

Commits: `029804f`, `81964db`, `1f62a8f`, `Fix selected-light shadow self-occlusion`.

Open: exact automated per-pass timings are unavailable on the production Metal 4 path; use Xcode Shader Timeline.
Disc/sphere hardware proxies remain tessellated (64 segments; sphere 32 rings), so strict BLAS/analytic surface
identity fails near polygon boundaries, including after non-uniform transforms. Exact procedural geometry is required.
