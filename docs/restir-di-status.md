Current phase: exact procedural analytic-light validation complete; ReSTIR remains experimental and default OFF.
Architecture: shared unit-sphere/disc BLAS, custom Metal intersections and TLAS instances; sampling/PDF use the same transformed surface.
Release, Apple M4 Pro, depth 4, 1 spp, 8 warm-up + 32 measured frames:

| sphere lights | old NEE 320p | NEE / ReSTIR 320p | NEE / ReSTIR 1080p | AS build / moving refit | TLAS memory |
|---:|---:|---:|---:|---:|---:|
| 1 | — | 2.81 / 3.88 ms | 57.98 / 85.55 ms | — | — |
| 512 | 48.98 ms | 3.02 / 4.44 ms | 63.37 / 93.34 ms | 4.35 / 0.194 ms | 0.101 MiB |
| 2048 | — | 3.56 / 5.16 ms | 82.07 / 111.97 ms | 3.92 / — ms | 0.399 MiB |
Incremental procedural AS: 512 B BLAS per used shape + 48 B bounds total; renderer/ReSTIR buffers unchanged (96 B reservoir/pixel).
Tests: 7 focused analytic cases (20 assertions), sampling audit, Debug CTest 3/3, Debug/Release Metal compile, MTLDevice.
NEE vs Initial RIS at 128 spp: mean -0.03%, relative MSE 0.00041; temporal/spatial path and moving-light refit executed.
Commits: `05d3010`, `65e26b1`, `Validate procedural analytic lights`.
Commands: `python3 tools/benchmark_analytic_light_scaling.py --shape sphere`; enable: `StrelkaCLI <scene> --restir-di`.
Open: Metal static `intersection_query` cannot bind an intersection-function table; alpha shadow uses bounded restart traversal.
