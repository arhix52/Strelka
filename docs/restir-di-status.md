# ReSTIR DI status

Current phase: B — Temporal reuse

Completed:
- Preflight: Debug CTest 4/4; existing light-sampling and MIS fixes present.
- Runtime switch and initial candidate count; reference NEE path remains selectable.
- Stable analytic/environment/emissive-triangle sample identifiers.
- Initial RIS at the first camera surface with one final visibility ray.
- Target: luminance of unoccluded `Li * cos * BSDF * MIS`.
- Candidate weight: target / complete marginal light proposal PDF.
- Update: weighted reservoir replacement; `M` counts zero-weight candidates.
- Normalization: `weightSum / (M * selectedTarget)`; RGB uses selected integrand times normalization.
- Double-buffered reservoirs and surface history with motion reprojection.
- History rejects cuts, reset/resolution/topology changes, disocclusion, normal/material mismatch and excessive age.
- Reused samples are reconstructed and evaluated with the current target; visibility is not reused.

Open:
- Spatial reuse, explicit wavefront passes, image validation and benchmark.
- Actual MTLDevice execution for the completed implementation.

Commits:
- `c8e9ded` ReSTIR reservoir and initial RIS
- Add ReSTIR temporal reuse (this phase)

Tests:
- `cd build/Debug && ctest --output-on-failure`
- `./build/Debug/unit_tests -tc="*ReSTIR*"`
- `./build.sh Debug`

Benchmark NEE vs ReSTIR: pending
GPU time: pending
Reservoir memory: 96 bytes/pixel double-buffered; surface history adds 64 bytes/pixel
External MTLDevice test: `STRELKA_STAGES=1 ./build/Release/StrelkaCLI --config <scene.toml>`
