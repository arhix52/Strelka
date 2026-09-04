# ReSTIR DI status

Current phase: A — Initial RIS

Completed:
- Preflight: Debug CTest 4/4; existing light-sampling and MIS fixes present.
- Runtime switch and initial candidate count; reference NEE path remains selectable.
- Stable analytic/environment/emissive-triangle sample identifiers.
- Initial RIS at the first camera surface with one final visibility ray.
- Target: luminance of unoccluded `Li * cos * BSDF * MIS`.
- Candidate weight: target / complete marginal light proposal PDF.
- Update: weighted reservoir replacement; `M` counts zero-weight candidates.
- Normalization: `weightSum / (M * selectedTarget)`; RGB uses selected integrand times normalization.

Open:
- Temporal and spatial reuse, explicit wavefront passes, image validation and benchmark.
- Actual MTLDevice execution for the completed implementation.

Commits:
- `33d3bb8` ReSTIR reservoir and initial RIS

Tests:
- `cd build/Debug && ctest --output-on-failure`
- `./build/Debug/unit_tests -tc="*ReSTIR*"`
- `./build.sh Debug`

Benchmark NEE vs ReSTIR: pending
GPU time: pending
Reservoir memory: 48 bytes/pixel per history buffer; storage integration pending
External MTLDevice test: `STRELKA_STAGES=1 ./build/Release/StrelkaCLI --config <scene.toml>`
