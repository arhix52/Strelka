# ReSTIR DI status

Current phase: D — Metal wavefront integration

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
- Spatial reuse reads an immutable reservoir set, samples up to 16 deterministic frame-rotated neighbors and writes a second set.
- Depth, geometric normal, material and geometry validity gate every neighbor; one final pass emits one shadow ray.
- Explicit temporal, spatial and final Metal stages run between shade/initial RIS and shadow preparation.
- CLI/TOML exposes all seven runtime settings; debug modes show age or selected emitter class.
- Apple M4 Pro execution passed for analytic area lighting; one-candidate relative L1 vs NEE was 1.39e-4 at 4 spp.

Open:
- Representative image validation, Release benchmark and focused review.

Commits:
- `c8e9ded` ReSTIR reservoir and initial RIS
- `c710fb6` Add ReSTIR temporal reuse
- `0a0e748` Add ReSTIR spatial reuse
- Integrate ReSTIR DI into Metal wavefront pipeline (this phase)

Tests:
- `cd build/Debug && ctest --output-on-failure`
- `./build/Debug/unit_tests -tc="*ReSTIR*"`
- `./build.sh Debug`

Benchmark NEE vs ReSTIR: pending
GPU time: pending
Reservoir memory: 96 bytes/pixel double-buffered; history/work data adds 200 bytes/pixel
MTLDevice test: Apple M4 Pro; `./build/Debug/StrelkaCLI --config scenes/validation/cornell_box/cornell_box.toml --restir-di`
