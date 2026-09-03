# Render correctness fix plan

Baseline: `172c35b Correct Metal environment sampling measure` on branch `arhix/wavefront`.

The pre-existing working-tree changes captured before this plan are intentionally preserved and are not cleanup targets:
modified `tests/CMakeLists.txt`; untracked `docs/restir/`, sampling-audit reports under `docs/`,
`tests/sampling/`, and the sampling-audit generators/runners under `tools/`.

| finding | reproducer | test | implementation | CPU | Metal | OptiX | commit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Standard PBR mixture PDF | Pending current-HEAD reproduction | Pending | Return the full compatible-lobe marginal density from non-delta samples; keep delta mass discrete | OPEN | OPEN | OPEN | — | IN_PROGRESS |
| 2. Distant/dome support and MIS completeness | Pending | Pending | Separate delta distant, finite spherical-cap, and environment-miss measures and complete complementary MIS paths | OPEN | OPEN | OPEN | — | OPEN |
| 3. Metal back-face NEE/MIS double counting | Pending | Pending | Make sidedness and strategy enumeration agree with the CPU oracle | OPEN | OPEN | N/A unless shared | — | OPEN |
| 4. Non-uniform transforms for analytic lights | Pending | Pending | Sample intersected proxy geometry and apply the world-area Jacobian and inverse-transpose normals | OPEN | OPEN | OPEN | — | OPEN |
| 5. Robust large-distribution support | Pending | Pending | Select a GPU-friendly PMF representation that preserves every finite positive bin | OPEN | OPEN | OPEN | — | OPEN |
| 6. Emissive mesh NEE | Pending | Pending | Add selection, transformed-area sampling, solid-angle conversion, and BSDF-hit MIS | OPEN | OPEN | OPEN | — | OPEN |
| 7. Cross-backend consistency | Pending | Pending | Audit common probability conventions and compile/execute each available backend | OPEN | OPEN | OPEN | — | OPEN |

## Per-finding probability records

Each finding receives, before its production change, a concrete record of the sampled random variable, measure,
support, conditional and marginal PDFs, selection PMFs, delta/continuous classification, and participating MIS
strategies. Numerical before/after evidence and backend execution status are recorded here and expanded in the final
report.

## Validation states

`FIXED` requires a regression test plus a numerical result. Shader compilation, host/reference simulation, and
execution on an actual `MTLDevice` are reported separately. The known pre-existing `sharc_grid.h` sanitizer finding
is out of scope unless it blocks validation.
