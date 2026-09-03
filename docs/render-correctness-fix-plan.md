# Render correctness fix plan

Baseline: `172c35b Correct Metal environment sampling measure` on branch `arhix/wavefront`.

The pre-existing working-tree changes captured before this plan are intentionally preserved and are not cleanup targets:
modified `tests/CMakeLists.txt`; untracked `docs/restir/`, sampling-audit reports under `docs/`,
`tests/sampling/`, and the sampling-audit generators/runners under `tools/`.

| finding | reproducer | test | implementation | CPU | Metal | OptiX | commit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Standard PBR mixture PDF | 28,923/36,794 legacy audit mismatches; independent regression 28,882 eval and 28,866 oracle mismatches / 36,818 | `test_sample_eval_consistency`: marginal double oracle, split-lobe invariance, 1.3M edge/domain assertions; audit guard | Shared `eval()` finishes every non-delta sample; exit-side diffuse/interface proposals are conditionally renormalized; delta mass unchanged | FIXED | Shared header compiled | Shared header; backend build pending | pending | PARTIAL |
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

## Finding 1: Standard PBR mixture PDF

- Random variable: first a lobe index `K`, then a direction `Wi` (and, inside dielectric transmission, a discrete
  Fresnel reflect/refract choice).
- Measure: `K` and smooth Fresnel outcomes use discrete probability mass; every rough or diffuse direction uses
  solid-angle density `dP/domega`.
- Support: diffuse/specular/coat reflection share the reflection hemisphere; diffuse and rough specular transmission
  share the opposite hemisphere. A component with zero selection weight has no sampling support.
- Conditional and marginal PDF: for a non-delta direction, `p(Wi|K=k)` is the component's cosine or GGX density and
  `p(Wi)=sum_k alpha_k p(Wi|K=k)` over every compatible lobe. The old lower-hemisphere sample path returned only the
  chosen term, while `eval()` returned the sum.
- Selection PMF: `alpha_k = w_k / sum_j w_j`, from `pbr_lobe_weights`; the transmission component also includes its
  Fresnel branch probability in `p(Wi|K=transmission)`.
- Delta classification: `alpha < BSDF_DELTA_ALPHA` remains a singular mirror/refraction event. Its discrete mass is
  not added to, or compared with, a continuous solid-angle PDF.
- MIS strategies: continuous light NEE and non-delta BSDF continuation use the complete marginal `p(Wi)`; a delta
  BSDF event is exclusive and carries no continuous competitor.
- Current-HEAD reproducer: 50,000 fixed-seed draws produced 36,794 eligible lower-hemisphere non-delta samples;
  28,923 (78.61%) disagreed with `bsdf_eval()` at the existing relative threshold `1e-3`.
- Corrected result: 0/36,818 disagreements against both `eval()` and the independent double oracle; the separate audit
  stream reports 0/36,794. The domain/roughness grid executed 1,306,384 assertions including both sides of
  `BSDF_DELTA_ALPHA`, thin/solid transmission, grazing exit/TIR cases, and zero component weights.
- Validation so far: targeted consistency 1,792/1,792 assertions, furnace 588/588, Debug CTest 4/4, Release CTest
  4/4, targeted ASan+UBSan clean, and production `wavefront.metal` compilation passed. OptiX cannot be built by the
  macOS CMake configuration and remains to be compile-checked in finding 7.
