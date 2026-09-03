# Render correctness fix plan

Baseline: `172c35b Correct Metal environment sampling measure` on branch `arhix/wavefront`.

The pre-existing working-tree changes captured before this plan are intentionally preserved and are not cleanup targets:
modified `tests/CMakeLists.txt`; untracked `docs/restir/`, sampling-audit reports under `docs/`,
`tests/sampling/`, and the sampling-audit generators/runners under `tools/`.

| finding | reproducer | test | implementation | CPU | Metal | OptiX | commit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Standard PBR mixture PDF | 28,923/36,794 legacy audit mismatches; independent regression 28,882 eval and 28,866 oracle mismatches / 36,818 | `test_sample_eval_consistency`: marginal double oracle, split-lobe invariance, 1.3M edge/domain assertions; audit guard | Shared `eval()` finishes every non-delta sample; exit-side diffuse/interface proposals are conditionally renormalized; delta mass unchanged | FIXED | Shared header compiled | Shared header; backend build pending | `f845f2c` | PARTIAL |
| 2. Distant/dome support and MIS completeness | Dome packed as type `-1`; sharp distant reports continuous PDF 0 but is not delta; analytic dome one-bounce ratio `0.2988202609` | Scene/JSON packing, common support/delta oracle, cap normalization, analytic one-bounce MIS, omitted-miss mutation | Separate delta distant, finite spherical-cap, and dome measures; evaluate continuous infinite emitters on miss with the same selection PMF/support as NEE | FIXED | Shader compiled; analytic execution pending final GPU audit | Shared math/source changed; toolchain unavailable on macOS | `ad4c3d9` | PARTIAL |
| 3. Metal back-face NEE/MIS double counting | Back-face direction accepted by NEE but unpaired bounce gives MIS shares `0.4 + 1.0 = 1.4` | Exact support-equivalence sweep, front/back/flipped/transmission/fibre cases, 0.4/0.6 balance-share mutation | Evaluate proposal and bounce pairing in one shaded frame and make their continuous supports identical | Oracle FIXED | Shader compiled; path execution pending final GPU audit | Shared predicate fixed; toolchain unavailable on macOS | pending | PARTIAL |
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
- Validation: targeted consistency 1,792/1,792 assertions, furnace 588/588, Debug CTest 4/4, Release CTest 4/4,
  targeted ASan+UBSan clean, clang-tidy clean for project code, and production `wavefront.metal` compilation passed.
  The post-commit audit passed 754/754 tests and 58,590,420 assertions; its environment diagnostic executed on the
  actual Apple M4 Pro with 0/262,144 measure mismatches. That GPU kernel does not exercise Standard PBR. OptiX cannot
  be built by the macOS CMake configuration and remains to be compile-checked in finding 7.

## Finding 2: Distant/dome support and MIS completeness

- Random variable: next-event estimation first selects an infinite analytic light `J`, then, only for a continuous
  light, selects direction `W`. The complementary strategy selects `W` from the BSDF and identifies every infinite
  analytic emitter whose support contains it.
- Measure: `J` is a discrete probability mass. A finite distant light and a dome have densities with respect to solid
  angle `domega`. A zero-angle distant light is a Dirac mass at `-normal`, not a solid-angle density.
- Support: a finite distant light supports `dot(W, -normal) >= cos(halfAngle)`; a dome supports the full sphere. A
  sharp distant supports only its singular axis. Radiance from a continuous infinite emitter is nonzero exactly on
  that same directional support.
- Conditional and marginal PDF: `p(W|J=distant)=1/[4 pi sin^2(halfAngle/2)]` inside the cap and zero outside;
  `p(W|J=dome)=1/(4 pi)`. For emitter `J`, the MIS competitor is
  `p_light(J,W)=P(infinite-light group) P(J|group) p(W|J)`. The sharp distant carries only the corresponding
  discrete selection mass and has no `p(W|J)` in `domega`.
- Selection PMF: Metal uses the uploaded per-light `selectionPdf` times `1-envSelectionPdf` when an environment map
  competes. OptiX currently selects analytic lights uniformly and uses group mass `1/2` when an environment map also
  exists. Both the NEE and miss-side evaluation must use their backend's same PMF until finding 7 unifies any
  remaining backend convention.
- Delta classification: zero-angle distant and renderer-unhittable punctual proxies are delta/exclusive for MIS;
  finite distant and dome are continuous. A delta mass is never passed to a balance or power heuristic as a float
  solid-angle PDF.
- MIS strategies: light-sampled NEE and a non-delta BSDF ray that escapes to an infinite emitter are complementary.
  Camera/specular rays, or a vertex where NEE was disabled/not performed, retain unit BSDF-side weight. Overlapping
  infinite emitters are treated as separate integrand components; each is weighed against its own selected-light
  density, rather than incorrectly summing unrelated radiances under one emitter label.
- Current-HEAD reproducer: scene packing leaves `LIGHT_TYPE_DOME` at the default `-1`; `coneLightSolidAnglePdf(0)` is
  correctly zero but `lightIsDeltaForMis(LIGHT_TYPE_DISTANT)` is false, so NEE divides by zero instead of using a
  discrete event. Neither Metal nor OptiX adds finite distant/dome radiance on the BSDF miss path. The fixed-seed
  analytic dome audit reports `0.2988202609` instead of the reference `1.0`.
- Corrected result: finite-cap density integrates to `1` within the `2%` Monte Carlo bound; the deterministic
  double-precision one-bounce dome oracle is exactly `1.0`. Removing the BSDF-miss term reproduces
  `0.2988202609`. Sharp distant keeps continuous PDF `0`, is classified delta, and carries its selected-light mass
  through the existing connection PDF field. Dome descriptors now survive scene and JSON packing.
- Validation: targeted 45/45 assertions and broader dome/distant/cone 81/81 assertions passed; Debug CTest 4/4,
  Release CTest 4/4, targeted ASan+UBSan, the full 758-test sampling audit (58,590,444 assertions), and production
  `wavefront.metal` compilation passed. The environment-only diagnostic executed on an actual Apple M4 Pro with
  0/262,144 measure mismatches; it does not execute the new analytic-infinite miss path. The OptiX source uses the
  same common support/PDF functions but its toolchain is unavailable on this macOS host, so cross-backend execution
  remains for finding 7.

## Finding 3: Metal back-face NEE/MIS double counting

- Random variable: strategy `S` (selected-light NEE or BSDF continuation) and the continuous direction `W` leaving
  one surface vertex; a successful BSDF endpoint additionally identifies the same emitter selected by NEE.
- Measure: `S` and light identity use discrete probability mass; both non-delta directional proposals use density in
  solid angle `domega` in the BSDF's shaded frame.
- Support: for an ordinary surface, both strategies share exactly the hemisphere accepted by
  `neeProposesDirection(throughFibre, shadedFrontFace, signed n dot W)`. Fibres share both sides. Reflection and
  transmission outside that common set are BSDF-only. Mirroring a geometry/shading frame changes the signs but not
  this equivalence.
- Conditional and marginal PDF: `p_L(W)=P(light group) P(light identity|group) p(W|light)` and `p_B(W)` is the full
  marginal BSDF PDF from finding 1. Each endpoint uses balance/power shares `p_L^k/(p_L^k+p_B^k)` and
  `p_B^k/(p_L^k+p_B^k)` only where both supports overlap; otherwise the sole strategy has weight one.
- Selection PMF: unchanged from the light selector and already included in `LightConnection::pdf`. This finding
  changes only whether that continuous density is a valid competitor for the sampled direction.
- Delta classification: delta light or BSDF events remain exclusive and never enter this continuous pairing. The
  defect is specifically a non-delta direction that Metal NEE accepted but whose later path flag denied the
  complementary strategy.
- MIS strategies: one selected-light NEE sample (or local-RIS survivor representing that estimator) and one BSDF
  continuation sample. They enumerate the same path-space event only when their shaded-frame support predicates
  agree.
- Current-HEAD reproducer: on a raw back face with `n dot W=-0.7`, `neeProposesDirection` is true while
  `neePairsWithBounce` is false. With `p_L=0.4`, `p_B=0.6`, the NEE balance share is `0.4` and the unweighted BSDF hit
  contributes `1.0`, for total share `1.4`.
- Corrected result: the old predicate fails 11 cells of the exhaustive fibre/front-face/sign sweep. The paired
  predicate now equals the proposal predicate in every cell. The concrete back-face event returns balance shares
  `0.4 + 0.6 = 1.0`; clearing the pairing bit as a mutation returns `1.4`. Metal now constructs one `ShadedFrame`
  and uses it for both NEE validation and the outgoing path flag; OptiX already constructed that frame and inherits
  the corrected common support predicate for transmissive back faces.
- Validation: targeted pairing/frame cases passed 68/68 assertions; the full Debug audit passed 760/760 tests and
  58,590,488 assertions; Release CTest passed 4/4; targeted ASan+UBSan and production `wavefront.metal` compilation
  passed. The actual-MTLDevice audit still exercises only environment mapping, so renderer execution of this path is
  deferred to the cross-backend validation finding.
