# Render correctness fix plan

Baseline: `172c35b Correct Metal environment sampling measure` on branch `arhix/wavefront`.

The pre-existing working-tree changes captured before this plan are intentionally preserved and are not cleanup targets:
modified `tests/CMakeLists.txt`; untracked `docs/restir/`, sampling-audit reports under `docs/`,
`tests/sampling/`, and the sampling-audit generators/runners under `tools/`.

| finding | reproducer | test | implementation | CPU | Metal | OptiX | commit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Standard PBR mixture PDF | 28,923/36,794 legacy audit mismatches; independent regression 28,882 eval and 28,866 oracle mismatches / 36,818 | `test_sample_eval_consistency`: marginal double oracle, split-lobe invariance, 1.3M edge/domain assertions; audit guard | Shared `eval()` finishes every non-delta sample; exit-side diffuse/interface proposals are conditionally renormalized; delta mass unchanged | FIXED | Shared header compiled | Shared header; backend build pending | `f845f2c` | PARTIAL |
| 2. Distant/dome support and MIS completeness | Dome packed as type `-1`; sharp distant reports continuous PDF 0 but is not delta; analytic dome one-bounce ratio `0.2988202609` | Scene/JSON packing, common support/delta oracle, cap normalization, analytic one-bounce MIS, omitted-miss mutation | Separate delta distant, finite spherical-cap, and dome measures; evaluate continuous infinite emitters on miss with the same selection PMF/support as NEE | FIXED | Shader compiled; analytic execution pending final GPU audit | Shared math/source changed; toolchain unavailable on macOS | `ad4c3d9` | PARTIAL |
| 3. Metal back-face NEE/MIS double counting | Back-face direction accepted by NEE but unpaired bounce gives MIS shares `0.4 + 1.0 = 1.4` | Exact support-equivalence sweep, front/back/flipped/transmission/fibre cases, 0.4/0.6 balance-share mutation | Evaluate proposal and bounce pairing in one shaded frame and make their continuous supports identical | Oracle FIXED | Shader compiled; path execution pending final GPU audit | Shared predicate fixed; toolchain unavailable on macOS | `b214620` | PARTIAL |
| 4. Non-uniform transforms for analytic lights | Disc geometry/sampler area ratio `6.0`; a smooth-disc sample outside the 16-gon is invisible to old traversal; sphere record loses affine axes | Double-precision affine-Jacobian oracle, smooth sample/intersection/PDF agreement, mirrored normals, CPU picking, coarse-proxy mutation | Sample and intersect the same smooth affine disc/ellipsoid; keep tessellation editor-only | FIXED | Shader compiled; shared analytic math executed on MTLDevice | Source updated; toolchain unavailable on macOS | `495e2a7` | PARTIAL |
| 5. Robust large-distribution support | 209,715/1,048,576 positive-PMF bins have zero float-CDF interval | Million-bin support/normalization, sparse zeros, 1e12 dynamic range, PMF lookup, GOF, old-CDF mutation | Replace the cumulative float table with an O(1) Walker/Vose alias draw and its represented PMF | FIXED | 1,048,576-entry support kernel passed on MTLDevice | OptiX remains uniform until finding 7 | `cd2dd98` | PARTIAL |
| 6. Emissive mesh NEE | A scene containing only one material-emissive triangle has `hasEmitter=false` and NEE proposal probability 0 despite nonzero emitted radiance | Single/multiple triangle frequencies, affine instances, texture evaluation, sample/PDF oracle, hit MIS, NEE-off expectation, omitted-NEE and visibility mutations | Hierarchical mesh-instance/triangle selection, exact transformed-area sampling, endpoint-consistent visibility, and the same marginal density at BSDF hits | FIXED | FIXED on MTLDevice | Source implemented; toolchain unavailable | `d53e662` | PARTIAL |
| 7. Cross-backend consistency | OptiX uses uniform analytic-light identity, fixed 1/2 environment/local selection, and centre-Jacobian environment rows while Metal uses power aliases and exact row solid angle | Shared hierarchy/marginal-PMF oracle, sharp-distant support, exact environment degeneracies, legacy-selector mutation, source/toolchain validation | Use one host probability specification and represented PMFs; migrate OptiX environment and analytic selection to it | FIXED | FIXED on MTLDevice | Source fixed; external CUDA validation required | `0b85cd9` | UNVERIFIED |

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

## Finding 4: Non-uniform transforms for analytic lights

- Random variable: analytic-light identity `J`, object-space point/direction `Q`, transformed surface point
  `X=c+A Q`, and induced direction `W=(X-P)/|X-P|` from shading point `P`.
- Measure: `J` is discrete. `Q` is area on the unit disc or unit sphere; `X` is world-space area `dA_world`; `W` is
  solid angle `domega` at the shading vertex.
- Support: the smooth affine image of the unit disc or sphere. The old tessellated mesh remains an editor proxy only
  and is removed from radiance traversal for these two light types, so it cannot add or remove renderer support.
- Conditional and marginal PDF: the disc draws `p_Q=1/pi` and has constant
  `J_A=length(cross(axisX,axisY))`, hence `p_A=1/(pi J_A)`. The ellipsoid draws `p_Q=1/(4pi)` and uses
  `J_A(n)=abs(det(A))*length(transpose(inverse(A))*n_object)`, hence `p_A(X)=1/(4pi J_A(n))`. Both convert with
  `p_omega=p_A distance^2/abs(n_light dot -W)`. Hit-side inverse mapping reconstructs the same `n_object`, Jacobian,
  outward inverse-transpose normal, and density.
- Selection PMF: the existing analytic-light PMF remains an outer factor. Disc power uses its exact smooth area;
  ellipsoid selection power uses deterministic equal-area quadrature of `J_A`. That quadrature changes variance only:
  the returned selection PMF is still the exact one sampled and is included in MIS.
- Delta classification: transformed discs and ellipsoids are continuous area lights. Degenerate transforms have zero
  area/support and return PDF zero without NaN/Inf; no delta fallback is invented.
- MIS strategies: NEE maps `Q` to the analytic surface; a BSDF/camera ray uses the same analytic intersection before
  any farther hardware hit. Both use the same hit normal and local area density. Shadow visibility still traces scene
  geometry to just before the analytic endpoint; light proxies have never belonged to the shadow mask.
- Current-HEAD reproducer: a radius-0.5 disc under x/y scales 2/3 is intersected as an ellipse with semi-axes 1/1.5,
  while NEE normalizes both stored axes and samples a radius-0.5 circle, giving geometry/sampler area ratio `6.0`.
  At the midpoint of a 16-gon edge, smooth-disc radii in `(cos(pi/16),1]` are sampled but old ray traversal cannot hit
  them. A sphere stores only one scalar radius, so an ellipsoid transform is lost.
- Corrected result: 4,096 fixed-seed samples per transformed shape agree with the independent double-precision
  Jacobian and hit-side density; the complete analytic test group passes 76,705 assertions. Every sampled disc point
  is on the smooth affine disc used by CPU/Metal/OptiX intersection, and every well-conditioned ellipsoid sample
  round-trips through the quadratic intersection. Mirrored/sheared normals agree with `transpose(inverse(A))`.
  The old normalized-axis mutation retains the measured area ratio `6.0`; the old 16-gon mutation misses radius
  `0.99 > cos(pi/16)` while the analytic intersection hits it.
- Validation: full Debug and Release CTest pass 4/4, the full sampling audit passes 769/769 tests and 58,667,179
  assertions, targeted ASan+UBSan passes 76,705/76,705 assertions, and production `wavefront.metal` compilation
  passes. The expanded audit kernel executed 262,144 affine ellipsoid samples on an actual Apple M4 Pro in both fast
  and safe Metal math modes with zero sample/eval mismatches, intersection mismatches, NaN, Inf, or negative PDFs.
  The OptiX implementation uses the same header but cannot be compiled on this macOS host; it remains UNVERIFIED
  until the cross-backend finding.

## Finding 5: Robust large-distribution support

- Random variable: analytic-light identity `J`. The alias construction introduces bucket `B` and branch coin `C`.
- Measure: all three are discrete probability mass (`C` is represented by a uniform variate used to realize a
  Bernoulli mass); no directional or area density is involved until the selected light's conditional sampler runs.
- Support: exactly the indices whose cleaned finite power `w_i` is positive. A zero, negative, NaN, or infinite
  input weight has no support. An all-zero table has empty support and must return an empty connection.
- Conditional and marginal PMF: `P(B=b)=1/N`, `P(J=b|B=b)=q_b`, and
  `P(J=a_b|B=b)=1-q_b`. Therefore
  `P(J=i)=[q_i + sum_(b:a_b=i)(1-q_b)]/N`. The stored lookup PMF is reconstructed from the final float `q_b`
  values, so it is the distribution the GPU actually samples rather than the unrounded input target.
- Selection PMF: `P(J=i)` multiplies the selected light's conditional solid-angle/delta measure in NEE and is the
  same factor used by BSDF-hit MIS. There is no uniform floor: it selected zero-power lights and was the source of
  the collapsed tail. Positive probabilities below the normal float range are proposal-regularized upward before
  alias construction; every branch is also kept wider than the shipped samplers' smallest positive 23-bit step.
  The represented PMF remains normalized and is used consistently.
- Delta classification: `J` is discrete for every light. Whether the conditional light sample is delta (sharp
  distant/punctual) or continuous is orthogonal; the selection mass is never treated as a solid-angle PDF by itself.
- MIS strategies: selected-light NEE versus the compatible BSDF continuation. Both use the same marginal selection
  PMF stored in the selected GPU light record.
- Representation decision: a flat float CDF is 8 bytes/light and `O(log N)` but loses tail intervals. A blocked
  two-level CDF preserves local increments but needs block metadata, two searches, and another shader binding. A
  binary probability tree also uses local normalization but needs `N-1` branch values and about 20 dependent loads
  at one million lights. A Walker/Vose table is `O(1)` with one bucket and one alias load; its threshold and alias
  replace the old CDF field pair, while the otherwise-unused GPU `color.w` carries the represented PMF, keeping the
  128-byte light ABI and GPU memory unchanged.
- Current-HEAD reproducer: with 1,048,576 lights, one dominant power and the old 5% uniform floor, 209,715 positive
  stored PMFs (19.99998093%) have identical adjacent float CDF endpoints and can never be returned by the binary
  search.
- Corrected result: the production alias table loses 0/1,048,576 positive bins at dynamic range `1e12`, and sparse
  zero/NaN bins receive exactly zero represented PMF. The stored PMFs sum to one and match the distribution
  reconstructed independently from the rounded thresholds/aliases. Uniform tables are exact, even a positive
  double weight below float range is reachable after proposal regularization, and a 1,048,576-draw categorical GOF
  test gives chi-square `0.0187892` (three nonzero categories, acceptance bound `16`). The retained old-CDF mutation
  still collapses `0.19999980926513672` of the million bins.
- Performance at `-O3` on Apple M4 Pro, 1,048,576 weights and 16,777,216 CPU draws (median of five runs): build time
  changed from `1.06 ms` (flat CDF) to `8.93 ms` (alias); sampling changed from `923.9 ms` to `28.7 ms` (`32.2x`
  faster). GPU distribution metadata remains `8 MiB` (`8 bytes/light`) and `UniformLight` remains 128 bytes. Peak
  host RSS in isolated runs rose from `18.3 MiB` to `62.4 MiB` because the linear-time builder keeps temporary
  double arrays; this is a build-time cost, not resident GPU memory.
- Validation: targeted tests pass 5,243,704 assertions; the full audit passes 772/772 tests and 66,008,017 assertions.
  The production Metal shaders compile. On the actual Apple M4 Pro, the 1,048,576-entry, `1e12` audit reports zero
  positive-support loss and zero selection of zero bins in fast and safe math modes. Release, sanitizer, and final
  checks pass: Release CTest 4/4 and targeted ASan+UBSan 5,243,704/5,243,704. OptiX still uses its previous uniform
  analytic-light selector; the cross-backend convention is deliberately completed in finding 7.

## Finding 6: Emissive mesh NEE

- Random variable: emitter class `C` (environment or local), local class `K` (analytic or material-emissive mesh),
  mesh/instance identity `M`, triangle identity `T`, and barycentric surface point `X` on that triangle.
- Measure: `C`, `K`, `M`, and `T` are discrete probability masses. `X` has density with respect to world-space area
  `dA`; its induced direction `W=(X-P)/|X-P|` has density with respect to solid angle `domega` at shading point `P`.
- Support: every non-degenerate triangle instance whose material's multiplicative emission factor is finite and has
  a positive channel. Texture values are evaluated at the sampled barycentric UV; a black texel contributes zero but
  does not remove support from other positive texels. Existing surface shading emits on both traced sides, so mesh
  NEE uses the same two-sided support rather than inventing a one-sided material convention.
- Conditional and marginal PDF: within a selected triangle, `p_A(X|T,M)=1/A_world`. The area-to-direction Jacobian is
  `p_omega(W|T,M)=distance^2/[A_world abs(n_light dot (-W))]`. The complete light density is
  `P(C=local) P(K=mesh|local) P(M|mesh) P(T|M) p_omega(W|T,M)`. The hit-side lookup reconstructs this same product.
- Selection PMF: mesh-instance power is the sum of its triangle powers; the conditional triangle PMF is represented
  by its own alias table. Both stored PMFs are reconstructed from the float alias representation. Analytic lights and
  emissive meshes remain separate conditional tables, joined by an explicit local-class PMF; the environment/local
  PMF uses their combined local power.
- Delta classification: a finite-area mesh triangle is continuous even if it is very small. Degenerate world-space
  triangles have zero area and no continuous support; no delta fallback is synthesized.
- MIS strategies: selected-light NEE and non-delta BSDF continuation that hits the same emissive triangle. Camera,
  specular, NEE-disabled, and unsupported hits retain weight one. Otherwise both sides use the complete marginal
  light density above with the same heuristic.
- Current-HEAD reproducer: ordinary mesh hits add `si.emission`, but Metal's and OptiX's `hasEmitter` predicates and
  `connectToLight()` enumerate only analytic lights and the environment. For a scene containing one positive
  emissive triangle and no analytic/environment light, `P(NEE)=0`; the independent one-bounce triangle integral is
  positive. The existing BSDF-only path remains unbiased where its lobe has support, but the required direct-light
  strategy, selection PMFs, and complementary hit MIS are absent.
- Corrected result: the nested alias tables reproduce the represented PMFs `(0.1, 0.3, 0, 0.6)` in one million
  fixed-seed draws and never select the zero entry. Across 4,096 affine samples, the sampled point, transformed
  normal, world-space area density, solid-angle Jacobian, and full marginal PDF agree with an independent
  double-precision GLM oracle, including a mirrored non-uniform instance. An independent one-bounce QMC estimate
  agrees with midpoint quadrature; deleting the mesh proposal returns exactly zero and is retained as the mutation.
  Texture emission and alpha are evaluated at the sampled barycentric UV rather than replaced by a constant proxy.
- Visibility correction: the physical connection keeps the exact shading-point-to-light direction used by the
  BSDF and PDF, while the shadow ray is the exact segment between independently scale-offset source and target
  endpoints. The old `distance - 1e-5` mutation, measured from the unoffset source, crosses the emitter plane after
  the source is offset and self-occludes; the endpoint construction terminates strictly before the emitter without
  changing the sampled event.
- Validation: the five focused cases pass 2,646,048 assertions; Debug and Release CTest pass 4/4; the full audit
  passes 777/777 tests and 68,654,065 assertions; targeted ASan+UBSan passes 2,646,048 assertions (macOS reports that
  leak detection is unsupported). Production `wavefront.metal` compiles. On the actual Apple M4 Pro, a textured
  emissive-triangle scene at 96x96 and 32,768 spp gives receiver means `0.1022682866` with NEE and `0.1022836623`
  BSDF-only (relative difference `0.015035%`), with zero non-finite or negative pixels. The corresponding Debug GPU
  runs took 13.2 s and 10.4 s; these are correctness observations, not performance claims. OptiX uses the same
  analytic transformed-triangle and visibility specification, but cannot be compiled or executed on this macOS
  host, so its status remains UNVERIFIED until finding 7.

## Finding 7: Cross-backend consistency

- Random variable: emitter class `C` (environment or local), local class `K` (analytic or emissive mesh), analytic
  identity `J` or mesh identity `M` and triangle `T`, followed by the selected emitter's conditional point or
  direction variable.
- Measure: `C`, `K`, `J`, `M`, and `T` are discrete masses. Environment and finite distant/dome directions use
  `domega`; finite mesh and analytic surfaces first use `dA` and then the area-to-solid-angle Jacobian. Sharp distant
  and punctual events retain their separate delta measure.
- Support: every represented positive-power discrete entry and every direction/point in its conditional support.
  Zero-power entries are not selected. A positive sharp distant light receives positive discrete proposal power even
  though its conditional solid-angle density is correctly zero.
- Conditional and marginal PDF: `p(C,K,J,W)=P(C)P(K|C)P(J|K,C)p(W|J)` for analytic/environment sampling, with the
  analogous mesh product from finding 6. `P(environment)` and `P(mesh|local)` are binary power ratios; analytic,
  mesh, triangle, and environment-texel masses are the PMFs represented by their final float alias tables.
- Selection PMF: both hosts use the same double-precision power proxies and `binaryPowerProbability()`, including
  its finite endpoint regularization. Both devices multiply the returned conditional density by those uploaded
  probabilities, and BSDF-hit/miss MIS reads the same represented probabilities rather than reconstructing uniform
  or fixed-half alternatives.
- Delta classification: outer selections remain discrete for all emitters. A sharp distant's selected mass is paired
  only with the singular light strategy; no float `p_omega` is synthesized. All continuous conditional measures use
  the shared PDF and support functions from findings 1--6.
- MIS strategies: selected-light NEE and compatible non-delta BSDF continuation. Camera/specular/NEE-disabled paths
  have no competitor. Environment miss, finite distant/dome miss, analytic-area hit, and emissive-mesh hit each use
  the same full marginal density as their corresponding NEE branch.
- Current-HEAD reproducer: for environment/analytic/mesh powers `(9, 4, 2)` and analytic powers `(1, 3)`, Metal's
  hierarchy represents marginal masses `(0.6, 0.0666667, 0.2, 0.1333333)`. OptiX instead chooses environment with
  `0.5`, then analytic identities uniformly, yielding `(0.5, 0.1666667, 0.1666667, 0.1666667)`. Its environment
  host table additionally uses centre-row `sin(theta)` weights and the device jitters `v` uniformly, unlike Metal's
  exact row solid angles and uniform-cosine row sample. The common OptiX random-dimension enum also lacks the SSS
  dimensions already referenced by its shader source, which is a compile-time backend-consistency failure.
- Corrected result: both host paths construct the same nested class probabilities and Walker/Vose tables. The
  independent hierarchy oracle recovers marginal masses `(0.6, 0.0666667, 0.2, 0.1333333)`, summing to one within
  `1e-7`; the retained fixed-half/uniform OptiX mutation differs in both the environment and analytic terms. A sharp
  distant now keeps positive outer discrete mass while its continuous PDF remains zero. Environment entries carry
  the solid-angle density implied by their final float alias representation, so the direction sampler and lookup PDF
  agree even after the support-preserving proposal floor used for extreme dynamic range.
- Environment representation: texel `i` has represented discrete mass `P_i`. Its exact lat-long bin area is
  `DeltaOmega_i = (2 pi / width) [cos(theta_0) - cos(theta_1)]`; both devices draw azimuth uniformly and
  `cos(theta)` uniformly within the selected row, and return `p_omega = P_i / DeltaOmega_i`. A `1x1` constant map
  therefore returns `1/(4 pi)`. Invalid channels are rejected, finite positive channels retain support, and a
  bilinear-radiance direction in an otherwise zero-density texel remains owned by the BSDF strategy with MIS weight
  one rather than being dropped.
- Validation: the final focused groups pass 4,594,511 assertions, Debug and Release CTest pass 4/4, and the full
  audit passes 781/781 tests and 68,654,103 assertions. The environment integral is `1`, the Lambertian estimate is
  `1.002147074` with 95% CI `[0.9988432646, 1.005450884]`, and the frozen legacy mutation is `2.003353165` and is
  rejected. Targeted ASan+UBSan is clean with macOS leak detection disabled because that sanitizer mode is
  unsupported. Production MSL and both Metal 3/Metal 4 host binding paths compile in Debug, Release, and sanitizer
  builds. On the actual Apple M4 Pro, 262,144 samples in fast and safe math report zero sample/PDF, intersection,
  support, or non-finite mismatches; the available renderer selected Metal 4 because there is no diagnostic override
  for forcing its automatic Metal 3 fallback.
- Cross-strategy GPU result: an actual-device 96x96, 32,768-spp scene combining a textured environment, finite
  distant, analytic dome, and emissive mesh gives mean luminance `0.1808142214` with NEE and `0.1808202792` without
  NEE (relative difference `-0.003350%`), with no negative or non-finite pixels. This is a correctness comparison,
  not a performance claim.
- Resource delta: the environment alias entry grows from 8 to 12 bytes to store the represented solid-angle PDF;
  a 65,536-texel table therefore grows from 0.5 MiB to 0.75 MiB (reported as 0.8 MB by the renderer). The represented
  density removes an extra texture read from each PDF query. No backend-comparable timing claim is made.
- External blocker: this macOS host has neither `nvcc`, an OptiX SDK/toolchain, nor an NVIDIA GPU, so OptiX source
  compilation and execution are `UNVERIFIED`, not treated as Metal emulation. On a configured NVIDIA runner, use:
  `OPTIX_DIR=/path/to/OptiX-SDK-9.1.0 bash -lc 'tools/ci/linux_check.sh; s=$?; printf
  "{\"schema\":\"strelka.optix-correctness.v1\",\"exit_code\":%d,\"status\":\"%s\"}\\n" "$s"
  "$([ "$s" -eq 0 ] && echo PASS || echo FAIL)"; exit "$s"'`. The command builds both OptiX device modules,
  rejects empty `.optixir` outputs, runs unit tests and production smoke renders, and emits a machine-readable result.

## Adversarial correction A: finite-RNG hierarchy support

- Random variables and measure: environment/local class `C`, local emitter class `K`, alias bucket `B`, alias coin
  `A`, nested triangle bucket/coin, and within-texel coordinates are distinct discrete draws from the renderer's
  23-bit uniform lattice. The selected environment texel, light, mesh, and triangle remain categorical masses;
  environment direction remains a density in `domega` after independent continuous jitter.
- Support and PDFs: each hierarchy level uses its declared PMF without conditional remapping of the parent draw.
  The marginal PMF is still the product of the independent selection masses; the stored represented alias PMF and
  all MIS densities are unchanged. Delta/continuous classification and MIS strategy enumeration are unchanged.
- Reproducer: for an environment alias threshold `2^-22` at `2^20` texels, reusing the bucket fraction as the coin
  realizes conditional mass `1/8`, not `2^-22`. Remapping a parent draw from a class of probability `2^-22` into an
  eight-light bucket reaches only two buckets on the same finite lattice.
- Implementation: Metal and OptiX now use separate dimensions for emitter class, alias bucket/coin, nested triangle
  bucket/coin, and environment texel jitter. The old remaps and hash-scrambled reuse were removed.
- Validation: both mutations fail before the change; the corrected lattice checks pass exactly. Debug and Release
  CTest pass 4/4, targeted ASan+UBSan passes, production Metal shaders compile, and the host audit passes 784/784
  tests with 68,674,130 assertions. The actual Apple M4 Pro environment kernel passes 262,144 samples with zero
  measure mismatches. OptiX uses the same dimension/specification but remains externally compile/runtime
  `UNVERIFIED` on this host.

## Adversarial correction B: lat-long pole atoms

- Random variable and measure: the selected texel is a discrete mass; its two conditional coordinates must induce
  the continuous solid-angle density `P(texel)/DeltaOmega`. The finite 23-bit jitter lattice therefore represents
  equal-probability cells by their interior midpoints rather than by a closed endpoint.
- Support/PDF: the north and south poles have zero continuous measure and no synthetic atom. Moving the lattice to
  `(k+1/2)/2^23` preserves every conditional mass while keeping sampled directions in the selected texel interior;
  selection PMFs, delta classification, and environment-vs-BSDF MIS remain unchanged.
- Reproducer/mutation: `envSampleSolidAngleV(0, H, 0)` previously returned exactly `v=0`; all azimuth texels in that
  row collapsed to the same direction but returned different per-texel PDFs. The retained closed-endpoint mutation
  still maps different `u` values to the same north pole.
- Implementation: shared CPU/Metal/OptiX mapping centres both within-texel jitter dimensions on their finite random
  lattice before applying uniform azimuth and uniform-cosine row inversion.
- Validation: the regression passes 3/3 assertions; Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean,
  production Metal shaders compile, and the full harness passes 785/785 tests with 68,674,133 assertions. The actual
  Apple M4 Pro audit passes 262,144 samples in fast and safe math with zero measure mismatches. Shared OptiX source
  uses the same mapping; CUDA compilation/execution remains externally `UNVERIFIED` on this host. Status: FIXED.
