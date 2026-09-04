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
| H. Ill-conditioned ellipsoid intersection | Inverse-space quadratic overflows for `diag(1,1,1e-20)` although sampler/PDF are finite | Thin full-rank sampled-point round trip and inverse-quadratic mutation | Solve the ray/ellipsoid quadratic in scaled homogeneous cofactor coordinates | FIXED | Shader compiled; shared source | Shared source; external CUDA validation required | pending | FIXED |
| I. Sheared spherical-rectangle sampling | Normalized non-orthogonal edges move solid-angle samples off the affine proxy plane | Sheared parallelogram plane/PDF checks and old-frame mutation | Select the exact uniform-area path for non-rectangular affine parallelograms | FIXED | Shader compiled; shared source | Shared source; external CUDA validation required | pending | FIXED |
| J. Extreme analytic area transforms | Finite axes near `1e20` overflow a raw float cofactor/determinant and used to produce NaN normal/PDF | Extreme-scale sampler/intersection/host-power test and raw-denominator mutation | Represent the area density directly; reject only transforms whose homogeneous geometry or final density is unrepresentable | FIXED | Shader compiled; shared source executed | Shared source; external CUDA validation required | pending | FIXED |
| K. Invalid transformed light frames | Singular transforms normalize local emission axes to NaN while retaining positive selection power | Scene packing and host-power regressions for spot/projector/distant/IES point | Pack a finite zero sentinel and give invalid directional records zero outer PMF | FIXED | Shader compiled; packed ABI | Same packed ABI; external CUDA validation required | pending | FIXED |
| L. Edited light proxy topology | `SPHERE -> RECT` keeps the sphere mesh while sampling a packed rectangle; infinite→area has no proxy | Type-edit topology/create regression and stale-mesh mutation | Replace/create the editor/intersection proxy and rebuild geometry; mask infinite proxies in both backends | FIXED | Source compiled | Source; external CUDA validation required | pending | FIXED |
| M. Metal affine surface normals | Metal applies `A*n`; OptiX applies inverse-transpose, changing sidedness under non-uniform transforms | Independent inverse-transpose/sign regression and forward-transform mutation | Shared cofactor inverse-transpose normal helper used at both Metal surface reconstruction sites | Oracle FIXED | Shader compiled | Existing OptiX behavior | pending | FIXED |
| N. Runtime light topology rebuild | Geometry bit was ignored; Metal TLAS retained captured masks and both backends reused missing/stale BLAS after type edits | Repeated type-edit/cache regression plus backend rebuild source/compile validation | Cache unit proxies and route Geometry changes through buffer, BLAS, TLAS, and SBT rebuilds | FIXED | Source compiled; execution pending reviewer | Source; external CUDA validation required | pending | FIXED |
| O. Scale-safe analytic and mesh area measures | Finite `1e13` ellipsoid determinant and finite `1e38` triangle length overflow intermediate float products | Large ellipsoid/disc/triangle sample-intersection-PDF regressions and determinant/naive-length mutations | Scale vectors before normalization, solve analytic intersections homogeneously, and bound area-to-solid-angle arithmetic | FIXED | Shader compiled; shared math source | Shared source; external CUDA validation required | pending | FIXED |
| P. Extreme affine normal/support consistency | Common cofactor scaling underflows the only active normal; a finite `J_A` denominator can overflow although `p_A` and `p_omega` remain finite | Extreme inverse-transpose normal, `diag(2e19,2e19,1)` density, sample/eval/intersect, shear, and overflow mutations | Transform normals from an object tangent plane and carry scale-safe `p_A` directly through sample/PDF/intersection | FIXED | FIXED on MTLDevice | Shared source; external CUDA validation required | pending | FIXED |
| P2. Reciprocal-scale analytic support | A shared max scale turns `(1e20,0,0) cross (0,1e-20,0)` into an unstable `1e-40` intermediate; finite triangle endpoints can overflow `p1-p0`; a rounded unit cosine can overflow the PDF guard | Reciprocal/large affine and triangle sample-PDF-intersection, overflow/underflow, exact miss, condition-gate, and `dot(n,n)>1` regressions | Independently scaled compensated affine algebra, analytic disc/ellipsoid hits, exponent-carrying triangle measures, bounded Skeel-valid point maps, and scale-safe `p_A r^2/cos` | FIXED | FIXED on MTLDevice | Shared source; external CUDA validation required | `40e39fc` | FIXED |
| Q. True Standard-PBR/conductor delta measures | Rough lobes below the delta threshold are sampled continuously but labelled delta; positive PDFs are floored; sheen/tiny transmission can have evaluation without proposal support | Exact mirror/refraction, sub-`1e-10` marginal, sheen-with-transmission, near-critical Snell/Fresnel, and finite-lattice tiny-lobe regressions | Sum coincident atoms as discrete masses; keep every other lobe continuous; evaluate returned float endpoints with compensated inverse/Jacobian/Fresnel arithmetic | FIXED | Shader compiled; shared source | Shared headers; external toolchain required | pending | FIXED |
| R. Finite-RNG categorical representation | Alias buckets above `2^23` are unreachable and float thresholds do not equal Metal/OptiX event masses | `8,388,609` buckets, strict-threshold lattice, hierarchy probability, support and GOF mutations | Full-width integer bucket words and integer Bernoulli thresholds; reconstruct the exactly represented marginal PMF | FIXED | FIXED on MTLDevice | Shared source; external CUDA toolchain required | this commit | FIXED |
| S. Environment radiance/support lifecycle | Invalid HDR values reach textures, bilinear positive radiance can lie outside PMF support, runtime edits leave stale tables, and a `FLT_TRUE_MIN` 1x1 map loses its outer PMF through an infinite reciprocal | sanitization, 3x3 footprint support, seam/poles, tiny-positive power, edit/reload regressions | 3x3 oracle FIXED | FIXED on MTLDevice; lifecycle source/compile | Shared source fixed; external CUDA required | this commit | FIXED |
| T. Infinite-light exact support/MIS | Sharp distant versus mirror is not represented as a discrete match; tiny continuous caps and float round trips lose support; camera mask/tMax differ | delta match, tiny cap, boundary round-trip, camera visibility and backend-distance tests | Stable analytic cone inversion and chord support; explicit sharp-distant atom; shared infinite visibility/distance | FIXED | FIXED on MTLDevice | Source fixed; external CUDA required | this commit | UNVERIFIED |
| U. Analytic/punctual visibility agreement | Shadow rays ignore analytic emitters and finite-radius punctual proxies; soft punctual spheres are nevertheless sampled with a continuous area density but classified as MIS deltas | analytic segment blockers, stacked area lights, overlapping analytic surfaces, exact parallelograms and soft-punctual hit/MIS regressions | One analytic surface query for camera/BSDF hits and finite shadow segments; only radius-free punctual sources remain delta | FIXED | FIXED on MTLDevice | Shared source fixed; external CUDA required | this commit | UNVERIFIED |
| V. Transformed frame validity | Non-finite translations and collapsed/sheared projector/IES frames keep proposal power; OptiX transforms tangents as normals | translation, partial-rank/full-frame, shear, mirrored and tangent Gram-Schmidt tests | Orthonormal profile frames, matched packing/power/device validity, and forward-vector surface tangents | FIXED | FIXED on MTLDevice | Shared source fixed; external CUDA required | this commit | UNVERIFIED |
| W. Complete marginal PDF arithmetic | Conditional area PDFs saturate before outer PMFs, under-reporting a finite complete marginal density | tiny-area/low-selection analytic and mesh regressions | pending | OPEN | OPEN | OPEN | pending | OPEN |
| X. Emissive mesh animated/textured consistency | Power support ignores shutter/interior motion; OpenPBR and Metal LOD paths disagree with hit emission | motion extrema, OpenPBR texture bridge and texture-LOD regressions | pending | OPEN | OPEN | OPEN | pending | OPEN |
| Y. Runtime topology/mask safety | Headless host geometry is released too early; finite/infinite and camera-visibility edits do not rebuild every backend mask/descriptor | headless rebuild and repeated runtime mask transitions | pending | OPEN | OPEN | OPEN | pending | OPEN |
| Z. Projector transfer convention | Metal samples sRGB while OptiX decodes the same LDR source with a gamma-2.2 path | shared texel-value fixture and source mutation | pending | OPEN | OPEN | OPEN | pending | OPEN |

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

## Adversarial correction C: representable texel interiors

- Random variable/measure: the independent azimuth and solid-angle-row jitters remain uniform conditional variables,
  represented on a finite lattice. Their emitted float direction must inverse-map to the same categorical texel whose
  `P(texel)/DeltaOmega` density is returned.
- Reproducer: for `w=4`, selected `x=1`, and the largest float below one, `float(x)+jitter` rounds to `2`, so the old
  sample returns texel 1's PDF while `pdf(direction)` reads texel 2. At the lower endpoint, seam trigonometry can map
  texel 0 to `u` just below one. Row inversion likewise rounds many largest jitters onto the next row.
- Implementation: validate the representable UV and the complete direction-to-UV round trip; only a value that rounds
  across a bin boundary falls back to that selected bin's interior uniform-cosine midpoint. This preserves support
  and prevents a finite-probability PDF mismatch without adding an epsilon support region.
- Regression: both jitter endpoints, three rotations, widths `1/4/16/1024`, and heights `1/4/8` produce 5,774 exact
  selected/evaluated-bin assertions; the old arithmetic boundary is retained as a mutation.
- Validation: Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean, production Metal shaders compile, and
  the full harness passes 786/786 tests with 68,679,907 assertions. The actual Apple M4 Pro audit passes 262,144
  samples in fast and safe math with zero measure mismatches. OptiX shares the mapping but remains externally
  compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction D: receiver frame before NEE

- Random variable/measure: the light direction is continuous in `domega`; the receiver projected-solid-angle factor
  and support must be evaluated in the identical shaded frame used by the BSDF. Selection PMFs and conditional light
  PDFs are unchanged.
- Reproducer/mutation: an opaque geometric back face with raw `N.L=-0.7` is flipped by `standard_pbr`, but the former
  early `N.L>0` connection check rejected it before the later common pairing predicate. Its BSDF-hit strategy was
  nevertheless MIS-weighted as if NEE had support.
- Implementation: common CPU/Metal/OptiX helpers construct `shadedFrame()` for both receiver support and cosine;
  fibre, dielectric-exit, diffuse-transmission, mirrored, and ordinary front/back cases retain their intended sides.
- Validation: the focused case passes 7/7 assertions; Debug and Release CTest pass 4/4, targeted ASan+UBSan is
  clean, production Metal shaders compile, and the full harness passes 787/787 tests with 68,679,914 assertions.
  The actual Apple M4 Pro environment audit remains clean; it does not execute this surface path. OptiX shares the
  predicate but remains externally compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction E: infinite-light emitter sidedness

- Random variable/measure: finite distant and dome directions are continuous in `domega`; unlike area lights they
  have no emitter-surface normal. Receiver-side support remains the shaded-frame predicate, and the light PDF keeps
  its existing selection PMF times spherical-cap/full-sphere conditional density.
- Reproducer/mutation: at distant half-angle `pi`, the declared support is the sphere, but the old area-facing test
  `-dot(L, normal)>0` rejected one hemisphere in NEE while miss evaluation retained and MIS-weighted it.
- Implementation: a shared predicate applies emitter-facing only to finite area lights; punctual and infinite lights
  are not clipped. Delta distant classification and all MIS strategy identities are unchanged.
- Validation: the focused case passes 7/7 assertions; Debug and Release CTest pass 4/4, targeted ASan+UBSan is
  clean, production Metal shaders compile, and the full harness passes 788/788 tests with 68,679,921 assertions.
  The actual Apple M4 Pro environment audit remains clean but does not execute the distant path. OptiX shares the
  predicate but remains externally compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction F: singular analytic transforms

- Random variable/measure: disc and ellipsoid points require a two-dimensional world-area measure before conversion
  to `domega`. The ellipsoid additionally requires an invertible affine map for its analytic intersection and normal.
- Reproducer/mutation: `A=diag(1,1,0)` gives the old ellipsoid sampler denominator `4pi` for an object-space pole and
  a positive host power, while analytic intersection rejects `det(A)=0`. Scene packing gives the analogous singular
  disc a zero inverse-transpose normal but its old sampler/power/intersection still retained positive area.
- Implementation: singular ellipsoids and discs with invalid packed normals have zero sampler denominator, host
  selection power, and intersection support. Valid non-uniform and mirrored transforms keep the same analytic
  surface, cofactor Jacobian, and inverse-transpose orientation.
- Validation: focused device-math and host-power cases pass 11/11 assertions; Debug and Release CTest pass 4/4,
  targeted ASan+UBSan is clean, production Metal shaders compile, and the full harness passes 789/789 tests with
  68,679,930 assertions. The actual Apple M4 Pro audit remains clean. OptiX shares the analytic math but remains
  externally compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction G: mirrored rectangle orientation

- Random variable/measure: a rectangle point is sampled in world area and converted to `domega`; its one-sided
  emitter support depends on the inverse-transpose of authored local `-Z`, including transform orientation.
- Reproducer/mutation: identity and `diag(1,1,-1)` produce byte-identical coplanar rectangle corners, so the old
  `-normalize(cross(edge1,edge2))` reconstruction returned `-Z` for both although the mirrored normal is `+Z`.
- Implementation: scene packing stores the inverse-transpose/cofactor normal for rectangles and discs; both shaders
  consume it directly. Singular records have zero selection power. Point sampling and intersected proxy geometry are
  unchanged and remain the same transformed rectangle.
- Validation: focused tests pass 9/9 assertions; Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean,
  production Metal shaders compile, and the full audit passes 790/790 tests with 68,679,937 assertions. The first
  parallel Debug/audit invocation collided in pre-existing fixed-name temporary directories; the required sequential
  rerun passed. OptiX consumes the same packed normal but remains externally compile/runtime `UNVERIFIED`.
  Status: FIXED.

## Adversarial correction H: ill-conditioned ellipsoid intersection

- Random variable/measure: the outer ellipsoid proposal is a discrete light mass followed by uniform object-sphere
  area and the affine world-area Jacobian; the induced connection remains a continuous `domega` density.
- Support/PDF: every full-rank affine sphere sample with a positive finite cofactor Jacobian must lie on the same
  analytic surface intersected by visibility. Selection PMFs, one-sided support, and continuous MIS remain unchanged.
- Reproducer/mutation: for `A=diag(1,1,1e-20)`, the old inverse-space origin/direction are order `1e20`; their
  squared quadratic coefficients overflow and the discriminant becomes `Inf-Inf=NaN`, rejecting a finite sampled
  equator point with positive PDF.
- Implementation: solve `|adj(A)(O+tD)|^2=det(A)^2` directly and divide all homogeneous terms by one common scale.
  Normal/PDF reconstruction likewise normalizes adjugate coordinates before applying determinant orientation, so it
  never needs the overflowing inverse coordinates.
- Validation: the mutation is non-finite while the corrected sample round-trips with a finite hit/normal; focused
  tests pass 14/14 assertions and all analytic tests pass 76,714/76,714. Debug and Release CTest pass 4/4, targeted
  ASan+UBSan is clean, the production Metal shader compiles, and the full audit passes 791/791 tests with 68,679,944
  assertions. The actual Apple M4 Pro environment audit remains clean; OptiX source shares the equation but CUDA
  compile/runtime remains externally `UNVERIFIED`. Status: FIXED.

## Adversarial correction I: sheared spherical-rectangle sampling

- Random variable/measure: a transformed rectangle point is continuous in world area (then `domega`); the optional
  Urena/Fajardo/King strategy is continuous directly in `domega`. Its derivation requires two orthogonal edges.
- Support/PDF: a shear makes the proxy a general parallelogram. The exact area sampler covers that affine surface
  with `p_A=1/|e_x cross e_y|`, then uses `p_omega=p_A r^2/|n dot -wi|`. Both sample and hit-side PDF choose it from
  the same shared predicate; light selection PMFs, sidedness, and MIS strategies are unchanged.
- Reproducer/mutation: for `p0=(0,0,1)`, `ex=(1,0,0)`, `ey=(1,1,0)`, the old independently normalized frame has
  `z=(0,0,1/sqrt(2))` and reconstructs its central point at `z=0.5`, outside the actual `z=1` proxy plane.
- Implementation: reject zero edges and route non-orthogonal normalized edges to the existing affine area sampler.
  The spherical-rectangle path remains enabled for its actual rectangular domain.
- Validation: all twelve fixed-grid samples lie on the sheared proxy and sample/PDF agree; the old-frame mutation is
  off-plane. Focused tests pass 39/39 assertions, Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean,
  production Metal shaders compile, and the full audit passes 792/792 tests with 68,679,983 assertions. The actual
  Apple M4 Pro environment audit remains clean; OptiX shares the predicate but CUDA compile/runtime remains
  externally `UNVERIFIED`. Status: FIXED.

## Adversarial correction J: unrepresentable analytic area transforms

- Random variable/measure: ellipsoid identity is discrete; object-sphere area, affine world area, and the induced
  `domega` density are continuous. A transform whose float determinant/cofactors overflow cannot represent that
  measure in either device backend and is classified as an invalid, zero-support analytic record.
- Support/PDF: valid finite transforms retain their existing Jacobian and MIS. Invalid transforms have zero outer
  selection power, zero conditional sample/PDF, and no analytic intersection, so no NaN-valued event is introduced.
- Reproducer/mutation: raw float cofactors and determinants overflow for axes near `1e20`; the old exact-nonzero
  test accepted `Inf`, then computed `Inf/Inf` for the normal. The initial correction conservatively rejected those
  records, but `A=diag(2e19,2e19,1)` proves that policy too broad: `J_A` overflows at the pole while the final
  `p_A=1.98944e-40` and induced `p_omega=0.0795775` are finite.
- Implementation: correction P supersedes the interim determinant gate. Shared code carries reciprocal area density,
  evaluates tangent-plane Jacobians without forming the overflowing cofactor magnitude, and rejects only when either
  the homogeneous intersection loses rank or the final density itself has no positive float representation. Host
  power applies that device predicate before its double-precision area quadrature.
- Validation: the raw determinant/cross mutations remain non-finite while the corrected sampler, evaluator,
  intersection, host selection power, and solid-angle density retain support. Full current results are recorded in
  correction P. Status: FIXED.

## Adversarial correction K: invalid transformed light frames

- Random variable/measure: light identity is a discrete PMF. Spot/projector/distant and IES point conditionals also
  require a finite nonzero directional frame; an isotropic point delta does not.
- Support/PDF: a singular transformed axis has no declared angular support and receives zero selection power. Valid
  frames keep their existing cone/projector/delta measures and MIS strategies. The packed zero vector is an invalid
  sentinel, never a fabricated fallback direction.
- Reproducer/mutation: normalizing transformed local `-Z` under `diag(1,1,0)` produced a NaN direction while host
  power stayed positive. Cone sampling and miss support then consumed NaNs or silently lost positive radiance.
- Implementation: scene packing normalizes in double only after a finite positive length check; area cofactors get
  the same finite guard. Host selection requires the direction for spot/projector/distant and IES point, while plain
  isotropic point remains valid.
- Validation: singular spot/projector/distant records are finite zero sentinels; invalid host powers are zero and an
  isotropic point remains positive. Focused tests pass 21/21 assertions. Debug and Release CTest pass 4/4, targeted
  ASan+UBSan is clean, production Metal shaders compile, and the full audit passes 795/795 tests with 68,680,008
  assertions. Actual MTLDevice environment execution remains clean; OptiX consumes the same records but CUDA
  compile/runtime remains externally `UNVERIFIED`. Status: FIXED.

## Adversarial correction L: edited light proxy topology

- Random variable/measure: changing a light descriptor must not change the meaning of the existing discrete light
  identity: its new conditional surface and hit geometry must be the same shape. Rectangles use their transformed
  proxy surface; discs/ellipsoids remain analytic; punctual/infinite lights have no radiance proxy.
- Support/PDF: a type edit replaces or creates the proxy mesh before acceleration structures rebuild. Infinite
  descriptors are masked from hardware traversal in both backends even if an editor proxy remains from an earlier
  type. Selection PMFs and the new type's conditional density are otherwise unchanged.
- Reproducer/mutation: `SPHERE -> RECT` updated the packed rectangle/transform but retained the sphere mesh;
  `DISTANT -> RECT` had no instance at all. The stale sphere topology is retained as the mutation.
- Implementation: `setLight()` resolves the desired proxy topology, replaces `mMeshId` or creates the missing light
  instance, marks geometry dirty, and updates the transform. Metal and OptiX masks explicitly suppress infinite
  proxies.
- Validation: both edit directions now produce a six-index rectangle proxy and set `ChangeBits::Geometry`; focused
  tests pass 15/15 assertions. Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean, production Metal
  shaders compile, and the full audit passes 796/796 tests with 68,680,016 assertions. Actual MTLDevice environment
  execution remains clean; OptiX source is updated but CUDA compile/runtime remains externally `UNVERIFIED`.
  Status: FIXED.

## Adversarial correction M: Metal affine surface normals

- Random variable/measure: the BSDF/light connection direction remains continuous in `domega`; receiver support and
  projected cosine are defined by the world shading/geometric normals. A tangent is a vector (`A*t`), while a normal
  is a covector (`transpose(inverse(A))*n`).
- Support/PDF: Metal and OptiX now construct the same receiver frame under non-uniform and mirrored transforms.
  Light/BSDF selection PMFs, conditional PDFs, delta classification, and MIS enumeration are unchanged.
- Reproducer/mutation: under `A=diag(2,1,1)`, object `n=normalize(1,1,0)` must become
  `normalize(0.5,1,0)`. The old forward normal is `normalize(2,1,0)`; for `wi=normalize(-0.6,0.8,0)` their support
  signs are opposite.
- Implementation: shared cofactor math returns the normalized inverse-transpose including determinant orientation;
  both Metal triangle reconstruction sites use it for shading and geometric normals, retaining forward transforms
  for tangents and edges. OptiX already uses its inverse-transpose intrinsic.
- Validation: the independent value/sign/mirror test and old-forward mutation pass 6/6 assertions. Debug and Release
  CTest pass 4/4, targeted ASan+UBSan is clean, production Metal shaders compile, and the full audit passes 797/797
  tests with 68,680,022 assertions. Actual MTLDevice environment execution remains clean but does not exercise mesh
  normals; OptiX runtime remains externally `UNVERIFIED`. Status: FIXED.

## Adversarial correction N: runtime light topology rebuild

- Random variable/measure: unchanged from correction L. This closes the runtime state transition so the discrete
  light identity's newly selected geometry is actually the geometry uploaded and intersected by each backend.
- Support/PDF: a geometry/topology edit rebuilds GPU vertex/index storage plus BLAS/TLAS/SBT as required; Metal no
  longer reuses the old emitted-instance mask. No selection or conditional density changes.
- Reproducer/mutation: `ChangeBits::Geometry` was set but ignored. Metal `rebuildTLAS()` retained captured `asIndex`
  and mask, while OptiX resolved against the old GAS list. A first infinite→rect edit also failed to cache its newly
  created shared proxy and repeated edits grew the mesh array.
- Implementation: proxy IDs are cached when first created. Metal handles Geometry with a full geometry/AS rebuild;
  OptiX recreates vertex/index buffers, BLAS, TLAS, and SBT. Transform-only edits keep the cheaper existing path.
- Validation: 100 repeated edits leave the proxy mesh count unchanged; both topology transitions use the expected
  mesh and focused tests pass 9/9 assertions. Debug and Release CTest pass 4/4, targeted ASan+UBSan is clean,
  production Metal code/shaders compile, and the full audit passes 797/797 tests with 68,680,023 assertions. Actual
  MTLDevice environment execution remains clean; the edit path is source/compile validated. OptiX remains externally
  compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction O: scale-safe analytic and mesh area measures

- Random variable/measure: analytic object-sphere or object-disc sample and emissive-triangle barycentrics induce
  world-area density `dA`, then direction at the receiver has density in `domega`. Light identity remains a discrete
  outer PMF.
- Support/PDF: every finite surface whose world-area Jacobian and final float density are representable retains
  support. `p_omega = distance^2 / (cos(theta_light) J_A)` for analytic samples and
  `p_omega = p_A distance^2 / cos(theta_light)` for triangles. Selection PMFs and delta classification are unchanged.
- Reproducer/mutation: a uniform ellipsoid at scale `1e13` has finite `4 pi J_A` but its raw float determinant
  overflows; a triangle with orthogonal `1e19` edges has finite cross product and positive area density but
  `length(cross)` squares it to infinity. The old determinant and naive-length expressions are retained as failing
  mutations.
- Implementation: shared CPU/Metal/OptiX helpers normalize with a component scale, analytic disc/ellipsoid
  intersections solve scaled equations for the same smooth surfaces sampled by NEE, and area-to-solid-angle
  conversions avoid forming an overflowing square. Host selection rejects only records whose final device area
  measure is unrepresentable.
- MIS strategies: selected-light NEE and BSDF-hit evaluation continue to use the identical analytic intersection,
  light normal, area Jacobian, and complete marginal outer PMF. No mesh proxy or additional strategy is introduced.
- Validation: focused large-scale regressions pass 32/32 assertions; Debug and Release CTest pass 4/4, targeted
  ASan+UBSan is clean, production Metal shaders compile, and the full audit passes 800/800 tests with 68,680,047
  assertions. The actual Apple M4 Pro audit passes 262,144 samples in fast and safe math with zero measure
  mismatches. OptiX shares the same source but remains externally compile/runtime `UNVERIFIED`. Status: FIXED.

## Adversarial correction P: extreme affine normal/support consistency

- Random variable/measure: mesh normals define BSDF sidedness in `domega`; analytic ellipsoid samples use object
  sphere area, affine world area, and then `domega`. No selection PMF changes.
- Support/PDF: an arbitrary mesh normal is transformed by the inverse transpose even when irrelevant affine axes
  differ by `1e60`. Analytic lights carry `p_A` directly, so an overflowing area or Jacobian does not remove support
  when the reciprocal density is positive. A light is rejected only when the homogeneous float representation shared
  by sample, PDF evaluation, and intersection loses rank, or when no positive normal-float area density survives the
  production Metal arithmetic mode. P2 supersedes the interim attempt to retain subnormal `p_A` values.
- Reproducer/mutation: for `A=diag(1e-30,1e30,1)` and object normal `Y`, global cofactor scaling underflows the only
  active `1e-30` cofactor and returns zero. The ellipsoid sampler previously returned positive area at its `Y` pole,
  while global-axis-scaled PDF/intersection arithmetic underflowed the `X` axis and returned no support.
- Implementation: transform an object-space tangent plane with `A`, take a scale-safe cross direction, and apply
  determinant orientation, avoiding irrelevant cofactor magnitudes. Evaluate
  `p_A=1/(pi |A ex cross A ey|)` for discs and `p_A=1/(4 pi |A t cross A b|)` for ellipsoids directly, then compute
  `p_omega=p_A distance^2/cos(theta_light)` without first forming either `J_A` or `distance^2`. The validity predicate
  checks the same scaled determinant used by the homogeneous analytic intersection and a conservative positive
  density bound over the complete sphere.
- Delta/continuous and MIS: unchanged; meshes and ellipsoids remain continuous, and the light/BSDF strategies use
  the same support decision. The selected-light outer PMF is unchanged.
- Validation: the interim CPU result retained `p_A=1.98944e-40` for `diag(2e19,2e19,1)`, but P2's direct MTLDevice
  probe showed that production Metal flushes that conditional density. Final shared tests therefore reject it before
  selection while preserving the largest normal density. Current complete validation is recorded in P2. Status:
  FIXED.

## Adversarial correction P2: reciprocal-scale analytic support

- Random variable/measure: the analytic-light identity remains a discrete PMF. A disc samples a uniform object-disc
  point; an ellipsoid samples a uniform object-sphere normal `N`. Their affine images are continuous in world area
  `dA`, and the induced receiver direction is continuous in solid angle `domega`.
- Support: every finite affine disc with positive normal-float `p_A`; every emissive triangle whose exact float-vertex
  area has a normal-float reciprocal; and every ellipsoid whose complete area-density range is representable and whose
  float point map has scale-invariant Skeel condition at most `2^12`. For the linear map this bounds the three-FMA
  endpoint's recovered object-coordinate drift by
  `gamma_3 K < 7.33e-4`. Rejected ellipsoids receive zero host power, sample/evaluate density, and intersection
  support together. General affine solves, transformed normals, and discs do not inherit the sphere-map gate.
- Conditional and marginal PDF: for a disc, `p_A=1/(pi |axisX cross axisY|)`; for an ellipsoid,
  `p_A(N)=1/(4 pi |cofactor(A) N|)`. The directional density is
  `p_omega=P(light) p_A distance^2/clamp(cos(theta_light),0,1)`, with the unchanged outer selection PMF included in
  the complete light-strategy density. Power-of-two exponent factoring avoids forming an overflowing area,
  determinant, squared distance, or inverse.
- Delta/continuous and MIS: both surfaces remain continuous. NEE retains the sampler's endpoint, normal, and `p_A`
  and tests the endpoint segment for occluders. A BSDF/camera hit uses the analytic disc/ellipsoid intersection and
  that event's normal and `p_A`. They are the two correct float strategy events: a quantized far/grazing NEE ray is
  not snapped to a different analytic hit merely to make the numbers equal. No proxy geometry or delta fallback is
  introduced.
- Current reproducer: a common `1e20` scale makes the reciprocal axis `1e-20` become `1e-40`; dividing by this
  intermediate before cancelling both scale factors overflows and returns zero support although the true cross
  length is one. Two `1e20` axes instead overflow the raw cross used only by intersection. Triangle endpoints
  `(-FLT_MAX,0,0)`, `(FLT_MAX,0,0)`, and `(-FLT_MAX,FLT_MIN,0)` have finite area but make `p1-p0` infinite. Row
  scaling alone also loses small endpoints that determine the area after large products cancel. Separately, a
  normalized float vector can satisfy `dot(n,n)>1`, so `maxFloat*cos(theta_light)` becomes infinity and defeats the
  saturation guard.
- Root cause and implementation: shared-max normalization can underflow one axis before reciprocal scales cancel;
  raw cross/determinant/quadratic and distance-square intermediates overflow even when the final density and hit are
  finite. The common CPU/Metal/OptiX code now factors affine columns and world rows independently by powers of two,
  retains determinant/dot/cross remainders as two-float expansions, constructs the geometric sphere discriminant,
  reconstructs the homogeneous surface event, and accepts it only after a componentwise backward-residual check.
  Disc intersection uses the same scale-safe affine solve. Triangle area and orientation use exponent-carrying
  expansions of the original vertex products, so neither overflowing edges nor subnormal normalized intermediates
  are formed; barycentric point/UV interpolation uses bounded convex lerps. Host selection derives triangle area from
  that same represented `p_A`. Area-to-solid-angle conversion distinguishes true underflow from overflow and clamps
  a rounded unit cosine before the saturation bound.
- Independent review: 20 million exponent-boundary bases produced zero false gate decisions against a long-double
  Skeel oracle (largest accepted `4095.98809`, smallest rejected `4096.01165`). For 937,653 accepted random rays the
  analytic hit classification had zero errors and zero `p_A` errors above 1% (worst `3.21e-7`); five million direct
  cofactor checks had no bad normals or errors above 1% (worst `3.65e-7`). A Decimal-100 triangle oracle then tested
  one million random float triangles in each of exponent ranges `[-38,+38]` and `[-12,+12]`: zero false accepts,
  zero false rejects, and zero density errors above 1%. The worst observed relative error was `1.096e-3` for an
  extremely thin triangle with `sin(angle) about 1e-8`.
- Validation: focused analytic/affine/emissive tests pass 2,723,157/2,723,157 assertions; full Debug and Release CTest
  pass 4/4; targeted ASan+UBSan passes 2,723,157 assertions; production Metal shaders compile. The full audit passes
  845/845 tests and 68,680,287 assertions. Its actual Apple M4 Pro fast/safe kernels execute 262,144 samples with zero
  measure mismatches and threadgroup-invariant results. A separate actual-device mesh probe gives identical support
  and exact `p_A` bits in default/fast/safe Metal for three Decimal counterexamples; normals differ only in their last
  bits. OptiX consumes the same shared headers but CUDA compilation/runtime remains externally `UNVERIFIED` on this
  macOS host. Status: FIXED.

## Finding Q: true Standard-PBR/conductor delta measures

- Random variables and measure: the Standard-PBR proposal first selects lobe `K`; dielectric transmission then
  selects Fresnel outcome `F`. `K` and `F` are discrete masses. Diffuse and rough microfacet directions have density
  in `domega`. An isotropic GGX lobe with `alpha < BSDF_DELTA_ALPHA`, or an anisotropic lobe only when both represented
  axes are below it, is an atom at the exact macro mirror direction. Refraction collapses to a pass-through atom only
  for exactly equal stored interior/exterior indices. The conductor has a single continuous GGX direction above the
  threshold and one mirror atom below it.
- Support: a positive continuous BRDF/BTDF term must have a positive proposal on its hemisphere. Sheen shares a
  cosine proposal even when opaque diffuse is suppressed by full interface transmission. Every positive lobe weight
  is widened enough to contain a point of the production 23-bit uniform lattice; finding R now uses the exact mass
  of each represented selection interval. Zero weights retain empty selection intervals.
- Conditional and marginal PDF: for continuous `wi`,
  `p_omega(wi)=sum_(k continuous) P(K=k) p_omega(wi|k)`, excluding every delta lobe. For a shared mirror atom, the
  returned discrete mass is the sum of all compatible atoms:
  `P_delta(mirror)=P(base-spec)+P(clearcoat)+P(transmission) F`, with absent or rough terms omitted. Smooth
  refraction has mass `P(transmission)(1-F)`. The sample throughput is the sum of the corresponding physical
  `f*cos` atom coefficients divided by that complete discrete mass. No positive density or selection probability is
  replaced by an arbitrary numeric floor. For rough reflection,
  `p_omega=P(K) D_visible(H)/(4 abs(V dot H))`; for rough refraction,
  `p_omega=P(K)(1-F) p_H(H) abs(L dot H)/length(eta V+L)^2`.
- Selection PMF: `P(K=k)=w_k/sum_j w_j` after support-preserving proposal regularisation of positive weights only.
  On exit hits the diffuse-transmission/interface pair is conditionally renormalized exactly as in finding 1.
- Delta classification: exact mirror/refraction outcomes carry probability mass and are marked `SPECULAR`; VNDF and
  cosine draws carry `domega` density and are marked `GLOSSY`/`DIFFUSE`. A continuously sampled half-vector is never
  relabelled as a delta event, including a rough numerical TIR fallback.
- MIS strategies: continuous light NEE competes only with the complete continuous BSDF marginal. Delta BSDF atoms
  are exclusive. This prevents the former polished-plastic case where a VNDF direction was marked exclusive while
  the same continuous GGX term remained in `eval()`, making complementary shares sum above one.
- Current-HEAD reproducer: conductor and Standard-PBR at roughness `0.02` (`alpha=0.0004`) produce RNG-dependent
  directions separated by `0.00245714` but mark both `SPECULAR`; their `eval()` still reports a continuous density.
  Full-transmission sheen evaluates to positive `f=1.26624e-4` with PDF zero and disables NEE. A clearcoat weight
  `2.40000034e-11` rounds out of the Standard-PBR CDF, so none of the `2^23` Metal RNG values can select it.
- Numerical conditioning reproducer: at an exiting interface with `eta=1.40776122`, the returned float endpoint has
  exact inverse `V dot H=0.703850938414`. Rounding that scalar before Fresnel changed the transmission PDF from the
  double oracle `2.21938104e-6` to `3.24683742e-6` (46.3%). Separately, the old Snell expression falsely classified
  a positive `1.84979161e-8` transmitted-cosine square as TIR. An exhaustive near-critical case has true remainder
  `2.80032682e-16`, which the initial two-float correction rounded negative and would have removed from support.
- Implementation: samples below the roughness threshold now construct the exact mirror/refraction atom instead of a
  VNDF direction with a specular label. Standard PBR sums all coincident mirror atoms in both returned mass and
  numerator; continuous samples are finalized by the common evaluator at the actual rounded endpoint. Full-vector
  GGX density, Smith terms, reflection half vectors, and refraction Jacobians avoid cancellation/overflow. The
  inverse refraction carries a two-float `V dot H` into Fresnel; scalar Snell evaluation uses an adaptive exact
  binary32 expansion near the critical boundary. Fresnel and `refract_dir` share that discriminant, and the refracted
  vector is assembled from compensated products. Exact categorical realization of very small represented masses is
  deliberately the next, separate finding R.
- Independent numerical result: the reviewer compared 19,543,112 near-critical rough transmissions with zero PDF
  errors above `0.1%` (worst `3.87e-5`). An exhaustive exact-sign/root sweep over `52,428,795` float pairs around
  the critical boundary has zero false accepts, zero false rejects, zero root errors above `0.1%`, and worst relative
  error `5.9586e-8`. Rotated Snell fuzz covered 9,992,157 valid events with zero wrong-side, non-finite, or root
  errors above `0.1%`; Standard-PBR/conductor double-oracle PDF sweeps saw worst relative errors `5.7493e-6` and
  `1.1713e-6`. Delta mass/throughput fuzz covered 12,944,000 events with zero disagreements.
- Validation: focused Debug and ASan+UBSan runs each pass 42/42 cases and 57,306,907/57,306,907 assertions; full
  Debug and Release CTest pass 4/4; production `wavefront.metal` compiles. The full audit passes 878/878 tests and
  68,677,725 assertions, retains environment integral `1`, Lambertian estimate `1.002147074` inside its CI, and
  detects the `2.003353165` legacy environment mutation. Its actual Apple M4 Pro kernel passes 262,144 samples with
  zero measure mismatches, but that kernel does not execute the BSDF changes. `clang-tidy` is unavailable on this
  host; OptiX consumes the shared headers but CUDA compilation/execution remains externally `UNVERIFIED`. Status:
  FIXED.

## Finding R: finite-RNG categorical representation

- Random variables and measure: an alias draw uses a full-width unsigned bucket word `B` and an independent
  full-width unsigned coin word `C`; both are discrete uniform variables over the `2^32` integer lattice. The
  resulting light/environment/mesh/triangle identity is a categorical probability mass. Its conditional area or
  solid-angle sample remains a separate continuous variable and measure.
- Support: for `1 <= N <= 2^32`, bucket `i` owns every integer `b` for which
  `floor(b N / 2^32)=i`. Each of the first `N` categorical buckets therefore owns at least one integer state. An
  alias branch owns exactly `T_i` coin states through the predicate `C < T_i`; a positive represented branch uses a
  positive integer threshold, while zero input weights remain unreachable.
- Conditional and marginal PMF: let `M=2^32`, `K_i` be the exact number of bucket words mapped to bucket `i`,
  `T_i` the stored integer threshold, and `a_i` its alias. Then
  `P(B=i)=K_i/M`, `P(J=i|B=i)=T_i/M`, and `P(J=a_i|B=i)=1-T_i/M`. The represented marginal used by MIS is
  `P(J=j)=sum_i (K_i/M)[1(i=j)T_i/M + 1(a_i=j)(1-T_i/M)]`. A self-alias represents the whole bucket regardless of
  threshold. This exact finite-lattice distribution, rounded once into the uploaded PMF field, is authoritative for
  both sampling and PDF lookup.
- Selection PMFs and classification: environment/local and analytic/mesh binary class choices are discrete masses
  and must use the same integer-threshold convention. Alias identity selection remains discrete; the selected
  emitter's delta/continuous classification is unchanged. NEE and BSDF-hit/miss MIS multiply by the identical
  represented outer PMFs.
- Reproducer: the old Metal PCG path exposes only `2^23` distinct floats. Sweeping all of them through
  `floor(u * 8,388,609)` reaches exactly `8,388,608` buckets, leaving one positive bucket impossible. A general float
  alias threshold also denotes a real number that is not the strict-comparison mass realized by that finite float
  lattice. The regression fails on the current implementation with `8,388,608 != 8,388,609` before production
  changes.
- Implementation: shared CPU/Metal/OptiX code maps a full 32-bit random word to a bucket with multiplication-high,
  and represents every nontrivial alias coin as a 32-bit integer threshold. Host construction reconstructs the
  marginal PMF from the exact number of words owned by each nonuniform bucket and the exact coin count, then uploads
  that PMF for MIS lookup. Environment/local and analytic/mesh class choices use the same integer Bernoulli
  convention. The existing four-byte alias field is reinterpreted from `float` to `uint32_t`, so table ABI sizes and
  memory consumption do not increase. Continuous position/direction samples retain independent float dimensions.
  Standard-PBR lobe and Fresnel choices use a common exact 23-bit sub-lattice because their public sampling API also
  serves host tests and continuous draws as floats; their physical coefficients remain in the numerator, while the
  represented proposal mass is used in both sample and eval PDFs.
- Mutation sensitivity: the old float bucket mapping reaches only `8,388,608/8,388,609` positive buckets. A separate
  strict-comparison mutation shows that a nominal float probability `0.3` differs from its event mass on the old
  23-bit lattice. Integer boundary tests check both `T-1` and `T`, and PMF reconstruction explicitly accounts for
  unequal multiplication-high bucket populations.
- Validation: focused distribution and BSDF tests pass 87/87 cases and 66,002,781 assertions under ASan+UBSan.
  Debug and Release CTest pass 4/4, production Metal shaders compile, and changed CPU/Metal files are clean under the
  repository clang-tidy configuration. The full audit passes 881/881 cases and 68,677,694 assertions; environment
  normalization is `1`, the Lambertian estimate is `1.002147074` inside its CI, and the legacy environment mutation
  remains detected at `2.003353165`. The actual Apple M4 Pro fast/safe kernels execute 262,144 samples with zero
  measure mismatches. Direct OptiX lint/compilation is blocked on this macOS host by the absent `optix.h`; the CUDA
  path consumes the same integer table ABI and shared selection source but remains externally `UNVERIFIED`. Status:
  FIXED for CPU and Metal, UNVERIFIED for OptiX execution.

## Finding S: environment radiance/support lifecycle

- Random variables and measure: next-event sampling first selects the environment as a discrete emitter class, then
  a discrete lat-long texel bin `I`, and finally a continuous direction `W` uniformly in solid angle inside that bin.
  The map lookup itself is a bilinear, continuous reconstruction in UV; it is not another sampling strategy.
- Support: uploaded RGB is sanitized channel by channel to finite nonnegative radiance. Bin `I` has proposal support
  whenever any texel in the 3x3 reconstruction footprint that hardware linear filtering can reach from inside `I`
  has positive luminance. Horizontal neighbours wrap across the seam and vertical neighbours clamp at the poles.
  Open-interval solid-angle jitter continues to exclude the coordinate poles themselves.
- Conditional and marginal PDF: with footprint envelope `q_i`, exact bin solid angle `DeltaOmega_i`, and
  `Z=sum_j q_j DeltaOmega_j`, the target texel mass is `P(I=i)=q_i DeltaOmega_i/Z`; the represented alias mass is
  authoritative after integer quantization. Inside the bin, `p(W|I=i)=1/DeltaOmega_i`, hence
  `p_omega(W)=P(I=i)/DeltaOmega_i`. The outer light density is
  `P(environment class) p_omega(W)`.
- Selection PMF: environment-versus-local selection uses the finite double-precision map integral `Z` directly.
  It must not recover `Z` by taking the reciprocal of a float normalization scale, because a tiny positive map can
  make that reciprocal overflow to infinity and turn the recovered power into zero.
- Delta classification and MIS: a textured environment remains continuous in `domega`. Environment NEE and a
  BSDF-sampled miss are the complementary MIS strategies and use the same represented texel and outer class PMFs.
  Resource edits change the integrand and proposal together: map/background replacement or removal must update the
  texture, dimensions, alias table, power, flags, miss record, and accumulation state before the next launch.
- Current-HEAD reproducers: a 3x3 map with one bright centre has zero proposal PDF in an adjacent black bin although
  hardware bilinear filtering returns positive radiance in half of that bin. A 1x1 `{NaN,1,0}` texel is discarded
  entirely even though sanitizing the invalid channel leaves positive green radiance. A 1x1 `FLT_TRUE_MIN` map has
  positive double total power but stores an infinite float reciprocal; both backends invert that infinity back to
  zero when computing the environment's outer selection power. Metal reloads only the lighting map on an Env edit,
  never clears a removed map/background, and OptiX does not consume or apply `ChangeBits::Env` after scene load.
- Implementation: the shared host builder sanitizes every RGB channel independently to finite nonnegative radiance
  before either backend uploads the texture. Its proposal uses the maximum sanitized luminance over the exact 3x3
  bilinear reconstruction footprint, horizontally wrapped and vertically clamped; `solidAnglePdf` remains the
  represented alias mass divided by the exact row solid angle. The physical radiance integral is retained separately
  from the proposal normalizer and carried in double precision into environment/local power selection, while the
  legacy float normalization field is saturated instead of becoming infinity. Metal now releases and rebuilds both
  lighting/background resources on every Env edit or scene build. OptiX owns the environment arrays/objects as one
  resettable set, consumes `ChangeBits::Env`, refreshes map/background flags and the miss SBT, recomputes outer PMFs,
  resets accumulation, and invalidates SHARC before the next launch.
- Mutation sensitivity and numerical result: before the change the focused suite reported 11 failures: all eight
  neighbouring 3x3 bins had PDF zero, the `{NaN,1,+Inf}` texel had total power zero, and `FLT_TRUE_MIN` produced an
  infinite normalization scale. The regression retains the old reciprocal explicitly: it is infinity and recovers
  zero power. Seam/pole tests independently expect support only in wrapped `x={W-1,0,1}` and clamped `y={0,1}` bins,
  so a global nonzero floor or vertical wrap also fails.
- Validation: focused tests pass 43/43 cases and 986,236 assertions in Debug and ASan+UBSan. Debug and Release CTest
  pass 4/4 and production Metal shaders compile. The full audit passes 887/887 cases and 68,777,759 assertions;
  environment normalization remains `1`, the Lambertian estimate remains `1.002147074` inside its CI, and the legacy
  mutation remains detected at `2.003353165`. The actual Apple M4 Pro fast/safe kernels execute 262,144 environment
  samples with zero measure mismatches; that kernel validates the shared table ABI and sample/PDF pair, while runtime
  resource replacement is source/compile validated rather than claimed as GPU execution. OptiX consumes the shared
  table and sanitization but CUDA compilation/runtime remains externally `UNVERIFIED`. Status: FIXED.

## Finding T: infinite-light exact support and MIS

- Random variables and measure: analytic-light identity `J` is a discrete PMF. A finite-angle distant additionally
  draws direction `W` continuously in solid angle over a spherical cap; a dome draws `W` continuously over the full
  sphere. A renderer-sharp distant has only one directional atom `W=-normal` and no `domega` density.
- Support: a finite distant supports exactly the unit directions whose chord distance from its axis is at most
  `2 sin(halfAngle/2)`; this is the cancellation-free form of the cap boundary. A dome supports the full sphere. A
  sharp distant supports its exact represented axis only, and only a preceding discrete/specular BSDF event can
  reach that atom on the complementary miss path. Camera and secondary visibility bits apply to every analytic
  infinite emitter just as they already apply to analytic area hits.
- Conditional and marginal PDF: for finite distant,
  `p(W|J)=1/[4 pi sin^2(halfAngle/2)]` inside the cap and zero outside; for dome, `p(W|J)=1/(4 pi)`. The complete NEE
  density is `P(local) P(analytic|local) P(J|analytic) p(W|J)`. The sharp distant carries only those discrete
  selection masses and a unit conditional placeholder for estimator division; it never enters a continuous MIS
  heuristic.
- Selection PMF: unchanged from findings R/S and read from the represented integer-alias table. Camera visibility
  changes integrand support, not selection probability; a camera-hidden light remains available to secondary NEE
  and secondary BSDF paths.
- Delta/continuous classification: zero, invalid, and positive angles too small to retain a finite, normal-float cone
  measure on all supported GPU arithmetic modes are sharp atoms. Every other positive clamped angle is continuous.
  Host radiometric baking uses the same classification and exact cone measure rather than an unrelated `1e-8`
  solid-angle floor.
- MIS strategies: finite distant and dome use selected-light NEE plus the compatible non-delta BSDF miss strategy.
  A sharp distant uses selected-light delta NEE; a coincident specular BSDF atom is an exclusive discrete path and is
  evaluated on miss with unit MIS weight. A continuous BSDF ray never acquires sharp-distant radiance merely because
  it is numerically close to the axis.
- Current-HEAD reproducer: `distantLightIsDelta(1e-30f)` is false although its cone measure is subnormal/flushable and
  the conditional PDF is not portable across CPU and Metal arithmetic modes. At `halfAngle=1e-5`, irradiance baking
  divides by the unrelated `1e-8` floor instead of the analytic cone measure, so multiplying the baked radiance by
  the sampled measure recovers only about `0.15708` of the authored irradiance. Both sampler copies recover
  `sin(theta)` from `sqrt(1-cos(theta)^2)`, which collapses narrow samples after `cos(theta)` rounds to one; miss
  support repeats the cancellation through `cos(halfAngle)`. Sharp distant is explicitly skipped by both miss
  programs, all infinite analytic lights ignore their camera mask there, and distant/dome NEE traces to `1e9` while
  environment and BSDF-miss visibility use `1e16`.
- Implementation and mutation sensitivity: both shader samplers now call one stable inversion that computes
  `sin(theta)=2 sin(a/2) sqrt(q[1-q sin^2(a/2)])`; the old `sqrt(1-cos^2(theta))` mutation still collapses a
  positive-measure `1e-5`-radian sample to the axis. Sample and miss lookup use the same chord predicate, and only a
  finite RNG endpoint that rounds outside is remapped to a strict interior representative rather than widening
  support. Angles below `2^-62` use the sharp atom on host and device; the threshold keeps `sin^2(a/2)` normal even
  under fast-math reassociation. Exact sharp-distant/specular atom equality adds the missing BSDF miss contribution
  with unit weight. Shared visibility masks and `1e16` distance now cover dome and distant NEE consistently.
- Corrected result: the `1e-5`-radian irradiance bake recovers `5.0` instead of `0.15708`. A fixed-grid cap test and
  100,000 random directions have exact sample/evaluated-PDF agreement; all returned densities are positive and
  finite and agree with a double oracle within `2e-6` relative error. The tiny-cap, exact-atom, near-atom rejection,
  camera/secondary-mask, invalid-mask, full-sphere, and common-distance regressions all pass.
- Validation: focused Debug, Release, and ASan+UBSan pass 400,974/400,974 assertions. Full Debug/Release CTest pass
  4/4; production Metal shaders compile. The full audit passes 892/892 cases and 69,178,701 assertions, preserving
  environment integral `1`, Lambertian estimate `1.002147074` inside its CI, detected legacy mutation
  `2.003353165`, and dome MIS ratio `1`. On the actual Apple M4 Pro, a separate 262,144-direction cap/atom/mask
  kernel passes with six zero failure counters in both fast and safe math; the standard GPU audit also retains zero
  measure mismatches. OptiX shares the source but cannot compile or execute without the absent CUDA/OptiX toolchain,
  so that backend remains externally `UNVERIFIED` under the documented finding-7 command.

## Finding U: analytic and punctual visibility agreement

- Random variables and measure: the analytic-light identity `J` is a discrete mass. A rectangle, disc, ellipsoid,
  or positive-radius punctual sphere then samples a point `X` with density in world area `dA`; its connection
  direction `W=(X-P)/|X-P|` has density in receiver solid angle `domega`. A punctual source at or below the shared
  softness threshold samples one position atom and remains a delta event.
- Support: finite analytic surfaces use their exact represented geometry: an affine parallelogram, affine disc,
  affine ellipsoid, or radius-`r` sphere. Camera/BSDF intersection and shadow blocking use this same closed surface;
  emission itself retains its one-sided surface support. The offset target lies strictly on the receiver side of the
  sampled surface, so the selected emitter is not its own blocker. Infinite lights and sharp punctual atoms have no
  finite occluding surface.
- Conditional and marginal PDF: rectangle `p_A=1/|e_x cross e_y|`, disc
  `p_A=1/(pi|a_x cross a_y|)`, ellipsoid `p_A(n)=1/(4pi|cofactor(A)n|)`, and soft punctual
  `p_A=1/(4pi r^2)`. Each continuous direction uses
  `p_omega(X)=p_A(X)|X-P|^2/|n_X dot(-W)|`; the complete density multiplies by the unchanged environment/local,
  analytic-class, and represented light-selection PMFs. A sharp punctual connection carries only those discrete
  selection masses and its delta conditional placeholder.
- Selection PMF: unchanged from findings R/S. Turning a soft punctual source into a reachable continuous surface
  does not change which light identity the alias table selects; it changes only that identity's conditional measure
  and enables the matching BSDF-hit strategy.
- Delta/continuous classification: point, spot, and projector lights are delta only when their packed radius is at
  or below `STRELKA_SOFT_LIGHT_RADIUS_MIN`. Above it they are ordinary continuous spherical emitters. Rectangle,
  disc, ellipsoid, and soft punctual surfaces are opaque analytic visibility blockers even from their non-emitting
  side; sidedness controls emitted radiance, not geometry existence.
- MIS strategies: selected-light NEE and a non-delta BSDF ray hitting the identical analytic surface are
  complementary. Their light density contains the same outer PMF and area-to-solid-angle Jacobian. Sharp punctual
  NEE is exclusive. A second finite emitter intersecting the open shadow segment suppresses the selected-light
  contribution before either strategy adds it.
- Current-HEAD reproducer: the focused light suite passes 4,227 assertions because it codifies the old convention
  that every punctual light is delta. Source inspection shows `RAY_MASK_SHADOW` contains geometry only, every light
  proxy is absent from that mask, and the manual analytic hit loop contains only discs and ellipsoids. Thus a rear
  disc seen through a front disc is accepted by NEE even though a BSDF ray terminates at the front disc. A soft point
  already samples `p_A=1/(4pi r^2)` but `lightIsDeltaForMis()` returns true and the same sphere has neither a hit nor
  a shadow-intersection path.
- Implementation and mutation sensitivity: one shared dispatch now intersects rectangles, affine discs,
  ellipsoids, and soft punctual spheres for camera/BSDF rays and for every open shadow segment. The selected
  continuous emitter supplies an offset near-side endpoint, so it cannot shadow itself, while a nearer or stacked
  analytic surface remains an opaque blocker independent of emission sidedness. The old shadow-mask mutation still
  has no analytic support, and changing the soft punctual classification back to delta makes the new MIS assertion
  fail. Continuous light coordinates use centres of their represented 23-bit RNG cells; the closed-endpoint mutation
  puts finite probability on rectangle edges and sphere poles and produced one actual-device rectangle miss in
  262,144 samples before this correction.
- Corrected result: 4,096 exact parallelogram samples and 4,096 soft-sphere samples round-trip through the common
  intersection with positive finite area/solid-angle densities. Complementary balance weights sum to one, sharp
  punctual sources retain their discrete classification, the selected endpoint is excluded, and a nearer analytic
  surface blocks the segment. The actual M4 probe reports zero rectangle hit, open-endpoint, soft-sphere hit,
  occlusion, or classification failures in fast and safe math at threadgroup sizes 32, 64, and 128.
- Validation: focused Debug, Release, and ASan+UBSan pass 24,709/24,709 assertions. Full Debug and Release CTest pass
  4/4; production Metal shaders compile. The complete audit passes 895/895 cases and 69,215,589 assertions, with
  environment integral `1`, Lambertian estimate `1.002147074` inside its CI, detected legacy mutation `2.003353165`,
  and dome MIS ratio `1`. The standard and finding-specific actual Apple M4 Pro kernels each execute 262,144 samples
  with zero measure/support failures. OptiX consumes the same intersection, PDF, endpoint, and blocker source, but
  CUDA compilation/runtime remains externally `UNVERIFIED` under the finding-7 command. Status: FIXED for CPU and
  Metal, UNVERIFIED for OptiX execution.

## Finding V: transformed frame validity

- Random variables and measure: light identity `J` remains a discrete mass. A sharp punctual or distant source then
  has its existing directional atom; a soft punctual source samples world area before conversion to `domega`; spot,
  projector, and IES profiles deterministically modulate radiance as a function of direction. A mesh BSDF direction
  remains continuous in `domega`; its tangent frame only changes the integrand and lobe orientation.
- Support: every finite light needs a finite world position when its conditional is positional. Spot and distant
  lights need one finite nonzero emission axis. Projectors and IES lights additionally need a full-rank finite local
  frame. A general affine transform is represented by an orthonormal frame obtained from its forward-transformed
  tangents and inverse-transpose emission normal; collapsed frames have empty radiometric and proposal support.
- Conditional and marginal PDFs: valid affine reorientation does not change a punctual delta mass, sphere area
  density, or any outer selection PMF. The complete continuous density remains the represented outer PMFs times the
  existing conditional `p_omega`; an invalid record has both physical radiance and proposal mass zero. Surface BSDF
  PDFs are evaluated in a tangent produced by forward vector transformation followed by Gram-Schmidt against the
  inverse-transpose shading normal.
- Selection PMF: `analyticLightPower()` must reject the same invalid position/frame records that packing and device
  evaluation reject. Valid records continue through the integer alias construction unchanged.
- Delta/continuous classification and MIS: unchanged. Frame validity cannot turn a finite-area event into an atom or
  vice versa. NEE and BSDF-hit/miss strategies retain their prior densities; both merely see the same valid emission
  profile and orthonormal receiver frame.
- Current-HEAD reproducer: `Scene::updateLight()` writes a non-finite transformed translation into `points[1]` while
  retaining positive color and visibility. A projector under `diag(0,1,1)` retains a valid `-Z` axis and therefore
  positive selection power even though its `+X` image axis is zero. Under an XY shear, the two separately normalized
  packed axes are not orthogonal. OptiX reconstructs mesh and curve tangents with
  `optixTransformNormalFromObjectToWorldSpace`, while Metal correctly uses the forward vector transform; under
  `diag(2,1,1)` those operations point in different directions.
- Implementation and mutation sensitivity: scene packing builds a profile frame by modified Gram-Schmidt from
  forward-transformed local `+X/+Y` tangents and the inverse-transpose local `-Z` emission normal. Scale and shear no
  longer distort the angular coordinate system, while the transformed tangent signs retain mirrored image/profile
  orientation. A non-finite position or required collapsed frame zeros both radiance/visibility and host proposal
  power. Projector and IES evaluation revalidate the same packed frame on both devices. Mesh and curve tangents use
  forward vector transport and are then Gram-Schmidt orthogonalized against the inverse-transpose normal. Restoring
  OptiX's old normal transform leaves `abs(N dot T)>0.5` in the retained `diag(2,1,1)` mutation; independently
  normalizing the sheared light axes reproduces the pre-fix unit-length and orthogonality failures.
- Corrected result: the pre-fix regressions reported 8 failures across non-finite translation, collapsed projector,
  collapsed IES, and a sheared frame. They now pass, including mirrored orientation. An independent double-precision
  oracle checks 4,096 random full-rank affine frames; all three production axes agree within `2e-6` and every pair is
  orthogonal within `2e-6`. The transformed-tangent regression agrees with forward-vector transport and has unit
  length with `N dot T=0` to the same tolerance.
- Validation: focused Debug, Release, and ASan+UBSan each pass 15/15 cases and 28,765/28,765 assertions. Full Debug
  and Release suites pass; production Metal shaders compile. The complete audit passes 899/899 cases and 69,244,287
  assertions, retaining environment integral `1`, Lambertian estimate `1.002147074` inside its CI, detected legacy
  mutation `2.003353165`, and dome MIS ratio `1`. A finding-specific actual Apple M4 Pro kernel executes 262,144
  samples in fast and safe math at three threadgroup sizes with zero frame/tangent or existing audit failures. OptiX
  source uses the identical Gram-Schmidt helper and forward tangent transform, but CUDA compilation/runtime remains
  externally `UNVERIFIED`. Status: FIXED for CPU and Metal, UNVERIFIED for OptiX execution.
