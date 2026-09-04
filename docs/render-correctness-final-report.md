# Render correctness final report

## Scope and baseline

This report covers the correctness series beginning at
`172c35bcb661c72b44a2a627453ba48c55d6a36b` (`Correct Metal environment sampling measure`). The pre-run audit
artifacts and unrelated working-tree files were copied to `.audit-preserved/172c35b-pre-correctness-run/` and were
kept out of every commit. No ReSTIR, light BVH, GRIS, RTXDI, or unrelated renderer architecture work is included.

The full random-variable, measure, support, conditional/marginal PDF, selection-PMF, delta classification, and MIS
record for every finding is in [render-correctness-fix-plan.md](render-correctness-fix-plan.md). This report records
the final outcome and reproducible validation.

## Outcome

| finding | CPU/reference | Metal | OptiX | final status |
| --- | --- | --- | --- | --- |
| 1. Standard PBR mixture PDF | regression and double oracle pass | shared shader compiled; exercised in renderer | shared source changed; external build required | FIXED; OptiX execution unverified |
| 2. Distant/dome support and MIS | analytic oracle pass | compiled and actual-device mixed-emitter render agrees | source changed; external build required | FIXED; OptiX execution unverified |
| 3. Back-face NEE/MIS pairing | exhaustive support oracle pass | shared predicate compiled | shared predicate changed; external build required | FIXED; OptiX execution unverified |
| 4. Affine analytic lights | double sample/intersection/PDF oracle pass | compiled and actual-device kernel pass | source changed; external build required | FIXED; OptiX execution unverified |
| 5. Large discrete distributions | million-bin support and GOF pass | million-bin actual-device kernel pass | alias source changed; external build required | FIXED; OptiX execution unverified |
| 6. Emissive mesh NEE | hierarchy, transform, visibility, and integral oracles pass | actual renderer NEE on/off agrees | source changed; external build required | FIXED; OptiX execution unverified |
| 7. Cross-backend probability convention | shared host hierarchy passes | compiled and actual-device renderer pass | source migrated; external build required | UNVERIFIED: OptiX hardware/toolchain blocker |

`UNVERIFIED` is used deliberately for cross-backend execution: this Apple-silicon macOS host has no `nvcc`, OptiX
SDK, or NVIDIA GPU. Host emulation and Metal execution are not reported as OptiX execution. No known correctness
counterexample remains in the compiled CPU/Metal scope or in the shared OptiX math inspected by the reviewer.

## Root causes and corrections

### 1. Standard PBR mixture PDF

The sampler returned only the chosen lobe's weighted conditional density. Several non-delta lobes can generate the
same direction, so `sample.pdf` disagreed with both `evalPdf` and the actual marginal proposal.

Old, for selected lobe `K`:

`p_old(wi) = alpha_K p_K(wi)`

New, for a non-delta direction:

`p(wi) = sum_k alpha_k p_k(wi)`

The sum includes every compatible reflection or transmission lobe after exit-side conditional renormalization.
Singular mirror/refraction events retain discrete mass and are never added to a solid-angle density.

### 2. Distant/dome support and complementary MIS

Dome descriptors were lost during packing, zero-angle distant lights were treated as zero-density continuous lights,
and finite infinite emitters had a light-sampled strategy without the complementary BSDF-miss evaluation.

For finite distant half-angle `a`:

`p(w | distant) = 1 / [4 pi sin^2(a/2)]` inside the same spherical cap used by miss evaluation.

A dome uses `1/(4 pi)` on the sphere. A zero-angle distant is a Dirac event with categorical selection mass and no
continuous `p_omega`. NEE and BSDF-miss now enumerate the same support and use the same complete marginal light PDF.

### 3. Metal back-face NEE/MIS double counting

NEE tested direction support in one sidedness frame, while the bounce flag that enabled hit/miss MIS used another.
The same path event therefore received shares `0.4 + 1.0 = 1.4`.

One `ShadedFrame` and one support predicate now decide both proposal validity and complementary-strategy pairing.
For every shared continuous event, balance-heuristic shares are again

`p_L/(p_L+p_B) + p_B/(p_L+p_B) = 1`.

### 4. Non-uniform analytic-light transforms

Sampling normalized transformed axes and kept a scalar radius, while traversal intersected coarse or differently
transformed proxies. This produced a factor-six area mismatch for the reproducer and sampled points outside the
intersected 16-gon.

The production sampler and production intersection now use the same smooth affine disc or ellipsoid. For linear part
`A` and object normal `n_o`:

`J_A = abs(det(A)) length(transpose(inverse(A)) n_o)`

`p_A_world = p_A_object / J_A`

`p_omega = p_A_world distance^2 / abs(n_world dot (-wi))`

Normals use the inverse transpose and the absolute determinant keeps mirrored transforms valid. Tessellation remains
editor-only; visibility and radiance traversal are analytic.

### 5. Robust large distributions

The flat float CDF accumulated a dominant prefix until 209,715 of 1,048,576 positive tail probabilities had identical
adjacent endpoints and could not be selected. Rebuilding only in FP64 would not repair the uploaded float table.

A Walker/Vose alias table now stores the actual float-represented proposal. For bucket `b` and alias `a_b`:

`P(J=i) = [q_i + sum_(b:a_b=i)(1-q_b)] / N`.

The lookup PMF is reconstructed from the final float thresholds. Positive probabilities are regularized above the
smallest reachable 23-bit sampler branch; zeros remain unselectable. The returned marginal light density uses this
represented PMF.

### 6. Emissive mesh NEE

Material-emissive triangles contributed only when a BSDF ray happened to hit them. They were absent from direct-light
selection, and there was no matching hit-side marginal PDF.

The baseline hierarchy is now:

`P(local) P(mesh | local) P(instance | mesh) P(triangle | instance) p_A(x | triangle)`

with `p_A = 1/A_world` and

`p_omega = p_A distance^2 / abs(n_light dot (-wi))`.

The same nested represented PMFs are used by NEE and BSDF-hit MIS. Sampling evaluates textured emission at the
barycentric UV. Affine instances use exact transformed triangle area and inverse-transpose normals. Visibility traces
the segment between independently scale-offset source and analytic target endpoints, so the sampled point and the
visible endpoint remain the same event.

### 7. CPU/Metal/OptiX probability convention

Metal selected environment/local and analytic/mesh classes by power, while OptiX hard-coded a 1/2 split and uniform
analytic identities. OptiX also used centre-row environment weights with uniform-`v` jitter.

Both hosts now use the same nested power probabilities and float-represented alias PMFs. An environment texel has
discrete mass `P_i`; for row bounds `theta_0`, `theta_1`:

`DeltaOmega_i = (2 pi / width) [cos(theta_0) - cos(theta_1)]`

Both shaders draw `phi` uniformly and `cos(theta)` uniformly within the chosen row, then return and evaluate

`p_omega(w) = P_i / DeltaOmega_i`.

This is the exact-bin form of the lat-long Jacobian

`phi = 2 pi u`, `theta = pi v`, `domega = 2 pi^2 sin(theta) du dv`,

so a UV density obeys `p_omega = p_uv / [2 pi^2 sin(theta)]`. Poles are sampled through row solid angle rather than
division by `sin(theta)`, avoiding artificial polar support and non-finite PDFs. A 1x1 constant map is exactly
`1/(4 pi)` almost everywhere.

## Commits

| commit | scope |
| --- | --- |
| `172c35b` | correct Metal environment sampling measure |
| `24c49aa` | add the persistent correctness plan and probability records |
| `f845f2c` | return the Standard PBR marginal mixture PDF |
| `ad4c3d9` | complete finite distant/dome support and miss-side MIS |
| `b214620` | pair Metal back-face NEE and BSDF strategies |
| `495e2a7` | sample, intersect, and test smooth affine analytic lights |
| `cd2dd98` | replace the support-losing float CDF with represented alias PMFs |
| `d53e662` | add baseline emissive-mesh NEE and BSDF-hit MIS |
| `0b85cd9` | unify environment and light-selection conventions across backends |
| `1a25b06` through `955da31` | focused adversarial fixes recorded individually in the fix-plan table |
| `3188bfa` | remove environment round-trip fallback atoms |
| `6592691` | preserve authored POWER under transformed analytic-light area |
| `1cebdd5` | preserve the finite-distant continuous boundary measure |
| `a639086` | make primary sharp-distant visibility agree across shaders |
| `223233f` | retain every coincident analytic emitter MIS component |
| `44a2479` | remove dead sampling-audit production baggage after Ponytail review |
| `dc3173f` | define SHARC float conversion and distance arithmetic at numeric limits |

Each correctness finding has its own production commit. Later report/review commits contain documentation or scoped
simplification only.

## Numerical results before and after

| check | before | after |
| --- | ---: | ---: |
| mixed Standard PBR sampled/evaluated PDF mismatches | 28,923 / 36,794 | 0 / 36,818 |
| 1x1 environment PDF integral | legacy mutation `0.6366197724` | `1.0` |
| 1x1 Lambertian relative radiance | legacy mutation `2.003353165` | `1.002147074`, 95% CI `[0.998843265, 1.005450884]` |
| dome one-bounce Lambertian ratio | `0.2988202609` | `1.0` |
| back-face complementary MIS share | `1.4` | `1.0` |
| transformed-disc geometry/sampler area ratio | `6.0` | `1.0` oracle agreement |
| 16-gon proxy/ideal-disc area | `0.9744953584` | common smooth analytic support |
| transformed 600 W rect/disc/ellipsoid | `3600`, `3600`, `2333.94` W | all `600` W within `2e-4` relative error |
| collapsed positive million-light bins | 209,715 / 1,048,576 | 0 / 1,048,576 |
| mixed-emitter Metal NEE vs BSDF-only mean | no complete estimator | `0.1808142214` vs `0.1808202792` (`-0.003350%`) |

The retained mutations are detected, including selected-lobe PDF, delta-in-continuous-MIS, omitted infinite miss,
omitted back-face pairing, normalized analytic axes, coarse intersection proxy, flat float CDF, omitted light
PMF/Jacobian, legacy environment measure, finite-distant boundary collapse, local-area POWER, and single-emitter
tie-breaking. Exact per-finding counts are retained in the fix plan.

## Validation

- Final CPU suite: 929/929 cases and 69,256,863/69,256,863 assertions pass.
- Debug: CTest 4/4 passes; both production Metal shader configurations compile.
- Release: CTest 4/4 passes; both production Metal shader configurations compile.
- Sanitizers: the full 929-case ASan+UBSan suite passes with all 69,256,863 assertions and no diagnostic after setting
  `ASAN_OPTIONS=detect_leaks=0`; Apple reports leak detection itself unsupported. The former `sharc_grid.h` NaN-to-int
  and signed-overflow findings have dedicated regressions and are fixed in `dc3173f`.
- Static checks: changed host/test translation units pass the Homebrew LLVM `clang-tidy`; the analyzer still reports a
  pre-existing potential leak inside vendored `tinyexr`.
- Actual `MTLDevice`: Apple M4 Pro, 262,144 samples, threadgroups 32/64/128, fast and safe math. There are zero NaN,
  Inf, negative PDF, zero PDF, sample/eval mismatch, actual-measure mismatch, affine sample/intersection mismatch,
  positive-bin support loss, or zero-bin selections. The available renderer selected its Metal 4 path. Both Metal 3
  and Metal 4 host bindings compile, but the codebase has no runtime switch to force the automatic Metal 3 fallback.
- Actual production render: 96x96 at 32,768 spp with textured environment, finite distant, dome, and emissive mesh;
  NEE on/off means differ by `0.003350%`, with no negative or non-finite pixels.
- OptiX: not compiled or executed locally. Run on a configured NVIDIA host:

  ```sh
  OPTIX_DIR=/path/to/OptiX-SDK-9.1.0 bash -lc 'tools/ci/linux_check.sh; s=$?; printf "{\"schema\":\"strelka.optix-correctness.v1\",\"exit_code\":%d,\"status\":\"%s\"}\\n" "$s" "$([ "$s" -eq 0 ] && echo PASS || echo FAIL)"; exit "$s"'
  ```

  This builds both OptiX device modules, rejects empty `.optixir` outputs, runs unit tests, and runs production smoke
  renders while emitting a machine-readable final result.

The reference benchmark's two records named `current_env` are intentionally frozen legacy-mutation inputs and remain
classified as biased. The production exact-measure records and the complete audit pass; those two labels are not a
regression in the current renderer.

## Performance and memory deltas

The million-entry CPU proposal benchmark on Apple M4 Pro (`-O3`, median of five, 16,777,216 draws) changed alias/CDF
construction from 1.06 ms to 8.93 ms and sampling from 923.9 ms to 28.7 ms. GPU analytic-light distribution storage
remains 8 MiB and `UniformLight` remains 128 bytes. Peak isolated host RSS during construction rises from 18.3 MiB to
62.4 MiB because the linear-time builder uses temporary double arrays.

The environment alias entry grows from 8 to 12 bytes so its PDF is the one represented by the uploaded float alias
table. A 65,536-texel table grows from 0.5 MiB to 0.75 MiB. PDF evaluation no longer performs the extra point-texture
fetch. The final Ponytail pass removed 21 production/test lines net and shrank Metal `Uniforms` from 832 to 816
bytes. These are representation measurements, not claims of GPU competitiveness.

## Final adversarial and Ponytail review

An independent reviewer inspected normalization, sample/PDF agreement, support, MIS pairing, transformed geometry,
large aliases, and CPU/Metal/OptiX source agreement without editing production code. It found three ordinary-input
counterexamples: finite-distant retry mass collapse, missing primary-camera ownership of the sharp-distant OptiX
atom, and loss of coincident analytic emitter components. They were fixed independently in `1cebdd5`, `a639086`,
and `223233f`; their focused regressions and the final suites pass. The reviewer found no further ordinary-input
counterexample in the environment degeneracies, transformed analytic lights, emissive hierarchy, projector/IES, or
shared sidedness paths.

The full-range Ponytail review then traced every new helper/field to production callers. It removed the dead Metal
`envPdfScale`, a test-only production affine helper, and a private pi macro in `44a2479`; it deliberately retained the
small alias wrapper because the preserved standalone GPU audit consumes it. Production/test code decreased by 21
lines in that cleanup, rather than adding another abstraction layer.

## Remaining risks and blockers

- OptiX compilation and actual NVIDIA execution remain externally blocked. The source uses the shared mathematical
  specification, but that is not equivalent to a successful CUDA/OptiX build or launch.
- Actual Metal execution covered Metal 4. Metal 3 bindings and shaders compile, but the automatic fallback could not
  be forced without adding a production-only diagnostic switch outside this correctness scope.
- Float endpoint rejection in environment and finite-distant samplers is deliberately bounded for GPU progress. A
  last-resort valid-bin sentinel remains for hostile numeric inputs after every retry, although no retained ordinary
  input or exhaustive focused lattice reached it.
- Alias construction deliberately trades preprocessing time and transient CPU memory for guaranteed float-device
  support and O(1) sampling. Further optimization requires profiling and belongs to a later performance stage.
