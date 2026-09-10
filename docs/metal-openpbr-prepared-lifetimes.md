# Metal OpenPBR prepared-state lifetimes

`OpenPBR_PreparedBsdf` is 752 bytes in the packed Metal configuration. Only
40 bytes are public integrator outputs; the remaining 680 bytes form a nested
BSDF lobe tree. The table below classifies every stored field by the longest
lifetime it genuinely requires.

The categories are:

- **material-static**: changes only when resolved material parameters change;
- **per-hit**: depends on the shading frame, view direction, exterior IOR, or
  path throughput and is shared by NEE and continuation sampling;
- **NEE-only**: needed only while evaluating a light connection;
- **sample-only**: needed only while choosing/generating a continuation;
- **transient**: can be produced and consumed inside one operation.

## Top level

| Field | Bytes | Lifetime | Reason |
|---|---:|---|---|
| `volume.extinction_coefficient` | 12 | material-static | Interior-medium output; not read by BSDF eval/pdf/sample. |
| `volume.albedo` | 12 | material-static | Interior-medium output; not read by BSDF eval/pdf/sample. |
| `volume.anisotropy` | 4 | material-static | Interior-medium output; not read by BSDF eval/pdf/sample. |
| `emission` | 12 | per-hit | View-dependent coat/fuzz attenuation; not read by BSDF eval/pdf/sample. |
| `fuzz_lobe` | 668 | expanded below | Nested BSDF state. |
| `view_direction` | 12 | per-hit | Shared by eval, pdf and sample. |

## Base aggregate

`fuzz_lobe.coating_lobe.base_lobe` occupies 484 bytes.

| Field | Bytes | Lifetime | Consumers |
|---|---:|---|---|
| `specular_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `specular_lobe.microfacet_distr.alpha` | 8 | material-static | eval, pdf, sample |
| `specular_lobe.microfacet_distr.basis_ff.{t,b,n}` | 36 | per-hit | eval, pdf, sample |
| `specular_lobe.microfacet_distr.isotropic_alpha` | 4 | material-static | preparation/weight estimation; redundant after weights are made |
| `specular_lobe.refl_trans_coeff.eta_t_over_eta_i_for_transparent_part` | 12 | per-hit | eval/sample Fresnel; includes exterior IOR and dispersion |
| `specular_lobe.refl_trans_coeff.eta_t_over_eta_i_for_opaque_part` | 12 | per-hit | eval/sample Fresnel; includes exterior IOR and dispersion |
| `specular_lobe.refl_trans_coeff.scale_for_reflection_for_transparent_part` | 12 | material-static | eval and sample weight |
| `specular_lobe.refl_trans_coeff.scale_for_reflection_for_opaque_part` | 12 | material-static | eval and sample weight |
| `specular_lobe.refl_trans_coeff.transmission` | 12 | material-static | transmission eval and sample |
| `specular_lobe.refl_trans_coeff.f0_for_metal` | 12 | material-static | reflection eval and sample |
| `specular_lobe.refl_trans_coeff.f82_tint_for_metal` | 12 | material-static | reflection eval and sample |
| `specular_lobe.refl_trans_coeff.metal_amount` | 4 | material-static | reflection eval and sample |
| `specular_lobe.refl_trans_coeff.thin_film_weight` | 4 | material-static | reflection eval and sample |
| `specular_lobe.refl_trans_coeff.thin_film_thickness_nm` | 4 | material-static | reflection eval and sample |
| `specular_lobe.refl_trans_coeff.thin_film_exterior_ior` | 4 | per-hit | thin-film Fresnel; depends on the current side |
| `specular_lobe.refl_trans_coeff.thin_film_ior` | 4 | material-static | thin-film Fresnel |
| `specular_lobe.refl_trans_coeff.thin_film_interior_ior` | 12 | per-hit | thin-film Fresnel with dispersion/current side |
| `specular_lobe.refl_trans_coeff.rgb_wavelengths_nm` | 12 | material-static | dispersion/thin-film evaluation |
| `specular_lobe.refl_trans_coeff.thin_wall_constant_reflection_albedo` | 12 | per-hit | thin-wall view-dependent reflection |
| `specular_lobe.eta_t_over_eta_i` | 12 | per-hit | transmission eval, pdf and sample |
| `specular_lobe.path_throughput` | 12 | per-hit | reflection/transmission probabilities in pdf and sample |
| `dielectric_mms_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `dielectric_mms_lobe.alpha` | 4 | material-static | eval, pdf, sample |
| `dielectric_mms_lobe.eta_t_over_eta_i` | 4 | per-hit | eval, pdf, sample |
| `dielectric_mms_lobe.scale_refl` | 12 | material-static | eval and sample weight |
| `dielectric_mms_lobe.scale_trans` | 12 | material-static | eval and sample weight |
| `dielectric_mms_lobe.reflection_ratio_ti` | 4 | per-hit | pdf and sample |
| `dielectric_mms_lobe.energy_complement_ti_idotn` | 4 | per-hit | eval and sample weight |
| `metal_mms_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `metal_mms_lobe.alpha` | 4 | material-static | eval, pdf, sample |
| `metal_mms_lobe.scale` | 12 | material-static | eval and sample weight |
| `metal_mms_lobe.energy_complement_idotn` | 4 | per-hit | eval and sample weight |
| `diffuse_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `diffuse_lobe.diffuse_albedo` | 12 | material-static | eval and sample weight |
| `diffuse_lobe.diffuse_roughness` | 4 | material-static | eval and sample weight |
| `diffuse_lobe.specular_alpha` | 4 | material-static | energy compensation in eval/sample |
| `diffuse_lobe.specular_eta_t_over_eta_i` | 4 | per-hit | energy compensation; depends on current side |
| `diffuse_lobe.cached_specular_energy_compensation` | 4 | per-hit | view-dependent eval/sample cache |
| `thin_wall_specular_trans_lobe.flipped_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `thin_wall_specular_trans_lobe.flipped_lobe.microfacet_distr.alpha` | 8 | material-static | eval, pdf, sample |
| `thin_wall_specular_trans_lobe.flipped_lobe.microfacet_distr.basis_ff.{t,b,n}` | 36 | per-hit | eval, pdf, sample |
| `thin_wall_specular_trans_lobe.flipped_lobe.microfacet_distr.isotropic_alpha` | 4 | material-static | preparation only after weights are made |
| `thin_wall_specular_trans_lobe.flipped_lobe.refl_trans_coeff.color` | 12 | material-static | eval and sample weight |
| `thin_wall_diffuse_trans_lobe.flipped_lobe.*` | 40 | same as `diffuse_lobe` | Same fields and lifetimes, evaluated in the flipped hemisphere. |
| `lobe_weights[specular]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |
| `lobe_weights[dielectric_mms]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |
| `lobe_weights[metal_mms]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |
| `lobe_weights[diffuse]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |
| `lobe_weights[thin_wall_specular_trans]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |
| `lobe_weights[thin_wall_diffuse_trans]` | 4 | per-hit | NEE pdf and sample selection; path-throughput dependent |

## Coat and fuzz wrappers

| Field | Bytes | Lifetime | Consumers |
|---|---:|---|---|
| `coating_lobe.normal_ff` | 12 | per-hit | coat side tests in eval/pdf/sample |
| `coating_lobe.tint` | 12 | material-static | base-layer attenuation |
| `coating_lobe.presence` | 4 | material-static | coat mixture weight |
| `coating_lobe.inside` | 1 (+ padding) | per-hit | selects inside/outside behavior |
| `coating_lobe.coat_reflection_lobe.normal_ff` | 12 | per-hit | eval, pdf, sample |
| `coating_lobe.coat_reflection_lobe.microfacet_distr.alpha` | 8 | material-static | eval, pdf, sample |
| `coating_lobe.coat_reflection_lobe.microfacet_distr.basis_ff.{t,b,n}` | 36 | per-hit | eval, pdf, sample |
| `coating_lobe.coat_reflection_lobe.microfacet_distr.isotropic_alpha` | 4 | material-static | coat probability preparation |
| `coating_lobe.coat_reflection_lobe.refl_trans_coeff.eta_t_over_eta_i` | 4 | per-hit | coat Fresnel/current side |
| `coating_lobe.in_reflected` | 4 | per-hit | sample probability cache |
| `coating_lobe.in_base_layer_scale` | 12 | per-hit | eval/sample/emission attenuation cache |
| `fuzz_lobe.alpha` | 4 | material-static | eval, pdf, sample |
| `fuzz_lobe.tint` | 12 | material-static | eval and sample weight |
| `fuzz_lobe.presence` | 4 | material-static | fuzz mixture weight |
| `fuzz_lobe.basis.{t,b,n}` | 36 | per-hit | eval, pdf, sample |
| `fuzz_lobe.view_dir_local` | 12 | per-hit | eval, pdf, sample |
| `fuzz_lobe.view_reflected` | 4 | per-hit | sample probability cache |

## Operation-only values

No stored `OpenPBR_PreparedBsdf` field is genuinely NEE-only. NEE's light
direction, half vector, cosine products, Fresnel result, lobe values, weighted
PDF sum, and MIS result are transient intermediates. Sampling likewise keeps
its random remap, selected-lobe index/weight, sampled direction/type, individual
lobe PDF/value, and aggregate value only inside `sample()`.

The important overlap is that lobe weights and `path_throughput` are not
sample-only: OpenPBR's NEE PDF uses the same throughput-dependent mixture as
sampling. This is why rebuilding the full tree independently for NEE and sample
regressed the measured frame by about 4.7%.

## Metal base-queue refactor

The producer guarantees that shade bucket 0 contains no coat, fuzz, thin film,
transmission or subsurface materials. Its function constants also disable
translucency and sheen/coat code. `OpenPBR_BasePreparedBsdf` therefore retains
only the active specular, metal multiple-scattering and diffuse lobes, their
three mixture weights, and the view direction. It is 328 bytes rather than
752 bytes. Full preparation still happens once, but the full tree is immediately
compacted and dies; only the 328-byte state crosses NEE and continuation
sampling. `eval`, `pdf` and `sample` create their half vectors, Fresnel values,
weighted sums and selection state locally.

Layer and tail queues intentionally keep the upstream representation full tree: layer
materials require the wrapper caches and tail materials may activate every
lobe. Removing fields there would either duplicate OpenPBR initialization or
change its sampling distribution.
