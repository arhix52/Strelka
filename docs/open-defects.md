# Open defects

Only unresolved, reproducible correctness gaps belong here. Closed investigations
remain in Git history (`git show b720fc5^:docs/open-defects.md`). Performance work
belongs in `docs/open-perf.md`.

## Metal: layered OpenPBR geometry opacity

Native OpenPBR constant `geometry_opacity` and a direct red-channel opacity map
now use the same cutout coverage for camera and shadow rays. The remaining gap
is `OpenPBRLayeredTextureParams::output_mask & OPENPBR_LAYER_OUTPUT_OPACITY`:
its graph can depend on a selected data layer, UV transform and view-facing
term. The compact alpha traversal table cannot evaluate that graph. Metal now
rejects such a scene with the material name rather than silently rendering
wrong coverage. This remains an unsupported feature, not a valid approximation.

To close: evaluate that graph in all cutout paths (including the intersection
function and direct-static shadow path), or bake a matching opacity texture at
load time. Test a layered opacity graph against both primary and shadow rays,
with the glTF/Metal alpha path as the reference for binary coverage.

## Two-sided material with a distinct back face

One of the kids-bedroom `Mtl2Sided` materials links a separate Back material.
Eight others only use translucency and can use
`KHR_materials_diffuse_transmission`. Strelka has no back-face material slot, so
this one asset cannot be represented faithfully. This needs a material/scene
format decision, not a Metal shader-only patch.

## OpenPBR + subsurface reproducibility

The historical witness was the Open Chess Set at 960×540, 512 spp: eight of ten
runs differed from the first in only two pixels, `(329,482)` and `(336,494)`.
It is still live on 2026-09-23: two runs of `chess_set.toml` at that resolution,
512 spp, four spp per launch produced different EXR SHA-256 hashes
(`3df269b8…` and `cd121982…`). The current differing pixel count has not yet
been measured, so do not assume it is still exactly two.
The `25_subsurface` feature scene itself was bit-identical across seven runs;
it must not be labelled nondeterministic. The old ablation found that OpenPBR
and subsurface together were required, but did not localise the race.

Next: capture the first divergent sample and inspect the shared wavefront
state. Until localised, compare this combination statistically; exact-match
grading is valid for the isolated `25_subsurface` scene.
