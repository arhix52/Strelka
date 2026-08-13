# Open defects

Each entry is something measured and left unfixed, with the measurement that
found it and what has already been ruled out. They are written so that a reader
starting cold can act without repeating the elimination.

Most of what used to live here was found against the Isometric Bathroom /
Kids Bedroom V-Ray conversions. Anything that needs Chaos's closed formula or a
controlled V-Ray render to close -- `adv_base`, rect-light `directional`,
per-light diffuse/specular weights, `TexMulti` with empty slots, `BRDFCarPaint2`
flakes -- is gone from this file. Fitting those to one frame is a guess wearing
the clothes of a conversion; without V-Ray there is nothing to measure against.
The converter still drops them, and that is fine.

What remains is either a renderer defect that Cycles (or a Strelka A/B) can
answer, or an asset/converter note that does not need Chaos.

## Order of attack

| # | What | Where it lives | How we know it is done |
|---|---|---|---|
| ~~1~~ | ~~Clearcoat underside bounce~~ | done — see Closed | `15_clearcoat` 1.002 overall, IOR 2.2 within 1% |
| ~~2~~ | ~~Rough thin-wall blur~~ | done — see Closed | `22_thin_walled` 0.094 / 0.966 CLOSE |
| ~~3~~ | ~~Hair lobe (geometry is done)~~ | done — see Closed | `28_hair` 0.084 / 0.977 CLOSE |
| 4 | Height-map mip aliasing | `render/pt/textureLodMode` | kids bedroom walls A/B lod 0 vs footprint; measure with `patch_mean.py` |
| 5 | OptiX ior-stack counters | OptiX push/pop path | same three counters Metal already reports; no bathroom patch change expected |
| 6 | Two-sided different back face | material model / glTF | design first; one kids-bedroom material only |
| 7 | Bath water is a dish, not a volume | asset, not code | remodel in Blender; bath water R/G against Chaos PNG is a check, not a driver |

4–6 are smaller. 7 is not a renderer bug.

Build the bathroom / kids bedroom when an entry asks for it:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom \
    --rebuild-rug --no-fog-volumes

cd build/Release
./StrelkaCLI ../../scenes/iso_bathroom/iso_bathroom.gltf -o /tmp/iso.png \
    -w 1024 --height 1024 --spp 512 --depth 16 --camera 0 --clamp 8 --tonemap aces
```

Kids bedroom:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Kids_Bedroom/Iso_Kids_Room.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out /tmp/kids --name kids_room \
    --no-fog-volumes
```

Patch means (linear luminance over a rectangle):

```bash
python3 tools/iso_bathroom/patch_mean.py /tmp/kids_ref.png /tmp/kids.png \
    -- 920 380 30 30 right_wall  660 260 40 60 curtain
```

Feature-test ladder (the default verifier for 1–3):

```bash
# see tools/feature_tests/README.md -- Cycles EXR refs, one feature per scene
```

---

## 4. Baked height normals alias at mip 0

Fixed earlier, and it was the largest thing wrong with the kids bedroom: V-Ray
`bump_type` 0 is a height field, and the converter wired it into `normalTexture`
regardless. Seven of nine bump maps were height fields; the walls' was the
greyscale mix mask. A dark mask texel became a shading normal into the surface,
and the plaster walls measured 0.0000 against the reference. Conversion is
`TextureBaker.bake_height_normal`.

**Left open**: those baked maps are sampled at mip 0 by default
(`render/pt/textureLodMode` 0). A mask used as a height field has one-texel
edges, so the sparse 45-degree tilts at those edges alias. Nothing here yet
measures what that costs.

**Fix / measure**: A/B `textureLodMode` 0 vs ray-footprint LOD on the kids
bedroom walls; decide whether the default should change. **Verify**:
`patch_mean.py` on right/back wall; no Chaos required.

---

## 5. Nested dielectrics: Metal counts losses, OptiX does not

**Half fixed on Metal.** `ior_stack_pop` matches on the material being left.
`tests/material/test_ior_stack.cpp` pins it. Three counters report pushes onto a
full stack, unmatched pops, and paths that reach the environment still inside a
medium. Per sample at 1024² / depth 16 on the bathroom: 0 / 7 / 2 with holes
capped, 0 / 19 / 2 without. Residual is a handful of paths in a million; capping
holes moved no patch by more than 0.003.

**Still open**: the OptiX path calls push and pop without asking either question,
so the measurement exists on Metal only.

What cannot be fixed in the renderer: a ray leaving an open mesh goes out through
a hole with no exit event. The converter caps flat, small loops (`Brush_Fibers`);
it correctly leaves foam and water open -- capping the bath water is
arithmetically the best patch mean and plainly wrong on screen, because the water
is a 2 mm dish. See entry 7.

**Fix**: wire the same three counters into OptiX. **Verify**: counters non-zero
on a known-open mesh; bathroom patch means unchanged.

---

## 6. One two-sided material has a different back face

Eight of nine `Mtl2Sided` materials in the kids bedroom are translucency only,
which `KHR_materials_diffuse_transmission` carries. Exactly one has a Back
sub-material linked. The renderer has nowhere to put a genuinely different back
shader; it is reported rather than approximated.

Lampshade / curtain residuals after the translucency fix are dominated by lighting
differences that used to be filed under V-Ray `directional` (removed from this
file), not by this entry.

**Fix**: design a back-face material slot (or reject and document). **Verify**:
that one kids-bedroom material, once a slot exists.

---

## 7. The bath looks full because the stack is wrong

Not a renderer bug. The bath water mesh is a 2 mm dish; what makes it look full
is an unbalanced ior stack applying water absorption to every segment after an
entry through the open surface. Repair the mesh and the accident goes with it.
Needs a real water volume upstream in the .blend.

**Verify after remodel**: bath water R/G via `bubble_profile.py` / patch means;
nested-dielectric counters should drop on that mesh.

---

## Closed (kept for the measurement, not the work)

### Hair Chiang lobe

**Fixed.** Curves used to shade as a rough dielectric cylinder at IOR 1.55 --
silhouette right, lobe dark. `MATERIAL_TYPE_HAIR` now runs Chiang et al. 2016
(R / TT / TRT / TRRT+) on the curve shade path, ported from Cycles' Principled
Hair. `STRELKA_materials_hair` marks the material; pigment colour is Direct
Coloring reflectance, roughness is longitudinal, `radialRoughness` / `coat` ride
in the extension. Non-hair scenes stay on the triangle kernels
(`kFeatureCurves`).

`scenes/feature_tests/28_hair`: short groom vs bald control, 0.084 / 0.977 CLOSE.
Kids-bedroom monster/spider still want a re-export so the converter writes the
extension; the lobe is what those patches were waiting for.

### Rough thin-wall blur

**Fixed.** Smooth thin walls stay a delta at `-V`. Rough ones are a GGX
reflection of the view mirrored through the surface, with Kulla–Conty
transmission roughness (`α √(3.4 (η-1)(η-0.5)²/η³)`), matching Cycles /
OpenPBR. Fresnel for the reflect/transmit split is taken at the shading
normal once, the same way Cycles bakes weights into its two closures; a
microfacet F made the coin flip track the reflection distribution instead.

`scenes/feature_tests/22_thin_walled` is now a roughness ramp (0 → 0.45) plus a
solid control, with a striped card behind so the blur is visible. Measured
0.094 / 0.966 CLOSE. Above ~0.45 Cycles' multiscatter GGX and our single-
scatter disagree on energy more than on blur, so the ramp stops there. Smooth
(roughness 0) is unchanged -- soap bubbles stay bit-identical.

### Clearcoat underside bounce

**Fixed.** `clearcoat_base_scale` now returns the single-scatter
`(1-F_L)(1-F_V)` times the per-channel geometric series against
`½(F_L+F_V)`, capped at `1-F_ms` so a base whose albedo we have under-counted
(specular sits under the coat too) cannot climb past one.

The two earlier attempts failed for named reasons: the *internal* hemispherical
average (~0.6 with TIR) is the wrong Fresnel for a model that never refracts L
and V into the coat -- it put a white ceramic at 2.43 directional albedo; the
*external* hemispherical average is larger than F at normal incidence, so
dividing `(1-F0)²` by `(1-F_avg ρ)` put the same ceramic at 1.02. Using the
directional pair keeps the identity `F + (1-F)² ρ / (1-F ρ) ≤ 1` where
`F_L ≈ F_V`.

`scenes/feature_tests/15_clearcoat`: overall ratio 0.995 → 1.002; the IOR 2.2
end of the ramp, which was the 6% hole, is within 1% of Cycles. Grazing
directions where an *uncoated* dielectric already exceeds 1 (additive
diffuse + specular floor) are excluded from the absolute bound in
`test_clearcoat.cpp`; the coat is required not to make that overshoot worse.

### RIS vs NEE

**Does not reproduce.** An earlier cross-estimator disagreement (NEE 4096 vs RIS
4096 at ~0.049) re-measured at 0.0126 -- five times smaller than the noise either
estimator has left at 1024 spp. Likely closed by the thin-wall `eval` fix
(`93421cf`). `19_env_and_light` and a bathroom lighting ablation both exonerate
the env/analytic split. RIS costs ~40% more time per sample; default stays at 1
for that reason, not because the answer moves.

### Denoiser

**Fixed.** Headless `denoise = true` now writes the denoised texture. Floor at
~0.105 rel RMSE on the bathroom; past ~200 spp plain accumulation wins.
`guide_primary_hit` stays a switch (bathroom +6% at 16 spp, mirror scene −110%).
`20_mirror_and_floor` is the rung. `denoise_firefly_clamp` is a config key.
