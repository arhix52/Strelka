# Open defects

Each entry is something measured and left unfixed, with the measurement that
found it and what has already been ruled out. They are written so that a reader
starting cold can act without repeating the elimination.

Everything below reproduces from the Isometric Bathroom conversion. Build the
scene once:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom \
    --rebuild-rug --no-fog-volumes

cd build/Release
./StrelkaCLI ../../scenes/iso_bathroom/iso_bathroom.gltf -o /tmp/iso.png \
    -w 1024 --height 1024 --spp 512 --depth 16 --camera 0 --clamp 8 --tonemap aces
```

The reference render is
<https://documentation.chaos.com/download/attachments/117637916/Iso_Bathroom_GPU.png>.
Crop the same region from both with `sips`, which does not touch the bytes:

```bash
sips -c 150 190 --cropOffset 330 300 /tmp/iso.png --out /tmp/crop.png
```

`tools/iso_bathroom/bubble_profile.py` reads back a rendered PNG as mean
luminance per radial bin across one of the bubbles, against the wall just
outside it. A defect with a radius is far easier to name than one with only a
colour.

---

## 1. The foam clusters are open transmissive meshes

The two foam clusters in the bathtub carry 126 and 70 boundary edges between
them. An open transmissive mesh unbalances the IOR stack by construction — every
entry without a matching exit leaves the path believing it is inside glass, and
from there every exit it does find is read as an exit from a medium it never
entered.

Nothing in the current render has been traced to this, which is why it is here
rather than fixed: it is a measured property of the asset with a known
consequence, and no measurement yet says which pixels it costs. Closing the
meshes in the converter, or giving the material `thin_walled`, are both cheaper
than teaching the stack to recover.

---

## 2. A rough thin-walled surface transmits as a delta but is weighted as glossy

`standard_pbr_sample` sends thin-walled transmission straight through — `wi` is
exactly `-wo` — at every roughness, because a thin wall has no interior to
refract across. The pdf and the event type do not agree with that. Measured over
50k samples per roughness, on a thin-walled dielectric at IOR 1.6:

| Roughness | max distance from `wi` to `-wo` | mean returned pdf | event |
|---|---|---|---|
| 0.0 | 0 | 0.945 | `SPECULAR_TRANSMISSION` |
| 0.1 | 0 | 48.9 | `GLOSSY_TRANSMISSION` |
| 0.3 | 0 | 0.619 | `GLOSSY_TRANSMISSION` |
| 0.6 | 0 | 0.045 | `GLOSSY_TRANSMISSION` |

So a frosted thin sheet passes light as a perfect mirror-through while telling
MIS it sampled a spread lobe, and `standard_pbr_eval` compounds it: it builds
the half vector as `normalize(V + eta * wi)`, a refraction that never happened,
and evaluates a BTDF over directions the sampler cannot produce. A light seen
through such a sheet is therefore weighted against a density that describes a
different surface.

The soap bubbles are at roughness 0 and are not affected — the smooth row above
is self-consistent. What this costs has not been measured because no scene in
the tree has a rough thin-walled material; `tools/feature_tests/` would need a
new rung before the fix could be checked against Cycles, and inventing the
weighting without that is how the clearcoat term below got rejected twice.

---

## 3. Crossing a medium boundary costs the water its colour

**Symptom.** With the bathtub's `EnvironmentFog` gizmo present, the bath water
loses its cyan. Measured as the red-to-green ratio of the water against the
reference's 0.871:

| Configuration | R/G |
|---|---|
| gizmo present | 0.975 |
| density cut fourfold | 0.971 |
| density ~0 and no emission | 0.971 |
| gizmo removed | 0.734 |

Four orders of magnitude of density move it by 0.004; removing the boundary moves
it by 0.24. It is the crossing, not the medium.

One cause was found and fixed: the crossing branch in
`src/shaders/metal/wavefront.metal` returned before the block that attenuates
over the segment just travelled, so a ray leaving the water through the gizmo
lost the water's absorption. Worth 0.011 of the 0.24. The rest is unexplained.

`tools/iso_bathroom/vray2strelka.py --no-fog-volumes` is the lever meanwhile, and
is what the current export uses.

---

## 4. The clearcoat does not return what bounces under it

`scenes/feature_tests/15_clearcoat` runs 6% dark against Cycles at the strong end
of its IOR ramp and matches exactly at IOR 1.0 — the signature of a missing term
scaling with the coat's reflectance. Cycles models the light that goes through
the coat, off the base, and back down off the coat's underside.

Two formulations were tried and both produced a material brighter than the light
falling on it, measured by `tests/material/test_clearcoat.cpp`: summed against
the coat's internal hemispherical reflectance a glazed white ceramic reached 2.43
directional albedo, and against the external average 1.02. The round trip carries
a 1/eta² radiance compression that does not separate cleanly from the reflectance
when what sits under the coat is a full BSDF rather than a Lambertian.

A documented 6% beats an energy violation, so it stays out until it can be
derived rather than fitted.

---

## 5. `18_bounded_volume` still fails at 1.215

Shadow rays now attenuate through a bounded medium, which took the row from 1.899
to 1.215. What is left is the medium not shadowing itself as strongly as Cycles',
and the emission convention: Cycles adds volume emission with its own coefficient
and Strelka adds it per free-flight event. The scene sets emission to zero rather
than measure the mismatch as though it were an error, so the residual is
scattering alone.

---

## 6. V-Ray colour correction drops `adv_base`

`tools/iso_bathroom/vray2strelka.py`'s `bake_color_correction` implements
brightness, contrast, the advanced lightness curve and the hue tint, but not
`adv_base`. The wood materials set it to 2.0, and without it the contrast of 2.4
pivots around 0.5 and crushes the dark end to black — which is what makes the
window frame read as too dark and too contrasty against the reference.

V-Ray's formula for the advanced lightness mode is not documented anywhere this
conversion could check, and fitting one by eye would be a guess wearing the
clothes of a conversion.
