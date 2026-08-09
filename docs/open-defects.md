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

---

## 1. Thin-walled spheres render with a black rim

**Symptom.** The three soap bubbles floating by the window have a thick dark
ring where the sphere is seen edge-on. The reference has pale bubbles with a
bright thin rim. Crop offset `330 300`, size `150 190`.

**Where it is.** The thin-walled branch of the specular transmission lobe,
`src/material/include/strelka/material/bxdfs/standard_pbr.h:499` onward. The
material reaching it: `transmission 1`, `ior 1.6`, `thin_walled 1`,
`clearcoat 1` at roughness 0, iridescence at 400 nm.

### Ruled out, with the test that ruled it out

| Suspect | Test | Result |
|---|---|---|
| Path depth | `--depth 48` against `--depth 16` | identical, pixel for pixel |
| Clearcoat | removed from the material, and separately set to roughness 0.5 | rim unchanged |
| Thin film | `KHR_materials_iridescence` removed, *after* the film was wired into this lobe | rim unchanged |
| Geometry thickness | bmesh: the three bubbles are closed 386-vertex spheres, 0 open edges | not a shell with thickness |
| Total internal reflection | fixed — thin-walled no longer TIRs | rim unchanged |
| IOR stack imbalance | fixed — thin-walled no longer pushes the stack | rim unchanged |

Patch the glTF directly for these; the material is `Bubbles_Mtl`:

```python
import json
d = json.load(open("scenes/iso_bathroom/iso_bathroom.gltf"))
for m in d["materials"]:
    if m["name"] == "Bubbles_Mtl":
        m["extensions"].pop("KHR_materials_clearcoat", None)
json.dump(d, open("scenes/iso_bathroom/probe.gltf", "w"))
```
and copy `iso_bathroom_light.json` to `probe_light.json` beside it.

### The discriminator that matters

The same spheres render **correctly** two other ways:

- as opaque (drop `KHR_materials_transmission` and `KHR_materials_volume`, set a
  grey base colour) — clean spheres, no rim;
- as **solid** glass (drop only `KHR_materials_volume`, so `thin_walled` reads 0)
  — they look like glass marbles, and the rim is bright.

So it is specific to the thin-walled path, and not to the geometry, the lighting
or the lobe selection.

### Remaining hypothesis, and how to test it

Grazing self-intersection. At the silhouette the Fresnel coin flip takes the
reflection almost always, and a reflection there leaves nearly tangentially. If
`offset_ray` (`src/shaders/metal/shading_common.h:641`) does not clear the
sphere's curvature, the ray re-hits the same surface, reflects again at grazing,
and the throughput bleeds away over many hits. That would also explain the
indifference to depth: at grazing almost nothing is lost per hit, so more bounces
do not brighten it.

The solid case would be exempt because its refracted ray goes inward and away.

Test it by instrumenting rather than substituting: count hits per path on the
bubble instance and write the count to a debug view, or write the path's surface
crossing count into an AOV. `DebugMode` in `src/shaders/metal/ShaderTypes.h`
already has the enumeration and `--config` exposes `render.debug`. If the count
spikes at the rim, the fix is a curvature-aware offset or a shading-normal
reconciliation at grazing (Schüssler et al. 2017 and relatives), not another
change to the lobe.

### Related, and worth its own look

The foam clusters in the bathtub are two open meshes carrying 126 and 70
boundary edges between them. An open transmissive mesh unbalances the IOR stack
by construction — every entry without a matching exit leaves the path believing
it is inside glass. The window bubbles are closed, so this is not what causes
their rim, but it is a real hazard in the same scene.

---

## 2. Crossing a medium boundary costs the water its colour

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

## 3. The clearcoat does not return what bounces under it

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

## 4. `18_bounded_volume` still fails at 1.215

Shadow rays now attenuate through a bounded medium, which took the row from 1.899
to 1.215. What is left is the medium not shadowing itself as strongly as Cycles',
and the emission convention: Cycles adds volume emission with its own coefficient
and Strelka adds it per free-flight event. The scene sets emission to zero rather
than measure the mismatch as though it were an error, so the residual is
scattering alone.

---

## 5. V-Ray colour correction drops `adv_base`

`tools/iso_bathroom/vray2strelka.py`'s `bake_color_correction` implements
brightness, contrast, the advanced lightness curve and the hue tint, but not
`adv_base`. The wood materials set it to 2.0, and without it the contrast of 2.4
pivots around 0.5 and crushes the dark end to black — which is what makes the
window frame read as too dark and too contrasty against the reference.

V-Ray's formula for the advanced lightness mode is not documented anywhere this
conversion could check, and fitting one by eye would be a guess wearing the
clothes of a conversion.
