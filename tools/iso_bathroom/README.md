# Isometric Bathroom — feature gap analysis

Chaos' V-Ray sample scene ["Isometric Bathroom"][scene] as a Strelka test case. It was chosen
because it is a dense, deliberately hard interior: an orthographic camera, 70 materials over 118
objects, glass and water everywhere, cloth, foliage, and two bounded fog volumes.

[scene]: https://documentation.chaos.com/space/VBLD/117637916/Sample+Scenes
Reference render: `Iso_Bathroom_GPU.png` (1024×1024) from the same page.

## Exporting

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b \
    ~/Isometric_Bathroom_Scene/Iso_Bathroom.blend --factory-startup \
    -P tools/iso_bathroom/vray2strelka.py -- --out scenes/iso_bathroom

cd build/Release
./StrelkaCLI ../../scenes/iso_bathroom/iso_bathroom.gltf \
    -o iso.png -w 1024 --height 1024 --spp 512 --depth 16 --tonemap aces
```

The V-Ray for Blender addon is not installed and does not run on Apple silicon, so every V-Ray node
loads as `NodeUndefined`: the sockets are dead, but the plugin parameters survive as IDProperties
(`node['BRDFVRayMtl']`, `node['BRDFSSS2Complex']`, `node['BitmapBuffer']`, …). `vray2strelka.py`
reads those, rebuilds each material as a Principled BSDF, bakes the node effects glTF cannot carry
(colour correction, glossiness→roughness inversion, the procedural noise bump, the gradient ramp),
and injects the extensions Blender's exporter drops (`KHR_materials_sheen`, `_iridescence`,
`_volume`, and a `STRELKA_materials_subsurface` placeholder) into the `.gltf` afterwards.

Two things it must get right, because getting them wrong fails silently:

- **Copy the IDProperties out before clearing the node tree.** Removing a node frees its
  IDProperties and every later read returns the plugin default — which exported white glass and no
  sheen at all until it was fixed.
- **A missing IDProperty key is the plugin default, not zero.** V-Ray only stores what differs from
  the default, so `VRAYMTL_DEFAULTS` in the script is load-bearing.

## What the scene needs

| | Feature | Where it shows | Status |
|---|---|---|---|
| 1 | Orthographic camera | the whole framing | **done** |
| 2 | glTF camera index desync | any scene with an ortho camera | **fixed** |
| 3 | Sheen | 9 cloth materials — towels, rugs | **done** |
| 4 | Subsurface scattering | 9 materials — cactus, marble, soap, duck, sponge | **done** |
| 5 | Thin-film iridescence | `Bubbles_Mtl` | **done** |
| 5b | Thin-walled refraction | the bubbles on the bath water | **fixed** |
| 6 | Clearcoat with IOR | 4 ceramics, `coat_ior` up to 2.0 | **done** |
| 7 | Specular colour tint | every dielectric with a tinted reflection | **done** |
| 8 | Texture wrap modes | marble worktop (2×2), plant ramp (50×50) | **done** |
| 9 | Percent-decoded image URIs | any texture with a space in the name | **fixed** |
| 10 | Per-slot `KHR_texture_transform` | wood materials | partial |
| 11 | Radiance clamp | fireflies over the whole frame | **done** (`--clamp`) |
| 12 | Mesh-emitter NEE | `Light_Plane` | worked around in the exporter |
| 13 | One-sided mesh emitters | `emitOnBackSide = 0` | absent |
| 14 | Per-light "invisible to camera" | `VRayRectLight_Window` has `invisible = 1` | **done** |
| 15 | Rect-light spread | `directional` 0.1 / 0.5 | absent |
| 16 | Bounded volumetrics | bath water + shower spray glow | **done** |
| 17 | `.vrmesh` proxy | the round rug | preview mesh only |

## What landed

Rendered at 1024x1024, 2048 spp, `--depth 16 --clamp 8 --camera 0`:

- **Orthographic camera** -- `Camera::setOrthographic` / `magForAspect`
  (`src/scene/src/camera.cpp`), the glTF `orthographic` branch in `loadCameras`,
  `Uniforms::projectionType` plus the film half-extents, and the branch in
  `generateCameraRay` (`src/shaders/metal/shading_common.h`). An orthographic
  camera has no centre of projection, so the pixel moves the ray's origin and not
  its direction -- that is a branch in ray generation, not a different matrix.
  The CPU pick ray branches identically, since a pick that maps pixels
  differently than the renderer selects something other than what was clicked.
- **Camera index map** -- `loadCameras` returns glTF-index to scene-index and
  `processNode` goes through it instead of writing past the end of a vector.
- **Percent-decoded image URIs** via `tinygltf::URIDecode`.
- **Indirect radiance clamp** -- `render/pt/clampIndirect`, CLI `--clamp`, TOML
  `render.clamp_indirect`; applied to shadow-ray weights where the ray is built
  and to the emission and environment contributions past depth 0. Off by default:
  it is a bias, and the feature tests measure an unbiased estimator.
- **Sheen** -- Charlie distribution and Ashikhmin visibility (`microfacet.h`), the
  lobe in `standard_pbr.h`, `KHR_materials_sheen` in the loader, and the
  directional-albedo table in `sheen_albedo_lut.h` from
  `tools/material/gen_sheen_albedo_lut.py`. Sheen rides the diffuse lobe's cosine
  sampling rather than getting its own, which is what keeps the pdf a single
  cosine term in every branch.
- **Subsurface scattering** -- a bounded scattering medium entered through the
  diffuse-transmission lobe and left by a random walk: free flight in `extend`,
  Henyey-Greenstein scattering in `shade`, and next-event estimation at the exit
  vertex (`src/shaders/metal/subsurface.h`). Gated by a `kFeatureSubsurface`
  function constant, so a scene without a translucent material compiles the
  kernels it compiled before. A cold side table carries two words -- which
  medium and how far into the walk, plus the textured entry albedo -- rather
  than widening the `PathState` every stage streams or copying the medium's
  parameters, which would have been 28 MB at 1024x1024 to avoid a load from a
  table that fits in cache.
- **Texture wrap** -- material samplers are `address::repeat`. glTF's default
  wrap is REPEAT and Metal's is clamp_to_edge; under clamping a tiled texture
  smears its edge texel across the whole surface, which reads as a texture that
  failed to load rather than as a wrap-mode bug.
- **Bounded volumetrics** -- a participating medium bounded by the geometry
  carrying its material, which is what a V-Ray `EnvironmentFog` gizmo is. It
  shares the machinery the subsurface walk established: same free flight in
  `extend`, same Henyey-Greenstein scattering, same one-word medium slot on the
  path. What it adds is a boundary crossing that toggles the medium and carries
  the ray on unshaded, volumetric emission, and next-event estimation at the
  scattering vertex -- a bounded volume is thin and lit from outside, so the
  glow and the shafts are single scattering, which is exactly what a subsurface
  walk cannot afford. The boundary geometry sits on its own ray mask
  (`GEOMETRY_MASK_MEDIUM`) so shadow rays pass through it; on the triangle mask a
  gizmo would black out everything it encloses.
- **Thin-walled refraction** -- `thin_walled` was hardcoded to 0 in the loader,
  so the bubbles floating on the bath water rendered as dark specks: a solid
  sphere of IOR 1.6 refracts into itself and the path dies before it gets out.
  The loader now reads `KHR_materials_volume.thicknessFactor`. It does *not*
  follow the spec's reading that a transmissive material without the extension is
  thin-walled -- Blender writes that extension only for a specific node setup, so
  the literal reading turns every ordinary glass export into a bubble. Absence is
  read as solid and thin-walledness has to be stated, which the exporter now does
  on both sides.
- **Clearcoat IOR** -- `clearcoat_ior` on the material, read from a
  `clearcoatIor` field the exporter adds to `KHR_materials_clearcoat` (the
  extension has none, and fixes the coat at a clear lacquer). The ceramics here
  are authored at 2.0, an F0 of 0.111 against 0.04. The coat also stopped being
  additive: what it reflects is now taken out of the base, once for the way in
  and once for the way out.
- **Thin-film iridescence** -- Belcour & Barla's Airy summation in
  `src/material/include/strelka/material/iridescence.h`, in the form
  `KHR_materials_iridescence` is specified against, replacing the specular
  Fresnel. Evaluated at `VdotH` rather than at `NdotV` as the glTF reference
  does: inside a microfacet BRDF the angle the Fresnel is taken at is the
  microfacet's, and using anything else makes the film disagree with the lobe it
  is modifying.
- **Specular colour** -- `KHR_materials_specular.specularColorFactor`. The
  material carried a scalar `specular_tint` that the loader hardcoded to zero, so
  nothing ever exercised it; it was also a different parameter wearing the same
  name -- a Disney-style weight blending F0 toward the base colour, where the
  extension is an independent multiplier. It is now a `float3 specular_color`.
  Nothing in this scene changes: every V-Ray reflection colour here is grey.
- **Per-light `visibleToCamera`** -- a second mask bit
  (`GEOMETRY_MASK_LIGHT_HIDDEN`) and a per-dispatch ray mask on `extend`, chosen
  by bounce. Per dispatch and not per ray: the only thing it distinguishes is the
  camera bounce, and `extend` is encoded once per bounce anyway, so reading the
  path depth there would put a load in the hottest kernel in the renderer.

Several of these were caught by a test rather than by eye, and each was wrong in
a way that looked plausible:

- **Sheen was manufacturing energy.** Ashikhmin's visibility term does not
  conserve it -- the measured directional albedo peaks at 2.78 -- so an additive
  layer put a plain white cloth at 1.40 directional albedo, a towel brighter than
  the light falling on it. The lobe is now normalised by its own albedo and the
  base scaled by what the layer took. `tests/material/test_sheen.cpp` pins both,
  along with the property the lobe exists for: sheen leaves the head-on response
  where it was and more than doubles the grazing one.
- **The clearcoat was manufacturing energy too**, for the same reason and with
  the same fix, but it took two goes: scaling the base by the view-side Fresnel
  alone -- which is what the glTF sample viewer does -- still left a glazed white
  ceramic at 1.06 directional albedo. Light crosses the coat twice, so the base
  is scaled twice. `tests/material/test_clearcoat.cpp` measures it.
- **A zero-thickness film was not no film.** The extension fades the film's IOR
  to the outside medium's as the thickness goes to zero, and that alone does not
  get there: with both IORs equal the outer interface reflects nothing, the
  series collapses, and what is left is the first interference fringe scaled by
  the floor `r123` is clamped to -- about 60% of the base reflectance, out of
  nowhere. `iridescence_fresnel` states the degenerate case instead of asking a
  clamped series to recover it.
- **The subsurface albedo was the wrong quantity.** A DCC's subsurface colour is
  a diffuse albedo; the walk wants the probability that one extinction event
  scatters rather than absorbs. Feeding 0.47 straight in and scattering three
  times gives 0.1 -- which is what turned the rubber duck into a dark blob. The
  exporter now inverts Van de Hulst's approximation to get the single-scattering
  albedo, and the extension documents which of the two it carries.

### Evidence

Measured on the way in, and worth keeping because each one settled a question
that guesswork had a plausible wrong answer for:

- The framing, the glass partition, the chrome, the tiles and the wood all read
  correctly once the export was right, so the geometry and the bulk of the
  material conversion were sound before any renderer change.
- **The fireflies were the analytic rect lights, not the environment and not the
  mesh light.** They persisted unchanged with the environment map removed *and*
  with `Light_Plane`'s emission zeroed, which is what pointed at specular and
  glass chains to the two rect lights (radiance 35 and 10) rather than at
  environment sampling. `--clamp 8` removes them.
- The orthographic camera did not merely render wrong, it hung: the accumulator
  reset to 1 spp every frame. See the camera-index bug below -- the symptom and
  the cause look nothing alike.
- `Light_Plane` rendered as a bright quad in front of the wall, and the
  reference does not show it. It is now lifted out of the geometry into an
  analytic rect light marked invisible to the camera, which also gives it
  next-event estimation -- as emissive geometry it was a light Strelka could only
  find by chance, so a 0.2 x 0.2 quad was mostly a noise source. The room came out
  visibly brighter for it.
- **The fog conversion had the multiplier on the wrong side, and the bath water
  hid it.** V-Ray's fog depth is in centimetres and its multiplier scales that
  depth -- a *larger* `fog_mult` is more transparent, not less -- so the glTF
  equivalent is `attenuationDistance = fog_mult x one centimetre`, or
  `fog_mult * 0.01` in a scene modelled at a unit to the metre. The exporter had
  `1.0 / fog_mult`, and at the water's `fog_mult` of 10 both expressions come out
  at 0.1: the one material anyone looks at was right by coincidence while the
  shower glass, at `fog_mult` 1, was a hundred times too weak and showed no green
  at all.

  Caught by measuring hue rather than brightness. Normalised R:G:B on the shower
  panel read 0.75 : 0.93 : 1.00 against the reference's 0.50 : 0.97 : 1.00, and
  no exposure change moves a ratio. After the fix, 0.52 : 0.95 : 1.00.

  Nothing here is fitted -- `--fog-scale` is one centimetre expressed in scene
  units and only changes if the scene does. No material in this scene authors
  V-Ray's "Depth (cm)" at all; they are all on the plugin default, which is what
  the centimetre is.
- The subsurface scale was an order of magnitude too large at first, and the
  symptom was the opposite of the intuition: translucent objects *vanished*
  rather than glowing, because a mean free path longer than the object is a
  medium light passes straight through.
- The bounded volumes were wrong twice, and both times the symptom was global
  rather than local, which is what made them worth chasing rather than tuning
  around:
  - Deciding enter-versus-exit from `dot(rayDir, geomNormal)` inverted one of the
    two volumes, because a fog gizmo's winding is arbitrary -- V-Ray decides
    inside from an inside/outside test and never reads the normal. An inverted
    volume is not a subtle error: it is a medium filling all of space except the
    gizmo. Crossing now toggles the medium, which needs no winding convention.
  - Volumetric emission was added outside the clamp every other contribution goes
    through. Reached through glass the throughput is well above one, and the
    result was fireflies over the entire frame, including the backdrop outside
    the room. That the noise was *outside* the volumes is what identified the
    term rather than the medium: zeroing the emission made it vanish while the
    volumes stayed.
- **The frame reading harder than the reference is not a calibration problem**,
  which was worth settling rather than repeating. `tools/feature_tests/` run
  against Cycles in linear EXR puts `00_calibration` at ratio 1.010 and every
  other row between 0.986 and 1.061 -- light units and exposure agree with another
  path tracer to within a percent, and the 8x difference image is noise with no
  structure in it. What is left between this render and the reference PNG is
  V-Ray's own colour mapping, which is not any of `none|reinhard|aces|filmic`, and
  matching it is a tone-curve fit rather than anything to fix in the renderer.
- V-Ray's *volumetric* fog density -- the `EnvironmentFog` gizmos, not the
  material fog above -- is in units this conversion cannot recover; taken at face
  value it renders the bathtub as a white blob. `--fog-density-scale` is a fit to
  the reference and is the only number in the script flagged as one.
- The bath water's remaining hue difference is not the material fog: its
  attenuation distance was accidentally correct all along. What is left there is
  the volumetric gizmo's emission whitening it.

### The two bugs, precisely

**glTF camera index desync** (fixed). `loadCameras` appended only cameras whose
`type == "perspective"` and silently dropped the rest, while `processNode` wrote
the node transform to `scene.getCamera(node.camera)` -- indexed by the *glTF*
camera index. One skipped camera shifted every later one and the last camera node
wrote past the end of the vector. The symptom was not a crash: it was an
accumulator that reset to 1 spp every frame, because the corrupted projection
matrix compared unequal to itself in the "did the camera move" test at
`MetalRender.mm:2802-2806`. `loadCameras` now returns a glTF-index to
scene-index map and `processNode` goes through it, so an unknown camera type
warns and is skipped without disturbing the others.

**Image URIs were not percent-decoded** (fixed). `getTextureUri` handed
`images[].uri` straight to the file opener, so a texture written as
`Material #449_ramp.png` arrived as `Material%20%23449_ramp.png` and failed to
load. It now goes through `tinygltf::URIDecode`. The exporter also sanitises the
names it generates, which is belt and braces rather than a substitute -- any
third-party glTF with a space in a texture name hit this.

## Open defects

Five things are measured and unfixed, and they live in `docs/open-defects.md`
rather than here: each one is written so a reader starting cold can act without
repeating the elimination. The one to start with is the black rim on the
thin-walled bubbles -- six suspects are already ruled out with the test that
ruled each one out, and the remaining hypothesis says to instrument rather than
substitute.

## Plan

Ordered so that each phase produces a visibly better image than the last, and so the cheap
correctness fixes land before the expensive features.

### Phase 1 -- remaining material correctness

1. **Per-slot `KHR_texture_transform`.** Read per material today, first slot
   wins, applied to all (`gltfloader.cpp`, `readTextureTransform`). The wood
   materials in this scene transform only some of their slots.
2. **Occlusion maps.** Loaded, uploaded, bound, and sampled by nobody on either
   backend.
3. **`anisotropy_rotation`.** Parsed from `KHR_materials_anisotropy` and stored,
   never consumed by a shader.

### Phase 4 -- documenting the two extensions

12. `STRELKA_materials_subsurface` (`subsurfaceFactor`, `scatterColor`,
    `scatterRadius`, `anisotropy`, `ior`) and `STRELKA_materials_medium`
    (`density`, `scatterColor`, `emissionColor`, `anisotropy`) are both written by
    the exporter and read by the loader, and neither is written down anywhere a
    third party could find. They belong in `docs/`, with the one thing about them
    that is easy to get wrong stated plainly: `scatterColor` is the
    *single-scattering* albedo, not the diffuse colour a DCC shows.
13. Keep `KHR_materials_diffuse_transmission` as the graceful fallback it
    currently is -- a renderer without the extension draws a translucent object
    as translucent rather than as matte paint.

### Phase 5 -- the remaining lighting

15. **Mesh-emitter NEE.** Build an emissive-triangle list at scene build time,
    sample it in `connectToLight` (`shading_common.h`), and MIS it against the
    BSDF hit in `shade`. The bathroom scene no longer needs it -- the exporter
    lifts its one light material out into an analytic light -- but any scene that
    lights itself with emissive geometry does. Uniform light selection is fine at
    three lights; adding mesh emitters makes a light BVH worth doing at the same
    time.
16. **One-sided emitters** -- a `double_sided_emission` bit on `MaterialParams`,
    tested where emission is picked up in `shade`. V-Ray's Light Mtl carries
    `emitOnBackSide`.
17. **Rect-light spread** -- V-Ray's `directional` narrows the emission lobe; a
    cosine-power falloff on the rect light sampler in `lights_metal.h`.
18. **Shadow rays through a bounded medium.** They currently ignore its
    extinction, so a volume does not shadow itself. That needs the shadow ray to
    intersect the boundary, i.e. a second traversal -- worth it for a room full of
    haze, not for a bathtub.

### Phase 6 — calibration

19. Extend `tools/feature_tests/` with `14_sheen`, `15_subsurface`, `16_bounded_volume` and
    `17_orthographic` against Cycles, in linear EXR, the same way the existing ladder works. Read
    `00_calibration` first — if it does not sit at ratio ≈1.0 the rest measures exposure, not the
    feature.
20. Sheen and clearcoat are on the ladder now (`14_sheen`, `15_clearcoat`).
    `15_clearcoat` measures a real approximation worth closing: our coat takes
    energy from the base twice, on the way in and on the way out, and never
    returns what bounces between the coat's underside and the base. That is 6%
    at the strongest coat in the ramp.
21. Only then match this scene to the reference PNG: solve for exposure and tonemap, since V-Ray's
    colour mapping is not any of `none|reinhard|aces|filmic`. Everything before this point should
    be validated against Cycles in linear, not against a tonemapped JPEG.

## Known approximations in the export

Recorded in `<name>_materials.json` under `approx`, and worth remembering when reading a diff:

- `TexFalloff` driving the subsurface colour on the cactus and hoya leaves collapses to a flat
  facing colour.
- The gradient ramp on the Sansevieria is baked as the plugin-default black-to-white V-ramp — the
  actual stops are not in the IDProperties.
- The procedural `TexNoiseMax` bump on the water is baked to a 256² tiling normal map.
- V-Ray's `fog_color`/`fog_mult` become `attenuationDistance = --fog-scale / fog_mult`, which is a
  fit, not the V-Ray formula.
- The scatter radius is authored in centimetres against a scene whose room is 0.5 units across;
  `--sss-scale` converts it and the default was chosen by eye.
- `Rug_02_Mtl` / `Rug_03_Mtl` are dropped — they are slots 2 and 3 on the proxy preview mesh, which
  only uses slot 0.
