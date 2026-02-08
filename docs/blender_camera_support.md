# Blender Camera Support in Strelka

## Current State

Strelka loads only basic perspective camera properties from glTF:

| Property | Source | Status |
|---|---|---|
| Vertical FOV (`yfov`) | glTF | Working |
| Near/far clip | glTF | Working |
| Position/rotation | glTF node transform | Working |
| Camera name | glTF | Working |

Ray generation uses inverse matrices (`clipToView`, `viewToWorld`) — a simple pinhole model:

```cpp
// pixel → NDC → clip → view → world
float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;
float4 clip{pixelNDC.x, pixelNDC.y, 1.0f, 1.0f};
float4 viewSpace = clipToView * clip;
float4 wdir = viewToWorld * make_float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);
```

No thin lens model, no lens shift, no sensor dimensions — just a basic pinhole.

## Priority Features

### Tier 1 — High Impact, Moderate Effort

#### 1. Depth of Field (DOF)

The most visually impactful missing feature. Requires switching from pinhole to a
thin lens model in ray generation.

**Blender properties to capture:**

| Property | Blender API | Effect |
|---|---|---|
| Enable DOF | `camera.dof.use_dof` | Toggle |
| Focus distance | `camera.dof.focus_distance` | Focal plane distance |
| Focus object | `camera.dof.focus_object` | Auto-focus on object (resolve to distance at export) |
| F-stop | `camera.dof.aperture_fstop` | Aperture size (controls blur amount) |
| Blades | `camera.dof.aperture_blades` | Bokeh shape (0 = circular, 3+ = polygonal) |
| Blade rotation | `camera.dof.aperture_rotation` | Rotates bokeh shape |
| Anamorphic ratio | `camera.dof.aperture_ratio` | Oval bokeh (1.0 = circular) |

**Renderer changes needed:**

- Add `focalDistance`, `lensRadius`, `blades`, `bladeRotation`, `anamorphicRatio`
  to render params.
- Modify `generateCameraRay()` to jitter ray origin on the lens aperture:

```cpp
// Thin lens DOF
float2 lensSample = sampleAperture(rng, blades, bladeRotation);
lensSample *= lensRadius;
float3 focalPoint = origin + direction * focalDistance;
origin += right * lensSample.x + up * lensSample.y;
direction = normalize(focalPoint - origin);
```

**Lens radius derivation from f-stop:**

```
lensRadius = focalLengthMm / (2.0 * fStop * 1000.0)
```

#### 2. Lens Shift (Tilt-Shift / Perspective Correction)

**Blender properties:**

| Property | Blender API | Effect |
|---|---|---|
| Shift X | `camera.shift_x` | Horizontal offset (-10 to 10, fraction of sensor) |
| Shift Y | `camera.shift_y` | Vertical offset (-10 to 10, fraction of sensor) |

**Renderer changes:**

Offset the center of projection in the perspective matrix. In the ray generator,
this is an offset to `pixelNDC`:

```cpp
pixelNDC.x += shiftX * 2.0f;
pixelNDC.y += shiftY * 2.0f;
```

Or preferably, build the shift into the perspective matrix as an off-center
projection.

#### 3. Sensor Dimensions / Focal Length in mm

Currently only `yfov` is used. Blender artists typically work with focal length
and sensor size. These determine FOV via:

```
fov = 2 * atan(sensorHeight / (2 * focalLength))
```

| Property | Blender API | Effect |
|---|---|---|
| Focal length (mm) | `camera.lens` | Lens focal length |
| Sensor fit | `camera.sensor_fit` | `AUTO` / `HORIZONTAL` / `VERTICAL` |
| Sensor width | `camera.sensor_width` | mm (default 36.0) |
| Sensor height | `camera.sensor_height` | mm (default 24.0) |

These can be resolved to `yfov` at export time, but storing them allows
interactive focal length changes in the editor and is required for correct DOF
lens radius calculation.

### Tier 2 — Nice to Have

#### 4. Camera Motion Blur

Strelka already has `enableCameraMotionBlur` as a setting (currently unused).
Needs two camera transforms (current + previous frame) and interpolation during
ray generation.

| Property | Blender API | Effect |
|---|---|---|
| Shutter time | `scene.render.motion_blur_shutter` | Exposure duration (frames) |
| Shutter position | `scene.render.motion_blur_position` | `START` / `CENTER` / `END` |

#### 5. Orthographic Camera

Strelka's loader explicitly skips orthographic cameras.

| Property | Blender API | Effect |
|---|---|---|
| Ortho scale | `camera.ortho_scale` | Orthographic viewport width |

**Renderer changes:** Rays have parallel directions instead of converging to a
point. Modify `generateCameraRay()` to offset origin instead of direction:

```cpp
origin = cameraPos + right * pixelNDC.x * orthoWidth * 0.5
                   + up    * pixelNDC.y * orthoHeight * 0.5;
direction = cameraForward;
```

### Tier 3 — Advanced / Niche

#### 6. Panoramic Cameras

Equirectangular, fisheye equisolid/equidistant, mirror ball, cubemap face.
Requires completely different ray generation math per sub-type. Useful for VR
and 360 rendering.

#### 7. Stereo Rendering

Two-camera rig with interocular distance and convergence. Needed for VR.

| Property | Blender API | Effect |
|---|---|---|
| Interocular distance | `camera.stereo.interocular_distance` | Eye separation (default 0.065 m) |
| Convergence distance | `camera.stereo.convergence_distance` | Convergence plane (default 1.95 m) |
| Convergence mode | `camera.stereo.convergence_mode` | `OFFAXIS` / `PARALLEL` / `TOE` |

## Export Design

### Camera JSON (`_camera.json`)

Similar to how lights use `_light.json`, camera properties not representable in
glTF are exported to a sidecar JSON file. The loader matches cameras by name.

```json
{
    "cameras": [
        {
            "name": "Splash Cam",
            "type": "perspective",
            "focal_length_mm": 35.0,
            "sensor_width": 36.0,
            "sensor_height": 24.0,
            "sensor_fit": "AUTO",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "clip_near": 0.1,
            "clip_far": 100.0,
            "dof": {
                "enabled": true,
                "focus_distance": 5.2,
                "fstop": 2.8,
                "blades": 5,
                "blade_rotation": 0.0,
                "anamorphic_ratio": 1.0
            }
        }
    ]
}
```

### Export script changes (`blend2strelka.py`)

Extract all camera properties to `<name>_camera.json`:

```python
def extract_camera(cam_obj):
    cam = cam_obj.data
    desc = {
        "name": cam.name,
        "type": cam.type.lower(),             # "persp" / "ortho" / "pano"
        "focal_length_mm": cam.lens,
        "sensor_width": cam.sensor_width,
        "sensor_height": cam.sensor_height,
        "sensor_fit": cam.sensor_fit,         # "AUTO" / "HORIZONTAL" / "VERTICAL"
        "shift_x": cam.shift_x,
        "shift_y": cam.shift_y,
        "clip_near": cam.clip_start,
        "clip_far": cam.clip_end,
    }
    if cam.dof.use_dof:
        focus_dist = cam.dof.focus_distance
        if cam.dof.focus_object:
            focus_dist = (cam_obj.matrix_world.translation
                          - cam.dof.focus_object.matrix_world.translation).length
        desc["dof"] = {
            "enabled": True,
            "focus_distance": focus_dist,
            "fstop": cam.dof.aperture_fstop,
            "blades": cam.dof.aperture_blades,
            "blade_rotation": cam.dof.aperture_rotation,
            "anamorphic_ratio": cam.dof.aperture_ratio,
        }
    return desc
```

### Loader changes (`gltfloader.cpp`)

Add `loadCamerasFromJson()` (analogous to `loadLightsFromJson()`):

1. Look for `<modelname>_camera.json` next to the model file.
2. Parse camera entries and match to already-loaded glTF cameras by name.
3. Augment `Camera` objects with DOF, shift, sensor properties.

## Implementation Changes by File

### `camera.h` — New fields

```cpp
// Depth of field
bool useDof = false;
float focalDistance = 10.0f;
float fStop = 2.8f;
int apertureBlades = 0;       // 0 = circular
float bladeRotation = 0.0f;   // radians
float anamorphicRatio = 1.0f;

// Lens
float focalLengthMm = 50.0f;
float sensorWidth = 36.0f;    // mm
float sensorHeight = 24.0f;   // mm
float shiftX = 0.0f;
float shiftY = 0.0f;
```

### `OptixRenderParams.h` — GPU-side DOF params

```cpp
// DOF
int   useDof;
float focalDistance;
float lensRadius;
int   apertureBlades;
float bladeRotation;
float anamorphicRatio;

// Lens shift
float shiftX;
float shiftY;
```

### `OptixRender.cu` — Ray generation

Modify `generateCameraRay()` to support thin lens DOF and lens shift. See the
pseudocode in the DOF section above.

### `OptixRender.cpp` — Upload params

Compute `lensRadius` from focal length and f-stop, copy DOF/shift params to
GPU.

### `gltfloader.cpp` — JSON loading

Add `loadCamerasFromJson()` function, called after `loadCameras()`.

### `PropertyPanel.cpp` — Editor UI

Add DOF controls: focus distance slider, f-stop slider, blade count, enable
checkbox.

## Effort Estimates

| Feature | Effort | Files Changed |
|---|---|---|
| DOF (thin lens) | Medium | camera.h, OptixRenderParams.h, OptixRender.cu, OptixRender.cpp, blend2strelka.py, gltfloader.cpp, PropertyPanel.cpp |
| Lens shift | Small | camera.h, OptixRender.cu or camera.cpp, blend2strelka.py, gltfloader.cpp |
| Sensor dims / focal length | Small | camera.h, blend2strelka.py, gltfloader.cpp, PropertyPanel.cpp |
| Camera motion blur | Medium | OptixRender.cu, OptixRenderParams.h, OptixRender.cpp |
| Orthographic | Small | camera.cpp, OptixRender.cu, gltfloader.cpp |
| Panoramic | Large | OptixRender.cu (new ray gen functions per type) |

## Suggested Implementation Order

1. **Sensor dims + focal length** — small change, enables correct DOF math.
2. **DOF (thin lens)** — biggest visual payoff.
3. **Lens shift** — small, useful for architectural visualization.
4. **Orthographic** — small, widens the set of supported scenes.
5. **Camera motion blur** — medium, leverages existing motion blur infrastructure.
6. **Panoramic** — large, only if VR/360 rendering is a goal.
