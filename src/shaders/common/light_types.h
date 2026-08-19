#pragma once
// Light type constants shared between CPU scene code and GPU shader code.
// Keep this header free of CUDA/Metal/platform-specific includes.

enum LightType : int
{
    LIGHT_TYPE_RECT    = 0,
    LIGHT_TYPE_DISC    = 1,
    LIGHT_TYPE_SPHERE  = 2,
    LIGHT_TYPE_DISTANT = 3,
    LIGHT_TYPE_DOME    = 4,
    LIGHT_TYPE_POINT   = 5,
    LIGHT_TYPE_SPOT    = 6,
    // A spot whose cone is a rectangular pyramid and whose intensity across it
    // is an image: a home cinema beamer, a slide projector, a theatre gobo.
    LIGHT_TYPE_PROJECTOR = 7,
};

// How UniformLightDesc::intensity is interpreted before baking into the GPU
// light. Existing sidecar JSON without a unit field is Radiance, which keeps
// the historical "colour × intensity" behaviour unchanged.
enum LightIntensityUnit : int
{
    // Area / distant: colour×intensity is radiance (W/sr/m²).
    // Point / spot: colour×intensity is radiant intensity (W/sr) — rare, for
    // hand-authored legacy values that already matched the shader.
    LIGHT_UNIT_RADIANCE = 0,
    // Watts. Converted at bake time: area → Lambertian radiance, point →
    // I = Φ/(4π), spot → I = Φ/Ω_outer.
    LIGHT_UNIT_POWER = 1,
    // Candela (lm/sr ≈ W/sr radiometrically). Point and spot only; matches
    // KHR_lights_punctual and Blender's "Intensity" mode.
    LIGHT_UNIT_INTENSITY = 2,
    // Lux (lm/m² ≈ W/m²). Distant only; matches KHR directional intensity
    // and Blender's sun strength. Converted to radiance via the cone solid angle.
    LIGHT_UNIT_IRRADIANCE = 3,
};
