#pragma once
// ============================================================================
// light_emission_metal.h -- Light emission sampling for BDPT
//
// Samples a point on a light AND an outgoing direction, for generating
// light subpaths in bidirectional path tracing.
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "ShaderTypes.h"
#include "lights_metal.h"
#include "env_light_metal.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Emission sample result
// ---------------------------------------------------------------------------
struct LightEmissionSample
{
    float3 position;       // point on light surface
    float3 normal;         // surface normal at position
    float3 direction;      // sampled outgoing direction
    float3 Le;             // emitted radiance
    float  pdf_pos;        // area PDF for position
    float  pdf_dir;        // directional PDF (solid angle)
    int    light_index;    // which light was chosen (-1 for env map)
};

// ---------------------------------------------------------------------------
// Sample cosine-weighted hemisphere from a surface normal
// ---------------------------------------------------------------------------
static inline float3 cosineSampleHemisphere(float u1, float u2, float3 normal, thread float& pdf)
{
    // Local cosine-weighted sample
    float r = sqrt(u1);
    float phi = 2.0f * M_PI_F * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - u1));

    pdf = z * M_1_PI_F; // cos(theta) / pi

    // Build ONB from normal
    float3 T, B;
    if (fabs(normal.z) < 0.999f)
    {
        T = normalize(cross(float3(0.0f, 0.0f, 1.0f), normal));
    }
    else
    {
        T = normalize(cross(float3(1.0f, 0.0f, 0.0f), normal));
    }
    B = cross(normal, T);

    return normalize(x * T + y * B + z * normal);
}

// ---------------------------------------------------------------------------
// Sample uniform point on a sphere surface
// ---------------------------------------------------------------------------
static inline float3 uniformSampleSphere(float u1, float u2)
{
    float cosTheta = 1.0f - 2.0f * u1;
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * M_PI_F * u2;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// ---------------------------------------------------------------------------
// Emission sampling per light type
// ---------------------------------------------------------------------------

// Rect light (type 0): uniform on quad, cosine hemisphere direction
static inline LightEmissionSample sampleRectLightEmission(
    device const UniformLight& light,
    float u_pos1, float u_pos2,
    float u_dir1, float u_dir2)
{
    LightEmissionSample result;

    float3 p0 = float3(light.points[0]);
    float3 e1 = float3(light.points[1]) - p0;
    float3 e2 = float3(light.points[3]) - p0;

    result.position = p0 + e1 * u_pos1 + e2 * u_pos2;
    result.normal   = -normalize(cross(e1, e2)); // outward normal
    result.Le       = float3(light.color);

    float area = length(cross(e1, e2));
    result.pdf_pos = 1.0f / area;

    result.direction = cosineSampleHemisphere(u_dir1, u_dir2, result.normal, result.pdf_dir);

    return result;
}

// Disc light (type 1): uniform on disc, cosine hemisphere direction
static inline LightEmissionSample sampleDiscLightEmission(
    device const UniformLight& light,
    float u_pos1, float u_pos2,
    float u_dir1, float u_dir2)
{
    LightEmissionSample result;

    float radius = light.points[0].x;
    float3 center = float3(light.points[1]);
    result.normal = float3(light.normal);

    // Uniform point on disc
    float r = radius * sqrt(u_pos1);
    float theta = 2.0f * M_PI_F * u_pos2;

    float3 T, B;
    if (fabs(result.normal.z) < 0.999f)
        T = normalize(cross(float3(0.0f, 0.0f, 1.0f), result.normal));
    else
        T = normalize(cross(float3(1.0f, 0.0f, 0.0f), result.normal));
    B = cross(result.normal, T);

    result.position = center + r * cos(theta) * T + r * sin(theta) * B;
    result.Le       = float3(light.color);

    float area = M_PI_F * radius * radius;
    result.pdf_pos = 1.0f / area;

    result.direction = cosineSampleHemisphere(u_dir1, u_dir2, result.normal, result.pdf_dir);

    return result;
}

// Sphere light (type 2): uniform on sphere, cosine hemisphere from outward normal
static inline LightEmissionSample sampleSphereLightEmission(
    device const UniformLight& light,
    float u_pos1, float u_pos2,
    float u_dir1, float u_dir2)
{
    LightEmissionSample result;

    float radius = light.points[0].x;
    float3 center = float3(light.points[1]);

    float3 sphereDir = uniformSampleSphere(u_pos1, u_pos2);
    result.position = center + radius * sphereDir;
    result.normal   = sphereDir; // outward normal on sphere
    result.Le       = float3(light.color);

    float area = 4.0f * M_PI_F * radius * radius;
    result.pdf_pos = 1.0f / area;

    result.direction = cosineSampleHemisphere(u_dir1, u_dir2, result.normal, result.pdf_dir);

    return result;
}

// Distant light (type 3): point on scene-bounding disc, fixed direction (delta)
static inline LightEmissionSample sampleDistantLightEmission(
    device const UniformLight& light,
    float u_pos1, float u_pos2,
    float sceneBoundRadius,
    float3 sceneBoundCenter)
{
    LightEmissionSample result;

    float3 lightDir = -float3(light.normal); // direction light shines toward

    // Sample point on disc perpendicular to light direction at scene boundary
    float3 T, B;
    if (fabs(lightDir.z) < 0.999f)
        T = normalize(cross(float3(0.0f, 0.0f, 1.0f), lightDir));
    else
        T = normalize(cross(float3(1.0f, 0.0f, 0.0f), lightDir));
    B = cross(lightDir, T);

    float r = sceneBoundRadius * sqrt(u_pos1);
    float theta = 2.0f * M_PI_F * u_pos2;

    result.position = sceneBoundCenter - lightDir * sceneBoundRadius
                    + r * cos(theta) * T + r * sin(theta) * B;
    result.normal    = lightDir;
    result.direction = lightDir;
    result.Le        = float3(light.color);

    float discArea = M_PI_F * sceneBoundRadius * sceneBoundRadius;
    result.pdf_pos = 1.0f / discArea;
    result.pdf_dir = 1.0f; // delta in direction

    return result;
}

// ---------------------------------------------------------------------------
// Environment map emission sampling
// ---------------------------------------------------------------------------
static inline LightEmissionSample sampleEnvMapEmission(
    float u_dir1, float u_dir2,
    float u_pos1, float u_pos2,
    device const float* envCdfX,
    device const float* envCdfY,
    uint32_t envMapWidth,
    uint32_t envMapHeight,
    float envMapRotation,
    float envMapIntensity,
    float3 envMapColorTint,
    texture2d<float> envMapTexture,
    float sceneBoundRadius,
    float3 sceneBoundCenter)
{
    LightEmissionSample result;

    // Sample direction from environment map CDF
    float envPdf = 0.0f;
    float3 dir = sampleEnvMap(float2(u_dir1, u_dir2),
                               envCdfX, envCdfY,
                               envMapWidth, envMapHeight,
                               envMapRotation, envPdf);

    if (envPdf <= 0.0f)
    {
        result.Le = float3(0.0f);
        result.pdf_pos = 0.0f;
        result.pdf_dir = 0.0f;
        return result;
    }

    result.direction = dir;
    result.normal    = -dir; // points inward toward scene
    result.pdf_dir   = envPdf;

    // Sample position on a disc at the scene boundary, perpendicular to direction
    float3 T, B;
    if (fabs(dir.z) < 0.999f)
        T = normalize(cross(float3(0.0f, 0.0f, 1.0f), dir));
    else
        T = normalize(cross(float3(1.0f, 0.0f, 0.0f), dir));
    B = cross(dir, T);

    float r = sceneBoundRadius * sqrt(u_pos1);
    float theta = 2.0f * M_PI_F * u_pos2;

    result.position = sceneBoundCenter - dir * sceneBoundRadius
                    + r * cos(theta) * T + r * sin(theta) * B;

    float discArea = M_PI_F * sceneBoundRadius * sceneBoundRadius;
    result.pdf_pos = 1.0f / discArea;

    // Look up environment radiance
    constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
    float2 uv = dirToEnvUV(dir, envMapRotation);
    float4 envSample = envMapTexture.sample(envSampler, uv);
    result.Le = envSample.xyz * envMapIntensity * envMapColorTint;

    result.light_index = -1;

    return result;
}

// ---------------------------------------------------------------------------
// Top-level emission sampling: choose a light and sample emission
// ---------------------------------------------------------------------------
static inline LightEmissionSample sampleLightEmission(
    constant Uniforms& uniforms,
    device UniformLight* lights,
    thread SamplerState& sampler,
    device const float* envCdfX,
    device const float* envCdfY,
    texture2d<float> envMapTexture)
{
    float u_select = random<SampleDimension::eLightId>(sampler, uniforms.samplerType);
    float u_pos1   = random<SampleDimension::eLightPointX>(sampler, uniforms.samplerType);
    float u_pos2   = random<SampleDimension::eLightPointY>(sampler, uniforms.samplerType);
    float u_dir1   = random<SampleDimension::eBSDF0>(sampler, uniforms.samplerType);
    float u_dir2   = random<SampleDimension::eBSDF1>(sampler, uniforms.samplerType);

    uint32_t numLights = uniforms.numLights;
    bool hasEnv = uniforms.hasEnvMap != 0;

    // Selection: if env map present, 50/50 between env and local lights
    LightEmissionSample result;

    if (hasEnv && (numLights == 0 || u_select >= 0.5f))
    {
        // Environment map
        float selectionPdf = hasEnv && numLights > 0 ? 0.5f : 1.0f;
        result = sampleEnvMapEmission(u_dir1, u_dir2, u_pos1, u_pos2,
                                       envCdfX, envCdfY,
                                       uniforms.envMapWidth, uniforms.envMapHeight,
                                       uniforms.envMapRotation,
                                       uniforms.envMapIntensity,
                                       float3(uniforms.envMapColorTint),
                                       envMapTexture,
                                       uniforms.sceneBoundRadius,
                                       float3(uniforms.sceneBoundCenter));
        result.pdf_pos *= selectionPdf;
        result.light_index = -1;
    }
    else
    {
        // Local light
        float selectionPdf = hasEnv ? 0.5f / float(numLights) : 1.0f / float(numLights);
        float remapped = hasEnv ? (u_select * 2.0f) : u_select;
        uint32_t lightId = min(uint32_t(numLights * remapped), numLights - 1);
        device const UniformLight& light = lights[lightId];

        switch (light.type)
        {
        case 0: // Rect
            result = sampleRectLightEmission(light, u_pos1, u_pos2, u_dir1, u_dir2);
            break;
        case 1: // Disc
            result = sampleDiscLightEmission(light, u_pos1, u_pos2, u_dir1, u_dir2);
            break;
        case 2: // Sphere
            result = sampleSphereLightEmission(light, u_pos1, u_pos2, u_dir1, u_dir2);
            break;
        case 3: // Distant
            result = sampleDistantLightEmission(light, u_pos1, u_pos2,
                                                  uniforms.sceneBoundRadius,
                                                  float3(uniforms.sceneBoundCenter));
            break;
        default:
            result.Le = float3(0.0f);
            result.pdf_pos = 0.0f;
            result.pdf_dir = 0.0f;
            break;
        }

        result.pdf_pos *= selectionPdf;
        result.light_index = (int)lightId;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Compute PDF for a given emission configuration (for MIS)
// ---------------------------------------------------------------------------
static inline float lightEmissionPdf_pos(
    device const UniformLight& light,
    float3 position)
{
    float area = calcLightArea(light);
    return (area > 0.0f) ? (1.0f / area) : 0.0f;
}

static inline float lightEmissionPdf_dir(
    device const UniformLight& light,
    float3 direction,
    float3 lightNormal)
{
    // Cosine hemisphere PDF
    float cosTheta = max(dot(lightNormal, direction), 0.0f);
    return cosTheta * M_1_PI_F;
}
