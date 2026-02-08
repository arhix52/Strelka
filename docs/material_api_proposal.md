# Strelka Material API Proposal

## Overview

A self-contained BSDF material system that replaces the MDL SDK dependency with pure header-only
GPU code. Four core functions, five material types, one flat parameter struct. Compiles on both
CUDA (OptiX) and Metal with zero external dependencies.

Replaces: `mdlRuntime`, `mdlPtxCodeGen`, `mdlNeurayLoader`, `mdlMaterialCompiler`,
`mtlxMdlCodeGen`, `texture_support_cuda.h`, LLVM 12, MDL SDK, MaterialX.

---

## File Layout

```
include/material/
    bsdf.h                  // top-level dispatch (the 4 API functions)
    bsdf_types.h            // BsdfSampleResult, BsdfEvalResult, BsdfEventType
    material_params.h       // MaterialParams, MaterialType
    surface_interaction.h   // SurfaceInteraction
    material_math.h         // cross-platform float3/float2 math + DEVICE_FUNC macro
    sampling.h              // cosine_hemisphere, ggx_vndf, uniform_sphere, etc.
    fresnel.h               // schlick, conductor, dielectric Fresnel
    microfacet.h            // GGX NDF, Smith G, VNDF sampling
    texture_sample.h        // platform-specific texture sampling
    bxdfs/
        diffuse.h           // Lambert + Oren-Nayar
        conductor.h         // GGX metallic reflection
        dielectric.h        // GGX glass (reflect + refract)
        standard_pbr.h      // multi-lobe: metallic blend + clearcoat + sheen
```

---

## Surface Interaction

Input to all BSDF functions. Filled from geometry intersection data.

```cpp
// material/surface_interaction.h

#ifndef MATERIAL_SURFACE_INTERACTION_H
#define MATERIAL_SURFACE_INTERACTION_H

#include "material_math.h"

struct SurfaceInteraction
{
    float3 position;       // world-space hit point
    float3 normal;         // shading normal (after normal map)
    float3 geom_normal;    // geometric normal
    float3 tangent;        // shading tangent
    float3 bitangent;      // shading bitangent
    float2 uv;             // texture coordinates
    float3 wo;             // outgoing direction (toward camera), in world space
    bool   front_face;     // true if ray hit front face
};

#endif
```

---

## BSDF Result Types

```cpp
// material/bsdf_types.h

#ifndef MATERIAL_BSDF_TYPES_H
#define MATERIAL_BSDF_TYPES_H

#include "material_math.h"

enum BsdfEventType : uint32_t
{
    BSDF_EVENT_ABSORB       = 0,
    BSDF_EVENT_DIFFUSE      = 1 << 0,
    BSDF_EVENT_GLOSSY       = 1 << 1,
    BSDF_EVENT_SPECULAR     = 1 << 2,
    BSDF_EVENT_REFLECTION   = 1 << 3,
    BSDF_EVENT_TRANSMISSION = 1 << 4,
};

struct BsdfSampleResult
{
    float3   wi;             // sampled incoming direction (toward light)
    float    pdf;            // probability density
    float3   bsdf_over_pdf;  // bsdf * cos(theta) / pdf  (ready to multiply with throughput)
    uint32_t event_type;     // BsdfEventType flags
};

struct BsdfEvalResult
{
    float3 diffuse;   // diffuse lobe contribution
    float3 specular;  // specular lobe contribution
    float  pdf;       // probability of sampling this direction
};

#endif
```

---

## Material Parameters

A single flat POD struct that lives in a GPU buffer. This is the "argument block" for each material.

```cpp
// material/material_params.h

#ifndef MATERIAL_PARAMS_H
#define MATERIAL_PARAMS_H

#include "material_math.h"

enum MaterialType : uint32_t
{
    MATERIAL_DIFFUSE      = 0,  // Lambert / Oren-Nayar
    MATERIAL_CONDUCTOR    = 1,  // metallic GGX reflection
    MATERIAL_DIELECTRIC   = 2,  // glass (reflection + transmission)
    MATERIAL_STANDARD_PBR = 3,  // full metallic-roughness PBR (covers glTF)
    MATERIAL_EMISSIVE     = 4,  // pure emitter (area lights)
};

struct MaterialParams
{
    MaterialType type;

    // --- core ---
    float3 base_color;                  // albedo / reflectance
    float  roughness;                   // GGX alpha roughness [0,1]
    float  metallic;                    // metallic blend [0,1]  (STANDARD_PBR)
    float  ior;                         // index of refraction (DIELECTRIC, STANDARD_PBR specular)
    float  transmission;                // transmission weight [0,1] (STANDARD_PBR)

    // --- textures (bindless handles, 0 = no texture) ---
    uint64_t base_color_tex;
    uint64_t normal_tex;
    uint64_t roughness_metallic_tex;    // G=roughness, B=metallic (glTF convention)
    uint64_t emission_tex;

    // --- emission ---
    float3 emission_color;
    float  emission_intensity;

    // --- advanced lobes ---
    float  anisotropy;                  // anisotropic roughness [-1,1]
    float  clearcoat;                   // clearcoat weight [0,1]
    float  clearcoat_roughness;         // clearcoat roughness [0,1]
    float  specular_tint;               // Fresnel tint toward base_color
    float  sheen;                       // sheen weight (fabric)
    float  sheen_tint;                  // sheen color tint

    float  _pad[2];                     // alignment to 16 bytes
};

#endif
```

### Material type coverage

| Field | Diffuse | Conductor | Dielectric | Standard PBR | Emissive |
|-------|---------|-----------|------------|-------------|----------|
| `base_color` | albedo | F0 reflectance | tint | albedo | - |
| `roughness` | Oren-Nayar sigma | GGX alpha | GGX alpha | GGX alpha | - |
| `metallic` | - | - | - | diffuse/conductor blend | - |
| `ior` | - | - | refraction index | specular F0 | - |
| `transmission` | - | - | always 1 | glass blend weight | - |
| `anisotropy` | - | aniso stretch | aniso stretch | aniso stretch | - |
| `clearcoat` | - | - | - | extra GGX layer | - |
| `sheen` | - | - | - | fabric sheen | - |
| `emission_*` | - | - | - | emissive add | emitter |

---

## Core API: Four Functions

```cpp
// material/bsdf.h

#ifndef MATERIAL_BSDF_H
#define MATERIAL_BSDF_H

#include "bsdf_types.h"
#include "material_params.h"
#include "surface_interaction.h"
#include "texture_sample.h"
#include "bxdfs/diffuse.h"
#include "bxdfs/conductor.h"
#include "bxdfs/dielectric.h"
#include "bxdfs/standard_pbr.h"

// ---------------------------------------------------------------------------
// bsdf_init
// ---------------------------------------------------------------------------
// Initialize material state from textures (normal mapping, texture lookups).
// Modifies `si` in place (updates shading normal from normal map).
// Writes resolved params into `resolved` (texture-sampled base_color, etc.)

DEVICE_FUNC void bsdf_init(
    const MaterialParams& params,
    SurfaceInteraction&   si,
    MaterialParams&       resolved);

// ---------------------------------------------------------------------------
// bsdf_sample
// ---------------------------------------------------------------------------
// Sample a direction from the BSDF.
// Returns sampled direction + bsdf*cos/pdf (ready to multiply with throughput).

DEVICE_FUNC BsdfSampleResult bsdf_sample(
    const MaterialParams&     params,   // resolved (post-init)
    const SurfaceInteraction& si,
    float4                    xi);      // 4 random numbers [0,1)

// ---------------------------------------------------------------------------
// bsdf_eval
// ---------------------------------------------------------------------------
// Evaluate the BSDF for a given incoming direction wi.
// Returns diffuse + specular contributions and combined PDF.
// Used for next-event estimation (direct light sampling + MIS).

DEVICE_FUNC BsdfEvalResult bsdf_eval(
    const MaterialParams&     params,   // resolved (post-init)
    const SurfaceInteraction& si,
    float3                    wi);      // incoming direction (toward light)

// ---------------------------------------------------------------------------
// bsdf_pdf
// ---------------------------------------------------------------------------
// Probability density for sampling direction wi.
// Used when only the PDF is needed (e.g., MIS weight for light hits).

DEVICE_FUNC float bsdf_pdf(
    const MaterialParams&     params,
    const SurfaceInteraction& si,
    float3                    wi);

#endif
```

---

## Dispatch Implementation

```cpp
// material/bsdf.h (continued — inline implementation)

DEVICE_FUNC void bsdf_init(
    const MaterialParams& params,
    SurfaceInteraction&   si,
    MaterialParams&       resolved)
{
    resolved = params;

    // Sample textures if bound
    if (params.base_color_tex) {
        float4 tex = sample_texture(params.base_color_tex, si.uv);
        resolved.base_color = params.base_color * make_float3(tex.x, tex.y, tex.z);
    }
    if (params.roughness_metallic_tex) {
        float4 rm = sample_texture(params.roughness_metallic_tex, si.uv);
        resolved.roughness *= rm.y;   // green channel
        resolved.metallic  *= rm.z;   // blue channel
    }
    if (params.emission_tex) {
        float4 em = sample_texture(params.emission_tex, si.uv);
        resolved.emission_color *= make_float3(em.x, em.y, em.z);
    }

    // Normal mapping
    if (params.normal_tex) {
        float4 n = sample_texture(params.normal_tex, si.uv);
        float3 tangent_normal = make_float3(n.x, n.y, n.z) * 2.0f - 1.0f;
        si.normal = normalize(
            si.tangent   * tangent_normal.x +
            si.bitangent * tangent_normal.y +
            si.normal    * tangent_normal.z);
    }

    // Clamp roughness to avoid singularities
    resolved.roughness = max(resolved.roughness, 0.001f);
}

DEVICE_FUNC BsdfSampleResult bsdf_sample(
    const MaterialParams&     params,
    const SurfaceInteraction& si,
    float4                    xi)
{
    switch (params.type)
    {
    case MATERIAL_DIFFUSE:      return diffuse_sample(params, si, xi);
    case MATERIAL_CONDUCTOR:    return conductor_sample(params, si, xi);
    case MATERIAL_DIELECTRIC:   return dielectric_sample(params, si, xi);
    case MATERIAL_STANDARD_PBR: return standard_pbr_sample(params, si, xi);
    case MATERIAL_EMISSIVE:
    default:
        BsdfSampleResult r = {};
        r.event_type = BSDF_EVENT_ABSORB;
        return r;
    }
}

DEVICE_FUNC BsdfEvalResult bsdf_eval(
    const MaterialParams&     params,
    const SurfaceInteraction& si,
    float3                    wi)
{
    switch (params.type)
    {
    case MATERIAL_DIFFUSE:      return diffuse_eval(params, si, wi);
    case MATERIAL_CONDUCTOR:    return conductor_eval(params, si, wi);
    case MATERIAL_DIELECTRIC:   return dielectric_eval(params, si, wi);
    case MATERIAL_STANDARD_PBR: return standard_pbr_eval(params, si, wi);
    case MATERIAL_EMISSIVE:
    default:
        BsdfEvalResult r = {};
        return r;
    }
}

DEVICE_FUNC float bsdf_pdf(
    const MaterialParams&     params,
    const SurfaceInteraction& si,
    float3                    wi)
{
    switch (params.type)
    {
    case MATERIAL_DIFFUSE:      return diffuse_pdf(params, si, wi);
    case MATERIAL_CONDUCTOR:    return conductor_pdf(params, si, wi);
    case MATERIAL_DIELECTRIC:   return dielectric_pdf(params, si, wi);
    case MATERIAL_STANDARD_PBR: return standard_pbr_pdf(params, si, wi);
    case MATERIAL_EMISSIVE:
    default:
        return 0.0f;
    }
}
```

---

## BxDF Building Blocks

### Microfacet Utilities

```cpp
// material/microfacet.h

// GGX Normal Distribution Function
DEVICE_FUNC float ggx_ndf(float3 h, float3 n, float alpha)
{
    float ndoth = max(dot(n, h), 0.0f);
    float a2 = alpha * alpha;
    float denom = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom);
}

// Smith G2 (height-correlated)
DEVICE_FUNC float ggx_smith_g2(float3 wo, float3 wi, float3 n, float alpha)
{
    float a2 = alpha * alpha;
    float ndotv = max(abs(dot(n, wo)), 1e-7f);
    float ndotl = max(abs(dot(n, wi)), 1e-7f);
    float g1_v = 2.0f * ndotv / (ndotv + sqrt(a2 + (1.0f - a2) * ndotv * ndotv));
    float g1_l = 2.0f * ndotl / (ndotl + sqrt(a2 + (1.0f - a2) * ndotl * ndotl));
    return g1_v * g1_l;
}

// GGX Visible Normal Distribution Function (VNDF) sampling
// Heitz 2018: "Sampling the GGX Distribution of Visible Normals"
DEVICE_FUNC float3 ggx_vndf_sample(float3 wo_local, float alpha, float2 xi)
{
    // Stretch wo
    float3 v = normalize(make_float3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));

    // Orthonormal basis around v
    float3 t1 = (v.z < 0.9999f) ? normalize(cross(v, make_float3(0, 0, 1))) : make_float3(1, 0, 0);
    float3 t2 = cross(t1, v);

    // Parameterization of projected area
    float r = sqrt(xi.x);
    float phi = 2.0f * M_PI_F * xi.y;
    float t_1 = r * cos(phi);
    float t_2 = r * sin(phi);
    float s = 0.5f * (1.0f + v.z);
    t_2 = (1.0f - s) * sqrt(1.0f - t_1 * t_1) + s * t_2;

    // Reprojection onto hemisphere
    float3 nh = t_1 * t1 + t_2 * t2 + sqrt(max(0.0f, 1.0f - t_1 * t_1 - t_2 * t_2)) * v;

    // Unstretch
    return normalize(make_float3(alpha * nh.x, alpha * nh.y, max(0.0f, nh.z)));
}
```

### Fresnel

```cpp
// material/fresnel.h

// Schlick approximation (for conductors and dielectrics at grazing angles)
DEVICE_FUNC float3 fresnel_schlick(float3 F0, float cos_theta)
{
    float t = 1.0f - cos_theta;
    float t2 = t * t;
    return F0 + (make_float3(1.0f) - F0) * (t2 * t2 * t);
}

// Exact dielectric Fresnel
DEVICE_FUNC float fresnel_dielectric(float cos_theta_i, float eta)
{
    float sin2_t = eta * eta * (1.0f - cos_theta_i * cos_theta_i);
    if (sin2_t > 1.0f) return 1.0f; // total internal reflection

    float cos_theta_t = sqrt(1.0f - sin2_t);
    float rs = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float rp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    return 0.5f * (rs * rs + rp * rp);
}
```

### Sampling Utilities

```cpp
// material/sampling.h

// Cosine-weighted hemisphere sampling
DEVICE_FUNC float3 cosine_hemisphere_sample(float2 xi)
{
    float r   = sqrt(xi.x);
    float phi = 2.0f * M_PI_F * xi.y;
    return make_float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - xi.x)));
}

DEVICE_FUNC float cosine_hemisphere_pdf(float cos_theta)
{
    return cos_theta * M_1_PI_F;
}

// Transform direction from local (z-up) to world space
DEVICE_FUNC float3 local_to_world(float3 local, float3 n, float3 t, float3 b)
{
    return t * local.x + b * local.y + n * local.z;
}

// Transform direction from world to local (z-up) space
DEVICE_FUNC float3 world_to_local(float3 world, float3 n, float3 t, float3 b)
{
    return make_float3(dot(world, t), dot(world, b), dot(world, n));
}
```

---

## BxDF Implementations

### Diffuse (Lambert)

```cpp
// material/bxdfs/diffuse.h

DEVICE_FUNC BsdfSampleResult diffuse_sample(
    const MaterialParams& p, const SurfaceInteraction& si, float4 xi)
{
    BsdfSampleResult r;
    float3 local_dir = cosine_hemisphere_sample(make_float2(xi.x, xi.y));
    r.wi = local_to_world(local_dir, si.normal, si.tangent, si.bitangent);
    float cos_theta = max(dot(si.normal, r.wi), 0.0f);
    r.pdf = cosine_hemisphere_pdf(cos_theta);
    r.bsdf_over_pdf = p.base_color;  // (albedo/pi * cos) / (cos/pi) = albedo
    r.event_type = BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION;
    return r;
}

DEVICE_FUNC BsdfEvalResult diffuse_eval(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = {};
    float cos_theta = dot(si.normal, wi);
    if (cos_theta <= 0.0f) return r;
    r.diffuse = p.base_color * M_1_PI_F;
    r.pdf = cosine_hemisphere_pdf(cos_theta);
    return r;
}

DEVICE_FUNC float diffuse_pdf(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    float cos_theta = max(dot(si.normal, wi), 0.0f);
    return cosine_hemisphere_pdf(cos_theta);
}
```

### Conductor (Metallic GGX Reflection)

```cpp
// material/bxdfs/conductor.h

DEVICE_FUNC BsdfSampleResult conductor_sample(
    const MaterialParams& p, const SurfaceInteraction& si, float4 xi)
{
    BsdfSampleResult r;
    float alpha = p.roughness * p.roughness;

    float3 wo_local = world_to_local(si.wo, si.normal, si.tangent, si.bitangent);
    float3 h_local = ggx_vndf_sample(wo_local, alpha, make_float2(xi.x, xi.y));
    float3 h = local_to_world(h_local, si.normal, si.tangent, si.bitangent);

    r.wi = reflect_dir(-si.wo, h);  // 2 * dot(wo, h) * h - wo
    float ndotwi = dot(si.normal, r.wi);
    if (ndotwi <= 0.0f) {
        r.event_type = BSDF_EVENT_ABSORB;
        return r;
    }

    float3 F = fresnel_schlick(p.base_color, dot(si.wo, h));  // base_color = F0 for metals
    float G = ggx_smith_g2(si.wo, r.wi, si.normal, alpha);
    float D = ggx_ndf(h, si.normal, alpha);

    float ndotwo = max(abs(dot(si.normal, si.wo)), 1e-7f);
    // VNDF PDF = D * G1(wo) * max(dot(wo,h),0) / ndotwo
    // Jacobian for reflection: 1 / (4 * dot(wo, h))
    r.pdf = D * G / (4.0f * ndotwo);  // simplified VNDF reflection PDF
    r.bsdf_over_pdf = F * G / (G /* G1 cancellation from VNDF */);
    r.event_type = (alpha < 0.001f)
        ? (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
        : (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION);
    return r;
}

DEVICE_FUNC BsdfEvalResult conductor_eval(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = {};
    float ndotwi = dot(si.normal, wi);
    float ndotwo = dot(si.normal, si.wo);
    if (ndotwi <= 0.0f || ndotwo <= 0.0f) return r;

    float alpha = p.roughness * p.roughness;
    float3 h = normalize(si.wo + wi);
    float3 F = fresnel_schlick(p.base_color, dot(si.wo, h));
    float D = ggx_ndf(h, si.normal, alpha);
    float G = ggx_smith_g2(si.wo, wi, si.normal, alpha);

    r.specular = F * D * G / (4.0f * ndotwo * ndotwi);
    r.pdf = D * G / (4.0f * ndotwo);
    return r;
}

DEVICE_FUNC float conductor_pdf(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    float ndotwo = max(dot(si.normal, si.wo), 1e-7f);
    float alpha = p.roughness * p.roughness;
    float3 h = normalize(si.wo + wi);
    float D = ggx_ndf(h, si.normal, alpha);
    float G = ggx_smith_g2(si.wo, wi, si.normal, alpha);
    return D * G / (4.0f * ndotwo);
}
```

### Dielectric (Glass)

```cpp
// material/bxdfs/dielectric.h

DEVICE_FUNC BsdfSampleResult dielectric_sample(
    const MaterialParams& p, const SurfaceInteraction& si, float4 xi)
{
    BsdfSampleResult r;
    float alpha = p.roughness * p.roughness;
    float eta = si.front_face ? (1.0f / p.ior) : p.ior;

    float3 wo_local = world_to_local(si.wo, si.normal, si.tangent, si.bitangent);
    float3 h_local = ggx_vndf_sample(wo_local, alpha, make_float2(xi.x, xi.y));
    float3 h = local_to_world(h_local, si.normal, si.tangent, si.bitangent);

    float cos_theta_i = dot(si.wo, h);
    float F = fresnel_dielectric(cos_theta_i, eta);

    if (xi.z < F)
    {
        // Reflection
        r.wi = reflect_dir(-si.wo, h);
        if (dot(si.normal, r.wi) <= 0.0f) { r.event_type = BSDF_EVENT_ABSORB; return r; }
        r.bsdf_over_pdf = p.base_color;  // tint
        r.event_type = (alpha < 0.001f)
            ? (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
            : (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION);
    }
    else
    {
        // Transmission (refraction)
        float3 refracted;
        if (!refract_dir(-si.wo, h, eta, refracted)) {
            // Total internal reflection fallback
            r.wi = reflect_dir(-si.wo, h);
            r.bsdf_over_pdf = p.base_color;
            r.event_type = BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION;
            return r;
        }
        r.wi = refracted;
        r.bsdf_over_pdf = p.base_color;  // tint
        r.event_type = (alpha < 0.001f)
            ? (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)
            : (BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION);
    }

    r.pdf = 1.0f;  // delta or near-delta; MIS handled via specular flag
    return r;
}

DEVICE_FUNC BsdfEvalResult dielectric_eval(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    // For smooth dielectrics, eval returns 0 (delta distribution).
    // For rough dielectrics, evaluate microfacet reflection + refraction lobes.
    BsdfEvalResult r = {};
    float alpha = p.roughness * p.roughness;
    if (alpha < 0.001f) return r;  // smooth = delta

    // Rough dielectric: evaluate both reflection and transmission lobes
    // (implementation follows microfacet refraction model, Walter et al. 2007)
    // ... full implementation here ...

    return r;
}

DEVICE_FUNC float dielectric_pdf(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    float alpha = p.roughness * p.roughness;
    if (alpha < 0.001f) return 0.0f;
    // ... rough dielectric PDF ...
    return 0.0f;
}
```

### Standard PBR (glTF Metallic-Roughness)

The workhorse material. Maps 1:1 to glTF's PBR model.

```cpp
// material/bxdfs/standard_pbr.h

DEVICE_FUNC BsdfSampleResult standard_pbr_sample(
    const MaterialParams& p, const SurfaceInteraction& si, float4 xi)
{
    // Lobe weights
    float diffuse_w     = (1.0f - p.metallic) * (1.0f - p.transmission);
    float specular_w    = 1.0f;
    float transmission_w = (1.0f - p.metallic) * p.transmission;
    float total = diffuse_w + specular_w + transmission_w;
    diffuse_w     /= total;
    specular_w    /= total;
    transmission_w /= total;

    // Select lobe based on xi.w
    float selector = xi.w;
    float cdf = 0.0f;

    if (selector < (cdf += diffuse_w))
    {
        // Diffuse lobe: cosine-weighted hemisphere
        float3 local_dir = cosine_hemisphere_sample(make_float2(xi.x, xi.y));
        BsdfSampleResult r;
        r.wi = local_to_world(local_dir, si.normal, si.tangent, si.bitangent);
        float cos_theta = max(dot(si.normal, r.wi), 0.0f);

        // Energy conservation: (1 - F) * albedo
        float3 h = normalize(si.wo + r.wi);
        float3 F0 = mix(make_float3(0.04f), p.base_color, p.metallic);
        float3 F = fresnel_schlick(F0, max(dot(h, r.wi), 0.0f));

        r.bsdf_over_pdf = (make_float3(1.0f) - F) * p.base_color / diffuse_w;
        r.pdf = diffuse_w * cosine_hemisphere_pdf(cos_theta);
        r.event_type = BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION;
        return r;
    }
    else if (selector < (cdf += specular_w))
    {
        // Specular lobe: GGX VNDF reflection
        float alpha = p.roughness * p.roughness;
        float3 wo_local = world_to_local(si.wo, si.normal, si.tangent, si.bitangent);
        float3 h_local = ggx_vndf_sample(wo_local, alpha, make_float2(xi.x, xi.y));
        float3 h = local_to_world(h_local, si.normal, si.tangent, si.bitangent);

        BsdfSampleResult r;
        r.wi = reflect_dir(-si.wo, h);
        if (dot(si.normal, r.wi) <= 0.0f) { r.event_type = BSDF_EVENT_ABSORB; return r; }

        float3 F0 = mix(make_float3(0.04f), p.base_color, p.metallic);
        float3 F = fresnel_schlick(F0, dot(si.wo, h));
        float G = ggx_smith_g2(si.wo, r.wi, si.normal, alpha);

        r.bsdf_over_pdf = F * G / (specular_w * /* G1 from VNDF */ 1.0f);
        r.pdf = specular_w * ggx_vndf_pdf(wo_local, h_local, alpha);
        r.event_type = (alpha < 0.001f)
            ? (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
            : (BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION);
        return r;
    }
    else
    {
        // Transmission lobe: dielectric refraction
        MaterialParams glass_params = p;
        glass_params.type = MATERIAL_DIELECTRIC;
        return dielectric_sample(glass_params, si, xi);
    }
}

DEVICE_FUNC BsdfEvalResult standard_pbr_eval(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult r = {};
    float cos_theta = dot(si.normal, wi);
    if (cos_theta <= 0.0f) return r;  // below surface

    float3 h = normalize(si.wo + wi);
    float alpha = p.roughness * p.roughness;

    // Fresnel
    float3 F0 = mix(make_float3(0.04f), p.base_color, p.metallic);
    float3 F = fresnel_schlick(F0, max(dot(h, wi), 0.0f));

    // Diffuse lobe
    float diffuse_w = (1.0f - p.metallic) * (1.0f - p.transmission);
    r.diffuse = diffuse_w * (make_float3(1.0f) - F) * p.base_color * M_1_PI_F;

    // Specular lobe
    float D = ggx_ndf(h, si.normal, alpha);
    float G = ggx_smith_g2(si.wo, wi, si.normal, alpha);
    float ndotwo = max(abs(dot(si.normal, si.wo)), 1e-7f);
    float ndotwi = max(abs(dot(si.normal, wi)), 1e-7f);
    r.specular = F * D * G / (4.0f * ndotwo * ndotwi);

    // Combined PDF
    float total = diffuse_w + 1.0f;  // specular_w = 1
    float pdf_diffuse  = cosine_hemisphere_pdf(cos_theta);
    float pdf_specular = D * G / (4.0f * ndotwo);  // VNDF PDF
    r.pdf = (diffuse_w * pdf_diffuse + 1.0f * pdf_specular) / total;

    return r;
}

DEVICE_FUNC float standard_pbr_pdf(
    const MaterialParams& p, const SurfaceInteraction& si, float3 wi)
{
    BsdfEvalResult eval = standard_pbr_eval(p, si, wi);
    return eval.pdf;
}
```

---

## Cross-Platform Macros

```cpp
// material/material_math.h

#ifndef MATERIAL_MATH_H
#define MATERIAL_MATH_H

#if defined(__CUDA_ARCH__)
    // ---- CUDA / OptiX ----
    #define DEVICE_FUNC   __device__ __forceinline__
    #define M_PI_F        3.14159265358979323846f
    #define M_1_PI_F      0.31830988618379067154f
    // float3, float2, float4 are CUDA built-ins
    // make_float3, make_float2, make_float4 are CUDA built-ins
    // dot, cross, normalize, length available via sutil/vec_math.h

    DEVICE_FUNC float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }

    DEVICE_FUNC float3 reflect_dir(float3 incident, float3 normal) {
        return incident - 2.0f * dot(incident, normal) * normal;
    }

    DEVICE_FUNC bool refract_dir(float3 incident, float3 normal, float eta, float3& out) {
        float cosi = dot(normal, incident);
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + sqrt(1.0f - sin2t)) * normal;
        return true;
    }

#elif defined(__METAL_VERSION__)
    // ---- Metal ----
    #define DEVICE_FUNC   inline
    // float3, float2, float4 are Metal built-ins
    // dot, cross, normalize, length, mix, reflect are Metal built-ins

    inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
    inline float3 make_float3(float v) { return float3(v); }
    inline float2 make_float2(float x, float y) { return float2(x, y); }
    inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }

    inline float3 reflect_dir(float3 incident, float3 normal) {
        return reflect(incident, normal);
    }

    inline bool refract_dir(float3 incident, float3 normal, float eta, thread float3& out) {
        float cosi = dot(normal, incident);
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        if (sin2t > 1.0f) return false;
        out = eta * incident - (eta * cosi + sqrt(1.0f - sin2t)) * normal;
        return true;
    }

#else
    // ---- CPU (tests, previews) ----
    #define DEVICE_FUNC   inline
    #include <glm/glm.hpp>
    using float2 = glm::vec2;
    using float3 = glm::vec3;
    using float4 = glm::vec4;
    inline float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
    inline float3 make_float3(float v) { return float3(v); }
    inline float2 make_float2(float x, float y) { return float2(x, y); }
    inline float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
    using glm::dot; using glm::cross; using glm::normalize; using glm::mix;
    using glm::reflect; using glm::refract; using glm::max; using glm::min;
    #define M_PI_F   3.14159265358979323846f
    #define M_1_PI_F 0.31830988618379067154f
#endif

#endif
```

---

## Texture Sampling (Platform-Specific)

```cpp
// material/texture_sample.h

#ifndef MATERIAL_TEXTURE_SAMPLE_H
#define MATERIAL_TEXTURE_SAMPLE_H

#include "material_math.h"

#if defined(__CUDA_ARCH__)

DEVICE_FUNC float4 sample_texture(uint64_t handle, float2 uv)
{
    if (!handle) return make_float4(1, 1, 1, 1);
    cudaTextureObject_t tex = (cudaTextureObject_t)handle;
    return tex2D<float4>(tex, uv.x, uv.y);
}

#elif defined(__METAL_VERSION__)

// Metal: texture array passed via argument buffer.
// handle is an index into the array.
// Caller passes the texture array; see integration notes below.
//
// Because Metal cannot access textures via raw uint64 handles,
// the raytracingKernel receives an array<texture2d<float>, MAX_TEXTURES>
// and bsdf_init/sample_texture are called with that array in scope.
// A lightweight wrapper:

struct TextureContext
{
    array<texture2d<float>, 128> textures;
    sampler                       tex_sampler;
};

inline float4 sample_texture(TextureContext ctx, uint64_t handle, float2 uv)
{
    if (!handle) return float4(1, 1, 1, 1);
    return ctx.textures[(uint)handle].sample(ctx.tex_sampler, uv);
}

#else

// CPU stub
inline float4 sample_texture(uint64_t handle, float2 uv)
{
    return make_float4(1, 1, 1, 1);  // white fallback
}

#endif

#endif
```

---

## glTF Mapping

In `GltfLoader`, instead of generating MDL code strings, fill `MaterialParams` directly:

```cpp
MaterialParams mat = {};
mat.type                     = MATERIAL_STANDARD_PBR;
mat.base_color               = gltfPBR.baseColorFactor;
mat.metallic                 = gltfPBR.metallicFactor;
mat.roughness                = gltfPBR.roughnessFactor;
mat.ior                      = gltfMat.ior;           // KHR_materials_ior
mat.transmission             = gltfMat.transmissionFactor; // KHR_materials_transmission
mat.base_color_tex           = loadTexture(gltfPBR.baseColorTexture);
mat.roughness_metallic_tex   = loadTexture(gltfPBR.metallicRoughnessTexture);
mat.normal_tex               = loadTexture(gltfMat.normalTexture);
mat.emission_color           = gltfMat.emissiveFactor;
mat.emission_intensity       = gltfMat.emissiveStrength; // KHR_materials_emissive_strength
mat.clearcoat                = gltfMat.clearcoatFactor;   // KHR_materials_clearcoat
mat.clearcoat_roughness      = gltfMat.clearcoatRoughnessFactor;
mat.sheen                    = gltfMat.sheenRoughnessFactor; // KHR_materials_sheen
// No compilation, no code generation, no external SDK.
```

---

## Integration with Path Tracer

### OptiX Closest Hit (replaces MDL calls)

```cuda
extern "C" __global__ void __closesthit__radiance()
{
    // ... geometry interpolation (unchanged) ...

    SurfaceInteraction si;
    si.position   = surfaceHit.position;
    si.normal     = surfaceHit.normal;
    si.geom_normal = surfaceHit.geom_normal;
    si.tangent    = surfaceHit.worldTangent[0];
    si.bitangent  = surfaceHit.worldBinormal[0];
    si.uv         = make_float2(surfaceHit.text_coords[0].x, surfaceHit.text_coords[0].y);
    si.wo         = -ray_dir;
    si.front_face = !prd->inside;

    // Load material from GPU buffer (replaces MDL arg block + function pointers)
    const MaterialParams& mat = params.scene.materials[hit_data->materialId];
    MaterialParams resolved;
    bsdf_init(mat, si, resolved);

    // Sample BSDF (replaces mdlcode_sample)
    float4 xi = make_float4(z1, z2, z3, z4);
    BsdfSampleResult sample = bsdf_sample(resolved, si, xi);

    if (sample.event_type == BSDF_EVENT_ABSORB) {
        prd->throughput = make_float3(0);
        return;
    }

    prd->specularBounce = (sample.event_type & BSDF_EVENT_SPECULAR) != 0;

    // NEE for diffuse/glossy (replaces mdlcode_evaluate)
    if (sample.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) {
        float3 toLight;
        float lightPdf;
        float3 Li = estimateDirectLighting(prd->sampler, si, toLight, lightPdf);
        if (lightPdf > 0.0f) {
            BsdfEvalResult eval = bsdf_eval(resolved, si, toLight);
            if (eval.pdf > 0.0f) {
                float mis = misWeightBalance(lightPdf, eval.pdf);
                prd->radiance += prd->throughput * (Li / lightPdf) * mis
                               * (eval.diffuse + eval.specular);
            }
        }
    }

    // Continue path
    if (sample.event_type & BSDF_EVENT_TRANSMISSION) {
        prd->inside = !prd->inside;
        prd->origin = offset_ray(si.position, -si.geom_normal);
    } else {
        prd->origin = offset_ray(si.position, si.geom_normal);
    }
    prd->lastBsdfPdf = prd->specularBounce ? 1.0f : sample.pdf;
    prd->dir = sample.wi;
    prd->throughput *= sample.bsdf_over_pdf;
}
```

### Metal Kernel (replaces hardcoded Lambert)

Same logic, same headers, same function calls. Only `texture_sample.h` differs.

---

## Editor Integration

The `MaterialParams` struct is trivially inspectable. No reflection API needed.

```cpp
void MaterialPanel::draw(MaterialParams& mat)
{
    const char* types[] = {"Diffuse", "Conductor", "Dielectric", "Standard PBR", "Emissive"};
    ImGui::Combo("Type", (int*)&mat.type, types, IM_ARRAYSIZE(types));

    ImGui::ColorEdit3("Base Color", &mat.base_color.x);
    ImGui::SliderFloat("Roughness", &mat.roughness, 0.0f, 1.0f);

    if (mat.type == MATERIAL_STANDARD_PBR) {
        ImGui::SliderFloat("Metallic",     &mat.metallic,     0.0f, 1.0f);
        ImGui::SliderFloat("IOR",          &mat.ior,          1.0f, 3.0f);
        ImGui::SliderFloat("Transmission", &mat.transmission, 0.0f, 1.0f);
        ImGui::SliderFloat("Anisotropy",   &mat.anisotropy,  -1.0f, 1.0f);
        ImGui::SliderFloat("Clearcoat",    &mat.clearcoat,    0.0f, 1.0f);
        ImGui::SliderFloat("Sheen",        &mat.sheen,        0.0f, 1.0f);
    }
    if (mat.type == MATERIAL_DIELECTRIC) {
        ImGui::SliderFloat("IOR", &mat.ior, 1.0f, 3.0f);
    }

    ImGui::ColorEdit3("Emission",  &mat.emission_color.x);
    ImGui::DragFloat("Emission Intensity", &mat.emission_intensity, 1.0f, 0.0f, 10000.0f);

    // Texture pickers (file browser or drag-drop)
    // texturePickerWidget("Albedo Map",    &mat.base_color_tex);
    // texturePickerWidget("Normal Map",    &mat.normal_tex);
    // texturePickerWidget("Roughness/Met", &mat.roughness_metallic_tex);
}
```

---

## Dependencies Removed

| Dependency | Was Used For | Replaced By |
|------------|-------------|-------------|
| MDL SDK | Material compilation | `MaterialParams` struct + BxDF headers |
| LLVM 12.0.1 | PTX code generation from MDL | Not needed (material code compiled with shaders) |
| Neuray | MDL runtime loading | Not needed |
| MaterialX | MaterialX-to-MDL translation | Not needed (direct glTF-to-params mapping) |

---

## References

- [GGX NDF + Smith G](https://doi.org/10.2312/EGWR/EGSR07/195-206) - Walter et al. 2007
- [VNDF Sampling](https://doi.org/10.1111/cgf.14867) - Heitz 2018
- [Energy-conserving Lambert](https://seblagarde.wordpress.com/2012/01/08/) - Lagarde
- [glTF PBR spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material)
- [PBRT-v4 material system](https://pbr-book.org/) - reference implementation
