#pragma once
// ============================================================================
// bdpt_common.h -- Shared utilities for BDPT kernels
//
// MIS weight functions following SmallVCM / Georgiev 2012 formulation.
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

#include "ShaderTypes.h"
#include "bdpt_types.h"
#include <strelka/material/bsdf.h>

// ---------------------------------------------------------------------------
// MIS weight computation helpers (partial weights dVCM, dVC)
//
// SmallVCM formulation:
//   dVCM and dVC are propagated along each subpath to allow O(1) MIS weight
//   computation at connection time.
//
// Camera vertex 0 (on lens):
//   dVCM = numPixels / (imagePdf * cosAtCamera)
//   dVC  = 0
//
// Light vertex 0 (on light surface):
//   dVCM = pdf_dir / pdf_pos
//   dVC  = 0
//
// Subsequent vertices i+1:
//   dVCM = (dist^2 / (abs(cos_theta) * pdf_fwd)) * dVC_prev
//   dVC  = (abs(cos_theta_prev) / (dist^2 * pdf_fwd)) * (dVCM_prev + dVC_prev * pdf_rev)
//
// For delta events (specular), set dVCM = 0 (cannot connect to delta vertex).
// ---------------------------------------------------------------------------

// Compute the MIS weight for a connection strategy (s, t)
// using the partial weights stored in the subpath vertices.
//
// wLight = (dVCM_light * numPixels + dVC_light) for the light vertex
// wCamera = (dVCM_camera * numPixels + dVC_camera) for the camera vertex
//
// The MIS weight is: 1 / (wLight + 1 + wCamera)
static inline float bdptMISWeight(float dVCM_camera, float dVC_camera,
                                   float dVCM_light, float dVC_light,
                                   float pdf_connect_camera, float pdf_connect_light,
                                   float numPixels)
{
    // For general connection (s >= 2, t >= 2):
    // wCamera captures all strategies that could have generated the camera subpath
    // wLight captures all strategies that could have generated the light subpath
    float wCamera = dVCM_camera * pdf_connect_light + dVC_camera * pdf_connect_light * pdf_connect_camera;
    float wLight  = dVCM_light * pdf_connect_camera + dVC_light * pdf_connect_camera * pdf_connect_light;

    return 1.0f / (wLight + 1.0f + wCamera);
}

// Simplified MIS weight using the balance heuristic
// For (s=0,t>=2): camera path hits light directly
static inline float bdptMISWeightDirect(float bsdfPdf, float lightPdf)
{
    if (bsdfPdf <= 0.0f) return 1.0f;
    return bsdfPdf / (bsdfPdf + lightPdf);
}

// For (s=1,t>=1): NEE - sample a point on light, connect to camera vertex
static inline float bdptMISWeightNEE(float lightPdf, float bsdfPdf)
{
    if (lightPdf <= 0.0f) return 0.0f;
    return lightPdf / (lightPdf + bsdfPdf);
}

// ---------------------------------------------------------------------------
// Geometry term helpers
// ---------------------------------------------------------------------------
static inline float geometryTerm(float3 p1, float3 n1, float3 p2, float3 n2)
{
    float3 d = p2 - p1;
    float dist2 = dot(d, d);
    if (dist2 < 1e-12f) return 0.0f;
    d = d / sqrt(dist2);
    float g = fabs(dot(n1, d)) * fabs(dot(n2, -d)) / dist2;
    return g;
}

static inline float distanceSquared(float3 a, float3 b)
{
    float3 d = b - a;
    return dot(d, d);
}

// ---------------------------------------------------------------------------
// Convert between area and solid angle PDFs
// ---------------------------------------------------------------------------
static inline float pdfAreaToSolidAngle(float pdfArea, float dist2, float cosTheta)
{
    if (cosTheta <= 0.0f) return 0.0f;
    return pdfArea * dist2 / cosTheta;
}

static inline float pdfSolidAngleToArea(float pdfSolidAngle, float dist2, float cosTheta)
{
    if (dist2 <= 0.0f) return 0.0f;
    return pdfSolidAngle * cosTheta / dist2;
}

// ---------------------------------------------------------------------------
// Reconstruct a minimal SurfaceInteraction from BDPTVertex for BSDF evaluation
// ---------------------------------------------------------------------------
static SurfaceInteraction vertexToSI(device const BDPTVertex& v, device Material* materials)
{
    SurfaceInteraction si;
    si.position        = float3(v.position);
    si.geometry_normal = float3(v.geometry_normal);
    si.shading_normal  = float3(v.shading_normal);
    si.wo              = float3(v.wo);
    si.uv              = float2(v.uv_x, v.uv_y);
    si.front_face      = dot(si.geometry_normal, si.wo) > 0.0f;

    if (v.material_index < 0xFFFFFFFF && !v.is_on_light && !v.is_on_camera)
    {
        const device Material& mat = materials[v.material_index];
        si.albedo     = float3(mat.base_color);
        si.roughness  = max(mat.roughness, 0.0001f);
        si.metallic   = saturate(mat.metallic);
        si.ior        = mat.ior;
        si.transmission = mat.transmission;
        si.clearcoat  = mat.clearcoat;
        si.clearcoat_roughness = max(mat.clearcoat_roughness, 0.0001f);
        si.anisotropy = mat.anisotropy;
        si.specular   = mat.specular;
        si.specular_tint = mat.specular_tint;
        si.material_type = mat.material_type;
        si.thin_walled = mat.thin_walled;
        si.dielectric_priority = mat.dielectric_priority;
        si.exterior_ior = 1.0f;

        constexpr sampler texSampler(mag_filter::linear, min_filter::linear);
        if (!is_null_texture(mat.baseColorTexture))
        {
            float4 texVal = mat.baseColorTexture.sample(texSampler, si.uv);
            si.albedo *= texVal.rgb;
        }
        if (!is_null_texture(mat.metallicRoughnessTexture))
        {
            float4 mrTex = mat.metallicRoughnessTexture.sample(texSampler, si.uv);
            si.roughness = max(si.roughness * mrTex.g, 0.0001f);
            si.metallic = saturate(si.metallic * mrTex.b);
        }
    }
    else
    {
        si.albedo     = float3(1.0f);
        si.roughness  = 1.0f;
        si.metallic   = 0.0f;
        si.ior        = 1.5f;
        si.transmission = 0.0f;
        si.clearcoat  = 0.0f;
        si.clearcoat_roughness = 0.0001f;
        si.anisotropy = 0.0f;
        si.specular   = 0.5f;
        si.specular_tint = 0.0f;
        si.material_type = MATERIAL_TYPE_DIFFUSE;
        si.thin_walled = 0;
        si.dielectric_priority = 0;
        si.exterior_ior = 1.0f;
    }

    float3 T, B;
    build_onb(si.shading_normal, T, B);
    si.tangent   = T;
    si.bitangent = B;
    si.emission  = float3(0.0f);

    return si;
}

// ---------------------------------------------------------------------------
// VCM merge MIS weight (SmallVCM / Georgiev 2012)
// ---------------------------------------------------------------------------
static inline float vcmMergeMISWeight(
    float cameraDVCM, float cameraDVM,
    float lightDVCM,  float lightDVM,
    float cameraBsdfFwdPdf, float cameraBsdfRevPdf,
    float vcWeightFactor)
{
    float wLight  = lightDVCM  * vcWeightFactor + lightDVM  * cameraBsdfFwdPdf;
    float wCamera = cameraDVCM * vcWeightFactor + cameraDVM * cameraBsdfRevPdf;
    return 1.0f / (wLight + 1.0f + wCamera + 1e-10f);
}
