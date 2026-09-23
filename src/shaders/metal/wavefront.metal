#include "shading_common.h"

#define STRELKA_OPENPBR_FEATURE_EnableSheenAndCoat SPEC_OPENPBR_SHEEN_AND_COAT
#define STRELKA_OPENPBR_FEATURE_EnableDispersion SPEC_OPENPBR_DISPERSION
#define STRELKA_OPENPBR_FEATURE_EnableTranslucency SPEC_OPENPBR_TRANSLUCENCY
#define STRELKA_OPENPBR_FEATURE_EnableMetallic SPEC_OPENPBR_METALLIC
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) STRELKA_OPENPBR_FEATURE_##name
#include <strelka/material/openpbr/openpbr_bridge.h>
#undef OPENPBR_GET_SPECIALIZATION_CONSTANT
#undef STRELKA_OPENPBR_FEATURE_EnableMetallic
#undef STRELKA_OPENPBR_FEATURE_EnableTranslucency
#undef STRELKA_OPENPBR_FEATURE_EnableDispersion
#undef STRELKA_OPENPBR_FEATURE_EnableSheenAndCoat
#include <sharc_query_eligibility.h>

// ShadeBase never reaches coat, fuzz, transmission, thin-film or subsurface
// lobes. Keep the common 272-byte device block out of thread memory and retain
// only the fields that feed its base/specular preparation.
static __attribute__((always_inline)) void loadOpenPBRBaseParams(device const OpenPBRParams& p,
                                                                 thread OpenPBR_BaseParams& base)
{
    base.base_color = p.base_color;
    base.base_weight = p.base_weight;
    base.base_diffuse_roughness = p.base_diffuse_roughness;
    base.base_metalness = p.base_metalness;
    base.specular_weight = p.specular_weight;
    base.specular_roughness = p.specular_roughness;
    base.specular_color = p.specular_color;
    base.specular_roughness_anisotropy = p.specular_roughness_anisotropy;
    base.specular_ior = p.specular_ior;
    base.specular_anisotropy_rotation_cos = p.specular_anisotropy_rotation_cos;
    base.specular_anisotropy_rotation_sin = p.specular_anisotropy_rotation_sin;
}

// Texture counterpart of loadOpenPBRBaseParams. The source block stays in
// device memory; emission and the normal map update SurfaceInteraction at the
// same point as the generic path.
static void applyOpenPBRBaseTextures(thread OpenPBR_BaseParams& p,
                                     device const OpenPBRParams& source,
                                     device const OpenPBRTextures& t,
                                     thread SurfaceInteraction& si,
                                     float2 uv,
                                     float lodBase = -1e30f,
                                     bool uvPretransformed = false,
                                     float bumpUvFootprint = -1.0f,
                                     float bumpWorldFootprint = -1.0f)
{
    const float2 tuv = uvPretransformed ? uv :
                                          applyOpenPBRTextureTransform(uv, source.uv_rotation,
                                                                       float2(source.uv_scale_x, source.uv_scale_y),
                                                                       float2(source.uv_offset_x, source.uv_offset_y));
    const bool hasLod = lodBase > -1e29f;
    constexpr sampler openpbrSampler(mag_filter::linear, min_filter::linear, mip_filter::linear, address::repeat);
#define SAMPLE_OPENPBR_BASE_TEXTURE(slot)                                                                              \
    t.tex[slot].sample(openpbrSampler, tuv, level(texLod(t.tex[slot], lodBase, hasLod)))

    const uint32_t mask = source.texture_mask;
    if (t.layered.output_mask != 0u)
    {
        OpenPBRParams layered = source;
        applyOpenPBRLayeredTexture(layered, t, si, uv, lodBase, bumpUvFootprint, bumpWorldFootprint);
        p.base_color = layered.base_color;
        p.specular_color = layered.specular_color;
        p.specular_roughness = layered.specular_roughness;
    }
    if ((mask & (1u << OPENPBR_TEX_BASE_COLOR)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_BASE_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_BASE_COLOR).rgb;
        p.base_color = (source.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u ?
                           OpenPBRColor{ p.base_color.r * v.r, p.base_color.g * v.g, p.base_color.b * v.b } :
                           OpenPBRColor{ v.r, v.g, v.b };
    }
    if ((mask & (1u << OPENPBR_TEX_BASE_METALNESS)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_BASE_METALNESS]))
    {
        p.base_metalness = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_BASE_METALNESS).r;
    }
    if ((mask & (1u << OPENPBR_TEX_SPECULAR_ROUGHNESS)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_ROUGHNESS]))
    {
        const float4 v = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_SPECULAR_ROUGHNESS);
        const float roughness = v[source.texture_scalar_flags & OPENPBR_ROUGHNESS_CHANNEL_MASK];
        p.specular_roughness = (source.texture_scalar_flags & OPENPBR_ROUGHNESS_MULTIPLY) != 0u ?
                                   p.specular_roughness * roughness :
                                   roughness;
    }
    if ((mask & (1u << OPENPBR_TEX_SPECULAR_ANISOTROPY)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_ANISOTROPY]))
    {
        p.specular_roughness_anisotropy = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_SPECULAR_ANISOTROPY).r;
    }
    if ((mask & (1u << OPENPBR_TEX_SPECULAR_COLOR)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_SPECULAR_COLOR]))
    {
        const float3 v = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_SPECULAR_COLOR).rgb;
        p.specular_color = OpenPBRColor{ v.r, v.g, v.b };
    }
    if ((mask & (1u << OPENPBR_TEX_EMISSION_COLOR)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_EMISSION_COLOR]))
    {
        constexpr sampler emissionSampler(mag_filter::linear, min_filter::linear, address::repeat);
        const float3 v = t.tex[OPENPBR_TEX_EMISSION_COLOR].sample(emissionSampler, tuv).rgb;
        const float3 tint = (source.texture_scalar_flags & OPENPBR_TEXTURES_GLTF) != 0u ?
                                float3(source.emission_color.r, source.emission_color.g, source.emission_color.b) * v :
                                v;
        si.emission = tint * source.emission_luminance;
    }
    if ((mask & (1u << OPENPBR_TEX_GEOMETRY_NORMAL)) != 0u && !is_null_texture(t.tex[OPENPBR_TEX_GEOMETRY_NORMAL]))
    {
        const float2 xy = SAMPLE_OPENPBR_BASE_TEXTURE(OPENPBR_TEX_GEOMETRY_NORMAL).xy * 2.0f - 1.0f;
        const float z = sqrt(saturate(1.0f - dot(xy, xy)));
        const float3x3 TBN = float3x3(si.tangent, si.bitangent, si.shading_normal);
        si.shading_normal = normalize(TBN * float3(xy * source.texture_normal_scale, z));
        si.bump_normal = si.shading_normal;
        if (dot(si.shading_normal, si.wo) <= 0.0f)
        {
            const float3 facingGeom = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            si.shading_normal = ensureValidSpecularReflection(facingGeom, si.wo, si.shading_normal);
            si.diffuse_faces_away = true;
        }
    }
#undef SAMPLE_OPENPBR_BASE_TEXTURE
}

// Path state stays pixel-indexed while queues compact live indices between stages.
// Shadow traversal is deferred so its incoherent work stays out of shade.

#include "fog.h"
#include "subsurface.h"
#include "sharc.h"

#define HIT_FOG_BIT (1u << 30)

// Bit 29 marks a scattering event inside a subsurface medium: like the fog bit,
// the ray never reached a surface, but the medium is the one bounded by the
// object the path is currently inside rather than the atmosphere.
#define HIT_SSS_BIT (1u << 29)

// The walk's albedo, packed to one word. Eight bits a channel is more precision
// than a scattering albedo carries meaning at, and this rides on every live path.
static inline uint32_t packMediumAlbedo(float3 a)
{
    return (uint32_t)(saturate(a.x) * 255.0f + 0.5f) | ((uint32_t)(saturate(a.y) * 255.0f + 0.5f) << 8) |
           ((uint32_t)(saturate(a.z) * 255.0f + 0.5f) << 16);
}

static inline float3 unpackMediumAlbedo(uint32_t v)
{
    constexpr float s = 1.0f / 255.0f;
    return float3((v & 0xffu) * s, ((v >> 8) & 0xffu) * s, ((v >> 16) & 0xffu) * s);
}

// Bit 31 of HitRecord::geomEntryIndex marks a hit on emissive geometry, in which
// case the remaining bits hold the light index rather than a geometry entry.
#define HIT_LIGHT_BIT 0x80000000u

struct AnalyticPrimitiveIntersection
{
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

#define WF_ANALYTIC_INTERSECTION_ENTRY(NAME, BODY, ...)                                                                \
    [[intersection(bounding_box, __VA_ARGS__)]]                                                                        \
    AnalyticPrimitiveIntersection NAME(float3 origin [[origin]], float3 direction [[direction]],                       \
                                       float minDistance [[min_distance]], float maxDistance [[max_distance]])         \
    {                                                                                                                  \
        const CanonicalAnalyticIntersection hit = BODY(origin, direction, minDistance, maxDistance);                   \
        return { hit.hit, hit.distance };                                                                              \
    }

#define WF_ANALYTIC_INTERSECTION_FAMILY(FAMILY)                                                                         \
    WF_ANALYTIC_INTERSECTION_ENTRY(                                                                                     \
        analyticSphereIntersection##FAMILY, intersectCanonicalSphere, triangle_data, instancing)                        \
    WF_ANALYTIC_INTERSECTION_ENTRY(analyticDiscIntersection##FAMILY, intersectCanonicalDisc, triangle_data, instancing) \
    WF_ANALYTIC_INTERSECTION_ENTRY(analyticSphereIntersection##FAMILY##Motion, intersectCanonicalSphere,                \
                                   triangle_data, instancing, primitive_motion)                                         \
    WF_ANALYTIC_INTERSECTION_ENTRY(                                                                                     \
        analyticDiscIntersection##FAMILY##Motion, intersectCanonicalDisc, triangle_data, instancing, primitive_motion)  \
    WF_ANALYTIC_INTERSECTION_ENTRY(                                                                                     \
        analyticSphereIntersection##FAMILY##Curve, intersectCanonicalSphere, triangle_data, curve_data, instancing)     \
    WF_ANALYTIC_INTERSECTION_ENTRY(                                                                                     \
        analyticDiscIntersection##FAMILY##Curve, intersectCanonicalDisc, triangle_data, curve_data, instancing)         \
    WF_ANALYTIC_INTERSECTION_ENTRY(analyticSphereIntersection##FAMILY##MotionCurve, intersectCanonicalSphere,           \
                                   triangle_data, curve_data, instancing, primitive_motion)                             \
    WF_ANALYTIC_INTERSECTION_ENTRY(analyticDiscIntersection##FAMILY##MotionCurve, intersectCanonicalDisc,               \
                                   triangle_data, curve_data, instancing, primitive_motion)

// Keep traversal pipelines' exported intersection symbols distinct for Xcode
// Metal capture replay compatibility.
WF_ANALYTIC_INTERSECTION_FAMILY(Extend)
WF_ANALYTIC_INTERSECTION_FAMILY(Shadow)
WF_ANALYTIC_INTERSECTION_FAMILY(Guide)
WF_ANALYTIC_INTERSECTION_FAMILY(RestirShadeDiagnostic)
WF_ANALYTIC_INTERSECTION_FAMILY(RestirSpatialDiagnostic)

#undef WF_ANALYTIC_INTERSECTION_FAMILY
#undef WF_ANALYTIC_INTERSECTION_ENTRY

#define WF_CTRL_COUNT0 0
#define WF_CTRL_COUNT1 1
#define WF_CTRL_DISPATCH 2
#define WF_CTRL_ACTIVE 5
#define WF_CTRL_SHADOW 6
#define WF_CTRL_SHADOW_N 7
#define WF_CTRL_SHADOW_DIS 8
#define WF_CTRL_HIT 11
#define WF_CTRL_HIT_N 12
#define WF_CTRL_HIT_DIS 13
#define WF_CTRL_MISS 16
#define WF_CTRL_MISS_N 17
#define WF_CTRL_MISS_DIS 18
#define WF_CTRL_GUIDE 21
#define WF_CTRL_GUIDE_N 22
#define WF_CTRL_CAPACITY 24
#define WF_CTRL_GUIDE_DIS 25
#define WF_CTRL_RESTIR 28
#define WF_CTRL_RESTIR_N 28
#define WF_CTRL_RESTIR_DIS 29
#define WF_CTRL_SHADE_BASE_DIS 80
#define WF_CTRL_SHADE_LAYER_START 83
#define WF_CTRL_SHADE_LAYER_DIS 84
#define WF_CTRL_SHADE_TRANSLUCENT_START 87
#define WF_CTRL_SHADE_TRANSLUCENT_DIS 88
#define WF_CTRL_SHADE_TAIL_START 91
#define WF_CTRL_SHADE_TAIL_DIS 92
// Profiling only: live path count and shadow ray count per bounce, so the
// per-stage timings can be read as a cost per ray rather than a cost per stage.
#define WF_CTRL_STATS_PATHS 32
#define WF_CTRL_STATS_SHADOW 64
#define WF_HIT_BUCKET_COUNT 4u
#define WF_SHADE_BASE 0u
#define WF_SHADE_LAYER 1u
#define WF_SHADE_TRANSLUCENT 2u
#define WF_SHADE_TAIL 3u
#define WF_SHADE_GENERIC 4u

// Shared pre-traversal breadcrumbs. The prepare dispatch completes before
// extend starts, so these survive even when one of the first few rays wedges in
// the ray tracing unit and the command buffer is terminated by the watchdog.
#define WF_DIAG_BOUNCES 96
#define WF_DIAG_LANES 4
#define WF_DIAG_LANE_WORDS 11
#define WF_DIAG_STRIDE (2 + WF_DIAG_LANES * WF_DIAG_LANE_WORDS)
#define WF_STAGE_BREADCRUMB 96
#define WF_DIAG_BASE 97

static inline void auditWork(constant Uniforms& uniforms, uint32_t counter, uint32_t amount = 1u)
{
    if (SPEC_RENDER_WORK_AUDIT)
    {
        atomic_fetch_add_explicit(&uniforms.renderWorkCounters[counter], amount, memory_order_relaxed);
    }
}

static inline void auditExtendWork(constant Uniforms& uniforms, uint32_t bounce)
{
    if (SPEC_RENDER_WORK_AUDIT)
    {
        const uint32_t rays = simd_sum(1u);
        if (simd_is_first())
        {
            auditWork(uniforms, WORK_EXTEND_RAYS_BASE + min(bounce, WORK_BOUNCE_SLOTS - 1u), rays);
            auditWork(uniforms, WORK_INTERSECTION_QUERIES, rays);
            auditWork(uniforms, WORK_EXTENSION_QUERIES, rays);
        }
    }
}

static inline void addFilteredRadiance(device float4* radianceOut, uint32_t pixelIndex, float3 radiance)
{
    // The signed reconstruction weight is applied once in wavefrontResolve.
    radianceOut[pixelIndex] += float4(radiance, 0.0f);
}

static inline device RestirDiagnosticRecord* restirDiagnosticRecord(constant Uniforms& uniforms, uint32_t pixelIndex)
{
    if (!SPEC_RENDER_WORK_AUDIT || uniforms.renderWorkCounters == nullptr)
    {
        return nullptr;
    }
    device uint32_t* words = (device uint32_t*)uniforms.renderWorkCounters;
    device uint32_t* pixels = words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS;
    device RestirDiagnosticRecord* records = (device RestirDiagnosticRecord*)(pixels + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
    for (uint32_t i = 0u; i < RESTIR_DIAGNOSTIC_PIXEL_COUNT; ++i)
    {
        if (pixels[i] == pixelIndex)
        {
            return &records[i];
        }
    }
    return nullptr;
}

static inline device RestirCandidateAuditRecord* restirCandidateAuditRecord(constant Uniforms& uniforms,
                                                                            uint32_t pixelIndex)
{
    device RestirDiagnosticRecord* diagnostic = restirDiagnosticRecord(uniforms, pixelIndex);
    if (diagnostic == nullptr)
        return nullptr;
    device uint32_t* words = (device uint32_t*)uniforms.renderWorkCounters;
    device uint32_t* pixels = words + WORK_COUNTER_COUNT + RESTIR_AUDIT_LIGHT_ID_WORDS;
    device RestirDiagnosticRecord* records = (device RestirDiagnosticRecord*)(pixels + RESTIR_DIAGNOSTIC_PIXEL_COUNT);
    const uint32_t slot = uint32_t(diagnostic - records);
    return ((device RestirCandidateAuditRecord*)(records + RESTIR_DIAGNOSTIC_PIXEL_COUNT)) + slot;
}

static inline void auditRestirLightId(constant Uniforms& uniforms, uint32_t lightId)
{
    if (SPEC_RENDER_WORK_AUDIT && lightId < RESTIR_AUDIT_LIGHT_ID_WORDS * 32u)
    {
        device atomic_uint* words = uniforms.renderWorkCounters + WORK_COUNTER_COUNT;
        atomic_fetch_or_explicit(&words[lightId >> 5u], 1u << (lightId & 31u), memory_order_relaxed);
    }
}

static inline uint32_t restirMHistogramBin(uint32_t M)
{
    return M <= 1u ? 0u : min(32u - clz(M - 1u), RESTIR_AUDIT_M_BINS - 1u);
}

static inline uint32_t restirTargetRatioHistogramBin(float ratio)
{
    return uint32_t(clamp(floor(log2(ratio)) + 4.0f, 0.0f, float(RESTIR_AUDIT_TARGET_RATIO_BINS - 1u)));
}

static inline void restirDiagnosticVisibility(constant Uniforms& uniforms, uint32_t pixelIndex, float3 contribution)
{
    device RestirDiagnosticRecord* record = restirDiagnosticRecord(uniforms, pixelIndex);
    if (record != nullptr)
    {
        record->finalVisibility = 1u;
        record->contribution[0] = contribution.x;
        record->contribution[1] = contribution.y;
        record->contribution[2] = contribution.z;
    }
}

static inline void queuePush(constant Uniforms& uniforms,
                             device atomic_uint* counter,
                             device uint32_t* queueOut,
                             uint32_t pathIndex,
                             uint32_t capacity)
{
    const uint32_t rank = simd_prefix_exclusive_sum(1u);
    const uint32_t total = simd_sum(1u);
    uint32_t base = 0u;
    if (simd_is_first())
    {
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
        auditWork(uniforms, WORK_QUEUE_APPENDS, total);
        if (base >= capacity || total > capacity - base)
        {
            auditWork(uniforms, WORK_QUEUE_OVERFLOWS, base >= capacity ? total : total - (capacity - base));
        }
    }
    base = simd_broadcast_first(base);
    if (base < capacity && rank < capacity - base)
    {
        queueOut[base + rank] = pathIndex;
    }
}

// Static and motion acceleration structures require different intersector types,
// so traversal specialization cannot be a runtime branch or function constant.

constant float kShadowTransmittanceCutoff = 0.05f;

// Coverage helpers for the inline-query and restart fallbacks.
static inline float cutoutOpacityAt(uint primitive_id,
                                    uint geometry_id,
                                    uint geometry_entry_base,
                                    float2 barycentric_coord,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer)
{
    const GeometryEntry entry = geometryEntries[geometry_entry_base + geometry_id];
    device const Material& mat = materials[entry.materialId];

    if (mat.alpha_mode == ALPHA_MODE_SHADOW_TRANSPARENT)
    {
        return 0.0f;
    }
    if (mat.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f; // blocks outright
    }

    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t uvOff = 20;
    float2 uvv[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitive_id * 3 + k];
        device const char* vtx = vertexBuffer + (entry.vbOffset + idx) * vtxStride;
        uvv[k] = unpackUV(*(device const uint32_t*)(vtx + uvOff), *(device const uint32_t*)(vtx + uvOff + 4u));
    }
    const float2 uv = interpolateAttrib(uvv[0], uvv[1], uvv[2], barycentric_coord);
    return resolveOpacity(mat, uv);
}

static inline float2 decodeInterpolatedCutoutUv(device const PrimitiveAlphaData& primitive,
                                                uint32_t recordIndex,
                                                device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                                float2 barycentricCoord)
{
    const float3 weights = float3(1.0f - barycentricCoord.x - barycentricCoord.y, barycentricCoord);
    const float3 u = unpack_unorm10a2_to_float(primitive.u).xyz;
    const float3 v = unpack_unorm10a2_to_float(primitive.v).xyz;
    const float2 encoded = float2(dot(u, weights), dot(v, weights));
    const float4 decode = float4(primitiveAlphaDecode[recordIndex >> PRIMITIVE_ALPHA_BLOCK_SHIFT].offsetScale);
    return fma(encoded, decode.zw, decode.xy);
}

static inline float2 decodeInterpolatedCutoutUv(uint32_t uv0, uint32_t uv1, uint32_t uv2, float2 barycentricCoord)
{
    const float2 encoded = interpolateAttrib(unpack_unorm2x16_to_float(uv0), unpack_unorm2x16_to_float(uv1),
                                             unpack_unorm2x16_to_float(uv2), barycentricCoord);
    return encoded * 79.998779346f - 10.0f;
}

static inline float cutoutOpacityAtPrimitive(uint primitive_id,
                                             uint geometry_id,
                                             uint geometry_entry_base,
                                             float2 barycentric_coord,
                                             device const Material* materials,
                                             device const PrimitiveAlphaData* primitiveAlphaData,
                                             device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                             device const GeometryEntry* geometryEntries)
{
    const GeometryEntry entry = geometryEntries[geometry_entry_base + geometry_id];
    device const Material& mat = materials[entry.materialId];
    if (mat.alpha_mode == ALPHA_MODE_SHADOW_TRANSPARENT)
    {
        return 0.0f;
    }
    if (mat.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f;
    }
    const uint32_t base = entry.flags & GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK;
    const uint32_t recordIndex = base + primitive_id;
    device const PrimitiveAlphaData& primitive = primitiveAlphaData[recordIndex];
    const float2 uv = decodeInterpolatedCutoutUv(primitive, recordIndex, primitiveAlphaDecode, barycentric_coord);
    return resolveOpacity(mat, uv);
}

static inline float cutoutOpacityAtSelected(uint primitive_id,
                                            uint geometry_id,
                                            uint geometry_entry_base,
                                            float2 barycentric_coord,
                                            device const Material* materials,
                                            device const PrimitiveAlphaData* primitiveAlphaData,
                                            device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                            device const GeometryEntry* geometryEntries,
                                            device const char* vertexBuffer,
                                            device const uint32_t* indexBuffer,
                                            bool usePrimitiveAlphaData)
{
    if (usePrimitiveAlphaData)
    {
        return cutoutOpacityAtPrimitive(primitive_id, geometry_id, geometry_entry_base, barycentric_coord, materials,
                                        primitiveAlphaData, primitiveAlphaDecode, geometryEntries);
    }
    return cutoutOpacityAt(primitive_id, geometry_id, geometry_entry_base, barycentric_coord, materials,
                           geometryEntries, vertexBuffer, indexBuffer);
}

// Static inline traversal only reports geometry descriptors that the builder
// marked non-opaque. Read its alpha state from the dense shadow-walk table: the
// full Material spreads these few fields across a 296-byte shading record.
static inline float resolveKnownCutoutOpacity(device const AlphaMaterialData& material, float2 uv)
{
    if (material.alphaMode == ALPHA_MODE_SHADOW_TRANSPARENT)
    {
        return 0.0f;
    }
    constexpr sampler alphaSampler(mag_filter::linear, min_filter::linear, address::repeat);
    constexpr sampler alphaNearestSampler(mag_filter::nearest, min_filter::nearest, address::repeat);
    float alpha = SPEC_ALPHA_BASE_COLOR_ONE ? 1.0f : material.baseColorAlpha;
    // The feature test is logically redundant with the null handle, but it
    // keeps the texture-dependent values out of the live set on the fallback
    // path. Removing it raised this kernel from 49 to 53 registers on G16S.
    if ((material.features & MATERIAL_TEX_BASE_COLOR) != 0u && !is_null_texture(material.baseColorTexture))
    {
        if (!SPEC_ALPHA_UV_IDENTITY)
        {
            uv = float2(dot(uv, float2(material.uvTransformX)), dot(uv, float2(material.uvTransformY))) +
                 float2(material.uvOffset);
        }
        const float4 texel = SPEC_NEAREST_ALPHA_TEXTURE ? material.baseColorTexture.sample(alphaNearestSampler, uv) :
                                                          material.baseColorTexture.sample(alphaSampler, uv);
        alpha *= (material.features & MATERIAL_FEATURE_OPENPBR_OPACITY_RED) != 0u ? texel.r : texel.a;
    }
    if (!SPEC_ALL_ALPHA_BLEND && material.alphaMode == ALPHA_MODE_MASK)
    {
        return alpha >= material.alphaCutoff ? 1.0f : 0.0f;
    }
    return saturate(alpha);
}

static inline uint32_t cutoutIftHash(uint32_t v)
{
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}

static inline bool cutoutShadowAccept(float2 barycentricCoord,
                                      uint32_t primitiveId,
                                      uint32_t geometryId,
                                      uint32_t geometryEntryBase,
                                      float3 origin,
                                      float3 direction,
                                      GeometryEntry entry,
                                      device const PrimitiveAlphaData& primitive,
                                      uint32_t recordIndex,
                                      device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                      device const AlphaMaterialData* alphaMaterials)
{
    device const AlphaMaterialData& material = alphaMaterials[entry.materialId];
    if (material.alphaMode == ALPHA_MODE_SHADOW_TRANSPARENT)
    {
        return false;
    }
    float2 uv = decodeInterpolatedCutoutUv(primitive, recordIndex, primitiveAlphaDecode, barycentricCoord);
    if ((material.features & MATERIAL_TEX_BASE_COLOR) != 0u && !is_null_texture(material.baseColorTexture))
    {
        if (!SPEC_ALPHA_UV_IDENTITY)
        {
            uv = float2(dot(uv, float2(material.uvTransformX)), dot(uv, float2(material.uvTransformY))) +
                 float2(material.uvOffset);
        }
    }
    constexpr sampler alphaLinearSampler(mag_filter::linear, min_filter::linear, address::repeat);
    constexpr sampler alphaNearestSampler(mag_filter::nearest, min_filter::nearest, address::repeat);
    float opacity = SPEC_ALPHA_BASE_COLOR_ONE ? 1.0f : material.baseColorAlpha;
    if ((material.features & MATERIAL_TEX_BASE_COLOR) != 0u && !is_null_texture(material.baseColorTexture))
    {
        const float4 texel = SPEC_NEAREST_ALPHA_TEXTURE ? material.baseColorTexture.sample(alphaNearestSampler, uv) :
                                                          material.baseColorTexture.sample(alphaLinearSampler, uv);
        opacity *= (material.features & MATERIAL_FEATURE_OPENPBR_OPACITY_RED) != 0u ? texel.r : texel.a;
    }
    if (!SPEC_ALL_ALPHA_BLEND && material.alphaMode == ALPHA_MODE_MASK)
    {
        opacity = opacity >= material.alphaCutoff ? 1.0f : 0.0f;
    }
    opacity = saturate(opacity);
    uint32_t randomBits = primitiveId * 0x9e3779b9u ^ geometryId * 0x85ebca6bu ^ geometryEntryBase * 0xc2b2ae35u;
    randomBits ^= as_type<uint32_t>(origin.x) ^ as_type<uint32_t>(origin.y) ^ as_type<uint32_t>(origin.z);
    randomBits ^= as_type<uint32_t>(direction.x) * 0x27d4eb2du ^ as_type<uint32_t>(direction.y) * 0x165667b1u ^
                  as_type<uint32_t>(direction.z) * 0xd3a2646cu;
    const float threshold = float(cutoutIftHash(randomBits) >> 8u) * (1.0f / 16777216.0f);
    return threshold < opacity;
}

[[intersection(triangle, triangle_data, instancing)]]
bool cutoutShadowIntersection(float2 barycentricCoord [[barycentric_coord]],
                              uint32_t primitiveId [[primitive_id]],
                              uint32_t geometryId [[geometry_id]],
                              uint32_t geometryEntryBase [[user_instance_id]],
                              float3 origin [[origin]],
                              float3 direction [[direction]],
                              device const AlphaMaterialData* alphaMaterials [[buffer(0)]],
                              device const GeometryEntry* geometryEntries [[buffer(1)]],
                              device const PrimitiveAlphaData* primitiveAlphaData [[buffer(2)]],
                              device const PrimitiveAlphaDecode* primitiveAlphaDecode [[buffer(3)]])
{
    const GeometryEntry entry = geometryEntries[geometryEntryBase + geometryId];
    const uint32_t base = entry.flags & GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK;
    const uint32_t recordIndex = base + primitiveId;
    return cutoutShadowAccept(barycentricCoord, primitiveId, geometryId, geometryEntryBase, origin, direction, entry,
                              primitiveAlphaData[recordIndex], recordIndex, primitiveAlphaDecode, alphaMaterials);
}

// Retained as a controlled experiment. Enable STRELKA_ALPHA_IFT_EMBEDDED_UV=1
// to reproduce the AS-resident 12-byte path and measure future Metal versions.
[[intersection(triangle, triangle_data, instancing)]]
bool cutoutShadowIntersectionEmbedded(float2 barycentricCoord [[barycentric_coord]],
                                      uint32_t primitiveId [[primitive_id]],
                                      uint32_t geometryId [[geometry_id]],
                                      uint32_t geometryEntryBase [[user_instance_id]],
                                      float3 origin [[origin]],
                                      float3 direction [[direction]],
                                      device const void* primitiveData [[primitive_data]],
                                      device const AlphaMaterialData* alphaMaterials [[buffer(0)]],
                                      device const GeometryEntry* geometryEntries [[buffer(1)]],
                                      device const PrimitiveAlphaDecode* primitiveAlphaDecode [[buffer(3)]])
{
    if (primitiveData == nullptr)
    {
        return true;
    }
    const GeometryEntry entry = geometryEntries[geometryEntryBase + geometryId];
    const uint32_t recordIndex = (entry.flags & GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK) + primitiveId;
    device const PrimitiveAlphaData& primitive = *(device const PrimitiveAlphaData*)primitiveData;
    return cutoutShadowAccept(barycentricCoord, primitiveId, geometryId, geometryEntryBase, origin, direction, entry,
                              primitive, recordIndex, primitiveAlphaDecode, alphaMaterials);
}

static inline float cutoutOpacityAtCompact(uint primitive_id,
                                           uint geometry_id,
                                           uint geometry_entry_base,
                                           float2 barycentric_coord,
                                           device const AlphaMaterialData* alphaMaterials,
                                           device const GeometryEntry* geometryEntries,
                                           device const char* vertexBuffer,
                                           device const uint32_t* indexBuffer,
                                           bool knownCutout)
{
    const GeometryEntry entry = geometryEntries[geometry_entry_base + geometry_id];
    device const AlphaMaterialData& material = alphaMaterials[entry.materialId];
    if (!knownCutout && material.alphaMode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f;
    }
    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t uvOff = 20;
    float2 uvv[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitive_id * 3 + k];
        device const char* vtx = vertexBuffer + (entry.vbOffset + idx) * vtxStride;
        uvv[k] = unpackUV(*(device const uint32_t*)(vtx + uvOff), *(device const uint32_t*)(vtx + uvOff + 4u));
    }
    const float2 uv = interpolateAttrib(uvv[0], uvv[1], uvv[2], barycentric_coord);
    return resolveKnownCutoutOpacity(material, uv);
}

static inline float cutoutOpacityAtPrimitiveCompact(uint primitive_id,
                                                    uint geometry_id,
                                                    uint geometry_entry_base,
                                                    float2 barycentric_coord,
                                                    device const AlphaMaterialData* alphaMaterials,
                                                    device const PrimitiveAlphaData* primitiveAlphaData,
                                                    device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                                    device const GeometryEntry* geometryEntries,
                                                    bool knownCutout)
{
    const GeometryEntry entry = geometryEntries[geometry_entry_base + geometry_id];
    device const AlphaMaterialData& material = alphaMaterials[entry.materialId];
    if (!knownCutout && material.alphaMode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f;
    }
    const uint32_t base = entry.flags & GEOM_PRIMITIVE_ALPHA_DATA_INDEX_MASK;
    const uint32_t recordIndex = base + primitive_id;
    device const PrimitiveAlphaData& primitive = primitiveAlphaData[recordIndex];
    const float2 uv = decodeInterpolatedCutoutUv(primitive, recordIndex, primitiveAlphaDecode, barycentric_coord);
    return resolveKnownCutoutOpacity(material, uv);
}

static inline float cutoutOpacityAtSelectedCompact(uint primitive_id,
                                                   uint geometry_id,
                                                   uint geometry_entry_base,
                                                   float2 barycentric_coord,
                                                   device const AlphaMaterialData* alphaMaterials,
                                                   device const PrimitiveAlphaData* primitiveAlphaData,
                                                   device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                                                   device const GeometryEntry* geometryEntries,
                                                   device const char* vertexBuffer,
                                                   device const uint32_t* indexBuffer,
                                                   bool usePrimitiveAlphaData,
                                                   bool knownCutout)
{
    if (usePrimitiveAlphaData)
    {
        return cutoutOpacityAtPrimitiveCompact(primitive_id, geometry_id, geometry_entry_base, barycentric_coord,
                                               alphaMaterials, primitiveAlphaData, primitiveAlphaDecode,
                                               geometryEntries, knownCutout);
    }
    return cutoutOpacityAtCompact(primitive_id, geometry_id, geometry_entry_base, barycentric_coord, alphaMaterials,
                                  geometryEntries, vertexBuffer, indexBuffer, knownCutout);
}

struct MotionTraversal
{
    // intersection_query rejects the motion tags, so this one keeps the
    // restart walk.
    enum
    {
        kInlineQuery = 0,
        kHardwareAlpha = 0,
        kDirect = 0
    };
    using structure = acceleration_structure<instancing, primitive_motion>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, instancing, primitive_motion>;
    using volume_isect = isect;
    using table = intersection_function_table<triangle_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static uint32_t instanceId(thread const isect::result_type& r, uint32_t)
    {
        return r.instance_id;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time, table t)
    {
        return i.intersect(r, as, mask, time, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
};

struct StaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, instancing>;
    using query = intersection_query<triangle_data, instancing>;
    enum
    {
        kInlineQuery = 1,
        kHardwareAlpha = 0,
        kDirect = 0
    };
    using volume_isect = isect;
    using table = intersection_function_table<triangle_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static uint32_t instanceId(thread const isect::result_type& r, uint32_t)
    {
        return r.instance_id;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
    static void resetShadowQuery(thread query& q, ray r, structure as, intersection_params params)
    {
        q.reset(r, as, RAY_MASK_SHADOW, params);
    }
    static uint32_t shadowGeometryEntryBase(thread query& q, uint32_t)
    {
        return q.get_candidate_user_instance_id();
    }
};

// Same static TLAS as StaticTraversal, but non-opaque triangles are resolved
// by cutoutShadowIntersection in the RT any-hit path rather than surfaced to an
// inline query in this compute kernel.
struct StaticAlphaIftTraversal
{
    using structure = acceleration_structure<instancing>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, instancing>;
    using volume_isect = isect;
    using table = intersection_function_table<triangle_data, instancing>;
    enum
    {
        kInlineQuery = 0,
        kHardwareAlpha = 1,
        kDirect = 0
    };
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
};

struct DirectStaticTraversal
{
    using query = intersection_query<triangle_data>;
    enum
    {
        kInlineQuery = 1,
        kHardwareAlpha = 0,
        kDirect = 1
    };
    using structure = primitive_acceleration_structure;
    using volume_structure = acceleration_structure<instancing>;
    using isect = intersector<triangle_data>;
    using volume_isect = intersector<triangle_data, instancing>;
    using table = uint32_t;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle;
    }
    static float curveParameter(thread const isect::result_type&)
    {
        return 0.0f;
    }
    static uint32_t instanceId(thread const isect::result_type&, uint32_t directInstanceIndex)
    {
        return directInstanceIndex;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t, float, table)
    {
        return i.intersect(r, as);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, volume_structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
    static void resetShadowQuery(thread query& q, ray r, structure as, intersection_params params)
    {
        q.reset(r, as, params);
    }
    static uint32_t shadowGeometryEntryBase(thread query&, uint32_t directGeometryBase)
    {
        return directGeometryBase;
    }
};

struct CurveMotionTraversal
{
    // intersection_query rejects the motion tags, so this one keeps the
    // restart walk.
    enum
    {
        kInlineQuery = 0,
        kHardwareAlpha = 0,
        kDirect = 0
    };
    using structure = acceleration_structure<instancing, primitive_motion>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, curve_data, instancing, primitive_motion>;
    using volume_isect = intersector<triangle_data, instancing, primitive_motion>;
    using table = intersection_function_table<triangle_data, curve_data, instancing, primitive_motion>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static uint32_t instanceId(thread const isect::result_type& r, uint32_t)
    {
        return r.instance_id;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float time, table t)
    {
        return i.intersect(r, as, mask, time, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float time)
    {
        return i.intersect(r, as, mask, time);
    }
};

struct CurveStaticTraversal
{
    using structure = acceleration_structure<instancing>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, curve_data, instancing>;
    using query = intersection_query<triangle_data, curve_data, instancing>;
    enum
    {
        kInlineQuery = 0,
        kHardwareAlpha = 0,
        kDirect = 0
    };
    using volume_isect = intersector<triangle_data, instancing>;
    using table = intersection_function_table<triangle_data, curve_data, instancing>;
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve | geometry_type::bounding_box;
    }
    static float curveParameter(thread const isect::result_type& r)
    {
        return r.curve_parameter;
    }
    static uint32_t instanceId(thread const isect::result_type& r, uint32_t)
    {
        return r.instance_id;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
};

// Static hair/curve scenes still benefit from hardware alpha on their triangle
// geometry. Curves keep the ordinary analytic intersection functions in slots
// 0/1; cutout triangles use slot 2.
struct CurveStaticAlphaIftTraversal
{
    using structure = acceleration_structure<instancing>;
    using volume_structure = structure;
    using isect = intersector<triangle_data, curve_data, instancing>;
    using volume_isect = intersector<triangle_data, instancing>;
    using table = intersection_function_table<triangle_data, curve_data, instancing>;
    enum
    {
        kInlineQuery = 0,
        kHardwareAlpha = 1,
        kDirect = 0
    };
    static geometry_type geometryTypes()
    {
        return geometry_type::triangle | geometry_type::curve | geometry_type::bounding_box;
    }
    static isect::result_type trace(thread isect& i, ray r, structure as, uint32_t mask, float, table t)
    {
        return i.intersect(r, as, mask, t);
    }
    static volume_isect::result_type traceVolume(thread volume_isect& i, ray r, structure as, uint32_t mask, float)
    {
        return i.intersect(r, as, mask);
    }
};

struct ExtendIntersection
{
    intersection_type type;
    uint32_t instanceId;
    uint32_t geometryId;
    uint32_t primitiveId;
    float distance;
    float2 barycentrics;
    float curveParameter;
    device const void* primitiveData;
};

static void fetchTriangleBlended(device const char* vertexBuffer,
                                 device const char* prevVertexBuffer,
                                 device const uint32_t* indexBuffer,
                                 GeometryEntry entry,
                                 uint32_t primitiveId,
                                 bool interpolateMotion,
                                 float motionTime,
                                 float2 bary,
                                 thread float3& outNormal,
                                 thread float3& outTangent,
                                 thread float2& outUv,
                                 thread float3& outColor,
                                 thread float& tangentSign,
                                 thread float3& outGeomNormal,
                                 thread float& outUvArea2);

static inline uint32_t packSurfaceNormal(float3 normal)
{
    const float l1 = max(abs(normal.x) + abs(normal.y) + abs(normal.z), 1e-20f);
    const float3 n = normal / l1;
    const float2 signNotZero = float2(n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f);
    const float2 oct = n.z >= 0.0f ? n.xy : (1.0f - abs(n.yx)) * signNotZero;
    return pack_float_to_snorm2x16(oct);
}

static inline uint32_t packSurfaceTangent(float3 tangent)
{
    const uint32_t packed16 = packSurfaceNormal(tangent);
    constexpr uint32_t mask = 0x7fffu;
    return ((packed16 >> 1u) & mask) | (((packed16 >> 17u) & mask) << 15u);
}

static inline uint32_t packSurfaceUv(float2 uv, uint32_t lod)
{
    const uint32_t packed16 = pack_float_to_unorm2x16(fract(uv));
    constexpr uint32_t mask = 0x1fffu;
    const uint32_t packed13 = ((packed16 >> 3u) & mask) | (((packed16 >> 19u) & mask) << 13u);
    return packed13 | (lod << 26u);
}

static inline uint32_t packSurfaceLod(float lodBase)
{
    // Code zero is the no-LOD sentinel. A non-negative base always reaches the
    // texture's last mip after texLod adds log2(texture size), while -15.5 is
    // already below the first mip for every Metal texture size we support.
    if (!(lodBase > -1e29f))
    {
        return 0u;
    }
    const uint32_t quantized = uint32_t(round((clamp(lodBase, -15.5f, 0.0f) + 15.5f) * 4.0f));
    return quantized + 1u;
}

static inline void packSurfaceGeometry(float3 shadingNormal,
                                       float3 geometryNormal,
                                       float3 tangent,
                                       float2 uv,
                                       float tangentSign,
                                       float lodBase,
                                       thread SurfaceGeometryPayload& payload)
{
    const uint32_t lod = packSurfaceLod(lodBase);
    payload.shadingNormal = packSurfaceNormal(shadingNormal);
    payload.geometryNormal = packSurfaceNormal(geometryNormal);
    payload.tangentAndFlags = packSurfaceTangent(tangent) | SURFACE_GEOMETRY_VALID |
                              (tangentSign < 0.0f ? SURFACE_GEOMETRY_TANGENT_NEGATIVE : 0u);
    payload.uvAndLod = packSurfaceUv(uv, lod);
}

static inline float3 unpackSurfaceNormal(uint32_t packed)
{
    const float2 oct = unpack_snorm2x16_to_float(packed);
    float3 normal = float3(oct, 1.0f - abs(oct.x) - abs(oct.y));
    if (normal.z < 0.0f)
    {
        const float2 signNotZero = float2(normal.x >= 0.0f ? 1.0f : -1.0f, normal.y >= 0.0f ? 1.0f : -1.0f);
        normal.xy = (1.0f - abs(normal.yx)) * signNotZero;
    }
    return normalize(normal);
}

static inline float3 unpackSurfaceTangent(uint32_t tangentAndFlags)
{
    const uint32_t packed = tangentAndFlags & 0x3fffffffu;
    const uint32_t expanded = ((packed & 0x7fffu) << 1u) | (((packed >> 15u) & 0x7fffu) << 17u);
    return unpackSurfaceNormal(expanded);
}

static inline float2 unpackSurfaceUv(uint32_t uvAndLod)
{
    const uint32_t packed = uvAndLod & 0x03ffffffu;
    const uint32_t expanded = ((packed & 0x1fffu) << 3u) | (((packed >> 13u) & 0x1fffu) << 19u);
    return unpack_unorm2x16_to_float(expanded);
}

static inline float unpackSurfaceTangentSign(uint32_t tangentAndFlags)
{
    return (tangentAndFlags & SURFACE_GEOMETRY_TANGENT_NEGATIVE) != 0u ? -1.0f : 1.0f;
}

static inline float unpackSurfaceLod(uint32_t uvAndLod)
{
    const uint32_t code = uvAndLod >> 26u;
    if (code == 0u)
    {
        return -1e30f;
    }
    return float(code - 1u) * 0.25f - 15.5f;
}

static inline float openpbrUvLodOffset(device const OpenPBRParams& material)
{
    const float scale = max(max(abs(material.uv_scale_x), abs(material.uv_scale_y)), 1.0e-8f);
    return log2(scale);
}

static inline float surfaceUvLodOffset(constant Uniforms& uniforms, device const Material* materials, uint32_t materialId)
{
    if (SPEC_ALL_NATIVE_OPENPBR)
    {
        return openpbrUvLodOffset(uniforms.openpbrParams[materialId]);
    }

    device const Material& material = materials[materialId];
    const bool nativeOpenPBR = SPEC_OPENPBR && material.material_type == MATERIAL_TYPE_OPENPBR &&
                               (material.features & MATERIAL_FEATURE_NATIVE_OPENPBR) != 0u;
    if (nativeOpenPBR)
    {
        return openpbrUvLodOffset(uniforms.openpbrParams[materialId]);
    }

    const float2 columnX = float2(material.uv_transform_x.x, material.uv_transform_y.x);
    const float2 columnY = float2(material.uv_transform_x.y, material.uv_transform_y.y);
    const float squaredScale = max(max(dot(columnX, columnX), dot(columnY, columnY)), 1.0e-16f);
    return 0.5f * log2(squaredScale);
}

// Match the camera-ray footprint used by OptiX.  A perspective cone narrows
// towards the edge of the image because the unnormalised film ray gets longer;
// an orthographic footprint is constant and must not grow with hit distance.
static inline float surfaceRayFootprint(constant Uniforms& uniforms, float3 rayDirection, float distance, bool primaryRay)
{
    const float width = float(max(uniforms.width, 1u));
    const float height = float(max(uniforms.height, 1u));
    const float3 cameraRight = uniforms.viewToWorld[0].xyz;
    const float3 cameraUp = uniforms.viewToWorld[1].xyz;

    if (primaryRay && uniforms.projectionType == PROJECTION_ORTHOGRAPHIC)
    {
        const float3 dOdx = cameraRight * (2.0f * uniforms.orthoHalfWidth / width);
        const float3 dOdy = cameraUp * (2.0f * uniforms.orthoHalfHeight / height);
        return 0.5f * (length(dOdx) + length(dOdy));
    }

    const float pixelSpread = 2.0f * abs(uniforms.clipToView[1][1]) / height;
    if (!primaryRay)
    {
        return pixelSpread * distance;
    }

    const float3 cameraForward = -uniforms.viewToWorld[2].xyz;
    const float rayNormalization = max(dot(rayDirection, cameraForward), 1.0e-4f);
    const float dxScale = 2.0f * abs(uniforms.clipToView[0][0]) / width;
    const float dyScale = 2.0f * abs(uniforms.clipToView[1][1]) / height;
    const float3 dDdx = (cameraRight - rayDirection * dot(rayDirection, cameraRight)) * (dxScale * rayNormalization);
    const float3 dDdy = (cameraUp - rayDirection * dot(rayDirection, cameraUp)) * (dyScale * rayNormalization);
    return (distance + uniforms.cameraNear) * 0.5f * (length(dDdx) + length(dDdy));
}

template <typename R>
static inline ExtendIntersection captureExtendIntersection(thread const R& r, float curveParameter, uint32_t instanceId)
{
    ExtendIntersection out;
    out.type = r.type;
    out.instanceId = instanceId;
    out.geometryId = 0u;
    out.primitiveId = 0u;
    out.distance = INFINITY;
    out.barycentrics = float2(0.0f);
    out.curveParameter = 0.0f;
    out.primitiveData = nullptr;
    if (r.type != intersection_type::none)
    {
        out.geometryId = r.geometry_id;
        out.primitiveId = r.primitive_id;
        out.distance = r.distance;
        if (r.type == intersection_type::triangle)
        {
            out.barycentrics = r.triangle_barycentric_coord;
            out.primitiveData = r.primitive_data;
        }
        else if (r.type == intersection_type::curve)
        {
            out.curveParameter = curveParameter;
        }
    }
    return out;
}

static inline uint32_t pathDepth(uint32_t depthAndFlags)
{
    return depthAndFlags & PATH_DEPTH_MASK;
}

// The sampler is never stored in the bandwidth-critical path queues. Rebuild it
// once per stage and reuse that register state for alternate depths locally.
static inline uint32_t sampleSequenceIndex(constant Uniforms& uniforms, uint32_t sampleIdx)
{
    const uint32_t sequenceBase = restirSampleSequenceBase(
        uniforms.useFrameJitter != 0u, uniforms.enableAccumulation != 0u, uniforms.restirDIEnabled != 0u,
        uniforms.frameIndex, uniforms.samples_per_launch, uniforms.subframeIndex);
    return SPEC_SHARC_UPDATE ? uniforms.sharcFrameIndex : sequenceBase + sampleIdx;
}

static inline SamplerState samplerFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx, uint32_t depth)
{
    const uint32_t sequenceIndex = sampleSequenceIndex(uniforms, sampleIdx);
    SamplerState s =
        initSampler(pixelIndex, sequenceIndex, uniforms.width, uniforms.widthDivMultiplier, uniforms.widthDivShiftAdd,
                    uniforms.blueNoiseSwitchSpp, uniforms.sobolSampleBlockBits, uniforms.samplerType);
    s.depth = depth;
    return s;
}

// Use only where the returned state is consumed at this depth. SSS/volume
// walkers intentionally retarget one sampler to later dimensions and must keep
// the per-pixel seed even when they start at the primary vertex.
static inline SamplerState samplerForFixedDepth(constant Uniforms& uniforms,
                                                uint32_t pixelIndex,
                                                uint32_t sampleIdx,
                                                uint32_t depth)
{
    const uint32_t sequenceIndex = sampleSequenceIndex(uniforms, sampleIdx);
    const uint32_t samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : uniforms.samplerType;
    const bool blueNoisePrimary =
        depth == 0u && (samplerType == 3u || (samplerType == 4u && sequenceIndex < uniforms.blueNoiseSwitchSpp));
    if (blueNoisePrimary)
    {
        return initPrimaryBlueNoiseSampler(pixelIndex, sequenceIndex, uniforms.width, uniforms.widthDivMultiplier,
                                           uniforms.widthDivShiftAdd, uniforms.blueNoiseSwitchSpp);
    }
    SamplerState s =
        initSampler(pixelIndex, sequenceIndex, uniforms.width, uniforms.widthDivMultiplier, uniforms.widthDivShiftAdd,
                    uniforms.blueNoiseSwitchSpp, uniforms.sobolSampleBlockBits, samplerType);
    s.depth = depth;
    return s;
}

// Camera generation already receives a 2D grid position. Keep it all the way
// into the primary blue-noise lookup instead of recovering x/y with two integer
// divisions from the linear path index.
static inline SamplerState samplerForFixedDepth(
    constant Uniforms& uniforms, uint32_t pixelIndex, uint2 pixel, uint32_t sampleIdx, uint32_t depth)
{
    const uint32_t sequenceIndex = sampleSequenceIndex(uniforms, sampleIdx);
    const uint32_t samplerType = FIXED_SAMPLER_TYPE != 0xffffffffu ? FIXED_SAMPLER_TYPE : uniforms.samplerType;
    const bool blueNoisePrimary =
        depth == 0u && (samplerType == 3u || (samplerType == 4u && sequenceIndex < uniforms.blueNoiseSwitchSpp));
    if (blueNoisePrimary)
    {
        return initPrimaryBlueNoiseSampler(pixel, sequenceIndex, uniforms.blueNoiseSwitchSpp);
    }
    SamplerState s =
        initSampler(pixelIndex, sequenceIndex, uniforms.width, uniforms.widthDivMultiplier, uniforms.widthDivShiftAdd,
                    uniforms.blueNoiseSwitchSpp, uniforms.sobolSampleBlockBits, samplerType);
    s.depth = depth;
    return s;
}

static inline float motionTimeFromSampler(constant Uniforms& uniforms, uint32_t sampleIdx, thread SamplerState& sampler)
{
    if (!SPEC_MOTION_BLUR || !uniforms.enableMotionBlur)
    {
        return 0.0f;
    }
    if (!uniforms.isMotionBlurVisible)
    {
        return 1.0f;
    }
    const uint32_t sampleCount = max(uniforms.samples_per_launch, 1u);
    return ((float)sampleIdx + random<SampleDimension::eTime>(sampler, uniforms.samplerType)) / (float)sampleCount;
}

static inline float motionTimeFor(constant Uniforms& uniforms, uint32_t pixelIndex, uint32_t sampleIdx)
{
    if (!SPEC_MOTION_BLUR || !uniforms.enableMotionBlur)
    {
        return 0.0f;
    }
    if (!uniforms.isMotionBlurVisible)
    {
        return 1.0f;
    }
    SamplerState s = samplerFor(uniforms, pixelIndex, sampleIdx, 0u);
    return motionTimeFromSampler(uniforms, sampleIdx, s);
}

static inline bool shouldWriteAov(constant Uniforms& uniforms, uint32_t sampleIdx)
{
    return SPEC_AOV && !SPEC_SHARC_UPDATE && uniforms.writeAov && sampleIdx == 0u;
}

static inline float backgroundDepth(constant Uniforms& uniforms);

static inline uint32_t sharcUpdateStateIndex(constant Uniforms& uniforms, uint32_t pixelIndex)
{
    const uint32_t scale = max(uniforms.sharcUpdateDownscale, 1u);
    const uint32_t tileWidth = (uniforms.width + scale - 1u) / scale;
    const uint2 pixel = uint2(pixelIndex % uniforms.width, pixelIndex / uniforms.width);
    return (pixel.y / scale) * tileWidth + pixel.x / scale;
}

static inline uint32_t packSharcRoughness(float roughness)
{
    return uint32_t(round(saturate(roughness) * 255.0f)) << PATH_SHARC_ROUGHNESS_SHIFT;
}

static inline float unpackSharcRoughness(uint32_t depthAndFlags)
{
    return float((depthAndFlags & PATH_SHARC_ROUGHNESS_MASK) >> PATH_SHARC_ROUGHNESS_SHIFT) / 255.0f;
}

// Angular bandwidth of the outgoing radiance represented by one SHARC entry.
// Use the narrowest active reflective layer: one sharp coat over a diffuse base
// is still view dependent when all layers share a single cached value.
static inline float sharcReceiverRoughness(bool isOpenPBR,
                                           thread const OpenPBRParams& openpbr,
                                           thread const SurfaceInteraction& si)
{
    float roughness = 1.0f;
    if (isOpenPBR)
    {
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.specular_roughness, openpbr.specular_weight);
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.coat_roughness, openpbr.coat_weight);
        roughness = sharcReceiverLobeRoughness(roughness, openpbr.fuzz_roughness, openpbr.fuzz_weight);
        return roughness;
    }

    // Standard PBR always has a Fresnel reflection lobe. Clearcoat and sheen
    // add their own view-dependent layers only when their weights are nonzero.
    roughness = sharcReceiverLobeRoughness(roughness, si.roughness, 1.0f);
    roughness = sharcReceiverLobeRoughness(roughness, si.clearcoat_roughness, si.clearcoat);
    roughness = sharcReceiverLobeRoughness(roughness, si.sheen_roughness, si.sheen);
    return roughness;
}

static inline float sharcReceiverRoughness(bool isOpenPBR,
                                           thread const OpenPBR_BaseParams& openpbr,
                                           thread const SurfaceInteraction& si)
{
    if (isOpenPBR)
    {
        return sharcReceiverLobeRoughness(1.0f, openpbr.specular_roughness, openpbr.specular_weight);
    }

    float roughness = sharcReceiverLobeRoughness(1.0f, si.roughness, 1.0f);
    roughness = sharcReceiverLobeRoughness(roughness, si.clearcoat_roughness, si.clearcoat);
    return sharcReceiverLobeRoughness(roughness, si.sheen_roughness, si.sheen);
}

// ---------------------------------------------------------------------------
// generate -- camera rays
// ---------------------------------------------------------------------------

struct MediumProps
{
    float3 sigmaT;
    float3 albedo;
    float anisotropy;
};

static float3 mediumSigmaT(constant Uniforms& uniforms, device const Material* materials, uint32_t materialIndex)
{
    device const Material& mm = materials[materialIndex];
    if ((SPEC_ALL_OPENPBR || (SPEC_OPENPBR && mm.material_type == MATERIAL_TYPE_OPENPBR)) &&
        uniforms.openpbrParams != nullptr)
    {
        const OpenPBRParams mat = uniforms.openpbrParams[materialIndex];
        return openpbr_interior_volume(mat).extinction_coefficient;
    }
    return sssSigmaT(float3(mm.subsurface_radius));
}

static MediumProps mediumPropsFor(constant Uniforms& uniforms,
                                  device const Material* materials,
                                  uint32_t materialIndex,
                                  uint32_t packedAlbedo)
{
    device const Material& mm = materials[materialIndex];
    MediumProps out;

    if ((SPEC_ALL_OPENPBR || (SPEC_OPENPBR && mm.material_type == MATERIAL_TYPE_OPENPBR)) &&
        uniforms.openpbrParams != nullptr)
    {
        const OpenPBRParams mat = uniforms.openpbrParams[materialIndex];
        const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(mat);
        out.sigmaT = v.extinction_coefficient;
        // The texture-resolved single-scattering albedo was captured where the
        // path crossed the surface. There is no UV inside the volume with which
        // to reconstruct it from the material table here.
        out.albedo = unpackMediumAlbedo(packedAlbedo);
        out.anisotropy = v.anisotropy;
        return out;
    }

    out.sigmaT = mediumSigmaT(uniforms, materials, materialIndex);
    // A bounded volume has no entry surface to have textured, so it keeps the
    // material's constant; a subsurface walk takes what the boundary resolved.
    out.albedo = ((mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u) ? float3(mm.diffuse_transmission_color) :
                                                                    unpackMediumAlbedo(packedAlbedo);
    out.anisotropy = mm.subsurface_anisotropy;
    return out;
}

struct PrimaryPathData
{
    PathRay ray;
    float motionTime;
};

// Initialise the state consumed after the first intersection. Keeping this in
// one routine lets the ordinary generate pass and the fused static-primary
// traversal share exactly the same sampling and AOV semantics.
static inline PrimaryPathData initializePrimaryPath(uint2 pixel,
                                                    uint32_t pixelIndex,
                                                    constant Uniforms& uniforms,
                                                    constant uint32_t& sampleIdx,
                                                    device PathState* paths,
                                                    device PathRay* rays,
                                                    device float4* radianceOut,
                                                    device AovSample* aov,
                                                    device MediumPathState* mediumPaths)
{
    const uint reconstructionFilter =
        (uniforms.textureLodMode & RECONSTRUCTION_FILTER_MASK) >> RECONSTRUCTION_FILTER_SHIFT;
    if (sampleIdx == 0u)
    {
        radianceOut[pixelIndex] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    if (SPEC_RESTIR && uniforms.restirDIEnabled != 0u && !SPEC_SHARC_UPDATE)
    {
        device RestirReservoir* currentReservoirs =
            (uniforms.frameIndex & 1u) != 0u ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
        device RestirSurfaceHistory* currentHistory =
            (uniforms.frameIndex & 1u) != 0u ? uniforms.restirHistory1 : uniforms.restirHistory0;
        currentReservoirs[pixelIndex] = {};
        currentHistory[pixelIndex] = {};
    }

    SamplerState rng = samplerForFixedDepth(uniforms, pixelIndex, pixel, sampleIdx, 0u);
    const float motionTime = motionTimeFromSampler(uniforms, sampleIdx, rng);

    float3 origin, direction;
    float reconstructionWeight;
    generateCameraRay(pixel, rng, origin, direction, uniforms, motionTime, reconstructionWeight);
    const bool signedReconstruction = reconstructionFilterIsSigned(reconstructionFilter);
    if (signedReconstruction)
    {
        if (sampleIdx == 0u)
        {
            radianceOut[pixelIndex].w = reconstructionWeight;
        }
        else
        {
            float4 encoded = radianceOut[pixelIndex];
            encoded.xyz *= reconstructionBatchScale(encoded.w, reconstructionWeight);
            encoded.w = reconstructionWeight;
            radianceOut[pixelIndex] = encoded;
        }
    }
    if (shouldWriteAov(uniforms, sampleIdx))
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-direction);
        a.roughness = 1.0f;
        a.depth = backgroundDepth(uniforms);
        a.motionX = 0.0f;
        a.motionY = 0.0f;
        a.specularHitDistance = 0.0f;
        // Until a primary surface replaces it, this is background. The AOV
        // resolve derives MetalFX's denoise-strength mask from its depth.
        a.reactive = 1.0f;
        a.guideStateOrBounceDepth = -1.0f;
        aov[pixelIndex] = a;
        uniforms.guideRays[pixelIndex].flags = 0u;
    }

    PathRay r;
    r.origin = packed_float3(origin);
    r.direction = packed_float3(direction);
    rays[pixelIndex] = r;

    PathState p;
    p.throughput = packed_float3(float3(1.0f));
    // The camera segment has no scattering footprint. The specular flag keeps
    // it non-queryable independently; subsequent bounces accumulate their cone
    // width into the packed SHARC roughness.
    p.depthAndFlags = PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | packSharcRoughness(0.0f);
    p.lastBsdfPdf = 0.0f;
    p.misDistance = 0.0f;
    paths[pixelIndex] = p;
    if (SPEC_SSS)
    {
        // Outside every medium. A camera that starts inside a translucent object
        // is not handled -- nothing tells the path which medium it is in.
        MediumPathState mediumState;
        mediumState.medium = 0u;
        mediumState.mediumAlbedo = 0u;
        mediumPaths[pixelIndex] = mediumState;
    }

    PrimaryPathData result;
    result.ray = r;
    result.motionTime = motionTime;
    return result;
}

static inline void initializePrimaryControl(constant Uniforms& uniforms,
                                            constant uint32_t& sampleIdx,
                                            uint32_t pathCount,
                                            device uint32_t* control,
                                            device uint32_t* iorStats)
{
    control[WF_CTRL_COUNT0] = pathCount;
    control[WF_CTRL_COUNT1] = 0u;
    control[WF_CTRL_SHADOW] = 0u;
    control[WF_CTRL_HIT] = 0u;
    control[WF_CTRL_MISS] = 0u;
    if (!SPEC_SHARC_UPDATE && sampleIdx == 0u)
    {
        control[WF_CTRL_GUIDE] = 0u;
    }
    control[WF_CTRL_CAPACITY] = pathCount;
    iorStats[IOR_STAT_OVERFLOW] = 0u;
    iorStats[IOR_STAT_UNMATCHED] = 0u;
    iorStats[IOR_STAT_ESCAPED_INSIDE] = 0u;
    if (!SPEC_SHARC_UPDATE)
    {
        // The primary queue size is known before launch; audit mode needs one
        // counter update, not one contended atomic per camera ray.
        auditWork(uniforms, WORK_PRIMARY_RAYS, pathCount);
    }
}

// Counter-only companion for the fused Metal 4 camera traversal. Keeping this
// separate from wavefrontGenerate avoids running camera RNG for a token lane
// just to publish the known full-frame path count.
kernel void wavefrontInitPrimary(constant Uniforms& uniforms [[buffer(0)]],
                                 constant uint32_t& sampleIdx [[buffer(4)]],
                                 device uint32_t* control [[buffer(6)]],
                                 device uint32_t* iorStats [[buffer(9)]])
{
    initializePrimaryControl(uniforms, sampleIdx, uniforms.width * uniforms.height, control, iorStats);
}

kernel void wavefrontGenerate(uint2 gridPosition [[thread_position_in_grid]],
                              constant Uniforms& uniforms [[buffer(0)]],
                              device PathState* paths [[buffer(1)]],
                              device PathRay* rays [[buffer(8)]],
                              device float4* radianceOut [[buffer(2)]],
                              device IorStack* iorStacks [[buffer(3)]],
                              constant uint32_t& sampleIdx [[buffer(4)]],
                              device uint32_t* queueOut [[buffer(5)]],
                              device uint32_t* control [[buffer(6)]],
                              device AovSample* aov [[buffer(7)]],
                              // Zeroed here rather than on the host: `generate` already owns resetting the
                              // per-sample counters, and a host-side clear would race the frame in flight.
                              // The tally is therefore per sample, which is the rate rather than a total.
                              device uint32_t* iorStats [[buffer(9)]],
                              device SharcUpdateState* sharcUpdates [[buffer(10)]],
                              device MediumPathState* mediumPaths [[buffer(11)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    const uint32_t scale = SPEC_SHARC_UPDATE ? max(uniforms.sharcUpdateDownscale, 1u) : 1u;
    const uint32_t gridWidth = (uniforms.width + scale - 1u) / scale;
    const uint32_t gridHeight = (uniforms.height + scale - 1u) / scale;
    if (gridPosition.x >= gridWidth || gridPosition.y >= gridHeight)
    {
        return;
    }
    const uint32_t tid = gridPosition.y * gridWidth + gridPosition.x;
    const uint32_t pathCount = SPEC_SHARC_UPDATE ? uniforms.sharcUpdatePathCount : pixelCount;
    if (tid == 0u)
    {
        initializePrimaryControl(uniforms, sampleIdx, pathCount, control, iorStats);
    }
    if (tid >= pathCount)
    {
        return;
    }

    uint2 pixel = gridPosition;
    uint32_t pixelIndex = pixel.y * uniforms.width + pixel.x;
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t scramble = sharcHash(tid ^ (uniforms.sharcFrameIndex * 0x9e3779b9u));
        const uint2 offset = uint2(scramble % scale, (scramble / scale) % scale);
        pixel = min(gridPosition * scale + offset, uint2(uniforms.width - 1u, uniforms.height - 1u));
        pixelIndex = pixel.y * uniforms.width + pixel.x;
        SharcUpdateState updateState;
        sharcInitUpdateState(updateState, pixelIndex);
        sharcUpdates[tid] = updateState;
    }

    // Sparse updates preserve the same tile order while queueing the selected
    // full-resolution path slots.
    queueOut[tid] = pixelIndex;
    initializePrimaryPath(pixel, pixelIndex, uniforms, sampleIdx, paths, rays, radianceOut, aov, mediumPaths);

    // The inactive bit in PathState makes stale IOR side-table bytes
    // unreachable. Initialise the table lazily if this path actually enters a
    // solid dielectric instead of streaming 36 bytes for every camera sample.
}

// ---------------------------------------------------------------------------
// extend -- closest hit
// ---------------------------------------------------------------------------
static inline device HitRecord* wavefrontHitRecord(device char* records, uint32_t index)
{
    const size_t stride = SPEC_RESTIR ? sizeof(RestirReservoir) : sizeof(HitRecord);
    return (device HitRecord*)(records + size_t(index) * stride);
}

static inline uint32_t packHitBarycentrics(float2 barycentrics)
{
    return pack_float_to_unorm2x16(saturate(barycentrics));
}

static inline float2 unpackHitBarycentrics(uint32_t barycentrics)
{
    return unpack_unorm2x16_to_float(barycentrics);
}

static inline void bucketPush(constant Uniforms& uniforms,
                              device atomic_uint* counter,
                              device uint32_t* queueOut,
                              uint32_t pathIndex,
                              uint32_t capacity)
{
    const uint32_t rank = simd_prefix_exclusive_sum(1u);
    const uint32_t total = simd_sum(1u);
    uint32_t base = 0u;
    if (simd_is_first())
    {
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
        auditWork(uniforms, WORK_QUEUE_APPENDS, total);
        if (base >= capacity || total > capacity - base)
        {
            auditWork(uniforms, WORK_QUEUE_OVERFLOWS, base >= capacity ? total : total - (capacity - base));
        }
    }
    queueOut[simd_broadcast_first(base) + rank] = pathIndex;
}

static inline void hitQueuePush(
    constant Uniforms& uniforms, device uint32_t* queue, uint32_t pathIndex, uint32_t capacity, uint32_t bucket)
{
    device atomic_uint* counters = (device atomic_uint*)queue;
    device uint32_t* buckets = queue + WF_HIT_BUCKET_COUNT;
    switch (bucket)
    {
    case 0u:
        bucketPush(uniforms, &counters[0], buckets, pathIndex, capacity);
        break;
    case 1u:
        bucketPush(uniforms, &counters[1], buckets + capacity, pathIndex, capacity);
        break;
    case 2u:
        bucketPush(uniforms, &counters[2], buckets + 2u * capacity, pathIndex, capacity);
        break;
    default:
        bucketPush(uniforms, &counters[3], buckets + 3u * capacity, pathIndex, capacity);
        break;
    }
}

template <uint ShadeBucket>
static inline uint32_t bucketedHitIndex(uint32_t index, device const uint32_t* queue, device const uint32_t* control)
{
    const uint32_t capacity = control[WF_CTRL_CAPACITY];

    // Each OpenPBR shade PSO knows which contiguous logical segment it consumes,
    // so none of them needs to walk the preceding bucket counters.
    if (ShadeBucket == WF_SHADE_BASE)
    {
        return WF_HIT_BUCKET_COUNT + index;
    }
    if (ShadeBucket == WF_SHADE_LAYER)
    {
        return WF_HIT_BUCKET_COUNT + capacity + index - control[WF_CTRL_SHADE_LAYER_START];
    }
    if (ShadeBucket == WF_SHADE_TRANSLUCENT)
    {
        return WF_HIT_BUCKET_COUNT + 2u * capacity + index - control[WF_CTRL_SHADE_TRANSLUCENT_START];
    }
    if (ShadeBucket == WF_SHADE_TAIL)
    {
        return WF_HIT_BUCKET_COUNT + 3u * capacity + index - control[WF_CTRL_SHADE_TAIL_START];
    }

    const uint32_t count0 = min(queue[0], capacity);
    if (index < count0)
    {
        return WF_HIT_BUCKET_COUNT + index;
    }
    index -= count0;
    const uint32_t count1 = min(queue[1], capacity);
    if (index < count1)
    {
        return WF_HIT_BUCKET_COUNT + capacity + index;
    }
    index -= count1;
    const uint32_t count2 = min(queue[2], capacity);
    if (index < count2)
    {
        return WF_HIT_BUCKET_COUNT + 2u * capacity + index;
    }
    return WF_HIT_BUCKET_COUNT + 3u * capacity + index - count2;
}

static void enqueueExtendSurfaceResult(constant Uniforms& uniforms,
                                       thread const ExtendIntersection& hit,
                                       device char* hits,
                                       uint32_t tid,
                                       device uint32_t* hitQueue,
                                       device uint32_t* missQueue,
                                       device atomic_uint* missCounter,
                                       device const uint32_t* control,
                                       uint32_t geometryEntryIndex,
                                       uint32_t shadeBucket,
                                       bool forceTail)
{
    if (hit.type == intersection_type::none)
    {
        queuePush(uniforms, missCounter, missQueue, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    HitRecord rec;
    rec.geomEntryIndex = geometryEntryIndex;
    rec.instanceIndex = hit.instanceId;
    rec.primitiveId = hit.primitiveId;
    rec.barycentrics = packHitBarycentrics((hit.type == intersection_type::curve) ? float2(hit.curveParameter, 0.0f) :
                                                                                    hit.barycentrics);
    rec.distance = hit.distance;
    *wavefrontHitRecord(hits, tid) = rec;
    const uint32_t bucket = forceTail ? WF_SHADE_TAIL : shadeBucket;
    hitQueuePush(uniforms, hitQueue, tid, control[WF_CTRL_CAPACITY], bucket);
}

// Compact only dense subsurface paths. Normal paths stay in the original queue:
// extend can reject the same dense predicate cheaply, avoiding a second full-
// size queue while the expensive SSS traversal still runs over coherent lanes.
kernel void wavefrontClassifySss(uint gid [[thread_position_in_grid]],
                                 constant Uniforms& uniforms [[buffer(0)]],
                                 device const uint32_t* queue [[buffer(1)]],
                                 device const uint32_t* control [[buffer(2)]],
                                 device const MediumPathState* mediumPaths [[buffer(3)]],
                                 device const Material* materials [[buffer(4)]],
                                 device uint32_t* sssQueue [[buffer(5)]],
                                 device atomic_uint* sssCounter [[buffer(6)]],
                                 constant uint32_t& queueOffset [[buffer(7)]])
{
    gid += queueOffset;
    if (gid >= control[WF_CTRL_ACTIVE])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    if (tid >= uniforms.width * uniforms.height)
    {
        return;
    }
    const uint32_t medium = mediumPaths[tid].medium & MEDIUM_INDEX_MASK;
    if (medium != 0u && (materials[medium - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u)
    {
        const uint32_t rank = simd_prefix_exclusive_sum(1u);
        const uint32_t total = simd_sum(1u);
        uint32_t base = 0u;
        if (simd_is_first())
        {
            base = atomic_fetch_add_explicit(sssCounter, total, memory_order_relaxed);
        }
        base = simd_broadcast_first(base);
        if (base < control[WF_CTRL_CAPACITY] && rank < control[WF_CTRL_CAPACITY] - base)
        {
            sssQueue[base + rank] = tid;
        }
    }
}

kernel void wavefrontPrepareSss(device uint32_t* sssControl [[buffer(0)]],
                                constant uint32_t& threadsPerGroup [[buffer(1)]],
                                device uint32_t* stageStats [[buffer(2)]],
                                constant uint32_t& bounceIdx [[buffer(3)]],
                                constant uint32_t& diagnosticsEnabled [[buffer(4)]])
{
    const uint32_t n = sssControl[0];
    sssControl[1] = n;
    sssControl[2] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    sssControl[3] = 1u;
    sssControl[4] = 1u;
    if (diagnosticsEnabled != 0u)
    {
        const uint32_t diagBase = WF_DIAG_BASE + min(bounceIdx, WF_DIAG_BOUNCES - 1u) * WF_DIAG_STRIDE;
        stageStats[diagBase + 1u] = n;
    }
}

template <typename T>
static void sssWalkImpl(uint gid,
                        constant Uniforms& uniforms,
                        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                        typename T::structure volumeAccelerationStructure,
                        device PathRay* rays,
                        device PathState* paths,
                        device MediumPathState* mediumPaths,
                        device const Material* materials,
                        constant uint32_t& sampleIdx,
                        device const uint32_t* sssQueue,
                        device const uint32_t* sssControl,
                        device char* hits,
                        device uint32_t* hitQueue,
                        device uint32_t* missQueue,
                        device atomic_uint* missCounter,
                        device uint32_t* queueOut,
                        device atomic_uint* outCounter,
                        device const uint32_t* control,
                        uint32_t rayMask,
                        device const GeometryEntry* geometryEntries)
{
    if (gid >= sssControl[1])
    {
        return;
    }
    const uint32_t tid = sssQueue[gid];
    if (tid >= uniforms.width * uniforms.height)
    {
        return;
    }

    MediumPathState mediumState = mediumPaths[tid];
    const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
    if (medium == 0u || (materials[medium - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
    {
        return;
    }

    PathState p = paths[tid];
    const PathRay pr = rays[tid];
    float3 throughput = float3(p.throughput);
    float3 rayOrigin = float3(pr.origin);
    float3 rayDirection = float3(pr.direction);
    uint32_t step = mediumState.medium >> MEDIUM_STEP_SHIFT;
    const uint32_t maxSteps = min(uniforms.subsurfaceIterations, (uint32_t)MEDIUM_MAX_STEPS);
    const MediumProps mp = mediumPropsFor(uniforms, materials, medium - 1u, mediumState.mediumAlbedo);
    const float anisotropy = materials[medium - 1u].subsurface_anisotropy;
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);
    SamplerState wrng = samplerFor(uniforms, tid, sampleIdx, 0u);

    for (uint32_t fusedStep = 0u; fusedStep < SSS_FUSED_STEPS; ++fusedStep)
    {
        if (step >= maxSteps)
        {
            return;
        }
        const uint32_t depth = pathDepth(p.depthAndFlags);
        auditWork(uniforms, WORK_EXTEND_RAYS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
        auditWork(uniforms, WORK_INTERSECTION_QUERIES);
        auditWork(uniforms, WORK_EXTENSION_QUERIES);

        const float3 channelPdf = sssChannelPdf(throughput, mp.albedo);
        wrng.depth = depth + step;
        const float2 distanceRandom =
            random2<SampleDimension::eSssChannel, SampleDimension::eSssDistance>(wrng, uniforms.samplerType).value;
        float scatterDistance = 0.0f;
        const bool sampledScatter = sssSampleDistance(
            mp.sigmaT, channelPdf, uniforms.sceneExtent, distanceRandom.x, distanceRandom.y, scatterDistance);

        ray sssRay;
        sssRay.min_distance = 1e-6f;
        sssRay.max_distance = sampledScatter ? max(scatterDistance, sssRay.min_distance + 1e-6f) : INFINITY;
        sssRay.origin = rayOrigin;
        sssRay.direction = rayDirection;

        typename T::volume_isect volumeIsect;
        volumeIsect.assume_geometry_type(geometry_type::triangle);
        volumeIsect.force_opacity(forced_opacity::opaque);
        volumeIsect.accept_any_intersection(false);
        const typename T::volume_isect::result_type volumeHit =
            T::traceVolume(volumeIsect, sssRay, volumeAccelerationStructure, rayMask & ~GEOMETRY_MASK_CURVE, motionTime);
        const ExtendIntersection hit = captureExtendIntersection(volumeHit, 0.0f, volumeHit.instance_id);
        const float surfaceDistance = hit.type == intersection_type::none ? sssRay.max_distance : hit.distance;

        if (!sampledScatter || (hit.type != intersection_type::none && scatterDistance >= surfaceDistance))
        {
            PathRay nextRay;
            nextRay.origin = packed_float3(rayOrigin);
            nextRay.direction = packed_float3(rayDirection);
            rays[tid] = nextRay;
            p.throughput = packed_float3(throughput);
            paths[tid] = p;
            mediumState.medium = medium | (step << MEDIUM_STEP_SHIFT);
            mediumPaths[tid] = mediumState;
            // This surface was produced by the fused SSS traversal, not by the
            // regular extend path that prepares compact geometry. Force shade
            // onto its exact vertex-buffer fallback for this uncommon exit.
            uniforms.surfaceGeometry[tid].tangentAndFlags = 0u;
            uint32_t geometryEntryIndex = 0u;
            if (hit.type != intersection_type::none)
            {
                geometryEntryIndex = instances[hit.instanceId].userID + hit.geometryId;
            }
            enqueueExtendSurfaceResult(uniforms, hit, hits, tid, hitQueue, missQueue, missCounter, control,
                                       geometryEntryIndex, WF_SHADE_TAIL, true);
            return;
        }

        auditWork(uniforms, WORK_SHADE_ITEMS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
        throughput *= sssScatterWeight(mp.sigmaT, mp.albedo, channelPdf, scatterDistance);
        const float3 scatterPoint = rayOrigin + rayDirection * scatterDistance;
        float phasePdf = 0.0f;
        const float2 phaseRandom =
            random2<SampleDimension::eSssPhaseU, SampleDimension::eSssPhaseV>(wrng, uniforms.samplerType).value;
        const float3 nextDirection = hgSample(-rayDirection, anisotropy, phaseRandom.x, phaseRandom.y, phasePdf);
        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(wrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;
        rayOrigin = scatterPoint;
        rayDirection = nextDirection;
        ++step;
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;
        p.depthAndFlags =
            depth | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE));
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
    }

    PathRay nextRay;
    nextRay.origin = packed_float3(rayOrigin);
    nextRay.direction = packed_float3(rayDirection);
    rays[tid] = nextRay;
    p.throughput = packed_float3(throughput);
    paths[tid] = p;
    mediumState.medium = medium | (step << MEDIUM_STEP_SHIFT);
    mediumPaths[tid] = mediumState;
    queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
}

#define WF_SSS_WALK_ENTRY(NAME, TRAITS)                                                                                \
    kernel void NAME(uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                  \
                     constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],             \
                     TRAITS::structure volumeAccelerationStructure [[buffer(2)]], device PathRay* rays [[buffer(3)]],  \
                     device PathState* paths [[buffer(4)]], device MediumPathState* mediumPaths [[buffer(5)]],         \
                     device const Material* materials [[buffer(6)]], constant uint32_t& sampleIdx [[buffer(7)]],       \
                     device const uint32_t* sssQueue [[buffer(8)]], device const uint32_t* sssControl [[buffer(9)]],   \
                     device char* hits [[buffer(10)]], device uint32_t* hitQueue [[buffer(11)]],                       \
                     device atomic_uint* hitCounter [[buffer(12)]], device uint32_t* missQueue [[buffer(13)]],         \
                     device atomic_uint* missCounter [[buffer(14)]], device uint32_t* queueOut [[buffer(15)]],         \
                     device atomic_uint* outCounter [[buffer(16)]], device const uint32_t* control [[buffer(17)]],     \
                     constant uint32_t& rayMask [[buffer(18)]],                                                        \
                     device const GeometryEntry* geometryEntries [[buffer(19)]])                                       \
    {                                                                                                                  \
        sssWalkImpl<TRAITS>(gid, uniforms, instances, volumeAccelerationStructure, rays, paths, mediumPaths,           \
                            materials, sampleIdx, sssQueue, sssControl, hits, hitQueue, missQueue, missCounter,        \
                            queueOut, outCounter, control, rayMask, geometryEntries);                                  \
    }

WF_SSS_WALK_ENTRY(wavefrontSssWalk, MotionTraversal)
WF_SSS_WALK_ENTRY(wavefrontSssWalkStatic, StaticTraversal)

static bool storeSurfaceGeometry(constant Uniforms& uniforms,
                                 constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                 device const Material* materials,
                                 device const char* vertexBuffer,
                                 device const char* prevVertexBuffer,
                                 device const uint32_t* indexBuffer,
                                 thread const ExtendIntersection& hit,
                                 uint32_t tid,
                                 float motionTime,
                                 float3 rayDirection,
                                 bool primaryRay,
                                 uint32_t geometryEntryIndex,
                                 GeometryEntry entry,
                                 bool isLight)
{
    if (!SPEC_PREPARE_SURFACE_GEOMETRY_IN_EXTEND)
    {
        return false;
    }
    if (hit.type == intersection_type::none)
    {
        return false;
    }

    if (hit.type != intersection_type::triangle || isLight)
    {
        return false;
    }
    if ((entry.flags & GEOM_FLAG_VERTEX_COLOR) != 0u)
    {
        // Preserve arbitrary COLOR_0 exactly on its rare path instead of
        // charging every surface record another four bytes.
        uniforms.surfaceGeometry[tid].tangentAndFlags = 0u;
        return false;
    }

    SurfaceGeometryPayload payload;
    float lodBase = -1e30f;
    float uvLodOffset = 0.0f;
    const bool interpolateMotion =
        SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer && indexBuffer;
    float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
    float2 uv;
    float tangentSign = 1.0f;
    float uvArea2 = 0.0f;
    const bool hasPrimitiveSurfaceData =
        (entry.flags & GEOM_FLAG_PRIMITIVE_SURFACE_DATA) != 0u && hit.primitiveData != nullptr && !interpolateMotion;
    if (hasPrimitiveSurfaceData)
    {
        device const PrimitiveSurfaceData& primitive = *(device const PrimitiveSurfaceData*)hit.primitiveData;
        const float3 weight = float3(1.0f - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
        objectNormal = float3(0.0f);
        uv = float2(0.0f);
        vertexColor = float3(1.0f);
        for (uint32_t k = 0; k < 3u; ++k)
        {
            objectNormal += unpackNormal(primitive.normal[k]) * weight[k];
        }
        tangentSign = 1.0f;
        objectGeomNormal = unpackNormal(primitive.geometryNormal);
    }
    else
    {
        fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, hit.primitiveId, interpolateMotion,
                             motionTime, hit.barycentrics, objectNormal, objectTangent, uv, vertexColor, tangentSign,
                             objectGeomNormal, uvArea2);
    }

    if ((entry.flags & GEOM_FLAG_SURFACE_UV) != 0u)
    {
        if (SPEC_ALL_NATIVE_OPENPBR)
        {
            device const OpenPBRParams& p = uniforms.openpbrParams[entry.materialId];
            const float2 scale = float2(p.uv_scale_x, p.uv_scale_y);
            uv = applyOpenPBRTextureTransform(uv, p.uv_rotation, scale, float2(p.uv_offset_x, p.uv_offset_y));
        }
        else
        {
            device const Material& material = materials[entry.materialId];
            const bool nativeOpenPBR = SPEC_OPENPBR && material.material_type == MATERIAL_TYPE_OPENPBR &&
                                       (material.features & MATERIAL_FEATURE_NATIVE_OPENPBR) != 0u;
            if (nativeOpenPBR)
            {
                device const OpenPBRParams& p = uniforms.openpbrParams[entry.materialId];
                const float2 scale = float2(p.uv_scale_x, p.uv_scale_y);
                uv = applyOpenPBRTextureTransform(uv, p.uv_rotation, scale, float2(p.uv_offset_x, p.uv_offset_y));
            }
            else
            {
                uv = applyTextureTransform(uv, material);
            }
        }
        if (SPEC_TEXTURE_LOD_CODE && (uniforms.textureLodMode & TEXTURE_LOD_MODE_MASK) != 0u)
        {
            uvLodOffset = surfaceUvLodOffset(uniforms, materials, entry.materialId);
        }
    }

    const float4x4 objectToWorld = geometryObjectToWorld(uniforms, instances, hit.instanceId, geometryEntryIndex, entry);
    const float3 axisX = objectToWorld[0].xyz;
    const float3 axisY = objectToWorld[1].xyz;
    const float3 axisZ = objectToWorld[2].xyz;
    const uint32_t transformIndex = geometryTransformIndex(uniforms, hit.instanceId, geometryEntryIndex, entry);
    const bool uniformOrthogonal = (instances[transformIndex].mask & GEOMETRY_MASK_UNIFORM_ORTHOGONAL_TRANSFORM) != 0u;
    float3 shadingNormal;
    float3 worldGeomNormal;
    float normalOrientation = 1.0f;
    if (uniformOrthogonal)
    {
        shadingNormal = normalize(transformDirection(objectNormal, axisX, axisY, axisZ));
        worldGeomNormal = transformDirection(objectGeomNormal, axisX, axisY, axisZ);
    }
    else
    {
        const FastNormalTransform normalTransform = makeFastNormalTransform(axisX, axisY, axisZ);
        shadingNormal = transformNormalFast(objectNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                            normalTransform.cofactorZ, normalTransform.orientation);
        worldGeomNormal = normalTransform.cofactorX * objectGeomNormal.x +
                          normalTransform.cofactorY * objectGeomNormal.y + normalTransform.cofactorZ * objectGeomNormal.z;
        normalOrientation = normalTransform.orientation;
    }
    const float worldGeomLength = length(worldGeomNormal);
    const float worldArea2 = uniformOrthogonal ? worldGeomLength * sqrt(max(dot(axisX, axisX), 0.0f)) : worldGeomLength;
    if (!(worldArea2 > 1e-20f) || !all(isfinite(shadingNormal)))
    {
        // Tail will reconstruct this uncommon hit from the vertex buffer. It
        // only needs the validity bit cleared; the remaining stale words are
        // never observed.
        uniforms.surfaceGeometry[tid].tangentAndFlags = 0u;
        return false;
    }
    const float3 geometryNormal = normalOrientation * (worldGeomNormal / worldGeomLength);
    float3 tangent;
    if (hasPrimitiveSurfaceData)
    {
        const float3 tangentAxis = abs(shadingNormal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
        tangent = normalize(cross(tangentAxis, shadingNormal));
    }
    else
    {
        tangent = orthonormalizeTangent(shadingNormal, transformDirection(objectTangent, axisX, axisY, axisZ));
    }

    const float coneWidthHere = surfaceRayFootprint(uniforms, rayDirection, hit.distance, primaryRay);
    if (SPEC_TEXTURE_LOD_CODE && (uniforms.textureLodMode & TEXTURE_LOD_MODE_MASK) != 0u && uvArea2 > 0.0f &&
        coneWidthHere > 0.0f)
    {
        const float ndotd = max(abs(dot(geometryNormal, rayDirection)), 1e-4f);
        lodBase = 0.5f * log2(uvArea2 / worldArea2) + log2(coneWidthHere) - log2(ndotd) + uvLodOffset +
                  uniforms.textureLodBias;
    }

    packSurfaceGeometry(shadingNormal, geometryNormal, tangent, uv, tangentSign, lodBase, payload);
    uniforms.surfaceGeometry[tid] = payload;
    return true;
}

template <typename T, bool InitializePrimary = false>
static void extendImpl(uint gid,
                       uint2 primaryPixel,
                       constant Uniforms& uniforms,
                       constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                       typename T::structure accelerationStructure,
                       typename T::volume_structure volumeAccelerationStructure,
                       device const PathRay* rays,
                       device char* hits,
                       constant uint32_t& sampleIdx,
                       device const uint32_t* queue,
                       device const uint32_t* control,
                       device uint32_t* hitQueue,
                       device uint32_t* missQueue,
                       device atomic_uint* missCounter,
                       // Only for the fog: the path's depth decorrelates the free-flight draw
                       // across bounces, and without it every bounce of a path scatters at the same
                       // fraction of its segment, which shows up as banding in the haze.
                       device const PathState* paths,
                       // Only for the subsurface walk, which needs the medium's mean free path to
                       // sample a free flight and reads it from the material the path is inside.
                       device const Material* materials,
                       device const MediumPathState* mediumPaths,
                       device const GeometryEntry* geometryEntries,
                       typename T::table functionTable,
                       device const char* vertexBuffer,
                       device const char* prevVertexBuffer,
                       device const uint32_t* indexBuffer,
                       device const UniformLight* lights,
                       uint32_t directGeometryBase,
                       uint32_t directInstanceIndex,
                       uint32_t rayMask,
                       device PathState* primaryPaths,
                       device PathRay* primaryRays,
                       device MediumPathState* primaryMediumPaths,
                       device float4* primaryRadiance,
                       device AovSample* primaryAov)
{
    uint32_t tid;
    MediumPathState mediumState = {};
    PathRay pr;
    uint32_t pathFlags;
    float motionTime;
    const bool primaryRay = (rayMask & GEOMETRY_MASK_LIGHT_HIDDEN) == 0u;
    if (InitializePrimary)
    {
        if (primaryPixel.x >= uniforms.width || primaryPixel.y >= uniforms.height)
        {
            return;
        }
        tid = primaryPixel.y * uniforms.width + primaryPixel.x;
        const PrimaryPathData primary =
            initializePrimaryPath(primaryPixel, tid, uniforms, sampleIdx, primaryPaths, primaryRays, primaryRadiance,
                                  primaryAov, primaryMediumPaths);
        pr = primary.ray;
        pathFlags = 0u;
        motionTime = primary.motionTime;
    }
    else
    {
        // Indirect dispatch can only launch whole threadgroups, so the tail of
        // the last one runs past the queue and has to be discarded here.
        if (gid >= control[WF_CTRL_ACTIVE])
        {
            return;
        }
        tid = queue[gid];
        // A corrupted append count must not turn into an out-of-bounds PathRay
        // load and then an invalid hardware traversal. This is cold-path
        // protection: a valid queue always contains the pixel slot its path owns.
        if (tid >= uniforms.width * uniforms.height)
        {
            return;
        }
        if (SPEC_SSS)
        {
            mediumState = mediumPaths[tid];
            const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
            if (!SPEC_SHARC_UPDATE && medium != 0u && (materials[medium - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u)
            {
                return;
            }
        }
        pr = rays[tid];
        pathFlags = primaryRay ? 0u : paths[tid].depthAndFlags;
        motionTime = motionTimeFor(uniforms, tid, sampleIdx);
    }

    const uint32_t auditBounce = min(pathDepth(pathFlags), WORK_BOUNCE_SLOTS - 1u);
    auditExtendWork(uniforms, auditBounce);

    ray r;
    r.min_distance = pathDepth(pathFlags) == 0u ? 0.0f : 1e-6f;
    r.max_distance = INFINITY;
    r.origin = float3(pr.origin);
    r.direction = float3(pr.direction);

    // Triangle traversal turns a malformed ray into a miss. Curve traversal can
    // fail to make progress and trip the watchdog, so only its specialization
    // pays for the defensive finite/range checks.
    if (SPEC_CURVES)
    {
        const float directionLength2 = dot(r.direction, r.direction);
        if (!all(isfinite(r.origin)) || !all(isfinite(r.direction)) ||
            !(directionLength2 > 0.25f && directionLength2 < 4.0f))
        {
            return;
        }
    }

    bool insideSss = false;
    uint32_t mediumHitBit = 0u;
    float mediumScatterT = 0.0f;
    if (SPEC_SSS)
    {
        const uint32_t sss = mediumState.medium;
        const uint32_t medium = sss & MEDIUM_INDEX_MASK;
        insideSss = medium != 0u;
        if (insideSss)
        {
            const uint32_t step = sss >> MEDIUM_STEP_SHIFT;
            if (step < MEDIUM_MAX_STEPS)
            {
                const MediumProps mp = mediumPropsFor(uniforms, materials, medium - 1u, mediumState.mediumAlbedo);
                const float3 sigmaT = mp.sigmaT;
                const float3 albedo = mp.albedo;
                const float3 channelPdf = sssChannelPdf(float3(paths[tid].throughput), albedo);
                SamplerState srng = samplerFor(uniforms, tid, sampleIdx, pathDepth(pathFlags) + step);
                const float2 distanceRandom =
                    random2<SampleDimension::eSssChannel, SampleDimension::eSssDistance>(srng, uniforms.samplerType).value;
                // Bound free flights by the scene: a longer draw left the medium and can wedge traversal on leaked
                // paths.
                if (sssSampleDistance(
                        sigmaT, channelPdf, uniforms.sceneExtent, distanceRandom.x, distanceRandom.y, mediumScatterT))
                {
                    mediumHitBit = HIT_SSS_BIT;
                }
            }
        }
    }
    if (SPEC_FOG && uniforms.hasFog && !insideSss)
    {
        SamplerState frng = samplerFor(uniforms, tid, sampleIdx, pathDepth(pathFlags));
        if (fogSampleDistance(r.origin, r.direction, 1e16f, uniforms.fogHeight, uniforms.fogSigmaT,
                              random<SampleDimension::eFogDistance>(frng, uniforms.samplerType), mediumScatterT))
        {
            mediumHitBit = HIT_FOG_BIT;
        }
    }
    if (mediumHitBit != 0u)
    {
        // Keep max >= min even for a legitimate zero-valued random draw. Any
        // surface in this tiny padded interval is compared with the actual
        // sampled distance after traversal.
        r.max_distance = max(mediumScatterT, r.min_distance + 1e-6f);
    }

    ExtendIntersection hit;
    if (insideSss)
    {
        typename T::volume_isect volumeIsect;
        volumeIsect.assume_geometry_type(geometry_type::triangle);
        volumeIsect.force_opacity(forced_opacity::opaque);
        volumeIsect.accept_any_intersection(false);
        const typename T::volume_isect::result_type volumeHit =
            T::traceVolume(volumeIsect, r, volumeAccelerationStructure, rayMask & ~GEOMETRY_MASK_CURVE, motionTime);
        hit = captureExtendIntersection(volumeHit, 0.0f, volumeHit.instance_id);
    }
    else
    {
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        // Cutout coverage is tested in shade, not extend.
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        const typename T::isect::result_type surfaceHit =
            T::trace(isect, r, accelerationStructure, rayMask, motionTime, functionTable);
        const float curveParameter = surfaceHit.type == intersection_type::curve ? T::curveParameter(surfaceHit) : 0.0f;
        hit = captureExtendIntersection(surfaceHit, curveParameter, T::instanceId(surfaceHit, directInstanceIndex));
    }

    float surfaceDistance = hit.type == intersection_type::none ? r.max_distance : hit.distance;
    if (T::kDirect && SPEC_LIGHTS)
    {
        for (uint32_t lightId = 0u; lightId < uniforms.numLights; ++lightId)
        {
            device const UniformLight& light = lights[lightId];
            // Scenes containing only rectangles compile the type test out. A
            // punctual/infinite light has no TLAS proxy surface and remains
            // NEE-only when it accompanies the rectangle fast path.
            if ((!SPEC_ALL_ANALYTIC_LIGHTS_RECT && light.type != LIGHT_TYPE_RECT) ||
                !analyticLightVisibilityAllowsRay(light.normal.w, !primaryRay))
            {
                continue;
            }
            const PackedAnalyticHit lightHit =
                intersectPackedRectangle(light, r.origin, r.direction, r.min_distance, surfaceDistance);
            if (lightHit.hit && lightHit.distance < surfaceDistance)
            {
                hit.type = intersection_type::triangle;
                hit.instanceId = directInstanceIndex;
                hit.geometryId = HIT_LIGHT_BIT | lightId;
                hit.primitiveId = 0u;
                hit.distance = lightHit.distance;
                hit.barycentrics = float2(0.0f);
                hit.curveParameter = 0.0f;
                hit.primitiveData = nullptr;
                surfaceDistance = lightHit.distance;
            }
        }
    }

    // Escaped rays skip shade and go to miss. Fog/SSS are decided here
    // because only extend knows whether a surface precedes the medium event.
    if (mediumHitBit != 0u && (hit.type == intersection_type::none || mediumScatterT < surfaceDistance))
    {
        HitRecord mediumRec;
        mediumRec.geomEntryIndex = mediumHitBit;
        mediumRec.instanceIndex = 0u;
        mediumRec.primitiveId = 0u;
        mediumRec.barycentrics = 0u;
        mediumRec.distance = mediumScatterT;
        *wavefrontHitRecord(hits, tid) = mediumRec;
        hitQueuePush(uniforms, hitQueue, tid, control[WF_CTRL_CAPACITY], 3u);
        return;
    }

    uint32_t geometryEntryIndex = 0u;
    uint32_t shadeBucket = WF_SHADE_TAIL;
    bool isLight = false;
    GeometryEntry entry = {};
    if (hit.type != intersection_type::none)
    {
        if (T::kDirect)
        {
            isLight = (hit.geometryId & HIT_LIGHT_BIT) != 0u;
            geometryEntryIndex = isLight ? hit.geometryId : directGeometryBase + hit.geometryId;
        }
        else
        {
            const auto inst = instances[hit.instanceId];
            const uint32_t geometryMask = inst.mask & ~GEOMETRY_MASK_UNIFORM_ORTHOGONAL_TRANSFORM;
            isLight = geometryMask == GEOMETRY_MASK_LIGHT || geometryMask == GEOMETRY_MASK_LIGHT_HIDDEN;
            geometryEntryIndex = isLight ? (HIT_LIGHT_BIT | inst.userID) : (inst.userID + hit.geometryId);
        }
        if (!isLight && hit.type == intersection_type::triangle)
        {
            entry = geometryEntries[geometryEntryIndex];
            shadeBucket = (entry.flags & GEOM_SHADE_BUCKET_MASK) >> GEOM_SHADE_BUCKET_SHIFT;
        }
    }

    const bool hasSurfaceGeometry =
        storeSurfaceGeometry(uniforms, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer, hit, tid,
                             motionTime, r.direction, primaryRay, geometryEntryIndex, entry, isLight);
    enqueueExtendSurfaceResult(uniforms, hit, hits, tid, hitQueue, missQueue, missCounter, control, geometryEntryIndex,
                               shadeBucket, SPEC_PREPARE_SURFACE_GEOMETRY_IN_EXTEND && !hasSurfaceGeometry);
}

#define WF_EXTEND_ENTRY(NAME, TRAITS)                                                                                   \
    kernel void NAME(                                                                                                   \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                                \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                           \
        TRAITS::structure accelerationStructure [[buffer(2)]], device const PathRay* rays [[buffer(3)]],                \
        device char* hits [[buffer(4)]], constant uint32_t& sampleIdx [[buffer(5)]],                                    \
        device const uint32_t* queue [[buffer(6)]], device const uint32_t* control [[buffer(7)]],                       \
        device uint32_t* hitQueue [[buffer(8)]], device atomic_uint* hitCounter [[buffer(9)]],                          \
        device uint32_t* missQueue [[buffer(10)]], device atomic_uint* missCounter [[buffer(11)]],                      \
        device const PathState* paths [[buffer(12)]], device const Material* materials [[buffer(13)]],                  \
        constant uint32_t& rayMask [[buffer(14)]], TRAITS::volume_structure volumeAccelerationStructure [[buffer(15)]], \
        constant uint32_t& queueOffset [[buffer(16)]], device const MediumPathState* mediumPaths [[buffer(17)]],        \
        device const GeometryEntry* geometryEntries [[buffer(18)]], TRAITS::table functionTable [[buffer(19)]],         \
        device const char* vertexBuffer [[buffer(20)]], device const char* prevVertexBuffer [[buffer(21)]],             \
        device const uint32_t* indexBuffer [[buffer(22)]], device const UniformLight* lights [[buffer(23)]],            \
        constant uint32_t& directGeometryBase [[buffer(24)]], constant uint32_t& directInstanceIndex [[buffer(25)]])    \
    {                                                                                                                   \
        extendImpl<TRAITS>(gid + queueOffset, uint2(0u), uniforms, instances, accelerationStructure,                    \
                           volumeAccelerationStructure, rays, hits, sampleIdx, queue, control, hitQueue, missQueue,     \
                           missCounter, paths, materials, mediumPaths, geometryEntries, functionTable, vertexBuffer,    \
                           prevVertexBuffer, indexBuffer, lights, directGeometryBase, directInstanceIndex, rayMask,     \
                           nullptr, nullptr, nullptr, nullptr, nullptr);                                                \
    }

WF_EXTEND_ENTRY(wavefrontExtend, MotionTraversal)
WF_EXTEND_ENTRY(wavefrontExtendStatic, StaticTraversal)
WF_EXTEND_ENTRY(wavefrontExtendCurve, CurveMotionTraversal)
WF_EXTEND_ENTRY(wavefrontExtendStaticCurve, CurveStaticTraversal)

kernel void wavefrontExtendDirectStatic(uint gid [[thread_position_in_grid]],
                                        constant Uniforms& uniforms [[buffer(0)]],
                                        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances
                                        [[buffer(1)]],
                                        DirectStaticTraversal::structure accelerationStructure [[buffer(2)]],
                                        device const PathRay* rays [[buffer(3)]],
                                        device char* hits [[buffer(4)]],
                                        constant uint32_t& sampleIdx [[buffer(5)]],
                                        device const uint32_t* queue [[buffer(6)]],
                                        device const uint32_t* control [[buffer(7)]],
                                        device uint32_t* hitQueue [[buffer(8)]],
                                        device atomic_uint* hitCounter [[buffer(9)]],
                                        device uint32_t* missQueue [[buffer(10)]],
                                        device atomic_uint* missCounter [[buffer(11)]],
                                        device const PathState* paths [[buffer(12)]],
                                        device const Material* materials [[buffer(13)]],
                                        constant uint32_t& rayMask [[buffer(14)]],
                                        DirectStaticTraversal::volume_structure volumeAccelerationStructure
                                        [[buffer(15)]],
                                        constant uint32_t& queueOffset [[buffer(16)]],
                                        device const MediumPathState* mediumPaths [[buffer(17)]],
                                        device const GeometryEntry* geometryEntries [[buffer(18)]],
                                        device const char* vertexBuffer [[buffer(20)]],
                                        device const char* prevVertexBuffer [[buffer(21)]],
                                        device const uint32_t* indexBuffer [[buffer(22)]],
                                        device const UniformLight* lights [[buffer(23)]],
                                        constant uint32_t& directGeometryBase [[buffer(24)]],
                                        constant uint32_t& directInstanceIndex [[buffer(25)]])
{
    extendImpl<DirectStaticTraversal>(gid + queueOffset, uint2(0u), uniforms, instances, accelerationStructure,
                                      volumeAccelerationStructure, rays, hits, sampleIdx, queue, control, hitQueue,
                                      missQueue, missCounter, paths, materials, mediumPaths, geometryEntries, 0u,
                                      vertexBuffer, prevVertexBuffer, indexBuffer, lights, directGeometryBase,
                                      directInstanceIndex, rayMask, nullptr, nullptr, nullptr, nullptr, nullptr);
}

#define WF_EXTEND_PRIMARY_ENTRY(NAME, TRAITS)                                                                           \
    kernel void NAME(                                                                                                   \
        uint2 pixel [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                             \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                           \
        TRAITS::structure accelerationStructure [[buffer(2)]], device PathRay* rays [[buffer(3)]],                      \
        device char* hits [[buffer(4)]], constant uint32_t& sampleIdx [[buffer(5)]],                                    \
        device const uint32_t* queue [[buffer(6)]], device const uint32_t* control [[buffer(7)]],                       \
        device uint32_t* hitQueue [[buffer(8)]], device atomic_uint* hitCounter [[buffer(9)]],                          \
        device uint32_t* missQueue [[buffer(10)]], device atomic_uint* missCounter [[buffer(11)]],                      \
        device PathState* paths [[buffer(12)]], device const Material* materials [[buffer(13)]],                        \
        constant uint32_t& rayMask [[buffer(14)]], TRAITS::volume_structure volumeAccelerationStructure [[buffer(15)]], \
        device MediumPathState* mediumPaths [[buffer(17)]],                                                             \
        device const GeometryEntry* geometryEntries [[buffer(18)]], TRAITS::table functionTable [[buffer(19)]],         \
        device const char* vertexBuffer [[buffer(20)]], device const char* prevVertexBuffer [[buffer(21)]],             \
        device const uint32_t* indexBuffer [[buffer(22)]], device const UniformLight* lights [[buffer(23)]],            \
        constant uint32_t& directGeometryBase [[buffer(24)]], constant uint32_t& directInstanceIndex [[buffer(25)]],    \
        device float4* radianceOut [[buffer(26)]], device AovSample* aov [[buffer(27)]])                                \
    {                                                                                                                   \
        const uint32_t gid = pixel.y * uniforms.width + pixel.x;                                                        \
        extendImpl<TRAITS, true>(gid, pixel, uniforms, instances, accelerationStructure, volumeAccelerationStructure,   \
                                 rays, hits, sampleIdx, queue, control, hitQueue, missQueue, missCounter, paths,        \
                                 materials, mediumPaths, geometryEntries, functionTable, vertexBuffer,                  \
                                 prevVertexBuffer, indexBuffer, lights, directGeometryBase, directInstanceIndex,        \
                                 rayMask, paths, rays, mediumPaths, radianceOut, aov);                                  \
    }

WF_EXTEND_PRIMARY_ENTRY(wavefrontExtendPrimaryStatic, StaticTraversal)

kernel void wavefrontExtendPrimaryDirectStatic(uint2 pixel [[thread_position_in_grid]],
                                               constant Uniforms& uniforms [[buffer(0)]],
                                               constant MTLIndirectAccelerationStructureInstanceDescriptor* instances
                                               [[buffer(1)]],
                                               DirectStaticTraversal::structure accelerationStructure [[buffer(2)]],
                                               device PathRay* rays [[buffer(3)]],
                                               device char* hits [[buffer(4)]],
                                               constant uint32_t& sampleIdx [[buffer(5)]],
                                               device const uint32_t* queue [[buffer(6)]],
                                               device const uint32_t* control [[buffer(7)]],
                                               device uint32_t* hitQueue [[buffer(8)]],
                                               device atomic_uint* hitCounter [[buffer(9)]],
                                               device uint32_t* missQueue [[buffer(10)]],
                                               device atomic_uint* missCounter [[buffer(11)]],
                                               device PathState* paths [[buffer(12)]],
                                               device const Material* materials [[buffer(13)]],
                                               constant uint32_t& rayMask [[buffer(14)]],
                                               DirectStaticTraversal::volume_structure volumeAccelerationStructure
                                               [[buffer(15)]],
                                               device MediumPathState* mediumPaths [[buffer(17)]],
                                               device const GeometryEntry* geometryEntries [[buffer(18)]],
                                               device const char* vertexBuffer [[buffer(20)]],
                                               device const char* prevVertexBuffer [[buffer(21)]],
                                               device const uint32_t* indexBuffer [[buffer(22)]],
                                               device const UniformLight* lights [[buffer(23)]],
                                               constant uint32_t& directGeometryBase [[buffer(24)]],
                                               constant uint32_t& directInstanceIndex [[buffer(25)]],
                                               device float4* radianceOut [[buffer(26)]],
                                               device AovSample* aov [[buffer(27)]])
{
    const uint32_t gid = pixel.y * uniforms.width + pixel.x;
    extendImpl<DirectStaticTraversal, true>(gid, pixel, uniforms, instances, accelerationStructure,
                                            volumeAccelerationStructure, rays, hits, sampleIdx, queue, control, hitQueue,
                                            missQueue, missCounter, paths, materials, mediumPaths, geometryEntries, 0u,
                                            vertexBuffer, prevVertexBuffer, indexBuffer, lights, directGeometryBase,
                                            directInstanceIndex, rayMask, paths, rays, mediumPaths, radianceOut, aov);
}

static void fetchTriangle(device const char* vertexBuffer,
                          device const char* prevVertexBuffer,
                          device const uint32_t* indexBuffer,
                          GeometryEntry entry,
                          uint32_t primitiveId,
                          bool interpolateMotion,
                          float motionTime,
                          thread float3* p,
                          thread float3* n,
                          thread float3* t,
                          thread float2* uv,
                          thread float& tangentSign,
                          thread float3* vcol)
{
    // Scene::Vertex is 32 bytes; these offsets are shared with ShaderTypes.h and guarded by host static_asserts.
    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff = 16;
    constexpr uint32_t uvOff = 20;
    constexpr uint32_t colorOff = 28; // Scene::Vertex::color, packed RGBA8

    uint32_t idx[3];
    idx[0] = indexBuffer[entry.indexOffset + primitiveId * 3 + 0];
    idx[1] = indexBuffer[entry.indexOffset + primitiveId * 3 + 1];
    idx[2] = indexBuffer[entry.indexOffset + primitiveId * 3 + 2];

    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t byteOff = (entry.vbOffset + idx[k]) * vtxStride;
        device const char* v = vertexBuffer + byteOff;

        const float3 pos = float3(*(device const packed_float3*)v);
        const float3 nrm = unpackNormal(*(device const uint32_t*)(v + normalOff));
        const uint32_t tanPacked = *(device const uint32_t*)(v + tangentOff);
        const float4 tanUnorm = unpack_unorm10a2_to_float(tanPacked);
        const float3 tan = tanUnorm.xyz * 2.0f - 1.0f;
        uv[k] = unpackUV(*(device const uint32_t*)(v + uvOff), *(device const uint32_t*)(v + uvOff + 4u));
        // Vertex colour is not skinned and does not animate, so it is read from
        // the current frame even when the rest is motion-interpolated.
        vcol[k] = unpackVertexColor(*(device const uint32_t*)(v + colorOff));

        // Handedness is a per-mesh property in every exporter that writes it, so
        // one vertex settles it -- there is nothing sensible to interpolate.
        if (k == 0)
            tangentSign = tanUnorm.w > 0.25f ? -1.0f : 1.0f;

        if (interpolateMotion)
        {
            device const char* pv = prevVertexBuffer + byteOff;
            const float3 posPrev = float3(*(device const packed_float3*)pv);
            const float3 nrmPrev = unpackNormal(*(device const uint32_t*)(pv + normalOff));
            const float3 tanPrev = unpackNormal(*(device const uint32_t*)(pv + tangentOff));
            // BVH keyframe 0 is prevVB at t=0 and keyframe 1 is VB at t=1.
            p[k] = mix(posPrev, pos, motionTime);
            n[k] = mix(nrmPrev, nrm, motionTime);
            t[k] = mix(tanPrev, tan, motionTime);
        }
        else
        {
            p[k] = pos;
            n[k] = nrm;
            t[k] = tan;
        }
    }
}

static void fetchCurve(device const packed_float3* curvePoints,
                       device const uint32_t* curveSegments,
                       GeometryEntry entry,
                       uint32_t primitiveId,
                       float curveParam,
                       float3 worldHit,
                       float4x4 objectToWorld,
                       thread float3& outNormal,
                       thread float3& outTangent,
                       thread float2& outUv,
                       thread float& outRadius)
{
    const bool cubic = (entry.flags & GEOM_CURVE_CUBIC) != 0u;
    // Segment indices are local to the curve set, matching the range exposed to
    // its Metal geometry descriptor. Add the set's scene-wide point offset only
    // when refetching from the shared shader buffer.
    const uint32_t base = entry.vbOffset + curveSegments[entry.indexOffset + primitiveId];
    const float u = saturate(curveParam);
    // Control points are placed by the same transform as the hit, so the axis is
    // built in world space and the radial offset needs no change of basis.
#define WF_CURVE_CP(i) ((objectToWorld * float4(float3(curvePoints[base + (i)]), 1.0f)).xyz)

    float3 axis, dAxis;
    if (cubic)
    {
        // Cubic B-spline, the basis the acceleration structure was built with.
        const float3 p0 = WF_CURVE_CP(0), p1 = WF_CURVE_CP(1);
        const float3 p2 = WF_CURVE_CP(2), p3 = WF_CURVE_CP(3);
        const float u2 = u * u, u3 = u2 * u;
        const float b0 = (1.0f - 3.0f * u + 3.0f * u2 - u3) / 6.0f;
        const float b1 = (4.0f - 6.0f * u2 + 3.0f * u3) / 6.0f;
        const float b2 = (1.0f + 3.0f * u + 3.0f * u2 - 3.0f * u3) / 6.0f;
        const float b3 = u3 / 6.0f;
        const float d0 = (-1.0f + 2.0f * u - u2) * 0.5f;
        const float d1 = (-4.0f * u + 3.0f * u2) * 0.5f;
        const float d2 = (1.0f + 2.0f * u - 3.0f * u2) * 0.5f;
        const float d3 = u2 * 0.5f;
        axis = p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
        dAxis = p0 * d0 + p1 * d1 + p2 * d2 + p3 * d3;
    }
    else
    {
        const float3 p0 = WF_CURVE_CP(0), p1 = WF_CURVE_CP(1);
        axis = mix(p0, p1, u);
        dAxis = p1 - p0;
    }
#undef WF_CURVE_CP

    outTangent = normalize(dAxis);
    const float3 radial = worldHit - axis;
    // Remove whatever component of the offset runs along the strand: at a
    // spherical cap the closest axis point is not the one the parameter names,
    // and leaving it in tilts the normal toward the tip.
    const float3 perp = radial - outTangent * dot(radial, outTangent);
    const float lenSq = dot(perp, perp);
    // A hit exactly on the axis has no radial direction; anything perpendicular
    // to the strand will do, and this is far rarer than a denormal guard.
    outNormal = lenSq > 1e-16f ? perp * rsqrt(lenSq) : normalize(cross(outTangent, float3(0.0f, 0.0f, 1.0f)));
    outRadius = sqrt(lenSq);

    const uint32_t perStrand = entry.flags & GEOM_CURVE_STRAND_MASK;
    const float alongStrand = perStrand != 0u ? ((float)(primitiveId % perStrand) + u) / (float)perStrand : 0.0f;
    outUv = float2(alongStrand, 0.0f);
}

// The whole-fibre lobe already accounts for absorption across the chord, so transmission starts at the far surface.
// For a cylinder, the chord distance is -2r(n.u) divided by the direction's length across the strand.
static inline float3 fibreExitOrigin(float3 position, float3 tangent, float3 normal, float radius, float3 dir)
{
    const float3 dPerp = dir - tangent * dot(dir, tangent);
    const float m2 = dot(dPerp, dPerp);
    // Straight along the strand there is no far wall, and the chord below would
    // divide by zero on the way to saying so.
    if (m2 < 1e-8f || radius <= 0.0f)
    {
        return offset_ray(position, normal);
    }
    const float m = sqrt(m2);
    const float3 u = dPerp / m;
    const float chord = -2.0f * radius * dot(normal, u);
    // Leaving on the side it arrived from: an ordinary surface offset is enough.
    if (chord <= 0.0f)
    {
        return offset_ray(position, normal);
    }
    // Offset along the outward normal at the exit, not at the entry: they point
    // to opposite sides of the strand.
    return offset_ray(position + dir * (chord / m), normalize(normal * radius + u * chord));
}

static void fetchTriangleBlended(device const char* vertexBuffer,
                                 device const char* prevVertexBuffer,
                                 device const uint32_t* indexBuffer,
                                 GeometryEntry entry,
                                 uint32_t primitiveId,
                                 bool interpolateMotion,
                                 float motionTime,
                                 float2 bary,
                                 thread float3& outNormal,
                                 thread float3& outTangent,
                                 thread float2& outUv,
                                 thread float3& outColor,
                                 thread float& tangentSign,
                                 thread float3& outGeomNormal,
                                 // For the ray-cone texture LOD: twice the triangle's
                                 // area in uv. It falls out of loads this function
                                 // already does, so the footprint costs no extra traffic.
                                 thread float& outUvArea2)
{
    constexpr uint32_t vtxStride = 32;
    constexpr uint32_t tangentOff = 12;
    constexpr uint32_t normalOff = 16;
    constexpr uint32_t uvOff = 20;
    constexpr uint32_t colorOff = 28;

    const float w0 = 1.0f - bary.x - bary.y;
    const float3 weight = float3(w0, bary.x, bary.y);

    outNormal = float3(0.0f);
    outTangent = float3(0.0f);
    outColor = float3(0.0f);
    outUv = float2(0.0f);
    float3 p0 = float3(0.0f), e1 = float3(0.0f), e2 = float3(0.0f);
    float2 uv0 = float2(0.0f), uvE1 = float2(0.0f), uvE2 = float2(0.0f);

    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitiveId * 3 + k];
        const uint32_t byteOff = (entry.vbOffset + idx) * vtxStride;
        device const char* v = vertexBuffer + byteOff;

        float3 pos = float3(*(device const packed_float3*)v);
        float3 nrm = unpackNormal(*(device const uint32_t*)(v + normalOff));
        const uint32_t tanPacked = *(device const uint32_t*)(v + tangentOff);
        const float4 tanUnorm = unpack_unorm10a2_to_float(tanPacked);
        float3 tan = tanUnorm.xyz * 2.0f - 1.0f;

        if (k == 0)
        {
            tangentSign = tanUnorm.w > 0.25f ? -1.0f : 1.0f;
        }

        if (interpolateMotion)
        {
            device const char* pvb = prevVertexBuffer + byteOff;
            pos = mix(float3(*(device const packed_float3*)pvb), pos, motionTime);
            nrm = mix(unpackNormal(*(device const uint32_t*)(pvb + normalOff)), nrm, motionTime);
            tan = mix(unpackNormal(*(device const uint32_t*)(pvb + tangentOff)), tan, motionTime);
        }

        outNormal += nrm * weight[k];
        outTangent += tan * weight[k];
        const float2 vertUv = unpackUV(*(device const uint32_t*)(v + uvOff), *(device const uint32_t*)(v + uvOff + 4u));
        outUv += vertUv * weight[k];
        if (k == 0)
            uv0 = vertUv;
        else if (k == 1)
            uvE1 = vertUv - uv0;
        else
            uvE2 = vertUv - uv0;
        outColor += unpackVertexColor(*(device const uint32_t*)(v + colorOff)) * weight[k];

        if (k == 0)
            p0 = pos;
        else if (k == 1)
            e1 = pos - p0;
        else
            e2 = pos - p0;
    }
    outGeomNormal = cross(e1, e2);
    outUvArea2 = abs(uvE1.x * uvE2.y - uvE2.x * uvE1.y);
}

static inline float3 previousWorldPosition(device const char* prevFrameVertexBuffer,
                                           device const uint32_t* indexBuffer,
                                           device const MTLIndirectAccelerationStructureInstanceDescriptor* prevInstances,
                                           constant Uniforms& uniforms,
                                           GeometryEntry entry,
                                           uint32_t instanceIndex,
                                           uint32_t geometryEntryIndex,
                                           uint32_t primitiveId,
                                           float2 bary)
{
    constexpr uint32_t vtxStride = 32; // see fetchTriangle for the layout

    float3 p[3];
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t idx = indexBuffer[entry.indexOffset + primitiveId * 3 + k];
        p[k] = float3(*(device const packed_float3*)(prevFrameVertexBuffer + (entry.vbOffset + idx) * vtxStride));
    }
    const float3 objectPos = interpolateAttrib(p[0], p[1], p[2], bary);

    const float4x4 prevObjectToWorld =
        geometryObjectToWorld(uniforms, prevInstances, instanceIndex, geometryEntryIndex, entry);
    return (prevObjectToWorld * float4(objectPos, 1.0f)).xyz;
}

// Depth in whichever convention the denoiser is currently being fed; see
// kDenoiseDepth* for why this is a switch rather than a decision.
static inline float viewDepth(constant Uniforms& uniforms, float3 worldPosition)
{
    if (uniforms.denoiseDepthMode == kDenoiseDepthDevice)
    {
        const float4 clip = uniforms.worldToClip * float4(worldPosition, 1.0f);
        return clip.w > 0.0f ? clip.z / clip.w : 1.0f;
    }
    const float3 eye = (uniforms.viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    if (uniforms.denoiseDepthMode == kDenoiseDepthViewZ)
    {
        // Along the camera axis, which is what a depth buffer holds before the
        // projection is applied. The third column of viewToWorld is the camera's
        // backward axis, so forward is its negation.
        const float3 forward = -uniforms.viewToWorld[2].xyz;
        return dot(worldPosition - eye, forward);
    }
    return length(worldPosition - eye);
}

// The value the background writes. Device depth has a finite far plane, so the
// sentinel has to match the convention or the denoiser reads the sky as being
// nearer than the geometry.
static inline float backgroundDepth(constant Uniforms& uniforms)
{
    return denoiseBackgroundDepth(uniforms.denoiseDepthMode, uniforms.projectionType);
}

struct ScreenMotion
{
    float2 offset;
    float reactive;
};

static inline ScreenMotion screenMotion(constant Uniforms& uniforms, float4 prevClip, uint2 pixel)
{
    ScreenMotion result;
    result.offset = float2(0.0f);
    result.reactive = 0.0f;
    // Near-zero clip w has no valid reprojection. This is the exceptional case
    // the reactive mask is for: favor the current frame because there is no
    // meaningful history location to sample.
    const float kMinW = 1e-4f;
    if (prevClip.w <= kMinW)
    {
        result.reactive = 1.0f;
        return result;
    }
    const float2 prevNdc = prevClip.xy / prevClip.w;
    const float2 prevPixel = float2(
        (prevNdc.x * 0.5f + 0.5f) * (float)uniforms.width, (1.0f - (prevNdc.y * 0.5f + 0.5f)) * (float)uniforms.height);
    // This is the raster location of the world-space hit under the current
    // unjittered camera. Using the centre here would make a still scene report
    // exactly the jitter as motion, which is not a dejittered vector.
    const float2 motion = float2(strelkaScreenMotionAxis(prevPixel.x, (float)pixel.x + 0.5f, uniforms.jitterX),
                                 strelkaScreenMotionAxis(prevPixel.y, (float)pixel.y + 0.5f, uniforms.jitterY));
    const float limit = (float)(uniforms.width + uniforms.height);
    result.offset = clamp(motion, -limit, limit);
    return result;
}

struct DenoiserMaterialGuides
{
    float3 diffuse;
    float3 specular;
    float roughness;
    float transmission;
};

static inline bool openpbrBaseMapDetailsSubsurface(thread const OpenPBRParams& p)
{
    return p.subsurface_weight > 0.0f && openpbrHasMap(p, OPENPBR_TEX_BASE_COLOR) &&
           !openpbrHasMap(p, OPENPBR_TEX_SUBSURFACE_COLOR);
}

static inline DenoiserMaterialGuides standardDenoiserGuides(thread const SurfaceInteraction& si)
{
    DenoiserMaterialGuides g;
    const float3 base = float3(si.albedo);
    const float dielectric = 1.0f - si.metallic;
    g.diffuse = base * dielectric * (1.0f - si.transmission) * (1.0f - si.diffuse_transmission);

    const float cosView = saturate(abs(dot(float3(si.shading_normal), float3(si.wo))));
    const float3 f0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);
    g.specular = fresnel_schlick_roughness(f0, cosView, si.roughness);

    const float coatFresnel = si.clearcoat * fresnel_schlick_scalar(f0_from_ior(max(si.clearcoat_ior, 1.0f)), cosView);
    g.specular = g.specular + (float3(1.0f) - g.specular) * coatFresnel;
    const float3 fuzz = si.sheen_color * si.sheen * powr(1.0f - cosView, 5.0f);
    g.specular = saturate(g.specular + (float3(1.0f) - g.specular) * fuzz);

    const float mainEnergy = max(luminance(g.specular), 1e-4f);
    const float coatEnergy = max(coatFresnel, 0.0f);
    const float fuzzEnergy = max(luminance(fuzz), 0.0f);
    g.roughness =
        saturate((si.roughness * mainEnergy + si.clearcoat_roughness * coatEnergy + si.sheen_roughness * fuzzEnergy) /
                 (mainEnergy + coatEnergy + fuzzEnergy));
    g.transmission = max(si.transmission, si.diffuse_transmission);
    return g;
}

static inline DenoiserMaterialGuides openpbrDenoiserGuides(thread const OpenPBR_ResolvedInputs& in,
                                                           thread const SurfaceInteraction& si,
                                                           bool baseMapDetailsSubsurface)
{
    DenoiserMaterialGuides g;
    const float dielectric = 1.0f - in.base_metalness;
    const float opaque = 1.0f - in.transmission_weight;
    const float3 weightedBase = in.base_color * in.base_weight;
    // MaterialX exports in the test scenes use a mapped base plus a constant
    // subsurface tint. Preserve the mapped detail in that case; a genuinely
    // mapped subsurface colour remains an independent OpenPBR input.
    const float3 subsurfaceColor = baseMapDetailsSubsurface ? weightedBase * in.subsurface_color : in.subsurface_color;
    const float3 diffuseColor = mix(weightedBase, subsurfaceColor, in.subsurface_weight);
    g.diffuse = diffuseColor * dielectric * opaque;

    const float cosView = saturate(abs(dot(float3(si.shading_normal), float3(si.wo))));
    const float dielectricF0 = f0_from_ior(max(in.specular_ior, 1.0f)) * in.specular_weight;
    const float3 dielectricSpecular = saturate(in.specular_color * dielectricF0);
    const float3 metalSpecular = saturate(weightedBase * in.specular_weight);
    const float3 f0 = mix(dielectricSpecular, metalSpecular, in.base_metalness);
    g.specular = fresnel_schlick_roughness(f0, cosView, in.specular_roughness);

    const float coatFresnel = in.coat_weight * fresnel_schlick_scalar(f0_from_ior(max(in.coat_ior, 1.0f)), cosView);
    const float3 coated = g.specular + (float3(1.0f) - g.specular) * in.coat_color * coatFresnel;
    const float3 fuzz = in.fuzz_color * in.fuzz_weight * powr(1.0f - cosView, 5.0f);
    g.specular = saturate(coated + (float3(1.0f) - coated) * fuzz);

    const float mainEnergy = max(luminance(g.specular), 1e-4f);
    const float coatEnergy = max(coatFresnel, 0.0f);
    const float fuzzEnergy = max(luminance(fuzz), 0.0f);
    const float roughnessEnergy =
        in.specular_roughness * mainEnergy + in.coat_roughness * coatEnergy + in.fuzz_roughness * fuzzEnergy;
    g.roughness = saturate(roughnessEnergy / (mainEnergy + coatEnergy + fuzzEnergy));
    g.transmission = in.transmission_weight;
    return g;
}

static inline DenoiserMaterialGuides openpbrBaseDenoiserGuides(thread const OpenPBR_BaseParams& in,
                                                               thread const SurfaceInteraction& si)
{
    DenoiserMaterialGuides g;
    const float3 weightedBase = openpbr_color_to_float3(in.base_color) * in.base_weight;
    const float dielectric = 1.0f - in.base_metalness;
    g.diffuse = weightedBase * dielectric;

    const float cosView = saturate(abs(dot(float3(si.shading_normal), float3(si.wo))));
    const float dielectricF0 = f0_from_ior(max(in.specular_ior, 1.0f)) * in.specular_weight;
    const float3 dielectricSpecular = saturate(openpbr_color_to_float3(in.specular_color) * dielectricF0);
    const float3 metalSpecular = saturate(weightedBase * in.specular_weight);
    const float3 f0 = mix(dielectricSpecular, metalSpecular, in.base_metalness);
    g.specular = fresnel_schlick_roughness(f0, cosView, in.specular_roughness);
    g.roughness = in.specular_roughness;
    g.transmission = 0.0f;
    return g;
}

static inline float denoiserInterfaceIor(bool isOpenPBR,
                                         thread const OpenPBRParams& openpbr,
                                         thread const SurfaceInteraction& si,
                                         bool entering)
{
    if (!isOpenPBR)
    {
        return max(si.ior, 1.0f);
    }

    const float exterior = max(si.exterior_ior, 1e-4f);
    const bool entersCoat = entering && openpbr.coat_weight > 0.0f;
    const float surrounding = entersCoat ? mix(exterior, max(openpbr.coat_ior, 1.0f), openpbr.coat_weight) : exterior;
    return surrounding *
           openpbr_apply_specular_weight_to_ior(max(openpbr.specular_ior, 1.0f) / surrounding, openpbr.specular_weight);
}

static inline float denoiserInterfaceIor(bool isOpenPBR,
                                         thread const OpenPBR_BaseParams& openpbr,
                                         thread const SurfaceInteraction& si,
                                         bool entering)
{
    (void)entering;
    if (!isOpenPBR)
    {
        return max(si.ior, 1.0f);
    }
    const float exterior = max(si.exterior_ior, 1e-4f);
    return exterior *
           openpbr_apply_specular_weight_to_ior(max(openpbr.specular_ior, 1.0f) / exterior, openpbr.specular_weight);
}

static inline uint32_t packGuideIors(float currentIor, float exteriorIor)
{
    return as_type<uint32_t>(half2(currentIor, exteriorIor));
}

static inline float2 unpackGuideIors(uint32_t packed)
{
    return float2(as_type<half2>(packed));
}

kernel void wavefrontMiss(uint gid [[thread_position_in_grid]],
                          constant Uniforms& uniforms [[buffer(0)]],
                          device const PathState* paths [[buffer(1)]],
                          device const PathRay* rays [[buffer(2)]],
                          device float4* radianceOut [[buffer(3)]],
                          device const uint32_t* queue [[buffer(4)]],
                          device const uint32_t* control [[buffer(5)]],
                          device AovSample* aov [[buffer(6)]],
                          constant uint32_t& sampleIdx [[buffer(7)]],
                          device const IorStack* iorStacks [[buffer(8)]],
                          device atomic_uint* iorStats [[buffer(9)]],
                          device SharcUpdateState* sharcUpdates [[buffer(10)]],
                          device SharcAccumulationEntry* sharcAccumulation [[buffer(11)]],
                          device const UniformLight* lights [[buffer(12)]],
                          device const uint2* envAliasTable [[buffer(13)]],
                          texture2d<float> envMapTexture [[texture(0)]],
                          texture2d<float> envBackgroundTexture [[texture(1)]])
{
    if (gid >= control[WF_CTRL_MISS_N])
    {
        return;
    }
    const uint32_t tid = queue[gid];
    const PathState p = paths[tid];
    const float3 rayDir = float3(rays[tid].direction);
    const float3 throughput = float3(p.throughput);
    const uint32_t depth = pathDepth(p.depthAndFlags);
    auditWork(uniforms, WORK_MISS_ITEMS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    // Record path depth for the SHARC bounce heatmap.
    if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcBounces)
    {
        aov[tid].guideStateOrBounceDepth = (float)depth;
    }

    // Counted here because here is the only place it is visible: the path is
    // gone and it still thinks it is inside glass. See ior_stack.h.
    if ((p.depthAndFlags & PATH_FLAG_IOR_STACK_ACTIVE) != 0u)
    {
        atomic_fetch_add_explicit(&iorStats[IOR_STAT_ESCAPED_INSIDE], 1u, memory_order_relaxed);
    }

    // Escapes must overwrite stale denoiser guides, including those reached after a specular primary hit.
    if (shouldWriteAov(uniforms, sampleIdx) && (depth == 0u || (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u))
    {
        AovSample a;
        a.diffuseAlbedo = packed_float3(float3(0.0f));
        a.specularAlbedo = packed_float3(float3(0.0f));
        a.normal = packed_float3(-rayDir);
        a.roughness = 1.0f;
        a.depth = backgroundDepth(uniforms);
        // Camera rays reproject the sky at infinity; bounced rays retain the primary surface's motion instead.
        const ScreenMotion motion = (depth == 0u) ?
                                        screenMotion(uniforms, uniforms.prevWorldToClip * float4(rayDir, 0.0f),
                                                     uint2(tid % uniforms.width, tid / uniforms.width)) :
                                        ScreenMotion{ float2(0.0f), 0.0f };
        a.specularHitDistance = 0.0f;
        // Primary background is noise-free but still has valid camera motion.
        // Denoise strength handles the former; rejecting temporal history here
        // throws away the subpixel samples the upscaler needs for the latter.
        a.reactive = depth == 0u ? motion.reactive : 0.0f;
        a.guideStateOrBounceDepth = depth == 0u ? -1.0f : 0.0f;
        if (depth == 0u)
        {
            a.motionX = motion.offset.x;
            a.motionY = motion.offset.y;
            aov[tid] = a;
        }
        else
        {
            // Bounced sky keeps the camera-visible surface's depth and motion for reprojection.
            const float state = aov[tid].guideStateOrBounceDepth;
            const bool blendTransmission = state >= 1.0f;
            const float interfaceFresnel = blendTransmission ? saturate(state - 1.0f) : 0.0f;
            if (blendTransmission)
            {
                const float3 primaryNormal = float3(aov[tid].normal);
                const float3 skyNormal = -rayDir;
                const float3 blendedNormal = primaryNormal * interfaceFresnel + skyNormal * (1.0f - interfaceFresnel);
                a.specularAlbedo = packed_float3(float3(interfaceFresnel));
                a.normal =
                    packed_float3(dot(blendedNormal, blendedNormal) > 1e-8f ? normalize(blendedNormal) : skyNormal);
                a.roughness = mix(aov[tid].roughness, 1.0f, 1.0f - interfaceFresnel);
            }
            a.depth = aov[tid].depth;
            a.motionX = aov[tid].motionX;
            a.motionY = aov[tid].motionY;
            a.specularHitDistance = aov[tid].specularHitDistance;
            a.reactive = aov[tid].reactive;
            aov[tid] = a;
        }
    }
    // A specular primary that escaped into the environment still has a hit
    // distance -- infinity -- and leaving it at zero tells MetalFX the
    // reflection sits on the mirror.
    if (shouldWriteAov(uniforms, sampleIdx) && depth > 0u && specularBounce &&
        (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
    {
        aov[tid].specularHitDistance += max(uniforms.sceneExtent, 1e3f);
    }

    // A single-hit debug view and the SHaRC surface diagnostics are answers about
    // a surface. The background has nothing to say in either, and an environment
    // brighter than the debug colours drowns them where it does appear.
    if (!SPEC_SHARC_UPDATE &&
        ((SPEC_DEBUG && DEBUG_MODE_IS_SINGLE_HIT(uniforms.debug)) ||
         (SPEC_SHARC && uniforms.sharcCapacity != 0u && SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug))))
    {
        return;
    }

    float3 radiance = float3(0.0f);
    float3 sharcEnvironment = float3(0.0f);
    if (SPEC_ENV_MAP && uniforms.hasEnvMap)
    {
        auditWork(uniforms, WORK_ENVIRONMENT_EVALUATIONS);
        constexpr sampler envSampler(
            mag_filter::linear, min_filter::linear, s_address::repeat, t_address::clamp_to_edge, coord::normalized);
        const float2 envUV = metalDirToEnvUv(rayDir, uniforms.envMapRotationSinCos);
        float3 envColor = envMapTexture.sample(envSampler, envUV).xyz;
        envColor *= uniforms.envMapIntensity * uniforms.envMapColorTint.xyz;

        if (depth == 0u || specularBounce || !neeDone)
        {
            if (uniforms.hasEnvBackground && depth == 0u)
            {
                envColor = envBackgroundTexture.sample(envSampler, envUV).xyz * uniforms.envBackgroundIntensity *
                           uniforms.envMapColorTint.xyz;
            }
            radiance += throughput * envColor;
            sharcEnvironment = envColor;
        }
        else
        {
            const float envPdf = envMapPdf(rayDir, uniforms.envPdfTable, uniforms.envMapWidth, uniforms.envMapHeight,
                                           uniforms.envMapRotationSinCos);
            const float envSelectionPdf =
                (uniforms.numLights > 0 || (SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0)) ?
                    uniforms.envMapColorTint.w :
                    1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            const float mis =
                effectiveEnvPdf > 0.0f ? computeMisWeight(p.lastBsdfPdf, effectiveEnvPdf, uniforms.misHeuristic) : 1.0f;
            radiance += throughput * envColor * mis;
            sharcEnvironment = envColor * mis;
        }
    }
    else
    {
        radiance += throughput * uniforms.missColor;
        sharcEnvironment = uniforms.missColor;
    }

    if (SPEC_LIGHTS)
    {
        const float localSelectionPdf = (SPEC_ENV_MAP && uniforms.hasEnvMap) ? (1.0f - uniforms.envMapColorTint.w) : 1.0f;
        const float analyticClassPdf = SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0u ?
                                           (1.0f - uniforms.meshLightSelectionPdf) :
                                           1.0f;
        for (uint32_t infiniteIndex = 0; infiniteIndex < uniforms.numInfiniteLights; ++infiniteIndex)
        {
            auditWork(uniforms, WORK_MISS_LIGHT_EVALUATIONS);
            const uint32_t lightId = uniforms.infiniteLightIndices[infiniteIndex];
            device const UniformLight& light = lights[lightId];
            if (!analyticLightVisibilityAllowsRay(light.normal.w, depth != 0u))
            {
                continue;
            }
            const float3 axis = -float3(light.normal);
            if (light.type == LIGHT_TYPE_DISTANT && distantLightIsDelta(light.halfAngle))
            {
                if (distantLightDeltaPathMatches(depth, specularBounce, rayDir, axis))
                {
                    const float3 infiniteRadiance = float3(light.color);
                    radiance += throughput * infiniteRadiance;
                    sharcEnvironment += infiniteRadiance;
                }
                continue;
            }
            const float conditionalPdf = infiniteLightConditionalPdf(light.type, light.halfAngle, rayDir, axis);
            if (!(conditionalPdf > 0.0f))
            {
                continue;
            }
            const float effectivePdf =
                localSelectionPdf * analyticClassPdf * analyticLightSelectionPdf(light) * conditionalPdf;
            const float mis = (depth == 0u || specularBounce || !neeDone || !(effectivePdf > 0.0f)) ?
                                  1.0f :
                                  computeMisWeight(p.lastBsdfPdf, effectivePdf, uniforms.misHeuristic);
            const float3 infiniteRadiance = float3(light.color) * mis;
            radiance += throughput * infiniteRadiance;
            sharcEnvironment += infiniteRadiance;
        }
    }
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcUpdateMiss(updateState, uniforms, sharcAccumulation, sharcEnvironment, -rayDir);
        sharcUpdates[updateIndex] = updateState;
    }
    addFilteredRadiance(
        radianceOut, tid, clampPathContribution(radiance, depth, uniforms.clampDirect, uniforms.clampIndirect));
}

// ---------------------------------------------------------------------------
// shade -- material evaluation, next-event estimation, next ray
// ---------------------------------------------------------------------------
struct LightConnectionEvaluation
{
    float3 integrand;
    float target;
};

template <typename OpenPBRPrepared>
static float restirTargetOnly(thread const LightConnection& connection,
                              thread const SurfaceInteraction& si,
                              bool isFibre,
                              thread const ShadedFrame& neeFrame,
                              bool isOpenPBR,
                              thread const OpenPBRPrepared& openpbrPrepared,
                              uint32_t misHeuristic)
{
    if (!connection.needsRay || !(connection.pdf > 0.0f) ||
        !neeProposesDirection(
            isFibre, neeFrame.frontFace, neeFrame.normalSign * dot(connection.toLight, si.shading_normal)))
    {
        return 0.0f;
    }
    const BsdfEvalResult evalResult =
        isOpenPBR ? openpbr_bsdf_eval(openpbrPrepared, si, connection.toLight) : bsdf_eval(si, connection.toLight);
    if (!(evalResult.pdf > 0.0f))
    {
        return 0.0f;
    }
    const float misWeight = connection.isDelta ? 1.0f : computeMisWeight(connection.pdf, evalResult.pdf, misHeuristic);
    return luminance(connection.radiance * evalResult.bsdf * misWeight);
}

template <typename OpenPBRPrepared>
static LightConnectionEvaluation evaluateLightConnection(thread const LightConnection& connection,
                                                         thread const SurfaceInteraction& si,
                                                         bool isFibre,
                                                         thread const ShadedFrame& neeFrame,
                                                         bool isOpenPBR,
                                                         thread const OpenPBRPrepared& openpbrPrepared,
                                                         uint32_t misHeuristic)
{
    LightConnectionEvaluation result = {};
    // A fibre has no back side to reject, and neither does a leaf: see
    // neeCrossesSurface().
    if (!connection.needsRay || !(connection.pdf > 0.0f) ||
        !neeProposesDirection(neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission), neeFrame.frontFace,
                              neeFrame.normalSign * dot(connection.toLight, si.shading_normal)))
    {
        return result;
    }
    const BsdfEvalResult evalResult =
        isOpenPBR ? openpbr_bsdf_eval(openpbrPrepared, si, connection.toLight) : bsdf_eval(si, connection.toLight);
    if (!(evalResult.pdf > 0.0f))
    {
        return result;
    }
    const float misWeight = connection.isDelta ? 1.0f : computeMisWeight(connection.pdf, evalResult.pdf, misHeuristic);
    result.integrand = connection.radiance * evalResult.bsdf * misWeight;
    result.target = luminance(result.integrand);
    return result;
}

static bool restirSurfaceHistoryCompatible(thread const RestirSurfaceHistory& current,
                                           thread const RestirSurfaceHistory& previous)
{
    return restirSurfaceCompatible(current.depth, previous.depth,
                                   dot(float3(current.geometryNormal), float3(previous.geometryNormal)),
                                   current.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK,
                                   previous.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK,
                                   (previous.materialIdAndFlags & RESTIR_SURFACE_VALID) != 0u);
}

static uint32_t restirVisibilityQuantizedHash(float3 value, float inverseStep)
{
    const int3 q = int3(floor(value * inverseStep));
    uint32_t key = hash_combine(as_type<uint32_t>(q.x), as_type<uint32_t>(q.y));
    return hash_combine(key, as_type<uint32_t>(q.z));
}

static uint32_t restirVisibilityReceiverKey(constant Uniforms& uniforms,
                                            thread const SurfaceInteraction& si,
                                            uint32_t materialId,
                                            uint32_t geometryKey)
{
    const float positionStep = clamp(uniforms.sceneExtent * 0.005f, 0.02f, 0.1f);
    const int3 q = int3(floor(si.position / positionStep));
    uint32_t key = hash_combine(as_type<uint32_t>(q.x), as_type<uint32_t>(q.y));
    key = hash_combine(key, as_type<uint32_t>(q.z));
    key = hash_combine(key, restirVisibilityQuantizedHash(si.geometry_normal, 4.0f));
    key = hash_combine(key, materialId);
    key = hash_combine(key, geometryKey);
    return ((uniforms.restirVisibilityRevision & 0xffu) << 24u) | (key & RESTIR_VISIBILITY_LIGHT_KEY_MASK);
}

static uint32_t restirVisibilityGeometryKey(uint32_t geomEntryIndex, uint32_t instanceIndex, uint32_t primitiveId)
{
    return hash_combine(hash_combine(geomEntryIndex, instanceIndex), primitiveId) & RESTIR_TARGET_GEOMETRY_MASK;
}

static uint32_t restirVisibilityGeometryKey(thread const RestirTargetSurface& stored)
{
    if ((stored.geomEntryIndex & RESTIR_TARGET_DIRECT) != 0u)
        return ((thread const RestirDirectTargetSurface*)&stored)->flags & RESTIR_TARGET_GEOMETRY_MASK;
    return restirVisibilityGeometryKey(stored.geomEntryIndex, stored.instanceIndex, stored.primitiveId);
}

static uint32_t restirVisibilityLightKey(constant Uniforms& uniforms, thread const LightConnection& connection)
{
    uint32_t key = hash_combine(connection.sample.typeAndLightId, connection.sample.data0);
    key = hash_combine(key, connection.sample.data1);
    key = hash_combine(key, connection.sample.data2);
    const bool infinite = connection.tMax >= 1e15f;
    const float3 endpoint = infinite                       ? connection.toLight :
                            connection.hasVisibilityTarget ? connection.visibilityTarget :
                                                             connection.origin + connection.toLight * connection.tMax;
    const float positionStep = clamp(uniforms.sceneExtent * 0.01f, 0.1f, 0.25f);
    key = hash_combine(key, restirVisibilityQuantizedHash(endpoint, infinite ? 50.0f : 1.0f / positionStep));
    return key & RESTIR_VISIBILITY_LIGHT_KEY_MASK;
}

static bool restirVisibilityReceiverMatches(constant Uniforms& uniforms,
                                            thread const RestirReservoir& reservoir,
                                            thread const SurfaceInteraction& si,
                                            uint32_t materialId,
                                            uint32_t geometryKey)
{
    return reservoir.visibility.receiverKey == restirVisibilityReceiverKey(uniforms, si, materialId, geometryKey);
}

static bool restirVisibilityCacheMatches(constant Uniforms& uniforms,
                                         thread const RestirReservoir& reservoir,
                                         thread const SurfaceInteraction& si,
                                         uint32_t materialId,
                                         uint32_t geometryKey,
                                         uint32_t lightKey,
                                         bool countRejects)
{
    if (countRejects)
        auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_ATTEMPTS);
    if ((reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) == 0u)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_EMPTY);
        return false;
    }
    if (restirVisibilityAge(reservoir.visibility) > uniforms.restirFinalVisibilityMaxAge)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_AGE);
        return false;
    }
    if ((reservoir.visibility.receiverKey >> 24u) != (uniforms.restirVisibilityRevision & 0xffu))
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_REVISION);
        return false;
    }
    if (!restirVisibilityReceiverMatches(uniforms, reservoir, si, materialId, geometryKey))
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_RECEIVER);
        return false;
    }
    if ((reservoir.visibility.lightKeyAndAge & RESTIR_VISIBILITY_LIGHT_KEY_MASK) != lightKey)
    {
        if (countRejects)
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_REJECT_LIGHT);
        return false;
    }
    return true;
}

static bool restirCanStoreDirectTarget(thread const SurfaceInteraction& si, bool isFibre, bool isOpenPBR)
{
    if (isFibre || isOpenPBR || si.material_type == MATERIAL_TYPE_HAIR || any(si.shading_normal != si.geometry_normal))
    {
        return false;
    }
    if (si.material_type != MATERIAL_TYPE_STANDARD_PBR)
    {
        return true;
    }
    return si.transmission == 0.0f && si.clearcoat == 0.0f && si.anisotropy == 0.0f && si.specular == 0.5f &&
           all(si.specular_color == float3(1.0f)) && si.iridescence == 0.0f && si.diffuse_transmission == 0.0f &&
           si.sheen == 0.0f && si.subsurface == 0.0f && si.ior == 1.5f && !si.diffuse_faces_away;
}

static bool restirTargetIsDirect(thread const RestirTargetSurface& stored)
{
    return (stored.geomEntryIndex & RESTIR_TARGET_DIRECT) != 0u;
}

static float4x4 restirTargetObjectToWorld(constant Uniforms& uniforms,
                                          constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                          device const GeometryEntry* geometryEntries,
                                          thread const RestirTargetSurface& stored)
{
    if (restirTargetIsDirect(stored))
    {
        return float4x4(1.0f);
    }
    const GeometryEntry entry = geometryEntries[stored.geomEntryIndex];
    return geometryObjectToWorld(uniforms, instances, stored.instanceIndex, stored.geomEntryIndex, entry);
}

static void rebuildRestirTargetSurface(constant Uniforms& uniforms,
                                       float4x4 objectToWorld,
                                       device const Material* materials,
                                       device const GeometryEntry* geometryEntries,
                                       device const char* vertexBuffer,
                                       device const char* prevVertexBuffer,
                                       device const uint32_t* indexBuffer,
                                       device const packed_float3* curvePoints,
                                       device const uint32_t* curveSegments,
                                       thread const RestirTargetSurface& stored,
                                       bool interpolateMotion,
                                       float motionTime,
                                       thread SurfaceInteraction& si,
                                       thread bool& isFibre,
                                       thread bool& isOpenPBR,
                                       thread ShadedFrame& neeFrame,
                                       thread OpenPBR_PreparedBsdf& openpbrPrepared,
                                       thread float& curveRadius);

static inline void auditNeeSampledConnection(constant Uniforms& uniforms, thread const LightConnection& conn)
{
    auditWork(uniforms, WORK_NEE_LIGHT_SAMPLER_CALLS);
    if (!conn.needsRay)
    {
        auditWork(uniforms, WORK_NEE_REJECT_CONNECTION);
        return;
    }
    auditWork(uniforms, WORK_NEE_CONDITIONAL_PDF_EVALUATIONS);
    const uint32_t sampleType = restirSampleType(conn.sample);
    if (sampleType == RESTIR_SAMPLE_ANALYTIC)
        auditWork(uniforms, WORK_NEE_FINITE_LIGHT_INSPECTIONS, 2u);
    else if (sampleType == RESTIR_SAMPLE_EMISSIVE_TRIANGLE)
        auditWork(uniforms, WORK_NEE_EMISSIVE_LIGHT_INSPECTIONS, 4u);
    else if (sampleType == RESTIR_SAMPLE_ENVIRONMENT)
        auditWork(uniforms, WORK_NEE_ENVIRONMENT_SAMPLES);
}

static bool restirDiagnosticVisible(constant Uniforms& uniforms,
                                    CurveStaticTraversal::structure accelerationStructure,
                                    CurveStaticTraversal::table functionTable,
                                    thread const LightConnection& connection,
                                    float3 shadowOrigin,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer,
                                    thread float& transmittance);

// Reserve adjacent shadow-queue entries with one global atomic per active
// SIMD-group instead of one per ray.
static inline uint32_t allocateShadowSlot(device atomic_uint* shadowCounter)
{
    const uint32_t rank = simd_prefix_exclusive_sum(1u);
    const uint32_t count = simd_sum(1u);
    uint32_t base = 0u;
    if (simd_is_first())
    {
        base = atomic_fetch_add_explicit(shadowCounter, count, memory_order_relaxed);
    }
    base = simd_broadcast_first(base);
    return base + rank;
}

static inline void storeShadowRay(device char* data, uint32_t index, uint32_t capacity, thread const ShadowRay& ray)
{
    if (SPEC_SHARC_UPDATE || SPEC_RESTIR || SPEC_RENDER_WORK_AUDIT)
    {
        ((device ShadowRay*)data)[index] = ray;
        return;
    }
    device CompactShadowTraversal* traversal = (device CompactShadowTraversal*)data;
    device CompactShadowContribution* contribution =
        (device CompactShadowContribution*)(data + sizeof(CompactShadowTraversal) * capacity);
    CompactShadowTraversal compactTraversal;
    compactTraversal.origin = ray.origin;
    compactTraversal.direction = ray.direction;
    compactTraversal.maxDistance = ray.maxDistance;
    compactTraversal.alphaThreshold = ray.alphaThreshold;
    traversal[index] = compactTraversal;
    CompactShadowContribution compactContribution;
    compactContribution.weight = ray.weight;
    compactContribution.pixelIndex = ray.pixelIndex;
    contribution[index] = compactContribution;
    if (SPEC_SSS)
    {
        device uint32_t* media =
            (device uint32_t*)(data + (sizeof(CompactShadowTraversal) + sizeof(CompactShadowContribution)) * capacity);
        media[index] = ray.medium;
    }
}

static inline CompactShadowTraversal loadShadowTraversal(device const char* data, uint32_t index)
{
    if (SPEC_SHARC_UPDATE || SPEC_RESTIR || SPEC_RENDER_WORK_AUDIT)
    {
        const device ShadowRay& ray = ((device const ShadowRay*)data)[index];
        CompactShadowTraversal traversal;
        traversal.origin = ray.origin;
        traversal.direction = ray.direction;
        traversal.maxDistance = ray.maxDistance;
        traversal.alphaThreshold = ray.alphaThreshold;
        return traversal;
    }
    return ((device const CompactShadowTraversal*)data)[index];
}

static inline CompactShadowContribution loadShadowContribution(device const char* data, uint32_t index, uint32_t capacity)
{
    if (SPEC_SHARC_UPDATE || SPEC_RESTIR || SPEC_RENDER_WORK_AUDIT)
    {
        const device ShadowRay& ray = ((device const ShadowRay*)data)[index];
        CompactShadowContribution contribution;
        contribution.weight = ray.weight;
        contribution.pixelIndex = ray.pixelIndex;
        return contribution;
    }
    const device char* contributionBase = data + sizeof(CompactShadowTraversal) * capacity;
    return ((device const CompactShadowContribution*)contributionBase)[index];
}

static inline uint32_t loadShadowMedium(device const char* data, uint32_t index, uint32_t capacity)
{
    if (SPEC_SHARC_UPDATE || SPEC_RESTIR || SPEC_RENDER_WORK_AUDIT)
    {
        return ((device const ShadowRay*)data)[index].medium;
    }
    const device char* mediumBase =
        data + (sizeof(CompactShadowTraversal) + sizeof(CompactShadowContribution)) * capacity;
    return ((device const uint32_t*)mediumBase)[index];
}

static inline uint32_t loadShadowPathIndex(device const char* data, uint32_t index, uint32_t pixelIndex)
{
    if (SPEC_SHARC_UPDATE || SPEC_RESTIR || SPEC_RENDER_WORK_AUDIT)
    {
        return ((device const ShadowRay*)data)[index].sharcPathIndex;
    }
    return pixelIndex;
}

static inline float3 loadShadowSharcRadiance(device const char* data, uint32_t index)
{
    if (SPEC_SHARC_UPDATE)
    {
        return float3(((device const ShadowRay*)data)[index].sharcRadiance);
    }
    return float3(0.0f);
}

static inline BaseLightConnectionPayload packBaseLightConnection(thread const LightConnection& connection)
{
    BaseLightConnectionPayload payload;
    payload.radiance = packed_float3(connection.radiance);
    payload.toLight = packed_float3(connection.toLight);
    payload.origin = packed_float3(connection.origin);
    payload.visibilityTarget = packed_float3(connection.visibilityTarget);
    payload.pdf = connection.pdf;
    payload.tMax = connection.tMax;
    payload.flags = (connection.needsRay ? BASE_LIGHT_CONNECTION_NEEDS_RAY : 0u) |
                    (connection.hasVisibilityTarget ? BASE_LIGHT_CONNECTION_VISIBILITY_TARGET : 0u) |
                    (connection.isDelta ? BASE_LIGHT_CONNECTION_DELTA : 0u);
    return payload;
}

static inline LightConnection unpackBaseLightConnection(device const BaseLightConnectionPayload& payload)
{
    LightConnection connection = makeEmptyConnection();
    connection.radiance = float3(payload.radiance);
    connection.toLight = float3(payload.toLight);
    connection.origin = float3(payload.origin);
    connection.visibilityTarget = float3(payload.visibilityTarget);
    connection.pdf = payload.pdf;
    connection.tMax = payload.tMax;
    connection.needsRay = (payload.flags & BASE_LIGHT_CONNECTION_NEEDS_RAY) != 0u;
    connection.hasVisibilityTarget = (payload.flags & BASE_LIGHT_CONNECTION_VISIBILITY_TARGET) != 0u;
    connection.isDelta = (payload.flags & BASE_LIGHT_CONNECTION_DELTA) != 0u;
    return connection;
}

struct PackedLightConnectionProbe
{
    packed_float3 radiance;
    uint2 toLight;
    uint2 originOffset;
    uint2 visibilityTargetOffset;
    float pdf;
    float tMax;
    uint32_t flags;
};

static inline uint2 packProbeHalf3(float3 value)
{
    return uint2(as_type<uint>(half2(value.xy)), uint(as_type<ushort>(half(value.z))));
}

static inline PackedLightConnectionProbe packLightConnectionProbe(thread const LightConnection& connection,
                                                                  float3 surfacePosition)
{
    PackedLightConnectionProbe payload;
    payload.radiance = packed_float3(connection.radiance);
    payload.toLight = packProbeHalf3(connection.toLight);
    payload.originOffset = packProbeHalf3(connection.origin - surfacePosition);
    payload.visibilityTargetOffset = packProbeHalf3(connection.visibilityTarget - surfacePosition);
    payload.pdf = connection.pdf;
    payload.tMax = connection.tMax;
    payload.flags = (connection.needsRay ? BASE_LIGHT_CONNECTION_NEEDS_RAY : 0u) |
                    (connection.hasVisibilityTarget ? BASE_LIGHT_CONNECTION_VISIBILITY_TARGET : 0u) |
                    (connection.isDelta ? BASE_LIGHT_CONNECTION_DELTA : 0u);
    return payload;
}

static inline float baseSampleStateChecksum(thread const OpenPBR_BasePreparedBsdf& prepared)
{
    const thread OpenPBR_BaseMicrofacetDistribution& distribution = prepared.specular_lobe.microfacet_distr;
    float checksum = dot(float2(distribution.alpha), float2(1.0f, 2.0f));
    checksum += dot(float3(distribution.tangent), float3(3.0f, 5.0f, 7.0f));
    checksum += dot(float3(distribution.bitangent), float3(11.0f, 13.0f, 17.0f));
    checksum += dot(float3(distribution.normal), float3(19.0f, 23.0f, 29.0f));
    checksum += float(distribution.isotropic_alpha) * 31.0f;
    checksum += float(prepared.specular_lobe.eta_t_over_eta_i_for_opaque_part) * 37.0f;
    checksum += float(prepared.specular_lobe.dielectric_amount) * 41.0f;
    checksum += float(prepared.specular_lobe.metal_amount) * 43.0f;
    checksum += prepared.metal_mms_lobe.energy_complement_idotn * 47.0f;
    checksum += prepared.diffuse_lobe.diffuse_roughness * 53.0f;
    checksum += prepared.diffuse_lobe.cached_specular_energy_compensation * 59.0f;

    uint packedHash = hash_combine(prepared.specular_lobe.specular_color.x, prepared.specular_lobe.specular_color.y);
    packedHash = hash_combine(packedHash, prepared.specular_lobe.f0_for_metal.x);
    packedHash = hash_combine(packedHash, prepared.specular_lobe.f0_for_metal.y);
    packedHash = hash_combine(packedHash, prepared.metal_mms_lobe.scale.x);
    packedHash = hash_combine(packedHash, prepared.metal_mms_lobe.scale.y);
    packedHash = hash_combine(packedHash, prepared.diffuse_lobe.diffuse_albedo.x);
    packedHash = hash_combine(packedHash, prepared.diffuse_lobe.diffuse_albedo.y);
    packedHash = hash_combine(packedHash, prepared.lobe_thresholds);
    return checksum + float(packedHash) * (1.0f / 4294967296.0f);
}

// Plain one-candidate NEE does not need material parameters to propose a light.
// Generate the proposal from extend's compact geometry before Base shade so the
// light sampler and OpenPBR prepare/eval never occupy the same register live set.
kernel void wavefrontConnectBase(uint gid [[thread_position_in_grid]],
                                 constant Uniforms& uniforms [[buffer(0)]],
                                 constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],
                                 device const IesGpuBufferHeader* iesProfiles [[buffer(2)]],
                                 device UniformLight* lights [[buffer(3)]],
                                 device const Material* materials [[buffer(4)]],
                                 device const PathState* paths [[buffer(5)]],
                                 device const PathRay* rays [[buffer(6)]],
                                 device char* hits [[buffer(7)]],
                                 constant uint32_t& sampleIdx [[buffer(8)]],
                                 device const uint32_t* queue [[buffer(9)]],
                                 device const uint32_t* control [[buffer(10)]],
                                 device const uint2* envAliasTable [[buffer(11)]],
                                 device const char* vertexBuffer [[buffer(12)]],
                                 device const char* prevVertexBuffer [[buffer(13)]],
                                 device const uint32_t* indexBuffer [[buffer(14)]],
                                 texture2d<float> envMapTexture [[texture(0)]])
{
    if (gid >= control[WF_CTRL_SHADE_LAYER_START])
    {
        return;
    }
    const uint32_t tid = queue[bucketedHitIndex<WF_SHADE_BASE>(gid, queue, control)];
    const SurfaceGeometryPayload geometry = uniforms.surfaceGeometry[tid];
    if ((geometry.tangentAndFlags & SURFACE_GEOMETRY_VALID) == 0u)
    {
        const LightConnection empty = makeEmptyConnection();
        uniforms.baseLightConnections[tid] = packBaseLightConnection(empty);
        return;
    }

    const PathRay pathRay = rays[tid];
    const float3 rayDirection = float3(pathRay.direction);
    const HitRecord hit = *wavefrontHitRecord(hits, tid);
    SurfaceInteraction lightSurface = {};
    lightSurface.position = float3(pathRay.origin) + rayDirection * hit.distance;
    lightSurface.shading_normal = unpackSurfaceNormal(geometry.shadingNormal);
    lightSurface.geometry_normal = unpackSurfaceNormal(geometry.geometryNormal);
    lightSurface.wo = -rayDirection;
    lightSurface.front_face = dot(lightSurface.geometry_normal, lightSurface.wo) > 0.0f;
    lightSurface.material_type = MATERIAL_TYPE_OPENPBR;

    const uint32_t depth = pathDepth(paths[tid].depthAndFlags);
    SamplerState rng = samplerForFixedDepth(uniforms, tid, sampleIdx, depth);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);
    const LightConnection connection =
        connectToLight(uniforms, uniforms.numLights, lights, instances, materials, vertexBuffer, prevVertexBuffer,
                       indexBuffer, motionTime, rng, lightSurface, envAliasTable, envMapTexture, iesProfiles);
    uniforms.baseLightConnections[tid] = packBaseLightConnection(connection);
}

template <uint ShadeBucket>
static inline void wavefrontShadeImpl(uint gid,
                                      constant Uniforms& uniforms,
                                      constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                      device const IesGpuBufferHeader* iesProfiles,
                                      device UniformLight* lights,
                                      device Material* materials,
                                      device PathState* paths,
                                      device PathRay* rays,
                                      device char* hits,
                                      device float4* radianceOut,
                                      device IorStack* iorStacks,
                                      device const GeometryEntry* geometryEntries,
                                      device const uint2* envAliasTable,
                                      device const char* vertexBuffer,
                                      device const char* prevVertexBuffer,
                                      device const uint32_t* indexBuffer,
                                      constant uint32_t& sampleIdx,
                                      device const uint32_t* queue,
                                      device uint32_t* queueOut,
                                      device atomic_uint* outCounter,
                                      device uint32_t* control,
                                      device char* shadowRays,
                                      device atomic_uint* shadowCounter,
                                      device AovSample* aov,
                                      device char* sharcPassBuffer0,
                                      device const char* sharcPassBuffer1,
                                      CurveStaticTraversal::structure diagnosticAccelerationStructure,
                                      CurveStaticTraversal::table diagnosticFunctionTable,
                                      device const uint32_t* curveSegments,
                                      device atomic_uint* iorStats,
                                      device char* sharcPassState,
                                      device MediumPathState* mediumPaths,
                                      texture2d<float> envMapTexture)
{
    constexpr bool kShadeBase = ShadeBucket == WF_SHADE_BASE;
    constexpr bool kShadeLayer = ShadeBucket == WF_SHADE_LAYER;
    constexpr bool kShadeTranslucent = ShadeBucket == WF_SHADE_TRANSLUCENT;
    constexpr bool kShadeTail = ShadeBucket == WF_SHADE_TAIL;
    constexpr bool kHandlesSpecialHits = kShadeTail || ShadeBucket == WF_SHADE_GENERIC;
    gid += kShadeTail ? control[WF_CTRL_SHADE_TAIL_START] :
                        (kShadeTranslucent ? control[WF_CTRL_SHADE_TRANSLUCENT_START] :
                                             (kShadeLayer ? control[WF_CTRL_SHADE_LAYER_START] : 0u));
    const uint32_t segmentEnd =
        kShadeTail ? control[WF_CTRL_HIT_N] :
                     (kShadeTranslucent ?
                          control[WF_CTRL_SHADE_TAIL_START] :
                          (kShadeLayer ? control[WF_CTRL_SHADE_TRANSLUCENT_START] :
                                         (kShadeBase ? control[WF_CTRL_SHADE_LAYER_START] : control[WF_CTRL_HIT_N])));
    if (gid >= segmentEnd)
    {
        return;
    }
    const uint32_t tid = queue[bucketedHitIndex<ShadeBucket>(gid, queue, control)];
    device SharcHashEntry* sharcHashEntries = (device SharcHashEntry*)uniforms.sharcHashData;
    device const packed_float3* curvePoints = (device const packed_float3*)uniforms.curvePointData;
    device const char* prevFrameVertexBuffer = sharcPassBuffer0;
    device const MTLIndirectAccelerationStructureInstanceDescriptor* prevInstances =
        (device const MTLIndirectAccelerationStructureInstanceDescriptor*)sharcPassBuffer1;
    device SharcUpdateState* sharcUpdates = (device SharcUpdateState*)sharcPassState;
    device SharcAccumulationEntry* sharcAccumulation = (device SharcAccumulationEntry*)sharcPassBuffer0;
    device const SharcResolvedEntry* sharcResolved = SPEC_SHARC_UPDATE ?
                                                         (device const SharcResolvedEntry*)sharcPassBuffer1 :
                                                         (device const SharcResolvedEntry*)sharcPassState;
    PathState p = paths[tid];
    MediumPathState mediumState = {};
    if (SPEC_SSS)
    {
        mediumState = mediumPaths[tid];
    }
    const PathRay pr = rays[tid];

    const uint32_t depth = pathDepth(p.depthAndFlags);
    auditWork(uniforms, WORK_SHADE_ITEMS_BASE + min(depth, WORK_BOUNCE_SLOTS - 1u));
    const bool specularBounce = (p.depthAndFlags & PATH_FLAG_SPECULAR) != 0u;
    const bool neeDone = (p.depthAndFlags & PATH_FLAG_NEE_DONE) != 0u;

    // Bounce heatmap. Every stage that sees a path records how deep it was, so
    // whichever one it dies in has left the answer behind.
    if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcBounces)
    {
        aov[tid].guideStateOrBounceDepth = (float)depth;
    }

    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);

    const float3 rayOrigin = float3(pr.origin);
    const float3 rayDir = float3(pr.direction);
    float3 throughput = float3(p.throughput);
    const float3 throughputAtStageEntry = throughput;
    const float3 sampledThroughput = throughput;

    float3 radiance = float3(0.0f);
    const HitRecord rec = *wavefrontHitRecord(hits, tid);

    // Apply enclosing IOR absorption once before all vertex branches; rec.distance is the segment just travelled.
    // Subsurface walks carry their own extinction, while bounded volumes can still overlap enclosing glass.
    const bool hasIorStack = (p.depthAndFlags & PATH_FLAG_IOR_STACK_ACTIVE) != 0u;
    {
        const uint32_t walk = mediumState.medium & MEDIUM_INDEX_MASK;
        const bool inSubsurfaceWalk =
            SPEC_SSS && walk != 0u && (materials[walk - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u;
        if (!inSubsurfaceWalk && hasIorStack)
        {
            // The active bit guarantees a valid head; only these paths touch
            // the strided IOR side table.
            const int iorTop = iorStacks[tid].top;
            const uint32_t inside = iorStacks[tid].entries[iorTop].packed & IOR_ENTRY_MATERIAL_MASK;
            device const Material& im = materials[inside];
            const float3 sigma_t =
                volume_extinction(float3(im.attenuation_color), im.attenuation_distance, uniforms.volumeModel);
            throughput *= beer_lambert_transmittance(sigma_t, rec.distance);
            // Every branch below that persists the path either recomputes this
            // or writes `p` unchanged, so it is written once here.
            p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        }
    }

    if (kHandlesSpecialHits && SPEC_FOG && (rec.geomEntryIndex & HIT_FOG_BIT) != 0u)
    {
        SamplerState rng = samplerFor(uniforms, tid, sampleIdx, depth);
        const float3 scatterPoint = rayOrigin + rayDir * rec.distance;
        throughput *= float3(uniforms.fogAlbedo);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            sharcUpdates[updateIndex] = updateState;
        }

        // A medium event has a position and no normal, which is what the
        // volumeEvent flag tells the light connection.
        SurfaceInteraction si = {};
        si.position = scatterPoint;
        si.shading_normal = -rayDir;
        si.geometry_normal = -rayDir;
        si.front_face = true;

        const bool didNee = volumeNeePairsWithBounce(
            uniforms.estimatorMode == 0,
            (SPEC_LIGHTS && (uniforms.numLights > 0 || (SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0))) ||
                (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (didNee)
        {
            auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                        vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, rng,
                                                        si, envAliasTable, envMapTexture, iesProfiles, true);
            auditNeeSampledConnection(uniforms, conn);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                // The phase cosine compares travel directions: dot(rayDir, toLight).
                const float phase = hgPhase(dot(rayDir, conn.toLight), uniforms.fogAnisotropy);
                // The phase function is the medium's BSDF and its own pdf, so
                // MIS pairs it against the light density exactly as a surface
                // lobe would.
                const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, uniforms.misHeuristic);
                const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                if (any(weight > 1e-6f) && visibility.valid)
                {
                    auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                    ShadowRay sr;
                    sr.origin = packed_float3(scatterPoint);
                    sr.direction = packed_float3(visibility.direction);
                    sr.weight =
                        packed_float3(clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect));
                    sr.maxDistance = visibility.maxDistance;
                    sr.pixelIndex = tid;
                    sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                    sr.sharcRadiance =
                        packed_float3(SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * phase : float3(0.0f));
                    sr.sharcPathIndex = tid;
                    sr.alphaThreshold = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType);
                    const uint32_t slot = allocateShadowSlot(shadowCounter);
                    auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                    storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
                }
            }
            else if (conn.needsRay)
            {
                auditWork(uniforms, WORK_NEE_REJECT_PDF);
            }
        }

        float phasePdf = 0.0f;
        const float2 phaseRandom =
            random2<SampleDimension::eFogPhaseU, SampleDimension::eFogPhaseV>(rng, uniforms.samplerType).value;
        const float3 nextDir = hgSample(-rayDir, uniforms.fogAnisotropy, phaseRandom.x, phaseRandom.y, phasePdf);

        addFilteredRadiance(radianceOut, tid, radiance);

        // Roulette on the medium's albedo, which is the only thing multiplying
        // the throughput here. Without it a thin haze is a very long random walk
        // that contributes almost nothing per step.
        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(rng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;

        // Depth advances: a scattering event is a bounce, and a medium with no
        // depth budget of its own would let a path wander forever.
        p.depthAndFlags =
            (depth + 1u) | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNee ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    if (kHandlesSpecialHits && SPEC_SSS && (rec.geomEntryIndex & HIT_SSS_BIT) != 0u)
    {
        const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = mediumState.medium >> MEDIUM_STEP_SHIFT;
        device const Material& mm = materials[medium - 1u];
        const MediumProps mp = mediumPropsFor(uniforms, materials, medium - 1u, mediumState.mediumAlbedo);
        const float3 sigmaT = mp.sigmaT;
        const bool isBoundedMedium = (mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        const float3 albedo = mp.albedo;

        throughput *= sssScatterWeight(sigmaT, albedo, sssChannelPdf(sampledThroughput, albedo), rec.distance);

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            if (isBoundedMedium)
            {
                sharcPropagate(updateState, sharcAccumulation, float3(mm.medium_emission), uniforms,
                               (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
            }
            sharcUpdates[updateIndex] = updateState;
        }

        const float3 scatterPoint = rayOrigin + rayDir * rec.distance;
        SamplerState wrng = samplerFor(uniforms, tid, sampleIdx, depth);
        wrng.depth = depth + step;

        const bool isBounded = isBoundedMedium;
        // As in the fog path: available, not delivered.
        const bool didNeeVolume =
            isBounded &&
            volumeNeePairsWithBounce(uniforms.estimatorMode == 0,
                                     (SPEC_LIGHTS && (uniforms.numLights > 0 ||
                                                      (SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0))) ||
                                         (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (isBounded)
        {
            radiance += clampPathContribution(
                throughput * float3(mm.medium_emission), depth, uniforms.clampDirect, uniforms.clampIndirect);

            if (didNeeVolume)
            {
                auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
                SurfaceInteraction vsi = {};
                vsi.position = scatterPoint;
                vsi.shading_normal = -rayDir;
                vsi.geometry_normal = -rayDir;
                vsi.front_face = true;
                const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                            vertexBuffer, prevVertexBuffer, indexBuffer, motionTime,
                                                            wrng, vsi, envAliasTable, envMapTexture, iesProfiles, true);
                auditNeeSampledConnection(uniforms, conn);
                if (conn.needsRay && conn.pdf > 0.0f)
                {
                    // Match the fog path's travel-direction phase convention.
                    const float phase = hgPhase(dot(rayDir, conn.toLight), mm.subsurface_anisotropy);
                    const float misWeight =
                        conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, uniforms.misHeuristic);
                    const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                    if (any(weight > 1e-6f) && visibility.valid)
                    {
                        auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                        ShadowRay sr;
                        sr.origin = packed_float3(scatterPoint);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(
                            clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                        sr.sharcRadiance = packed_float3(
                            SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * phase : float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.alphaThreshold = random<SampleDimension::eShadowRR>(wrng, uniforms.samplerType);
                        const uint32_t slot = allocateShadowSlot(shadowCounter);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
                    }
                }
                else if (conn.needsRay)
                {
                    auditWork(uniforms, WORK_NEE_REJECT_PDF);
                }
            }
        }

        float phasePdf = 0.0f;
        const float2 phaseRandom =
            random2<SampleDimension::eSssPhaseU, SampleDimension::eSssPhaseV>(wrng, uniforms.samplerType).value;
        const float3 nextDir = hgSample(-rayDir, mm.subsurface_anisotropy, phaseRandom.x, phaseRandom.y, phasePdf);

        addFilteredRadiance(radianceOut, tid, radiance);

        // Roulette on what the walk has left. The step ceiling is a backstop for
        // a medium dense enough that roulette alone would take thousands of
        // steps to end; this is what actually terminates the walk.
        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(wrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(scatterPoint);
        nextRay.direction = packed_float3(nextDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = phasePdf;
        p.misDistance = 0.0f;
        // Dense walks advance only their step counter; the whole walk consumes one path bounce.
        mediumState.medium = medium | ((step + 1u) << MEDIUM_STEP_SHIFT);
        // Bounded-volume scattering advances path depth so it cannot wander without a bounce budget.
        const uint32_t nextDepth = isBounded ? (depth + 1u) : depth;
        p.depthAndFlags =
            nextDepth | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNeeVolume ? PATH_FLAG_NEE_DONE : 0u);
        if (nextDepth >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // --- Emissive geometry --------------------------------------------------
    if (kHandlesSpecialHits && SPEC_LIGHTS && (rec.geomEntryIndex & HIT_LIGHT_BIT) != 0u)
    {
        const uint32_t lightId = rec.geomEntryIndex & ~HIT_LIGHT_BIT;
        device const UniformLight& currLight = lights[lightId];
        // Traversal already decided the nearest surface and returned its
        // distance. Reconstruct the point and only evaluate normal/PDF; running
        // the full ray/surface intersection here made the same decision twice.
        const float3 hitPoint = rayOrigin + rayDir * rec.distance;
        float3 lightNormal;
        float lightAreaPdf;
        if (currLight.type == LIGHT_TYPE_SPHERE)
        {
            lightAreaPdf = analyticEllipsoidAreaPdfUnchecked(float3(currLight.points[1]), float3(currLight.points[0]),
                                                             float3(currLight.points[2]), float3(currLight.points[3]),
                                                             hitPoint, lightNormal);
        }
        else
        {
            lightNormal = calcLightNormal(currLight, hitPoint);
            lightAreaPdf = calcLightAreaPdf(currLight, hitPoint);
        }
        // A light's geometry is still a surface the denoiser has to reconstruct.
        // Its emission is unaffected by denoising, so it gets a black albedo and
        // its own geometry, which keeps the guides continuous across the edge.
        if (shouldWriteAov(uniforms, sampleIdx) && depth == 0u)
        {
            AovSample a;
            a.diffuseAlbedo = packed_float3(float3(0.0f));
            a.specularAlbedo = packed_float3(float3(0.0f));
            a.normal = packed_float3(-rayDir);
            a.roughness = 1.0f;
            a.depth = viewDepth(uniforms, hitPoint);
            // Analytic lights do not move, so the camera is the only thing that
            // can have displaced them.
            const ScreenMotion motion = screenMotion(uniforms, uniforms.prevWorldToClip * float4(hitPoint, 1.0f),
                                                     uint2(tid % uniforms.width, tid / uniforms.width));
            a.motionX = motion.offset.x;
            a.motionY = motion.offset.y;
            a.specularHitDistance = 0.0f;
            a.reactive = motion.reactive;
            // A directly visible emitter is deterministic. Let MetalFX scale it
            // temporally but do not ask the denoiser to suppress its energy.
            a.guideStateOrBounceDepth = -1.0f;
            aov[tid] = a;
        }
        else if (shouldWriteAov(uniforms, sampleIdx) && (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
        {
            // An emitter reached through glass is a valid replacement surface,
            // just with no material albedo of its own. Retain the primary
            // interface's Fresnel share and reprojection data.
            const float state = aov[tid].guideStateOrBounceDepth;
            const float interfaceFresnel = state >= 1.0f ? saturate(state - 1.0f) : 0.0f;
            AovSample a;
            a.diffuseAlbedo = packed_float3(float3(0.0f));
            a.specularAlbedo = packed_float3(float3(interfaceFresnel));
            a.normal = packed_float3(-rayDir);
            a.roughness = mix(aov[tid].roughness, 1.0f, 1.0f - interfaceFresnel);
            a.depth = aov[tid].depth;
            a.motionX = aov[tid].motionX;
            a.motionY = aov[tid].motionY;
            a.specularHitDistance = aov[tid].specularHitDistance;
            a.reactive = 0.0f;
            a.guideStateOrBounceDepth = 0.0f;
            aov[tid] = a;
        }
        if (shouldWriteAov(uniforms, sampleIdx) && depth > 0u && specularBounce &&
            (p.depthAndFlags & PATH_FLAG_AOV_DONE) == 0u)
        {
            aov[tid].specularHitDistance += rec.distance;
        }
        if (!SPEC_SHARC_UPDATE && depth == 0u)
        {
            if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcGrid)
            {
                radianceOut[tid] = float4(sharcDebugColoredHash(uniforms, hitPoint, lightNormal), 0.0f);
                return;
            }
            if (SPEC_DEBUG && SPEC_SHARC && uniforms.sharcCapacity != 0u &&
                (DebugMode)uniforms.debug == DebugMode::eSharcRadiance)
            {
                radianceOut[tid] = float4(sharcDebugRadiance(uniforms, sharcHashEntries, sharcResolved, hitPoint,
                                                             lightNormal, -rayDir, float3(1.0f)),
                                          0.0f);
                return;
            }
            if (SPEC_SHARC && uniforms.sharcCapacity != 0u && SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug))
            {
                radianceOut[tid] = float4(sharcDebugSurface(uniforms, sharcHashEntries, sharcResolved, hitPoint,
                                                            lightNormal, -rayDir, float3(1.0f)),
                                          0.0f);
                return;
            }
        }
        float3 sharcLight = float3(0.0f);
        const float3 scatteringOrigin = rayOrigin - rayDir * p.misDistance;
        const float lightCosine = -dot(rayDir, lightNormal);
        if (analyticLightVisibilityAllowsRay(currLight.normal.w, depth != 0u) &&
            lightConnectionFacesVertex(currLight.type, lightCosine, 0.0f))
        {
            const float hitDistance = finiteVectorLength(hitPoint - scatteringOrigin);
            const float3 Le = emittedLightRadiance(currLight, -rayDir, hitDistance, iesProfiles);
            float3 weightedLe;
            if (depth == 0u || specularBounce || !neeDone)
            {
                weightedLe = Le;
            }
            else
            {
                const float localSelectionPdf = uniforms.hasEnvMap ? 1.0f - uniforms.envMapColorTint.w : 1.0f;
                const float analyticClassPdf = SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0u ?
                                                   1.0f - uniforms.meshLightSelectionPdf :
                                                   1.0f;
                const float lightIdentityPdf = analyticLightSelectionPdf(currLight);
                const float lightPdf = areaPdfToSolidAngleMarginalPdf(
                    hitDistance, lightCosine, lightAreaPdf, localSelectionPdf, analyticClassPdf, lightIdentityPdf, 1.0f);
                const float mis = computeMisWeight(p.lastBsdfPdf, lightPdf, uniforms.misHeuristic);
                weightedLe = Le * mis;
            }
            radiance += throughput * weightedLe;
            sharcLight += weightedLe;
        }
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcUpdateMiss(updateState, uniforms, sharcAccumulation, sharcLight, -rayDir);
            sharcUpdates[updateIndex] = updateState;
        }
        addFilteredRadiance(
            radianceOut, tid, clampPathContribution(radiance, depth, uniforms.clampDirect, uniforms.clampIndirect));
        return;
    }

    // --- Surface ------------------------------------------------------------
    const GeometryEntry entry = geometryEntries[rec.geomEntryIndex];
    const bool interpolateMotion =
        SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer && indexBuffer;

    const float2 bary = unpackHitBarycentrics(rec.barycentrics);
    const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;
    float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
    float2 uv;
    bool uvPretransformed = false;
    float tangentSign = 1.0f;
    float lodBase = -1e30f;
    // Only a curve hit has one, and only the fibre paths below read it.
    float curveRadius = 0.0f;

    const float3 worldPosition = rayOrigin + rayDir * rec.distance;

    float3 shadingNormal, shadingTangent, shadingGeomNormal;
    if (isCurve)
    {
        const float4x4 objectToWorld =
            geometryObjectToWorld(uniforms, instances, rec.instanceIndex, rec.geomEntryIndex, entry);
        fetchCurve(curvePoints, curveSegments, entry, rec.primitiveId, bary.x, worldPosition, objectToWorld,
                   shadingNormal, shadingTangent, uv, curveRadius);
        // A strand has no separate geometric normal: the surface *is* the
        // cylinder, so the shading normal is the geometric one.
        shadingGeomNormal = shadingNormal;
        vertexColor = float3(1.0f);
    }
    else
    {
        const SurfaceGeometryPayload preparedGeometry = uniforms.surfaceGeometry[tid];
        if (!kHandlesSpecialHits || (preparedGeometry.tangentAndFlags & SURFACE_GEOMETRY_VALID) != 0u)
        {
            shadingNormal = unpackSurfaceNormal(preparedGeometry.shadingNormal);
            shadingGeomNormal = unpackSurfaceNormal(preparedGeometry.geometryNormal);
            shadingTangent = orthonormalizeTangent(shadingNormal, unpackSurfaceTangent(preparedGeometry.tangentAndFlags));
            tangentSign = unpackSurfaceTangentSign(preparedGeometry.tangentAndFlags);
            uv = unpackSurfaceUv(preparedGeometry.uvAndLod);
            uvPretransformed = (entry.flags & GEOM_FLAG_SURFACE_UV) != 0u;
            vertexColor = float3(1.0f);
            lodBase = unpackSurfaceLod(preparedGeometry.uvAndLod);
        }
        else
        {
            float uvArea2 = 0.0f;
            fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId, interpolateMotion,
                                 motionTime, bary, objectNormal, objectTangent, uv, vertexColor, tangentSign,
                                 objectGeomNormal, uvArea2);
            const float4x4 objectToWorld =
                geometryObjectToWorld(uniforms, instances, rec.instanceIndex, rec.geomEntryIndex, entry);
            const float3 axisX = objectToWorld[0].xyz;
            const float3 axisY = objectToWorld[1].xyz;
            const float3 axisZ = objectToWorld[2].xyz;
            const FastNormalTransform normalTransform = makeFastNormalTransform(axisX, axisY, axisZ);
            shadingNormal = transformNormalFast(objectNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                                normalTransform.cofactorZ, normalTransform.orientation);
            const float3 worldGeomNormal = normalTransform.cofactorX * objectGeomNormal.x +
                                           normalTransform.cofactorY * objectGeomNormal.y +
                                           normalTransform.cofactorZ * objectGeomNormal.z;
            const float worldArea2 = length(worldGeomNormal);
            shadingGeomNormal = normalTransform.orientation * (worldGeomNormal / worldArea2);
            shadingTangent = orthonormalizeTangent(shadingNormal, transformDirection(objectTangent, axisX, axisY, axisZ));

            const bool primarySurface = depth == 0u;
            const float cameraPathDistance = rec.distance + (primarySurface ? p.misDistance : 0.0f);
            const float coneWidthHere = surfaceRayFootprint(uniforms, rayDir, cameraPathDistance, primarySurface);
            if (SPEC_TEXTURE_LOD_CODE && (uniforms.textureLodMode & TEXTURE_LOD_MODE_MASK) != 0u && uvArea2 > 0.0f &&
                worldArea2 > 1e-20f && coneWidthHere > 0.0f)
            {
                const float ndotd = max(abs(dot(shadingGeomNormal, rayDir)), 1e-4f);
                const float uvLodOffset = (entry.flags & GEOM_FLAG_SURFACE_UV) != 0u ?
                                              surfaceUvLodOffset(uniforms, materials, entry.materialId) :
                                              0.0f;
                lodBase = 0.5f * log2(uvArea2 / worldArea2) + log2(coneWidthHere) - log2(ndotd) + uvLodOffset +
                          uniforms.textureLodBias;
            }
        }
    }

    const float3 worldNormal = shadingNormal;
    const float3 worldTangent = shadingTangent;
    // glTF TANGENT.w. Without it the bitangent points the wrong way and every
    // normal map is mirrored along it -- bumps light from the opposite side.
    const float3 worldBinormal = cross(worldNormal, worldTangent) * tangentSign;

    const float3 geomNormal = shadingGeomNormal;

    if (kHandlesSpecialHits && SPEC_SSS && (materials[entry.materialId].medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
    {
        const uint32_t passes = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
        if (passes >= PATH_PASSTHROUGH_MAX)
        {
            addFilteredRadiance(radianceOut, tid, radiance);
            return;
        }
        p.depthAndFlags = (p.depthAndFlags & ~PATH_PASSTHROUGH_MASK) |
                          (((passes + 1u) << PATH_PASSTHROUGH_SHIFT) & PATH_PASSTHROUGH_MASK);

        const uint32_t here = (entry.materialId + 1u) & MEDIUM_INDEX_MASK;
        const bool leaving = (mediumState.medium & MEDIUM_INDEX_MASK) == here;
        mediumState.medium = leaving ? 0u : here;

        // Push past the surface on the side the ray is heading, which needs the
        // sign of the normal and not its direction.
        const float3 exitSide = (dot(rayDir, geomNormal) > 0.0f) ? geomNormal : -geomNormal;

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(worldPosition, exitSide));
        nextRay.direction = packed_float3(rayDir);
        rays[tid] = nextRay;

        p.misDistance += rec.distance;
        addFilteredRadiance(radianceOut, tid, radiance);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcUpdates[updateIndex] = updateState;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    if (kHandlesSpecialHits && SPEC_SSS && (mediumState.medium & MEDIUM_INDEX_MASK) != 0u &&
        (materials[(mediumState.medium & MEDIUM_INDEX_MASK) - 1u].medium_flags & MEDIUM_FLAG_BOUNDARY) == 0u)
    {
        const uint32_t medium = mediumState.medium & MEDIUM_INDEX_MASK;
        const uint32_t step = mediumState.medium >> MEDIUM_STEP_SHIFT;
        const float3 exitAlbedo = unpackMediumAlbedo(mediumState.mediumAlbedo);
        throughput *= sssBoundaryWeight(
            mediumSigmaT(uniforms, materials, medium - 1u), sssChannelPdf(sampledThroughput, exitAlbedo), rec.distance);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
            sharcApplyPendingThroughput(updateState, uniforms);
            sharcUpdates[updateIndex] = updateState;
        }

        // Use the geometric normal for offsets and the interpolated normal for the exit lobe and light connection.
        const float3 outwardGeom = (dot(geomNormal, rayDir) > 0.0f) ? geomNormal : -geomNormal;
        const float3 outward = (dot(worldNormal, outwardGeom) > 0.0f) ? worldNormal : -worldNormal;
        SamplerState xrng = samplerFor(uniforms, tid, sampleIdx, depth);
        xrng.depth = depth + step;

        // As in the fog path: available, not delivered. The exit lobe is a cosine
        // hemisphere, which is smooth at every parameter, so there is nothing
        // about the material to ask.
        const bool didNeeExit = volumeNeePairsWithBounce(
            uniforms.estimatorMode == 0,
            (SPEC_LIGHTS && (uniforms.numLights > 0 || (SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0))) ||
                (SPEC_ENV_MAP && uniforms.hasEnvMap));
        if (didNeeExit)
        {
            auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
            // NEE here and not inside the walk: this is the vertex light can
            // actually reach, and leaving it to BSDF sampling alone is what makes
            // a translucent object the noisiest thing in a frame.
            SurfaceInteraction xsi = {};
            xsi.position = worldPosition;
            xsi.shading_normal = outward;
            xsi.geometry_normal = outward;
            xsi.wo = -rayDir;
            xsi.front_face = true;
            const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                        vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, xrng,
                                                        xsi, envAliasTable, envMapTexture, iesProfiles, false);
            auditNeeSampledConnection(uniforms, conn);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                const float cosOut = dot(outward, conn.toLight);
                if (cosOut > 0.0f)
                {
                    // The exit lobe is 1/pi; cos/pi is only its MIS density because conn.radiance includes the cosine.
                    const float lobePdf = cosOut * M_1_PI_F;
                    const float misWeight =
                        conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, lobePdf, uniforms.misHeuristic);
                    const float3 weight = throughput * (conn.radiance / conn.pdf) * misWeight * M_1_PI_F;
                    const float3 shadowOrigin = offset_ray(worldPosition, outwardGeom);
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, shadowOrigin);
                    if (any(weight > 1e-6f) && visibility.valid)
                    {
                        auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                        ShadowRay sr;
                        sr.origin = packed_float3(shadowOrigin);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(
                            clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        sr.medium = 0u;
                        sr.sharcRadiance = packed_float3(
                            SPEC_SHARC_UPDATE ? (conn.radiance / conn.pdf) * misWeight * M_1_PI_F : float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.alphaThreshold = random<SampleDimension::eShadowRR>(xrng, uniforms.samplerType);
                        const uint32_t slot = allocateShadowSlot(shadowCounter);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
                    }
                }
            }
            else if (conn.needsRay)
            {
                auditWork(uniforms, WORK_NEE_REJECT_PDF);
            }
        }

        const float2 phaseRandom =
            random2<SampleDimension::eSssPhaseU, SampleDimension::eSssPhaseV>(xrng, uniforms.samplerType).value;
        const float3 exitDir = sssCosineDirection(outward, phaseRandom.x, phaseRandom.y);

        addFilteredRadiance(radianceOut, tid, radiance);

        const float survive = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05f, 1.0f);
        if (random<SampleDimension::eRussianRoulette>(xrng, uniforms.samplerType) >= survive)
        {
            return;
        }
        throughput /= survive;

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, float3(1.0f / max(survive, 1e-5f)));
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(worldPosition, outwardGeom));
        nextRay.direction = packed_float3(exitDir);
        rays[tid] = nextRay;

        p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : throughput);
        p.lastBsdfPdf = fmax(dot(outward, exitDir), 0.0f) * M_1_PI_F;
        p.misDistance = 0.0f;
        mediumState.medium = 0u;
        // Depth advances once for the whole walk, here rather than at the entry:
        // charging it at both ends would cost a translucent surface two bounces
        // to do what an opaque one does in one.
        p.depthAndFlags =
            (depth + 1u) | PATH_FLAG_ALIVE |
            (p.depthAndFlags & ~(PATH_DEPTH_MASK | PATH_FLAG_ALIVE | PATH_FLAG_SPECULAR | PATH_FLAG_NEE_DONE)) |
            (didNeeExit ? PATH_FLAG_NEE_DONE : 0u);
        if (depth + 1u >= uniforms.maxDepth)
        {
            return;
        }
        paths[tid] = p;
        mediumPaths[tid] = mediumState;
        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    device const Material& hitMaterial = materials[entry.materialId];
    const bool isOpenPBR = SPEC_ALL_OPENPBR || (SPEC_OPENPBR && hitMaterial.material_type == MATERIAL_TYPE_OPENPBR);
    const bool nativeOpenPBR =
        SPEC_ALL_NATIVE_OPENPBR || (isOpenPBR && (hitMaterial.features & MATERIAL_FEATURE_NATIVE_OPENPBR) != 0u);

    SurfaceInteraction si;
    if (nativeOpenPBR)
    {
        initSurfaceGeometry(si, worldPosition, worldNormal, geomNormal, worldTangent, worldBinormal, uv, rayDir);
        initOpenPBRSurfaceMaterial(si, uniforms.openpbrParams[entry.materialId], vertexColor);
        si.opacity = resolveOpacity(hitMaterial, uv, uvPretransformed);
    }
    else
    {
        initSurfaceInteraction(si, hitMaterial, worldPosition, worldNormal, geomNormal, worldTangent, worldBinormal, uv,
                               rayDir, vertexColor, lodBase, uvPretransformed);
    }

    const bool fibreMaterial = !isOpenPBR && scattersThroughFibre(si);
    const bool isFibre = isCurve && fibreMaterial;

    // None of the surface setup above consumes randomness. Keep these five
    // registers out of triangle reconstruction and material initialisation;
    // the fog and volume exits construct their own branch-local sampler state.
    SamplerState rng = samplerForFixedDepth(uniforms, tid, sampleIdx, depth);

    if (si.opacity < 1.0f)
    {
        const uint32_t layer = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
        SamplerState orng = rng;
        float u = random<SampleDimension::eOpacity>(orng, uniforms.samplerType);
        if (layer != 0u)
        {
            uint32_t r = sharcHash(layer * 2654435761u + tid * 2246822519u);
            u = fract(u + float(r) * (1.0f / 4294967296.0f));
        }
        if (u >= si.opacity)
        {
            const uint32_t passes = (p.depthAndFlags & PATH_PASSTHROUGH_MASK) >> PATH_PASSTHROUGH_SHIFT;
            addFilteredRadiance(radianceOut, tid, radiance);
            if (passes >= PATH_PASSTHROUGH_MAX)
            {
                return;
            }
            // Step off the surface on the side the ray was travelling, so the
            // next trace cannot re-hit the triangle it just passed through.
            const float3 faceNg = dot(geomNormal, rayDir) > 0.0f ? geomNormal : -geomNormal;
            PathRay through;
            through.origin = packed_float3(offset_ray(worldPosition, faceNg));
            through.direction = packed_float3(rayDir);
            rays[tid] = through;
            p.misDistance += rec.distance;
            p.depthAndFlags = (p.depthAndFlags & ~PATH_PASSTHROUGH_MASK) |
                              (((passes + 1u) << PATH_PASSTHROUGH_SHIFT) & PATH_PASSTHROUGH_MASK);
            if (SPEC_SHARC_UPDATE)
            {
                const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
                SharcUpdateState updateState = sharcUpdates[updateIndex];
                sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
                sharcUpdates[updateIndex] = updateState;
                p.throughput = packed_float3(float3(1.0f));
            }
            paths[tid] = p;
            auditWork(uniforms, WORK_PATH_CONTINUATIONS);
            queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
            return;
        }
    }

    OpenPBR_BaseParams openpbrBaseMat;
    OpenPBRParams openpbrMat;
    if (isOpenPBR)
    {
        device const OpenPBRParams& source = uniforms.openpbrParams[entry.materialId];
        float bumpUvFootprint = -1.0f;
        float bumpWorldFootprint = -1.0f;
        if (!isCurve && lodBase > -1e29f && source.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
        {
            device const OpenPBRTextures& textures = uniforms.openpbrTextures[entry.materialId];
            const uint32_t layeredOutputs = textures.layered.output_mask;
            if ((layeredOutputs & OPENPBR_LAYER_OUTPUT_NORMAL) != 0u)
            {
                const bool primarySurface = depth == 0u;
                const float cameraPathDistance = rec.distance + (primarySurface ? p.misDistance : 0.0f);
                bumpWorldFootprint = surfaceRayFootprint(uniforms, rayDir, cameraPathDistance, primarySurface);
                const float ndotd = max(abs(dot(geomNormal, rayDir)), 1.0e-4f);
                // lodBase already contains UV/world density, footprint, grazing correction and bias.
                // Undo the last two terms to recover a compact UV footprint without refetching the triangle.
                bumpUvFootprint = exp2(lodBase - uniforms.textureLodBias) * ndotd;
            }
        }
        if (kShadeBase)
        {
            loadOpenPBRBaseParams(source, openpbrBaseMat);
            if (source.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
            {
                applyOpenPBRBaseTextures(openpbrBaseMat, source, uniforms.openpbrTextures[entry.materialId], si, uv,
                                         lodBase, uvPretransformed, bumpUvFootprint, bumpWorldFootprint);
            }
        }
        else
        {
            openpbrMat = source;
            if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
            {
                applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv, lodBase,
                                     uvPretransformed, bumpUvFootprint, bumpWorldFootprint);
            }
        }
    }

    const DebugMode debugMode = (DebugMode)uniforms.debug;
    if (!SPEC_SHARC_UPDATE && SPEC_DEBUG && debugMode == DebugMode::eSharcGrid && depth == 0u)
    {
        radianceOut[tid] = float4(sharcDebugColoredHash(uniforms, si.position, si.geometry_normal), 0.0f);
        return;
    }
    if (SPEC_DEBUG && (debugMode == DebugMode::eMotionBlur || debugMode == DebugMode::eNormal))
    {
        // Debug views replace the radiance outright rather than accumulating.
        float3 dbg;
        if (debugMode == DebugMode::eNormal)
        {
            dbg = (si.shading_normal + float3(1.0f)) * 0.5f;
        }
        else
        {
            // Visualize displacement between the motion-interpolated and current-frame normals.
            float3 motionNormal, motionTangent, motionColor, motionGeomNormal;
            float2 motionUv;
            float motionTangentSign = 1.0f;
            float motionUvArea2 = 0.0f;
            fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId, interpolateMotion,
                                 motionTime, bary, motionNormal, motionTangent, motionUv, motionColor,
                                 motionTangentSign, motionGeomNormal, motionUvArea2);
            float3 nCur[3], pCur[3], tCur[3], cCur[3];
            float2 uvCur[3];
            float signCur = 1.0f; // unused by this debug view
            fetchTriangle(vertexBuffer, prevVertexBuffer, indexBuffer, entry, rec.primitiveId, false, motionTime, pCur,
                          nCur, tCur, uvCur, signCur, cCur);
            dbg = float3(
                motionTime, clamp(length(normalize(motionNormal) - normalize(nCur[0])) * 10.0f, 0.0f, 1.0f), 0.0f);
        }
        addFilteredRadiance(radianceOut, tid, dbg);
        return;
    }

    const bool entering = si.front_face;
    if (entering && !hasIorStack)
    {
        si.exterior_ior = 1.0f;
    }
    else if (entering)
    {
        const int iorTop = iorStacks[tid].top;
        si.exterior_ior = iorTop >= 0 ? iorStacks[tid].entries[iorTop].ior : 1.0f;
    }
    else if (hasIorStack)
    {
        // Exits need to look below the matching material. They are the uncommon
        // case that pays for copying the full stack.
        const IorStack iorStack = iorStacks[tid];
        si.exterior_ior = ior_stack_peek_after_pop_material(iorStack, entry.materialId);
    }
    else
    {
        // The camera may start inside a mesh, or an open mesh may expose a back
        // face before this path has entered any tracked dielectric. Do not read
        // the deliberately uninitialised cold side table in that case.
        si.exterior_ior = 1.0f;
    }

    const float3 surfaceEmission = float3(si.emission);

    constexpr float kGuideRoughnessFloor = 0.05f;
    const bool writingAov = shouldWriteAov(uniforms, sampleIdx);
    if (writingAov && depth == 0u)
    {
        DenoiserMaterialGuides materialGuides = {};
        if (isOpenPBR)
        {
            if (kShadeBase)
            {
                materialGuides = openpbrBaseDenoiserGuides(openpbrBaseMat, si);
            }
            else
            {
                const OpenPBR_ResolvedInputs openpbrInputs = openpbr_resolve_inputs(openpbrMat, si);
                materialGuides = openpbrDenoiserGuides(openpbrInputs, si, openpbrBaseMapDetailsSubsurface(openpbrMat));
            }
        }
        else
        {
            materialGuides = standardDenoiserGuides(si);
        }
        const bool guideOpaque = materialGuides.transmission <= kGuideRoughnessFloor;

        // Depth and motion always belong to the camera-visible surface, even
        // when its material attributes will be replaced.
        const float3 prevPrimary =
            uniforms.hasPrevFramePose ?
                previousWorldPosition(prevFrameVertexBuffer, indexBuffer, prevInstances, uniforms, entry,
                                      rec.instanceIndex, rec.geomEntryIndex, rec.primitiveId, bary) :
                worldPosition;
        const ScreenMotion primaryMotion = screenMotion(uniforms, uniforms.prevWorldToClip * float4(prevPrimary, 1.0f),
                                                        uint2(tid % uniforms.width, tid / uniforms.width));
        aov[tid].depth = viewDepth(uniforms, worldPosition);
        aov[tid].motionX = primaryMotion.offset.x;
        aov[tid].motionY = primaryMotion.offset.y;
        aov[tid].reactive = primaryMotion.reactive;
        aov[tid].guideStateOrBounceDepth = 0.0f;

        aov[tid].diffuseAlbedo = packed_float3(materialGuides.diffuse);
        aov[tid].specularAlbedo = packed_float3(materialGuides.specular);
        aov[tid].normal = packed_float3(normalize(float3(si.shading_normal)));
        aov[tid].roughness = materialGuides.roughness;
        aov[tid].specularHitDistance = 0.0f;

        const bool hasPrimaryDiffuse = luminance(materialGuides.diffuse) > 1e-3f;
        const bool replaceMaterial =
            !uniforms.guidePrimaryHit &&
            (!guideOpaque || (!hasPrimaryDiffuse && materialGuides.roughness <= kGuideRoughnessFloor));
        const bool needsSpecularDistance =
            guideOpaque && materialGuides.roughness < 0.5f && luminance(materialGuides.specular) > 0.02f;

        float interfaceIor = max(si.exterior_ior, 1.0f);
        float interfaceFresnel = 0.0f;
        float interfaceCosine = 0.0f;
        if (!guideOpaque)
        {
            interfaceIor = kShadeBase ? denoiserInterfaceIor(isOpenPBR, openpbrBaseMat, si, entering) :
                                        denoiserInterfaceIor(isOpenPBR, openpbrMat, si, entering);
            const float eta = entering ? si.exterior_ior / interfaceIor : interfaceIor / max(si.exterior_ior, 1e-4f);
            interfaceCosine = abs(dot(float3(si.shading_normal), float3(si.wo)));
            interfaceFresnel = fresnel_dielectric(interfaceCosine, eta);
            aov[tid].diffuseAlbedo = packed_float3(float3(0.0f));
            aov[tid].specularAlbedo = packed_float3(float3(interfaceFresnel));
            // Values >= 1 carry the glass Fresnel into the replacement blend.
            aov[tid].guideStateOrBounceDepth = interfaceFresnel < 0.5f ? 1.0f + interfaceFresnel : 0.0f;
        }

        if (SPEC_AOV && !any(surfaceEmission > 0.0f) && uniforms.writeAov && (replaceMaterial || needsSpecularDistance))
        {
            const float3 V = normalize(float3(si.wo));
            const float3 Ns = normalize(float3(si.shading_normal));
            const float3 Nf = dot(Ns, V) >= 0.0f ? Ns : -Ns;
            float3 guideDirection = reflect_dir(-V, Nf);
            bool transmitted = false;
            if (!guideOpaque)
            {
                const float eta = entering ? si.exterior_ior / interfaceIor : interfaceIor / max(si.exterior_ior, 1e-4f);
                float3 refracted;
                const bool validRefraction = si.thin_walled ? true : refract_dir(-V, Nf, eta, interfaceCosine, refracted);
                if (validRefraction && interfaceFresnel < 0.5f)
                {
                    guideDirection = si.thin_walled ? -V : refracted;
                    transmitted = true;
                }
            }

            const float3 faceNg =
                dot(float3(si.geometry_normal), V) > 0.0f ? float3(si.geometry_normal) : -float3(si.geometry_normal);
            GuideRay guide;
            guide.origin = packed_float3(offset_ray(si.position, transmitted ? -faceNg : faceNg));
            guide.flags = GUIDE_RAY_ACTIVE | (replaceMaterial ? GUIDE_RAY_REPLACE_MATERIAL : 0u);
            guide.direction = packed_float3(normalize(guideDirection));
            IorStack guideIorStack;
            if (hasIorStack)
            {
                guideIorStack = iorStacks[tid];
            }
            else
            {
                ior_stack_init(guideIorStack);
            }
            if (!si.thin_walled && !guideOpaque)
            {
                if (transmitted)
                {
                    if (entering)
                    {
                        ior_stack_push(guideIorStack, si.dielectric_priority, interfaceIor, entry.materialId);
                    }
                    else
                    {
                        ior_stack_pop_material(guideIorStack, entry.materialId);
                    }
                }
                else if (!entering && !ior_stack_has_material(guideIorStack, entry.materialId))
                {
                    // A camera may start inside a dielectric without ever
                    // having pushed it onto the path stack.
                    ior_stack_push(guideIorStack, si.dielectric_priority, interfaceIor, entry.materialId);
                }
            }
            const float guideCurrentIor = ior_stack_current_ior(guideIorStack);
            const float guideExteriorIor =
                guideIorStack.top > 0 ? guideIorStack.entries[guideIorStack.top - 1].ior : 1.0f;
            guide.mediaIors = packGuideIors(guideCurrentIor, guideExteriorIor);
            uniforms.guideRays[tid] = guide;
            queuePush(uniforms, (device atomic_uint*)&control[WF_CTRL_GUIDE], uniforms.guideQueue, tid,
                      control[WF_CTRL_CAPACITY]);
        }

        // The stochastic radiance continuation must not overwrite the primary
        // record. Any requested secondary information now belongs to GuideRay.
        p.depthAndFlags |= PATH_FLAG_AOV_DONE;
        paths[tid].depthAndFlags = p.depthAndFlags;

        if (any(surfaceEmission > 0.0f))
        {
            // Direct emission has no Monte Carlo variance. Preserve small HDR
            // emitters instead of filtering them toward their neighbourhood.
            aov[tid].guideStateOrBounceDepth = -1.0f;
        }
    }

    // --- Sparse Hash Radiance Cache ----------------------------------------
    const float3 diffuseAlbedo = float3(si.albedo) * (1.0f - si.metallic);
    const float3 specularF0 = gltf_f0(si.ior, si.specular, si.specular_color, si.albedo, si.metallic);
    const float3 materialDemodulation = sharcMaterialDemodulation(diffuseAlbedo, specularF0);
    const float receiverRoughness = kShadeBase ? sharcReceiverRoughness(isOpenPBR, openpbrBaseMat, si) :
                                                 sharcReceiverRoughness(isOpenPBR, openpbrMat, si);
    const bool cacheableReceiver = sharcReceiverCacheEligible(
        receiverRoughness,
        isOpenPBR ? (kShadeBase ? 0.0f : openpbrMat.transmission_weight) : max(si.transmission, si.diffuse_transmission),
        isFibre, uniforms.sharcRoughnessThreshold);

    if (SPEC_SHARC_UPDATE && uniforms.sharcCapacity != 0u)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcMultiplyPendingThroughput(updateState, throughput / max(throughputAtStageEntry, float3(1e-6f)));
        const bool responsive = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
        const bool continueTracing = sharcUpdateHit(
            updateState, uniforms, sharcHashEntries, sharcAccumulation, sharcResolved,
            uniforms.sharcDebug != 0u ? iorStats + IOR_STAT_COUNT : nullptr, si.position, si.geometry_normal, -rayDir,
            0.0f, materialDemodulation, float3(0.0f), surfaceEmission, cacheableReceiver,
            float(sharcHash(tid ^ uniforms.sharcFrameIndex)) * (1.0f / 4294967296.0f), responsive);
        sharcUpdates[updateIndex] = updateState;
        if (!continueTracing)
        {
            return;
        }
    }
    else if (SPEC_SHARC && uniforms.sharcCapacity != 0u)
    {
        if (SPEC_DEBUG && (DebugMode)uniforms.debug == DebugMode::eSharcRadiance && depth == 0u)
        {
            radianceOut[tid] = float4(sharcDebugRadiance(uniforms, sharcHashEntries, sharcResolved, si.position,
                                                         si.geometry_normal, -rayDir, materialDemodulation),
                                      0.0f);
            return;
        }
        if (SHARC_DEBUG_IS_SURFACE_VIEW(uniforms.sharcDebug) && depth == 0u)
        {
            radianceOut[tid] = float4(sharcDebugSurface(uniforms, sharcHashEntries, sharcResolved, si.position,
                                                        si.geometry_normal, -rayDir, materialDemodulation),
                                      0.0f);
            return;
        }
        if (depth >= uniforms.sharcDepth)
        {
            const SharcAddress address = sharcAddress(uniforms, si.position, si.geometry_normal, false);
            const float roughness = min(max(unpackSharcRoughness(p.depthAndFlags), 0.0f), 0.99f);
            const float alpha = roughness * roughness;
            const float alpha2 = alpha * alpha;
            const float footprint = rec.distance * sqrt(max(0.5f * alpha2 / max(1.0f - alpha2, 1e-5f), 0.0f));
            const bool validSegment = rec.distance > address.voxelSize * sqrt(3.0f);
            const bool validLobe = footprint > address.voxelSize;
            device atomic_uint* sharcStats = uniforms.sharcDebug != 0u ? iorStats + IOR_STAT_COUNT : nullptr;
            if (sharcStats && !validSegment)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_SEGMENT_REJECT], 1u, memory_order_relaxed);
            }
            if (sharcStats && validSegment && !validLobe)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_FOOTPRINT_REJECT], 1u, memory_order_relaxed);
            }
            if (sharcStats && validSegment && validLobe && !cacheableReceiver)
            {
                atomic_fetch_add_explicit(&sharcStats[SHARC_STAT_RECEIVER_REJECT], 1u, memory_order_relaxed);
            }
            float3 cached = float3(0.0f);
            uint32_t cachedSamples = 0u;
            if (validSegment && validLobe && cacheableReceiver &&
                sharcQuery(uniforms, sharcHashEntries, sharcResolved, sharcStats, si.position, si.geometry_normal,
                           -rayDir, materialDemodulation, cached, cachedSamples))
            {
                if ((uniforms.sharcFlags & SHARC_FLAG_SEPARATE_EMISSIVE) != 0u)
                {
                    cached += surfaceEmission;
                }
                radiance += throughput * cached;
                addFilteredRadiance(radianceOut, tid, radiance);
                paths[tid] = p;
                return;
            }
        }
    }

    if (any(surfaceEmission > 0.0f))
    {
        float emissionMis = 1.0f;
        if (SPEC_LIGHTS && SPEC_EMISSIVE_MESH_LIGHTS && depth > 0u && !specularBounce && neeDone && !isCurve &&
            uniforms.numEmissiveMeshes > 0u)
        {
            const uint32_t geometryId = rec.geomEntryIndex - instances[rec.instanceIndex].userID;
            const float3 misOrigin = rayOrigin - rayDir * p.misDistance;
            const float lightPdf =
                emissiveMeshHitPdf(uniforms, instances, vertexBuffer, prevVertexBuffer, indexBuffer, rec.instanceIndex,
                                   geometryId, rec.primitiveId, misOrigin, worldPosition, motionTime);
            if (lightPdf > 0.0f)
            {
                emissionMis = computeMisWeight(p.lastBsdfPdf, lightPdf, uniforms.misHeuristic);
            }
        }
        radiance += throughput * surfaceEmission * emissionMis;
    }

    if (SPEC_SHADE_PROBE == 1u)
    {
        // Keep the actual resolved surface prefix observable while compiling
        // prepare, NEE and sampling out of this diagnostic specialization.
        const float3 checksum = si.position + si.geometry_normal + si.shading_normal + si.tangent + si.bitangent +
                                si.wo + surfaceEmission + float3(si.exterior_ior);
        radianceOut[tid] = float4(checksum, 1.0f);
        return;
    }

    const bool smoothLobe =
        isOpenPBR ? (kShadeBase ? openpbr_has_smooth_lobe(openpbrBaseMat) : openpbr_has_smooth_lobe(openpbrMat)) :
                    bsdf_has_smooth_lobe(si);
    bool openpbrEntersSubsurface = false;
    float3 openpbrWalkAlbedo = float3(0.0f);
    OpenPBR_PreparedBsdf openpbrPrepared;
    OpenPBR_BasePreparedBsdf openpbrBasePrepared;
    if (isOpenPBR)
    {
        // Base and layer queues cannot enter a volume. For the tail, finish
        // the short-lived volume derivation before constructing the surface
        // lobes so the two large OpenPBR states never overlap.
        if (SPEC_SSS && !kShadeBase && !kShadeLayer)
        {
            openpbrEntersSubsurface = openpbrMat.subsurface_weight > 0.0f && openpbrMat.geometry_thin_walled == 0u;
            if (openpbrEntersSubsurface)
            {
                if (openpbrBaseMapDetailsSubsurface(openpbrMat))
                {
                    OpenPBRParams walkMat = openpbrMat;
                    const float3 base = openpbr_color_to_float3(walkMat.base_color);
                    const float3 subsurface = openpbr_color_to_float3(walkMat.subsurface_color) * base;
                    walkMat.subsurface_color = OpenPBRColor{ subsurface.x, subsurface.y, subsurface.z };
                    openpbrWalkAlbedo = openpbr_interior_volume(walkMat).albedo;
                }
                else
                {
                    openpbrWalkAlbedo = openpbr_interior_volume(openpbrMat).albedo;
                }
            }
        }

        if (kShadeBase)
        {
            openpbrBasePrepared = openpbr_prepare_base_at(openpbrBaseMat, si, throughput);
        }
        else if (!kShadeBase)
        {
            openpbrPrepared = openpbr_prepare_surface_at(openpbrMat, si, throughput);
        }
    }

    const bool hasEmitter =
        (SPEC_LIGHTS && (uniforms.numLights > 0 || (SPEC_EMISSIVE_MESH_LIGHTS && uniforms.numEmissiveMeshes > 0))) ||
        (SPEC_ENV_MAP && uniforms.hasEnvMap);
    bool didNee = neeRunsAtVertex(uniforms.estimatorMode == 0, hasEmitter, smoothLobe);
    if (!smoothLobe)
    {
        auditWork(uniforms, WORK_NEE_DELTA_HITS);
    }
    const ShadedFrame neeFrame =
        shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
    if (didNee && SPEC_SHADE_PROBE != 3u && SPEC_SHADE_PROBE != 4u)
    {
        auditWork(uniforms, WORK_NEE_ELIGIBLE_HITS);
        if (SPEC_RIS_ONE)
        {
            if (depth == 0u)
            {
                auditWork(uniforms, WORK_FIRST_BOUNCE_NEE_SAMPLES, 1u);
            }
            else
            {
                auditWork(uniforms, WORK_SECONDARY_NEE_SAMPLES, 1u);
            }

            LightConnection conn;
            if (SPEC_SHADE_PROBE == 6u)
            {
                // Production surface + prepare + eval, with the scene light
                // sampler removed. This distinguishes eval/prepared overlap
                // from connectToLight's own peak without a synthetic kernel.
                conn = makeEmptyConnection();
                conn.toLight = normalize(si.shading_normal + 0.25f * si.tangent);
                conn.radiance = float3(1.0f);
                conn.pdf = 0.5f;
                conn.tMax = 1.0f;
                conn.needsRay = true;
            }
            else if (kShadeBase && SPEC_SPLIT_BASE_NEE)
            {
                conn = unpackBaseLightConnection(uniforms.baseLightConnections[tid]);
            }
            else
            {
                conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials, vertexBuffer,
                                      prevVertexBuffer, indexBuffer, motionTime, rng, si, envAliasTable, envMapTexture,
                                      iesProfiles);
            }
            auditNeeSampledConnection(uniforms, conn);
            if (SPEC_SHADE_PROBE == 10u)
            {
                const PackedLightConnectionProbe packed = packLightConnectionProbe(conn, si.position);
                uint packedHash = hash_combine(packed.toLight.x, packed.toLight.y);
                packedHash = hash_combine(packedHash, packed.originOffset.x);
                packedHash = hash_combine(packedHash, packed.originOffset.y);
                packedHash = hash_combine(packedHash, packed.visibilityTargetOffset.x);
                packedHash = hash_combine(packedHash, packed.visibilityTargetOffset.y);
                const float packedChecksum =
                    packed.pdf + packed.tMax + float(packed.flags) + float(packedHash) * (1.0f / 4294967296.0f);
                radianceOut[tid] = float4(float3(packed.radiance), packedChecksum);
                return;
            }
            if (SPEC_SHADE_PROBE == 5u)
            {
                // Production surface + connectToLight, stopping before BSDF
                // eval. Consume every deferred-shadow field so the compiler
                // cannot turn this into a direction-only light micro-test.
                const float3 checksum = conn.radiance + conn.toLight + conn.origin + conn.visibilityTarget +
                                        float3(conn.pdf + conn.tMax + float(conn.needsRay) + float(conn.isDelta) +
                                               float(conn.hasVisibilityTarget));
                radianceOut[tid] = float4(checksum, 1.0f);
                return;
            }
            LightConnectionEvaluation candidate;
            if (SPEC_SHADE_PROBE == 8u)
            {
                candidate.integrand = conn.radiance * conn.pdf;
                candidate.target = conn.needsRay && conn.pdf > 0.0f ? max(luminance(candidate.integrand), 1.0e-6f) : 0.0f;
            }
            else
            {
                candidate = kShadeBase ? evaluateLightConnection(conn, si, isFibre, neeFrame, isOpenPBR,
                                                                 openpbrBasePrepared, uniforms.misHeuristic) :
                                         evaluateLightConnection(conn, si, isFibre, neeFrame, isOpenPBR,
                                                                 openpbrPrepared, uniforms.misHeuristic);
            }
            if (kShadeBase && SPEC_SHADE_PROBE == 11u)
            {
                // Exact eval/sample phase boundary. Keep both the completed
                // evaluation and the prepared state required by the future
                // sampler alive, while compiling the sampler itself out.
                const float retainedSampleState = isOpenPBR ? baseSampleStateChecksum(openpbrBasePrepared) : 0.0f;
                radianceOut[tid] = float4(candidate.integrand, candidate.target + retainedSampleState * 1.0e-6f);
                return;
            }
            if (SPEC_SHADE_PROBE == 6u || SPEC_SHADE_PROBE == 7u)
            {
                // 6: prepare/eval against a synthetic connection.
                // 7: production proposal + prepare/eval, before shadow append.
                radianceOut[tid] = float4(candidate.integrand, candidate.target);
                return;
            }
            if (candidate.target > 0.0f)
            {
                auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                // Keep the generic one-candidate RIS operation order to
                // minimize numerical differences from that path.
                const float proposalWeight = candidate.target / conn.pdf;
                const float W = proposalWeight / candidate.target;
                const float3 weight = throughput * candidate.integrand * W;
                if (SPEC_SHADE_PROBE == 9u)
                {
                    radiance += clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect);
                }
                else
                {
                    const float3 shadowOrigin = isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal,
                                                                          curveRadius, conn.toLight) :
                                                          conn.origin;
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, shadowOrigin);
                    if (any(weight != 0.0f) && visibility.valid)
                    {
                        ShadowRay sr;
                        sr.origin = packed_float3(shadowOrigin);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(
                            clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                        sr.sharcRadiance = packed_float3(float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.alphaThreshold = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType);
                        const uint32_t slot = allocateShadowSlot(shadowCounter);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
                    }
                    else if (!any(weight != 0.0f))
                    {
                        auditWork(uniforms, WORK_NEE_SHADOW_REJECT_WEIGHT);
                    }
                    else
                    {
                        auditWork(uniforms, WORK_NEE_SHADOW_REJECT_SEGMENT);
                    }
                }
            }
            else if (conn.needsRay && !(conn.pdf > 0.0f))
            {
                auditWork(uniforms, WORK_NEE_REJECT_PDF);
            }
            else if (conn.needsRay && !neeProposesDirection(isFibre, neeFrame.frontFace,
                                                            neeFrame.normalSign * dot(conn.toLight, si.shading_normal)))
            {
                auditWork(uniforms, WORK_NEE_REJECT_COSINE);
            }
            else if (conn.needsRay)
            {
                auditWork(uniforms, WORK_NEE_REJECT_TARGET);
            }
        }
        else
        {
            const bool restirInitial = SPEC_RESTIR && uniforms.restirDIEnabled != 0u && depth == 0u;
            const uint32_t candidates = restirInitial ? max(uniforms.initialCandidateCount, 1u) :
                                                        (SPEC_RESTIR ? 1u : max(uniforms.risCandidates, 1u));

            if (restirInitial)
            {
                auditWork(uniforms, WORK_RESTIR_ELIGIBLE_HITS);
                auditWork(uniforms, WORK_RESTIR_INITIAL_CANDIDATES, candidates);
            }
            else if (depth == 0u)
            {
                auditWork(uniforms, WORK_FIRST_BOUNCE_NEE_SAMPLES, candidates);
            }
            else
            {
                auditWork(uniforms, WORK_SECONDARY_NEE_SAMPLES, candidates);
            }

            LightConnection bestConn = makeEmptyConnection();
            float3 bestF = float3(0.0f);
            RestirReservoir reservoir = {};
            const bool singleCandidateNee = !restirInitial && candidates == 1u;
            float singleCandidateW = 0.0f;

            // Keep one candidate live at a time. The separate SPEC_RIS_ONE path
            // bypasses this loop entirely; this controls the 2/4/8-candidate
            // editor modes where unrolling otherwise multiplies the NEE state.
#pragma clang loop unroll(disable)
            for (uint32_t i = 0; i < candidates; ++i)
            {
                SamplerState crng = rng;
                if (i != 0u)
                {
                    crng.seed = hash_combine(rng.seed, i * 0x9E3779B9u);
                }

                const LightConnection conn = connectToLight(uniforms, uniforms.numLights, lights, instances, materials,
                                                            vertexBuffer, prevVertexBuffer, indexBuffer, motionTime,
                                                            crng, si, envAliasTable, envMapTexture, iesProfiles);
                auditNeeSampledConnection(uniforms, conn);
                const LightConnectionEvaluation candidate =
                    kShadeBase ? evaluateLightConnection(conn, si, isFibre, neeFrame, isOpenPBR, openpbrBasePrepared,
                                                         uniforms.misHeuristic) :
                                 evaluateLightConnection(
                                     conn, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
                if (!(candidate.target > 0.0f))
                {
                    if (conn.needsRay && !(conn.pdf > 0.0f))
                        auditWork(uniforms, WORK_NEE_REJECT_PDF);
                    else if (conn.needsRay &&
                             !neeProposesDirection(isFibre, neeFrame.frontFace,
                                                   neeFrame.normalSign * dot(conn.toLight, si.shading_normal)))
                        auditWork(uniforms, WORK_NEE_REJECT_COSINE);
                    else if (conn.needsRay)
                        auditWork(uniforms, WORK_NEE_REJECT_TARGET);
                    continue;
                }
                auditWork(uniforms, WORK_NEE_VALID_CANDIDATES);
                const float w = candidate.target / conn.pdf;
                if (singleCandidateNee)
                {
                    bestConn = conn;
                    bestF = candidate.integrand;
                    singleCandidateW = w / candidate.target;
                    continue;
                }
                // Reuse eLightId under a new scramble so adding RIS does not shift later Sobol dimensions.
                SamplerState arng = crng;
                arng.seed = restirRngStreamSeed(crng.seed, RESTIR_RNG_INITIAL_SALT);
                if (restirReservoirUpdate(reservoir.state, w, candidate.target, 1u,
                                          random<SampleDimension::eLightId>(arng, uniforms.samplerType)))
                {
                    bestConn = conn;
                    bestF = candidate.integrand;
                    reservoir.sample = conn.sample;
                }
            }

            reservoir.state.M = candidates;
            if (restirInitial && uniforms.restirInitialVisibility != 0u &&
                (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u)
            {
                auditWork(uniforms, WORK_RESTIR_INITIAL_VISIBILITY_QUERIES);
                const float3 initialShadowOrigin = isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal,
                                                                             curveRadius, bestConn.toLight) :
                                                             bestConn.origin;
                float initialTransmittance;
                if (!restirDiagnosticVisible(uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable,
                                             bestConn, initialShadowOrigin, instances, materials, geometryEntries,
                                             vertexBuffer, indexBuffer, initialTransmittance))
                {
                    restirReservoirDiscardSample(reservoir.state);
                }
                else
                {
                    restirReservoirStoreInitialVisibility(reservoir.state, initialTransmittance);
                    const uint32_t geometryKey =
                        restirVisibilityGeometryKey(rec.geomEntryIndex, rec.instanceIndex, rec.primitiveId);
                    restirVisibilityStore(reservoir.visibility,
                                          restirVisibilityReceiverKey(uniforms, si, entry.materialId, geometryKey),
                                          restirVisibilityLightKey(uniforms, bestConn), 0u);
                }
            }
            if (restirInitial)
            {
                const bool oddFrame = (uniforms.frameIndex & 1u) != 0u;
                device RestirCandidateAuditRecord* candidateAudit = restirCandidateAuditRecord(uniforms, tid);
                if (candidateAudit != nullptr)
                {
                    candidateAudit->pixelIndex = tid;
                    candidateAudit->frameIndex = uniforms.frameIndex;
                    candidateAudit->sampleIndex = sampleIdx;
                    candidateAudit->rngSeed = rng.seed;
                    candidateAudit->rngSampleIndex = rng.sampleIdx;
                    candidateAudit->rngDepth = rng.depth;
                    candidateAudit->lightDimension = uint32_t(SampleDimension::eLightId);
                    candidateAudit->initialStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_INITIAL_SALT);
                    candidateAudit->temporalStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_TEMPORAL_SALT);
                    candidateAudit->spatialStreamSeed = restirRngStreamSeed(rng.seed, RESTIR_RNG_SPATIAL_SALT);
                    candidateAudit->currentSample = reservoir.sample;
                    candidateAudit->currentBufferIndex = restirCurrentBufferIndex(uniforms.frameIndex);
                    candidateAudit->historyBufferIndex = restirHistoryBufferIndex(uniforms.frameIndex);
                    candidateAudit->proposalCollision = uniforms.restirProposalCollision;
                    candidateAudit->proposalEntropy = uniforms.restirProposalEntropy;
                }
                device RestirSurfaceHistory* currentHistory =
                    oddFrame ? uniforms.restirHistory1 : uniforms.restirHistory0;
                device char* currentSurfaceData =
                    uniforms.restirBiasCorrection != 0u ?
                        (oddFrame ? uniforms.restirSurfaceData1 : uniforms.restirSurfaceData0) :
                        uniforms.restirSurfaceData0;

                const float3 previousPosition =
                    uniforms.hasPrevFramePose != 0u && !isCurve ?
                        previousWorldPosition(prevFrameVertexBuffer, indexBuffer, prevInstances, uniforms, entry,
                                              rec.instanceIndex, rec.geomEntryIndex, rec.primitiveId, bary) :
                        worldPosition;
                const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
                const ScreenMotion motion =
                    screenMotion(uniforms, uniforms.prevWorldToClip * float4(previousPosition, 1.0f), pixel);
                RestirSurfaceHistory surface;
                surface.geometryNormal = packed_float3(si.geometry_normal);
                surface.depth = viewDepth(uniforms, worldPosition);
                surface.materialIdAndFlags = entry.materialId | RESTIR_SURFACE_VALID;

                const float2 previousPixelPosition =
                    float2(pixel) + float2(0.5f + uniforms.jitterX, 0.5f + uniforms.jitterY) + motion.offset;

                const uint32_t sampleAndMedium = (sampleIdx & RESTIR_TARGET_SAMPLE_MASK) |
                                                 ((mediumState.medium & MEDIUM_INDEX_MASK) << RESTIR_TARGET_MEDIUM_SHIFT);
                if (restirCanStoreDirectTarget(si, isFibre, isOpenPBR))
                {
                    RestirDirectTargetSurface targetSurface;
                    targetSurface.position = packed_float3(worldPosition);
                    targetSurface.shadingNormal = packed_float3(si.shading_normal);
                    targetSurface.rayDirection = packed_float3(rayDir);
                    targetSurface.albedo = packed_float3(si.albedo);
                    targetSurface.flags =
                        RESTIR_TARGET_DIRECT |
                        ((si.material_type & RESTIR_TARGET_MATERIAL_MASK) << RESTIR_TARGET_MATERIAL_SHIFT) |
                        (si.front_face ? RESTIR_TARGET_FRONT_FACE : 0u) |
                        (si.thin_walled ? RESTIR_TARGET_THIN_WALLED : 0u) |
                        (si.transmission > 0.0f ? RESTIR_TARGET_HAS_TRANSMISSION : 0u) |
                        (si.diffuse_transmission > 0.0f ? RESTIR_TARGET_HAS_DIFFUSE_TRANSMISSION : 0u);
                    targetSurface.flags |=
                        restirVisibilityGeometryKey(rec.geomEntryIndex, rec.instanceIndex, rec.primitiveId);
                    targetSurface.roughness = si.roughness;
                    targetSurface.metallicOrIor = si.material_type == MATERIAL_TYPE_STANDARD_PBR ? si.metallic : si.ior;
                    targetSurface.sampleIdxAndMedium = sampleAndMedium;
                    ((device RestirDirectTargetSurface*)currentSurfaceData)[tid] = targetSurface;
                }
                else
                {
                    RestirTargetSurface targetSurface;
                    targetSurface.position = packed_float3(worldPosition);
                    targetSurface.rayDirection = packed_float3(rayDir);
                    targetSurface.throughput = packed_float3(throughput);
                    targetSurface.barycentrics = bary;
                    targetSurface.lodBase = lodBase;
                    targetSurface.geomEntryIndex = rec.geomEntryIndex;
                    targetSurface.instanceIndex = rec.instanceIndex;
                    targetSurface.primitiveId = rec.primitiveId;
                    targetSurface.sampleIdxAndMedium = sampleAndMedium;
                    ((device RestirTargetSurface*)currentSurfaceData)[tid] = targetSurface;
                }

                // Temporal reuse consumes the same surface interaction and prepared
                // BSDF as initial sampling. Keeping it here avoids reconstructing and
                // evaluating the first hit again in a full hit-queue dispatch.
                if (uniforms.temporalReuseEnabled != 0u && uniforms.restirHistoryValid != 0u && sampleIdx == 0u &&
                    motion.reactive == 0.0f)
                {
                    const int2 previousPixel = int2(floor(previousPixelPosition));
                    if (all(previousPixel >= int2(0)) && previousPixel.x < int(uniforms.width) &&
                        previousPixel.y < int(uniforms.height))
                    {
                        const uint32_t previousIndex =
                            uint32_t(previousPixel.y) * uniforms.width + uint32_t(previousPixel.x);
                        device const RestirReservoir* previousReservoirs =
                            oddFrame ? uniforms.restirReservoir0 : uniforms.restirReservoir1;
                        device const RestirSurfaceHistory* previousHistory =
                            oddFrame ? uniforms.restirHistory0 : uniforms.restirHistory1;
                        device const char* previousSurfaceData =
                            oddFrame ? uniforms.restirSurfaceData0 : uniforms.restirSurfaceData1;
                        RestirReservoir previousReservoir = previousReservoirs[previousIndex];
                        const RestirSurfaceHistory oldSurface = previousHistory[previousIndex];
                        const uint32_t age = previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
                        const bool previousValid = (previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u;
                        if (previousReservoir.state.M != 0u && (!previousValid || age < uniforms.reservoirMaxAge) &&
                            restirSurfaceHistoryCompatible(surface, oldSurface))
                        {
                            restirReservoirLimitHistoryM(
                                previousReservoir.state, reservoir.state.M, max(uniforms.reservoirMaxAge, 1u));
                            LightConnectionEvaluation previousEvaluation = {};
                            RestirLightSample previousSampleAtCurrent = {};
                            const uint32_t previousLightMapping =
                                previousValid ? remapRestirSample(uniforms, lights, previousReservoir.sample, true,
                                                                  previousSampleAtCurrent) :
                                                RESTIR_LIGHT_UNMAPPED;
                            if (candidateAudit != nullptr)
                            {
                                candidateAudit->historySample = previousReservoir.sample;
                                candidateAudit->mappedHistorySample = previousSampleAtCurrent;
                                candidateAudit->mappedLightId = previousLightMapping;
                            }
                            const bool previousLightValid = previousLightMapping != RESTIR_LIGHT_UNMAPPED &&
                                                            previousLightMapping != RESTIR_LIGHT_TYPE_CHANGED;
                            if (previousValid && !previousLightValid)
                            {
                                const uint32_t previousSampleType = restirSampleType(previousReservoir.sample);
                                auditWork(uniforms, previousLightMapping == RESTIR_LIGHT_TYPE_CHANGED ?
                                                        WORK_RESTIR_TEMPORAL_REJECT_TYPE :
                                                    previousSampleType == RESTIR_SAMPLE_ENVIRONMENT ?
                                                        WORK_RESTIR_TEMPORAL_REJECT_ENVIRONMENT :
                                                    previousSampleType == RESTIR_SAMPLE_EMISSIVE_TRIANGLE ?
                                                        WORK_RESTIR_TEMPORAL_REJECT_MESH :
                                                    previousSampleType == RESTIR_SAMPLE_INVALID ?
                                                        WORK_RESTIR_TEMPORAL_REJECT_SURFACE :
                                                        WORK_RESTIR_TEMPORAL_REJECT_UNMAPPED);
                            }
                            bool previousDuplicateCurrent = false;
                            if (previousValid && previousLightValid)
                            {
                                if (previousSampleAtCurrent.typeAndLightId == reservoir.sample.typeAndLightId)
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_SAME_LIGHT);
                                previousDuplicateCurrent = restirSamplesEqual(previousSampleAtCurrent, reservoir.sample);
                                if (previousDuplicateCurrent)
                                {
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_DUPLICATE_CURRENT);
                                }
                                const LightConnection previousConnection = reconnectRestirSample(
                                    uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer,
                                    motionTime, si, envAliasTable, envMapTexture, iesProfiles, previousSampleAtCurrent);
                                previousEvaluation =
                                    kShadeBase ?
                                        evaluateLightConnection(previousConnection, si, isFibre, neeFrame, isOpenPBR,
                                                                openpbrBasePrepared, uniforms.misHeuristic) :
                                        evaluateLightConnection(previousConnection, si, isFibre, neeFrame, isOpenPBR,
                                                                openpbrPrepared, uniforms.misHeuristic);
                                if (previousEvaluation.target > 0.0f)
                                {
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_TARGET_POSITIVE);
                                    if (reservoir.state.target > 0.0f)
                                    {
                                        const float targetRatio = previousEvaluation.target / reservoir.state.target;
                                        auditWork(uniforms, WORK_RESTIR_TARGET_RATIO_HISTOGRAM_BASE +
                                                                restirTargetRatioHistogramBin(targetRatio));
                                    }
                                }
                            }
                            const bool temporalSourceCompatible = !previousValid || previousLightValid;
                            bool selectedHistory = false;
                            if (temporalSourceCompatible)
                            {
                                const uint32_t currentM = reservoir.state.M;
                                SamplerState temporalRng = rng;
                                temporalRng.seed = restirRngStreamSeed(temporalRng.seed, RESTIR_RNG_TEMPORAL_SALT);
                                const bool selectedSampleHasCurrentVisibility =
                                    (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u &&
                                    previousLightValid && restirSamplesEqual(reservoir.sample, previousSampleAtCurrent);
                                const uint32_t currentVisibilityFlags =
                                    reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                                const RestirVisibilityCache currentVisibility = reservoir.visibility;
                                selectedHistory = restirReservoirUpdate(
                                    reservoir.state,
                                    restirReservoirMergeWeight(previousReservoir.state, previousEvaluation.target),
                                    previousEvaluation.target, previousReservoir.state.M,
                                    random<SampleDimension::eLightId>(temporalRng, uniforms.samplerType));
                                if (previousLightValid)
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_MERGES);
                                if (selectedHistory)
                                {
                                    reservoir.sample = previousSampleAtCurrent;
                                    if (selectedSampleHasCurrentVisibility)
                                    {
                                        reservoir.state.ageAndFlags |= currentVisibilityFlags;
                                        reservoir.visibility = currentVisibility;
                                    }
                                    else if ((previousReservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) !=
                                             0u)
                                    {
                                        reservoir.state.ageAndFlags |= previousReservoir.state.ageAndFlags &
                                                                       RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                                        reservoir.visibility = previousReservoir.visibility;
                                        restirVisibilityStore(
                                            reservoir.visibility, reservoir.visibility.receiverKey,
                                            reservoir.visibility.lightKeyAndAge & RESTIR_VISIBILITY_LIGHT_KEY_MASK,
                                            min(restirVisibilityAge(previousReservoir.visibility) + 1u, 255u));
                                    }
                                    auditWork(uniforms, WORK_RESTIR_TEMPORAL_SELECTED);
                                    if (previousDuplicateCurrent)
                                        auditWork(uniforms, WORK_RESTIR_TEMPORAL_SELECTED_DUPLICATE_CURRENT);
                                }
                                if (uniforms.restirBiasCorrection != 0u)
                                {
                                    const float currentTarget = reservoir.state.target;
                                    float previousTarget = previousReservoir.state.target;
                                    LightConnection selectedAtPrevious = {};
                                    RestirLightSample selectedAtPreviousSample = previousReservoir.sample;
                                    const uint32_t selectedPreviousMapping =
                                        selectedHistory ? previousLightMapping :
                                                          remapRestirSample(uniforms, lights, reservoir.sample, false,
                                                                            selectedAtPreviousSample);
                                    if (selectedPreviousMapping != RESTIR_LIGHT_UNMAPPED &&
                                        selectedPreviousMapping != RESTIR_LIGHT_TYPE_CHANGED &&
                                        (!selectedHistory ||
                                         (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u)))
                                    {
                                        const RestirTargetSurface previousStored =
                                            ((device const RestirTargetSurface*)previousSurfaceData)[previousIndex];
                                        SurfaceInteraction previousSi;
                                        bool previousIsFibre, previousIsOpenPBR;
                                        ShadedFrame previousNeeFrame;
                                        OpenPBR_PreparedBsdf previousOpenpbrPrepared;
                                        float previousCurveRadius;
                                        float4x4 previousObjectToWorld = float4x4(1.0f);
                                        if (!restirTargetIsDirect(previousStored))
                                        {
                                            const GeometryEntry previousEntry =
                                                geometryEntries[previousStored.geomEntryIndex];
                                            previousObjectToWorld = geometryObjectToWorld(
                                                uniforms, prevInstances, previousStored.instanceIndex,
                                                previousStored.geomEntryIndex, previousEntry);
                                        }
                                        rebuildRestirTargetSurface(
                                            uniforms, previousObjectToWorld, materials, geometryEntries,
                                            prevFrameVertexBuffer, prevFrameVertexBuffer, indexBuffer, curvePoints,
                                            curveSegments, previousStored, false, 0.0f, previousSi, previousIsFibre,
                                            previousIsOpenPBR, previousNeeFrame, previousOpenpbrPrepared,
                                            previousCurveRadius);
                                        selectedAtPrevious = reconnectRestirSampleContext(
                                            uniforms, (device UniformLight*)uniforms.previousLights,
                                            uniforms.previousNumLights, uniforms.previousNumEmissiveMeshes,
                                            uniforms.previousMeshLightSelectionPdf,
                                            uniforms.hasEnvMap != 0u && uniforms.restirEnvironmentHistoryValid != 0u,
                                            uniforms.previousEnvSelectionPdf, prevInstances, materials,
                                            prevFrameVertexBuffer, prevFrameVertexBuffer, indexBuffer, 0.0f, previousSi,
                                            envAliasTable, envMapTexture, iesProfiles, selectedAtPreviousSample);
                                        previousTarget = restirTargetOnly(
                                            selectedAtPrevious, previousSi, previousIsFibre, previousNeeFrame,
                                            previousIsOpenPBR, previousOpenpbrPrepared, uniforms.misHeuristic);
                                    }
                                    else if (selectedPreviousMapping == RESTIR_LIGHT_UNMAPPED ||
                                             selectedPreviousMapping == RESTIR_LIGHT_TYPE_CHANGED)
                                    {
                                        previousTarget = 0.0f;
                                    }
                                    if (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u &&
                                        previousTarget > 0.0f)
                                    {
                                        auditWork(uniforms, WORK_RESTIR_DIAGNOSTIC_QUERIES);
                                        float ignoredTransmittance;
                                        if (!restirDiagnosticVisible(
                                                uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable,
                                                selectedAtPrevious, selectedAtPrevious.origin, instances, materials,
                                                geometryEntries, vertexBuffer, indexBuffer, ignoredTransmittance))
                                        {
                                            previousTarget = 0.0f;
                                        }
                                    }
                                    const float sourceTargetSum = float(currentM) * currentTarget +
                                                                  float(previousReservoir.state.M) * previousTarget;
                                    restirReservoirApplyBasicNormalization(
                                        reservoir.state, selectedHistory ? previousTarget : currentTarget,
                                        sourceTargetSum);
                                }
                            }
                            reservoir.state.ageAndFlags =
                                (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_FLAGS_MASK) |
                                (selectedHistory ? min(age + 1u, RESTIR_RESERVOIR_AGE_MASK) : 0u);
                        }
                        else
                        {
                            auditWork(uniforms, WORK_RESTIR_TEMPORAL_REJECT_SURFACE);
                        }
                    }
                    else
                    {
                        auditWork(uniforms, WORK_RESTIR_TEMPORAL_REJECT_SURFACE);
                    }
                }
                *((device RestirReservoir*)(hits + size_t(tid) * sizeof(RestirReservoir))) = reservoir;
                currentHistory[tid] = surface;
                const uint32_t restirSlot =
                    atomic_fetch_add_explicit((device atomic_uint*)&control[WF_CTRL_RESTIR], 1u, memory_order_relaxed);
                if (restirSlot < control[WF_CTRL_CAPACITY])
                {
                    uniforms.restirQueue[restirSlot] = tid;
                    if ((restirSlot & 63u) == 0u)
                    {
                        atomic_fetch_add_explicit(
                            (device atomic_uint*)&control[WF_CTRL_RESTIR_DIS], 1u, memory_order_relaxed);
                    }
                }
            }
            if (!restirInitial)
            {
                const float W = singleCandidateNee ? singleCandidateW : restirReservoirNormalization(reservoir.state);
                if (W > 0.0f)
                {
                    const float3 weight = throughput * bestF * W;
                    const float3 shadowOrigin = isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal,
                                                                          curveRadius, bestConn.toLight) :
                                                          bestConn.origin;
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(bestConn, shadowOrigin);
                    if (any(weight != 0.0f) && visibility.valid)
                    {
                        ShadowRay sr;
                        sr.origin = packed_float3(shadowOrigin);
                        sr.direction = packed_float3(visibility.direction);
                        sr.weight = packed_float3(
                            clampPathContribution(weight, depth, uniforms.clampDirect, uniforms.clampIndirect));
                        sr.maxDistance = visibility.maxDistance;
                        sr.pixelIndex = tid;
                        sr.medium = mediumState.medium & MEDIUM_INDEX_MASK;
                        sr.sharcRadiance = packed_float3(SPEC_SHARC_UPDATE ? bestF * W : float3(0.0f));
                        sr.sharcPathIndex = tid;
                        sr.alphaThreshold = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType);
                        const uint32_t slot = allocateShadowSlot(shadowCounter);
                        auditWork(uniforms, WORK_NEE_SHADOW_APPENDS);
                        storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
                    }
                    else if (!any(weight != 0.0f))
                    {
                        auditWork(uniforms, WORK_NEE_SHADOW_REJECT_WEIGHT);
                    }
                    else
                    {
                        auditWork(uniforms, WORK_NEE_SHADOW_REJECT_SEGMENT);
                    }
                }
                else
                {
                    auditWork(uniforms, WORK_NEE_SHADOW_REJECT_NORMALIZATION);
                }
            }
        }
    }

    if (SPEC_SHADE_PROBE == 2u || SPEC_SHADE_PROBE == 8u)
    {
        // NEE has already emitted its shadow record. Probe 2 retains the full
        // path, while probe 8 omits BSDF prepare/eval; both remove continuation.
        addFilteredRadiance(radianceOut, tid, radiance);
        return;
    }

    if (!SPEC_SHARC_UPDATE && depth + 1u >= uniforms.maxDepth)
    {
        addFilteredRadiance(radianceOut, tid, radiance);
        return;
    }

    if (kShadeBase)
    {
        addFilteredRadiance(radianceOut, tid, radiance);
        radiance = float3(0.0f);
    }

    half3 baseContinuationShadingNormal = half3(0.0h);
    half3 baseContinuationGeometryNormal = half3(0.0h);
    half baseContinuationRoughness = half(0.0h);
    if (kShadeBase)
    {
        baseContinuationShadingNormal = half3(si.shading_normal);
        baseContinuationGeometryNormal = half3(si.geometry_normal);
        baseContinuationRoughness = half(si.roughness);
    }

    const RandomSample4 bsdfRandom =
        random4<SampleDimension::eBSDF0, SampleDimension::eBSDF1, SampleDimension::eBSDF2, SampleDimension::eBSDF3>(
            rng, uniforms.samplerType);
    const float4 xi = bsdfRandom.value;
    const uint32_t lobeWord = bsdfRandom.bits.z >> 9u;
    const uint32_t fresnelWord = bsdfRandom.bits.w >> 9u;
    BsdfSampleResult sampleResult = isOpenPBR ? (kShadeBase ? openpbr_bsdf_sample(openpbrBasePrepared, si.wo, xi) :
                                                              openpbr_bsdf_sample(openpbrPrepared, xi)) :
                                                bsdf_sample(si, xi, lobeWord, fresnelWord);

    if (SPEC_SHADE_PROBE == 4u)
    {
        // Keep preparation and the complete BSDF sample observable while
        // deleting continuation, roulette and path write-back. This separates
        // the sampler's peak from state that only the next segment consumes.
        const float eventChecksum = float(sampleResult.event_type & 255u) * 1.0e-6f;
        radianceOut[tid] =
            float4(float3(sampleResult.wi) + float3(sampleResult.bsdf_over_pdf) + eventChecksum, sampleResult.pdf);
        return;
    }

    if (sampleResult.event_type == BSDF_EVENT_ABSORB)
    {
        // Whatever next-event estimation queued above stays: it is this vertex's
        // direct lighting and does not depend on where the path went next.
        addFilteredRadiance(radianceOut, tid, radiance);
        return;
    }

    const bool nextSpecular = ((sampleResult.event_type & BSDF_EVENT_SPECULAR) != 0);
    if (SPEC_SHARC_UPDATE && cacheableReceiver)
    {
        const float directionWeight =
            (sampleResult.event_type & BSDF_EVENT_DIFFUSE) != 0 ? 0.0f : 1.0f - saturate(si.roughness);
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcSetRadianceDirectionWeight(updateState, directionWeight);
        sharcUpdates[updateIndex] = updateState;
    }

    if (kShadeBase)
    {
        const PathRay continuationRay = rays[tid];
        const HitRecord continuationHit = *wavefrontHitRecord(hits, tid);
        const float3 continuationRayOrigin = float3(continuationRay.origin);
        const float3 continuationRayDirection = float3(continuationRay.direction);
        const float3 continuationPosition = continuationRayOrigin + continuationRayDirection * continuationHit.distance;
        const float3 continuationShadingNormal = float3(baseContinuationShadingNormal);
        const float3 continuationGeometryNormal = float3(baseContinuationGeometryNormal);
        const float3 continuationWo = -continuationRayDirection;
        const float3 continuationFaceNg = dot(continuationGeometryNormal, continuationWo) > 0.0f ?
                                              continuationGeometryNormal :
                                              -continuationGeometryNormal;
        const float3 continuationDirection = normalize(float3(sampleResult.wi));
        float3 segmentThroughput = float3(sampleResult.bsdf_over_pdf);
        float3 nextThroughput = SPEC_SHARC_UPDATE ? segmentThroughput : throughput * segmentThroughput;

        const ShadedFrame continuationFrame = shadedFrame(dot(continuationGeometryNormal, continuationWo) > 0.0f,
                                                          dot(continuationShadingNormal, continuationWo), 0.0f, 0.0f);
        const bool continuationDidNee =
            neePairsWithBounce(didNee, false, continuationFrame.frontFace,
                               continuationFrame.normalSign * dot(continuationShadingNormal, continuationDirection));

        bool alive = dot(nextThroughput, nextThroughput) >= 1e-4f;
        if (alive && depth > 3u)
        {
            const float q = min(max(nextThroughput.x, max(nextThroughput.y, nextThroughput.z)), 1.0f);
            SamplerState rrng = samplerFor(uniforms, tid, sampleIdx, depth);
            if (random<SampleDimension::eRussianRoulette>(rrng, uniforms.samplerType) > q)
            {
                alive = false;
            }
            else
            {
                const float invQ = 1.0f / max(q, 1e-5f);
                nextThroughput *= invQ;
                segmentThroughput *= invQ;
            }
        }
        if (depth + 1u >= uniforms.maxDepth || !alive)
        {
            return;
        }

        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
            SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcSetThroughput(updateState, segmentThroughput);
            sharcUpdates[updateIndex] = updateState;
        }

        PathRay nextRay;
        nextRay.origin = packed_float3(offset_ray(continuationPosition, continuationFaceNg));
        nextRay.direction = packed_float3(continuationDirection);
        rays[tid] = nextRay;

        PathState nextPath = paths[tid];
        nextPath.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : nextThroughput);
        nextPath.lastBsdfPdf = nextSpecular ? 1.0f : sampleResult.pdf;
        nextPath.misDistance = 0.0f;
        const float previousSharcRoughness = unpackSharcRoughness(nextPath.depthAndFlags);
        const float sharcRoughness =
            min(previousSharcRoughness +
                    (((sampleResult.event_type & BSDF_EVENT_DIFFUSE) != 0) ? 1.0f : float(baseContinuationRoughness)),
                1.0f);
        nextPath.depthAndFlags = (depth + 1u) | PATH_FLAG_ALIVE | (nextSpecular ? PATH_FLAG_SPECULAR : 0u) |
                                 (continuationDidNee ? PATH_FLAG_NEE_DONE : 0u) |
                                 (nextPath.depthAndFlags & (PATH_FLAG_AOV_DONE | PATH_FLAG_IOR_STACK_ACTIVE)) |
                                 packSharcRoughness(sharcRoughness);
        paths[tid] = nextPath;

        auditWork(uniforms, WORK_PATH_CONTINUATIONS);
        queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
        return;
    }

    // --- Next segment -------------------------------------------------------
    const float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
    float3 nextOrigin;
    const bool openpbrOpaqueBucket = isOpenPBR && (kShadeBase || kShadeLayer);
    float3 sssEntryTint = float3(1.0f);
    // Use the refracted entry direction instead of the lobe's cosine draw.
    bool sssRefractedEntry = false;
    if (openpbrOpaqueBucket)
    {
        nextOrigin = offset_ray(si.position, faceNg);
    }
    else if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0 && !isFibre)
    {
        // A diffuse-transmission event crosses an infinitesimally thin sheet.
        // A subsurface event enters the separately tracked random walk. Neither
        // makes the following surface segment part of a dielectric volume.
        const bool entersMedium = isOpenPBR ? openpbrEntersSubsurface : (si.subsurface > 0.0f);
        const uint32_t entryEvent = isOpenPBR ? (sampleResult.event_type & BSDF_EVENT_TRANSMISSION) :
                                                (sampleResult.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION);
        const bool startsSubsurfaceWalk = SPEC_SSS && entersMedium && entryEvent != 0u;
        const bool diffuseTransmission = (sampleResult.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION) != 0u;

        if (!si.thin_walled && !startsSubsurfaceWalk && !diffuseTransmission && !fibreMaterial)
        {
            IorStack iorStack;
            if (hasIorStack)
            {
                iorStack = iorStacks[tid];
            }
            else
            {
                ior_stack_init(iorStack);
            }
            if (entering)
            {
                // Count stack failures because either leaves the path carrying the wrong medium.
                if (ior_stack_full(iorStack))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_OVERFLOW], 1u, memory_order_relaxed);
                }
                ior_stack_push(iorStack, si.dielectric_priority, si.ior, entry.materialId);
            }
            else
            {
                if (!ior_stack_has_material(iorStack, entry.materialId))
                {
                    atomic_fetch_add_explicit(&iorStats[IOR_STAT_UNMATCHED], 1u, memory_order_relaxed);
                }
                ior_stack_pop_material(iorStack, entry.materialId);
            }
            iorStacks[tid] = iorStack;
            if (iorStack.top >= 0)
            {
                p.depthAndFlags |= PATH_FLAG_IOR_STACK_ACTIVE;
            }
            else
            {
                p.depthAndFlags &= ~PATH_FLAG_IOR_STACK_ACTIVE;
            }
        }
        nextOrigin = offset_ray(si.position, -faceNg);

        // OpenPBR enters on textured subsurface weight plus any transmission; glTF uses diffuse transmission.
        // Pure-absorption interiors stay Beer-Lambert instead of entering the random walk.
        if (startsSubsurfaceWalk)
        {
            mediumState.medium = (entry.materialId + 1u) & MEDIUM_INDEX_MASK;

            float3 walkAlbedo;
            if (isOpenPBR)
            {
                walkAlbedo = openpbrWalkAlbedo;
            }
            else
            {
                walkAlbedo = float3(materials[entry.materialId].diffuse_transmission_color);
                const float3 reference = float3(materials[entry.materialId].subsurface_reference);
                if (reference.x > 1e-4f && reference.y > 1e-4f && reference.z > 1e-4f)
                {
                    walkAlbedo *= si.albedo / reference;
                }
            }
            mediumState.mediumAlbedo = packMediumAlbedo(saturate(walkAlbedo));
            mediumPaths[tid] = mediumState;
            if (!isOpenPBR)
            {
                sssEntryTint = max(float3(si.diffuse_transmission_color), float3(1e-4f));
            }

            // The lobe chooses and weights entry, but the random walk starts along the refracted direction.
            sssRefractedEntry = true;
        }
    }
    else
    {
        nextOrigin = offset_ray(si.position, faceNg);
    }
    float3 nextDir = normalize(sampleResult.wi);
    if (sssRefractedEntry)
    {
        const float2 entryRandom =
            random2<SampleDimension::eSssChannel, SampleDimension::eSssDistance>(rng, uniforms.samplerType).value;
        nextDir = subsurface_entry_direction(float3(si.wo),
                                             (dot(float3(si.shading_normal), float3(si.wo)) > 0.0f) ?
                                                 float3(si.shading_normal) :
                                                 -float3(si.shading_normal),
                                             entryRandom.x, entryRandom.y);
    }

    if (sssRefractedEntry && dot(faceNg, nextDir) >= 0.0f)
    {
        addFilteredRadiance(radianceOut, tid, radiance);
        return;
    }

    if (isFibre)
    {
        // Both branches above assume a surface with an inside and an outside. A
        // strand has neither: the bounce leaves from wherever the crossing the lobe
        // already accounted for comes out.
        nextOrigin = fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, nextDir);
    }
    float3 segmentThroughput = float3(sampleResult.bsdf_over_pdf) / sssEntryTint;
    float3 nextThroughput = SPEC_SHARC_UPDATE ? segmentThroughput : throughput * segmentThroughput;

    // Record exactly the support NEE offered in the same shaded frame. A raw
    // back face may be an opaque two-sided surface whose BSDF flipped its frame,
    // or a transmissive exit that did not; shadedFrame distinguishes them.
    const bool neeCrosses =
        openpbrOpaqueBucket ? false : neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission);
    didNee = neePairsWithBounce(
        didNee, neeCrosses, neeFrame.frontFace, neeFrame.normalSign * dot(si.shading_normal, nextDir));

    addFilteredRadiance(radianceOut, tid, radiance);

    bool alive = dot(nextThroughput, nextThroughput) >= 1e-4f;
    if (alive && depth > 3u)
    {
        const float q = min(max(nextThroughput.x, max(nextThroughput.y, nextThroughput.z)), 1.0f);
        float rr;
        if (kShadeBase)
        {
            SamplerState rrng = samplerFor(uniforms, tid, sampleIdx, depth);
            rr = random<SampleDimension::eRussianRoulette>(rrng, uniforms.samplerType);
        }
        else
        {
            rr = random<SampleDimension::eRussianRoulette>(rng, uniforms.samplerType);
        }
        if (rr > q)
        {
            alive = false;
        }
        else
        {
            nextThroughput *= 1.0f / max(q, 1e-5f);
            segmentThroughput *= 1.0f / max(q, 1e-5f);
        }
    }
    if (depth + 1u >= uniforms.maxDepth)
    {
        alive = false;
    }

    if (!alive)
    {
        return;
    }

    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, tid);
        SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcSetThroughput(updateState, segmentThroughput);
        sharcUpdates[updateIndex] = updateState;
    }

    PathRay nextRay;
    nextRay.origin = packed_float3(nextOrigin);
    nextRay.direction = packed_float3(nextDir);
    rays[tid] = nextRay;

    p.throughput = packed_float3(SPEC_SHARC_UPDATE ? float3(1.0f) : nextThroughput);
    p.lastBsdfPdf = nextSpecular ? 1.0f : sampleResult.pdf;
    p.misDistance = 0.0f;
    const float previousSharcRoughness = unpackSharcRoughness(p.depthAndFlags);
    const float sharcRoughness = min(
        previousSharcRoughness + (((sampleResult.event_type & BSDF_EVENT_DIFFUSE) != 0) ? 1.0f : si.roughness), 1.0f);
    p.depthAndFlags =
        (depth + 1u) | PATH_FLAG_ALIVE | (nextSpecular ? PATH_FLAG_SPECULAR : 0u) | (didNee ? PATH_FLAG_NEE_DONE : 0u) |
        (p.depthAndFlags & (PATH_FLAG_AOV_DONE | PATH_FLAG_IOR_STACK_ACTIVE)) | packSharcRoughness(sharcRoughness);
    paths[tid] = p;

    auditWork(uniforms, WORK_PATH_CONTINUATIONS);
    queuePush(uniforms, outCounter, queueOut, tid, control[WF_CTRL_CAPACITY]);
}

#define STRELKA_WAVEFRONT_SHADE_ENTRY(Name, ShadeBucket)                                                                \
    kernel void Name(                                                                                                   \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                                \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                           \
        device const IesGpuBufferHeader* iesProfiles [[buffer(2)]], device UniformLight* lights [[buffer(3)]],          \
        device Material* materials [[buffer(4)]], device PathState* paths [[buffer(5)]],                                \
        device PathRay* rays [[buffer(21)]], device char* hits [[buffer(6)]], device float4* radianceOut [[buffer(7)]], \
        device IorStack* iorStacks [[buffer(8)]], device const GeometryEntry* geometryEntries [[buffer(9)]],            \
        device const uint2* envAliasTable [[buffer(10)]], device const char* vertexBuffer [[buffer(11)]],               \
        device const char* prevVertexBuffer [[buffer(12)]], device const uint32_t* indexBuffer [[buffer(13)]],          \
        constant uint32_t& sampleIdx [[buffer(14)]], device const uint32_t* queue [[buffer(15)]],                       \
        device uint32_t* queueOut [[buffer(16)]], device atomic_uint* outCounter [[buffer(17)]],                        \
        device uint32_t* control [[buffer(18)]], device char* shadowRays [[buffer(19)]],                                \
        device atomic_uint* shadowCounter [[buffer(20)]], device AovSample* aov [[buffer(22)]],                         \
        device char* sharcPassBuffer0 [[buffer(23)]], device const char* sharcPassBuffer1 [[buffer(24)]],               \
        CurveStaticTraversal::structure diagnosticAccelerationStructure [[buffer(25)]],                                 \
        CurveStaticTraversal::table diagnosticFunctionTable [[buffer(26)]],                                             \
        device const uint32_t* curveSegments [[buffer(27)]], device atomic_uint* iorStats [[buffer(28)]],               \
        device char* sharcPassState [[buffer(29)]], device MediumPathState* mediumPaths [[buffer(30)]],                 \
        texture2d<float> envMapTexture [[texture(0)]])                                                                  \
    {                                                                                                                   \
        wavefrontShadeImpl<ShadeBucket>(gid, uniforms, instances, iesProfiles, lights, materials, paths, rays, hits,    \
                                        radianceOut, iorStacks, geometryEntries, envAliasTable, vertexBuffer,           \
                                        prevVertexBuffer, indexBuffer, sampleIdx, queue, queueOut, outCounter,          \
                                        control, shadowRays, shadowCounter, aov, sharcPassBuffer0, sharcPassBuffer1,    \
                                        diagnosticAccelerationStructure, diagnosticFunctionTable, curveSegments,        \
                                        iorStats, sharcPassState, mediumPaths, envMapTexture);                          \
    }

STRELKA_WAVEFRONT_SHADE_ENTRY(wavefrontShade, WF_SHADE_GENERIC)
STRELKA_WAVEFRONT_SHADE_ENTRY(wavefrontShadeBase, WF_SHADE_BASE)
STRELKA_WAVEFRONT_SHADE_ENTRY(wavefrontShadeLayer, WF_SHADE_LAYER)
STRELKA_WAVEFRONT_SHADE_ENTRY(wavefrontShadeTranslucent, WF_SHADE_TRANSLUCENT)
STRELKA_WAVEFRONT_SHADE_ENTRY(wavefrontShadeTail, WF_SHADE_TAIL)

#undef STRELKA_WAVEFRONT_SHADE_ENTRY

static void rebuildRestirTargetSurface(constant Uniforms& uniforms,
                                       float4x4 objectToWorld,
                                       device const Material* materials,
                                       device const GeometryEntry* geometryEntries,
                                       device const char* vertexBuffer,
                                       device const char* prevVertexBuffer,
                                       device const uint32_t* indexBuffer,
                                       device const packed_float3* curvePoints,
                                       device const uint32_t* curveSegments,
                                       thread const RestirTargetSurface& stored,
                                       bool interpolateMotion,
                                       float motionTime,
                                       thread SurfaceInteraction& si,
                                       thread bool& isFibre,
                                       thread bool& isOpenPBR,
                                       thread ShadedFrame& neeFrame,
                                       thread OpenPBR_PreparedBsdf& openpbrPrepared,
                                       thread float& curveRadius)
{
    if (restirTargetIsDirect(stored))
    {
        const RestirDirectTargetSurface direct = *(thread const RestirDirectTargetSurface*)&stored;
        si = {};
        si.position = float3(direct.position);
        si.shading_normal = float3(direct.shadingNormal);
        si.geometry_normal = si.shading_normal;
        build_onb(si.shading_normal, si.tangent, si.bitangent);
        si.wo = -float3(direct.rayDirection);
        si.albedo = float3(direct.albedo);
        si.roughness = direct.roughness;
        si.material_type = (direct.flags >> RESTIR_TARGET_MATERIAL_SHIFT) & RESTIR_TARGET_MATERIAL_MASK;
        si.front_face = (direct.flags & RESTIR_TARGET_FRONT_FACE) != 0u;
        si.thin_walled = (direct.flags & RESTIR_TARGET_THIN_WALLED) != 0u;
        si.transmission = (direct.flags & RESTIR_TARGET_HAS_TRANSMISSION) != 0u ? 1.0f : 0.0f;
        si.diffuse_transmission = (direct.flags & RESTIR_TARGET_HAS_DIFFUSE_TRANSMISSION) != 0u ? 1.0f : 0.0f;
        si.exterior_ior = 1.0f;
        si.bump_normal = si.shading_normal;
        si.clearcoat_ior = 1.5f;
        si.iridescence_ior = 1.3f;
        si.specular_color = float3(1.0f);
        if (si.material_type == MATERIAL_TYPE_STANDARD_PBR)
        {
            si.metallic = direct.metallicOrIor;
            si.ior = 1.5f;
            si.specular = 0.5f;
        }
        else
        {
            si.ior = direct.metallicOrIor;
        }
        isFibre = false;
        isOpenPBR = false;
        curveRadius = 0.0f;
        neeFrame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
        return;
    }
    const GeometryEntry entry = geometryEntries[stored.geomEntryIndex];
    const float3 position = float3(stored.position);
    const float2 bary = stored.barycentrics;
    const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;
    float3 normal, tangent, geometryNormal, vertexColor;
    float2 uv;
    float tangentSign = 1.0f;
    curveRadius = 0.0f;
    if (isCurve)
    {
        fetchCurve(curvePoints, curveSegments, entry, stored.primitiveId, bary.x, position, objectToWorld, normal,
                   tangent, uv, curveRadius);
        geometryNormal = normal;
        vertexColor = float3(1.0f);
    }
    else
    {
        float3 objectNormal, objectTangent;
        float uvArea2;
        fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, stored.primitiveId, interpolateMotion,
                             motionTime, bary, objectNormal, objectTangent, uv, vertexColor, tangentSign,
                             geometryNormal, uvArea2);
        const float3 axisX = objectToWorld[0].xyz;
        const float3 axisY = objectToWorld[1].xyz;
        const float3 axisZ = objectToWorld[2].xyz;
        const FastNormalTransform normalTransform = makeFastNormalTransform(axisX, axisY, axisZ);
        normal = transformNormalFast(objectNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                     normalTransform.cofactorZ, normalTransform.orientation);
        geometryNormal = transformNormalFast(geometryNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                             normalTransform.cofactorZ, normalTransform.orientation);
        tangent = orthonormalizeTangent(normal, transformDirection(objectTangent, axisX, axisY, axisZ));
    }
    initSurfaceInteraction(si, materials[entry.materialId], position, normal, geometryNormal, tangent,
                           cross(normal, tangent) * tangentSign, uv, float3(stored.rayDirection), vertexColor,
                           stored.lodBase);
    si.exterior_ior = 1.0f;
    isOpenPBR = SPEC_ALL_OPENPBR || (SPEC_OPENPBR && si.material_type == MATERIAL_TYPE_OPENPBR);
    if (isOpenPBR)
    {
        OpenPBRParams openpbrMat = uniforms.openpbrParams[entry.materialId];
        if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
        {
            applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv, stored.lodBase);
        }
        openpbrPrepared = openpbr_prepare_surface_at(openpbrMat, si, float3(stored.throughput));
    }
    isFibre = isCurve && scattersThroughFibre(si);
    neeFrame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
}

constant int2 kRestirNeighborOffsets[] = { int2(-1, 0),  int2(0, -1), int2(1, 0), int2(0, 1),
                                           int2(-1, -1), int2(1, -1), int2(1, 1), int2(-1, 1),
                                           int2(-2, 0),  int2(0, -2), int2(2, 0), int2(0, 2),
                                           int2(-2, -1), int2(1, -2), int2(2, 1), int2(-1, 2) };

kernel void wavefrontRestirSpatialFinal(uint gid [[thread_position_in_grid]],
                                        constant Uniforms& uniforms [[buffer(0)]],
                                        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances
                                        [[buffer(1)]],
                                        device const IesGpuBufferHeader* iesProfiles [[buffer(2)]],
                                        device UniformLight* lights [[buffer(3)]],
                                        device const Material* materials [[buffer(4)]],
                                        device const uint2* envAliasTable [[buffer(5)]],
                                        device const char* vertexBuffer [[buffer(6)]],
                                        device const char* prevVertexBuffer [[buffer(7)]],
                                        device const uint32_t* indexBuffer [[buffer(8)]],
                                        device const uint32_t* hitQueue [[buffer(9)]],
                                        device char* shadowRays [[buffer(10)]],
                                        device atomic_uint* shadowCounter [[buffer(11)]],
                                        device float4* radianceOut [[buffer(12)]],
                                        device const RestirReservoir* initialReservoirs [[buffer(13)]],
                                        device const uint32_t* control [[buffer(14)]],
                                        device const GeometryEntry* geometryEntries [[buffer(15)]],
                                        device const packed_float3* curvePoints [[buffer(16)]],
                                        device const uint32_t* curveSegments [[buffer(17)]],
                                        CurveStaticTraversal::structure diagnosticAccelerationStructure [[buffer(18)]],
                                        CurveStaticTraversal::table diagnosticFunctionTable [[buffer(19)]],
                                        texture2d<float> envMapTexture [[texture(0)]])
{
    if (gid >= control[WF_CTRL_RESTIR_N])
    {
        return;
    }
    const uint32_t tid = hitQueue[gid];
    const bool oddFrame = (uniforms.frameIndex & 1u) != 0u;
    device const char* surfaceData = uniforms.restirBiasCorrection != 0u ?
                                         (oddFrame ? uniforms.restirSurfaceData1 : uniforms.restirSurfaceData0) :
                                         uniforms.restirSurfaceData0;
    device RestirReservoir* currentReservoirs = oddFrame ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
    device const RestirSurfaceHistory* history = oddFrame ? uniforms.restirHistory1 : uniforms.restirHistory0;
    const RestirSurfaceHistory currentSurface = history[tid];
    if (uniforms.restirDIEnabled == 0u || (currentSurface.materialIdAndFlags & RESTIR_SURFACE_VALID) == 0u)
    {
        return;
    }
    auditWork(uniforms, WORK_RESTIR_FINAL_ITEMS);

    device RestirDiagnosticRecord* diagnosticRecord = restirDiagnosticRecord(uniforms, tid);
    if (diagnosticRecord != nullptr)
    {
        RestirDiagnosticRecord empty = {};
        empty.pixelIndex = tid;
        *diagnosticRecord = empty;
    }

    RestirReservoir reservoir = initialReservoirs[tid];
    const uint32_t centerM = reservoir.state.M;
    uint32_t selectedAge = reservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
    uint32_t selectedSourceIndex = tid;

    SurfaceInteraction si;
    bool isFibre, isOpenPBR;
    ShadedFrame neeFrame;
    OpenPBR_PreparedBsdf openpbrPrepared;
    float3 storedThroughput;
    float curveRadius;
    uint32_t medium;
    uint32_t sampleIdx;
    const RestirTargetSurface stored = ((device const RestirTargetSurface*)surfaceData)[tid];
    sampleIdx = stored.sampleIdxAndMedium & RESTIR_TARGET_SAMPLE_MASK;
    medium = stored.sampleIdxAndMedium >> RESTIR_TARGET_MEDIUM_SHIFT;
    storedThroughput = restirTargetIsDirect(stored) ? float3(1.0f) : float3(stored.throughput);
    const float storedMotionTime = motionTimeFor(uniforms, tid, sampleIdx);
    const bool interpolateMotion =
        SPEC_MOTION_BLUR && uniforms.enableMotionBlur && storedMotionTime < 1.0f && prevVertexBuffer && indexBuffer;
    rebuildRestirTargetSurface(uniforms, restirTargetObjectToWorld(uniforms, instances, geometryEntries, stored),
                               materials, geometryEntries, vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints,
                               curveSegments, stored, interpolateMotion, storedMotionTime, si, isFibre, isOpenPBR,
                               neeFrame, openpbrPrepared, curveRadius);

    const uint2 pixel = uint2(tid % uniforms.width, tid / uniforms.width);
    const float motionTime = motionTimeFor(uniforms, tid, sampleIdx);
    SamplerState rng = samplerFor(uniforms, tid, sampleIdx, 0u);
    const uint32_t rotation = hash_combine(tid, uniforms.frameIndex * 0x9e3779b9u) & 15u;
    const uint32_t neighborCount = uniforms.spatialReuseEnabled != 0u ? min(uniforms.spatialNeighborCount, 16u) : 0u;
    if (neighborCount != 0u)
    {
        auditWork(uniforms, WORK_RESTIR_SPATIAL_ITEMS);
    }
    for (uint32_t i = 0u; i < neighborCount; ++i)
    {
        const int2 neighborPixel = int2(pixel) + kRestirNeighborOffsets[(rotation + i * 5u) & 15u];
        if (any(neighborPixel < int2(0)) || neighborPixel.x >= int(uniforms.width) ||
            neighborPixel.y >= int(uniforms.height))
        {
            continue;
        }
        const uint32_t neighborIndex = uint32_t(neighborPixel.y) * uniforms.width + uint32_t(neighborPixel.x);
        const RestirReservoir neighbor = initialReservoirs[neighborIndex];
        const RestirSurfaceHistory neighborSurface = history[neighborIndex];
        if (neighbor.state.M == 0u || !restirSurfaceHistoryCompatible(currentSurface, neighborSurface))
        {
            continue;
        }

        auditWork(uniforms, WORK_RESTIR_SPATIAL_MERGES);

        LightConnectionEvaluation evaluated = {};
        if ((neighbor.state.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u)
        {
            const LightConnection connection = reconnectRestirSample(
                uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer, motionTime, si,
                envAliasTable, envMapTexture, iesProfiles, neighbor.sample);
            evaluated = evaluateLightConnection(
                connection, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
        }
        SamplerState neighborRng = rng;
        neighborRng.seed = restirRngStreamSeed(rng.seed, RESTIR_RNG_SPATIAL_SALT + i * 0x9e3779b9u);
        const bool selectedSampleHasCurrentVisibility =
            (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u &&
            restirSamplesEqual(reservoir.sample, neighbor.sample);
        const uint32_t currentVisibilityFlags = reservoir.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
        const RestirVisibilityCache currentVisibility = reservoir.visibility;
        if (restirReservoirUpdate(reservoir.state, restirReservoirMergeWeight(neighbor.state, evaluated.target),
                                  evaluated.target, neighbor.state.M,
                                  random<SampleDimension::eLightId>(neighborRng, uniforms.samplerType)))
        {
            reservoir.sample = neighbor.sample;
            if (selectedSampleHasCurrentVisibility)
            {
                reservoir.state.ageAndFlags |= currentVisibilityFlags;
                reservoir.visibility = currentVisibility;
            }
            else if ((neighbor.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBLE) != 0u)
            {
                reservoir.state.ageAndFlags |= neighbor.state.ageAndFlags & RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
                reservoir.visibility = neighbor.visibility;
            }
            selectedAge = neighbor.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK;
            selectedSourceIndex = neighborIndex;
        }
    }
    reservoir.state.ageAndFlags =
        (reservoir.state.ageAndFlags & RESTIR_RESERVOIR_FLAGS_MASK) | min(selectedAge, RESTIR_RESERVOIR_AGE_MASK);
    if ((reservoir.state.ageAndFlags & RESTIR_RESERVOIR_VALID) == 0u)
    {
        const uint32_t maxM = max(uniforms.initialCandidateCount, 1u) * max(uniforms.reservoirMaxAge, 1u) *
                              (min(uniforms.spatialNeighborCount, 16u) + 1u);
        restirReservoirLimitM(reservoir.state, maxM);
        auditWork(uniforms, WORK_RESTIR_EFFECTIVE_M, reservoir.state.M);
        auditWork(uniforms, WORK_RESTIR_M_HISTOGRAM_BASE + restirMHistogramBin(reservoir.state.M));
        currentReservoirs[tid] = reservoir;
        return;
    }

    const bool selectedSpatial = selectedSourceIndex != tid;
    const bool selectedHistory = selectedAge != 0u;
    auditWork(uniforms, selectedSpatial ? WORK_RESTIR_FINAL_SOURCE_SPATIAL :
                        selectedHistory ? WORK_RESTIR_FINAL_SOURCE_TEMPORAL :
                                          WORK_RESTIR_FINAL_SOURCE_INITIAL);
    if (selectedHistory)
        auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY);
    auditWork(uniforms, WORK_RESTIR_AGE_HISTOGRAM_BASE + min(selectedAge, RESTIR_AUDIT_AGE_BINS - 1u));
    if (restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ANALYTIC)
        auditRestirLightId(uniforms, restirSampleLightId(reservoir.sample));

    const LightConnection connection =
        reconnectRestirSample(uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer,
                              motionTime, si, envAliasTable, envMapTexture, iesProfiles, reservoir.sample);
    const LightConnectionEvaluation evaluated =
        evaluateLightConnection(connection, si, isFibre, neeFrame, isOpenPBR, openpbrPrepared, uniforms.misHeuristic);
    reservoir.state.target = evaluated.target;
    if (diagnosticRecord != nullptr)
    {
        diagnosticRecord->stableLightId = restirSampleLightId(reservoir.sample);
        diagnosticRecord->selectedSourceIndex = selectedSourceIndex;
        diagnosticRecord->currentTarget = evaluated.target;
        diagnosticRecord->weightSum = reservoir.state.weightSum;
    }
    if (uniforms.restirBiasCorrection != 0u)
    {
        float selectedSourceTarget = selectedSourceIndex == tid ? evaluated.target : 0.0f;
        float sourceTargetSum = float(centerM) * evaluated.target;
        if (diagnosticRecord != nullptr)
        {
            diagnosticRecord->sourceCount = 1u;
            diagnosticRecord->sourceIndices[0] = tid;
            diagnosticRecord->sourceM[0] = centerM;
            diagnosticRecord->sourceTargets[0] = evaluated.target;
        }
        for (uint32_t i = 0u; i < neighborCount; ++i)
        {
            const int2 neighborPixel = int2(pixel) + kRestirNeighborOffsets[(rotation + i * 5u) & 15u];
            if (any(neighborPixel < int2(0)) || neighborPixel.x >= int(uniforms.width) ||
                neighborPixel.y >= int(uniforms.height))
            {
                continue;
            }
            const uint32_t neighborIndex = uint32_t(neighborPixel.y) * uniforms.width + uint32_t(neighborPixel.x);
            const RestirReservoir neighbor = initialReservoirs[neighborIndex];
            const RestirSurfaceHistory neighborSurface = history[neighborIndex];
            if (neighbor.state.M == 0u || !restirSurfaceHistoryCompatible(currentSurface, neighborSurface))
            {
                continue;
            }
            if (selectedSourceIndex == neighborIndex &&
                !(SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u))
            {
                selectedSourceTarget = neighbor.state.target;
                sourceTargetSum += float(neighbor.state.M) * selectedSourceTarget;
                if (diagnosticRecord != nullptr && diagnosticRecord->sourceCount < RESTIR_DIAGNOSTIC_SOURCE_COUNT)
                {
                    const uint32_t source = diagnosticRecord->sourceCount++;
                    diagnosticRecord->sourceIndices[source] = neighborIndex;
                    diagnosticRecord->sourceM[source] = neighbor.state.M;
                    diagnosticRecord->sourceTargets[source] = selectedSourceTarget;
                }
                continue;
            }
            const RestirTargetSurface neighborStored = ((device const RestirTargetSurface*)surfaceData)[neighborIndex];
            SurfaceInteraction neighborSi;
            bool neighborIsFibre, neighborIsOpenPBR;
            ShadedFrame neighborNeeFrame;
            OpenPBR_PreparedBsdf neighborOpenpbrPrepared;
            float neighborCurveRadius;
            const uint32_t neighborSampleIdx = neighborStored.sampleIdxAndMedium & RESTIR_TARGET_SAMPLE_MASK;
            const float neighborMotionTime = motionTimeFor(uniforms, neighborIndex, neighborSampleIdx);
            const bool neighborInterpolateMotion = SPEC_MOTION_BLUR && uniforms.enableMotionBlur &&
                                                   neighborMotionTime < 1.0f && prevVertexBuffer && indexBuffer;
            rebuildRestirTargetSurface(
                uniforms, restirTargetObjectToWorld(uniforms, instances, geometryEntries, neighborStored), materials,
                geometryEntries, vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints, curveSegments,
                neighborStored, neighborInterpolateMotion, neighborMotionTime, neighborSi, neighborIsFibre,
                neighborIsOpenPBR, neighborNeeFrame, neighborOpenpbrPrepared, neighborCurveRadius);
            const LightConnection selectedAtNeighbor = reconnectRestirSample(
                uniforms, lights, instances, materials, vertexBuffer, prevVertexBuffer, indexBuffer, neighborMotionTime,
                neighborSi, envAliasTable, envMapTexture, iesProfiles, reservoir.sample);
            float neighborTarget = restirTargetOnly(selectedAtNeighbor, neighborSi, neighborIsFibre, neighborNeeFrame,
                                                    neighborIsOpenPBR, neighborOpenpbrPrepared, uniforms.misHeuristic);
            if (SPEC_RESTIR_RAY_TRACED_DIAGNOSTIC && uniforms.restirBiasCorrection == 2u && neighborTarget > 0.0f)
            {
                auditWork(uniforms, WORK_RESTIR_DIAGNOSTIC_QUERIES);
                float ignoredTransmittance;
                if (!restirDiagnosticVisible(uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable,
                                             selectedAtNeighbor, selectedAtNeighbor.origin, instances, materials,
                                             geometryEntries, vertexBuffer, indexBuffer, ignoredTransmittance))
                {
                    neighborTarget = 0.0f;
                }
            }
            if (selectedSourceIndex == neighborIndex)
            {
                selectedSourceTarget = neighborTarget;
            }
            sourceTargetSum += float(neighbor.state.M) * neighborTarget;
            if (diagnosticRecord != nullptr && diagnosticRecord->sourceCount < RESTIR_DIAGNOSTIC_SOURCE_COUNT)
            {
                const uint32_t source = diagnosticRecord->sourceCount++;
                diagnosticRecord->sourceIndices[source] = neighborIndex;
                diagnosticRecord->sourceM[source] = neighbor.state.M;
                diagnosticRecord->sourceTargets[source] = neighborTarget;
            }
        }
        if (diagnosticRecord != nullptr)
        {
            diagnosticRecord->basicDenominator = sourceTargetSum;
        }
        restirReservoirApplyBasicNormalization(reservoir.state, selectedSourceTarget, sourceTargetSum);
    }
    const uint32_t maxM = max(uniforms.initialCandidateCount, 1u) * max(uniforms.reservoirMaxAge, 1u) *
                          (min(uniforms.spatialNeighborCount, 16u) + 1u);
    restirReservoirLimitM(reservoir.state, maxM);
    auditWork(uniforms, WORK_RESTIR_EFFECTIVE_M, reservoir.state.M);
    auditWork(uniforms, WORK_RESTIR_M_HISTOGRAM_BASE + restirMHistogramBin(reservoir.state.M));
    currentReservoirs[tid] = reservoir;
    if (diagnosticRecord != nullptr)
    {
        diagnosticRecord->normalization = restirReservoirNormalization(reservoir.state);
    }
    if (uniforms.restirDebugMode != 0u)
    {
        const float3 debugColor =
            uniforms.restirDebugMode == 1u ?
                float3(float(reservoir.state.ageAndFlags & RESTIR_RESERVOIR_AGE_MASK) /
                       float(max(uniforms.reservoirMaxAge, 1u))) :
                (restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ANALYTIC    ? float3(1.0f, 0.2f, 0.1f) :
                 restirSampleType(reservoir.sample) == RESTIR_SAMPLE_ENVIRONMENT ? float3(0.1f, 0.4f, 1.0f) :
                                                                                   float3(0.1f, 1.0f, 0.2f));
        radianceOut[tid] = float4(debugColor, 0.0f);
        return;
    }
    const float W = restirReservoirNormalization(reservoir.state);
    if (!(W > 0.0f))
    {
        return;
    }

    const float3 shadowOrigin =
        isFibre ? fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, connection.toLight) :
                  connection.origin;
    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(connection, shadowOrigin);
    const float3 weight = clampPathContribution(
        storedThroughput * evaluated.integrand * W, 0u, uniforms.clampDirect, uniforms.clampIndirect);
    if (!any(weight != 0.0f) || !visibility.valid)
    {
        return;
    }
    const uint32_t receiverMaterial = currentSurface.materialIdAndFlags & RESTIR_SURFACE_MATERIAL_MASK;
    const uint32_t receiverGeometry = restirVisibilityGeometryKey(stored);
    uint32_t receiverKey = 0u;
    uint32_t lightKey = 0u;
    bool exactInitialVisibility = false;
    bool conservativeVisibility = false;
    const bool visibilityCacheAllowed = !SPEC_SSS || !uniforms.hasBoundedMedium;
    if (visibilityCacheAllowed && (uniforms.restirInitialVisibility != 0u || uniforms.restirFinalVisibilityReuse != 0u))
    {
        receiverKey = restirVisibilityReceiverKey(uniforms, si, receiverMaterial, receiverGeometry);
        lightKey = restirVisibilityLightKey(uniforms, connection);
        exactInitialVisibility =
            uniforms.restirInitialVisibility != 0u && restirVisibilityAge(reservoir.visibility) == 0u &&
            restirVisibilityCacheMatches(uniforms, reservoir, si, receiverMaterial, receiverGeometry, lightKey, false);
        conservativeVisibility =
            uniforms.restirFinalVisibilityReuse != 0u && !exactInitialVisibility &&
            restirVisibilityCacheMatches(uniforms, reservoir, si, receiverMaterial, receiverGeometry, lightKey, true);
    }
    const bool reuseVisibility = exactInitialVisibility || conservativeVisibility;
    if (reuseVisibility)
    {
        float3 visibleWeight = weight * restirReservoirInitialVisibility(reservoir.state);
        if (conservativeVisibility)
        {
            auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_HITS);
            if (SPEC_RENDER_WORK_AUDIT && restirReservoirInitialVisibility(reservoir.state) > 0.0f)
            {
                auditWork(uniforms, WORK_RESTIR_VISIBILITY_CACHE_ORACLE_QUERIES);
                float oracleTransmittance;
                const bool oracleVisible = restirDiagnosticVisible(
                    uniforms, diagnosticAccelerationStructure, diagnosticFunctionTable, connection, shadowOrigin,
                    instances, materials, geometryEntries, vertexBuffer, indexBuffer, oracleTransmittance);
                auditWork(uniforms, oracleVisible ? WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_VISIBLE :
                                                    WORK_RESTIR_VISIBILITY_CACHE_VISIBLE_OCCLUDED);
            }
        }
        if (SPEC_FOG && uniforms.hasFog)
        {
            const float tau = fogOpticalDepth(
                shadowOrigin, visibility.direction, visibility.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
            visibleWeight *= exp(-tau);
        }
        if (all(visibleWeight <= 1e-6f))
        {
            return;
        }
        addFilteredRadiance(radianceOut, tid, visibleWeight);
        restirDiagnosticVisibility(uniforms, tid, visibleWeight);
        if (restirVisibilityAge(reservoir.visibility) == 0u)
            auditWork(uniforms, WORK_RESTIR_INITIAL_VISIBILITY_REUSED);
        if (selectedHistory)
        {
            auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
        }
        return;
    }
    ShadowRay sr;
    sr.origin = packed_float3(shadowOrigin);
    sr.direction = packed_float3(visibility.direction);
    sr.weight = packed_float3(weight);
    sr.maxDistance = visibility.maxDistance;
    sr.pixelIndex = tid;
    sr.medium = medium;
    sr.sharcRadiance = packed_float3(float3(0.0f));
    const bool updateVisibilityCache = visibilityCacheAllowed && uniforms.restirFinalVisibilityReuse != 0u;
    if (updateVisibilityCache)
    {
        reservoir.state.ageAndFlags &= ~RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK;
        restirVisibilityStore(reservoir.visibility, receiverKey, lightKey, 0u);
        currentReservoirs[tid] = reservoir;
    }
    sr.sharcPathIndex = tid | (SPEC_RENDER_WORK_AUDIT && selectedHistory ? RESTIR_AUDIT_HISTORY_BIT : 0u) |
                        (updateVisibilityCache ? RESTIR_VISIBILITY_UPDATE_BIT : 0u);
    sr.alphaThreshold = random<SampleDimension::eShadowRR>(rng, uniforms.samplerType);
    const uint32_t slot = atomic_fetch_add_explicit(shadowCounter, 1u, memory_order_relaxed);
    storeShadowRay(shadowRays, slot, uniforms.width * uniforms.height, sr);
    auditWork(uniforms, WORK_RESTIR_FINAL_VISIBILITY_RAYS);
    if (selectedHistory)
        auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_RAYS);
}

// ---------------------------------------------------------------------------
// guide -- compact deterministic continuation for MetalFX auxiliary buffers
// ---------------------------------------------------------------------------
static inline void storeGuideMaterial(
    device AovSample* aov, uint32_t tid, thread const DenoiserMaterialGuides& guides, float3 normal, float hitDistance)
{
    AovSample out = aov[tid];
    const float state = out.guideStateOrBounceDepth;
    const bool blendTransmission = state >= 1.0f;
    const float interfaceFresnel = blendTransmission ? saturate(state - 1.0f) : 0.0f;
    const float transmissionWeight = 1.0f - interfaceFresnel;
    const float3 guideNormal = normalize(normal);
    const float3 blendedNormal = float3(out.normal) * interfaceFresnel + guideNormal * transmissionWeight;
    out.diffuseAlbedo = packed_float3(guides.diffuse * transmissionWeight);
    out.specularAlbedo = packed_float3(float3(interfaceFresnel) + guides.specular * transmissionWeight);
    out.normal = packed_float3(
        blendTransmission && dot(blendedNormal, blendedNormal) > 1e-8f ? normalize(blendedNormal) : guideNormal);
    out.roughness = blendTransmission ? mix(out.roughness, guides.roughness, transmissionWeight) : guides.roughness;
    out.specularHitDistance = hitDistance;
    out.guideStateOrBounceDepth = 0.0f;
    aov[tid] = out;
}

static inline void storeGuideMiss(device AovSample* aov, uint32_t tid, float3 rayDirection, float hitDistance)
{
    DenoiserMaterialGuides sky;
    sky.diffuse = float3(0.0f);
    sky.specular = float3(0.0f);
    sky.roughness = 1.0f;
    sky.transmission = 0.0f;
    storeGuideMaterial(aov, tid, sky, -rayDirection, hitDistance);
}

template <typename T>
static void guideImpl(uint gid,
                      constant Uniforms& uniforms,
                      constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                      typename T::structure accelerationStructure,
                      device AovSample* aov,
                      device const Material* materials,
                      device const GeometryEntry* geometryEntries,
                      device const char* vertexBuffer,
                      device const char* prevVertexBuffer,
                      device const uint32_t* indexBuffer,
                      device const packed_float3* curvePoints,
                      device const uint32_t* curveSegments,
                      typename T::table functionTable,
                      device const uint32_t* control)
{
    if (gid >= control[WF_CTRL_GUIDE_N])
    {
        return;
    }
    if (gid == 0u)
    {
        auditWork(uniforms, WORK_GUIDE_DISPATCHES);
    }
    gid = uniforms.guideQueue[gid];

    GuideRay guide = uniforms.guideRays[gid];
    if ((guide.flags & GUIDE_RAY_ACTIVE) == 0u)
    {
        return;
    }
    auditWork(uniforms, WORK_GUIDE_ACTIVE_ITEMS);

    const bool replaceMaterial = (guide.flags & GUIDE_RAY_REPLACE_MATERIAL) != 0u;
    const float motionTime = motionTimeFor(uniforms, gid, 0u);
    float totalDistance = 0.0f;
    uint32_t describedSurfaces = 0u;

    // Six traversal attempts allow deterministic pass-through of a few alpha
    // layers while material continuation itself remains capped at two hits.
    for (uint32_t attempt = 0u; attempt < 6u; ++attempt)
    {
        const float3 origin = float3(guide.origin);
        const float3 direction = normalize(float3(guide.direction));
        if (!all(isfinite(origin)) || !all(isfinite(direction)))
        {
            return;
        }

        ray r;
        r.origin = origin;
        r.direction = direction;
        r.min_distance = 1e-6f;
        r.max_distance = INFINITY;

        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        auditWork(uniforms, WORK_GUIDE_ONLY_RAYS);
        auditWork(uniforms, WORK_INTERSECTION_QUERIES);
        const typename T::isect::result_type rawHit =
            T::trace(isect, r, accelerationStructure, uniforms.primaryRayMask | GEOMETRY_MASK_LIGHT_HIDDEN, motionTime,
                     functionTable);
        const float curveParameter = rawHit.type == intersection_type::curve ? T::curveParameter(rawHit) : 0.0f;
        const ExtendIntersection hit = captureExtendIntersection(rawHit, curveParameter, rawHit.instance_id);
        if (hit.type == intersection_type::none)
        {
            const float missDistance = totalDistance + max(uniforms.sceneExtent, 1e3f);
            if (replaceMaterial)
            {
                storeGuideMiss(aov, gid, direction, missDistance);
            }
            else
            {
                aov[gid].specularHitDistance = missDistance;
            }
            return;
        }

        totalDistance += hit.distance;
        const auto inst = instances[hit.instanceId];
        const uint32_t geometryMask = inst.mask & ~GEOMETRY_MASK_UNIFORM_ORTHOGONAL_TRANSFORM;
        if (geometryMask == GEOMETRY_MASK_LIGHT || geometryMask == GEOMETRY_MASK_LIGHT_HIDDEN)
        {
            if (replaceMaterial)
            {
                storeGuideMiss(aov, gid, direction, totalDistance);
            }
            else
            {
                aov[gid].specularHitDistance = totalDistance;
            }
            return;
        }

        const uint32_t geometryEntryIndex = inst.userID + hit.geometryId;
        const GeometryEntry entry = geometryEntries[geometryEntryIndex];
        // The common glossy case needs only the first opaque hit distance. It
        // avoids all vertex/material texture traffic here.
        if (!replaceMaterial && materials[entry.materialId].alpha_mode == ALPHA_MODE_OPAQUE)
        {
            aov[gid].specularHitDistance = totalDistance;
            return;
        }

        const float4x4 objectToWorld =
            geometryObjectToWorld(uniforms, instances, hit.instanceId, geometryEntryIndex, entry);
        const float3 worldPosition = origin + direction * hit.distance;
        const bool isCurve = SPEC_CURVES && (entry.flags & GEOM_FLAG_CURVE) != 0u;

        float3 objectNormal, objectTangent, vertexColor, objectGeomNormal;
        float3 shadingNormal, shadingTangent, shadingGeomNormal;
        float2 uv;
        float tangentSign = 1.0f;
        if (isCurve)
        {
            float curveRadius = 0.0f;
            fetchCurve(curvePoints, curveSegments, entry, hit.primitiveId, hit.curveParameter, worldPosition,
                       objectToWorld, shadingNormal, shadingTangent, uv, curveRadius);
            shadingGeomNormal = shadingNormal;
            vertexColor = float3(1.0f);
        }
        else
        {
            float uvArea2 = 0.0f;
            const bool interpolateMotion =
                SPEC_MOTION_BLUR && uniforms.enableMotionBlur && motionTime < 1.0f && prevVertexBuffer && indexBuffer;
            fetchTriangleBlended(vertexBuffer, prevVertexBuffer, indexBuffer, entry, hit.primitiveId, interpolateMotion,
                                 motionTime, hit.barycentrics, objectNormal, objectTangent, uv, vertexColor,
                                 tangentSign, objectGeomNormal, uvArea2);
            const float3 axisX = objectToWorld[0].xyz;
            const float3 axisY = objectToWorld[1].xyz;
            const float3 axisZ = objectToWorld[2].xyz;
            const FastNormalTransform normalTransform = makeFastNormalTransform(axisX, axisY, axisZ);
            shadingNormal = transformNormalFast(objectNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                                normalTransform.cofactorZ, normalTransform.orientation);
            shadingGeomNormal =
                transformNormalFast(objectGeomNormal, normalTransform.cofactorX, normalTransform.cofactorY,
                                    normalTransform.cofactorZ, normalTransform.orientation);
            shadingTangent = orthonormalizeTangent(shadingNormal, transformDirection(objectTangent, axisX, axisY, axisZ));
        }

        const float3 binormal = cross(shadingNormal, shadingTangent) * tangentSign;
        SurfaceInteraction si;
        initSurfaceInteraction(si, materials[entry.materialId], worldPosition, shadingNormal, shadingGeomNormal,
                               shadingTangent, binormal, uv, direction, vertexColor, -1e30f);

        // Guides must be stable, so BLEND coverage uses a fixed threshold rather
        // than consuming the radiance path's random opacity decision.
        if (si.opacity < 0.5f)
        {
            const float3 faceNg = dot(shadingGeomNormal, direction) > 0.0f ? shadingGeomNormal : -shadingGeomNormal;
            guide.origin = packed_float3(offset_ray(worldPosition, faceNg));
            continue;
        }
        if (!replaceMaterial)
        {
            aov[gid].specularHitDistance = totalDistance;
            return;
        }

        const bool isOpenPBR = SPEC_ALL_OPENPBR || (SPEC_OPENPBR && si.material_type == MATERIAL_TYPE_OPENPBR);
        OpenPBRParams openpbrMat;
        if (isOpenPBR)
        {
            openpbrMat = uniforms.openpbrParams[entry.materialId];
            if (openpbrMat.texture_mask != 0u && uniforms.openpbrTextures != nullptr)
            {
                applyOpenPBRTextures(openpbrMat, uniforms.openpbrTextures[entry.materialId], si, uv);
            }
        }

        const bool entering = si.front_face;
        const float2 mediaIors = unpackGuideIors(guide.mediaIors);
        const float currentIor = max(mediaIors.x, 1.0f);
        const float exteriorIor = max(mediaIors.y, 1.0f);
        si.exterior_ior = entering ? currentIor : exteriorIor;
        DenoiserMaterialGuides materialGuides;
        if (isOpenPBR)
        {
            const OpenPBR_ResolvedInputs openpbrInputs = openpbr_resolve_inputs(openpbrMat, si);
            materialGuides = openpbrDenoiserGuides(openpbrInputs, si, openpbrBaseMapDetailsSubsurface(openpbrMat));
        }
        else
        {
            materialGuides = standardDenoiserGuides(si);
        }

        constexpr float kGuideRoughnessFloor = 0.05f;
        const bool opaque = materialGuides.transmission <= kGuideRoughnessFloor;
        const bool hasDiffuse = luminance(materialGuides.diffuse) > 1e-3f;
        ++describedSurfaces;
        if ((opaque && (hasDiffuse || materialGuides.roughness > kGuideRoughnessFloor)) || describedSurfaces >= 2u)
        {
            storeGuideMaterial(aov, gid, materialGuides, float3(si.shading_normal), totalDistance);
            return;
        }

        const float3 V = normalize(float3(si.wo));
        const float3 Ns = normalize(float3(si.shading_normal));
        const float3 Nf = dot(Ns, V) >= 0.0f ? Ns : -Ns;
        float3 nextDirection = reflect_dir(-V, Nf);
        bool transmitted = false;
        float nextCurrentIor = currentIor;
        float nextExteriorIor = exteriorIor;
        if (!opaque)
        {
            const float materialIor = denoiserInterfaceIor(isOpenPBR, openpbrMat, si, entering);
            const float outsideIor = entering ? currentIor : exteriorIor;
            const float eta = entering ? outsideIor / materialIor : materialIor / outsideIor;
            const float interfaceCosine = abs(dot(Nf, V));
            const float fresnel = fresnel_dielectric(interfaceCosine, eta);
            float3 refracted;
            const bool validRefraction = si.thin_walled ? true : refract_dir(-V, Nf, eta, interfaceCosine, refracted);
            if (validRefraction && fresnel < 0.5f)
            {
                nextDirection = si.thin_walled ? -V : refracted;
                transmitted = true;
                if (!si.thin_walled)
                {
                    nextCurrentIor = entering ? materialIor : outsideIor;
                    nextExteriorIor = entering ? outsideIor : 1.0f;
                }
            }
        }

        const float3 faceNg =
            dot(float3(si.geometry_normal), V) > 0.0f ? float3(si.geometry_normal) : -float3(si.geometry_normal);
        guide.origin = packed_float3(offset_ray(si.position, transmitted ? -faceNg : faceNg));
        guide.direction = packed_float3(normalize(nextDirection));
        guide.mediaIors = packGuideIors(nextCurrentIor, nextExteriorIor);
    }

    // A pathological alpha stack is still a valid reflected depth. Using the
    // scene bound is safer for temporal reprojection than leaving zero.
    const float fallbackDistance = totalDistance + max(uniforms.sceneExtent, 1e3f);
    if (replaceMaterial)
    {
        storeGuideMiss(aov, gid, float3(guide.direction), fallbackDistance);
    }
    else
    {
        aov[gid].specularHitDistance = fallbackDistance;
    }
}

#define WF_GUIDE_ENTRY(NAME, TRAITS)                                                                                   \
    kernel void NAME(                                                                                                  \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                               \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],                          \
        TRAITS::structure accelerationStructure [[buffer(2)]], device AovSample* aov [[buffer(4)]],                    \
        device const Material* materials [[buffer(5)]], device const GeometryEntry* geometryEntries [[buffer(6)]],     \
        device const char* vertexBuffer [[buffer(7)]], device const char* prevVertexBuffer [[buffer(8)]],              \
        device const uint32_t* indexBuffer [[buffer(9)]], device const packed_float3* curvePoints [[buffer(10)]],      \
        device const uint32_t* curveSegments [[buffer(11)]], TRAITS::table functionTable [[buffer(13)]],               \
        device const uint32_t* control [[buffer(14)]])                                                                 \
    {                                                                                                                  \
        guideImpl<TRAITS>(gid, uniforms, instances, accelerationStructure, aov, materials, geometryEntries,            \
                          vertexBuffer, prevVertexBuffer, indexBuffer, curvePoints, curveSegments, functionTable,      \
                          control);                                                                                    \
    }

WF_GUIDE_ENTRY(wavefrontGuide, MotionTraversal)
WF_GUIDE_ENTRY(wavefrontGuideStatic, StaticTraversal)
WF_GUIDE_ENTRY(wavefrontGuideCurve, CurveMotionTraversal)
WF_GUIDE_ENTRY(wavefrontGuideStaticCurve, CurveStaticTraversal)

kernel void wavefrontStageBreadcrumb(device atomic_uint* stage [[buffer(0)]], constant uint32_t& stageIndex [[buffer(1)]])
{
    atomic_store_explicit(stage, stageIndex, memory_order_relaxed);
}

static inline void prepareTraversalDispatches(
    device uint32_t* args, uint32_t active, uint32_t threadsPerGroup, uint32_t batchThreads, uint32_t batchCount)
{
    for (uint32_t batch = 0u; batch < batchCount; ++batch)
    {
        const uint32_t offset = batch * batchThreads;
        const uint32_t n = active > offset ? min(active - offset, batchThreads) : 0u;
        args[batch * 3u + 0u] = (n + threadsPerGroup - 1u) / threadsPerGroup;
        args[batch * 3u + 1u] = 1u;
        args[batch * 3u + 2u] = 1u;
    }
}

kernel void wavefrontPrepare(device uint32_t& controlRef [[buffer(0)]],
                             constant uint32_t& srcIdx [[buffer(1)]],
                             constant uint32_t& threadsPerGroup [[buffer(2)]],
                             constant uint32_t& bounceIdx [[buffer(3)]],
                             device uint32_t* stageStats [[buffer(4)]],
                             device const PathState* paths [[buffer(5)]],
                             device const PathRay* rays [[buffer(6)]],
                             device const uint32_t* queue [[buffer(7)]],
                             constant uint32_t& diagnosticsEnabled [[buffer(8)]],
                             device uint32_t* traversalDispatches [[buffer(9)]],
                             constant uint32_t& traversalBatchThreads [[buffer(10)]],
                             constant uint32_t& traversalBatchCount [[buffer(11)]],
                             device const MediumPathState* mediumPaths [[buffer(12)]],
                             constant uint32_t& diagnosticsMediumEnabled [[buffer(13)]],
                             device uint32_t* sssControl [[buffer(14)]],
                             device uint32_t* hitQueue [[buffer(15)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = min(control[srcIdx], control[WF_CTRL_CAPACITY]);
    if (diagnosticsEnabled != 0u)
    {
        const uint32_t diagBase = WF_DIAG_BASE + min(bounceIdx, WF_DIAG_BOUNCES - 1u) * WF_DIAG_STRIDE;
        stageStats[diagBase] = n;
        for (uint32_t lane = 0u; lane < min(n, (uint32_t)WF_DIAG_LANES); ++lane)
        {
            const uint32_t laneBase = diagBase + 2u + lane * WF_DIAG_LANE_WORDS;
            const uint32_t tid = queue[lane];
            stageStats[laneBase] = tid;
            if (tid >= control[WF_CTRL_CAPACITY])
            {
                stageStats[laneBase + 1u] = 0xffffffffu;
                continue;
            }
            const PathState p = paths[tid];
            const PathRay r = rays[tid];
            const float3 origin = float3(r.origin);
            const float3 direction = float3(r.direction);
            stageStats[laneBase + 1u] = diagnosticsMediumEnabled != 0u ? mediumPaths[tid].medium : 0u;
            stageStats[laneBase + 2u] = p.depthAndFlags;
            stageStats[laneBase + 3u] = as_type<uint32_t>(origin.x);
            stageStats[laneBase + 4u] = as_type<uint32_t>(origin.y);
            stageStats[laneBase + 5u] = as_type<uint32_t>(origin.z);
            stageStats[laneBase + 6u] = as_type<uint32_t>(direction.x);
            stageStats[laneBase + 7u] = as_type<uint32_t>(direction.y);
            stageStats[laneBase + 8u] = as_type<uint32_t>(direction.z);
            stageStats[laneBase + 9u] = as_type<uint32_t>(dot(direction, direction));
            stageStats[laneBase + 10u] =
                as_type<uint32_t>(max(max(float3(p.throughput).x, float3(p.throughput).y), float3(p.throughput).z));
        }
        control[WF_CTRL_STATS_PATHS + min(bounceIdx, 31u)] = n;
    }
    control[WF_CTRL_ACTIVE] = n;
    control[WF_CTRL_DISPATCH + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_DISPATCH + 1] = 1u;
    control[WF_CTRL_DISPATCH + 2] = 1u;
    prepareTraversalDispatches(traversalDispatches, n, threadsPerGroup, traversalBatchThreads, traversalBatchCount);
    // The stages about to run append into these, so clear their counts before
    // anything can add to them.
    control[1u - srcIdx] = 0u;
    control[WF_CTRL_SHADOW] = 0u;
    control[WF_CTRL_HIT] = 0u;
    control[WF_CTRL_MISS] = 0u;
    sssControl[0] = 0u;
    for (uint32_t bucket = 0u; bucket < WF_HIT_BUCKET_COUNT; ++bucket)
    {
        hitQueue[bucket] = 0u;
    }
}

// Between `extend` and the two stages that consume its classification.
kernel void wavefrontPrepareHitMiss(device uint32_t& controlRef [[buffer(0)]],
                                    constant uint32_t& threadsPerGroup [[buffer(1)]],
                                    device const uint32_t* hitQueue [[buffer(2)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t capacity = control[WF_CTRL_CAPACITY];
    const uint32_t baseCount = min(hitQueue[0], capacity);
    const uint32_t layerCount = min(hitQueue[1], capacity);
    const uint32_t translucentCount = min(hitQueue[2], capacity);
    const uint32_t tailCount = min(hitQueue[3], capacity);
    const uint32_t h = min(baseCount + layerCount + translucentCount + tailCount, capacity);
    control[WF_CTRL_HIT_N] = h;
    control[WF_CTRL_HIT_DIS + 0] = (h + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_HIT_DIS + 1] = 1u;
    control[WF_CTRL_HIT_DIS + 2] = 1u;

    const uint32_t base = min(baseCount, h);
    control[WF_CTRL_SHADE_BASE_DIS + 0] = (base + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADE_BASE_DIS + 1] = 1u;
    control[WF_CTRL_SHADE_BASE_DIS + 2] = 1u;
    const uint32_t layer = min(layerCount, h - base);
    control[WF_CTRL_SHADE_LAYER_START] = base;
    control[WF_CTRL_SHADE_LAYER_DIS + 0] = (layer + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADE_LAYER_DIS + 1] = 1u;
    control[WF_CTRL_SHADE_LAYER_DIS + 2] = 1u;
    const uint32_t translucent = min(translucentCount, h - base - layer);
    control[WF_CTRL_SHADE_TRANSLUCENT_START] = base + layer;
    control[WF_CTRL_SHADE_TRANSLUCENT_DIS + 0] = (translucent + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADE_TRANSLUCENT_DIS + 1] = 1u;
    control[WF_CTRL_SHADE_TRANSLUCENT_DIS + 2] = 1u;
    control[WF_CTRL_SHADE_TAIL_START] = base + layer + translucent;
    control[WF_CTRL_SHADE_TAIL_DIS + 0] = (h - base - layer - translucent + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADE_TAIL_DIS + 1] = 1u;
    control[WF_CTRL_SHADE_TAIL_DIS + 2] = 1u;

    const uint32_t m = min(control[WF_CTRL_MISS], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_MISS_N] = m;
    control[WF_CTRL_MISS_DIS + 0] = (m + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_MISS_DIS + 1] = 1u;
    control[WF_CTRL_MISS_DIS + 2] = 1u;
    control[WF_CTRL_RESTIR] = 0u;
    control[WF_CTRL_RESTIR_DIS + 0] = 0u;
    control[WF_CTRL_RESTIR_DIS + 1] = 1u;
    control[WF_CTRL_RESTIR_DIS + 2] = 1u;
}

// Between `shade` and `shadow`: publish the number of shadow rays `shade`
// emitted and size their dispatch. Separate from wavefrontPrepare because the
// count does not exist until `shade` has run.
kernel void wavefrontPrepareShadow(device uint32_t& controlRef [[buffer(0)]],
                                   constant uint32_t& threadsPerGroup [[buffer(1)]],
                                   constant uint32_t& bounceIdx [[buffer(2)]],
                                   device uint32_t* traversalDispatches [[buffer(3)]],
                                   constant uint32_t& traversalBatchThreads [[buffer(4)]],
                                   constant uint32_t& traversalBatchCount [[buffer(5)]])
{
    device uint32_t* control = &controlRef;
    const uint32_t n = min(control[WF_CTRL_SHADOW], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_SHADOW_N] = n;
    control[WF_CTRL_STATS_SHADOW + min(bounceIdx, 31u)] = n;
    control[WF_CTRL_SHADOW_DIS + 0] = (n + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_SHADOW_DIS + 1] = 1u;
    control[WF_CTRL_SHADOW_DIS + 2] = 1u;
    prepareTraversalDispatches(traversalDispatches, n, threadsPerGroup, traversalBatchThreads, traversalBatchCount);
    const uint32_t guides = min(control[WF_CTRL_GUIDE], control[WF_CTRL_CAPACITY]);
    control[WF_CTRL_GUIDE_N] = guides;
    control[WF_CTRL_GUIDE_DIS + 0] = (guides + threadsPerGroup - 1u) / threadsPerGroup;
    control[WF_CTRL_GUIDE_DIS + 1] = 1u;
    control[WF_CTRL_GUIDE_DIS + 2] = 1u;
}

template <typename T>
static float3 mediumTransmittance(typename T::volume_structure accelerationStructure,
                                  constant Uniforms& uniforms,
                                  device const Material* materials,
                                  device const GeometryEntry* geometryEntries,
                                  constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                  float3 origin,
                                  float3 direction,
                                  float maxDistance,
                                  uint32_t startMedium,
                                  float motionTime)
{
    constexpr uint32_t kMaxCrossings = 8u;

    float3 optical = float3(0.0f);
    float travelled = 0.0f;
    uint32_t medium = startMedium;

    for (uint32_t i = 0u; i < kMaxCrossings; ++i)
    {
        const float remaining = maxDistance - travelled;
        if (remaining <= 1e-5f)
        {
            break;
        }

        typename T::volume_isect isect;
        isect.assume_geometry_type(geometry_type::triangle);
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);

        ray r;
        r.origin = origin + direction * travelled;
        r.direction = direction;
        r.min_distance = 1e-4f;
        r.max_distance = remaining;

        const auto hit = T::traceVolume(isect, r, accelerationStructure, GEOMETRY_MASK_MEDIUM, motionTime);
        const bool escaped = (hit.type == intersection_type::none);
        const float segment = escaped ? remaining : hit.distance;

        if (medium != 0u)
        {
            optical += mediumSigmaT(uniforms, materials, medium - 1u) * segment;
        }
        if (escaped)
        {
            break;
        }

        // The same toggle the crossing in `shade` uses, for the same reason: a
        // gizmo's winding is arbitrary, so the normal cannot say which way the
        // ray is going.
        const auto inst = instances[hit.instance_id];
        const uint32_t here = (geometryEntries[inst.userID + hit.geometry_id].materialId + 1u) & MEDIUM_INDEX_MASK;
        medium = (medium == here) ? 0u : here;

        travelled += segment + 1e-4f;
    }

    return exp(-optical);
}

static inline bool cutoutRouletteDone(thread float& transmittance, float opacity, float cutoff)
{
    transmittance *= (1.0f - opacity);
    if (SPEC_STOCHASTIC_ALPHA_VISIBILITY)
    {
        return transmittance <= cutoff;
    }
    return transmittance <= cutoff || transmittance <= 1e-6f;
}

// How many cutout crossings the restart walk below may take before it gives up
// and reports what it has. It bounds that walk only -- the inline query needs no
// bound, because it answers the whole ray in one traversal.
constant uint32_t kMaxCutoutCrossings = 16u;

template <typename T, bool Inline, bool HardwareAlpha>
struct CutoutShadowWalk
{
    static bool run(constant Uniforms& uniforms,
                    typename T::structure as,
                    ray shadowRay,
                    float motionTime,
                    float cutoff,
                    device const Material* materials,
                    device const AlphaMaterialData* alphaMaterials,
                    device const PrimitiveAlphaData* primitiveAlphaData,
                    device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                    device const GeometryEntry* geometryEntries,
                    device const char* vertexBuffer,
                    device const uint32_t* indexBuffer,
                    typename T::table functionTable,
                    uint32_t,
                    bool usePrimitiveAlphaData,
                    bool useCompactAlphaMaterials,
                    thread float& transmittance)
    {
        transmittance = 1.0f;
        ray probe = shadowRay;
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(false);
        for (uint32_t crossing = 0u; crossing < kMaxCutoutCrossings; ++crossing)
        {
            if (crossing != 0u)
                auditWork(uniforms, WORK_ALPHA_TRAVERSAL_RESTARTS);
            const auto hit = T::trace(isect, probe, as, RAY_MASK_SHADOW, motionTime, functionTable);
            if (hit.type == intersection_type::none)
            {
                return true; // nothing else in the way
            }
            // Curve geometry is built opaque and carries no cutout, so a strand
            // blocks outright rather than being alpha tested.
            float opacity = 1.0f;
            if (hit.type == intersection_type::triangle)
            {
                opacity = useCompactAlphaMaterials ?
                              cutoutOpacityAtSelectedCompact(hit.primitive_id, hit.geometry_id, hit.user_instance_id,
                                                             hit.triangle_barycentric_coord, alphaMaterials,
                                                             primitiveAlphaData, primitiveAlphaDecode, geometryEntries,
                                                             vertexBuffer, indexBuffer, usePrimitiveAlphaData, false) :
                              cutoutOpacityAtSelected(hit.primitive_id, hit.geometry_id, hit.user_instance_id,
                                                      hit.triangle_barycentric_coord, materials, primitiveAlphaData,
                                                      primitiveAlphaDecode, geometryEntries, vertexBuffer, indexBuffer,
                                                      usePrimitiveAlphaData);
            }
            if (cutoutRouletteDone(transmittance, opacity, cutoff))
            {
                return false;
            }
            // Past the candidate just tested. Relative, because an absolute
            // epsilon is either too small to clear a distant hit or big enough
            // to step over a near one.
            probe.min_distance = hit.distance * (1.0f + 1e-5f) + 1e-5f;
            if (probe.min_distance >= probe.max_distance)
            {
                return true;
            }
        }
        return true;
    }
};

template <typename T>
struct CutoutShadowWalk<T, true, false>
{
    static bool run(constant Uniforms&,
                    typename T::structure as,
                    ray shadowRay,
                    float,
                    float cutoff,
                    device const Material* materials,
                    device const AlphaMaterialData* alphaMaterials,
                    device const PrimitiveAlphaData* primitiveAlphaData,
                    device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                    device const GeometryEntry* geometryEntries,
                    device const char* vertexBuffer,
                    device const uint32_t* indexBuffer,
                    typename T::table,
                    uint32_t directGeometryBase,
                    bool usePrimitiveAlphaData,
                    bool useCompactAlphaMaterials,
                    thread float& transmittance)
    {
        transmittance = 1.0f;
        // Accept any hit and restrict geometry to avoid nearest-hit work and irrelevant bounding-box candidates.
        intersection_params params;
        params.accept_any_intersection(true);
        // RAY_MASK_SHADOW excludes the procedural light instances. StaticTraversal's
        // query is triangle-only, so neither a bounding-box type nor an
        // intersection-function table belongs on this path.
        params.assume_geometry_type(geometry_type::triangle);
        typename T::query q;
        T::resetShadowQuery(q, shadowRay, as, params);
        while (q.next())
        {
            const uint32_t geometryEntryBase = T::shadowGeometryEntryBase(q, directGeometryBase);
            const float opacity =
                useCompactAlphaMaterials ?
                    cutoutOpacityAtSelectedCompact(
                        q.get_candidate_primitive_id(), q.get_candidate_geometry_id(), geometryEntryBase,
                        q.get_candidate_triangle_barycentric_coord(), alphaMaterials, primitiveAlphaData,
                        primitiveAlphaDecode, geometryEntries, vertexBuffer, indexBuffer, usePrimitiveAlphaData, true) :
                    cutoutOpacityAtSelected(q.get_candidate_primitive_id(), q.get_candidate_geometry_id(),
                                            geometryEntryBase, q.get_candidate_triangle_barycentric_coord(), materials,
                                            primitiveAlphaData, primitiveAlphaDecode, geometryEntries, vertexBuffer,
                                            indexBuffer, usePrimitiveAlphaData);
            if (cutoutRouletteDone(transmittance, opacity, cutoff))
            {
                q.abort();
                return false;
            }
        }
        // Anything committed during that walk was opaque, and blocks outright.
        return q.get_committed_intersection_type() == intersection_type::none;
    }
};

template <typename T>
struct CutoutShadowWalk<T, false, true>
{
    static bool run(constant Uniforms&,
                    typename T::structure as,
                    ray shadowRay,
                    float,
                    float,
                    device const Material*,
                    device const AlphaMaterialData*,
                    device const PrimitiveAlphaData*,
                    device const PrimitiveAlphaDecode*,
                    device const GeometryEntry*,
                    device const char*,
                    device const uint32_t*,
                    typename T::table functionTable,
                    uint32_t,
                    bool,
                    bool,
                    thread float& transmittance)
    {
        transmittance = 1.0f;
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.accept_any_intersection(true);
        return T::trace(isect, shadowRay, as, RAY_MASK_SHADOW, 0.0f, functionTable).type == intersection_type::none;
    }
};

static bool restirDiagnosticVisible(constant Uniforms& uniforms,
                                    CurveStaticTraversal::structure accelerationStructure,
                                    CurveStaticTraversal::table functionTable,
                                    thread const LightConnection& connection,
                                    float3 shadowOrigin,
                                    constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                                    device const Material* materials,
                                    device const GeometryEntry* geometryEntries,
                                    device const char* vertexBuffer,
                                    device const uint32_t* indexBuffer,
                                    thread float& transmittance)
{
    const EmissiveVisibilitySegment segment = lightVisibilitySegment(connection, shadowOrigin);
    if (!segment.valid)
    {
        transmittance = 0.0f;
        return false;
    }
    ray shadowRay;
    shadowRay.origin = shadowOrigin;
    shadowRay.direction = segment.direction;
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = segment.maxDistance;
    if (!SPEC_ALPHA)
    {
        transmittance = 1.0f;
        CurveStaticTraversal::isect isect;
        isect.assume_geometry_type(CurveStaticTraversal::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        return CurveStaticTraversal::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, 0.0f, functionTable)
                   .type == intersection_type::none;
    }
    float alphaTransmittance;
    const bool visible = CutoutShadowWalk<CurveStaticTraversal, false, false>::run(
        uniforms, accelerationStructure, shadowRay, 0.0f, 0.0f, materials, nullptr, nullptr, nullptr, geometryEntries,
        vertexBuffer, indexBuffer, functionTable, 0u, false, false, alphaTransmittance);
    transmittance = alphaTransmittance;
    return visible;
}

// ---------------------------------------------------------------------------
// shadow -- resolve the deferred connections
// ---------------------------------------------------------------------------
static void restirStoreFinalVisibility(constant Uniforms& uniforms, uint32_t pixelIndex, float transmittance)
{
    device RestirReservoir* reservoirs =
        (uniforms.frameIndex & 1u) != 0u ? uniforms.restirReservoir1 : uniforms.restirReservoir0;
    const uint32_t encoded = uint32_t(clamp(transmittance, 0.0f, 1.0f) * float(RESTIR_RESERVOIR_VISIBILITY_BITS) + 0.5f);
    device uint32_t& flags = reservoirs[pixelIndex].state.ageAndFlags;
    flags = (flags & ~RESTIR_RESERVOIR_INITIAL_VISIBILITY_MASK) | RESTIR_RESERVOIR_INITIAL_VISIBLE |
            (encoded << RESTIR_RESERVOIR_VISIBILITY_SHIFT);
}

template <typename T>
static void shadowImpl(uint gid,
                       constant Uniforms& uniforms,
                       typename T::structure accelerationStructure,
                       typename T::volume_structure mediumAccelerationStructure,
                       device const char* shadowRays,
                       device float4* radianceOut,
                       device const uint32_t* control,
                       constant uint32_t& sampleIdx,
                       constant MTLIndirectAccelerationStructureInstanceDescriptor* instances,
                       device const Material* materials,
                       device const AlphaMaterialData* alphaMaterials,
                       device const PrimitiveAlphaData* primitiveAlphaData,
                       device const PrimitiveAlphaDecode* primitiveAlphaDecode,
                       device const GeometryEntry* geometryEntries,
                       device const char* vertexBuffer,
                       device const uint32_t* indexBuffer,
                       typename T::table functionTable,
                       device SharcUpdateState* sharcUpdates,
                       device SharcAccumulationEntry* sharcAccumulation,
                       uint32_t directGeometryBase,
                       uint32_t bounce)
{
    if (gid >= control[WF_CTRL_SHADOW_N])
    {
        return;
    }
    const uint32_t capacity = uniforms.width * uniforms.height;
    const CompactShadowTraversal traversal = loadShadowTraversal(shadowRays, gid);
    auditWork(uniforms, WORK_SHADOW_RAYS_BASE + min(bounce, WORK_BOUNCE_SLOTS - 1u));
    // Opaque validation scenes issue exactly one hardware query per shadow ray.
    // Alpha restart walks may issue more and remain classified separately in
    // the static ledger.
    auditWork(uniforms, WORK_INTERSECTION_QUERIES);

    ray shadowRay;
    shadowRay.origin = float3(traversal.origin);
    shadowRay.direction = float3(traversal.direction);
    shadowRay.min_distance = 0.0f;
    shadowRay.max_distance = traversal.maxDistance;

    uint32_t motionPixelIndex = 0u;
    if (SPEC_MOTION_BLUR)
    {
        motionPixelIndex = loadShadowContribution(shadowRays, gid, capacity).pixelIndex;
    }
    const float motionTime = motionTimeFor(uniforms, motionPixelIndex, sampleIdx);
    if (!SPEC_ALPHA)
    {
        // No cutouts in this scene: one any-hit trace, exactly as before.
        typename T::isect isect;
        isect.assume_geometry_type(T::geometryTypes());
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        const bool visible =
            T::trace(isect, shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime, functionTable).type ==
            intersection_type::none;
        if (!visible)
        {
            if (SPEC_RESTIR)
            {
                const CompactShadowContribution contribution = loadShadowContribution(shadowRays, gid, capacity);
                const uint32_t pathIndex = loadShadowPathIndex(shadowRays, gid, contribution.pixelIndex);
                const bool update = bounce == 0u && uniforms.restirFinalVisibilityReuse != 0u &&
                                    (pathIndex & RESTIR_VISIBILITY_UPDATE_BIT) != 0u;
                if (update)
                    restirStoreFinalVisibility(uniforms, contribution.pixelIndex, 0.0f);
            }
            return;
        }
        // The contribution is a separate dense array for plain path tracing.
        // Blocked rays never touch it, and its values do not span traversal.
        const CompactShadowContribution contribution = loadShadowContribution(shadowRays, gid, capacity);
        const uint32_t pathIndex = loadShadowPathIndex(shadowRays, gid, contribution.pixelIndex);
        const bool restirHistoryRay =
            SPEC_RENDER_WORK_AUDIT && bounce == 0u && (pathIndex & RESTIR_AUDIT_HISTORY_BIT) != 0u;
        const bool restirVisibilityUpdate = SPEC_RESTIR && bounce == 0u && uniforms.restirFinalVisibilityReuse != 0u &&
                                            (pathIndex & RESTIR_VISIBILITY_UPDATE_BIT) != 0u;
        if (restirVisibilityUpdate)
            restirStoreFinalVisibility(uniforms, contribution.pixelIndex, 1.0f);
        float3 weight = float3(contribution.weight);
        float3 sharcRadiance = loadShadowSharcRadiance(shadowRays, gid);
        if (SPEC_FOG && uniforms.hasFog)
        {
            const CompactShadowTraversal visibleRay = loadShadowTraversal(shadowRays, gid);
            const float tau = fogOpticalDepth(float3(visibleRay.origin), float3(visibleRay.direction),
                                              visibleRay.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
            const float fogTransmittance = exp(-tau);
            weight *= fogTransmittance;
            sharcRadiance *= fogTransmittance;
        }
        if (SPEC_SSS && uniforms.hasBoundedMedium)
        {
            const CompactShadowTraversal visibleRay = loadShadowTraversal(shadowRays, gid);
            const float3 transmittance =
                mediumTransmittance<T>(mediumAccelerationStructure, uniforms, materials, geometryEntries, instances,
                                       float3(visibleRay.origin), float3(visibleRay.direction), visibleRay.maxDistance,
                                       loadShadowMedium(shadowRays, gid, capacity), motionTime);
            weight *= transmittance;
            sharcRadiance *= transmittance;
        }
        if (bounce == 0u)
        {
            restirDiagnosticVisibility(uniforms, contribution.pixelIndex, weight);
            if (restirHistoryRay)
                auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
        }
        addFilteredRadiance(radianceOut, contribution.pixelIndex, weight);
        if (SPEC_SHARC_UPDATE)
        {
            const uint32_t sharcPathIndex = pathIndex & RESTIR_AUDIT_PATH_INDEX_MASK;
            const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, sharcPathIndex);
            const SharcUpdateState updateState = sharcUpdates[updateIndex];
            sharcPropagate(updateState, sharcAccumulation, sharcRadiance, uniforms,
                           (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
        }
        return;
    }

    float transmittance;
    const float alphaCutoff = SPEC_STOCHASTIC_ALPHA_VISIBILITY ? traversal.alphaThreshold :
                                                                 traversal.alphaThreshold * kShadowTransmittanceCutoff;
    if (!CutoutShadowWalk<T, T::kInlineQuery != 0, T::kHardwareAlpha != 0>::run(
            uniforms, accelerationStructure, shadowRay, motionTime, alphaCutoff, materials, alphaMaterials,
            primitiveAlphaData, primitiveAlphaDecode, geometryEntries, vertexBuffer, indexBuffer, functionTable,
            directGeometryBase, SPEC_PRIMITIVE_ALPHA_DATA, SPEC_COMPACT_ALPHA_MATERIALS, transmittance))
    {
        if (SPEC_RESTIR)
        {
            const CompactShadowContribution contribution = loadShadowContribution(shadowRays, gid, capacity);
            const uint32_t pathIndex = loadShadowPathIndex(shadowRays, gid, contribution.pixelIndex);
            const bool update = bounce == 0u && uniforms.restirFinalVisibilityReuse != 0u &&
                                (pathIndex & RESTIR_VISIBILITY_UPDATE_BIT) != 0u;
            if (update)
                restirStoreFinalVisibility(uniforms, contribution.pixelIndex, 0.0f);
        }
        return; // fully blocked
    }
    // The stochastic path returns binary visibility; its survival probability
    // already carries the alpha product. The compatibility path instead keeps
    // the old 5% Russian-roulette compensation for deep transparent stacks.
    if (SPEC_STOCHASTIC_ALPHA_VISIBILITY)
    {
        // P(roulette < product(1 - opacity)) equals that product, so a surviving
        // ray carries unit visibility. This is unbiased and adds no RNG work.
        transmittance = 1.0f;
    }
    else if (transmittance < kShadowTransmittanceCutoff)
    {
        transmittance = kShadowTransmittanceCutoff;
    }
    // Like the opaque path, defer the contribution payload until after the
    // potentially long cutout walk. Most foliage rays terminate in traversal.
    const CompactShadowContribution contribution = loadShadowContribution(shadowRays, gid, capacity);
    const uint32_t pathIndex = loadShadowPathIndex(shadowRays, gid, contribution.pixelIndex);
    const bool restirHistoryRay = SPEC_RENDER_WORK_AUDIT && bounce == 0u && (pathIndex & RESTIR_AUDIT_HISTORY_BIT) != 0u;
    const bool restirVisibilityUpdate = SPEC_RESTIR && bounce == 0u && uniforms.restirFinalVisibilityReuse != 0u &&
                                        (pathIndex & RESTIR_VISIBILITY_UPDATE_BIT) != 0u;
    if (restirVisibilityUpdate)
        restirStoreFinalVisibility(uniforms, contribution.pixelIndex, transmittance);
    float3 weight = float3(contribution.weight) * transmittance;
    float3 sharcRadiance = loadShadowSharcRadiance(shadowRays, gid) * transmittance;
    if (all(weight <= 1e-6f))
    {
        return;
    }

    if (SPEC_FOG && uniforms.hasFog)
    {
        const CompactShadowTraversal visibleRay = loadShadowTraversal(shadowRays, gid);
        const float tau = fogOpticalDepth(float3(visibleRay.origin), float3(visibleRay.direction),
                                          visibleRay.maxDistance, uniforms.fogHeight, uniforms.fogSigmaT);
        const float fogTransmittance = exp(-tau);
        weight *= fogTransmittance;
        sharcRadiance *= fogTransmittance;
    }
    if (SPEC_SSS && uniforms.hasBoundedMedium)
    {
        const CompactShadowTraversal visibleRay = loadShadowTraversal(shadowRays, gid);
        const float3 mediumTr =
            mediumTransmittance<T>(mediumAccelerationStructure, uniforms, materials, geometryEntries, instances,
                                   float3(visibleRay.origin), float3(visibleRay.direction), visibleRay.maxDistance,
                                   loadShadowMedium(shadowRays, gid, capacity), motionTime);
        weight *= mediumTr;
        sharcRadiance *= mediumTr;
    }
    if (bounce == 0u)
    {
        restirDiagnosticVisibility(uniforms, contribution.pixelIndex, weight);
        if (restirHistoryRay)
            auditWork(uniforms, WORK_RESTIR_FINAL_HISTORY_VISIBLE);
    }
    addFilteredRadiance(radianceOut, contribution.pixelIndex, weight);
    if (SPEC_SHARC_UPDATE)
    {
        const uint32_t sharcPathIndex = pathIndex & RESTIR_AUDIT_PATH_INDEX_MASK;
        const uint32_t updateIndex = sharcUpdateStateIndex(uniforms, sharcPathIndex);
        const SharcUpdateState updateState = sharcUpdates[updateIndex];
        sharcPropagate(updateState, sharcAccumulation, sharcRadiance, uniforms,
                       (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u);
    }
}

// Full reset clears all three resources. Responsive mode uses the same kernel
// with clearPersistent == 0 to clear only the per-frame atomic accumulation.
kernel void sharcClear(uint tid [[thread_position_in_grid]],
                       constant Uniforms& uniforms [[buffer(0)]],
                       device SharcHashEntry* hashEntries [[buffer(1)]],
                       device SharcAccumulationEntry* accumulationEntries [[buffer(2)]],
                       device SharcResolvedEntry* resolvedEntries [[buffer(3)]],
                       device atomic_uint* stats [[buffer(4)]],
                       constant uint32_t& clearPersistent [[buffer(5)]])
{
    if (tid < uniforms.sharcCapacity)
    {
        for (uint32_t channel = 0u; channel < 3u; ++channel)
        {
            atomic_store_explicit(&accumulationEntries[tid].radiance[channel], 0, memory_order_relaxed);
            atomic_store_explicit(&accumulationEntries[tid].direction[channel], 0, memory_order_relaxed);
        }
        atomic_store_explicit(&accumulationEntries[tid].sampleCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&accumulationEntries[tid].diagnosticFlags, 0u, memory_order_relaxed);
        if (clearPersistent != 0u)
        {
            atomic_store_explicit(&hashEntries[tid].key, 0u, memory_order_relaxed);
            SharcResolvedEntry empty = {};
            resolvedEntries[tid] = empty;
        }
    }
    if (tid < SHARC_STAT_COUNT)
    {
        atomic_store_explicit(&stats[tid], 0u, memory_order_relaxed);
    }
}

// The only pass that publishes update data to queries. Running one thread per
// cache entry also owns stale eviction, so no hash mutation races a query.
kernel void sharcResolve(uint tid [[thread_position_in_grid]],
                         constant Uniforms& uniforms [[buffer(0)]],
                         device SharcHashEntry* hashEntries [[buffer(1)]],
                         device SharcAccumulationEntry* accumulationEntries [[buffer(2)]],
                         device SharcResolvedEntry* resolvedEntries [[buffer(3)]],
                         device atomic_uint* stats [[buffer(4)]])
{
    if (tid >= uniforms.sharcCapacity)
    {
        return;
    }
    const uint32_t mainCapacity = sharcMainCapacity(uniforms);
    const bool responsiveEntry = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u && tid >= mainCapacity;
    const bool preserveAccumulation = (uniforms.sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
    const uint32_t key = atomic_load_explicit(&hashEntries[tid].key, memory_order_relaxed);
    if (key == 0u)
    {
        return;
    }

    device SharcAccumulationEntry& accumulation = accumulationEntries[tid];
    const uint32_t entrySampleCount = preserveAccumulation ?
                                          atomic_load_explicit(&accumulation.sampleCount, memory_order_relaxed) :
                                          atomic_exchange_explicit(&accumulation.sampleCount, 0u, memory_order_relaxed);
    const uint32_t diagnosticFlags =
        preserveAccumulation ? atomic_load_explicit(&accumulation.diagnosticFlags, memory_order_relaxed) :
                               atomic_exchange_explicit(&accumulation.diagnosticFlags, 0u, memory_order_relaxed);
    float3 sum = float3(0.0f);
    float3 directionSum = float3(0.0f);
    uint32_t maxRadianceFixed = 0u;
    const bool directional = (uniforms.sharcFlags & SHARC_FLAG_DIRECTIONAL) != 0u;
    const float inverseScale = 1.0f / max(uniforms.sharcRadianceScale, 1.0f);
    for (uint32_t channel = 0u; channel < 3u; ++channel)
    {
        const int32_t radiance = preserveAccumulation ?
                                     atomic_load_explicit(&accumulation.radiance[channel], memory_order_relaxed) :
                                     atomic_exchange_explicit(&accumulation.radiance[channel], 0, memory_order_relaxed);
        const int32_t direction =
            preserveAccumulation ? atomic_load_explicit(&accumulation.direction[channel], memory_order_relaxed) :
                                   atomic_exchange_explicit(&accumulation.direction[channel], 0, memory_order_relaxed);
        sum[channel] = directional ? float(radiance) * inverseScale : float(as_type<uint32_t>(radiance)) * inverseScale;
        directionSum[channel] = float(direction) * inverseScale;
        maxRadianceFixed =
            max(maxRadianceFixed, directional ? sharcSignedMagnitude(radiance) : as_type<uint32_t>(radiance));
        if (directional)
        {
            maxRadianceFixed = max(maxRadianceFixed, sharcSignedMagnitude(direction));
        }
    }
    if (uniforms.sharcDebug != 0u)
    {
        if ((diagnosticFlags & kSharcDiagnosticAccumulationClamp) != 0u)
        {
            atomic_fetch_add_explicit(&stats[SHARC_STAT_ACCUMULATION_CLAMP], 1u, memory_order_relaxed);
        }
        if ((diagnosticFlags & kSharcDiagnosticNonfiniteReject) != 0u)
        {
            atomic_fetch_add_explicit(&stats[SHARC_STAT_NONFINITE_REJECT], 1u, memory_order_relaxed);
        }
        sharcAtomicMax(&stats[SHARC_STAT_MAX_RADIANCE_FIXED], maxRadianceFixed);
        sharcAtomicMax(&stats[SHARC_STAT_MAX_SAMPLE_COUNT], entrySampleCount);
    }

    SharcResolvedEntry previous = resolvedEntries[tid];
    uint32_t accumulatedFrames = (previous.accumulatedFramesAndFlags & ~SHARC_RESOLVED_RESPONSIVE_ENTRY) + 1u;
    uint32_t staleFrames = entrySampleCount != 0u ? 0u : previous.staleFrames + 1u;
    const uint32_t staleLimit =
        responsiveEntry ? max(uniforms.sharcResponsiveFrames, 1u) : clamp(uniforms.sharcStaleFrameCount, 8u, 1024u);
    if (staleFrames >= staleLimit)
    {
        SharcResolvedEntry empty = {};
        resolvedEntries[tid] = empty;
        // Resolve is the sole hash mutator in this phase; update and query are
        // separated by barriers, so eviction needs no fallible weak CAS.
        atomic_store_explicit(&hashEntries[tid].key, 0u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[SHARC_STAT_EVICTION], 1u, memory_order_relaxed);
        return;
    }

    const uint32_t fadeBit = 1u << (uniforms.sharcFrameIndex & 31u);
    uint32_t count = entrySampleCount;
    if (responsiveEntry)
    {
        uint32_t mainIndex = SHARC_NO_ENTRY;
        SharcAddress mainAddress;
        mainAddress.key = key;
        mainAddress.hash = sharcHash(key);
        mainAddress.voxelSize = 0.0f;
        mainAddress.levelBlend = 0.0f;
        mainAddress.level = 0;
        if (sharcFindEntry(uniforms, hashEntries, mainAddress, false, false, nullptr, mainIndex))
        {
            count = atomic_load_explicit(&accumulationEntries[mainIndex].sampleCount, memory_order_relaxed);
        }
    }

    if (count == 0u)
    {
        previous.accumulatedFramesAndFlags =
            min(accumulatedFrames, 1024u) | (responsiveEntry ? SHARC_RESOLVED_RESPONSIVE_ENTRY : 0u);
        previous.staleFrames = staleFrames;
        if ((uniforms.sharcFlags & SHARC_FLAG_FADE_ACCELERATION) != 0u && !responsiveEntry)
        {
            previous.fadeMask |= fadeBit;
        }
        resolvedEntries[tid] = previous;
        return;
    }

    float4 currentRadiance;
    float4 currentDirection = float4(0.0f);
    if (directional)
    {
        currentRadiance = float4(sharcResolveDirection(sum / float(count)), directionSum.x / float(count));
        currentDirection.xy = directionSum.yz / float(count);
    }
    else
    {
        currentRadiance = float4(max(sum / float(count), float3(0.0f)), 0.0f);
    }
    if (previous.sampleCount == 0u)
    {
        const uint32_t regionEnd = responsiveEntry ? uniforms.sharcCapacity : mainCapacity;
        const uint32_t searchEnd = min(tid + 1u + kSharcLinearProbeWindow, regionEnd);
        for (uint32_t i = tid + 1u; i < searchEnd; ++i)
        {
            if (atomic_load_explicit(&hashEntries[i].key, memory_order_relaxed) != key)
            {
                continue;
            }
            const SharcResolvedEntry older = resolvedEntries[i];
            previous.radiance = older.radiance;
            previous.direction = older.direction;
            previous.sampleCount = older.sampleCount;
            previous.fadeMask = older.fadeMask;
            accumulatedFrames = (older.accumulatedFramesAndFlags & ~SHARC_RESOLVED_RESPONSIVE_ENTRY) + 1u;
            staleFrames = 0u;
            break;
        }
    }

    float previousCount = float(previous.sampleCount);
    const uint32_t historyLimit =
        clamp(responsiveEntry ? uniforms.sharcResponsiveFrames : uniforms.sharcAccumulationFrames, 1u, 1024u);
    if (accumulatedFrames > historyLimit)
    {
        previousCount *= float(historyLimit) / float(accumulatedFrames);
        accumulatedFrames = historyLimit;
    }

    if ((uniforms.sharcFlags & SHARC_FLAG_FADE_ACCELERATION) != 0u && !responsiveEntry)
    {
        const float currentLuminance = directional ? max(currentRadiance.w, 0.0f) : sharcLuminance(currentRadiance.xyz);
        const float previousLuminance =
            directional ? max(float(previous.radiance.w), 0.0f) : sharcLuminance(float3(previous.radiance.xyz));
        const bool fading = currentLuminance < previousLuminance;
        previous.fadeMask = (previous.fadeMask & ~fadeBit) | (fading ? fadeBit : 0u);
        if (popcount(previous.fadeMask) == 32u)
        {
            previousCount = float(count);
        }
    }

    float combinedCount = previousCount + float(count);
    const float alpha = float(count) / max(combinedCount, 1.0f);
    float4 resolvedRadiance = mix(float4(previous.radiance), currentRadiance, alpha);
    float4 resolvedDirection = mix(float4(previous.direction), currentDirection, alpha);

    const float3 cameraDelta = uniforms.viewToWorld[3].xyz - float3(uniforms.sharcCameraPrev);
    if ((uniforms.sharcFlags & SHARC_FLAG_BLEND_ADJACENT_LEVELS) != 0u && !responsiveEntry &&
        dot(cameraDelta, cameraDelta) > 1e-6f && accumulatedFrames <= 2u)
    {
        const SharcAddress adjacentAddress = sharcAdjacentLevelAddress(uniforms, key);
        uint32_t adjacentIndex = SHARC_NO_ENTRY;
        if (sharcFindEntry(uniforms, hashEntries, adjacentAddress, false, false, nullptr, adjacentIndex))
        {
            const SharcResolvedEntry adjacent = resolvedEntries[adjacentIndex];
            if (adjacent.sampleCount != 0u)
            {
                const float adjacentCount = float(adjacent.sampleCount);
                const float adjacentWeight = adjacentCount / (combinedCount + adjacentCount);
                resolvedRadiance = mix(resolvedRadiance, float4(adjacent.radiance), adjacentWeight);
                resolvedDirection = mix(resolvedDirection, float4(adjacent.direction), adjacentWeight);
                combinedCount += adjacentCount;
            }
        }
    }

    previous.radiance = half4(clamp(resolvedRadiance, float4(-65504.0f), float4(65504.0f)));
    previous.direction = half4(clamp(resolvedDirection, float4(-65504.0f), float4(65504.0f)));
    previous.sampleCount = uint32_t(min(round(combinedCount), 4294960000.0f));
    previous.accumulatedFramesAndFlags = accumulatedFrames | (responsiveEntry ? SHARC_RESOLVED_RESPONSIVE_ENTRY : 0u);
    previous.staleFrames = staleFrames;
    resolvedEntries[tid] = previous;
}

#define WF_SHADOW_ENTRY(NAME, TRAITS)                                                                                  \
    kernel void NAME(                                                                                                  \
        uint gid [[thread_position_in_grid]], constant Uniforms& uniforms [[buffer(0)]],                               \
        TRAITS::structure accelerationStructure [[buffer(1)]], device const char* shadowRays [[buffer(2)]],            \
        device float4* radianceOut [[buffer(3)]], device const uint32_t* control [[buffer(4)]],                        \
        constant uint32_t& sampleIdx [[buffer(5)]],                                                                    \
        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(6)]],                          \
        device const Material* materials [[buffer(7)]], device const GeometryEntry* geometryEntries [[buffer(8)]],     \
        device const char* vertexBuffer [[buffer(9)]], device const uint32_t* indexBuffer [[buffer(10)]],              \
        device const UniformLight* lights [[buffer(11)]], constant uint32_t& queueOffset [[buffer(12)]],               \
        device SharcUpdateState* sharcUpdates [[buffer(13)]],                                                          \
        device SharcAccumulationEntry* sharcAccumulation [[buffer(14)]], TRAITS::table functionTable [[buffer(15)]],   \
        constant uint32_t& bounce [[buffer(16)]], TRAITS::volume_structure mediumAccelerationStructure [[buffer(18)]], \
        device const PrimitiveAlphaData* primitiveAlphaData [[buffer(19)]],                                            \
        device const AlphaMaterialData* alphaMaterials [[buffer(20)]],                                                 \
        device const PrimitiveAlphaDecode* primitiveAlphaDecode [[buffer(21)]])                                        \
    {                                                                                                                  \
        shadowImpl<TRAITS>(gid + queueOffset, uniforms, accelerationStructure, mediumAccelerationStructure,            \
                           shadowRays, radianceOut, control, sampleIdx, instances, materials, alphaMaterials,          \
                           primitiveAlphaData, primitiveAlphaDecode, geometryEntries, vertexBuffer, indexBuffer,       \
                           functionTable, sharcUpdates, sharcAccumulation, 0u, bounce);                                \
    }

WF_SHADOW_ENTRY(wavefrontShadow, MotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStatic, StaticTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStaticIft, StaticAlphaIftTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStaticIftCurve, CurveStaticAlphaIftTraversal)
WF_SHADOW_ENTRY(wavefrontShadowCurve, CurveMotionTraversal)
WF_SHADOW_ENTRY(wavefrontShadowStaticCurve, CurveStaticTraversal)

// Immutable world-space geometry does not need a TLAS traversal, instance
// transform, or intersection-function table. The volume structure remains a
// separate argument because bounded-media crossings are an independent walk.
kernel void wavefrontShadowDirectStatic(uint gid [[thread_position_in_grid]],
                                        constant Uniforms& uniforms [[buffer(0)]],
                                        DirectStaticTraversal::structure accelerationStructure [[buffer(1)]],
                                        device const char* shadowRays [[buffer(2)]],
                                        device float4* radianceOut [[buffer(3)]],
                                        device const uint32_t* control [[buffer(4)]],
                                        constant uint32_t& sampleIdx [[buffer(5)]],
                                        constant MTLIndirectAccelerationStructureInstanceDescriptor* instances
                                        [[buffer(6)]],
                                        device const Material* materials [[buffer(7)]],
                                        device const GeometryEntry* geometryEntries [[buffer(8)]],
                                        device const char* vertexBuffer [[buffer(9)]],
                                        device const uint32_t* indexBuffer [[buffer(10)]],
                                        device const UniformLight* lights [[buffer(11)]],
                                        constant uint32_t& queueOffset [[buffer(12)]],
                                        device SharcUpdateState* sharcUpdates [[buffer(13)]],
                                        device SharcAccumulationEntry* sharcAccumulation [[buffer(14)]],
                                        constant uint32_t& bounce [[buffer(16)]],
                                        constant uint32_t& directGeometryBase [[buffer(17)]],
                                        DirectStaticTraversal::volume_structure mediumAccelerationStructure
                                        [[buffer(18)]],
                                        device const PrimitiveAlphaData* primitiveAlphaData [[buffer(19)]],
                                        device const AlphaMaterialData* alphaMaterials [[buffer(20)]],
                                        device const PrimitiveAlphaDecode* primitiveAlphaDecode [[buffer(21)]])
{
    shadowImpl<DirectStaticTraversal>(gid + queueOffset, uniforms, accelerationStructure, mediumAccelerationStructure,
                                      shadowRays, radianceOut, control, sampleIdx, instances, materials, alphaMaterials,
                                      primitiveAlphaData, primitiveAlphaDecode, geometryEntries, vertexBuffer,
                                      indexBuffer, 0u, sharcUpdates, sharcAccumulation, directGeometryBase, bounce);
}

kernel void wavefrontResolve(uint tid [[thread_position_in_grid]],
                             constant Uniforms& uniforms [[buffer(0)]],
                             device const float4* radianceIn [[buffer(1)]],
                             device float4* res [[buffer(2)]],
                             device float4* accum [[buffer(3)]],
                             constant uint32_t& sampleCount [[buffer(4)]],
                             device const AovSample* aov [[buffer(5)]],
                             device SharcHashEntry* sharcHashEntries [[buffer(6)]],
                             device const SharcResolvedEntry* sharcResolved [[buffer(7)]])
{
    const uint32_t pixelCount = uniforms.width * uniforms.height;
    if (tid >= pixelCount)
    {
        return;
    }

    // Guide views. A denoiser fed a broken guide degrades quietly, so the guides
    // have to be inspectable on their own.
    const uint32_t debugMode = uniforms.debug;
    if (uniforms.sharcCapacity != 0u && debugMode == (uint32_t)DebugMode::eSharcOccupancy)
    {
        res[tid] = float4(sharcDebugOccupancy(tid, uniforms, sharcHashEntries, sharcResolved), 1.0f);
        return;
    }
    if (debugMode == (uint32_t)DebugMode::eSharcBounces)
    {
        res[tid] = float4(sharcDebugBounceColor((uint32_t)max(aov[tid].guideStateOrBounceDepth, 0.0f)), 1.0f);
        return;
    }
    if (DEBUG_MODE_IS_AOV(debugMode))
    {
        const AovSample a = aov[tid];
        float3 v = float3(0.0f);
        switch ((DebugMode)debugMode)
        {
        case DebugMode::eAovDiffuseAlbedo:
            v = float3(a.diffuseAlbedo);
            break;
        case DebugMode::eAovSpecularAlbedo:
            v = float3(a.specularAlbedo);
            break;
        case DebugMode::eAovNormal:
            v = float3(a.normal) * 0.5f + 0.5f;
            break;
        case DebugMode::eAovRoughness:
            v = float3(a.roughness);
            break;
        // d/(1+d): monotonic and scale-free, so a scene of any size is readable
        // and nothing crosses zero the way a logarithm does at d == 1.
        case DebugMode::eAovDepth:
            v = float3(a.depth / (1.0f + a.depth));
            break;
        // Red/green for the two axes, scaled so a few pixels of motion is visible.
        case DebugMode::eAovMotion:
            v = float3(a.motionX, a.motionY, 0.0f) * 0.05f + 0.5f;
            break;
        // Red where the denoiser is being told to distrust its history, so the
        // extent of the mask is a thing you can look at rather than infer.
        case DebugMode::eAovReactive:
            v = float3(a.reactive, 0.0f, 0.0f);
            break;
        // d/(1+d) again: scale-free, and zero stays zero.
        case DebugMode::eAovSpecularHitDistance:
            v = float3(a.specularHitDistance / (1.0f + a.specularHitDistance));
            break;
        default:
            break;
        }
        res[tid] = float4(v, 1.0f);
        return;
    }

    const uint reconstructionFilter =
        (uniforms.textureLodMode & RECONSTRUCTION_FILTER_MASK) >> RECONSTRUCTION_FILTER_SHIFT;
    const bool signedReconstruction = reconstructionFilter == RECONSTRUCTION_FILTER_MITCHELL ||
                                      reconstructionFilter == RECONSTRUCTION_FILTER_LANCZOS2;
    const float reconstructionWeight = signedReconstruction ? radianceIn[tid].w : 1.0f;
    float3 result = radianceIn[tid].xyz * reconstructionWeight / (float)max(sampleCount, 1u);

    if (uniforms.enableAccumulation)
    {
        float3 accumColor = result;
        if (uniforms.subframeIndex > 0)
        {
            // subframeIndex counts samples already folded in; this launch adds
            // sampleCount more, so the new mean carries weight m / (n + m).
            const float a = (float)sampleCount / (float)(uniforms.subframeIndex + sampleCount);
            accumColor = mix(float3(accum[tid]), accumColor, a);
        }
        accum[tid] = float4(accumColor, 1.0f);
        result = accumColor;
    }

    res[tid] = float4(result, 1.0f);
}

kernel void wavefrontAovResolve(uint2 tid [[thread_position_in_grid]],
                                constant Uniforms& uniforms [[buffer(0)]],
                                device const AovSample* aov [[buffer(1)]],
                                device const float4* radiance [[buffer(2)]],
                                constant uint32_t& sampleCount [[buffer(3)]],
                                texture2d<float, access::write> colorTex [[texture(0)]],
                                texture2d<float, access::write> depthTex [[texture(1)]],
                                texture2d<float, access::write> motionTex [[texture(2)]],
                                texture2d<float, access::write> diffuseTex [[texture(3)]],
                                texture2d<float, access::write> specularTex [[texture(4)]],
                                texture2d<float, access::write> normalTex [[texture(5)]],
                                texture2d<float, access::write> roughTex [[texture(6)]],
                                texture2d<float, access::write> specHitTex [[texture(7)]],
                                texture2d<float, access::write> reactiveTex [[texture(8)]],
                                texture2d<float, access::write> denoiseStrengthTex [[texture(9)]])
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
    {
        return;
    }
    const uint32_t i = tid.y * uniforms.width + tid.x;
    const AovSample a = aov[i];

    const uint reconstructionFilter =
        (uniforms.textureLodMode & RECONSTRUCTION_FILTER_MASK) >> RECONSTRUCTION_FILTER_SHIFT;
    const bool signedReconstruction = reconstructionFilter == RECONSTRUCTION_FILTER_MITCHELL ||
                                      reconstructionFilter == RECONSTRUCTION_FILTER_LANCZOS2;
    const float reconstructionWeight = signedReconstruction ? radiance[i].w : 1.0f;
    float3 color = radiance[i].xyz * reconstructionWeight / (float)max(sampleCount, 1u);

    if (uniforms.denoiseFireflyClamp > 0.0f)
    {
        const float3 exposed = color * float3(uniforms.exposureValue);
        const float lum = dot(exposed, float3(0.2126f, 0.7152f, 0.0722f));
        if (lum > uniforms.denoiseFireflyClamp)
        {
            color *= uniforms.denoiseFireflyClamp / lum;
        }
    }
    colorTex.write(float4(color, 1.0f), tid);
    depthTex.write(float4(a.depth, 0.0f, 0.0f, 0.0f), tid);
    motionTex.write(float4(a.motionX, a.motionY, 0.0f, 0.0f), tid);
    diffuseTex.write(float4(float3(a.diffuseAlbedo), 1.0f), tid);
    specularTex.write(float4(float3(a.specularAlbedo), 1.0f), tid);
    normalTex.write(float4(float3(a.normal), 0.0f), tid);
    roughTex.write(float4(a.roughness, 0.0f, 0.0f, 0.0f), tid);
    // The texture is R16Float. Environment reflections use a large finite proxy
    // for infinity; clamp it before conversion so the guide stays finite rather
    // than becoming +inf and poisoning the network input.
    specHitTex.write(float4(clamp(a.specularHitDistance, 0.0f, 65504.0f), 0.0f, 0.0f, 0.0f), tid);
    reactiveTex.write(float4(saturate(a.reactive), 0.0f, 0.0f, 0.0f), tid);
    const float denoiseStrength = a.guideStateOrBounceDepth < 0.0f ? 1.0f : 0.0f;
    denoiseStrengthTex.write(float4(denoiseStrength, 0.0f, 0.0f, 0.0f), tid);
}
