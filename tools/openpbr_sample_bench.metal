#include <metal_stdlib>

using namespace metal;

// Keep the benchmark on the same generic OpenPBR path as wavefrontShadeTail.
// The material preset is a runtime uniform, so the compiler cannot remove
// inactive lobes merely because one benchmark case happens to set their weight
// to zero.
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) true
#include <strelka/material/openpbr/openpbr_bridge.h>
#undef OPENPBR_GET_SPECIALIZATION_CONSTANT

struct OpenPbrSampleBenchParams
{
    uint preset;
    uint iterations;
    uint seed;
    uint padding;
};

static inline uint benchNext(thread uint& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

static inline float benchRandom(thread uint& state)
{
    return float(benchNext(state) >> 8) * (1.0f / 16777216.0f);
}

static inline OpenPBR_ResolvedInputs benchInputs(uint preset)
{
    OpenPBR_ResolvedInputs p = openpbr_make_default_resolved_inputs();
    p.base_color = vec3(0.7f, 0.45f, 0.2f);
    p.specular_roughness = 0.25f;

    switch (preset)
    {
    case 0u: // Diffuse only.
        p.specular_weight = 0.0f;
        p.base_diffuse_roughness = 0.5f;
        break;
    case 1u: // Dielectric reflection only.
        p.base_color = vec3(0.0f);
        break;
    case 2u: // Metallic reflection.
        p.base_metalness = 1.0f;
        break;
    case 3u: // Thick subsurface boundary, as used by marble/plants.
        p.subsurface_weight = 1.0f;
        p.subsurface_color = vec3(0.55f, 0.75f, 0.45f);
        break;
    case 4u: // Rough thick glass/water.
        p.transmission_weight = 1.0f;
        p.transmission_color = vec3(0.92f, 0.97f, 1.0f);
        p.transmission_depth = 1.0f;
        p.specular_ior = 1.33f;
        p.specular_roughness = 0.12f;
        break;
    case 5u: // Thin-walled glass.
        p.transmission_weight = 1.0f;
        p.geometry_thin_walled = true;
        p.specular_roughness = 0.12f;
        break;
    case 6u: // Coat over a base layer.
        p.coat_weight = 1.0f;
        p.coat_roughness = 0.2f;
        break;
    case 7u: // Fuzz over a base layer.
        p.fuzz_weight = 1.0f;
        p.fuzz_color = vec3(0.9f, 0.35f, 0.2f);
        p.fuzz_roughness = 0.5f;
        break;
    case 8u: // Iridescent coat.
        p.coat_weight = 1.0f;
        p.coat_roughness = 0.15f;
        p.thin_film_weight = 1.0f;
        p.thin_film_thickness = 0.45f;
        break;
    case 9u: // iso_bathroom/S_Bubbles_Mtl.
        p.coat_weight = 1.0f;
        p.coat_roughness = 0.0f;
        p.coat_ior = 1.5f;
        p.geometry_thin_walled = true;
        p.specular_ior = 1.6f;
        p.specular_roughness = 0.0f;
        p.transmission_weight = 1.0f;
        p.thin_film_weight = 1.0f;
        p.thin_film_thickness = 400.0f;
        p.thin_film_ior = 1.4f;
        break;
    case 10u: // iso_bathroom/S_Bathtub_Water_Mtl.
        p.base_color = vec3(1.0f);
        p.specular_ior = 1.33f;
        p.specular_roughness = 0.0f;
        p.transmission_weight = 1.0f;
        p.transmission_color = vec3(0.547863f, 0.918096f, 1.0f);
        p.transmission_depth = 0.1f;
        break;
    default: // Deliberately busy material.
        p.base_metalness = 0.25f;
        p.transmission_weight = 0.35f;
        p.coat_weight = 0.7f;
        p.coat_roughness = 0.18f;
        p.fuzz_weight = 0.4f;
        p.thin_film_weight = 0.6f;
        p.thin_film_thickness = 0.4f;
        break;
    }
    return p;
}

static inline OpenPBRParams benchBaseParams(uint preset)
{
    const OpenPBR_ResolvedInputs inputs = benchInputs(preset);
    OpenPBRParams p = {};
    p.base_color = OpenPBRColor{ inputs.base_color.x, inputs.base_color.y, inputs.base_color.z };
    p.base_weight = inputs.base_weight;
    p.base_diffuse_roughness = inputs.base_diffuse_roughness;
    p.base_metalness = inputs.base_metalness;
    p.specular_weight = inputs.specular_weight;
    p.specular_roughness = inputs.specular_roughness;
    p.specular_color = OpenPBRColor{ inputs.specular_color.x, inputs.specular_color.y, inputs.specular_color.z };
    p.specular_roughness_anisotropy = inputs.specular_roughness_anisotropy;
    p.specular_ior = inputs.specular_ior;
    p.specular_anisotropy_rotation_cos = inputs.specular_anisotropy_rotation_cos_sin.x;
    p.specular_anisotropy_rotation_sin = inputs.specular_anisotropy_rotation_cos_sin.y;
    p.geometry_thin_walled = inputs.geometry_thin_walled;
    return p;
}

static inline OpenPBR_BaseParams benchCompactBaseParams(uint preset)
{
    const OpenPBRParams p = benchBaseParams(preset);
    OpenPBR_BaseParams base;
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
    return base;
}

static inline SurfaceInteraction benchSurface()
{
    SurfaceInteraction si = {};
    si.shading_normal = float3(0.0f, 0.0f, 1.0f);
    si.tangent = float3(1.0f, 0.0f, 0.0f);
    si.bitangent = float3(0.0f, 1.0f, 0.0f);
    si.wo = normalize(float3(0.35f, 0.1f, 1.0f));
    si.exterior_ior = 1.0f;
    return si;
}

static inline float4 benchBaseSample(const thread OpenPBR_BasePreparedBsdf& prepared, uint seed)
{
    uint rng = seed;
    const float4 xi = float4(benchRandom(rng), benchRandom(rng), benchRandom(rng), benchRandom(rng));
    const BsdfSampleResult sample = openpbr_bsdf_sample(prepared, benchSurface().wo, xi);
    return sample.pdf > 0.0f ? float4(sample.wi + sample.bsdf_over_pdf, sample.pdf) : 0.0f;
}

static inline float4 benchFullSample(const thread OpenPBR_PreparedBsdf& prepared, uint seed)
{
    uint rng = seed;
    const float4 xi = float4(benchRandom(rng), benchRandom(rng), benchRandom(rng), benchRandom(rng));
    const BsdfSampleResult sample = openpbr_bsdf_sample(prepared, xi);
    return sample.pdf > 0.0f ? float4(sample.wi + sample.bsdf_over_pdf, sample.pdf) : 0.0f;
}

static inline float4 benchBaseEval(const thread OpenPBR_BasePreparedBsdf& prepared, const thread SurfaceInteraction& si)
{
    const float3 wi = normalize(float3(-0.2f, 0.45f, 1.0f));
    const BsdfEvalResult eval = openpbr_bsdf_eval(prepared, si, wi);
    return float4(eval.bsdf, eval.pdf);
}

// Keep the same return-value and deferred-shadow payload shape as the
// production NEE path without pulling the scene-dependent light samplers into
// this first lifetime experiment.
struct BenchLightConnection
{
    float3 radiance;
    float3 toLight;
    float3 origin;
    float3 visibilityTarget;
    float pdf;
    float tMax;
    bool needsRay;
    bool hasVisibilityTarget;
    bool isDelta;
    uint4 sample;
};

struct BenchLightConnectionEvaluation
{
    float3 integrand;
    float target;
};

struct BenchCompactShadowRay
{
    packed_float3 origin;
    packed_float3 direction;
    packed_float3 weight;
    float maxDistance;
    uint pixelIndex;
    float rrCutoff;
    uint medium;
};

static_assert(sizeof(BenchCompactShadowRay) == 52, "Production compact shadow-ray layout changed");

static inline BenchLightConnection benchLightConnection(const thread SurfaceInteraction& si, uint gid, uint seed)
{
    const float u = float((gid ^ seed) & 1023u) * (1.0f / 1024.0f);
    BenchLightConnection connection;
    connection.toLight = normalize(float3(u - 0.45f, 0.35f - 0.25f * u, 0.7f + 0.2f * u));
    connection.radiance = float3(1.2f + u, 0.8f + 0.5f * u, 0.6f + 0.25f * u);
    connection.origin = si.position + si.geometry_normal * (1.0e-4f + u * 1.0e-5f);
    connection.visibilityTarget = connection.origin + connection.toLight * (2.0f + u);
    connection.pdf = 0.15f + 0.7f * u;
    connection.tMax = 2.0f + u;
    connection.needsRay = true;
    connection.hasVisibilityTarget = (gid & 1u) != 0u;
    connection.isDelta = (gid & 7u) == 0u;
    connection.sample = uint4(gid & 0x3fffffffu, as_type<uint>(u), seed, gid);
    return connection;
}

static inline BenchLightConnectionEvaluation benchEvaluateLightConnection(const thread BenchLightConnection& connection,
                                                                          const thread SurfaceInteraction& si,
                                                                          const thread OpenPBR_BasePreparedBsdf& prepared,
                                                                          uint misHeuristic)
{
    BenchLightConnectionEvaluation result = {};
    if (!connection.needsRay || !(connection.pdf > 0.0f) || dot(connection.toLight, si.shading_normal) <= 0.0f)
    {
        return result;
    }
    const BsdfEvalResult evaluated = openpbr_bsdf_eval(prepared, si, connection.toLight);
    if (!(evaluated.pdf > 0.0f))
    {
        return result;
    }
    const float ratio = evaluated.pdf / connection.pdf;
    const float misWeight = connection.isDelta ? 1.0f :
                            misHeuristic == 1u ? 1.0f / (1.0f + ratio * ratio) :
                                                 1.0f / (1.0f + ratio);
    result.integrand = connection.radiance * evaluated.bsdf * misWeight;
    result.target = dot(result.integrand, float3(0.2126f, 0.7152f, 0.0722f));
    return result;
}

static inline float4 benchBaseNeePayload(const thread OpenPBR_BasePreparedBsdf& prepared,
                                         const thread SurfaceInteraction& si,
                                         uint gid,
                                         uint seed,
                                         device BenchCompactShadowRay* shadowOutput)
{
    const BenchLightConnection connection = benchLightConnection(si, gid, seed);
    const BenchLightConnectionEvaluation evaluated = benchEvaluateLightConnection(connection, si, prepared, gid & 1u);
    const float normalization = evaluated.target > 0.0f ? (evaluated.target / connection.pdf) / evaluated.target : 0.0f;
    const float3 endpoint = connection.hasVisibilityTarget ? connection.visibilityTarget :
                                                             connection.origin + connection.toLight * connection.tMax;
    const float3 delta = endpoint - connection.origin;
    const float distance = length(delta);

    BenchCompactShadowRay shadow;
    shadow.origin = packed_float3(connection.origin);
    shadow.direction = packed_float3(delta / distance);
    shadow.weight = packed_float3(evaluated.integrand * normalization);
    shadow.maxDistance = distance;
    shadow.pixelIndex = gid;
    shadow.rrCutoff = float((connection.sample.y ^ seed) >> 8) * (1.0f / 16777216.0f);
    shadow.medium = connection.sample.x & 15u;
    shadowOutput[gid] = shadow;
    return float4(evaluated.integrand, evaluated.target);
}

kernel void openpbrPrepareBaseReference(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                        device float4* output [[buffer(1)]],
                                        uint gid [[thread_position_in_grid]])
{
    const OpenPBRParams p = benchBaseParams(params.preset);
    const SurfaceInteraction si = benchSurface();
    const OpenPBR_PreparedBsdf full = openpbr_prepare_surface_at(p, si, float3(1.0f));
    output[gid] = benchFullSample(full, params.seed ^ (gid * 0x9e3779b9u));
}

kernel void openpbrPrepareBaseDirect(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                     device float4* output [[buffer(1)]],
                                     uint gid [[thread_position_in_grid]])
{
    const OpenPBR_BaseParams p = benchCompactBaseParams(params.preset);
    const SurfaceInteraction si = benchSurface();
    const OpenPBR_BasePreparedBsdf prepared = openpbr_prepare_base_at(p, si, float3(1.0f));
    output[gid] = benchBaseSample(prepared, params.seed ^ (gid * 0x9e3779b9u));
}

kernel void openpbrPrepareBaseEval(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                   device float4* output [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]])
{
    const OpenPBR_BaseParams p = benchCompactBaseParams(params.preset);
    const SurfaceInteraction si = benchSurface();
    const OpenPBR_BasePreparedBsdf prepared = openpbr_prepare_base_at(p, si, float3(1.0f));
    output[gid] = benchBaseEval(prepared, si);
}

kernel void openpbrPrepareBaseEvalSample(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                         device float4* output [[buffer(1)]],
                                         uint gid [[thread_position_in_grid]])
{
    const OpenPBR_BaseParams p = benchCompactBaseParams(params.preset);
    const SurfaceInteraction si = benchSurface();
    const OpenPBR_BasePreparedBsdf prepared = openpbr_prepare_base_at(p, si, float3(1.0f));
    const float4 evaluated = benchBaseEval(prepared, si);
    const float4 sampled = benchBaseSample(prepared, params.seed ^ (gid * 0x9e3779b9u));
    output[gid] = evaluated + sampled;
}

kernel void openpbrPrepareBaseNeePayload(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                         device float4* output [[buffer(1)]],
                                         device BenchCompactShadowRay* shadowOutput [[buffer(2)]],
                                         uint gid [[thread_position_in_grid]])
{
    const OpenPBR_BaseParams p = benchCompactBaseParams(params.preset);
    SurfaceInteraction si = benchSurface();
    si.position = float3(float(gid & 255u) * 0.01f, float((gid >> 8u) & 255u) * 0.01f, 0.0f);
    si.geometry_normal = si.shading_normal;
    const OpenPBR_BasePreparedBsdf prepared = openpbr_prepare_base_at(p, si, float3(1.0f));
    output[gid] = benchBaseNeePayload(prepared, si, gid, params.seed, shadowOutput);
}

kernel void openpbrPrepareBaseNeePayloadSample(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                               device float4* output [[buffer(1)]],
                                               device BenchCompactShadowRay* shadowOutput [[buffer(2)]],
                                               uint gid [[thread_position_in_grid]])
{
    const OpenPBR_BaseParams p = benchCompactBaseParams(params.preset);
    SurfaceInteraction si = benchSurface();
    si.position = float3(float(gid & 255u) * 0.01f, float((gid >> 8u) & 255u) * 0.01f, 0.0f);
    si.geometry_normal = si.shading_normal;
    const OpenPBR_BasePreparedBsdf prepared = openpbr_prepare_base_at(p, si, float3(1.0f));
    const float4 nee = benchBaseNeePayload(prepared, si, gid, params.seed, shadowOutput);
    const float4 sampled = benchBaseSample(prepared, params.seed ^ (gid * 0x9e3779b9u));
    output[gid] = nee + sampled;
}

kernel void openpbrSampleBench(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                               device float4* output [[buffer(1)]],
                               uint gid [[thread_position_in_grid]])
{
    const OpenPBR_ResolvedInputs inputs = benchInputs(params.preset);
    OpenPBR_VolumeDerivedProps volumeDerived;
    OpenPBR_PreparedBsdf prepared;
    openpbr_prepare_volume(inputs, volumeDerived, prepared, false);
    const vec3 wo = normalize(vec3(0.35f, 0.1f, 1.0f));
    openpbr_prepare_lobes(inputs, volumeDerived, prepared, vec3(1.0f), vec3(600.0f, 550.0f, 450.0f), 1.0f, wo);

    uint rng = params.seed ^ (gid * 0x9e3779b9u);
    float3 sum = 0.0f;
    float pdfSum = 0.0f;
    uint typeSum = 0u;
#pragma clang loop unroll(disable)
    for (uint i = 0u; i < params.iterations; ++i)
    {
        const vec3 xi = vec3(benchRandom(rng), benchRandom(rng), benchRandom(rng));
        vec3 wi;
        OpenPBR_DiffuseSpecular weight;
        float pdf;
        OpenPBR_BsdfLobeType type;
        openpbr_sample(prepared, xi, wi, weight, pdf, type);
        if (pdf > 0.0f)
        {
            sum += float3(wi) + float3(weight.diffuse) + float3(weight.specular);
            pdfSum += pdf;
            typeSum += type;
        }
    }
    output[gid] = float4(sum + float(typeSum & 255u) * 1.0e-6f, pdfSum);
}

kernel void openpbrSampleBenchRng(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                  device float4* output [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]])
{
    uint rng = params.seed ^ (gid * 0x9e3779b9u);
    float4 sum = 0.0f;
#pragma clang loop unroll(disable)
    for (uint i = 0u; i < params.iterations; ++i)
    {
        sum += float4(benchRandom(rng), benchRandom(rng), benchRandom(rng), 1.0f);
    }
    output[gid] = sum;
}

// The generic benchmark above deliberately preserves the production Tail
// shader's runtime material choice. These entry points answer a different
// question: what register allocation and spills remain when the compiler knows
// the exact material configuration and there is no loop-carried state?
template <uint Preset>
static inline void openpbrSampleOnceImpl(constant OpenPbrSampleBenchParams& params, device float4* output, uint gid)
{
    const OpenPBR_ResolvedInputs inputs = benchInputs(Preset);
    OpenPBR_VolumeDerivedProps volumeDerived;
    OpenPBR_PreparedBsdf prepared;
    openpbr_prepare_volume(inputs, volumeDerived, prepared, false);
    const vec3 wo = normalize(vec3(0.35f, 0.1f, 1.0f));
    openpbr_prepare_lobes(inputs, volumeDerived, prepared, vec3(1.0f), vec3(600.0f, 550.0f, 450.0f), 1.0f, wo);

    uint rng = params.seed ^ (gid * 0x9e3779b9u);
    const vec3 xi = vec3(benchRandom(rng), benchRandom(rng), benchRandom(rng));
    vec3 wi;
    OpenPBR_DiffuseSpecular weight;
    float pdf;
    OpenPBR_BsdfLobeType type;
    openpbr_sample(prepared, xi, wi, weight, pdf, type);
    output[gid] =
        pdf > 0.0f ?
            float4(float3(wi) + float3(weight.diffuse) + float3(weight.specular) + float(type & 255u) * 1.0e-6f, pdf) :
            0.0f;
}

#define OPENPBR_SAMPLE_ONCE_ENTRY(Name, Preset)                                                                        \
    kernel void Name(constant OpenPbrSampleBenchParams& params [[buffer(0)]], device float4* output [[buffer(1)]],     \
                     uint gid [[thread_position_in_grid]])                                                             \
    {                                                                                                                  \
        openpbrSampleOnceImpl<Preset>(params, output, gid);                                                            \
    }

OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceDiffuse, 0u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceDielectric, 1u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceMetal, 2u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceThickSss, 3u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceThickGlass, 4u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceThinGlass, 5u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceCoat, 6u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceFuzz, 7u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceThinFilmCoat, 8u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceBathroomBubbles, 9u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceBathroomWater, 10u)
OPENPBR_SAMPLE_ONCE_ENTRY(openpbrSampleOnceMixed, 11u)

#undef OPENPBR_SAMPLE_ONCE_ENTRY

kernel void openpbrSampleOnceRng(constant OpenPbrSampleBenchParams& params [[buffer(0)]],
                                 device float4* output [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]])
{
    uint rng = params.seed ^ (gid * 0x9e3779b9u);
    output[gid] = float4(benchRandom(rng), benchRandom(rng), benchRandom(rng), 1.0f);
}
