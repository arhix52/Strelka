#include <optix.h>

#include <cuda.h>

#include <OptixRenderParams.h>

extern "C"
{
    __constant__ Params params;
}

static __device__ bool samplerBlueNoiseEnabled()
{
    return params.hasBlueNoise != 0u;
}
#define STRELKA_OPENPBR_FEATURE_EnableSheenAndCoat params.openpbrSheenAndCoat
#define STRELKA_OPENPBR_FEATURE_EnableDispersion params.openpbrDispersion
#define STRELKA_OPENPBR_FEATURE_EnableTranslucency params.openpbrTranslucency
#define STRELKA_OPENPBR_FEATURE_EnableMetallic params.openpbrMetallic
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) STRELKA_OPENPBR_FEATURE_##name
#include <cuda_helpers/helpers.h>
#include <cuda_helpers/curve.h>

#include <random.h>

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include <lights.h>
#include <env_light.h>
#include <light_alias_sampling.h>

#include <strelka/material/bsdf.h>
#include <strelka/material/volume.h>

#include <postprocessing/Guides.h>

#include "optix_device_utils.h"

static __forceinline__ __device__ unsigned int optixIorMaterial(const OptixIorStack& stack)
{
    return stack.top >= 0 ? stack.materials[stack.top] : 0xFFFFFFFFu;
}

static __forceinline__ __device__ float optixIorCurrent(const OptixIorStack& stack)
{
    const unsigned int material = optixIorMaterial(stack);
    return material != 0xFFFFFFFFu ? params.materials[material].ior : 1.0f;
}

static __forceinline__ __device__ int optixIorFind(const OptixIorStack& stack,
                                                    unsigned int priority,
                                                    unsigned int material)
{
    for (int i = stack.top; i >= 0; --i)
        if (stack.materials[i] == material)
            return i;
    for (int i = stack.top; i >= 0; --i)
        if ((params.materials[stack.materials[i]].dielectric_priority & 0xFFu) == (priority & 0xFFu))
            return i;
    return -1;
}

static __forceinline__ __device__ void optixIorPush(OptixIorStack& stack, unsigned int material)
{
    if (stack.top < IOR_STACK_SIZE - 1)
        stack.materials[++stack.top] = material & IOR_ENTRY_MATERIAL_MASK;
}

static __forceinline__ __device__ void optixIorPop(OptixIorStack& stack,
                                                   unsigned int priority,
                                                   unsigned int material)
{
    const int found = optixIorFind(stack, priority, material);
    if (found >= 0)
    {
        for (int i = found; i < stack.top; ++i)
            stack.materials[i] = stack.materials[i + 1];
        --stack.top;
    }
}

static __forceinline__ __device__ float optixIorAfterPop(const OptixIorStack& stack,
                                                         unsigned int priority)
{
    int found = -1;
    for (int i = stack.top; i >= 0; --i)
        if ((params.materials[stack.materials[i]].dielectric_priority & 0xFFu) == (priority & 0xFFu))
        {
            found = i;
            break;
        }
    if (found < 0 || found < stack.top)
        return optixIorCurrent(stack);
    return stack.top > 0 ? params.materials[stack.materials[stack.top - 1]].ior : 1.0f;
}
#include "shading/shading_common.h"
#undef OPENPBR_GET_SPECIALIZATION_CONSTANT
#undef STRELKA_OPENPBR_FEATURE_EnableMetallic
#undef STRELKA_OPENPBR_FEATURE_EnableTranslucency
#undef STRELKA_OPENPBR_FEATURE_EnableDispersion
#undef STRELKA_OPENPBR_FEATURE_EnableSheenAndCoat
#include "shading/medium.h"
#include "alpha.h"
#include "fog.h"
#include <curve_layout.h>

static __forceinline__ __device__ bool isOpenPBRMaterial(const MaterialParams& m)
{
    return params.hasOpenPBR && params.openpbrParams != nullptr && m.material_type == MATERIAL_TYPE_OPENPBR;
}

struct MediumProps
{
    float3 sigmaT;
    float3 albedo;
    float anisotropy;
};

static __forceinline__ __device__ MediumProps mediumPropsFor(uint32_t materialIndex, float3 walkAlbedo)
{
    const MaterialParams& mm = params.materials[materialIndex];
    MediumProps out;
    if (isOpenPBRMaterial(mm))
    {
        const OpenPBRParams mat = params.openpbrParams[materialIndex];
        const OpenPBR_HomogeneousVolume v = openpbr_interior_volume(mat);
        out.sigmaT = v.extinction_coefficient;
        out.albedo = walkAlbedo;
        out.anisotropy = v.anisotropy;
        return out;
    }
    out.sigmaT = fromSpectrum(oka::medium::sigmaTFromRadius(toSpectrum(mm.subsurface_radius)));
    // A bounded volume has no entry surface to have textured, so it keeps the
    // material's constant; a subsurface walk takes what the boundary resolved.
    out.albedo = ((mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u) ? mm.diffuse_transmission_color : walkAlbedo;
    out.anisotropy = mm.subsurface_anisotropy;
    return out;
}

/// Whether a mapped base colour is the only thing carrying this material's
/// subsurface detail. Same test as Metal's openpbrBaseMapDetailsSubsurface().
static __forceinline__ __device__ bool openpbrBaseMapDetailsSubsurface(const OpenPBRParams& p)
{
    return p.subsurface_weight > 0.0f && openpbr_has_texture(p, OPENPBR_TEX_BASE_COLOR) &&
           !openpbr_has_texture(p, OPENPBR_TEX_SUBSURFACE_COLOR);
}

struct OpenPBRGuides
{
    float3 diffuse;
    float3 specular;
    float roughness;
};

static __forceinline__ __device__ OpenPBRGuides openpbrDenoiserGuides(const OpenPBR_ResolvedInputs& in,
                                                                      const SurfaceInteraction& si,
                                                                      bool baseMapDetailsSubsurface)
{
    OpenPBRGuides g;
    const float3 baseColor = in.base_color;
    const float3 subsurfaceIn = in.subsurface_color;
    const float3 specularColor = in.specular_color;
    const float3 coatColor = in.coat_color;
    const float3 fuzzColor = in.fuzz_color;

    const float dielectric = 1.0f - in.base_metalness;
    const float opaque = 1.0f - in.transmission_weight;
    const float3 weightedBase = baseColor * in.base_weight;
    // MaterialX exports in the test scenes use a mapped base plus a constant
    // subsurface tint. Preserve the mapped detail in that case; a genuinely
    // mapped subsurface colour remains an independent OpenPBR input.
    const float3 subsurfaceColor = baseMapDetailsSubsurface ? weightedBase * subsurfaceIn : subsurfaceIn;
    const float3 diffuseColor = lerp(weightedBase, subsurfaceColor, in.subsurface_weight);
    g.diffuse = diffuseColor * dielectric * opaque;

    const float cosView = saturate(fabsf(dot(si.shading_normal, si.wo)));
    const float dielectricF0 = f0_from_ior(fmaxf(in.specular_ior, 1.0f)) * in.specular_weight;
    const float3 dielectricSpecular = saturate(specularColor * dielectricF0);
    const float3 metalSpecular = saturate(weightedBase * in.specular_weight);
    const float3 f0 = lerp(dielectricSpecular, metalSpecular, in.base_metalness);
    g.specular = fresnel_schlick_roughness(f0, cosView, in.specular_roughness);

    const float coatFresnel = in.coat_weight * fresnel_schlick_scalar(f0_from_ior(fmaxf(in.coat_ior, 1.0f)), cosView);
    const float3 coated = g.specular + (make_float3(1.0f) - g.specular) * coatColor * coatFresnel;
    const float3 fuzz = fuzzColor * in.fuzz_weight * powf(1.0f - cosView, 5.0f);
    g.specular = saturate(coated + (make_float3(1.0f) - coated) * fuzz);

    // Energy-weighted, so the lobe that actually fills the pixel is the one the
    // denoiser is told about. The 1e-4 floor keeps a fully black specular from
    // dividing by zero rather than expressing any physics.
    const float mainEnergy = fmaxf(luminance(g.specular), 1e-4f);
    const float coatEnergy = fmaxf(coatFresnel, 0.0f);
    const float fuzzEnergy = fmaxf(luminance(fuzz), 0.0f);
    const float roughnessEnergy =
        in.specular_roughness * mainEnergy + in.coat_roughness * coatEnergy + in.fuzz_roughness * fuzzEnergy;
    g.roughness = saturate(roughnessEnergy / (mainEnergy + coatEnergy + fuzzEnergy));
    return g;
}

static __forceinline__ __device__ float traceOcclusion(
    OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction, float tmin, float tmax)
{
    const float time = optixGetRayTime();

    unsigned int transmittance = __float_as_uint(1.0f);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               time, // rayTime
               OptixVisibilityMask(RAY_MASK_SHADOW), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION, // SBT offset
               RAY_TYPE_COUNT, // SBT stride
               RAY_TYPE_OCCLUSION, // missSBTIndex
               transmittance);
    float visible = __uint_as_float(transmittance);

    // Atmospheric transmittance applies even when geometry leaves the shadow ray unobstructed.
    if (visible > 0.0f && params.hasFog)
    {
        const float tau = fogOpticalDepth(ray_origin, ray_direction, tmax, params.fogHeight, params.fogSigmaT);
        if (tau > 0.0f)
        {
            visible *= expf(-tau);
        }
    }
    return visible;
}

static __forceinline__ __device__ float3 mediumTransmittance(float3 origin,
                                                             float3 direction,
                                                             float maxDistance,
                                                             uint32_t startMedium)
{
    constexpr uint32_t kMaxCrossings = 8u;
    const float time = optixGetRayTime();

    float3 optical = make_float3(0.0f);
    float travelled = 0.0f;
    uint32_t medium = startMedium;

    for (uint32_t i = 0; i < kMaxCrossings; ++i)
    {
        const float remaining = maxDistance - travelled;
        if (remaining <= 1e-5f)
        {
            break;
        }

        optixTraverse(params.handle, origin + direction * travelled, direction, 1e-4f, remaining, time,
                      OptixVisibilityMask(GEOMETRY_MASK_MEDIUM), OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAY_TYPE_OCCLUSION,
                      RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION);

        const bool escaped = !optixHitObjectIsHit();
        const float segment = escaped ? remaining : optixHitObjectGetRayTmax();

        if (medium != 0u)
        {
            optical += mediumPropsFor(medium - 1u, make_float3(0.0f)).sigmaT * segment;
        }
        if (escaped)
        {
            break;
        }

        // The same toggle the crossing in __closesthit__radiance uses, and for
        // the same reason: a gizmo's winding is arbitrary, so the normal cannot
        // say which way the ray is going.
        const HitGroupData* hd = reinterpret_cast<HitGroupData*>(optixHitObjectGetSbtDataPointer());
        const uint32_t here = static_cast<uint32_t>(hd->materialId) + 1u;
        medium = (medium == here) ? 0u : here;

        travelled += segment + 1e-4f;
    }

    return make_float3(expf(-optical.x), expf(-optical.y), expf(-optical.z));
}

static __forceinline__ __device__ float hitOpacity(const HitGroupData* hit_data, int32_t matId)
{
    const OptixAlphaMaterialData& material = params.alphaMaterials[matId];
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    float2 sourceUv;
    if (params.hasPrimitiveAlphaData)
    {
        const OptixPrimitiveAlphaData& primitive =
            params.scene.primitiveAlphaData[hit_data->alphaPrimitiveOffset + primitiveId];
        sourceUv = interpolateAttrib(unpackUV(primitive.uv0), unpackUV(primitive.uv1), unpackUV(primitive.uv2),
                                     optixGetTriangleBarycentrics());
    }
    else
    {
        const uint32_t i0 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 0];
        const uint32_t i1 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 1];
        const uint32_t i2 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 2];
        const uint32_t baseVbOffset = hit_data->vertexOffset;
        sourceUv = interpolateAttrib(unpackUV(params.scene.vb[baseVbOffset + i0].uv),
                                     unpackUV(params.scene.vb[baseVbOffset + i1].uv),
                                     unpackUV(params.scene.vb[baseVbOffset + i2].uv),
                                     optixGetTriangleBarycentrics());
    }
    const float2 uv = make_float2(sourceUv.x * material.uvTransformX.x +
                                      sourceUv.y * material.uvTransformY.x + material.uvOffset.x,
                                  sourceUv.x * material.uvTransformX.y +
                                      sourceUv.y * material.uvTransformY.y + material.uvOffset.y);

    return resolveOpacity(material, params.materialTextures[matId * MAX_MATERIAL_TEXTURES], uv);
}

/// Any-hit for shadow rays. Only bound on instances whose material is not
/// opaque -- everything else carries OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT, so a
/// trunk or a rock never enters this program alongside the leaves.
extern "C" __global__ void __anyhit__occlusion()
{
    // A curve has no uv and is always built opaque, so it blocks outright. This
    // is reachable because the shadow hit group is shared with curve geometry.
    if (optixGetPrimitiveType() != OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        return;
    }

    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int32_t matId = hit_data->materialId;
    if (params.alphaMaterials[matId].alphaMode == ALPHA_MODE_OPAQUE)
    {
        return; // accepted; TERMINATE_ON_FIRST_HIT ends the ray here
    }

    const float opacity = hitOpacity(hit_data, matId);

    const float transmittance = __uint_as_float(optixGetPayload_0()) * (1.0f - opacity);
    if (transmittance <= SHADOW_TRANSMITTANCE_CUTOFF)
    {
        return; // nothing measurable gets through; accept and stop traversing
    }
    optixSetPayload_0(__float_as_uint(transmittance));
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__radiance()
{
    if (optixGetPrimitiveType() != OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        return;
    }

    PerRayData* prd = getPRD();
    if (prd->passthrough >= PATH_PASSTHROUGH_MAX)
    {
        return;
    }

    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int32_t matId = hit_data->materialId;
    if (params.alphaMaterials[matId].alphaMode == ALPHA_MODE_OPAQUE)
    {
        return;
    }

    const float opacity = hitOpacity(hit_data, matId);
    if (opacity < 1.0f && opacitySample(prd->sampler, launchPixelIndex(params), prd->passthrough) >= opacity)
    {
        ++prd->passthrough;
        optixIgnoreIntersection();
    }
}

static __forceinline__ __device__ uint32_t selectLightIndex(uint32_t bucketWord, uint32_t coinWord, uint32_t numLights)
{
    const uint32_t bucket = lightAliasBucket(numLights, bucketWord);
    if (bucket >= numLights)
    {
        return numLights;
    }
    const UniformLight& entry = params.scene.lights[bucket];
    return lightAliasSelect(numLights, bucket, coinWord, entry.selectionAliasThreshold, entry.selectionAlias);
}

static __forceinline__ __device__ float analyticLightSelectionPdf(const UniformLight& light)
{
    return light.color.w;
}

struct LightConnection
{
    float3 radiance; // Li times the shading cosine, unshadowed
    float3 toLight;
    float3 visibilityTarget;
    float pdf; // solid-angle density, including the light-selection probability
    float tMax;
    bool needsRay;
    bool hasVisibilityTarget;
    /// Delta lights are unreachable by BSDF sampling, so their MIS weight is one.
    bool isDelta;
    bool isResponsive;
};

/// True when light `id` is one the scene marked responsive.
static __forceinline__ __device__ bool isResponsiveLight(uint32_t id)
{
    if (params.sharcResponsive == 0u || params.sharcResponsiveLights == nullptr)
    {
        return false;
    }
    return (params.sharcResponsiveLights[id >> 5u] & (1u << (id & 31u))) != 0u;
}

static __forceinline__ __device__ LightConnection makeEmptyConnection()
{
    LightConnection c;
    c.radiance = make_float3(0.0f);
    c.toLight = make_float3(0.0f);
    c.visibilityTarget = make_float3(0.0f);
    c.pdf = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.hasVisibilityTarget = false;
    c.isDelta = false;
    c.isResponsive = false;
    return c;
}

static __forceinline__ __device__ float3 shadowOrigin(const SurfaceInteraction& si, float curveRadius, float3 toLight)
{
    if (scattersThroughFibre(si) && curveRadius > 0.0f)
    {
        return fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, toLight);
    }
    return offset_ray(si.position, orientedFaceNormal(si.geometry_normal, toLight));
}

static __forceinline__ __device__ EmissiveVisibilitySegment lightVisibilitySegment(const LightConnection& connection,
                                                                                   float3 origin)
{
    if (connection.hasVisibilityTarget)
    {
        return emissiveVisibilitySegment(origin, connection.visibilityTarget);
    }
    EmissiveVisibilitySegment segment;
    segment.direction = connection.toLight;
    segment.maxDistance = connection.tMax;
    segment.valid = connection.needsRay && connection.tMax > 0.0f;
    return segment;
}

static __forceinline__ __device__ float3 projectorEmission(const UniformLight& light, const float3 dirFromLight)
{
    const ProjectorSample p = projectorSampleForLight(light, dirFromLight);
    if (!p.inside)
    {
        return make_float3(0.0f);
    }
    const int slot = projectorImageIndex(light);
    if (slot < 0 || (uint32_t)slot >= params.scene.numProjectorTextures || params.scene.projectorTextures == nullptr)
    {
        return make_float3(p.falloff);
    }
    const cudaTextureObject_t tex = params.scene.projectorTextures[slot];
    if (tex == 0)
    {
        return make_float3(p.falloff);
    }
    // The texture is created with clamp addressing (see createProjectorTextures):
    // a slide has an edge, and repeating it would tile the wall with copies of
    // the frame as soon as a bilinear tap reached a texel past the border.
    const float4 slide = tex2D<float4>(tex, p.u, p.v);
    return p.falloff * make_float3(slide);
}

static __forceinline__ __device__ float3 emittedLightRadiance(const UniformLight& light,
                                                              float3 directionFromLight,
                                                              float distance)
{
    float3 radiance = make_float3(light.color);
    if (lightIsPunctual(light.type))
    {
        const float safeDistance = fmaxf(distance, 1e-4f);
        const bool soft = punctualLightIsSoft(light.points[0].x);
        radiance *= rangeWindow(light, safeDistance) *
                    (soft ? sphereRadianceFromIntensity(light.points[0].x) : (1.0f / (safeDistance * safeDistance)));
        const bool hasIes = light.points[0].y >= 0.0f;
        if (light.type == LIGHT_TYPE_PROJECTOR)
        {
            radiance *= projectorEmission(light, directionFromLight);
        }
        else if (hasIes)
        {
            radiance *= sampleIesCandela(params.scene.iesProfiles, light, directionFromLight);
        }
        else if (light.type == LIGHT_TYPE_SPOT)
        {
            radiance *= spotAttenuation(light, directionFromLight);
        }
    }
    return radiance * areaFalloff(light, distance);
}

static __device__ LightConnection connectLight(SamplerState& sampler,
                                               const UniformLight& light,
                                               const SurfaceInteraction& si,
                                               float curveRadius,
                                               bool volumeEvent,
                                               float localSelectionPdf,
                                               float analyticSelectionPdf,
                                               float lightSelectionPdf)
{
    LightSampleData lightSampleData = {};
    const float2 uv = make_float2(lightOpenUnitInterval(random<SampleDimension::eLightPointX>(sampler)),
                                  lightOpenUnitInterval(random<SampleDimension::eLightPointY>(sampler)));
    switch (light.type)
    {
    case LIGHT_TYPE_RECT:
        if (params.rectLightSamplingMethod == 0)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case LIGHT_TYPE_DISC:
        lightSampleData = SampleDiscLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_SPHERE:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_DISTANT:
    {
        const uint2 retryWords = make_uint2(randomBits<SampleDimension::eLightRetryU>(sampler),
                                            randomBits<SampleDimension::eLightRetryV>(sampler));
        lightSampleData = SampleDistantLight(light, uv, retryWords, si.position);
        break;
    }
    case LIGHT_TYPE_DOME:
        lightSampleData = SampleDomeLight(light, uv, si.position);
        break;
    case LIGHT_TYPE_POINT:
    case LIGHT_TYPE_SPOT:
    case LIGHT_TYPE_PROJECTOR:
        // One sampler for all three: a lamp at a point, or the sphere a soft
        // radius turns it into. What differs is the angular profile applied
        // below, not where the sample is taken.
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    LightConnection c = makeEmptyConnection();
    c.toLight = lightSampleData.L;
    const float shapeParameter = lightIsPunctual(light.type) ? light.points[0].x : light.halfAngle;
    c.isDelta = lightIsDeltaForMis(light.type, shapeParameter);

    const float3 Li = emittedLightRadiance(light, -lightSampleData.L, lightSampleData.distToLight);

    // lightReachesShadingPoint() is `dot(N, L) > 0` for everything except a
    // fibre, where the hemisphere test is the wrong question -- see the note on
    // it in shading/shading_common.h.
    const bool lit = volumeEvent || lightReachesShadingPoint(si, lightSampleData.L);
    const bool facing = lit &&
                        lightConnectionFacesVertex(light.type, -dot(lightSampleData.L, lightSampleData.normal),
                                                   lightIsPunctual(light.type) ? light.points[0].x : 0.0f) &&
                        emitsLight(Li);
    if (facing)
    {
        c.radiance = volumeEvent ? Li : Li * shadingCosine(si, lightSampleData.L);
        LightPdfQuery query = buildLightPdfQuery(light, lightSampleData);
        query.solidAngle = params.rectLightSamplingMethod != 0 ? lightSampleData.solidAngle : 0.0f;
        c.pdf = marginalLightSolidAnglePdf(query, localSelectionPdf, analyticSelectionPdf, lightSelectionPdf);
        c.tMax = lightSampleData.distToLight;
        c.needsRay = true;
        if (lightUsesAnalyticSurfaceIntersection(light.type, lightIsPunctual(light.type) ? light.points[0].x : 0.0f))
        {
            c.visibilityTarget =
                offset_ray(lightSampleData.pointOnLight, orientedFaceNormal(lightSampleData.normal, -lightSampleData.L));
            c.hasVisibilityTarget = true;
        }
    }
    return c;
}

static __device__ LightConnection connectEnvLight(SamplerState& sampler,
                                                  const SurfaceInteraction& si,
                                                  float curveRadius,
                                                  bool volumeEvent = false)
{
    const uint2 aliasWords = make_uint2(
        randomBits<SampleDimension::eLightBucket>(sampler), randomBits<SampleDimension::eLightAlias>(sampler));
    const float2 jitter =
        make_float2(random<SampleDimension::eLightPointX>(sampler), random<SampleDimension::eLightPointY>(sampler));
    const uint2 retryWords = make_uint2(randomBits<SampleDimension::eLightRetryU>(sampler),
                                        randomBits<SampleDimension::eLightRetryV>(sampler));

    float envPdf = 0.0f;
    const float3 dir = sampleEnvMap(aliasWords, jitter, retryWords, params.envAliasTable, params.envMapWidth,
                                    params.envMapHeight, params.envMapRotation, envPdf);

    LightConnection c = makeEmptyConnection();
    c.toLight = dir;
    c.pdf = envPdf;

    if (envPdf <= 0.0f)
        return c;

    // Check if direction is above the surface -- an identity for everything but
    // a fibre, which has no dark side, and for a medium, which has no side.
    if (!volumeEvent && !lightReachesShadingPoint(si, dir))
        return c;

    // Bilinear for the radiance carried down the ray; the point fetch inside
    // sampleEnvMap is for the sampling density only, and using it here would
    // quantise the lighting to the map's texels.
    const float2 uv = dirToEnvUV(dir, params.envMapRotation);
    const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
    float3 Li = make_float3(envSample.x, envSample.y, envSample.z);
    Li *= params.envMapIntensity * params.envMapColorTint;

    // Cosine folded in here for the same reason as in connectLight(), and left
    // out for a medium for the same reason.
    c.radiance = volumeEvent ? Li : Li * shadingCosine(si, dir);
    c.tMax = 1e16f;
    c.needsRay = true;
    return c;
}

struct EmissiveTriangleGeometry
{
    float3 p0;
    float3 p1;
    float3 p2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
};

static __forceinline__ __device__ float3 transformEmissivePoint(uint32_t instanceId, float3 objectPoint, float motionTime)
{
    const EmissiveInstanceTransform& current = params.scene.emissiveInstanceTransforms[instanceId];
    const EmissiveInstanceTransform* previousTransforms = params.scene.prevEmissiveInstanceTransforms;
    const EmissiveInstanceTransform& previous = previousTransforms ? previousTransforms[instanceId] : current;
    const float t = params.enableMotionBlur ? motionTime : 1.0f;
    float m[12];
#pragma unroll
    for (uint32_t i = 0u; i < 12u; ++i)
    {
        m[i] = previous.matrix[i] + (current.matrix[i] - previous.matrix[i]) * t;
    }
    return make_float3(m[0] * objectPoint.x + m[1] * objectPoint.y + m[2] * objectPoint.z + m[3],
                       m[4] * objectPoint.x + m[5] * objectPoint.y + m[6] * objectPoint.z + m[7],
                       m[8] * objectPoint.x + m[9] * objectPoint.y + m[10] * objectPoint.z + m[11]);
}

static __forceinline__ __device__ EmissiveTriangleGeometry fetchEmissiveTriangle(const EmissiveMeshLight& mesh,
                                                                                 uint32_t primitiveId,
                                                                                 float motionTime)
{
    const uint32_t i0 = params.scene.ib[mesh.indexOffset + primitiveId * 3u + 0u] + mesh.vertexOffset;
    const uint32_t i1 = params.scene.ib[mesh.indexOffset + primitiveId * 3u + 1u] + mesh.vertexOffset;
    const uint32_t i2 = params.scene.ib[mesh.indexOffset + primitiveId * 3u + 2u] + mesh.vertexOffset;
    const Vertex& v0 = params.scene.vb[i0];
    const Vertex& v1 = params.scene.vb[i1];
    const Vertex& v2 = params.scene.vb[i2];

    float3 p0 = v0.position;
    float3 p1 = v1.position;
    float3 p2 = v2.position;
    if (params.enableMotionBlur && params.scene.vb_prev != nullptr)
    {
        p0 = lerp(params.scene.vb_prev[i0].position, p0, motionTime);
        p1 = lerp(params.scene.vb_prev[i1].position, p1, motionTime);
        p2 = lerp(params.scene.vb_prev[i2].position, p2, motionTime);
    }

    EmissiveTriangleGeometry triangle;
    triangle.p0 = transformEmissivePoint(mesh.instanceId, p0, motionTime);
    triangle.p1 = transformEmissivePoint(mesh.instanceId, p1, motionTime);
    triangle.p2 = transformEmissivePoint(mesh.instanceId, p2, motionTime);
    triangle.uv0 = unpackUV(v0.uv);
    triangle.uv1 = unpackUV(v1.uv);
    triangle.uv2 = unpackUV(v2.uv);
    return triangle;
}

static __forceinline__ __device__ float3 emissiveMeshRadiance(const EmissiveMeshLight& mesh, float2 uv)
{
    const MaterialParams& material = params.materials[mesh.materialId];
    const cudaTextureObject_t* textures = &params.materialTextures[mesh.materialId * MAX_MATERIAL_TEXTURES];
    float3 emission = material.emission * material.emission_strength;
    if (material.emission_tex >= 0)
    {
        emission *=
            make_float3(texture_sample_2d(textures, material.emission_tex, apply_texture_transform(uv, material)));
    }
    return emission * resolveOpacity(material, textures, uv);
}

static __forceinline__ __device__ uint32_t sampleEmissiveMesh(SamplerState& sampler, uint32_t bucketWord)
{
    const uint32_t bucket = lightAliasBucket(params.scene.numEmissiveMeshes, bucketWord);
    const EmissiveMeshLight& entry = params.scene.emissiveMeshes[bucket];
    return lightAliasSelect(params.scene.numEmissiveMeshes, bucket, randomBits<SampleDimension::eLightAlias>(sampler),
                            entry.aliasThreshold, entry.alias);
}

static __forceinline__ __device__ uint32_t sampleEmissiveTriangleIndex(SamplerState& sampler,
                                                                       const EmissiveMeshLight& mesh)
{
    const uint32_t bucket = lightAliasBucket(mesh.triangleCount, randomBits<SampleDimension::eTriangleBucket>(sampler));
    const EmissiveTriangleLight& entry = params.scene.emissiveTriangles[mesh.triangleOffset + bucket];
    return lightAliasSelect(mesh.triangleCount, bucket, randomBits<SampleDimension::eTriangleAlias>(sampler),
                            entry.aliasThreshold, entry.alias);
}

static __forceinline__ __device__ int findEmissiveMesh(uint32_t instanceId, uint32_t geometryId)
{
    uint32_t first = 0u;
    uint32_t count = params.scene.numEmissiveMeshes;
    while (count > 0u)
    {
        const uint32_t step = count / 2u;
        const uint32_t middle = first + step;
        const EmissiveMeshLight& light = params.scene.emissiveMeshes[middle];
        if (emissiveMeshKeyLess(light.instanceId, light.geometryId, instanceId, geometryId))
        {
            first = middle + 1u;
            count -= step + 1u;
        }
        else
        {
            count = step;
        }
    }
    if (first < params.scene.numEmissiveMeshes)
    {
        const EmissiveMeshLight& light = params.scene.emissiveMeshes[first];
        if (light.instanceId == instanceId && light.geometryId == geometryId)
        {
            return static_cast<int>(first);
        }
    }
    return -1;
}

static __forceinline__ __device__ LightConnection connectEmissiveMesh(SamplerState& sampler,
                                                                      const SurfaceInteraction& si,
                                                                      uint32_t meshWord,
                                                                      bool volumeEvent,
                                                                      float localSelectionPdf,
                                                                      float meshClassPdf)
{
    LightConnection c = makeEmptyConnection();
    const uint32_t meshId = sampleEmissiveMesh(sampler, meshWord);
    if (meshId >= params.scene.numEmissiveMeshes)
    {
        return c;
    }
    const EmissiveMeshLight& mesh = params.scene.emissiveMeshes[meshId];
    const uint32_t primitiveId = sampleEmissiveTriangleIndex(sampler, mesh);
    if (primitiveId >= mesh.triangleCount)
    {
        return c;
    }
    const EmissiveTriangleLight& triangleEntry = params.scene.emissiveTriangles[mesh.triangleOffset + primitiveId];
    if (!(mesh.selectionPdf > 0.0f) || !(triangleEntry.selectionPdf > 0.0f))
    {
        return c;
    }

    const EmissiveTriangleGeometry triangle = fetchEmissiveTriangle(mesh, primitiveId, optixGetRayTime());
    const EmissiveTriangleSample sample = sampleEmissiveTriangle(
        triangle.p0, triangle.p1, triangle.p2, triangle.uv0, triangle.uv1, triangle.uv2,
        random<SampleDimension::eLightPointX>(sampler), random<SampleDimension::eLightPointY>(sampler));
    if (!sample.valid)
    {
        return c;
    }
    const float3 offset = sample.point - si.position;
    float distance;
    const float3 direction = finiteDirectionAndDistance(offset, distance);
    if (!(distance > 1e-5f))
    {
        return c;
    }
    const float3 emission = emissiveMeshRadiance(mesh, sample.uv);
    if (!emitsLight(emission) || (!volumeEvent && !lightReachesShadingPoint(si, direction)))
    {
        return c;
    }
    const float selectedMeshPdf = emissiveMeshMarginalSolidAnglePdf(localSelectionPdf, meshClassPdf, mesh.selectionPdf,
                                                                    triangleEntry.selectionPdf, sample.areaPdf,
                                                                    si.position, sample.point, sample.normal);
    if (!(selectedMeshPdf > 0.0f))
    {
        return c;
    }

    c.radiance = volumeEvent ? emission : emission * shadingCosine(si, direction);
    c.toLight = direction;
    c.visibilityTarget = offset_ray(sample.point, orientedFaceNormal(sample.normal, -direction));
    c.pdf = selectedMeshPdf;
    c.tMax = distance;
    c.needsRay = true;
    c.hasVisibilityTarget = true;
    c.isDelta = false;
    c.isResponsive = false;
    return c;
}

static __forceinline__ __device__ float emissiveMeshHitPdf(
    uint32_t instanceId, uint32_t geometryId, uint32_t primitiveId, float3 shadingPoint, float3 pointOnLight)
{
    const int meshId = findEmissiveMesh(instanceId, geometryId);
    if (meshId < 0)
    {
        return 0.0f;
    }
    const EmissiveMeshLight& mesh = params.scene.emissiveMeshes[meshId];
    if (primitiveId >= mesh.triangleCount)
    {
        return 0.0f;
    }
    const EmissiveTriangleLight& triangleEntry = params.scene.emissiveTriangles[mesh.triangleOffset + primitiveId];
    const EmissiveTriangleGeometry triangle = fetchEmissiveTriangle(mesh, primitiveId, optixGetRayTime());
    const EmissiveTriangleSample geometry = sampleEmissiveTriangle(
        triangle.p0, triangle.p1, triangle.p2, triangle.uv0, triangle.uv1, triangle.uv2, 0.25f, 0.5f);
    if (!geometry.valid)
    {
        return 0.0f;
    }
    const float localSelectionPdf = params.hasEnvMap ? 1.0f - params.envSelectionPdf : 1.0f;
    const float meshClassPdf = params.scene.numLights > 0u ? params.scene.meshLightSelectionPdf : 1.0f;
    return emissiveMeshMarginalSolidAnglePdf(localSelectionPdf, meshClassPdf, mesh.selectionPdf,
                                             triangleEntry.selectionPdf, geometry.areaPdf, shadingPoint, pointOnLight,
                                             geometry.normal);
}

/// Choose a strategy and build the connection. Visibility is the caller's job.
static __device__ LightConnection connectToLight(SamplerState& sampler,
                                                 const SurfaceInteraction& si,
                                                 float curveRadius,
                                                 bool volumeEvent = false)
{
    const bool hasAnalytic = params.scene.numLights > 0u;
    const bool hasMesh = params.scene.numEmissiveMeshes > 0u;
    const bool hasLocal = hasAnalytic || hasMesh;
    const uint32_t emitterWord = randomBits<SampleDimension::eLightId>(sampler);
    float localSelectionPdf = 1.0f;
    if (params.hasEnvMap)
    {
        localSelectionPdf = 1.0f - params.envSelectionPdf;
        if (!hasLocal || discreteBernoulli(emitterWord, params.envSelectionPdf))
        {
            LightConnection c = connectEnvLight(sampler, si, curveRadius, volumeEvent);
            c.pdf *= hasLocal ? params.envSelectionPdf : 1.0f;
            c.isResponsive = false;
            return c;
        }
    }

    if (!hasLocal)
    {
        return makeEmptyConnection();
    }

    const uint32_t classWord = randomBits<SampleDimension::eLightClass>(sampler);
    const float meshPdf = params.scene.meshLightSelectionPdf;
    if (hasMesh && (!hasAnalytic || discreteBernoulli(classWord, meshPdf)))
    {
        const uint32_t meshWord = randomBits<SampleDimension::eLightBucket>(sampler);
        return connectEmissiveMesh(sampler, si, meshWord, volumeEvent, localSelectionPdf, hasAnalytic ? meshPdf : 1.0f);
    }
    const float analyticPdf = hasMesh ? 1.0f - meshPdf : 1.0f;
    if (!(analyticPdf > 0.0f))
    {
        return makeEmptyConnection();
    }
    const uint32_t analyticWord = randomBits<SampleDimension::eLightBucket>(sampler);
    const uint32_t aliasWord = randomBits<SampleDimension::eLightAlias>(sampler);
    const uint32_t lightId = selectLightIndex(analyticWord, aliasWord, params.scene.numLights);
    if (lightId >= params.scene.numLights)
    {
        return makeEmptyConnection();
    }
    LightConnection c =
        connectLight(sampler, params.scene.lights[lightId], si, curveRadius, volumeEvent, localSelectionPdf,
                     analyticPdf, analyticLightSelectionPdf(params.scene.lights[lightId]));
    c.isResponsive = isResponsiveLight(lightId);
    return c;
}

template <typename OpenPBRPrepared>
static __device__ float3 estimateDirectLighting(PerRayData* prd,
                                                const SurfaceInteraction& si,
                                                float curveRadius,
                                                const OpenPBRPrepared* openpbrPrepared,
                                                const PbrPrepared& pbrPrepared,
                                                float3 throughputAtVertex,
                                                uint32_t mediumAtVertex,
                                                bool& outResponsive)
{
    outResponsive = false;
    const uint32_t candidates = max(params.risCandidates, 1u);
    const bool isFibre = (curveRadius > 0.0f);

    LightConnection bestConn = makeEmptyConnection();
    float3 bestF = make_float3(0.0f);
    float bestTarget = 0.0f;
    float weightSum = 0.0f;

    for (uint32_t i = 0; i < candidates; ++i)
    {
        // Candidates differ by their scramble, not by their dimension: each one
        // stays a stratified sequence across samples, and the first is the
        // sequence this code drew before RIS existed.
        SamplerState crng = prd->sampler;
        if (i != 0u)
        {
            crng.seed = hash_combine(prd->sampler.seed, i * 0x9E3779B9u);
        }

        const LightConnection conn = connectToLight(crng, si, curveRadius);
        const ShadedFrame frame =
            shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
        const bool isNextEventValid =
            neeProposesDirection(neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission), frame.frontFace,
                                 frame.normalSign * dot(si.shading_normal, conn.toLight)) &&
            (conn.pdf > 0.0f);
        if (!isNextEventValid || !conn.needsRay)
        {
            continue;
        }

        const BsdfEvalResult evalData = (openpbrPrepared != nullptr)
                                            ? openpbr_bsdf_eval(*openpbrPrepared, si, conn.toLight)
                                            : bsdf_eval(si, conn.toLight, pbrPrepared);
        if (isnan(conn.radiance) || isnan(conn.pdf) || isnan(evalData.bsdf) || isnan(evalData.pdf))
        {
            // ERROR, terminate tracing
            prd->radiance = make_float3(10000.0f, 0.0f, 0.0f);
            prd->throughput = make_float3(0.0f);
            return make_float3(0.0f);
        }
        if (!(evalData.pdf > 0.0f))
        {
            continue;
        }

        const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, evalData.pdf, params.misHeuristic);
        const float3 f = conn.radiance * evalData.bsdf * misWeight;
        const float target = dot(f, make_float3(0.2126f, 0.7152f, 0.0722f));
        if (!(target > 0.0f))
        {
            continue;
        }

        const float w = target / conn.pdf;
        weightSum += w;
        // The acceptance draw reuses eLightId under a different scramble rather
        // than taking a dimension of its own: adding one to the enum shifts every
        // dimension index above it.
        SamplerState arng = crng;
        arng.seed = hash_combine(crng.seed, 0x51633e2du);
        if (random<SampleDimension::eLightId>(arng) * weightSum <= w)
        {
            bestConn = conn;
            bestF = f;
            bestTarget = target;
        }
    }

    if (!(bestTarget > 0.0f))
    {
        return make_float3(0.0f);
    }

    // The reservoir's contribution weight: the mean candidate weight over the
    // target the survivor was kept for. At one candidate it is 1 / pdf and every
    // line here is the arithmetic this code had.
    const float W = (weightSum / (float)candidates) / bestTarget;
    outResponsive = bestConn.isResponsive;
    const float3 weight = throughputAtVertex * bestF * W;
    if (weight.x == 0.0f && weight.y == 0.0f && weight.z == 0.0f)
    {
        return make_float3(0.0f);
    }

    // Mesh emitters remain in traversal, so their segment ends at an offset
    // point on the near side instead of letting the emitter shadow itself.
    const float3 origin = shadowOrigin(si, curveRadius, bestConn.toLight);
    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(bestConn, origin);
    if (!visibility.valid)
    {
        return make_float3(0.0f);
    }
    // Occlusion returns accumulated transmittance, not binary visibility.
    const float transmittance =
        traceOcclusion(params.handle, origin, visibility.direction, params.shadowRayTmin, visibility.maxDistance);
    if (transmittance <= 0.0f)
    {
        return make_float3(0.0f);
    }
    // Geometry transmittance is additionally attenuated through bounded media.
    float3 survived = make_float3(transmittance);
    if (params.hasBoundedMedium)
    {
        survived *= mediumTransmittance(origin, visibility.direction, visibility.maxDistance, mediumAtVertex);
    }
    return clampIndirectContribution(weight * survived, prd->depth, params.clampIndirect);
}

// Get curve hit-point in world coordinates.
static __forceinline__ __device__ float3 getHitPoint()
{
    const float t = optixGetRayTmax();
    const float3 rayOrigin = optixGetWorldRayOrigin();
    const float3 rayDirection = optixGetWorldRayDirection();

    return rayOrigin + t * rayDirection;
}

struct SurfaceHitData
{
    float3 position;
    float3 normal;
    float3 geom_normal;
    float2 uv;
    float3 worldTangent;
    float3 worldBinormal;
    /// glTF COLOR_0, interpolated. White when the mesh carries no colour set.
    float3 vertexColor;
    /// World-space radius of the strand at the hit. Zero for a triangle, and the
    /// one thing the fibre chord needs that a curve hit does not hand back.
    float curveRadius;
};

static __forceinline__ __device__ float3 vertexColorOf(const Vertex& v)
{
    return unpack_vertex_color(__float_as_uint(v.pad1));
}

static __forceinline__ __device__ SurfaceHitData fillTriangleGeomData(const HitGroupData* hit_data)
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];

    const uint32_t baseVbOffset = hit_data->vertexOffset;

    float3 p0, p1, p2, n0, n1, n2, t0, t1, t2;
    float2 uv0, uv1, uv2;
    float3 c0, c1, c2;
    // glTF TANGENT.w rides in bit 30 of the packed tangent. Handedness is a
    // per-mesh property in every exporter that writes it, so one vertex settles
    // it -- there is nothing sensible to interpolate.
    float tangentSign = 1.0f;

    if (params.enableMotionBlur)
    {
        const float t = optixGetRayTime();
        const Vertex* vb0 = params.scene.vb_prev;
        const Vertex* vb1 = params.scene.vb;

        const Vertex v0_0 = vb0[baseVbOffset + i0];
        const Vertex v1_0 = vb0[baseVbOffset + i1];
        const Vertex v2_0 = vb0[baseVbOffset + i2];

        const Vertex v0_1 = vb1[baseVbOffset + i0];
        const Vertex v1_1 = vb1[baseVbOffset + i1];
        const Vertex v2_1 = vb1[baseVbOffset + i2];

        // Interpolate all vertex attributes
        p0 = lerp(v0_0.position, v0_1.position, t);
        p1 = lerp(v1_0.position, v1_1.position, t);
        p2 = lerp(v2_0.position, v2_1.position, t);

        n0 = lerp(unpackNormal(v0_0.normal), unpackNormal(v0_1.normal), t);
        n1 = lerp(unpackNormal(v1_0.normal), unpackNormal(v1_1.normal), t);
        n2 = lerp(unpackNormal(v2_0.normal), unpackNormal(v2_1.normal), t);

        t0 = lerp(unpackNormal(v0_0.tangent), unpackNormal(v0_1.tangent), t);
        t1 = lerp(unpackNormal(v1_0.tangent), unpackNormal(v1_1.tangent), t);
        t2 = lerp(unpackNormal(v2_0.tangent), unpackNormal(v2_1.tangent), t);

        uv0 = lerp(unpackUV(v0_0.uv), unpackUV(v0_1.uv), t);
        uv1 = lerp(unpackUV(v1_0.uv), unpackUV(v1_1.uv), t);
        uv2 = lerp(unpackUV(v2_0.uv), unpackUV(v2_1.uv), t);

        // Vertex colour is not skinned and does not animate, so it is read from
        // the current frame even when the rest is motion-interpolated.
        c0 = vertexColorOf(v0_1);
        c1 = vertexColorOf(v1_1);
        c2 = vertexColorOf(v2_1);
        tangentSign = unpackTangentSign(v0_1.tangent);
    }
    else
    {
        const Vertex v0 = params.scene.vb[baseVbOffset + i0];
        const Vertex v1 = params.scene.vb[baseVbOffset + i1];
        const Vertex v2 = params.scene.vb[baseVbOffset + i2];
        p0 = v0.position;
        p1 = v1.position;
        p2 = v2.position;

        n0 = unpackNormal(v0.normal);
        n1 = unpackNormal(v1.normal);
        n2 = unpackNormal(v2.normal);

        t0 = unpackNormal(v0.tangent);
        t1 = unpackNormal(v1.tangent);
        t2 = unpackNormal(v2.tangent);

        uv0 = unpackUV(v0.uv);
        uv1 = unpackUV(v1.uv);
        uv2 = unpackUV(v2.uv);

        c0 = vertexColorOf(v0);
        c1 = vertexColorOf(v1);
        c2 = vertexColorOf(v2);
        tangentSign = unpackTangentSign(v0.tangent);
    }

    const float2 uvCoord = interpolateAttrib(uv0, uv1, uv2, barycentrics);

    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(interpolateAttrib(p0, p1, p2, barycentrics));
    const float3 object_normal = interpolateAttrib(n0, n1, n2, barycentrics);
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 geomNormal = cross(p1 - p0, p2 - p0);
    geomNormal = safe_normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));
    const float3 worldTangent = orthonormalizeTangent(
        worldNormal, optixTransformVectorFromObjectToWorldSpace(interpolateAttrib(t0, t1, t2, barycentrics)));
    // Without TANGENT.w the bitangent points the wrong way and every normal map
    // is mirrored along it -- bumps light from the opposite side.
    const float3 worldBinormal = cross(worldNormal, worldTangent) * tangentSign;

    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = geomNormal;
    res.position = worldPosition;
    res.uv = uvCoord;
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    res.vertexColor = interpolateAttrib(c0, c1, c2, barycentrics);
    res.curveRadius = 0.0f;
    return res;
}

static __forceinline__ __device__ float2 curveStrandUV(const HitGroupData* hit_data, unsigned int primitiveIndex, float u)
{
    return make_float2(oka::curve_layout::strandCoordinate(primitiveIndex, hit_data->curveSegmentsPerStrand, u), 0.0f);
}

static __forceinline__ __device__ SurfaceHitData fillCubicCurveGeomData(const HitGroupData* hit_data)
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIndex = optixGetSbtGASIndex();
    float4 controlPoints[4];
    optixGetCubicBSplineVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);
    CubicInterpolator interpolator;
    interpolator.initializeFromBSpline(controlPoints);
    const float u = optixGetCurveParameter();
    float3 hitPoint = getHitPoint();
    // interpolators work in object space
    hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint); // interpolators work in object space
    const float3 objectNormal = surfaceNormal(interpolator, u, hitPoint);
    float3 worldNormal = safe_normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
    const float3 worldTangent = orthonormalizeTangent(
        worldNormal, optixTransformVectorFromObjectToWorldSpace(curveTangent(interpolator, u)));
    const float3 worldBinormal = cross(worldNormal, worldTangent);
    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(hitPoint);
    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = worldNormal;
    res.position = worldPosition;
    res.uv = curveStrandUV(hit_data, primitiveIndex, u);
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    res.vertexColor = make_float3(1.0f);
    // Same radial-vector reasoning as the linear arm below; a cubic strand needs
    // the chord for fibre_exit() exactly as much as a linear one does.
    res.curveRadius = length(optixTransformVectorFromObjectToWorldSpace(objectNormal * interpolator.radius(u)));

    return res;
}

/// Round linear curves use two control points and cylindrical geometry with spherical caps.
static __forceinline__ __device__ SurfaceHitData fillLinearCurveGeomData(const HitGroupData* hit_data)
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIndex = optixGetSbtGASIndex();
    float4 controlPoints[2];
    optixGetLinearCurveVertexData(gas, primitiveIndex, gasSbtIndex, 0.0f, controlPoints);
    LinearInterpolator interpolator;
    interpolator.initialize(controlPoints);
    const float u = optixGetCurveParameter();
    float3 hitPoint = getHitPoint();
    hitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
    const float3 objectNormal = surfaceNormal(interpolator, u, hitPoint);
    // safe_normalize: see fillCubicCurveGeomData -- a tapered/near-axial hit
    // collapses this to (0,0,0), and normalize((0,0,0)) is NaN.
    float3 worldNormal = safe_normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
    const float3 worldTangent = orthonormalizeTangent(
        worldNormal, optixTransformVectorFromObjectToWorldSpace(curveTangent(interpolator, u)));
    const float3 worldBinormal = cross(worldNormal, worldTangent);
    const float3 worldPosition = optixTransformPointFromObjectToWorldSpace(hitPoint);
    SurfaceHitData res;
    res.normal = worldNormal;
    res.geom_normal = worldNormal;
    res.position = worldPosition;
    res.uv = curveStrandUV(hit_data, primitiveIndex, u);
    res.worldTangent = worldTangent;
    res.worldBinormal = worldBinormal;
    res.vertexColor = make_float3(1.0f);
    res.curveRadius = length(optixTransformVectorFromObjectToWorldSpace(objectNormal * interpolator.radius(u)));

    return res;
}

static __forceinline__ __device__ bool previousTriangleWorldPosition(const HitGroupData* hit_data, float3& outPosition)
{
    if (params.scene.vb_prev == nullptr)
    {
        return false;
    }
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 0)];
    const uint32_t i1 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 1)];
    const uint32_t i2 = params.scene.ib[(hit_data->indexOffset + primitiveId * 3 + 2)];
    const uint32_t base = hit_data->vertexOffset;
    const float3 objectPos =
        interpolateAttrib(params.scene.vb_prev[base + i0].position, params.scene.vb_prev[base + i1].position,
                          params.scene.vb_prev[base + i2].position, barycentrics);
    outPosition = optixTransformPointFromObjectToWorldSpace(objectPos);
    return true;
}

static __forceinline__ __device__ void writeSurfaceGuide(const HitGroupData* hit_data,
                                                         PerRayData* prd,
                                                         const SurfaceInteraction& si,
                                                         const float3 worldPosition,
                                                         const bool isTriangle,
                                                         const OpenPBRGuides* openpbrGuides)
{
    const uint32_t pixelIndex = launchPixelIndex(params);
    if (prd->depth == 0)
    {
        float3 prevPosition = worldPosition;
        if (isTriangle)
        {
            previousTriangleWorldPosition(hit_data, prevPosition);
        }
        const float2 motion =
            guideScreenMotion(params, make_float4(prevPosition, 1.0f), guideCurrentSample(params, pixelIndex));
        params.aov[pixelIndex].depth = guideViewDepth(params, worldPosition);
        params.aov[pixelIndex].motionX = motion.x;
        params.aov[pixelIndex].motionY = motion.y;
    }

    const float guideRoughness = (openpbrGuides != nullptr) ? openpbrGuides->roughness : si.roughness;
    if (oka::guides::shouldWriteGuide(true, prd->aovDone, params.guidePrimaryHit, prd->depth, guideRoughness))
    {
        const AovSample previous = params.aov[pixelIndex];
        AovSample a;
        // Metals put their colour in the specular lobe and have no diffuse one.
        //
        // Albedo guides are reflectance, so saturate unbounded material colour for the denoiser.
        const float3 base = saturate(si.albedo);
        a.diffuseAlbedo = (openpbrGuides != nullptr) ? saturate(openpbrGuides->diffuse) : base * (1.0f - si.metallic);
        a.specularAlbedo = (openpbrGuides != nullptr) ? openpbrGuides->specular
                                                      : lerp(make_float3(0.04f), base, si.metallic);
        a.normal = si.shading_normal;
        a.roughness = guideRoughness;
        // Taken from the block above, which wrote them for the primary surface
        // whatever this one is.
        a.depth = previous.depth;
        a.motionX = previous.motionX;
        a.motionY = previous.motionY;
        a.specularHitDistance = 0.0f;
        a.reactive = oka::guides::reactiveFor(prd->depth);
        a.pad2 = 0.0f;
        params.aov[pixelIndex] = a;
        prd->aovDone = true;
    }

    if (prd->depth == 1 && prd->specularBounce)
    {
        params.aov[pixelIndex].specularHitDistance = optixGetRayTmax();
    }
}

static __device__ void scatterInMedium(PerRayData* prd,
                                       const MaterialParams& mm,
                                       const MediumSample& m,
                                       const float anisotropy,
                                       const float3 rayOrigin,
                                       const float3 rayDir,
                                       const bool isBounded)
{
    prd->throughput *= mediumScatterWeight(m, m.t);
    const float3 scatterPoint = rayOrigin + rayDir * m.t;
    SamplerState mrng = mediumSampler(prd->sampler, prd->mediumStep + 1u);

    // Record NEE availability, not its sampled outcome, so the paired bounce keeps its MIS weight.
    bool didNee = volumeNeePairsWithBounce(
        params.estimatorMode == 0, params.scene.numLights > 0 || params.scene.numEmissiveMeshes > 0 || params.hasEnvMap);
    if (isBounded)
    {
        const float3 Le = mm.medium_emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f)
        {
            prd->radiance += clampIndirectContribution(prd->throughput * Le, prd->depth, params.clampIndirect);
        }

        if (didNee)
        {
            SurfaceInteraction vsi = {};
            vsi.position = scatterPoint;
            vsi.shading_normal = -rayDir;
            vsi.geometry_normal = -rayDir;
            vsi.wo = -rayDir;
            vsi.front_face = true;
            const LightConnection conn = connectToLight(mrng, vsi, 0.0f, true);
            if (conn.needsRay && conn.pdf > 0.0f)
            {
                const float phase =
                    hgPhaseFunction(dot(rayDir, conn.toLight), anisotropy);
                // The phase function is the medium's BSDF and its own pdf, so MIS
                // pairs it against the light density exactly as a surface lobe
                // would.
                const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, params.misHeuristic);
                const float3 weight = prd->throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
                {
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                    const float visible = visibility.valid ?
                                              traceOcclusion(params.handle, scatterPoint, visibility.direction,
                                                             params.shadowRayTmin, visibility.maxDistance) :
                                              0.0f;
                    if (visible > 0.0f)
                    {
                        // Starts inside this medium, so the optical depth begins
                        // accumulating immediately rather than at the first
                        // boundary the ray crosses on the way out.
                        float3 survived = make_float3(visible);
                        if (params.hasBoundedMedium)
                        {
                            survived *= mediumTransmittance(
                                scatterPoint, visibility.direction, visibility.maxDistance, prd->medium);
                        }
                        prd->radiance += clampIndirectContribution(weight * survived, prd->depth, params.clampIndirect);
                    }
                }
            }
        }
    }

    float phasePdf = 0.0f;
    const float3 nextDir =
        hgSampleDirection(-rayDir, anisotropy, random<SampleDimension::eBSDF0>(mrng),
                          random<SampleDimension::eBSDF1>(mrng), phasePdf);

    prd->origin = scatterPoint;
    prd->dir = nextDir;
    prd->lastBsdfPdf = phasePdf;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    // Medium scattering launches a broad lobe, so SHARC eligibility uses full roughness.
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    ++prd->mediumStep;

    if (isBounded)
    {
        return;
    }

    const float survive = clamp(fmaxf(prd->throughput.x, fmaxf(prd->throughput.y, prd->throughput.z)), 0.05f, 1.0f);
    if (random<SampleDimension::eRussianRoulette>(mrng) >= survive)
    {
        prd->throughput = make_float3(0.0f);
        prd->depth = params.max_depth;
        return;
    }
    prd->throughput /= survive;
    prd->passedThrough = true;
}

static __device__ void scatterInFog(PerRayData* prd, const float3 rayOrigin, const float3 rayDir, const float t)
{
    prd->throughput *= params.fogAlbedo;
    const float3 scatterPoint = rayOrigin + rayDir * t;

    const bool didNee = volumeNeePairsWithBounce(
        params.estimatorMode == 0, params.scene.numLights > 0 || params.scene.numEmissiveMeshes > 0 || params.hasEnvMap);
    if (didNee)
    {
        // A medium event has a position and no normal. Facing the ray back the
        // way it came is what tells the light connection this is a volume vertex
        // and stops it applying a surface cosine.
        SurfaceInteraction vsi = {};
        vsi.position = scatterPoint;
        vsi.shading_normal = -rayDir;
        vsi.geometry_normal = -rayDir;
        vsi.wo = -rayDir;
        vsi.front_face = true;
        const LightConnection conn = connectToLight(prd->sampler, vsi, 0.0f, true);
        if (conn.needsRay && conn.pdf > 0.0f)
        {
            const float phase = hgPhaseFunction(dot(rayDir, conn.toLight), params.fogAnisotropy);
            // The phase function is the medium's BSDF and its own pdf, so MIS
            // pairs it against the light density exactly as a surface lobe would.
            const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, params.misHeuristic);
            const float3 weight = prd->throughput * (conn.radiance / conn.pdf) * misWeight * phase;
            if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
            {
                // traceOcclusion carries the haze's own transmittance along this
                // ray, so the connection is dimmed by the medium it starts in.
                const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, scatterPoint);
                const float visible = visibility.valid ?
                                          traceOcclusion(params.handle, scatterPoint, visibility.direction,
                                                         params.shadowRayTmin, visibility.maxDistance) :
                                          0.0f;
                if (visible > 0.0f)
                {
                    prd->radiance += clampIndirectContribution(weight * visible, prd->depth, params.clampIndirect);
                }
            }
        }
    }

    float phasePdf = 0.0f;
    const float3 nextDir =
        hgSampleDirection(-rayDir, params.fogAnisotropy, random<SampleDimension::eFogPhaseU>(prd->sampler),
                          random<SampleDimension::eFogPhaseV>(prd->sampler), phasePdf);

    prd->origin = scatterPoint;
    prd->dir = nextDir;
    prd->lastBsdfPdf = phasePdf;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    // No passedThrough: an atmospheric scattering event is a bounce like any
    // other, so the raygen loop charges it a depth and rolls roulette on it. A
    // medium with no depth budget of its own is a path that wanders forever.
}

static __forceinline__ __device__ bool fogScatters(
    PerRayData* prd, const float3 rayOrigin, const float3 rayDir, const float tMax, float& t)
{
    // Single-hit debug views decline fog events because atmospheric vertices have no normal.
    if (!params.hasFog || prd->medium != 0u || DEBUG_MODE_IS_SINGLE_HIT(params.debug))
    {
        return false;
    }
    return fogSampleDistance(rayOrigin, rayDir, tMax, params.fogHeight, params.fogSigmaT,
                             random<SampleDimension::eFogDistance>(prd->sampler), t);
}

static __device__ void exitMedium(PerRayData* prd,
                                  const float3 worldPosition,
                                  const float3 shadingNormal,
                                  const float3 geomNormal,
                                  const float3 rayDir)
{
    const float3 outwardGeom = (dot(geomNormal, rayDir) > 0.0f) ? geomNormal : -geomNormal;
    const float3 outward = (dot(shadingNormal, outwardGeom) > 0.0f) ? shadingNormal : -shadingNormal;
    const float3 exitOrigin = offset_ray(worldPosition, outwardGeom);
    SamplerState xrng = mediumSampler(prd->sampler, prd->mediumStep + 1u);
    const float invPi = 1.0f / M_PIf;

    bool didNee = volumeNeePairsWithBounce(
        params.estimatorMode == 0, params.scene.numLights > 0 || params.scene.numEmissiveMeshes > 0 || params.hasEnvMap);
    if (didNee)
    {
        // Next-event estimation here and not inside the walk: this is the vertex
        // light can actually reach, and leaving it to BSDF sampling alone is what
        // makes a translucent object the noisiest thing in a frame.
        SurfaceInteraction xsi = {};
        xsi.position = worldPosition;
        xsi.shading_normal = outward;
        xsi.geometry_normal = outward;
        xsi.wo = -rayDir;
        xsi.front_face = true;
        const LightConnection conn = connectToLight(xrng, xsi, 0.0f);
        if (conn.needsRay && conn.pdf > 0.0f)
        {
            const float cosOut = dot(outward, conn.toLight);
            if (cosOut > 0.0f)
            {
                // The Lambertian exit density is MIS-only; conn.radiance already includes the cosine.
                const float lobePdf = cosOut * invPi;
                const float misWeight = conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, lobePdf, params.misHeuristic);
                const float3 weight = prd->throughput * (conn.radiance / conn.pdf) * misWeight * invPi;
                if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
                {
                    const EmissiveVisibilitySegment visibility = lightVisibilitySegment(conn, exitOrigin);
                    const float visible = visibility.valid ?
                                              traceOcclusion(params.handle, exitOrigin, visibility.direction,
                                                             params.shadowRayTmin, visibility.maxDistance) :
                                              0.0f;
                    if (visible > 0.0f)
                    {
                        float3 survived = make_float3(visible);
                        if (params.hasBoundedMedium)
                        {
                            survived *= mediumTransmittance(exitOrigin, visibility.direction, visibility.maxDistance, 0u);
                        }
                        prd->radiance += clampIndirectContribution(weight * survived, prd->depth, params.clampIndirect);
                    }
                }
            }
        }
    }

    const float3 exitDir =
        mediumCosineDirection(outward, random<SampleDimension::eBSDF0>(xrng), random<SampleDimension::eBSDF1>(xrng));

    prd->origin = exitOrigin;
    prd->dir = exitDir;
    // Cosine-sampled from a 1/pi lobe: f * cos / pdf is exactly one, so the
    // throughput is untouched here. What the medium took, it took during the
    // walk.
    prd->lastBsdfPdf = fmaxf(dot(outward, exitDir), 0.0f) * invPi;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    prd->medium = 0u;
    prd->mediumStep = 0u;
}

static __forceinline__ __device__ void shadeAnalyticAreaLightHit(PerRayData* prd,
                                                                 const AnalyticAreaLightHit& hit,
                                                                 float3 rayOrigin,
                                                                 float3 rayDirection)
{
    const UniformLight& light = params.scene.lights[hit.lightId];
    const float3 hitPoint = hit.point;
    const float3 lightNormal = hit.normal;

    if (prd->writeAov && !prd->aovDone && params.aov != nullptr)
    {
        AovSample a;
        const float3 lightColor = make_float3(light.color);
        const float peak = fmaxf(fmaxf(lightColor.x, lightColor.y), fmaxf(lightColor.z, 1e-6f));
        a.diffuseAlbedo = lightColor / peak;
        a.specularAlbedo = make_float3(0.0f);
        a.normal = lightNormal;
        a.roughness = 1.0f;
        a.depth = prd->depth == 0 ? guideViewDepth(params, hitPoint) : params.aov[launchPixelIndex(params)].depth;
        if (prd->depth == 0)
        {
            const uint32_t pixelIndex = launchPixelIndex(params);
            const float2 motion =
                guideScreenMotion(params, make_float4(hitPoint, 1.0f), guideCurrentSample(params, pixelIndex));
            a.motionX = motion.x;
            a.motionY = motion.y;
        }
        else
        {
            a.motionX = params.aov[launchPixelIndex(params)].motionX;
            a.motionY = params.aov[launchPixelIndex(params)].motionY;
        }
        a.specularHitDistance = 0.0f;
        a.reactive = oka::guides::reactiveFor(prd->depth);
        a.pad2 = 0.0f;
        params.aov[launchPixelIndex(params)] = a;
        prd->aovDone = true;
    }

    const float3 misOrigin = rayOrigin - rayDirection * prd->misDistance;
    float3 radiance = make_float3(0.0f);
    // Coincident emitters are separate integrand components even though they
    // share one geometric event. Each keeps its own selection PMF and MIS pair.
    for (uint32_t componentId = 0u; componentId < params.scene.numLights; ++componentId)
    {
        const UniformLight& component = params.scene.lights[componentId];
        if (!analyticLightVisibilityAllowsRay(component.normal.w, prd->depth != 0u))
        {
            continue;
        }
        const AnalyticLightIntersection componentHit = intersectAnalyticLightSurfaceUnchecked(
            component.type, make_float3(component.points[0]), make_float3(component.points[1]),
            make_float3(component.points[2]), make_float3(component.points[3]), make_float3(component.normal),
            rayOrigin, rayDirection, params.materialRayTmin, 1e16f);
        if (!analyticLightIntersectionSharesEvent(hit.distance, componentHit))
        {
            continue;
        }
        const float componentCosine = -dot(rayDirection, componentHit.normal);
        if (!lightConnectionFacesVertex(component.type, componentCosine,
                                        lightIsPunctual(component.type) ? component.points[0].x : 0.0f))
        {
            continue;
        }
        const float hitDistance = finiteVectorLength(componentHit.point - misOrigin);
        const float3 Le = emittedLightRadiance(component, -rayDirection, hitDistance);
        if (prd->depth == 0 || prd->specularBounce || !prd->neeDone)
        {
            radiance += prd->throughput * Le;
        }
        else
        {
            const float localSelectionPdf = params.hasEnvMap ? 1.0f - params.envSelectionPdf : 1.0f;
            const float analyticClassPdf =
                params.scene.numEmissiveMeshes > 0u ? 1.0f - params.scene.meshLightSelectionPdf : 1.0f;
            const float lightPdf = areaPdfToSolidAngleMarginalPdf(
                hitDistance, componentCosine, componentHit.areaPdf, localSelectionPdf, analyticClassPdf,
                analyticLightSelectionPdf(component), 1.0f);
            radiance +=
                prd->throughput * Le * computeMisWeight(prd->lastBsdfPdf, lightPdf, params.misHeuristic);
        }
    }
    prd->radiance += clampIndirectContribution(radiance, prd->depth, params.clampIndirect);
    prd->throughput = make_float3(0.0f);
}

extern "C" __global__ void __intersection__light()
{
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const uint32_t lightId = hit_data->lightIndices[optixGetPrimitiveIndex()];
    const UniformLight& light = params.scene.lights[lightId];

    const AnalyticLightIntersection hit = intersectAnalyticLightSurfaceUnchecked(
        light.type, make_float3(light.points[0]), make_float3(light.points[1]), make_float3(light.points[2]),
        make_float3(light.points[3]), make_float3(light.normal), optixGetObjectRayOrigin(),
        optixGetObjectRayDirection(), optixGetRayTmin(), optixGetRayTmax());
    if (hit.hit)
    {
        optixReportIntersection(hit.distance, 0u, lightId);
    }
}

extern "C" __global__ void __closesthit__analytic_light()
{
    PerRayData* prd = getPRD();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    AnalyticAreaLightHit hit;
    hit.lightId = optixGetAttribute_0();
    const UniformLight& light = params.scene.lights[hit.lightId];
    const AnalyticLightIntersection surface = intersectAnalyticLightSurfaceUnchecked(
        light.type, make_float3(light.points[0]), make_float3(light.points[1]), make_float3(light.points[2]),
        make_float3(light.points[3]), make_float3(light.normal), ray_origin, ray_dir, params.materialRayTmin, 1e16f);
    hit.distance = surface.distance;
    hit.point = surface.point;
    hit.normal = surface.normal;
    hit.areaPdf = surface.areaPdf;
    hit.hit = surface.hit;

    // The haze in front of the emitter is still in front of it.
    float fogT = 0.0f;
    if (fogScatters(prd, ray_origin, ray_dir, hit.distance, fogT))
    {
        scatterInFog(prd, ray_origin, ray_dir, fogT);
        return;
    }

    shadeAnalyticAreaLightHit(prd, hit, ray_origin, ray_dir);
}

extern "C" __global__ void __miss__ms()
{
    PerRayData* prd = getPRD();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_origin = optixGetWorldRayOrigin();

    // Before everything else, including the counters and the guide: if the haze
    // scatters, this path did not reach the environment and nothing below is
    // true of it.
    float fogT = 0.0f;
    if (fogScatters(prd, ray_origin, ray_dir, 1e16f, fogT))
    {
        scatterInFog(prd, optixGetWorldRayOrigin(), ray_dir, fogT);
        return;
    }

    // Report paths that escape while still inside a dielectric, indicating an open mesh or unmatched exit.
    if (params.iorStats != nullptr && prd->iorStack.top >= 0)
    {
        atomicAdd(&params.iorStats[IOR_STAT_ESCAPED_INSIDE], 1u);
    }

    // Background still needs a guide record, or the denoiser reads whatever the
    // previous frame left there and smears the silhouette across the sky.
    if (prd->writeAov && !prd->aovDone && params.aov != nullptr)
    {
        const uint32_t pixelIndex = launchPixelIndex(params);
        writeBackgroundGuide(params, pixelIndex, ray_dir, prd->depth, guideCurrentSample(params, pixelIndex));
        prd->aovDone = true;
    }

    float3 radiance = make_float3(0.0f);
    if (params.hasEnvMap)
    {
        const float2 uv = dirToEnvUV(ray_dir, params.envMapRotation);
        const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
        float3 envColor = make_float3(envSample.x, envSample.y, envSample.z);
        envColor *= params.envMapIntensity * params.envMapColorTint;

        if (prd->depth == 0 || prd->specularBounce || !prd->neeDone)
        {
            if (params.hasEnvBackground && prd->depth == 0)
            {
                // The backdrop is what the camera sees; the map above is what lights
                // the scene, and the MIS branch below stays on it because that is the
                // one that was importance sampled.
                const float4 bgSample = tex2D<float4>(params.envBackgroundTexture, uv.x, uv.y);
                envColor = make_float3(bgSample.x, bgSample.y, bgSample.z) * params.envBackgroundIntensity *
                           params.envMapColorTint;
            }
            radiance = prd->throughput * envColor;
        }
        else
        {
            // MIS weight with BSDF sampling vs env map PDF
            const float envPdf =
                envMapPdf(ray_dir, params.envAliasTable, params.envMapWidth, params.envMapHeight, params.envMapRotation);
            const bool hasLocal = params.scene.numLights > 0u || params.scene.numEmissiveMeshes > 0u;
            const float envSelectionPdf = hasLocal ? params.envSelectionPdf : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            const float misWeight = (effectiveEnvPdf > 0.0f) ?
                                        computeMisWeight(prd->lastBsdfPdf, effectiveEnvPdf, params.misHeuristic) :
                                        1.0f;
            radiance = prd->throughput * envColor * misWeight;
        }
    }
    else
    {
        MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
        radiance = prd->throughput * miss_data->bg_color;
    }

    const float localSelectionPdf = params.hasEnvMap ? 1.0f - params.envSelectionPdf : 1.0f;
    const float analyticClassPdf = params.scene.numEmissiveMeshes > 0u ? 1.0f - params.scene.meshLightSelectionPdf : 1.0f;
    for (uint32_t lightId = 0; lightId < params.scene.numLights; ++lightId)
    {
        const UniformLight& light = params.scene.lights[lightId];
        if (!lightIsInfinite(light.type))
        {
            continue;
        }
        if (!analyticLightVisibilityAllowsRay(light.normal.w, prd->depth != 0u))
        {
            continue;
        }
        const float3 axis = -make_float3(light.normal);
        if (light.type == LIGHT_TYPE_DISTANT && distantLightIsDelta(light.halfAngle))
        {
            // A sharp distant and a specular BSDF direction are discrete atoms.
            // Their exact represented match has unit MIS weight; a continuous
            // ray near the axis must not acquire invented angular support.
            if (distantLightDeltaPathMatches(prd->depth, prd->specularBounce, ray_dir, axis))
            {
                radiance += prd->throughput * make_float3(light.color);
            }
            continue;
        }
        const float conditionalPdf = infiniteLightConditionalPdf(light.type, light.halfAngle, ray_dir, axis);
        if (!(conditionalPdf > 0.0f))
        {
            continue;
        }
        const float effectivePdf =
            localSelectionPdf * analyticClassPdf * analyticLightSelectionPdf(light) * conditionalPdf;
        const float misWeight = (prd->depth == 0 || prd->specularBounce || !prd->neeDone || !(effectivePdf > 0.0f)) ?
                                    1.0f :
                                    computeMisWeight(prd->lastBsdfPdf, effectivePdf, params.misHeuristic);
        radiance += prd->throughput * make_float3(light.color) * misWeight;
    }

    prd->radiance += clampIndirectContribution(radiance, prd->depth, params.clampIndirect);

    prd->throughput = make_float3(0.0f);
    prd->depth = params.max_depth;
}

/// What the bounce decided, held until next-event estimation has used the state
/// it replaces.
struct NextBounce
{
    /// Multiplies the throughput -- applied after the estimate, which is owed
    /// the throughput this vertex arrived with.
    float3 weight;
    float3 mediumAlbedo;
    uint32_t medium;
    uint32_t mediumStep;
    /// False when the path stops here. The estimate still runs: the direct
    /// lighting at this vertex does not depend on where the path went next.
    bool alive;
};

template <typename OpenPBRPrepared>
static __forceinline__ __device__ NextBounce sampleNextBounce(PerRayData* prd,
                                                              SurfaceInteraction& si,
                                                              const MaterialParams& matParams,
                                                              const OpenPBRParams& openpbrMat,
                                                              const OpenPBRPrepared& openpbrPrepared,
                                                              const PbrPrepared& pbrPrepared,
                                                              bool isOpenPBR,
                                                              bool isFibre,
                                                              float curveRadius,
                                                              int32_t matId,
                                                              bool entering,
                                                              bool didNee)
{
    NextBounce out;
    out.weight = make_float3(1.0f);
    out.mediumAlbedo = prd->mediumAlbedo;
    out.medium = prd->medium;
    out.mediumStep = prd->mediumStep;
    out.alive = true;

    const float z1 = random<SampleDimension::eBSDF0>(prd->sampler);
    const float z2 = random<SampleDimension::eBSDF1>(prd->sampler);
    const float z3 = random<SampleDimension::eBSDF2>(prd->sampler);
    const float z4 = random<SampleDimension::eBSDF3>(prd->sampler);

    float4 xi = make_float4(z1, z2, z3, z4);
    const uint32_t lobeWord = randomBits<SampleDimension::eBSDF2>(prd->sampler) >> 9u;
    const uint32_t fresnelWord = randomBits<SampleDimension::eBSDF3>(prd->sampler) >> 9u;
    // openpbr_sample takes three uniforms; xi.w is the one standard_pbr spends
    // choosing a lobe and OpenPBR does not need.
    BsdfSampleResult sample_data = isOpenPBR ? openpbr_bsdf_sample(openpbrPrepared, si.wo, xi) :
                                                bsdf_sample(si, xi, lobeWord, fresnelWord, pbrPrepared);

    if (sample_data.event_type == BSDF_EVENT_ABSORB)
    {
        if (prd->depth == 0)
        {
            prd->setFirstEventType(EventType::eAbsorb);
        }
        // Stop on absorb. Whatever next-event estimation delivers below stays:
        // it is this vertex's direct lighting and does not depend on where the
        // path went next.
        out.alive = false;
        return out;
    }
    prd->specularBounce = ((sample_data.event_type & BSDF_EVENT_SPECULAR) != 0);

    if (prd->depth == 0)
    {
        if (sample_data.event_type & BSDF_EVENT_DIFFUSE)
        {
            prd->setFirstEventType(EventType::eDiffuse);
        }
        if (sample_data.event_type & BSDF_EVENT_GLOSSY)
        {
            prd->setFirstEventType(EventType::eSpecular);
        }
    }

    // setup next path segment
    // Face normal oriented toward the incoming ray (wo)
    float3 faceNg = orientedFaceNormal(si.geometry_normal, si.wo);
    float3 sssEntryTint = make_float3(1.0f);
    // A subsurface entry continues along the refracted direction, not the lobe's cosine draw.
    bool sssRefractedEntry = false;
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0 && !isFibre)
    {
        const bool entersMedium = isOpenPBR ? (openpbrMat.subsurface_weight > 0.0f &&
                                               openpbrMat.geometry_thin_walled == 0u)
                                            : (si.subsurface > 0.0f);
        const uint32_t entryEvent = isOpenPBR ? (sample_data.event_type & BSDF_EVENT_TRANSMISSION)
                                              : (sample_data.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION);
        const bool startsSubsurfaceWalk = params.hasSubsurface && entersMedium && entryEvent != 0u;

        if (!si.thin_walled && !(isOpenPBR && startsSubsurfaceWalk))
        {
            if (entering)
            {
                // Report nested-dielectric stack overflow here; the matching pop reports unmatched exits.
                if (params.iorStats != nullptr && prd->iorStack.top >= IOR_STACK_SIZE - 1)
                {
                    atomicAdd(&params.iorStats[IOR_STAT_OVERFLOW], 1u);
                }
                optixIorPush(prd->iorStack, (unsigned int)matId);
            }
            else
            {
                if (params.iorStats != nullptr &&
                    optixIorFind(prd->iorStack, si.dielectric_priority, (unsigned int)matId) < 0)
                {
                    atomicAdd(&params.iorStats[IOR_STAT_UNMATCHED], 1u);
                }
                optixIorPop(prd->iorStack, si.dielectric_priority, (unsigned int)matId);
            }
        }
        prd->origin = offset_ray(si.position, -faceNg);

        if (startsSubsurfaceWalk)
        {
            out.medium = static_cast<uint32_t>(matId) + 1u;
            out.mediumStep = 0u;

            // The walk's albedo, resolved here because this is the last place a
            // texture exists: inside the medium there is no surface to sample.
            float3 walkAlbedo;
            if (isOpenPBR)
            {
                OpenPBRParams walkMat = openpbrMat;
                if (openpbrBaseMapDetailsSubsurface(walkMat))
                {
                    const float3 base = openpbr_color_to_float3(walkMat.base_color);
                    const float3 subsurface = openpbr_color_to_float3(walkMat.subsurface_color) * base;
                    walkMat.subsurface_color = OpenPBRColor{ subsurface.x, subsurface.y, subsurface.z };
                }
                walkAlbedo = openpbr_interior_volume(walkMat).albedo;
            }
            else
            {
                walkAlbedo = matParams.diffuse_transmission_color;
                const float3 reference = matParams.subsurface_reference;
                if (reference.x > 1e-4f && reference.y > 1e-4f && reference.z > 1e-4f)
                {
                    walkAlbedo *= si.albedo / reference;
                }
            }
            out.mediumAlbedo = saturate(walkAlbedo);
            if (!isOpenPBR)
            {
                sssEntryTint = fmaxf(si.diffuse_transmission_color, make_float3(1e-4f));
            }

            sssRefractedEntry = true;
        }
    }
    else
    {
        prd->origin = offset_ray(si.position, faceNg);
    }
    prd->dir = sssRefractedEntry ?
                   subsurface_entry_direction(
                       si.wo, (dot(si.shading_normal, si.wo) > 0.0f) ? si.shading_normal : -si.shading_normal,
                       random<SampleDimension::eSssChannel>(prd->sampler),
                       random<SampleDimension::eSssDistance>(prd->sampler)) :
                   sample_data.wi;

    // The same rejection Cycles makes in subsurface_bounce(): a refracted entry
    // on the viewer's side of the geometric normal never entered anything. See
    // the note at the matching site in wavefront.metal.
    if (sssRefractedEntry && dot(faceNg, prd->dir) >= 0.0f)
    {
        // Stopped the way an absorbed sample is: whatever next-event estimation
        // delivers at this vertex stays, and nothing is carried onward.
        out.alive = false;
        return out;
    }

    if (isFibre)
    {
        prd->origin = fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, normalize(prd->dir));
    }
    const ShadedFrame bounceFrame =
        shadedFrame(si.front_face, dot(si.shading_normal, si.wo), si.transmission, si.diffuse_transmission);
    prd->neeDone = neePairsWithBounce(didNee, neeCrossesSurface(isFibre, si.transmission, si.diffuse_transmission),
                                      bounceFrame.frontFace,
                                      bounceFrame.normalSign * dot(si.shading_normal, prd->dir));
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->misDistance = 0.0f;
    out.weight = sample_data.bsdf_over_pdf / sssEntryTint;
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = prd->specularBounce ? 0.0f : si.roughness;
    }
    return out;
}

enum class RadianceMaterialMode
{
    Dynamic,
    Gltf,
    OpenPBR,
    OpenPBRBase
};

template <RadianceMaterialMode Mode,
          typename OpenPBRPrepared = OpenPBR_PreparedBsdf>
static __forceinline__ __device__ void closestHitRadiance()
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float surfaceT = optixGetRayTmax();

    SurfaceHitData surfaceHit = {};
    const bool isCubicCurve = params.hasCurves && (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    const bool isCurveHit = isCubicCurve || (params.hasCurves && primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR);
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data);
    }
    else if (isCubicCurve)
    {
        surfaceHit = fillCubicCurveGeomData(hit_data);
    }
    else if (params.hasCurves && primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
    {
        surfaceHit = fillLinearCurveGeomData(hit_data);
    }

    // Look up material from device buffer (indexed by materialId)
    const int32_t matId = hit_data->materialId;
    const MaterialParams& matParams = params.materials[matId];
    const cudaTextureObject_t* textures = &params.materialTextures[matId * MAX_MATERIAL_TEXTURES];

    float segment = surfaceT;
    MediumSample medium = {};
    bool insideMedium = false;
    bool mediumIsBounded = false;
    bool sampledFreeFlight = false;
    // Only meaningful while insideMedium; scatterInMedium() needs the mean cosine
    // and the source of it differs between the two material models.
    float mediumAnisotropy = 0.0f;
    if (params.hasSubsurface && prd->medium != 0u)
    {
        const MaterialParams& mm = params.materials[prd->medium - 1u];
        insideMedium = true;
        mediumIsBounded = (mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        const MediumProps props = mediumPropsFor(prd->medium - 1u, prd->mediumAlbedo);
        mediumAnisotropy = props.anisotropy;
        const uint32_t stepCeiling = min(params.subsurfaceIterations, MEDIUM_MAX_STEPS);
        // Past the ceiling the walk stops drawing free flights, so the next
        // surface is its boundary and the path leaves rather than hanging.
        if (prd->mediumStep < stepCeiling)
        {
            SamplerState mrng = mediumSampler(prd->sampler, prd->mediumStep);
            medium = sampleMedium(
                props.sigmaT, props.albedo, prd->throughput, surfaceT,
                random<SampleDimension::eBSDF2>(mrng), random<SampleDimension::eBSDF3>(mrng));
            sampledFreeFlight = true;
            if (medium.scattered)
            {
                segment = medium.t;
            }
        }
    }

    float fogT = 0.0f;
    const bool fogScattered = fogScatters(prd, optixGetWorldRayOrigin(), ray_dir, segment, fogT);
    if (fogScattered)
    {
        segment = fogT;
    }

    if (!insideMedium || mediumIsBounded)
    {
        const unsigned int inside = optixIorMaterial(prd->iorStack);
        if (inside != 0xFFFFFFFFu)
        {
            const MaterialParams& im = params.materials[inside];
            const float3 sigma_t = volume_extinction(im.attenuation_color, im.attenuation_distance, params.volumeModel);
            prd->throughput *= beer_lambert_transmittance(sigma_t, segment);
        }
    }

    if (fogScattered)
    {
        // The surface below this point never happened: the path stopped in the
        // haze short of it, and everything from the material lookup down
        // describes a vertex that only exists if the surface won.
        scatterInFog(prd, optixGetWorldRayOrigin(), ray_dir, fogT);
        return;
    }

    if (medium.scattered)
    {
        scatterInMedium(prd, params.materials[prd->medium - 1u], medium, mediumAnisotropy,
                        optixGetWorldRayOrigin(), ray_dir, mediumIsBounded);
        return;
    }
    if (sampledFreeFlight)
    {
        prd->throughput *= mediumBoundaryWeight(medium, segment);
    }

    if (params.hasSubsurface && (matParams.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
    {
        if (prd->passthrough >= PATH_PASSTHROUGH_MAX)
        {
            prd->throughput = make_float3(0.0f);
            return;
        }
        ++prd->passthrough;

        const uint32_t here = static_cast<uint32_t>(matId) + 1u;
        const bool leaving = (prd->medium == here);
        prd->medium = leaving ? 0u : here;
        prd->mediumStep = 0u;

        // Push past the surface on the side the ray is heading, which needs the
        // sign of the normal and not its direction.
        const float3 exitSide =
            (dot(ray_dir, surfaceHit.geom_normal) > 0.0f) ? surfaceHit.geom_normal : -surfaceHit.geom_normal;
        prd->origin = offset_ray(surfaceHit.position, exitSide);
        prd->dir = ray_dir;
        // The MIS distance keeps counting: as far as the light at the end of
        // this ray is concerned, the scattering vertex is still the one before
        // the boundary.
        prd->misDistance += segment;
        prd->passedThrough = true;
        return;
    }

    if (insideMedium && !mediumIsBounded)
    {
        exitMedium(prd, surfaceHit.position, surfaceHit.normal, surfaceHit.geom_normal, ray_dir);
        return;
    }

    if (params.hasCutout && (params.hasCurves || primType != OPTIX_PRIMITIVE_TYPE_TRIANGLE) &&
        prd->passthrough < PATH_PASSTHROUGH_MAX)
    {
        const float opacity = resolveOpacity(matParams, textures, apply_texture_transform(surfaceHit.uv, matParams));
        if (opacity < 1.0f && opacitySample(prd->sampler, launchPixelIndex(params), prd->passthrough) >= opacity)
        {
            // Step off on the side the ray was travelling, so the next trace
            // cannot re-hit the surface it just passed through.
            const float3 faceNg = (dot(surfaceHit.geom_normal, ray_dir) > 0.0f) ? surfaceHit.geom_normal
                                                                                : -surfaceHit.geom_normal;
            prd->origin = offset_ray(surfaceHit.position, faceNg);
            prd->dir = ray_dir;
            prd->misDistance += segment;
            ++prd->passthrough;
            prd->passedThrough = true;
            return;
        }
    }

    // Fill SurfaceInteraction from hit data, resolving textures, the uv
    // transform, the normal map, vertex colour and coverage on the way.
    SurfaceInteraction si;
    initSurfaceInteraction(si, matParams, textures, surfaceHit.position, surfaceHit.normal, surfaceHit.geom_normal,
                           surfaceHit.worldTangent, surfaceHit.worldBinormal, surfaceHit.uv, ray_dir,
                           surfaceHit.vertexColor);

    const bool isOpenPBR = Mode == RadianceMaterialMode::OpenPBR || Mode == RadianceMaterialMode::OpenPBRBase ? true :
                           Mode == RadianceMaterialMode::Gltf ? false :
                                                               isOpenPBRMaterial(matParams);
    OpenPBRParams openpbrMat = {};
    if (isOpenPBR)
    {
        openpbrMat = params.openpbrParams[matId];
        if (openpbrMat.texture_mask != 0u && params.openpbrTextures != nullptr)
        {
            openpbr_apply_textures(openpbrMat, &params.openpbrTextures[matId * MAX_OPENPBR_TEXTURES], si,
                                   surfaceHit.uv);
        }
    }

    const bool isFibre = isCurveHit && scattersThroughFibre(si) && surfaceHit.curveRadius > 0.0f;
    const float curveRadius = isFibre ? surfaceHit.curveRadius : 0.0f;

    if (prd->writeAov && params.aov != nullptr)
    {
        OpenPBRGuides openpbrGuides;
        if (isOpenPBR)
        {
            openpbrGuides = openpbrDenoiserGuides(openpbr_resolve_inputs(openpbrMat, si), si,
                                                  openpbrBaseMapDetailsSubsurface(openpbrMat));
        }
        writeSurfaceGuide(hit_data, prd, si, surfaceHit.position,
                          primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE, isOpenPBR ? &openpbrGuides : nullptr);
    }

    if (params.debug == (uint32_t)DebugMode::eNormal)
    {
        prd->radiance = (si.geometry_normal + make_float3(1.0f)) * 0.5f;
        return;
    }
    if (params.debug == (uint32_t)DebugMode::eMotionBlur)
    {
        // Red is where in the shutter this path was sampled; green is how far
        // this vertex moved between the two motion keys, so a mesh that deforms
        // and a mesh that only translates look different.
        float3 prevPosition = surfaceHit.position;
        const bool moved =
            (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) && previousTriangleWorldPosition(hit_data, prevPosition);
        prd->radiance = make_float3(
            optixGetRayTime(), moved ? saturate(length(surfaceHit.position - prevPosition) * 10.0f) : 0.0f, 0.0f);
        return;
    }

    if (params.debug == (uint32_t)DebugMode::eSharcGrid)
    {
        const float3 cameraPosition = make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        const unsigned long long key = sharcVoxel(si.position, si.shading_normal, cameraPosition, params.sharcBaseSize,
                                                  /*responsive=*/false);
        prd->radiance = sharcDebugColour(oka::sharc::keyHash(key));
        return;
    }
    if (params.debug == (uint32_t)DebugMode::eSharcRadiance && prd->depth == 0 && params.sharcCapacity != 0u)
    {
        const float3 cameraPosition = make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        const unsigned long long key = sharcVoxel(si.position, si.shading_normal, cameraPosition, params.sharcBaseSize,
                                                  /*responsive=*/false);
        uint32_t slot = 0u;
        // Never inserting: a debug view that populates the table changes the
        // thing it is there to observe.
        if (sharcFind(params.sharcEntries, params.sharcCapacity, key, false, slot))
        {
            float sampleNum = 0.0f;
            float3 cached = sharcRead(params.sharcEntries, slot, sampleNum);
            if (sampleNum > 0.0f)
            {
                // Shown the way a path would read it, responsive part included,
                // or the view would disagree with the render beside it exactly
                // where the responsive signal is the interesting thing.
                cached += sharcReadResponsive(params.sharcEntries, params.sharcCapacity, key);
                params.image[launchPixelIndex(params)] = make_float4(cached, 1.0f);
            }
        }
    }

    // Add emission
    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        float emissionMis = 1.0f;
        if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE && prd->depth != 0u && !prd->specularBounce && prd->neeDone &&
            params.scene.numEmissiveMeshes > 0u)
        {
            const float3 misOrigin = optixGetWorldRayOrigin() - ray_dir * prd->misDistance;
            const float lightPdf =
                emissiveMeshHitPdf(optixGetInstanceId(), 0u, optixGetPrimitiveIndex(), misOrigin, si.position);
            if (lightPdf > 0.0f)
            {
                emissionMis = computeMisWeight(prd->lastBsdfPdf, lightPdf, params.misHeuristic);
            }
        }
        prd->radiance +=
            clampIndirectContribution(prd->throughput * si.emission * emissionMis, prd->depth, params.clampIndirect);
    }

    // Set exterior IOR from the IOR stack for nested dielectrics
    bool entering = si.front_face;
    if (entering)
    {
        si.exterior_ior = optixIorCurrent(prd->iorStack);
    }
    else
    {
        si.exterior_ior = optixIorAfterPop(prd->iorStack, si.dielectric_priority);
    }

    OpenPBRPrepared openpbrPrepared;
    if (isOpenPBR)
    {
        if constexpr (Mode == RadianceMaterialMode::OpenPBRBase)
        {
            openpbrPrepared = openpbr_prepare_base_at(openpbr_base_params(openpbrMat), si, prd->throughput);
        }
        else
        {
            openpbrPrepared = openpbr_prepare_at(openpbrMat, si, prd->throughput);
        }
    }

    const PbrPrepared pbrPrepared = pbr_prepare_for(si);

    // --- Radiance cache ------------------------------------------------------
    // Read only beyond the configured depth and never after a specular arrival or a prior voxel visit.
    // Reads stop after the accumulation cap; non-accumulating frames remain eligible and deposits continue.
    const bool sharcMayRead = params.sharcReadMaxSubframe == 0u || !params.enableAccumulation ||
                              params.subframe_index < params.sharcReadMaxSubframe;

    if (params.sharcCapacity != 0u && params.sharcPath[launchPixelIndex(params)].index == SHARC_NO_ENTRY &&
        prd->depth >= params.sharcDepth && !prd->specularBounce)
    {
        const float3 cameraPosition = make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        const unsigned long long voxelKey =
            sharcVoxel(si.position, si.shading_normal, cameraPosition, params.sharcBaseSize, /*responsive=*/false);

        // Update paths bypass reads and provide cache-independent deposits.
        const bool updatePath = oka::sharc::isUpdatePath(launchPixelIndex(params), prd->sampler.sampleIdx);
        uint32_t slot = 0u;
        // Inserting, because this path is here to fill the slot in; a read that
        // misses simply carries on tracing.
        if (sharcFind(params.sharcEntries, params.sharcCapacity, voxelKey, true, slot))
        {
            // The resolved half, which is a frame behind and made of every
            // deposit the voxel has kept inside its window -- not this frame's
            // partial sums, which other paths are still writing.
            float cachedSamples = 0.0f;
            float3 cached = sharcRead(params.sharcEntries, slot, cachedSamples);
            const float dx = si.position.x - cameraPosition.x;
            const float dy = si.position.y - cameraPosition.y;
            const float dz = si.position.z - cameraPosition.z;
            const float voxelSize =
                oka::sharc::voxelForDistance(sqrtf(dx * dx + dy * dy + dz * dz), params.sharcBaseSize).size;
            const float segmentLength = optixGetRayTmax() + prd->misDistance;
            const bool eligible = oka::sharc::mayReadCache(
                segmentLength, params.sharcPath[launchPixelIndex(params)].launchRoughness, voxelSize);

            if (sharcMayRead && eligible && !updatePath && cachedSamples >= (float)params.sharcMinSamples)
            {
                if (params.sharcResponsive != 0u)
                {
                    cached += sharcReadResponsive(params.sharcEntries, params.sharcCapacity, voxelKey);
                }
                // The rest of this path is what the cache already knows.
                prd->radiance += prd->throughput * cached;
                prd->throughput = make_float3(0.0f);
                prd->depth = params.max_depth; // the raygen loop stops here
                return;
            }
            if (luminance(prd->throughput) > oka::sharc::kMinRecordThroughput)
            {
                const float floorT = oka::sharc::kThroughputFloor;
                SharcPathState visit;
                visit.index = slot;
                visit.radianceAtVisit = prd->radiance;
                visit.responsiveRadiance = make_float3(0.0f);
                visit.responsiveIndex = SHARC_NO_ENTRY;
                if (params.sharcResponsive != 0u)
                {
                    uint32_t responsiveSlot = 0u;
                    if (sharcFind(params.sharcEntries, params.sharcCapacity, oka::sharc::responsiveKey(voxelKey), true,
                                  responsiveSlot))
                    {
                        visit.responsiveIndex = responsiveSlot;
                    }
                }
                visit.invThroughput =
                    make_float3(1.0f / fmaxf(prd->throughput.x, floorT), 1.0f / fmaxf(prd->throughput.y, floorT),
                                1.0f / fmaxf(prd->throughput.z, floorT));
                params.sharcPath[launchPixelIndex(params)] = visit;
            }
        }
    }

    const bool didNee = neeRunsAtVertex(
        params.estimatorMode == 0, params.scene.numLights > 0 || params.scene.numEmissiveMeshes > 0 || params.hasEnvMap,
        isOpenPBR ? openpbr_has_smooth_lobe(openpbrMat) : bsdf_has_smooth_lobe(si, pbrPrepared));
    const NextBounce bounce = sampleNextBounce(prd, si, matParams, openpbrMat, openpbrPrepared, pbrPrepared,
                                               isOpenPBR, isFibre, curveRadius, matId, entering, didNee);

    const float3 throughputAtVertex = prd->throughput;
    const uint32_t mediumAtVertex = prd->medium;
    prd->throughput *= bounce.weight;
    prd->medium = bounce.medium;
    prd->mediumStep = bounce.mediumStep;
    prd->mediumAlbedo = bounce.mediumAlbedo;

    if (didNee)
    {
        bool responsiveLight = false;
        const float3 direct = estimateDirectLighting(prd, si, curveRadius,
                                                     isOpenPBR ? &openpbrPrepared : nullptr, pbrPrepared,
                                                     throughputAtVertex, mediumAtVertex, responsiveLight);
        prd->radiance += direct;
        if (params.sharcResponsive != 0u && responsiveLight)
        {
            SharcPathState& visit = params.sharcPath[launchPixelIndex(params)];
            if (visit.index != SHARC_NO_ENTRY)
            {
                visit.responsiveRadiance += direct;
            }
        }
        if (prd->throughput.x == 0.0f && prd->throughput.y == 0.0f && prd->throughput.z == 0.0f)
        {
            // estimateDirectLighting() found a NaN and painted the pixel.
            return;
        }
    }

    if (!bounce.alive)
    {
        prd->throughput = make_float3(0.0f);
    }

}

extern "C" __global__ void __closesthit__radiance()
{
    closestHitRadiance<RadianceMaterialMode::Dynamic>();
}

extern "C" __global__ void __closesthit__radiance_gltf()
{
    closestHitRadiance<RadianceMaterialMode::Gltf>();
}

extern "C" __global__ void __closesthit__radiance_openpbr()
{
    closestHitRadiance<RadianceMaterialMode::OpenPBR>();
}

extern "C" __global__ void __closesthit__radiance_openpbr_base()
{
    closestHitRadiance<RadianceMaterialMode::OpenPBRBase, OpenPBR_BasePreparedBsdf>();
}
