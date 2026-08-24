#include <optix.h>

#include <cuda.h>

#include <OptixRenderParams.h>
#include <cuda_helpers/helpers.h>
#include <cuda_helpers/curve.h>

#include <random.h>

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <sutil/vec_math_adv.h>

#include <lights.h>
#include <env_light.h>

#include <strelka/material/bsdf.h>
#include <strelka/material/volume.h>

#include <postprocessing/Guides.h>

#include "optix_device_utils.h"
#include "shading/shading_common.h"
#include "shading/medium.h"
#include "alpha.h"
#include "fog.h"
#include <curve_layout.h>

extern "C"
{
    __constant__ Params params;
}

/// Fraction of the light that survives the segment: 1 when nothing is in the
/// way, 0 when an opaque surface is, and the product of (1 - opacity) over the
/// cutout surfaces crossed otherwise.
///
/// The payload is one word holding that fraction as a float. It starts at 1;
/// `__anyhit__occlusion` multiplies it down and ignores the intersection so
/// traversal carries on, and only an opaque hit is accepted -- which with
/// TERMINATE_ON_FIRST_HIT ends the ray and lets `__closesthit__occlusion` write
/// the zero.
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

    // The atmosphere dims every shadow ray, including the ones that reach the
    // light unobstructed. Without this a shadow ray is a hole in the haze and
    // every light reads as if the medium were not there -- which is the same
    // note Metal's shadow kernel carries, in the same place.
    if (visible > 0.0f && params.hasFog)
    {
        const float tau = fogOpticalDepth(ray_origin, ray_direction, tmax, params.fogHeight,
                                          params.fogSigmaT);
        if (tau > 0.0f)
        {
            visible *= expf(-tau);
        }
    }
    return visible;
}

/// Optical depth a shadow ray picks up crossing the boundaries of bounded media.
///
/// The boundary of a medium is not on the shadow mask -- a fog gizmo left there
/// would black out everything it encloses -- so the segments inside it have to
/// be found with a traversal of their own. That is the second traversal this
/// feature costs, which is why it is gated on the scene having a bounded medium
/// at all and why it walks a bounded number of crossings rather than to
/// completion.
///
/// Alternating closest hits rather than an any-hit sweep: a convex volume
/// answers in two, and the alternative is a payload that sorts an unbounded set
/// of distances. `startMedium` is the one thing the ray cannot work out for
/// itself -- whether it began inside. A vertex within a fog volume and one just
/// outside it produce the same origin and direction.
///
/// optixTraverse without optixInvoke, so this costs a traversal and not a
/// program launch: the hit object carries the distance and the SBT record, and
/// those are the whole of what is wanted.
static __forceinline__ __device__ float3 mediumTransmittance(
    float3 origin, float3 direction, float maxDistance, uint32_t startMedium)
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

        optixTraverse(params.handle, origin + direction * travelled, direction,
                      1e-4f, remaining, time,
                      OptixVisibilityMask(GEOMETRY_MASK_MEDIUM), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                      RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION);

        const bool escaped = !optixHitObjectIsHit();
        const float segment = escaped ? remaining : optixHitObjectGetRayTmax();

        if (medium != 0u)
        {
            const MaterialParams& mm = params.materials[medium - 1u];
            optical += fromSpectrum(oka::medium::sigmaTFromRadius(toSpectrum(mm.subsurface_radius))) *
                       segment;
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
    const MaterialParams& material = params.materials[matId];
    if (material.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return; // accepted; TERMINATE_ON_FIRST_HIT ends the ray here
    }

    const unsigned int primitiveId = optixGetPrimitiveIndex();
    const uint32_t i0 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 0];
    const uint32_t i1 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 1];
    const uint32_t i2 = params.scene.ib[hit_data->indexOffset + primitiveId * 3 + 2];
    const uint32_t baseVbOffset = hit_data->vertexOffset;
    const float2 uv = interpolateAttrib(unpackUV(params.scene.vb[baseVbOffset + i0].uv),
                                        unpackUV(params.scene.vb[baseVbOffset + i1].uv),
                                        unpackUV(params.scene.vb[baseVbOffset + i2].uv),
                                        optixGetTriangleBarycentrics());

    const cudaTextureObject_t* textures = &params.materialTextures[matId * MAX_MATERIAL_TEXTURES];
    const float opacity = resolveOpacity(material, textures, uv);

    const float transmittance = __uint_as_float(optixGetPayload_0()) * (1.0f - opacity);
    if (transmittance <= SHADOW_TRANSMITTANCE_CUTOFF)
    {
        return; // nothing measurable gets through; accept and stop traversing
    }
    optixSetPayload_0(__float_as_uint(transmittance));
    optixIgnoreIntersection();
}

/// Uniform light choice from a canonical sample, clamped to the last light.
///
/// The clamp is not defensive noise: u is drawn from [0, 1) but the env-map branch
/// feeds it u*2 from a u already known to be below 0.5, and that product rounds to
/// exactly 1.0f for the largest such u. Without the clamp that one sample indexes
/// one past the end of the light buffer.
static __forceinline__ __device__ uint32_t selectLightIndex(float u, uint32_t numLights)
{
    const uint32_t index = (uint32_t)(numLights * u);
    return (index < numLights) ? index : (numLights - 1);
}

/// One proposed connection to a light, before visibility is known.
///
/// Separating the proposal from the shadow ray is what makes resampling possible:
/// several candidates can be drawn and weighted by what they would contribute,
/// and only the survivor costs a ray. It also fixes an ordering problem the old
/// code had -- it traced occlusion inside the light sampler, so a candidate that
/// loses the resampling draw would still have paid for a ray.
struct LightConnection
{
    float3 radiance; // Li times the shading cosine, unshadowed
    float3 toLight;
    float pdf; // solid-angle density, including the light-selection probability
    float tMax;
    bool needsRay;
    /// A sharp point or spot cannot be hit by a BSDF ray, so no other strategy
    /// can produce this direction and its MIS weight is exactly one. Weighting it
    /// against the BSDF pdf -- which is what this code used to do -- discards the
    /// share of the light the BSDF strategy is credited with and never delivers.
    bool isDelta;
    /// The light behind this connection is marked responsive, so whatever it
    /// delivers is cached in the short-window half of the voxel rather than the
    /// long-window one. Always false for the environment: a dome is the one
    /// emitter that cannot be swung around or switched on mid-shot, and giving
    /// it a responsive entry would put most of an outdoor scene's light on the
    /// short clock for nothing.
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
    c.pdf = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.isDelta = false;
    c.isResponsive = false;
    return c;
}

/// Where a next-event connection leaves from.
///
/// A surface offsets along the face the shadow ray actually departs through. The
/// raw geometry normal points to a fixed side of the triangle, so on a back-face
/// hit it would push the origin *into* the surface and the ray would immediately
/// hit the geometry it started on -- next-event estimation then reports occlusion
/// the BSDF strategy does not see, and the two halves of the MIS estimate stop
/// summing to the integral. The bounce ray orients its offset the same way.
///
/// A fibre has no such face: the Chiang lobe has already paid for the crossing,
/// so a connection leaving through the strand has to start past it or the fibre
/// occludes itself and the dominant lobe is never connected to at all.
static __forceinline__ __device__ float3 shadowOrigin(const SurfaceInteraction& si,
                                                      float curveRadius,
                                                      float3 toLight)
{
    if (scattersThroughFibre(si) && curveRadius > 0.0f)
    {
        return fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius, toLight);
    }
    return offset_ray(si.position, orientedFaceNormal(si.geometry_normal, toLight));
}

/// What a projector emits in a direction, as a multiplier on its intensity.
///
/// The image it throws, faded at the frame's edge, and black outside the frame,
/// so the caller multiplies unconditionally the way it does with an IES table.
/// A projector with no image throws a plain white rectangle -- a usable light,
/// and what an image that failed to load degrades to rather than darkness.
///
/// Here rather than in common/lights.h because tex2D is a device intrinsic and
/// that header is also compiled by the OptiX backend's host translation units.
/// The frame maths it does share; only this fetch is CUDA's.
static __forceinline__ __device__ float3 projectorEmission(const UniformLight& light, const float3 dirFromLight)
{
    const ProjectorSample p = projectorSampleForLight(light, dirFromLight);
    if (!p.inside)
    {
        return make_float3(0.0f);
    }
    const int slot = projectorImageIndex(light);
    if (slot < 0 || params.scene.projectorTextures == nullptr)
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

static __device__ LightConnection connectLight(SamplerState& sampler,
                                               const UniformLight& light,
                                               const SurfaceInteraction& si,
                                               float curveRadius,
                                               // A scattering event in a medium has a position and
                                               // no normal. The hemisphere test and the cosine below
                                               // are surface terms; applied to a volume they reject
                                               // half of every connection and darken the other half.
                                               bool volumeEvent = false)
{
    LightSampleData lightSampleData = {};
    const float2 uv =
        make_float2(random<SampleDimension::eLightPointX>(sampler), random<SampleDimension::eLightPointY>(sampler));
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
        lightSampleData = SampleDistantLight(light, uv, si.position);
        break;
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
    // Every point and spot, whatever its radius -- see lightIsDeltaForMis(). The
    // radius used to exempt a soft one from this on the theory that a BSDF ray
    // could hit the sphere it is sampled as, but createInstance() gives a point
    // or spot proxy a zero visibility mask, so no ray can. The MIS weight then
    // deducted a share for a strategy that is switched off and never delivered
    // it.
    c.isDelta = lightIsDeltaForMis(light.type);

    float3 Li = make_float3(light.color);
    if (lightIsPunctual(light.type))
    {
        const float dist = fmaxf(lightSampleData.distToLight, 1e-4f);
        // Colour is radiant intensity either way, but the two cases turn it into
        // what the surface receives differently. A sharp light is a point and
        // carries the inverse-square law. A soft one is sampled as a sphere, and
        // its solid-angle density already holds the d^2, so applying the falloff
        // as well counts the distance twice -- what the sphere needs is the
        // radiance a uniform emitter of that intensity has, I / (pi r^2). With
        // the falloff kept and the density wrong, a lamp jumped by 4 pi the
        // moment its radius crossed the softness threshold.
        const bool soft = punctualLightIsSoft(light.points[0].x);
        Li *= rangeWindow(light, dist) *
              (soft ? sphereRadianceFromIntensity(light.points[0].x) : (1.0f / (dist * dist)));
        // IES replaces the isotropic (and, for spots, the cone) angular shape:
        // the table is the whole distribution, and the light's own intensity is
        // a multiplier on top of it. No profile means the cone alone, as before.
        // Applying the cone as well would count the luminaire's aperture twice,
        // once from the file and once from the sidecar's outer angle.
        const bool hasIes = light.points[0].y >= 0.0f;
        if (light.type == LIGHT_TYPE_PROJECTOR)
        {
            // The image *is* the profile, so it replaces the cone exactly as an
            // IES table would -- and a projector never carries both, which
            // Scene::updateLight enforces by writing -1 into the IES slot.
            Li *= projectorEmission(light, -lightSampleData.L);
        }
        else if (hasIes)
        {
            Li *= sampleIesCandela(params.scene.iesProfiles, light, -lightSampleData.L);
        }
        else if (light.type == LIGHT_TYPE_SPOT)
        {
            Li *= spotAttenuation(light, -lightSampleData.L);
        }
    }

    // Blender's controlled falloff. Self-gated to area lights, so punctual
    // lights (whose pad1 is the KHR range) are untouched.
    Li *= areaFalloff(light, lightSampleData.distToLight);

    // lightReachesShadingPoint() is `dot(N, L) > 0` for everything except a
    // fibre, where the hemisphere test is the wrong question -- see the note on
    // it in shading/shading_common.h.
    const bool lit = volumeEvent || lightReachesShadingPoint(si, lightSampleData.L);
    const bool facing = lightIsPunctual(light.type) ?
                            (lit && emitsLight(Li)) :
                            (lit && -dot(lightSampleData.L, lightSampleData.normal) > 0.0f && emitsLight(Li));
    if (facing)
    {
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h. shadingCosine() is
        // saturate(dot(N, L)) for everything but a fibre -- and nothing at all
        // for a medium, which has no normal to take it against.
        c.radiance = volumeEvent ? Li : Li * shadingCosine(si, lightSampleData.L);
        c.pdf = lightSampleData.pdf;
        c.tMax = lightSampleData.distToLight;
        c.needsRay = true;
    }
    return c;
}

static __device__ LightConnection connectEnvLight(SamplerState& sampler,
                                                  const SurfaceInteraction& si,
                                                  float curveRadius,
                                                  bool volumeEvent = false)
{
    const float2 xi = make_float2(
        random<SampleDimension::eLightPointX>(sampler),
        random<SampleDimension::eLightPointY>(sampler));

    float envPdf = 0.0f;
    const float3 dir = sampleEnvMap(xi,
                                    params.envAliasTable, params.envMapTexturePoint,
                                    params.envMapWidth, params.envMapHeight,
                                    params.envMapRotation, params.envPdfScale,
                                    envPdf);

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

/// Choose a strategy and build the connection. Visibility is the caller's job.
static __device__ LightConnection connectToLight(SamplerState& sampler,
                                                 const SurfaceInteraction& si,
                                                 float curveRadius,
                                                 bool volumeEvent = false)
{
    if (params.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(sampler);

        if (params.scene.numLights == 0 || u >= 0.5f)
        {
            // Sample environment map
            const float selectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            LightConnection c = connectEnvLight(sampler, si, curveRadius, volumeEvent);
            c.pdf *= selectionPdf;
            c.isResponsive = false;
            return c;
        }
        // Sample local light (remap u from [0, 0.5) to [0, 1))
        const uint32_t lightId = selectLightIndex(u * 2.0f, params.scene.numLights);
        LightConnection c =
            connectLight(sampler, params.scene.lights[lightId], si, curveRadius, volumeEvent);
        c.pdf *= 0.5f / params.scene.numLights;
        c.isResponsive = isResponsiveLight(lightId);
        return c;
    }

    // A scene with neither an environment nor an analytic light has nothing to
    // connect to. This used to fall through and divide by numLights == 0, then
    // read lights[0] off a null device pointer -- mLightBuffer is constructed
    // empty, so the pointer really is null rather than merely unpopulated.
    if (params.scene.numLights == 0)
    {
        return makeEmptyConnection();
    }

    const float u = random<SampleDimension::eLightId>(sampler);
    const uint32_t lightId = selectLightIndex(u, params.scene.numLights);
    LightConnection c =
        connectLight(sampler, params.scene.lights[lightId], si, curveRadius, volumeEvent);
    c.pdf *= 1.0f / params.scene.numLights;
    c.isResponsive = isResponsiveLight(lightId);
    return c;
}

/// Next-event estimation, with resampled importance sampling over the candidates.
///
/// Draw M candidates from the light-sampling density, weight each by what it
/// would actually contribute -- BSDF, cosine and MIS weight included, none of
/// which the light's own density knows about -- and keep one. The shadow ray
/// count does not change: one candidate survives and one ray is traced.
///
/// The target is the luminance of the *unshadowed* contribution with the MIS
/// weight already folded in. Folding it in is what keeps this unbiased against
/// the BSDF strategy: the two weights still sum to one in every direction, so
/// resampling only improves the next-event half and leaves the other alone.
/// Resampling cannot help with visibility, by construction -- the target does not
/// know it.
///
/// The default of one candidate reduces every line below to plain next-event
/// estimation, bit for bit: the first candidate uses the sampler unmodified.
///
/// Returns the radiance to add at this vertex, already multiplied by throughput
/// and clamped, or zero.
/// `outResponsive` reports whether the survivor's light is one the scene marked
/// responsive, so the caller can route what it delivered to the short-window
/// half of the voxel. Resampling keeps exactly one candidate, so the answer is a
/// single flag rather than a split of the returned contribution.
static __device__ float3 estimateDirectLighting(PerRayData* prd,
                                                const SurfaceInteraction& si,
                                                float curveRadius,
                                                bool& outResponsive)
{
    outResponsive = false;
    const uint32_t candidates = max(params.risCandidates, 1u);
    // A fibre has no back side to reject: the Chiang lobe's TT term is light that
    // entered one side and left the other, and on a bright groom it is four fifths
    // of the albedo. The caller hands over a radius only for a strand actually
    // shaded that way, so this is the same gate it applies to the bounce ray.
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
        // The set of directions this half of the estimate is willing to offer.
        // Stated once, in common/nee_pairing.h, because the bounce ray at the
        // bottom of __closesthit__radiance has to deduct a MIS share against
        // exactly this set and no other.
        // Against the frame the BSDF shaded in. An opaque back hit is flipped
        // before the lobes see it (shading_frame.h), so the hemisphere it
        // scatters into is the one below the raw shading normal, and both halves
        // of the estimate have to be told the same thing about that.
        const ShadedFrame frame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo),
                                              si.transmission, si.diffuse_transmission);
        const bool isNextEventValid =
            neeProposesDirection(isFibre, frame.frontFace,
                                 frame.normalSign * dot(si.shading_normal, conn.toLight)) &&
            (conn.pdf > 0.0f);
        if (!isNextEventValid || !conn.needsRay)
        {
            continue;
        }

        const BsdfEvalResult evalData = bsdf_eval(si, conn.toLight);
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

        const float misWeight =
            conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, evalData.pdf, params.misHeuristic);
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
    const float3 weight = prd->throughput * bestF * W;
    if (weight.x == 0.0f && weight.y == 0.0f && weight.z == 0.0f)
    {
        return make_float3(0.0f);
    }

    // Only the survivor pays for a ray. shadowOrigin() picks where it departs:
    // the oriented face offset for a surface -- this is the same fix Metal's
    // connectEnvLight carries -- and the far side of the strand for a fibre.
    // A float, not a bool: with the any-hit alpha test in place a shadow ray comes
    // back with the product of the transmittances it crossed, so a connection
    // through a cutout leaf or a blended pane is dimmed rather than being all or
    // nothing. Reading it as a bool inverts the test outright -- an unoccluded ray
    // returns 1.0, which is true -- and discards every next-event connection.
    const float transmittance =
        traceOcclusion(params.handle, shadowOrigin(si, curveRadius, bestConn.toLight), bestConn.toLight,
                       params.shadowRayTmin, bestConn.tMax);
    if (transmittance <= 0.0f)
    {
        return make_float3(0.0f);
    }
    // Whatever survived the geometry still has to cross whatever bounded media
    // stand between this vertex and the light. Without it a shadow ray is a hole
    // in the fog, and every light reads as if the volume were not there -- which
    // is the term that made 18_bounded_volume 1.9x too bright on the backend
    // that had it first.
    float3 survived = make_float3(transmittance);
    if (params.hasBoundedMedium)
    {
        survived *= mediumTransmittance(shadowOrigin(si, curveRadius, bestConn.toLight),
                                        bestConn.toLight, bestConn.tMax, prd->medium);
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

// (normalCubic() used to sit here: a second, unreferenced copy of what
// fillCubicCurveGeomData() does. Removed rather than left as a warning, because
// the next person adding a curve basis would have had two places to change and
// only one of them would have mattered.)

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

/// glTF COLOR_0 out of oka::Scene::Vertex.
///
/// The device-side `Vertex` in OptixRenderParams.h spells the last eight bytes
/// `pad0` / `pad1`, but the buffer it aliases is oka::Scene::Vertex, whose last
/// two words are `uv1` and `color`. Both structs are 32 bytes with the same
/// field offsets, and OptiXRender::createVertexBuffer uploads the scene struct
/// whole, so the colour really is there -- it is only spelled as padding.
/// Renaming those two fields is a hand-off; reading the bits is not.
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
    // safe_normalize, not normalize: a zero/near-zero-area triangle (a thin
    // cutout leaf/needle card collapsed by an exporter or LOD) makes cross()
    // return (0,0,0), and normalize((0,0,0)) is NaN -- which the eNormal
    // debug view writes straight to the display buffer with no guard, as
    // isolated black dots on foliage.
    float3 geomNormal = cross(p1 - p0, p2 - p0);
    geomNormal = safe_normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(interpolateAttrib(t0, t1, t2, barycentrics)));
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

/// Where along its strand a curve hit landed, as a uv.
///
/// Segments of one set are laid out strand after strand, so with a uniform
/// segment count the index modulo that count is the segment's position within
/// its strand and the curve parameter interpolates inside it. A set whose
/// strands differ in length carries 0 and gets the strand root, which is what a
/// root-to-tip ramp reads as "no gradient" rather than as garbage.
///
/// The second coordinate is 0: a strand is a fibre, not a sheet, and there is
/// no meaningful coordinate around it. This is the same (alongStrand, 0) Metal
/// hands its curve hits.
static __forceinline__ __device__ float2 curveStrandUV(const HitGroupData* hit_data,
                                                       unsigned int primitiveIndex,
                                                       float u)
{
    return make_float2(
        oka::curve_layout::strandCoordinate(primitiveIndex, hit_data->curveSegmentsPerStrand, u), 0.0f);
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
    // safe_normalize: a tapered strand tip (radius -> 0) or a near-axial ray
    // collapses surfaceNormal()'s radial vector toward (0,0,0), and
    // normalize((0,0,0)) is NaN -- exactly the failure mode a groom hits far
    // more often than a triangle mesh does.
    float3 worldNormal = safe_normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(curveTangent(interpolator, u)));
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
    res.curveRadius =
        length(optixTransformVectorFromObjectToWorldSpace(objectNormal * interpolator.radius(u)));

    return res;
}

/// Round linear curves: two control points, a cylinder with spherical caps.
///
/// This arm used to be unreachable. `createCurve` hardcoded degree 3, so a
/// linear sidecar -- which is what every particle groom in the tree writes, and
/// what `28_hair` is -- was built and fetched as a cubic B-spline over the same
/// points. A B-spline does not interpolate its control points, so the strands
/// were built along a curve that ran inside the one the sidecar described, three
/// control points short at every strand.
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
    const float3 worldTangent =
        normalize(optixTransformNormalFromObjectToWorldSpace(curveTangent(interpolator, u)));
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
    // The chord across the strand is measured in world units, and the
    // interpolator's radius is in object ones. Carrying the radial *vector*
    // through the transform rather than the scalar is what makes that survive an
    // instance transform with a scale on it -- and the object normal is already
    // the radial direction everywhere except the two flat endcaps, where the
    // chord degenerates to zero and fibre_exit() falls back to a surface offset.
    res.curveRadius =
        length(optixTransformVectorFromObjectToWorldSpace(objectNormal * interpolator.radius(u)));

    return res;
}

/// Where this triangle hit was in the world one frame ago.
///
/// `vb_prev` is the previous frame's vertex buffer -- the same data motion blur
/// uses as its first key -- so a deforming or skinned mesh reprojects correctly.
/// What it does *not* carry is a rigid instance transform that moved between
/// frames: the previous frame's instance matrices are not on the device, so the
/// previous object-space position is put through the current transform. Camera
/// motion is exact either way, and camera motion is what a still scene's guides
/// are made of. Returns false when there is nothing better than "it did not
/// move" to say.
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
    const float3 objectPos = interpolateAttrib(params.scene.vb_prev[base + i0].position,
                                               params.scene.vb_prev[base + i1].position,
                                               params.scene.vb_prev[base + i2].position, barycentrics);
    outPosition = optixTransformPointFromObjectToWorldSpace(objectPos);
    return true;
}

/// Fill in the pixel's guide record from the surface being shaded.
///
/// Depth and motion are written first and separately, because unlike albedo or
/// roughness they belong to the pixel rather than to whatever surface the
/// material guides were eventually taken from. A specular primary hit hands its
/// material guides to the surface it reflects, and that surface sits somewhere
/// else on screen -- reprojecting the pixel by *its* motion is not a small
/// error.
static __forceinline__ __device__ void writeSurfaceGuide(const HitGroupData* hit_data,
                                                         PerRayData* prd,
                                                         const SurfaceInteraction& si,
                                                         const float3 worldPosition,
                                                         const bool isTriangle)
{
    const uint32_t pixelIndex = launchPixelIndex(params);
    if (prd->depth == 0)
    {
        float3 prevPosition = worldPosition;
        if (isTriangle)
        {
            previousTriangleWorldPosition(hit_data, prevPosition);
        }
        const float2 motion = guideScreenMotion(params, make_float4(prevPosition, 1.0f), prd->pixelSample);
        params.aov[pixelIndex].depth = guideViewDepth(params, worldPosition);
        params.aov[pixelIndex].motionX = motion.x;
        params.aov[pixelIndex].motionY = motion.y;
    }

    if (oka::guides::shouldWriteGuide(true, prd->aovDone, params.guidePrimaryHit, prd->depth, si.roughness))
    {
        const AovSample previous = params.aov[pixelIndex];
        AovSample a;
        // Metals put their colour in the specular lobe and have no diffuse one.
        //
        // Clamped because an albedo guide is a reflectance and the denoiser reads
        // it as one: it divides the colour through by this and multiplies back
        // afterwards, so a value above 1 tells it a surface returns more light
        // than fell on it and it under-filters that pixel. si.albedo is the raw
        // base colour and an emissive material can carry any magnitude there --
        // measured up to 25.5 on 20_mirror_and_floor, 1.8% of the frame.
        const float3 base = saturate(si.albedo);
        a.diffuseAlbedo = base * (1.0f - si.metallic);
        a.specularAlbedo = lerp(make_float3(0.04f), base, si.metallic);
        a.normal = si.shading_normal;
        a.roughness = si.roughness;
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

    // What the specular lobe of the primary hit is looking at. A denoiser takes
    // this separately so it can reproject a reflection at the depth of the thing
    // being reflected rather than at the mirror's own. `specularBounce` still
    // describes the previous bounce here -- this program has not overwritten it
    // yet -- which is exactly the question being asked.
    if (prd->depth == 1 && prd->specularBounce)
    {
        params.aov[pixelIndex].specularHitDistance = optixGetRayTmax();
    }
}

/// A scattering event inside a participating medium.
///
/// Shaped like a surface vertex and different from one in two ways: there is no
/// normal, so the phase function stands in for the BSDF and is its own density;
/// and next-event estimation only runs for a bounded volume. A bounded volume is
/// the one kind of medium worth connecting to a light from -- it is thin, it is
/// lit from outside, and the shafts and the glow are single scattering. A
/// subsurface walk gets neither: its boundary occludes almost every shadow ray
/// it would spawn, so the cost is real and the contribution is not. Cycles'
/// random walk makes the same call, and light still gets in and out through the
/// surface, where next-event estimation does run.
static __device__ void scatterInMedium(PerRayData* prd,
                                       const MaterialParams& mm,
                                       const MediumSample& m,
                                       const float3 rayOrigin,
                                       const float3 rayDir,
                                       const bool isBounded)
{
    prd->throughput *= mediumScatterWeight(m, m.t);
    const float3 scatterPoint = rayOrigin + rayDir * m.t;
    SamplerState mrng = mediumSampler(prd->sampler, prd->mediumStep + 1u);

    // Whether this vertex made a next-event estimate at all -- the question the
    // miss and light-hit programs need answered, and not the same question as
    // whether the estimate came back with anything. A vertex that drew a light
    // sample pointing the wrong way still had the light strategy available for
    // the direction the phase function eventually took, so the MIS weight there
    // is still owed. Deciding it from the outcome instead hands the bounce ray
    // the whole contribution on exactly the draws where the connection failed,
    // and the two strategies stop summing to one: it put the white-furnace
    // sphere 15% over unity, uniformly at every mean free path.
    bool didNee = volumeNeePairsWithBounce(params.estimatorMode == 0,
                                          params.scene.numLights > 0 || params.hasEnvMap);
    if (isBounded)
    {
        // Volumetric emission: what makes bath water glow rather than merely
        // tint what is behind it. Clamped like every other contribution --
        // emission reached through a glass or specular chain arrives with a
        // throughput well above one, and adding that unclamped puts fireflies
        // over the whole frame.
        const float3 Le = mm.medium_emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f)
        {
            prd->radiance +=
                clampIndirectContribution(prd->throughput * Le, prd->depth, params.clampIndirect);
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
                // dot(rayDir, toLight), not dot(-rayDir, toLight). The phase
                // function takes the angle between the two directions of
                // *travel*: light arrives along -toLight and leaves toward the
                // camera along -rayDir, so their cosine is dot(rayDir, toLight).
                // Negated, a forward-scattering medium becomes a backward-
                // scattering one.
                const float phase =
                    hgPhaseFunction(dot(rayDir, conn.toLight), mm.subsurface_anisotropy);
                // The phase function is the medium's BSDF and its own pdf, so MIS
                // pairs it against the light density exactly as a surface lobe
                // would.
                const float misWeight =
                    conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, params.misHeuristic);
                const float3 weight =
                    prd->throughput * (conn.radiance / conn.pdf) * misWeight * phase;
                if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
                {
                    const float visible = traceOcclusion(params.handle, scatterPoint, conn.toLight,
                                                         params.shadowRayTmin, conn.tMax);
                    if (visible > 0.0f)
                    {
                        // Starts inside this medium, so the optical depth begins
                        // accumulating immediately rather than at the first
                        // boundary the ray crosses on the way out.
                        float3 survived = make_float3(visible);
                        if (params.hasBoundedMedium)
                        {
                            survived *= mediumTransmittance(scatterPoint, conn.toLight, conn.tMax,
                                                            prd->medium);
                        }
                        prd->radiance += clampIndirectContribution(weight * survived, prd->depth,
                                                                   params.clampIndirect);
                    }
                }
            }
        }
    }

    float phasePdf = 0.0f;
    const float3 nextDir =
        hgSampleDirection(-rayDir, mm.subsurface_anisotropy, random<SampleDimension::eBSDF0>(mrng),
                          random<SampleDimension::eBSDF1>(mrng), phasePdf);

    prd->origin = scatterPoint;
    prd->dir = nextDir;
    prd->lastBsdfPdf = phasePdf;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    // A phase function, or a cosine lobe off a subsurface exit: both are as
    // broad as a lobe gets, so the cache's eligibility test should treat the
    // next segment as launched from a fully rough surface. Without this the
    // segment inherits whatever surface sent the ray into the medium -- for
    // water or glass that is a delta transmission, roughness zero, and the test
    // then refuses the cache for every vertex inside the medium.
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    ++prd->mediumStep;

    if (isBounded)
    {
        // A bounded volume's scattering event is a bounce like any other: the
        // raygen loop charges it a depth, rolls the dice on it and advances the
        // sampler. A medium with no depth budget of its own is a path that
        // wanders forever.
        return;
    }

    // A subsurface walk keeps its depth -- the whole walk is one scattering
    // event as far as the path budget is concerned, and charging it per step
    // would make a translucent object go black at any sane max_depth -- so it
    // reuses the pass-through exit, which is the one that does not spend a
    // bounce, and rolls its own roulette on what the walk has left. The step
    // ceiling is a backstop for a medium dense enough that roulette alone would
    // take thousands of steps to end; this is what actually terminates the walk.
    const float survive = clamp(
        fmaxf(prd->throughput.x, fmaxf(prd->throughput.y, prd->throughput.z)), 0.05f, 1.0f);
    if (random<SampleDimension::eRussianRoulette>(mrng) >= survive)
    {
        prd->throughput = make_float3(0.0f);
        prd->depth = params.max_depth;
        return;
    }
    prd->throughput /= survive;
    prd->passedThrough = true;
}

/// An atmospheric scattering event: the segment the ray was travelling ended in
/// the haze rather than on the surface it was heading for.
///
/// Shaped like scatterInMedium's bounded branch and kept separate from it,
/// because the atmosphere has no material behind it -- no boundary to have been
/// entered through, no extinction to look up, no emission -- and threading a
/// synthetic MaterialParams through that function to say so is how the two
/// models start sharing a bug.
///
/// Free-flight sampling was analog, so the only weight is the single-scattering
/// albedo: the fraction of an extinction event that scatters rather than absorbs.
static __device__ void scatterInFog(PerRayData* prd,
                                    const float3 rayOrigin,
                                    const float3 rayDir,
                                    const float t)
{
    prd->throughput *= params.fogAlbedo;
    const float3 scatterPoint = rayOrigin + rayDir * t;

    // Whether this vertex made a next-event estimate at all, which is the
    // question the miss and light-hit programs need answered and not the same as
    // whether the estimate came back with anything. Same reasoning, and the same
    // failure if it is decided from the outcome, as scatterInMedium records.
    const bool didNee = volumeNeePairsWithBounce(params.estimatorMode == 0,
                                                params.scene.numLights > 0 || params.hasEnvMap);
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
            // dot(rayDir, toLight), not dot(-rayDir, toLight). The phase function
            // takes the angle between the two directions of *travel*: light
            // arrives along -toLight and leaves toward the camera along -rayDir,
            // so their cosine is dot(rayDir, toLight). Negated, a forward-
            // scattering haze becomes a backward-scattering one -- and at the
            // pine forest's g = 0.8 that is the difference between a glow around
            // the sun and a uniform wash.
            const float phase = hgPhaseFunction(dot(rayDir, conn.toLight), params.fogAnisotropy);
            // The phase function is the medium's BSDF and its own pdf, so MIS
            // pairs it against the light density exactly as a surface lobe would.
            const float misWeight =
                conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, phase, params.misHeuristic);
            const float3 weight = prd->throughput * (conn.radiance / conn.pdf) * misWeight * phase;
            if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
            {
                // traceOcclusion carries the haze's own transmittance along this
                // ray, so the connection is dimmed by the medium it starts in.
                const float visible = traceOcclusion(params.handle, scatterPoint, conn.toLight,
                                                     params.shadowRayTmin, conn.tMax);
                if (visible > 0.0f)
                {
                    prd->radiance += clampIndirectContribution(weight * visible, prd->depth,
                                                               params.clampIndirect);
                }
            }
        }
    }

    float phasePdf = 0.0f;
    const float3 nextDir = hgSampleDirection(-rayDir, params.fogAnisotropy,
                                             random<SampleDimension::eFogPhaseU>(prd->sampler),
                                             random<SampleDimension::eFogPhaseV>(prd->sampler),
                                             phasePdf);

    prd->origin = scatterPoint;
    prd->dir = nextDir;
    prd->lastBsdfPdf = phasePdf;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    // A phase function, or a cosine lobe off a subsurface exit: both are as
    // broad as a lobe gets, so the cache's eligibility test should treat the
    // next segment as launched from a fully rough surface. Without this the
    // segment inherits whatever surface sent the ray into the medium -- for
    // water or glass that is a delta transmission, roughness zero, and the test
    // then refuses the cache for every vertex inside the medium.
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    // No passedThrough: an atmospheric scattering event is a bounce like any
    // other, so the raygen loop charges it a depth and rolls roulette on it. A
    // medium with no depth budget of its own is a path that wanders forever.
}

/// Whether the atmosphere scatters this segment before `tMax`, and where.
///
/// Gated on the path not being inside a participating medium. The atmosphere is
/// what is outside things, and a path partway through a subsurface walk or a fog
/// gizmo is not in it. Metal gates on the subsurface case alone; the difference
/// only shows in a scene that puts a bounded volume inside atmospheric haze,
/// where two free flights would otherwise be drawn for one segment and the
/// nearer one silently discarded.
///
/// What neither backend does: notice that the path is inside *glass*. The IOR
/// stack knows, and a ray crossing a windowpane will pick up haze it should not.
/// Left matching Metal rather than fixed on one side.
static __forceinline__ __device__ bool fogScatters(PerRayData* prd,
                                                   const float3 rayOrigin,
                                                   const float3 rayDir,
                                                   const float tMax,
                                                   float& t)
{
    // The single-hit debug views are declined alongside the medium, and for a
    // strictly worse failure than the one the medium avoids: a haze event
    // returns from the closest hit and from the miss program *above* the branch
    // that writes the normal, so the pixel keeps the zero the raygen
    // initialised it with.
    //
    // Measured on the pine forest, whose sidecar authors a 0.0032-density haze:
    // with this declined, 6.01% of camera rays stopped in the air before the
    // closest hit's normal branch and another 0.68% before the miss program's,
    // and 4.29% of the frame came back exactly black -- isolated speckle, 0.10
    // black neighbours out of 4, which reads as screen-space noise rather than
    // anything on a surface. The remainder of those rays are not black but are
    // no better: they carry whatever the fog vertex's next-event estimate
    // returned, which is light, in a buffer that is supposed to hold normals.
    // The fogless Cornell box has none of this, which is what hid it.
    //
    // Declined rather than handled at the vertex: an atmospheric vertex has a
    // position and no normal, so there is nothing for these two views to
    // report there, and what they are asked for is the surface behind it.
    if (!params.hasFog || prd->medium != 0u || DEBUG_MODE_IS_SINGLE_HIT(params.debug))
    {
        return false;
    }
    return fogSampleDistance(rayOrigin, rayDir, tMax, params.fogHeight, params.fogSigmaT,
                             random<SampleDimension::eFogDistance>(prd->sampler), t);
}

/// The walk reached the boundary of a subsurface medium.
///
/// Everything the surface path does below -- the material, the BSDF, the cutout
/// test -- describes what happens to a ray arriving from outside, and none of it
/// applies to one on its way out, so the exit is handled here and the rest is
/// skipped.
///
/// Whatever surface the walk hit is treated as the boundary, not only the object
/// it entered. For the closed shapes this serves that is the same surface; for
/// geometry that interpenetrates it is a simplification, and the alternative is
/// carrying the entry instance and rejecting hits on anything else -- which
/// turns an open mesh into a light leak instead.
static __device__ void exitMedium(PerRayData* prd,
                                  const float3 worldPosition,
                                  const float3 shadingNormal,
                                  const float3 geomNormal,
                                  const float3 rayDir)
{
    // The ray is travelling outwards, so the outward normal is the one it agrees
    // with. Geometric for the offset, which is what it is for; interpolated for
    // the lobe and the connection, which is what every other shading vertex uses.
    //
    // Taking the geometric one for all three draws the tessellation: on a sphere
    // of 32 latitude rings, every ring. A dense medium is what makes it visible,
    // because the walk then leaves within one triangle of where it entered and
    // nothing averages the flat normal away. See the same note in
    // wavefront.metal, and `29_subsurface_skin`, which is the row that shows it.
    const float3 outwardGeom = (dot(geomNormal, rayDir) > 0.0f) ? geomNormal : -geomNormal;
    const float3 outward = (dot(shadingNormal, outwardGeom) > 0.0f) ? shadingNormal : -shadingNormal;
    const float3 exitOrigin = offset_ray(worldPosition, outwardGeom);
    SamplerState xrng = mediumSampler(prd->sampler, prd->mediumStep + 1u);
    const float invPi = 1.0f / M_PIf;

    // Attempted, not succeeded -- see the note on the same flag in
    // scatterInMedium(). The exit lobe covers the whole outward hemisphere and
    // so does the light strategy, so the MIS weight is owed on every draw,
    // including the ones where the light sample landed below the boundary.
    bool didNee = volumeNeePairsWithBounce(params.estimatorMode == 0,
                                          params.scene.numLights > 0 || params.hasEnvMap);
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
                // The exit is Lambertian and the medium's albedo was already paid
                // for during the walk, so the lobe here is 1/pi and its own
                // density is cos/pi.
                //
                // The density is for the MIS weight only. connectLight folds the
                // cosine into conn.radiance -- it returns what a caller holding f
                // alone needs -- so multiplying the estimate by the density as
                // well applies that cosine twice: the exit connection comes back
                // at around 70% of its value while MIS has already deducted the
                // whole of it from the bounce ray.
                const float lobePdf = cosOut * invPi;
                const float misWeight =
                    conn.isDelta ? 1.0f : computeMisWeight(conn.pdf, lobePdf, params.misHeuristic);
                const float3 weight = prd->throughput * (conn.radiance / conn.pdf) * misWeight * invPi;
                if (weight.x > 1e-6f || weight.y > 1e-6f || weight.z > 1e-6f)
                {
                    const float visible = traceOcclusion(params.handle, exitOrigin, conn.toLight,
                                                         params.shadowRayTmin, conn.tMax);
                    if (visible > 0.0f)
                    {
                        // Outside the medium: this vertex is the walk leaving it,
                        // and the ray starts on the far side of the boundary.
                        // Tagging it with the walk's own medium attenuates the
                        // whole distance to the light by a dense extinction that
                        // nothing ever cancels -- and this connection is what
                        // lights a translucent object, so it arrives at zero.
                        float3 survived = make_float3(visible);
                        if (params.hasBoundedMedium)
                        {
                            survived *=
                                mediumTransmittance(exitOrigin, conn.toLight, conn.tMax, 0u);
                        }
                        prd->radiance += clampIndirectContribution(weight * survived, prd->depth,
                                                                   params.clampIndirect);
                    }
                }
            }
        }
    }

    const float3 exitDir = mediumCosineDirection(outward, random<SampleDimension::eBSDF0>(xrng),
                                                 random<SampleDimension::eBSDF1>(xrng));

    prd->origin = exitOrigin;
    prd->dir = exitDir;
    // Cosine-sampled from a 1/pi lobe: f * cos / pdf is exactly one, so the
    // throughput is untouched here. What the medium took, it took during the
    // walk.
    prd->lastBsdfPdf = fmaxf(dot(outward, exitDir), 0.0f) * invPi;
    prd->misDistance = 0.0f;
    prd->specularBounce = false;
    // A phase function, or a cosine lobe off a subsurface exit: both are as
    // broad as a lobe gets, so the cache's eligibility test should treat the
    // next segment as launched from a fully rough surface. Without this the
    // segment inherits whatever surface sent the ray into the medium -- for
    // water or glass that is a delta transmission, roughness zero, and the test
    // then refuses the cache for every vertex inside the medium.
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = 1.0f;
    }
    prd->neeDone = didNee;
    prd->medium = 0u;
    prd->mediumStep = 0u;
    // Depth advances once for the whole walk, here rather than at the entry:
    // charging it at both ends would cost a translucent surface two bounces to
    // do what an opaque one does in one. The raygen loop does the increment,
    // because this exit is a bounce.
}

/// The ray reached the environment -- or would have, if the atmosphere lets it.
///
/// Lives here rather than beside the raygen program because of that first
/// clause: a segment on its way to the sky is as long as segments get, so it is
/// the one the haze is most likely to stop, and stopping it means a next-event
/// estimate. OptiX modules do not share device functions, and connectToLight is
/// in this one. createProgramGroups() points the miss group at this module.
extern "C" __global__ void __miss__ms()
{
    PerRayData* prd = getPRD();
    const float3 ray_dir = optixGetWorldRayDirection();

    // Before everything else, including the counters and the guide: if the haze
    // scatters, this path did not reach the environment and nothing below is
    // true of it.
    float fogT = 0.0f;
    if (fogScatters(prd, optixGetWorldRayOrigin(), ray_dir, 1e16f, fogT))
    {
        scatterInFog(prd, optixGetWorldRayOrigin(), ray_dir, fogT);
        return;
    }

    // A path that reached the environment still inside a medium. Counted here
    // because here is the only place it is visible: the path is gone and it
    // still thinks it is inside glass, so every segment it travelled after the
    // exit it never had carried the wrong absorption. No exit event can catch
    // this one -- the ray left through a hole in the mesh. See ior_stack.h and
    // entry 5 of docs/open-defects.md.
    if (params.iorStats != nullptr && prd->iorStack.top >= 0)
    {
        atomicAdd(&params.iorStats[IOR_STAT_ESCAPED_INSIDE], 1u);
    }

    // Background still needs a guide record, or the denoiser reads whatever the
    // previous frame left there and smears the silhouette across the sky.
    if (prd->writeAov && !prd->aovDone && params.aov != nullptr)
    {
        writeBackgroundGuide(params, launchPixelIndex(params), ray_dir, prd->depth, prd->pixelSample);
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
            // A camera ray, a specular bounce, or a vertex that made no next-event
            // estimate: the BSDF strategy owns the whole contribution here, so no
            // MIS weight. That third case is what estimatorMode 1 needs -- weighting
            // against an estimate that was never made loses the difference.
            if (params.hasEnvBackground && prd->depth == 0)
            {
                // The backdrop is what the camera sees; the map above is what lights
                // the scene, and the MIS branch below stays on it because that is the
                // one that was importance sampled.
                const float4 bgSample = tex2D<float4>(params.envBackgroundTexture, uv.x, uv.y);
                envColor = make_float3(bgSample.x, bgSample.y, bgSample.z) *
                           params.envBackgroundIntensity * params.envMapColorTint;
            }
            radiance = prd->throughput * envColor;
        }
        else
        {
            // MIS weight with BSDF sampling vs env map PDF
            const float envPdf = envMapPdf(ray_dir,
                                           params.envMapTexturePoint,
                                           params.envMapWidth, params.envMapHeight,
                                           params.envMapRotation, params.envPdfScale);
            // Account for 50% selection probability when local lights exist
            const float envSelectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            const float effectiveEnvPdf = envPdf * envSelectionPdf;
            // A texel of zero luminance has zero sampling density, so light sampling
            // could never have produced this direction and the BSDF strategy owns it
            // outright. Dropping the contribution instead -- which this guard used to
            // do -- loses energy exactly along the edges of dark regions, where the
            // bilinear radiance is still non-zero.
            const float misWeight = (effectiveEnvPdf > 0.0f)
                                        ? computeMisWeight(prd->lastBsdfPdf, effectiveEnvPdf, params.misHeuristic)
                                        : 1.0f;
            radiance = prd->throughput * envColor * misWeight;
        }
    }
    else
    {
        MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
        radiance = prd->throughput * miss_data->bg_color;
    }

    prd->radiance += clampIndirectContribution(radiance, prd->depth, params.clampIndirect);

    prd->throughput = make_float3(0.0f);
    prd->depth = params.max_depth;
}

extern "C" __global__ void __closesthit__radiance()
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    SurfaceHitData surfaceHit = {};
    // Two separate questions, and conflating them is what kept every fibre rule
    // below off the only curve basis the tree actually exports. `isCubicCurve`
    // picks which vertex fetch to run; `isCurveHit` says the hit is on a strand at
    // all, which is what the fibre semantics are gated on. A round *linear* curve
    // -- what every particle groom writes, and what `28_hair` is -- answered no to
    // the second one for as long as the two were the same variable, so its shadow
    // rays offset into the strand, its far-side connections were rejected as
    // back-facing, and its transmitted bounces re-entered the fibre they had just
    // crossed.
    const bool isCubicCurve = (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    const bool isCurveHit = isCubicCurve || (primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR);
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data);
    }
    else if (isCubicCurve)
    {
        surfaceHit = fillCubicCurveGeomData(hit_data);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
    {
        surfaceHit = fillLinearCurveGeomData(hit_data);
    }

    // Look up material from device buffer (indexed by materialId)
    const int32_t matId = hit_data->materialId;
    const MaterialParams& matParams = params.materials[matId];
    const cudaTextureObject_t* textures = &params.materialTextures[matId * MAX_MATERIAL_TEXTURES];

    // --- Free flight through a participating medium -------------------------
    //
    // Drawn before anything else, because the segment the path just travelled
    // ends at whichever comes first -- a scattering event inside the medium or
    // this surface -- and everything below describes a vertex that only exists
    // if the surface won.
    //
    // The distance is sampled here rather than as a bound on the ray, which is
    // where Metal's `extend` puts it. The two are statistically identical: a
    // surface at or before the sampled distance wins either way, and the bound
    // is only a scheduling decision about how far traversal is allowed to run.
    const float surfaceT = optixGetRayTmax();
    float segment = surfaceT;
    MediumSample medium = {};
    bool insideMedium = false;
    bool mediumIsBounded = false;
    bool sampledFreeFlight = false;
    if (prd->medium != 0u)
    {
        const MaterialParams& mm = params.materials[prd->medium - 1u];
        insideMedium = true;
        mediumIsBounded = (mm.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        // A bounded volume has no entry surface to have textured, so it keeps the
        // material's constant; a subsurface walk takes what the boundary resolved.
        const float3 mediumAlbedo =
            mediumIsBounded ? mm.diffuse_transmission_color : prd->mediumAlbedo;
        const uint32_t stepCeiling = min(params.subsurfaceIterations, MEDIUM_MAX_STEPS);
        // Past the ceiling the walk stops drawing free flights, so the next
        // surface is its boundary and the path leaves rather than hanging.
        if (prd->mediumStep < stepCeiling)
        {
            SamplerState mrng = mediumSampler(prd->sampler, prd->mediumStep);
            medium = sampleMedium(
                fromSpectrum(oka::medium::sigmaTFromRadius(toSpectrum(mm.subsurface_radius))),
                mediumAlbedo, prd->throughput, surfaceT,
                random<SampleDimension::eBSDF2>(mrng), random<SampleDimension::eBSDF3>(mrng));
            sampledFreeFlight = true;
            if (medium.scattered)
            {
                segment = medium.t;
            }
        }
    }

    // --- Free flight through the atmosphere ---------------------------------
    //
    // Drawn against whatever the segment has already been shortened to, so a
    // bounded medium that scattered nearer keeps the vertex and the haze does
    // not overwrite it. In practice the two are mutually exclusive -- fogScatters
    // declines while the path is inside a medium at all -- and this is what makes
    // that safe rather than merely true today.
    float fogT = 0.0f;
    const bool fogScattered = fogScatters(prd, optixGetWorldRayOrigin(), ray_dir, segment, fogT);
    if (fogScattered)
    {
        segment = fogT;
    }

    // --- Absorption over the segment just travelled -------------------------
    //
    // The IOR stack already knows which medium the path is inside; it also
    // carries the material that medium came from, so the extinction is looked up
    // here rather than threaded through the payload. It has to happen before the
    // pop at the bottom of this program, and before emission and next-event
    // estimation, or everything seen through a dense medium keeps its own colour.
    //
    // volume.h was included by no .cu file at all before this, so the OptiX path
    // had no volumetric attenuation of any kind.
    //
    // Skipped inside a subsurface walk, which is a different medium model:
    // mediumScatterWeight already carries that medium's extinction, and a
    // material carrying both extensions would otherwise be attenuated twice for
    // one interior. A bounded volume is not skipped -- being inside a fog gizmo
    // says nothing about whether the path is also inside glass.
    if (!insideMedium || mediumIsBounded)
    {
        const unsigned int inside = ior_stack_current_material(prd->iorStack);
        if (inside != 0xFFFFFFFFu)
        {
            const MaterialParams& im = params.materials[inside];
            const float3 sigma_t = volume_extinction(im.attenuation_color, im.attenuation_distance,
                                                     params.volumeModel);
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
        scatterInMedium(prd, params.materials[prd->medium - 1u], medium, optixGetWorldRayOrigin(),
                        ray_dir, mediumIsBounded);
        return;
    }
    if (sampledFreeFlight)
    {
        // The other half of the analog estimator: the path reached a boundary
        // without scattering, and the weight for that is the transmittance over
        // the density of having drawn a distance at least this long. Exactly one
        // for a grey extinction, which is why a fog gizmo of uniform density
        // never notices it is here -- and not one at all for 25_subsurface,
        // whose three mean free paths differ by more than threefold.
        prd->throughput *= mediumBoundaryWeight(medium, segment);
    }

    // --- Crossing the boundary of a participating medium --------------------
    //
    // The gizmo of a bounded volume is not a surface: it is where the medium
    // starts and stops. A ray through it toggles which medium it is in and
    // carries on with the same direction and throughput -- unshaded, and without
    // spending a bounce, because a volume the light passes through twice would
    // otherwise cost two of them.
    if ((matParams.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
    {
        // Bounded by the same counter the cutout pass-through uses, and for the
        // same reason: neither advances `depth`, so neither has a natural end. A
        // boundary the ray re-hits through a self-intersection would otherwise
        // toggle the medium forever and the path would never terminate.
        if (prd->passthrough >= PATH_PASSTHROUGH_MAX)
        {
            prd->throughput = make_float3(0.0f);
            return;
        }
        ++prd->passthrough;

        // Toggle, rather than deciding from the normal.
        //
        // A gizmo's winding is arbitrary: a DCC decides inside from an
        // inside/outside test and never looks at the normal, so a box exported
        // from one may be wound either way. Reading `entering` off
        // dot(rayDir, geomNormal) inverts such a volume -- and an inverted volume
        // is not a subtle error, it is a medium that fills all of space except
        // the gizmo.
        const uint32_t here = static_cast<uint32_t>(matId) + 1u;
        const bool leaving = (prd->medium == here);
        prd->medium = leaving ? 0u : here;
        prd->mediumStep = 0u;

        // Push past the surface on the side the ray is heading, which needs the
        // sign of the normal and not its direction.
        const float3 exitSide = (dot(ray_dir, surfaceHit.geom_normal) > 0.0f)
                                    ? surfaceHit.geom_normal
                                    : -surfaceHit.geom_normal;
        prd->origin = offset_ray(surfaceHit.position, exitSide);
        prd->dir = ray_dir;
        // The MIS distance keeps counting: as far as the light at the end of
        // this ray is concerned, the scattering vertex is still the one before
        // the boundary.
        prd->misDistance += segment;
        // Nothing else about the path moves -- not the throughput, not
        // `specularBounce`, not `lastBsdfPdf` -- so a light or an environment
        // beyond the gizmo is still weighted against the bounce that actually
        // produced the direction.
        prd->passedThrough = true;
        return;
    }

    // Fill SurfaceInteraction from hit data, resolving textures, the uv
    // transform, the normal map, vertex colour and coverage on the way.
    SurfaceInteraction si;
    initSurfaceInteraction(si, matParams, textures, surfaceHit.position, surfaceHit.normal,
                           surfaceHit.geom_normal, surfaceHit.worldTangent,
                           surfaceHit.worldBinormal, surfaceHit.uv, ray_dir,
                           surfaceHit.vertexColor);

    // A strand shaded by the whole-fibre lobe: light crosses it in one event, so
    // neither the hemisphere tests nor the ray offsets below apply. Gated on the
    // geometry as well as the material because the chord needs a radius, and only
    // a curve hit has one.
    const bool isFibre = isCurveHit && scattersThroughFibre(si) && surfaceHit.curveRadius > 0.0f;
    const float curveRadius = isFibre ? surfaceHit.curveRadius : 0.0f;

    // --- Leaving a subsurface medium ----------------------------------------
    //
    // Before the guides, the cutout test, emission and the BSDF, all of which
    // describe a ray arriving from outside. This one is on its way out.
    if (insideMedium && !mediumIsBounded)
    {
        exitMedium(prd, surfaceHit.position, surfaceHit.normal, surfaceHit.geom_normal, ray_dir);
        return;
    }

    if (prd->writeAov && params.aov != nullptr)
    {
        writeSurfaceGuide(hit_data, prd, si, surfaceHit.position, primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE);
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
        const bool moved = (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) &&
                           previousTriangleWorldPosition(hit_data, prevPosition);
        prd->radiance = make_float3(
            optixGetRayTime(), moved ? saturate(length(surfaceHit.position - prevPosition) * 10.0f) : 0.0f, 0.0f);
        return;
    }

    // The two radiance-cache views that describe the primary surface. Both use
    // exactly the quantities the cache itself uses -- si.position and
    // si.shading_normal, not the geometry normal -- because a debug view that
    // addresses a different voxel than the cache does is worse than none: it
    // agrees often enough to be believed.
    if (params.debug == (uint32_t)DebugMode::eSharcGrid)
    {
        // The voxels themselves. This is the view the SDK's own guidance leans
        // on for choosing voxel size, and it needs no cache to be allocated --
        // the grid is arithmetic, so the size can be dialled in before the
        // cache is ever switched on.
        const float3 cameraPosition =
            make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        const unsigned long long key = sharcVoxel(si.position, si.shading_normal, cameraPosition, params.sharcBaseSize,
                                                  /*responsive=*/false);
        prd->radiance = sharcDebugColour(oka::sharc::keyHash(key));
        return;
    }
    if (params.debug == (uint32_t)DebugMode::eSharcRadiance && prd->depth == 0 && params.sharcCapacity != 0u)
    {
        // What the cache would answer at the primary surface, shown directly
        // instead of through however many bounces normally stand between a
        // lookup and the pixel. Black means the voxel is missing or has not
        // resolved yet -- which is the difference the occupancy view explains.
        //
        // Written straight to the image, and the path then carries on shading
        // normally, which is the whole trick: the voxels this view reads are
        // filled by the bounces of this same path. Returning here instead --
        // the obvious way to write it, and the way it was written first --
        // leaves the cache empty and the view 99% black.
        //
        // The raygen skips its own write for this mode, so this is the only
        // thing that puts a value in the pixel; it also clears the pixel first,
        // so a camera ray that misses everything stays black rather than
        // keeping the last frame's answer.
        const float3 cameraPosition =
            make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
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

    // Coverage. MASK resolves to 0 or 1 and BLEND to its alpha, so one
    // stochastic test covers both: with probability (1 - opacity) the path
    // continues straight through, unchanged and unshaded. Nothing else about
    // the path moves -- not the throughput, not `specularBounce`, not
    // `lastBsdfPdf` -- so a light or an environment seen through a cutout is
    // still weighted against the bounce that actually produced the direction.
    const float opacity = resolveOpacity(matParams, textures, si.uv);
    if (opacity < 1.0f && prd->passthrough < PATH_PASSTHROUGH_MAX)
    {
        if (opacitySample(prd->sampler, launchPixelIndex(params), prd->passthrough) >= opacity)
        {
            // Step off on the side the ray was travelling, so the next trace
            // cannot re-hit the surface it just passed through.
            const float3 faceNg =
                (dot(si.geometry_normal, ray_dir) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            prd->origin = offset_ray(si.position, faceNg);
            prd->dir = ray_dir;
            prd->misDistance += segment;
            ++prd->passthrough;
            prd->passedThrough = true;
            return;
        }
    }

    // Add emission
    if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
    {
        prd->radiance +=
            clampIndirectContribution(prd->throughput * si.emission, prd->depth, params.clampIndirect);
    }

    // Set exterior IOR from the IOR stack for nested dielectrics
    bool entering = si.front_face;
    if (entering)
    {
        si.exterior_ior = ior_stack_current_ior(prd->iorStack);
    }
    else
    {
        si.exterior_ior = ior_stack_peek_after_pop(prd->iorStack, si.dielectric_priority);
    }

    // --- Radiance cache ------------------------------------------------------
    //
    // Read only past the first few bounces and only off a rough surface: the
    // camera ray and the first bounce carry the detail a voxel average would
    // blur, and a mirror reflects a direction rather than a place.
    //
    // Placed here, after emission and the IOR stack and before the BSDF sample,
    // for the same reason Metal places it here: the snapshot has to exclude this
    // vertex's own emission (which the cache's readers add for themselves when
    // they land on the same surface) and include this vertex's direct lighting
    // (which is part of what leaves the point).
    //
    // A path that has already recorded a voxel is done with the cache -- it owes
    // that voxel an honest estimate of the rest of itself, so it may not read,
    // and it has nowhere left to record -- which is what this pixel's
    // SharcPathState::index being set means, and why it is part of the guard
    // rather than checked inside.
    // A surface reached by a delta bounce is still the surface the eye is
    // looking at, and must not be answered for by a voxel average.
    //
    // `prd->depth` counts segments, not scattering events, so a mirror at depth
    // 0 puts what it reflects at depth 1 -- where this gate used to let the
    // cache answer. What the viewer sees in the mirror was then replaced by the
    // average over a voxel, and the same went for anything behind the shower
    // glass, since a delta transmission is a specular event too. The SDK states
    // this as its own rule and its own mistake to avoid: a replaced primary
    // surface is still a primary surface.
    //
    // `specularBounce` is written at the end of this program for the bounce it
    // generates, so here it describes the bounce that arrived -- exactly the
    // question being asked. It is conservative in one direction: a mirror
    // reached after a diffuse bounce also blocks the cache one vertex longer
    // than it strictly must, which costs a caching opportunity and no accuracy.
    // Reads stop once the render has accumulated past the point where the
    // cache's own error is the larger of the two; deposits do not, so the table
    // stays current for the next camera movement. See Params::sharcReadMaxSubframe
    // for the measurement that sets the default.
    //
    // Only while the film is accumulating, and that clause is not a nicety. The
    // limit exists because the cache's error is correlated and so stops being
    // the smaller error once enough samples have been averaged -- which
    // presupposes that samples are being averaged. With accumulation off every
    // frame is a fresh one-sample image, there is nothing converging past the
    // cache, and the cache is the entire reason the frame is not black.
    //
    // Without this clause the limit did the opposite of its job: with
    // accumulation off the renderer reports mSubframeIndex as the *whole* sample
    // budget rather than zero (OptixRender.cpp, and it does so deliberately --
    // a counter that reset every frame hung StrelkaCLI on the single-hit debug
    // views), so subframe_index is 256 from the second frame on and the cache
    // was never read at all. Measured on iso_bathroom at one sample against a
    // warm cache: 88% less error than no cache, and none of it was reaching the
    // one configuration built to show it.
    const bool sharcMayRead = params.sharcReadMaxSubframe == 0u || !params.enableAccumulation ||
                              params.subframe_index < params.sharcReadMaxSubframe;

    if (params.sharcCapacity != 0u && params.sharcPath[launchPixelIndex(params)].index == SHARC_NO_ENTRY &&
        prd->depth >= params.sharcDepth && !prd->specularBounce)
    {
        const float3 cameraPosition =
            make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        const unsigned long long voxelKey = sharcVoxel(si.position, si.shading_normal, cameraPosition,
                                                       params.sharcBaseSize, /*responsive=*/false);

        // A fixed share of paths never read and always trace to the end, so the
        // cache keeps converging instead of freezing at whatever the first few
        // paths through a voxel happened to find. They are also the only paths
        // whose deposits are unconditioned on the cache's own output, which is
        // the loop that would amplify whatever error it starts with -- Metal saw
        // it as a classroom 11% bright with no single step being wrong.
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
            // The segment just traced, against the voxel it landed in. Both
            // numbers are here rather than in the gate above because a path
            // that may not *read* may still record: the deposit is an honest
            // estimate whatever lobe produced it, and refusing it would starve
            // exactly the voxels a tight lobe keeps looking at.
            const float dx = si.position.x - cameraPosition.x;
            const float dy = si.position.y - cameraPosition.y;
            const float dz = si.position.z - cameraPosition.z;
            const float voxelSize =
                oka::sharc::voxelForDistance(sqrtf(dx * dx + dy * dy + dz * dz), params.sharcBaseSize).size;
            // From the vertex that *scattered*, not from wherever the ray was
            // last restarted. Passing through a cutout or a medium boundary
            // restarts the ray at that surface, so optixGetRayTmax() alone
            // measures the last leg only -- and the lobe whose spread this test
            // is about was launched before it. `misDistance` is the distance
            // already travelled since that vertex, carried for the MIS weight
            // for exactly the same reason.
            //
            // Under-measuring the segment fails both halves of the test, so a
            // scene with alpha cutouts or glass in front of things -- this
            // bathroom has a shower screen -- would refuse the cache on paths
            // that qualify.
            const float segmentLength = optixGetRayTmax() + prd->misDistance;
            const bool eligible = oka::sharc::mayReadCache(
                segmentLength, params.sharcPath[launchPixelIndex(params)].launchRoughness, voxelSize);

            if (sharcMayRead && eligible && !updatePath && cachedSamples >= (float)params.sharcMinSamples)
            {
                // The two halves are an additive split of the same radiance, so
                // a reader adds them back. Compiled out entirely when no light
                // in the scene is responsive -- sharcResponsive is bound into
                // the pipeline as a constant.
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
            // Recorded only while the throughput is worth dividing by. The
            // deposit is what the path gathers from here on divided by its
            // throughput here, and at a throughput of a thousandth that
            // estimator has a variance to match.
            if (luminance(prd->throughput) > oka::sharc::kMinRecordThroughput)
            {
                const float floorT = oka::sharc::kThroughputFloor;
                SharcPathState visit;
                visit.index = slot;
                visit.radianceAtVisit = prd->radiance;
                visit.responsiveRadiance = make_float3(0.0f);
                // Both entries are claimed here, together, and both are
                // deposited into together at the end of the path -- including
                // when the responsive part turns out to be zero. That is what
                // makes their two means add up to the mean of the total: they
                // have to be averages over the same set of paths. Claiming the
                // responsive slot lazily, only for paths that saw a responsive
                // light, would make one mean a subset of the other's and the sum
                // would count some light twice.
                visit.responsiveIndex = SHARC_NO_ENTRY;
                if (params.sharcResponsive != 0u)
                {
                    uint32_t responsiveSlot = 0u;
                    if (sharcFind(params.sharcEntries, params.sharcCapacity, oka::sharc::responsiveKey(voxelKey),
                                  true, responsiveSlot))
                    {
                        visit.responsiveIndex = responsiveSlot;
                    }
                }
                visit.invThroughput = make_float3(1.0f / fmaxf(prd->throughput.x, floorT),
                                                  1.0f / fmaxf(prd->throughput.y, floorT),
                                                  1.0f / fmaxf(prd->throughput.z, floorT));
                params.sharcPath[launchPixelIndex(params)] = visit;
            }
        }
    }

    // Next-event estimation first, and decided by the material rather than by the
    // bounce.
    //
    // It used to run after bsdf_sample() and be gated on the event that came
    // back, which made the light half of the estimate depend on a draw belonging
    // to the other half. Two things were lost that way: the whole vertex
    // whenever the sample came back BSDF_EVENT_ABSORB (a microfacet draw that
    // landed below the horizon is not a material that absorbs), and the smooth
    // lobe's direct light on every draw the delta lobe won -- over half the
    // draws on a clearcoat with glTF's default coat roughness of 0. See
    // neeRunsAtVertex() and bsdf_has_smooth_lobe().
    //
    // Moving it costs nothing in sample values: random<Dim>() is a pure function
    // of (sampleIdx, dimension, seed, depth), so the order the dimensions are
    // drawn in does not change any of them.
    //
    // estimatorMode 1 drops next-event estimation entirely and lets BSDF sampling
    // carry the whole integral. The two are independent unbiased estimators, so at
    // convergence they must agree; the difference between them measures estimator
    // inconsistency directly, which is the only reason the switch exists.
    const bool didNee = neeRunsAtVertex(params.estimatorMode == 0,
                                        params.scene.numLights > 0 || params.hasEnvMap,
                                        bsdf_has_smooth_lobe(si));
    if (didNee)
    {
        bool responsiveLight = false;
        const float3 direct = estimateDirectLighting(prd, si, curveRadius, responsiveLight);
        prd->radiance += direct;
        // Responsive lighting: remember how much of what this path gathers comes
        // from a light on the fast clock, so the deposit at the end of the path
        // can be split between the voxel's two entries.
        //
        // Recorded in the same units as prd->radiance -- that is, still carrying
        // the throughput at this vertex -- because the deposit divides the whole
        // difference by the throughput *at the visit*, and the two have to be
        // divided by the same thing to subtract cleanly.
        //
        // Only while this pixel's path has a live visit to deposit into. Every
        // vertex from the visit onward counts, not just the visited one: what
        // the cache stores is the outgoing radiance of the visited point, and a
        // responsive light reaching it through two more bounces is just as
        // responsive as one reaching it directly.
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

    const float z1 = random<SampleDimension::eBSDF0>(prd->sampler);
    const float z2 = random<SampleDimension::eBSDF1>(prd->sampler);
    const float z3 = random<SampleDimension::eBSDF2>(prd->sampler);
    const float z4 = random<SampleDimension::eBSDF3>(prd->sampler);

    float4 xi = make_float4(z1, z2, z3, z4);
    BsdfSampleResult sample_data = bsdf_sample(si, xi);

    if (sample_data.event_type == BSDF_EVENT_ABSORB)
    {
        if (prd->depth == 0)
        {
            prd->setFirstEventType(EventType::eAbsorb);
        }
        // stop on absorb. Whatever next-event estimation delivered above stays:
        // it is this vertex's direct lighting and does not depend on where the
        // path went next.
        prd->throughput = make_float3(0.0f);
        return;
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
    // Update IOR stack on transmission.
    //
    // A fibre's transmission lobes do not put the path inside anything: the
    // strand is crossed within the one event, so there is no medium to enter and
    // no entry to match with an exit. Pushing here left every transmitted hair
    // path one level deeper than it came in, and a groom is thousands of hairs
    // deep.
    // Colour the diffuse-transmission lobe applies on the way into a subsurface
    // medium, divided back out below. The walk supplies the colour itself, once
    // per scattering event, so leaving the lobe's copy in charges the first event
    // twice: a sphere comes out at albedo times its correct reflectance, which
    // for a deep-red medium is about a third of the light it should return.
    // Cycles divides the same factor out at the same place.
    float3 sssEntryTint = make_float3(1.0f);
    // Set when the path enters a subsurface medium, so the direction below is the
    // refraction rather than the lobe's cosine draw. See docs/open-defects.md 16.
    bool sssRefractedEntry = false;
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0 && !isFibre)
    {
        // A thin-walled surface has no interior either, so crossing it does not
        // put the path inside anything. Pushing the stack anyway left a ray that
        // had gone through the front of a bubble believing it was inside glass,
        // so the far side read as an exit from a dense medium -- and every
        // grazing angle there is past the critical angle.
        if (!si.thin_walled)
        {
            if (entering)
            {
                // Counted, not merely survived. Both of these leave the path
                // carrying the wrong medium and neither used to say a word on
                // this backend; see entry 5 of docs/open-defects.md. The test is
                // a loop over at most four entries on a path that has already
                // established it is a transmission through a solid, and the
                // atomic only runs when something has actually gone wrong.
                if (params.iorStats != nullptr && ior_stack_full(prd->iorStack))
                {
                    atomicAdd(&params.iorStats[IOR_STAT_OVERFLOW], 1u);
                }
                ior_stack_push(prd->iorStack, si.dielectric_priority, si.ior, (unsigned int)matId);
            }
            else
            {
                if (params.iorStats != nullptr &&
                    !ior_stack_can_pop(prd->iorStack, si.dielectric_priority, (unsigned int)matId))
                {
                    atomicAdd(&params.iorStats[IOR_STAT_UNMATCHED], 1u);
                }
                ior_stack_pop(prd->iorStack, si.dielectric_priority, (unsigned int)matId);
            }
        }
        prd->origin = offset_ray(si.position, -faceNg);

        // Entering a subsurface medium. The lobe that got here is the diffuse
        // transmission one, which on its own puts the light straight out the far
        // side; what this adds is that it random-walks on the way. From here the
        // path is inside, and the next closest hit samples a free flight instead
        // of shading whatever it reaches.
        if (si.subsurface > 0.0f &&
            (sample_data.event_type & BSDF_EVENT_DIFFUSE_TRANSMISSION) != 0)
        {
            prd->medium = static_cast<uint32_t>(matId) + 1u;
            prd->mediumStep = 0u;

            // The walk's albedo, resolved here because this is the last place a
            // texture exists: inside the medium there is no surface to sample.
            // Scaled by how far this point's albedo departs from the one the
            // material's scatter colour was derived from, so a flat material
            // takes the ratio 1 and is unchanged, and marble carries its veining
            // in.
            float3 walkAlbedo = matParams.diffuse_transmission_color;
            const float3 reference = matParams.subsurface_reference;
            if (reference.x > 1e-4f && reference.y > 1e-4f && reference.z > 1e-4f)
            {
                walkAlbedo *= si.albedo / reference;
            }
            prd->mediumAlbedo = saturate(walkAlbedo);
            sssEntryTint = fmaxf(si.diffuse_transmission_color, make_float3(1e-4f));

            // Enter on the refraction, not on the lobe's cosine draw. The lobe
            // still decides whether the medium is entered and still supplies the
            // weight; only the direction changes, and it is the direction that
            // sets how far a path travels through the body. Same note as in
            // wavefront.metal.
            sssRefractedEntry = true;
        }
    }
    else
    {
        prd->origin = offset_ray(si.position, faceNg);
    }
    prd->dir = sssRefractedEntry ?
                   subsurface_entry_direction(si.wo,
                                              (dot(si.shading_normal, si.wo) > 0.0f) ? si.shading_normal :
                                                                                       -si.shading_normal) :
                   sample_data.wi;
    if (isFibre)
    {
        // Both branches above assume a surface with an inside and an outside. A
        // strand has neither: the bounce leaves from wherever the crossing the
        // lobe has already accounted for comes out. Without this the ray hits the
        // far wall and buys a second whole-fibre event -- and a third, which is
        // what made an isolated strand's cross-section climb with depth instead
        // of going flat.
        prd->origin = fibreExitOrigin(si.position, si.tangent, si.shading_normal, curveRadius,
                                      normalize(prd->dir));
    }
    // What the next vertex is allowed to weight its light hit against.
    //
    // Next-event estimation here only ever proposed directions on the side of the
    // shading normal that `estimateDirectLighting` accepts -- above it on a front
    // face, and nothing at all through a back one. So a bounce that leaves in any
    // other direction is a direction the light-sampling strategy could not have
    // produced, and the balance heuristic must not deduct a share for it: the
    // deduction is real and the delivery never happens. Recording `didNee` before
    // the direction was known charged rough transmission for an estimate that had
    // already rejected the whole transmitted hemisphere, which is why the frosted
    // end of `22_thin_walled` was dark while its smooth control -- a specular
    // event, and already exempt through `specularBounce` -- was not.
    //
    // A fibre is the exception: its connections reach the far side of the strand,
    // so withholding the weight there would count the light twice. This is the
    // same expression Metal's `wavefrontShade` applies at the same point.
    //
    // Against the shaded frame, for the reason given at the proposal above. An
    // opaque back hit used to absorb, so it had no bounce to weight at all;
    // now that it has one, passing the raw front_face would withhold the weight
    // from a direction next-event estimation did offer, and the light would land
    // about twice.
    const ShadedFrame bounceFrame = shadedFrame(si.front_face, dot(si.shading_normal, si.wo),
                                                si.transmission, si.diffuse_transmission);
    prd->neeDone = neePairsWithBounce(didNee, isFibre, bounceFrame.frontFace,
                                      bounceFrame.normalSign * dot(si.shading_normal, prd->dir));
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->misDistance = 0.0f;
    prd->throughput *= sample_data.bsdf_over_pdf / sssEntryTint;
    // What the next hit's cache-eligibility test asks about: the lobe this
    // vertex is sending the ray out of. A specular event is recorded as zero
    // roughness whatever the material says, because that is what the test means
    // by a lobe that still carries an image.
    if (params.sharcCapacity != 0u)
    {
        params.sharcPath[launchPixelIndex(params)].launchRoughness = prd->specularBounce ? 0.0f : si.roughness;
    }
}
