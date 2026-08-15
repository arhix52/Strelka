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
#include "alpha.h"
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
    return __uint_as_float(transmittance);
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
};

static __forceinline__ __device__ LightConnection makeEmptyConnection()
{
    LightConnection c;
    c.radiance = make_float3(0.0f);
    c.toLight = make_float3(0.0f);
    c.pdf = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.isDelta = false;
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
    const float3 offsetNg =
        (dot(si.geometry_normal, toLight) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
    return offset_ray(si.position, offsetNg);
}

static __device__ LightConnection connectLight(SamplerState& sampler,
                                               const UniformLight& light,
                                               const SurfaceInteraction& si,
                                               float curveRadius)
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
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    LightConnection c = makeEmptyConnection();
    c.toLight = lightSampleData.L;
    // Sharp point/spot only: give one a radius and it is sampled as a sphere,
    // which BSDF rays can hit and which therefore does need MIS.
    c.isDelta = (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT) &&
                !(light.points[0].x > 1e-4f);

    float3 Li = make_float3(light.color);
    if (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
    {
        const float dist = fmaxf(lightSampleData.distToLight, 1e-4f);
        Li *= rangeWindow(light, dist) / (dist * dist);
        if (light.type == LIGHT_TYPE_SPOT)
        {
            Li *= spotAttenuation(light, -lightSampleData.L);
        }
    }

    // lightReachesShadingPoint() is `dot(N, L) > 0` for everything except a
    // fibre, where the hemisphere test is the wrong question -- see the note on
    // it in shading/shading_common.h.
    const bool lit = lightReachesShadingPoint(si, lightSampleData.L);
    const bool facing = (light.type == LIGHT_TYPE_POINT || light.type == LIGHT_TYPE_SPOT)
                            ? (lit && emitsLight(Li))
                            : (lit && -dot(lightSampleData.L, lightSampleData.normal) > 0.0f &&
                               emitsLight(Li));
    if (facing)
    {
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h. shadingCosine() is
        // saturate(dot(N, L)) for everything but a fibre.
        c.radiance = Li * shadingCosine(si, lightSampleData.L);
        c.pdf = lightSampleData.pdf;
        c.tMax = lightSampleData.distToLight;
        c.needsRay = true;
    }
    return c;
}

static __device__ LightConnection connectEnvLight(SamplerState& sampler,
                                                  const SurfaceInteraction& si,
                                                  float curveRadius)
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
    // a fibre, which has no dark side.
    if (!lightReachesShadingPoint(si, dir))
        return c;

    // Bilinear for the radiance carried down the ray; the point fetch inside
    // sampleEnvMap is for the sampling density only, and using it here would
    // quantise the lighting to the map's texels.
    const float2 uv = dirToEnvUV(dir, params.envMapRotation);
    const float4 envSample = tex2D<float4>(params.envMapTexture, uv.x, uv.y);
    float3 Li = make_float3(envSample.x, envSample.y, envSample.z);
    Li *= params.envMapIntensity * params.envMapColorTint;

    c.radiance = Li * shadingCosine(si, dir);
    c.tMax = 1e16f;
    c.needsRay = true;
    return c;
}

/// Choose a strategy and build the connection. Visibility is the caller's job.
static __device__ LightConnection connectToLight(SamplerState& sampler,
                                                 const SurfaceInteraction& si,
                                                 float curveRadius)
{
    if (params.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(sampler);

        if (params.scene.numLights == 0 || u >= 0.5f)
        {
            // Sample environment map
            const float selectionPdf = (params.scene.numLights > 0) ? 0.5f : 1.0f;
            LightConnection c = connectEnvLight(sampler, si, curveRadius);
            c.pdf *= selectionPdf;
            return c;
        }
        // Sample local light (remap u from [0, 0.5) to [0, 1))
        const uint32_t lightId = selectLightIndex(u * 2.0f, params.scene.numLights);
        LightConnection c = connectLight(sampler, params.scene.lights[lightId], si, curveRadius);
        c.pdf *= 0.5f / params.scene.numLights;
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
    LightConnection c = connectLight(sampler, params.scene.lights[lightId], si, curveRadius);
    c.pdf *= 1.0f / params.scene.numLights;
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
static __device__ float3 estimateDirectLighting(PerRayData* prd,
                                                const SurfaceInteraction& si,
                                                float curveRadius)
{
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
        const bool isNextEventValid =
            (isFibre || ((dot(conn.toLight, si.shading_normal) > 0.0f) == si.front_face)) &&
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
    return clampIndirectContribution(weight * transmittance, prd->depth, params.clampIndirect);
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
    float3 geomNormal = cross(p1 - p0, p2 - p0);
    geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));
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
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
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
    float3 worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));
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
    const uint32_t pixelIndex = prd->linearPixelIndex;
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

extern "C" __global__ void __closesthit__radiance()
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    PerRayData* prd = getPRD();
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_dir = optixGetWorldRayDirection();

    SurfaceHitData surfaceHit = {};
    const bool isCurveHit = (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE)
    {
        surfaceHit = fillTriangleGeomData(hit_data);
    }
    else if (isCurveHit)
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
    {
        const unsigned int inside = ior_stack_current_material(prd->iorStack);
        if (inside != 0xFFFFFFFFu)
        {
            const MaterialParams& im = params.materials[inside];
            const float3 sigma_t = volume_extinction(im.attenuation_color, im.attenuation_distance,
                                                     params.volumeModel);
            prd->throughput *= beer_lambert_transmittance(sigma_t, optixGetRayTmax());
        }
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

    // Coverage. MASK resolves to 0 or 1 and BLEND to its alpha, so one
    // stochastic test covers both: with probability (1 - opacity) the path
    // continues straight through, unchanged and unshaded. Nothing else about
    // the path moves -- not the throughput, not `specularBounce`, not
    // `lastBsdfPdf` -- so a light or an environment seen through a cutout is
    // still weighted against the bounce that actually produced the direction.
    const float opacity = resolveOpacity(matParams, textures, si.uv);
    if (opacity < 1.0f && prd->passthrough < PATH_PASSTHROUGH_MAX)
    {
        if (opacitySample(prd->sampleIndex, prd->linearPixelIndex, prd->passthrough) >= opacity)
        {
            // Step off on the side the ray was travelling, so the next trace
            // cannot re-hit the surface it just passed through.
            const float3 faceNg =
                (dot(si.geometry_normal, ray_dir) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
            prd->origin = offset_ray(si.position, faceNg);
            prd->dir = ray_dir;
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
    // and it has nowhere left to record -- which is what `sharcIndex` being set
    // means and why it is part of the guard rather than checked inside.
    if (params.sharcCapacity != 0u && prd->sharcIndex == SHARC_NO_ENTRY &&
        prd->depth >= params.sharcDepth && si.roughness > oka::sharc::kMinRoughness)
    {
        const float3 cameraPosition =
            make_float3(params.viewToWorld[3], params.viewToWorld[7], params.viewToWorld[11]);
        uint32_t voxelHash = 0u, voxelKey = 0u;
        sharcVoxel(si.position, si.shading_normal, cameraPosition, params.sharcBaseSize, voxelHash, voxelKey);

        // A fixed share of paths never read and always trace to the end, so the
        // cache keeps converging instead of freezing at whatever the first few
        // paths through a voxel happened to find. They are also the only paths
        // whose deposits are unconditioned on the cache's own output, which is
        // the loop that would amplify whatever error it starts with -- Metal saw
        // it as a classroom 11% bright with no single step being wrong.
        const bool updatePath = oka::sharc::isUpdatePath(prd->linearPixelIndex, prd->sampleIndex);
        uint32_t slot = 0u;
        // Inserting, because this path is here to fill the slot in; a read that
        // misses simply carries on tracing.
        if (sharcFind(params.sharcEntries, params.sharcCapacity, voxelHash, voxelKey, true, slot))
        {
            uint32_t cachedCount = 0u;
            const float3 cached = sharcRead(params.sharcEntries, slot, cachedCount);
            if (!updatePath && cachedCount >= params.sharcMinSamples)
            {
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
                prd->sharcIndex = slot;
                prd->sharcRadianceAtVisit = prd->radiance;
                prd->sharcInvThroughput = make_float3(1.0f / fmaxf(prd->throughput.x, floorT),
                                                      1.0f / fmaxf(prd->throughput.y, floorT),
                                                      1.0f / fmaxf(prd->throughput.z, floorT));
            }
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
            prd->firstEventType = EventType::eAbsorb;
        }
        // stop on absorb
        prd->throughput = make_float3(0.0f);
        return;
    }
    prd->specularBounce = ((sample_data.event_type & BSDF_EVENT_SPECULAR) != 0);

    if (prd->depth == 0)
    {
        if (sample_data.event_type & BSDF_EVENT_DIFFUSE)
        {
            prd->firstEventType = EventType::eDiffuse;
        }
        if (sample_data.event_type & BSDF_EVENT_GLOSSY)
        {
            prd->firstEventType = EventType::eSpecular;
        }
    }

    // estimatorMode 1 drops next-event estimation entirely and lets BSDF sampling
    // carry the whole integral. The two are independent unbiased estimators, so at
    // convergence they must agree; the difference between them measures estimator
    // inconsistency directly, which is the only reason the switch exists.
    const bool didNee = (params.estimatorMode == 0) &&
                        ((sample_data.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) != 0) &&
                        (params.scene.numLights > 0 || params.hasEnvMap);
    if (didNee)
    {
        prd->radiance += estimateDirectLighting(prd, si, curveRadius);
        if (prd->throughput.x == 0.0f && prd->throughput.y == 0.0f && prd->throughput.z == 0.0f)
        {
            // estimateDirectLighting() found a NaN and painted the pixel.
            return;
        }
    }
    prd->neeDone = didNee;

    // setup next path segment
    // Face normal oriented toward the incoming ray (wo)
    float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f)
                  ? si.geometry_normal : -si.geometry_normal;
    // Update IOR stack on transmission.
    //
    // A fibre's transmission lobes do not put the path inside anything: the
    // strand is crossed within the one event, so there is no medium to enter and
    // no entry to match with an exit. Pushing here left every transmitted hair
    // path one level deeper than it came in, and a groom is thousands of hairs
    // deep.
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
                ior_stack_push(prd->iorStack, si.dielectric_priority, si.ior, (unsigned int)matId);
            else
                ior_stack_pop(prd->iorStack, si.dielectric_priority, (unsigned int)matId);
        }
        prd->origin = offset_ray(si.position, -faceNg);
    }
    else
    {
        prd->origin = offset_ray(si.position, faceNg);
    }
    prd->dir = sample_data.wi;
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
    prd->lastBsdfPdf = (prd->specularBounce) ? 1.0f : sample_data.pdf;
    prd->throughput *= sample_data.bsdf_over_pdf;
}
