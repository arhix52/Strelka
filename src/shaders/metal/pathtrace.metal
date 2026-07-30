#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "env_light_metal.h"

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h>
#include <strelka/material/bsdf.h>

using namespace metal;
using namespace raytracing;


struct PerRayData
{
    SamplerState sampler;
    uint32_t depth;
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
    float lastBsdfPdf;
    IorStack iorStack;
    bool specularBounce;
    bool shouldTerninate;
};

// Interpolates the vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection structure.
template<typename T, typename IndexType>
inline T interpolateVertexAttribute(device T *attributes,
                                    IndexType i0,
                                    IndexType i1,
                                    IndexType i2,
                                    float2 uv) {
    // Look up value for each vertex.
    const T T0 = attributes[i0];
    const T T1 = attributes[i1];
    const T T2 = attributes[i2];

    // Compute the sum of the vertex attributes weighted by the barycentric coordinates.
    // The barycentric coordinates sum to one.
    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

template<typename T>
inline T interpolateVertexAttribute(thread T *attributes, float2 uv) {
    // Look up the value for each vertex.
    const T T0 = attributes[0];
    const T T1 = attributes[1];
    const T T2 = attributes[2];

    // Compute the sum of the vertex attributes weighted by the barycentric coordinates.
    // The barycentric coordinates sum to one.
    return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

__attribute__((always_inline))
float3 transformPoint(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 1.0f)).xyz;
}

__attribute__((always_inline))
float3 transformDirection(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 0.0f)).xyz;
}

//  valid range of coordinates [-1; 1]
static float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

//  valid range of coordinates [-10; 10]
static float2 unpackUV(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

static __attribute__((always_inline)) float3 interpolateAttrib(const float3 attr1, const float3 attr2, const float3 attr3, const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) float2 interpolateAttrib(const float2 attr1, const float2 attr2, const float2 attr3, const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) bool all(const float3 v)
{
    return v.x != 0.0f && v.y != 0.0f && v.z != 0.0f;
}

__attribute__((always_inline))
float4x4 lerpMatrix(float4x4 a, float4x4 b, float t)
{
    float4x4 r;
    r[0] = mix(a[0], b[0], t);
    r[1] = mix(a[1], b[1], t);
    r[2] = mix(a[2], b[2], t);
    r[3] = mix(a[3], b[3], t);
    return r;
}

// Concentric disk mapping (Shirley & Chiu 1997)
float2 concentricDiskSample(float u1, float u2)
{
    float2 offset = float2(2.0f * u1 - 1.0f, 2.0f * u2 - 1.0f);
    if (offset.x == 0.0f && offset.y == 0.0f)
        return float2(0.0f, 0.0f);

    float theta, r;
    if (abs(offset.x) > abs(offset.y))
    {
        r = offset.x;
        theta = (M_PI_F / 4.0f) * (offset.y / offset.x);
    }
    else
    {
        r = offset.y;
        theta = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (offset.x / offset.y);
    }
    return float2(r * cos(theta), r * sin(theta));
}

// Sample regular polygon aperture (blades >= 3)
float2 samplePolygonAperture(float u1, float u2, int blades)
{
    float sectorAngle = 2.0f * M_PI_F / (float)blades;
    int sector = (int)(u1 * blades);
    if (sector >= blades) sector = blades - 1;
    float u = u1 * blades - (float)sector;

    float su = sqrt(u);
    float bary0 = 1.0f - su;
    float bary1 = u2 * su;

    float angle0 = sectorAngle * sector;
    float angle1 = sectorAngle * (sector + 1);

    float x = bary1 * cos(angle0) + (1.0f - bary0 - bary1) * cos(angle1);
    float y = bary1 * sin(angle0) + (1.0f - bary0 - bary1) * sin(angle1);
    return float2(x, y);
}

float2 sampleAperture(thread SamplerState& sampler, const constant Uniforms& params)
{
    float u1 = random<SampleDimension::eLensU>(sampler, params.samplerType);
    float u2 = random<SampleDimension::eLensV>(sampler, params.samplerType);

    float2 p;
    if (params.apertureBlades < 3)
        p = concentricDiskSample(u1, u2);
    else
        p = samplePolygonAperture(u1, u2, params.apertureBlades);

    if (params.bladeRotation != 0.0f)
    {
        float cosR = cos(params.bladeRotation);
        float sinR = sin(params.bladeRotation);
        p = float2(p.x * cosR - p.y * sinR, p.x * sinR + p.y * cosR);
    }

    p.y *= params.anamorphicRatio;
    return p;
}

void generateCameraRay(uint2 pixelIndex,
                        thread SamplerState& samplerRnd,
                        thread float3& origin,
                        thread float3& direction,
                        const constant Uniforms& params,
                        float motionTime)
{
    const float2 subpixel_jitter = {
        random<SampleDimension::ePixelX>(samplerRnd, params.samplerType),
        random<SampleDimension::ePixelY>(samplerRnd, params.samplerType)};
    float2 pixelPos {pixelIndex.x + subpixel_jitter.x, params.height - (pixelIndex.y + subpixel_jitter.y)};

    float2 dimension {(float)params.width, (float)params.height};
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    // Lens shift
    pixelNDC.x += params.shiftX * 2.0f;
    pixelNDC.y += params.shiftY * 2.0f;

    // Interpolate camera matrices for camera motion blur
    float4x4 clipToView = params.clipToView;
    float4x4 viewToWorld = params.viewToWorld;
    if (motionTime < 1.0f && params.enableCameraMotionBlur)
    {
        clipToView = lerpMatrix(params.prevClipToView, params.clipToView, motionTime);
        viewToWorld = lerpMatrix(params.prevViewToWorld, params.viewToWorld, motionTime);
    }

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    float4 viewSpace = clipToView * clip;

    float4 wdir = viewToWorld * float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = (viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    direction = normalize(wdir.xyz);

    // Thin lens depth of field
    if (params.useDof && params.lensRadius > 0.0f)
    {
        float3 camRight = float3(viewToWorld[0][0], viewToWorld[1][0], viewToWorld[2][0]);
        float3 camUp    = float3(viewToWorld[0][1], viewToWorld[1][1], viewToWorld[2][1]);
        float3 camFwd   = float3(-viewToWorld[0][2], -viewToWorld[1][2], -viewToWorld[2][2]);

        float t = params.focalDistance / max(dot(direction, camFwd), 1e-6f);
        float3 focalPoint = origin + direction * t;

        float2 lensSample = sampleAperture(samplerRnd, params) * params.lensRadius;
        origin += camRight * lensSample.x + camUp * lensSample.y;
        direction = normalize(focalPoint - origin);
    }
}

// Fill SurfaceInteraction from hit geometry and sample Material textures
void initSurfaceInteraction(
    thread SurfaceInteraction& si,
    const device Material& material,
    float3 worldPosition,
    float3 worldNormal,
    float3 geomNormal,
    float3 worldTangent,
    float3 worldBinormal,
    float2 uv,
    float3 rayDir)
{
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear);

    si.position       = worldPosition;
    si.shading_normal = worldNormal;
    si.geometry_normal = geomNormal;
    si.tangent        = worldTangent;
    si.bitangent      = worldBinormal;
    si.uv             = uv;
    si.wo             = -rayDir;
    si.front_face     = dot(geomNormal, -rayDir) > 0.0f;

    // Sample base color texture
    float3 baseColor = float3(material.base_color);
    if (!is_null_texture(material.baseColorTexture))
    {
        float4 texVal = material.baseColorTexture.sample(texSampler, uv);
        baseColor *= texVal.rgb;
    }
    si.albedo = baseColor;

    // Sample metallic-roughness texture (glTF: G = roughness, B = metallic)
    float resolvedRoughness = material.roughness;
    float resolvedMetallic = material.metallic;
    if (!is_null_texture(material.metallicRoughnessTexture))
    {
        float4 mrTex = material.metallicRoughnessTexture.sample(texSampler, uv);
        resolvedRoughness *= mrTex.g;
        resolvedMetallic *= mrTex.b;
    }

    // Sample normal map
    if (!is_null_texture(material.normalTexture))
    {
        float3 bumpNormal = material.normalTexture.sample(texSampler, uv).xyz * 2.0f - 1.0f;
        bumpNormal.xy *= material.normal_scale;
        float3x3 TBN = float3x3(worldTangent, worldBinormal, worldNormal);
        si.shading_normal = normalize(TBN * bumpNormal);
    }

    // Sample emission texture
    float3 emissionColor = float3(material.emission);
    if (!is_null_texture(material.emissionTexture))
    {
        float4 emTex = material.emissionTexture.sample(texSampler, uv);
        emissionColor *= emTex.rgb;
    }
    si.emission = emissionColor * material.emission_strength;

    // Fill remaining material parameters for bsdf_init
    MaterialParams matParams;
    matParams.roughness = resolvedRoughness;
    matParams.metallic = resolvedMetallic;
    matParams.ior = material.ior;
    matParams.transmission = material.transmission;
    matParams.clearcoat = material.clearcoat;
    matParams.clearcoat_roughness = material.clearcoat_roughness;
    matParams.anisotropy = material.anisotropy;
    matParams.specular = material.specular;
    matParams.specular_tint = material.specular_tint;
    matParams.material_type = material.material_type;
    matParams.thin_walled = material.thin_walled;
    matParams.dielectric_priority = material.dielectric_priority;

    // bsdf_init (Metal overload) clamps and finalizes derived values
    bsdf_init(si, matParams);
    // Restore texture-resolved values that bsdf_init may have overwritten
    si.roughness = max(resolvedRoughness, 0.0001f);
    si.metallic = saturate(resolvedMetallic);
}

bool traceOcclusion(
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    const float3 origin, 
    const float3 direction,
    const float tMin,
    const float tMax,
    const float motionTime)
{
    struct ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = direction;
    shadowRay.min_distance = tMin;
    shadowRay.max_distance = tMax;
    isect.accept_any_intersection(true);

    bool res = true;
    typename intersector<triangle_data, instancing, primitive_motion>::result_type intersection;
    intersection = isect.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime);
    if (intersection.type == intersection_type::none)
    {
        res = false;
    }
    isect.accept_any_intersection(false);
    return res;
}

float3 sampleLight(
    constant Uniforms& uniforms,
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    thread SamplerState& samplerRnd,
    device const UniformLight& light,
    thread SurfaceInteraction& si,
    thread float3& toLight,
    thread float& lightPdf,
    const float motionTime)
{
    LightSampleData lightSampleData = {};
    const float2 uv = float2(random<SampleDimension::eLightPointX>(samplerRnd, uniforms.samplerType), random<SampleDimension::eLightPointY>(samplerRnd, uniforms.samplerType));
    switch (light.type)
    {
    case 0:
        if (uniforms.rectLightSamplingMethod == 0)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case 2:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;
    case 3:
        lightSampleData = SampleDistantLight(light, uv, si.position);
        break;
    }

    toLight = lightSampleData.L;
    float3 Li = float3(light.color);

    if (dot(si.shading_normal, lightSampleData.L) > 0.0f && -dot(lightSampleData.L, lightSampleData.normal) > 0.001f && all(Li))
    {
        const bool occluded = traceOcclusion(accelerationStructure, isect, si.position, lightSampleData.L,
                                             0.001f, // tmin
                                             lightSampleData.distToLight - 1e-5f, // tmax
                                             motionTime
        );
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        return visibility * Li * saturate(dot(si.shading_normal, lightSampleData.L));
    }
    lightPdf = 0.0f;
    return float3(0.0f, 0.0f, 0.0f);
}

__attribute__((always_inline))
int __float_as_int(float x)
{
    return as_type<int>(x);
}
__attribute__((always_inline))
float __int_as_float(int x)
{
    return as_type<float>(x);
}

static float3 offset_ray(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                             __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                             __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                       abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                       abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

float3 sampleEnvLightNEE(
    constant Uniforms& uniforms,
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    thread SamplerState& samplerRnd,
    thread SurfaceInteraction& si,
    thread float3& toLight,
    thread float& lightPdf,
    device const float* envCdfX,
    device const float* envCdfY,
    texture2d<float> envMapTexture,
    const float motionTime)
{
    const float2 xi = float2(
        random<SampleDimension::eLightPointX>(samplerRnd, uniforms.samplerType),
        random<SampleDimension::eLightPointY>(samplerRnd, uniforms.samplerType));

    float envPdf = 0.0f;
    float3 dir = sampleEnvMap(xi,
                              envCdfX, envCdfY,
                              uniforms.envMapWidth, uniforms.envMapHeight,
                              uniforms.envMapRotation,
                              envPdf);

    toLight = dir;
    lightPdf = envPdf;

    if (envPdf <= 0.0f)
        return float3(0.0f);

    if (dot(si.shading_normal, dir) <= 0.0f)
        return float3(0.0f);

    const bool occluded = traceOcclusion(
        accelerationStructure, isect,
        offset_ray(si.position, si.geometry_normal),
        dir,
        0.001f,
        1e16f,
        motionTime);

    if (occluded)
        return float3(0.0f);

    constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
    const float2 uv = dirToEnvUV(dir, uniforms.envMapRotation);
    const float4 envSample = envMapTexture.sample(envSampler, uv);
    float3 Li = envSample.xyz;
    Li *= uniforms.envMapIntensity * float3(uniforms.envMapColorTint);

    return Li * max(dot(si.shading_normal, dir), 0.0f);
}

float3 estimateDirectLighting(
    constant Uniforms& uniforms,
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    const uint32_t numLights,
    device UniformLight* lights,
    thread SamplerState& samplerRnd,
    thread SurfaceInteraction& si,
    thread float3& toLight,
    thread float& lightPdf,
    device const float* envCdfX,
    device const float* envCdfY,
    texture2d<float> envMapTexture,
    const float motionTime)
{
    if (uniforms.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);

        if (numLights == 0 || u >= 0.5f)
        {
            // Sample environment map
            const float selectionPdf = (numLights > 0) ? 0.5f : 1.0f;
            const float3 r = sampleEnvLightNEE(uniforms, accelerationStructure, isect,
                samplerRnd, si, toLight, lightPdf, envCdfX, envCdfY, envMapTexture, motionTime);
            lightPdf *= selectionPdf;
            return r;
        }
        else
        {
            // Sample local light (remap u from [0, 0.5) to [0, 1))
            const float remappedU = u * 2.0f;
            const uint32_t lightId = min((uint32_t)(numLights * remappedU), numLights - 1);
            const float lightSelectionPdf = 0.5f / numLights;
            device const UniformLight& currLight = lights[lightId];
            const float3 r = sampleLight(uniforms, accelerationStructure, isect, samplerRnd, currLight, si, toLight, lightPdf, motionTime);
            lightPdf *= lightSelectionPdf;
            return r;
        }
    }

    // No env map and no analytic lights: nothing to connect to. Falling through
    // would divide by numLights == 0, produce a NaN light PDF, and trip the
    // isnan() guard in the caller that paints the pixel bright red.
    if (numLights == 0)
    {
        toLight = float3(0.0f);
        lightPdf = 0.0f;
        return float3(0.0f);
    }

    float u = random<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);
    const uint32_t lightId = min((uint32_t)(numLights * u), numLights - 1);
    const float lightSelectionPdf = 1.0f / numLights;
    device const UniformLight& currLight = lights[lightId];
    const float3 r = sampleLight(uniforms, accelerationStructure, isect, samplerRnd, currLight, si, toLight, lightPdf, motionTime);
    lightPdf *= lightSelectionPdf;
    return r;
}

// Main ray tracing kernel.
kernel void raytracingKernel(
    uint2                                                      tid                   [[thread_position_in_grid]],
    constant Uniforms&                                         uniforms              [[buffer(0)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor* instances             [[buffer(1)]],
    acceleration_structure<instancing, primitive_motion>        accelerationStructure [[buffer(2)]],
    device UniformLight* lights                                                      [[buffer(3)]],
    device Material* materials                                                       [[buffer(4)]],
    device float4* res                                                               [[buffer(5)]],
    device float4* accum                                                             [[buffer(6)]],
    device const char* prevVertexBuffer                                              [[buffer(7)]],
    device const uint32_t* indexBuffer                                               [[buffer(8)]],
    device const InstanceData* instanceDataBuffer                                    [[buffer(9)]],
    device const float* envCdfX                                                      [[buffer(10)]],
    device const float* envCdfY                                                      [[buffer(11)]],
    constant uint32_t&                                         tileOffsetY           [[buffer(12)]],
    texture2d<float>                                           envMapTexture         [[texture(0)]]
    )
{
    // The host splits a frame into horizontal bands, each dispatched from its own
    // command buffer, so that no single submission monopolises the GPU. tid.y is
    // band-local; tileOffsetY maps it back to the full image.
    const uint2 pixel = uint2(tid.x, tid.y + tileOffsetY);
    if (pixel.x >= uniforms.width || pixel.y >= uniforms.height)
    {
        return;
    }
    const uint32_t linearPixelIndex = pixel.y * uniforms.width + pixel.x;

    // samples_per_launch paths are traced per dispatch and averaged below.
    const uint32_t sampleCount = max(uniforms.samples_per_launch, 1u);
    float3 radianceSum = float3(0.0f);

    for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx)
    {

    PerRayData prd{};
    prd.radiance = float3(0.0f);
    prd.throughput = float3(1.0f);
    ior_stack_init(prd.iorStack);
    prd.depth = 0;
    prd.specularBounce = false;
    prd.lastBsdfPdf = 0.0f;
    prd.sampler = initSampler(linearPixelIndex, uniforms.subframeIndex + sampleIdx, 0u);

    DebugMode debugMode = (DebugMode) uniforms.debug;

    // Sample motion blur time per ray
    float motionTime = 0.0f;
    if (uniforms.enableMotionBlur)
    {
        motionTime = random<SampleDimension::eTime>(prd.sampler, uniforms.samplerType);
        if (!uniforms.isMotionBlurVisible)
            motionTime = 1.0f; // show current frame only (t=1 → kf1 = current VB)
    }

    generateCameraRay(pixel, prd.sampler, prd.origin, prd.direction, uniforms, motionTime);

    // Create intersector once outside the bounce loop (primitive_motion for native motion BVH)
    intersector<triangle_data, instancing, primitive_motion> i;
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    typename intersector<triangle_data, instancing, primitive_motion>::result_type intersection;

    while (prd.depth < uniforms.maxDepth)
    {
        ray ray;
        ray.min_distance = 0.0f;
        ray.max_distance = INFINITY;
        ray.origin = prd.origin;
        ray.direction = prd.direction;

        i.accept_any_intersection(false);
        intersection = i.intersect(ray, accelerationStructure, RAY_MASK_PRIMARY, motionTime);

        // Stop if the ray didn't hit anything and has bounced out of the scene.
        if (intersection.type == intersection_type::none)
        {
            // Miss
            if (uniforms.hasEnvMap)
            {
                constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
                const float2 envUV = dirToEnvUV(prd.direction, uniforms.envMapRotation);
                const float4 envSample = envMapTexture.sample(envSampler, envUV);
                float3 envColor = envSample.xyz;
                envColor *= uniforms.envMapIntensity * float3(uniforms.envMapColorTint);

                if (prd.depth == 0 || prd.specularBounce)
                {
                    prd.radiance += prd.throughput * envColor;
                }
                else
                {
                    const float envPdf = envMapPdf(prd.direction,
                                                   envCdfX, envCdfY,
                                                   uniforms.envMapWidth, uniforms.envMapHeight,
                                                   uniforms.envMapRotation);
                    const float envSelectionPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
                    const float effectiveEnvPdf = envPdf * envSelectionPdf;
                    if (effectiveEnvPdf > 0.0f)
                    {
                        const float misWeight = misWeightBalance(prd.lastBsdfPdf, effectiveEnvPdf);
                        prd.radiance += prd.throughput * envColor * misWeight;
                    }
                }
            }
            else
            {
                prd.radiance += prd.throughput * uniforms.missColor;
            }
            prd.throughput = float3(0.0f);
            break;
        }
        else
        {
            // Load instance descriptor once into registers
            const uint32_t instanceIndex = intersection.instance_id;
            const auto inst = instances[instanceIndex];
            if (inst.mask == GEOMETRY_MASK_LIGHT)
            {
                // Light hit
                const float3 hitPoint = ray.origin + ray.direction * intersection.distance;
                device const UniformLight& currLight = lights[inst.userID];
                const float3 lightNormal = calcLightNormal(currLight, hitPoint);
                if (-dot(prd.direction, lightNormal) > 0.0f)
                {
                    if (prd.depth == 0 || prd.specularBounce)
                    {
                        prd.radiance += prd.throughput * float3(currLight.color) * -dot(prd.direction, lightNormal);
                    }
                    else
                    {
                        // Account for env map selection probability in light PDF
                        const float lightSelectionPdf = uniforms.hasEnvMap
                            ? 0.5f / (float)uniforms.numLights
                            : 1.0f / (float)uniforms.numLights;
                        const float lightPdf = getLightPdf(currLight, hitPoint, ray.origin) * lightSelectionPdf;
                        const float misWeight = misWeightBalance(prd.lastBsdfPdf, lightPdf);
                        prd.radiance += prd.throughput * float3(currLight.color) * -dot(prd.direction, lightNormal) * misWeight;
                    }
                }
                prd.throughput = float3(0.0f);
                // stop tracing

                break;
            }

            const Triangle triangle = *(const device Triangle*)intersection.primitive_data;

            // Per-primitive positions are from keyframe 1 (current VB) only.
            // For motion blur, we also need keyframe 0 positions to:
            //  (a) compute correct worldPosition via interpolated positions + barycentrics
            //  (b) compute correct geomNormal from the interpolated triangle
            float3 p0, p1, p2;
            float3 n0, n1, n2;
            float3 t0, t1, t2;

            if (uniforms.enableMotionBlur && motionTime < 1.0f &&
                prevVertexBuffer && indexBuffer && instanceDataBuffer)
            {
                const uint32_t geomIndex = inst.accelerationStructureIndex;
                const uint32_t primitiveId = intersection.primitive_id;
                const InstanceData instData = instanceDataBuffer[geomIndex];

                const uint32_t i0 = indexBuffer[instData.indexOffset + primitiveId * 3 + 0];
                const uint32_t i1 = indexBuffer[instData.indexOffset + primitiveId * 3 + 1];
                const uint32_t i2 = indexBuffer[instData.indexOffset + primitiveId * 3 + 2];

                // Scene::Vertex layout (32 bytes):
                //   offset 0:  pos     (packed_float3, 12 bytes)
                //   offset 12: tangent (uint32_t, 4 bytes)
                //   offset 16: normal  (uint32_t, 4 bytes)
                constexpr uint32_t vtxStride  = 32;
                constexpr uint32_t tangentOff = 12;
                constexpr uint32_t normalOff  = 16;

                // Previous frame positions (keyframe 0 = prevVB)
                const float3 p0_prev = float3(*(device const packed_float3*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride));
                const float3 p1_prev = float3(*(device const packed_float3*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride));
                const float3 p2_prev = float3(*(device const packed_float3*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride));

                // Interpolate positions: BVH kf0=prevVB at t=0, kf1=VB at t=1
                // mix(a,b,t) = a*(1-t)+b*t → mix(prev, current, t) gives prev at t=0, current at t=1
                p0 = mix(p0_prev, triangle.positions[0], motionTime);
                p1 = mix(p1_prev, triangle.positions[1], motionTime);
                p2 = mix(p2_prev, triangle.positions[2], motionTime);

                // Previous frame normals
                const float3 n0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + normalOff));
                const float3 n1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + normalOff));
                const float3 n2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + normalOff));

                // Interpolate normals: t=0 → prev (matches kf0=prevVB), t=1 → current (matches kf1=VB)
                n0 = mix(n0_prev, unpackNormal(triangle.normals[0]), motionTime);
                n1 = mix(n1_prev, unpackNormal(triangle.normals[1]), motionTime);
                n2 = mix(n2_prev, unpackNormal(triangle.normals[2]), motionTime);

                // Previous frame tangents
                const float3 t0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + tangentOff));
                const float3 t1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + tangentOff));
                const float3 t2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + tangentOff));

                // Interpolate tangents: same direction as normals
                t0 = mix(t0_prev, unpackNormal(triangle.tangent[0]), motionTime);
                t1 = mix(t1_prev, unpackNormal(triangle.tangent[1]), motionTime);
                t2 = mix(t2_prev, unpackNormal(triangle.tangent[2]), motionTime);
            }
            else
            {
                p0 = triangle.positions[0];
                p1 = triangle.positions[1];
                p2 = triangle.positions[2];
                n0 = unpackNormal(triangle.normals[0]);
                n1 = unpackNormal(triangle.normals[1]);
                n2 = unpackNormal(triangle.normals[2]);
                t0 = unpackNormal(triangle.tangent[0]);
                t1 = unpackNormal(triangle.tangent[1]);
                t2 = unpackNormal(triangle.tangent[2]);
            }

            const float2 uv0 = unpackUV(triangle.uv[0]);
            const float2 uv1 = unpackUV(triangle.uv[1]);
            const float2 uv2 = unpackUV(triangle.uv[2]);

            // Build transform from local instance copy (avoids 12 scattered device reads)
            const float4x4 objectToWorldSpaceTransform = float4x4(
                float4(float3(inst.transformationMatrix[0]), 0.0f),
                float4(float3(inst.transformationMatrix[1]), 0.0f),
                float4(float3(inst.transformationMatrix[2]), 0.0f),
                float4(float3(inst.transformationMatrix[3]), 1.0f));

            const float2 barycentrics = intersection.triangle_barycentric_coord;

            // Use ray equation for world position — this is always correct for the
            // motion-interpolated geometry, unlike computing from per-primitive positions
            // which are only from keyframe 1.
            const float3 worldPosition = ray.origin + ray.direction * intersection.distance;
            const float2 uv = interpolateAttrib(uv0, uv1, uv2, barycentrics);

            const float3 objectNormal = normalize(interpolateAttrib(n0, n1, n2, barycentrics));
            const float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldSpaceTransform));

            const float3 worldTangent = normalize(transformDirection(normalize(interpolateAttrib(t0, t1, t2, barycentrics)), objectToWorldSpaceTransform));
            const float3 worldBinormal = cross(worldNormal, worldTangent);

            // Geometric normal from interpolated positions (correct for motion-blurred triangle)
            float3 geomNormal = cross(p1 - p0, p2 - p0);
            geomNormal = normalize(transformDirection(geomNormal, objectToWorldSpaceTransform));

            const uint32_t materialId = inst.userID;

            SurfaceInteraction si;
            initSurfaceInteraction(si, materials[materialId],
                worldPosition, worldNormal, geomNormal,
                worldTangent, worldBinormal, uv,
                prd.direction);

            if (debugMode == DebugMode::eMotionBlur)
            {
                float3 nDelta = n0 - unpackNormal(triangle.normals[0]);
                float deltaMag = length(nDelta);
                prd.radiance = float3(motionTime, clamp(deltaMag * 10.0f, 0.0f, 1.0f), 0.0f);
                break;
            }

            if (debugMode == DebugMode::eNormal)
            {
                prd.radiance = (si.shading_normal + float3(1.0f)) * 0.5f;
                break;
            }

            // Add emission
            if (si.emission.x > 0.0f || si.emission.y > 0.0f || si.emission.z > 0.0f)
            {
                prd.radiance += prd.throughput * si.emission;
            }

            // Set exterior IOR from the IOR stack for nested dielectrics
            bool entering = si.front_face;
            if (entering)
            {
                si.exterior_ior = ior_stack_current_ior(prd.iorStack);
            }
            else
            {
                si.exterior_ior = ior_stack_peek_after_pop(prd.iorStack, si.dielectric_priority);
            }

            // Sample BSDF to determine event type
            const float z1 = random<SampleDimension::eBSDF0>(prd.sampler, uniforms.samplerType);
            const float z2 = random<SampleDimension::eBSDF1>(prd.sampler, uniforms.samplerType);
            const float z3 = random<SampleDimension::eBSDF2>(prd.sampler, uniforms.samplerType);
            const float z4 = random<SampleDimension::eBSDF3>(prd.sampler, uniforms.samplerType);
            float4 xi = float4(z1, z2, z3, z4);

            BsdfSampleResult sampleResult = bsdf_sample(si, xi);

            if (sampleResult.event_type == BSDF_EVENT_ABSORB)
            {
                prd.throughput = float3(0.0f);
                break;
            }

            prd.specularBounce = ((sampleResult.event_type & BSDF_EVENT_SPECULAR) != 0);

            // Direct lighting (NEE) for diffuse/glossy events
            if ((sampleResult.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) &&
                (uniforms.numLights > 0 || uniforms.hasEnvMap))
            {
                float3 toLight;
                float lightPdf = 0.0f;
                const float3 radiance = estimateDirectLighting(uniforms, accelerationStructure, i,
                    uniforms.numLights, lights,
                    prd.sampler, si, toLight, lightPdf,
                    envCdfX, envCdfY, envMapTexture, motionTime);

                // `> 0` rather than `!= 0`: a NaN PDF must not be treated as valid.
                const bool isNextEventValid = ((dot(toLight, si.shading_normal) > 0.0f) == si.front_face) && lightPdf > 0.0f;
                if (isNextEventValid)
                {
                    BsdfEvalResult evalResult = bsdf_eval(si, toLight);
                    if (isnan(lightPdf) || isnan(evalResult.pdf))
                    {
                        prd.radiance = float3(1000000.0f, 0.0f, 0.0f);
                        prd.throughput = float3(0.0f);
                        break;
                    }
                    if (evalResult.pdf > 0.0f)
                    {
                        const float3 radianceOverPdf = radiance / lightPdf;
                        const float misWeight = misWeightBalance(lightPdf, evalResult.pdf);
                        prd.radiance += prd.throughput * radianceOverPdf * misWeight * evalResult.bsdf;
                    }
                }
            }

            // Setup next path segment
            // Face normal oriented toward the incoming ray (wo)
            float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f)
                          ? si.geometry_normal : -si.geometry_normal;
            if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0)
            {
                if (entering)
                    ior_stack_push(prd.iorStack, si.dielectric_priority, si.ior);
                else
                    ior_stack_pop(prd.iorStack, si.dielectric_priority);
                prd.origin = offset_ray(si.position, -faceNg);
            }
            else
            {
                prd.origin = offset_ray(si.position, faceNg);
            }
            prd.direction = normalize(sampleResult.wi);
            prd.throughput *= sampleResult.bsdf_over_pdf;
            prd.lastBsdfPdf = (prd.specularBounce) ? 1.0f : sampleResult.pdf;

            if (dot(prd.throughput, prd.throughput) < 1e-4f)
            {
                break;
            }

            if (prd.depth > 3)
            {
                // The survival probability must be <= 1. Without the clamp a
                // throughput above 1 (bright albedo, emissive gain) never
                // terminates yet still gets divided by p > 1, silently losing
                // energy on every bounce past the third.
                const float p = min(max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z)), 1.0f);
                if (random<SampleDimension::eRussianRoulette>(prd.sampler, uniforms.samplerType) > p)
                {
                    break;
                }
                prd.throughput *= 1.0f / max(p, 1e-5f);
            }
        }
        ++prd.depth;
        ++prd.sampler.depth;
    }

    radianceSum += prd.radiance;

    } // sample loop

    float3 result = radianceSum / static_cast<float>(sampleCount);

    if (uniforms.enableAccumulation)
    {
        float3 accum_color = result;

        if (uniforms.subframeIndex > 0)
        {
            // subframeIndex counts *samples* already folded into the accumulator,
            // and this launch contributes sampleCount more. Merging two means of
            // n and m samples weights the new mean by m / (n + m); the previous
            // 1 / (subframeIndex + 1) was only correct for sampleCount == 1 and
            // biased the running average for any larger SPP-per-subframe.
            const float a = static_cast<float>(sampleCount) /
                            static_cast<float>(uniforms.subframeIndex + sampleCount);
            const float3 accum_color_prev = float3(accum[linearPixelIndex]);
            accum_color = mix(accum_color_prev, accum_color, a);
        }
        accum[linearPixelIndex] = float4(accum_color, 1.0f);
        result = accum_color;
    }

    res[linearPixelIndex] = float4(result, 1.0f);
}
