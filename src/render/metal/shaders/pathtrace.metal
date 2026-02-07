#include <metal_stdlib>
#include <simd/simd.h>

#include "random.h"
#include "lights.h"
#include "tonemappers.h"

#include "ShaderTypes.h"

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
    bool inside;
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

    // Interpolate camera matrices for camera motion blur (t=0 → current, t=1 → previous)
    float4x4 clipToView = params.clipToView;
    float4x4 viewToWorld = params.viewToWorld;
    if (motionTime > 0.0f && params.enableCameraMotionBlur)
    {
        clipToView = lerpMatrix(params.clipToView, params.prevClipToView, motionTime);
        viewToWorld = lerpMatrix(params.viewToWorld, params.prevViewToWorld, motionTime);
    }

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    float4 viewSpace = clipToView * clip;

    float4 wdir = viewToWorld * float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = (viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    direction = normalize(wdir.xyz);
}

struct MaterialState
{
    float3 position;
    float3 normal; // shading normal (normal mapping)
    float3 geom_normal; // triangle normal
    float3 tangent_u;
    float3 tangent_v;
    float3 diffuse;
    float2 textureCoordinates;
};

struct MaterialEval
{
    // in
    float3 ior1;
    float3 ior2;
    float3 outDir; // MDL: k1
    float3 inDir; // MDL: k2

    // out
    float3 bsdf_diffuse;
    float3 bsdf_glossy;
    float pdf;
};

struct MaterialSample
{
    // in
    float3 ior1;
    float3 ior2;
    float3 k1; // MDL: k1
    float4 xi; // rnd
    // out
    float3 k2;
    float3 bsdf_over_pdf;
    float pdf;
    int event_type;
};

void materialInit(thread MaterialState& state, const device Material& material)
{
    constexpr sampler textureSampler (mag_filter::linear, min_filter::linear);
    state.diffuse = material.diffuse;
    if (!is_null_texture(material.diffuseTexture))
    {
        const float4 diffuseFromTex = material.diffuseTexture.sample(textureSampler, state.textureCoordinates);
        state.diffuse *= diffuseFromTex.rgb;
    }
    if (!is_null_texture(material.normalTexture))
    {
        const float3 bumpNormal = material.normalTexture.sample(textureSampler, state.textureCoordinates).xyz * 2.0 - 1.0;
        const float3x3 TBN = float3x3(state.tangent_u, state.tangent_v, state.normal);
        state.normal = normalize(TBN * bumpNormal);
    }
}

void materialEvaluate(thread MaterialEval& data, thread const MaterialState& state)
{
    data.bsdf_diffuse = state.diffuse * dot(state.normal, data.inDir) * M_1_PI_F; 
    data.bsdf_glossy = float3(0.0f);
    data.pdf = dot(state.normal, data.inDir) * M_1_PI_F;
}

void materialSample(thread MaterialSample& data, thread MaterialState& state)
{
    // Source: https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_16.pdf
    {
        float a = 1.0f - 2.0f * data.xi.x;
        float b = sqrt(1.0f - a * a);
        float phi = data.xi.y * 2.0f * M_PI_F;
        data.k2.x = state.normal.x + b * cos(phi);
        data.k2.y = state.normal.y + b * sin(phi);
        data.k2.z = state.normal.z + a;
        data.pdf = a * M_1_PI_F;
    }
    data.bsdf_over_pdf = state.diffuse;
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
    thread MaterialState& state, 
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
            lightSampleData = SampleRectLightUniform(light, uv, state.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, state.position);
        }
        break;
        // case 1:
        //     lightSampleData = SampleDiscLight(light, float2(rand(rngState), rand(rngState)), state.position);
        //     break;
    case 2:
        lightSampleData = SampleSphereLight(light, uv, state.position);
        break;
    case 3:
        lightSampleData = SampleDistantLight(light, uv, state.position);
        break;
    }

    toLight = lightSampleData.L;
    float3 Li = float3(light.color);

    // TODO: Added here because advanced light sampler produces NaNs in close to orthogonal cases -dot(lightSampleData.L, lightSampleData.normal) > 0.001f
    if (dot(state.normal, lightSampleData.L) > 0.0f && -dot(lightSampleData.L, lightSampleData.normal) > 0.001f && all(Li))
    {
        const bool occluded = traceOcclusion(accelerationStructure, isect, state.position, lightSampleData.L,
                                             0.001f, // tmin
                                             lightSampleData.distToLight - 1e-5f, // tmax
                                             motionTime
        );
        // bool occluded = false;
        float visibility = occluded ? 0.0f : 1.0f;
        lightPdf = lightSampleData.pdf;
        return visibility * Li * saturate(dot(state.normal, lightSampleData.L));
    }
    lightPdf = 0.0f;
    return float3(0.0f, 0.0f, 0.0f);
}

float3 estimateDirectLighting(
    constant Uniforms& uniforms,
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    const uint32_t numLights,
    device UniformLight* lights,
    thread SamplerState& samplerRnd,
    thread MaterialState& state, 
    thread float3& toLight,
    thread float& lightPdf,
    const float motionTime)
{
    float u = random<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);

    const uint32_t lightId = min((uint32_t)(numLights * u), numLights - 1);
    const float lightSelectionPdf = 1.0f / numLights;
    device const UniformLight& currLight = lights[lightId];
    const float3 r = sampleLight(uniforms, accelerationStructure, isect, samplerRnd, currLight, state, toLight, lightPdf, motionTime);
    lightPdf *= lightSelectionPdf;
    return r;
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
    device const InstanceData* instanceDataBuffer                                    [[buffer(9)]]
    )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) 
    {
        return;
    }
    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;

    PerRayData prd{};
    prd.radiance = float3(0.0f);
    prd.throughput = float3(1.0f);
    prd.inside = false;
    prd.depth = 0;
    prd.specularBounce = false;
    prd.lastBsdfPdf = 0.0f;
    prd.sampler = initSampler(linearPixelIndex, uniforms.subframeIndex, 0u);

    DebugMode debugMode = (DebugMode) uniforms.debug;

    // Sample motion blur time per ray
    float motionTime = 0.0f;
    if (uniforms.enableMotionBlur)
    {
        motionTime = random<SampleDimension::eTime>(prd.sampler, uniforms.samplerType);
        if (!uniforms.isMotionBlurVisible)
            motionTime = 0.0f; // show only current frame (t=0 → current)
    }

    generateCameraRay(tid, prd.sampler, prd.origin, prd.direction, uniforms, motionTime);

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
            prd.radiance += prd.throughput * uniforms.missColor;
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
                        const float lightPdf = getLightPdf(currLight, hitPoint, ray.origin) / (uniforms.numLights);
                        const float misWeight = misWeightBalance(prd.lastBsdfPdf, lightPdf);
                        prd.radiance += prd.throughput * float3(currLight.color) * -dot(prd.direction, lightNormal) * misWeight;
                    }
                }
                prd.throughput = float3(0.0f);
                // stop tracing

                break;
            }

            const Triangle triangle = *(const device Triangle*)intersection.primitive_data;

            // Positions: the motion BVH already intersected the interpolated triangle,
            // so barycentrics are correct for the motion-time triangle. Use the per-primitive
            // positions from keyframe 1 (current VB) — the actual hit position is computed
            // via barycentrics on the interpolated geometry by Metal.
            const float3 p0 = triangle.positions[0];
            const float3 p1 = triangle.positions[1];
            const float3 p2 = triangle.positions[2];

            // Normals and tangents: Metal only interpolates vertex positions for BVH,
            // not per-primitive data. Manually interpolate normals & tangents from prevVB.
            float3 n0, n1, n2;
            float3 t0, t1, t2;

            if (uniforms.enableMotionBlur && motionTime > 0.0f &&
                prevVertexBuffer && indexBuffer && instanceDataBuffer)
            {
                // Read previous frame normals/tangents and interpolate with current
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

                // Previous frame normals
                const float3 n0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + normalOff));
                const float3 n1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + normalOff));
                const float3 n2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + normalOff));

                // Interpolate normals: t=0 → current (from Triangle), t=1 → previous
                n0 = mix(unpackNormal(triangle.normals[0]), n0_prev, motionTime);
                n1 = mix(unpackNormal(triangle.normals[1]), n1_prev, motionTime);
                n2 = mix(unpackNormal(triangle.normals[2]), n2_prev, motionTime);

                // Previous frame tangents
                const float3 t0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + tangentOff));
                const float3 t1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + tangentOff));
                const float3 t2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + tangentOff));

                // Interpolate tangents
                t0 = mix(unpackNormal(triangle.tangent[0]), t0_prev, motionTime);
                t1 = mix(unpackNormal(triangle.tangent[1]), t1_prev, motionTime);
                t2 = mix(unpackNormal(triangle.tangent[2]), t2_prev, motionTime);
            }
            else
            {
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
            const float3 worldPosition = transformPoint(interpolateAttrib(p0, p1, p2, barycentrics), objectToWorldSpaceTransform);
            const float2 uv = interpolateAttrib(uv0, uv1, uv2, barycentrics);

            const float3 objectNormal = normalize(interpolateAttrib(n0, n1, n2, barycentrics));
            const float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldSpaceTransform));

            const float3 worldTangent = normalize(transformDirection(normalize(interpolateAttrib(t0, t1, t2, barycentrics)), objectToWorldSpaceTransform));
            const float3 worldBinormal = cross(worldNormal, worldTangent);

            float3 geomNormal = cross(p1 - p0, p2 - p0);
            geomNormal = normalize(transformDirection(geomNormal, objectToWorldSpaceTransform));

            MaterialState matState;

            matState.position = worldPosition;
            matState.normal = worldNormal;
            matState.tangent_u = worldTangent;
            matState.tangent_v = worldBinormal;
            matState.geom_normal = geomNormal;
            matState.textureCoordinates = uv;

            const uint32_t materialId = inst.userID;
            materialInit(matState, materials[materialId]);

            float3 toLight; // return value for estimateDirectLighting()
            float lightPdf = 0.0f; // return value for estimateDirectLighting()
            const float3 radiance = estimateDirectLighting(uniforms, accelerationStructure, i,
                uniforms.numLights, lights, 
                prd.sampler, matState, toLight, lightPdf, motionTime);
            
            if (debugMode == DebugMode::eMotionBlur)
            {
                // Red = motionTime, Green = normal delta (shows skeletal motion magnitude)
                float3 nDelta = n0 - unpackNormal(triangle.normals[0]);
                float deltaMag = length(nDelta);
                prd.radiance = float3(motionTime, clamp(deltaMag * 10.0f, 0.0f, 1.0f), 0.0f);
                break;
            }

            if (debugMode == DebugMode::eNormal)
            {
                prd.radiance = (matState.normal + float3(1.0f)) * 0.5f;
                break;
            }
            
            const bool isNextEventValid = ((dot(toLight, matState.normal) > 0.0f) != prd.inside) && lightPdf != 0.0f;
            if (isNextEventValid)
            {
                const float3 radianceOverPdf = radiance / lightPdf;
                
                MaterialEval evalData {};
                // evalData.ior1 = ior1;
                // evalData.ior2 = ior2;
                evalData.outDir = -prd.direction;
                evalData.inDir = toLight;

                materialEvaluate(evalData, matState);
                if (isnan(lightPdf))
                {
                    // ERROR, terminate tracing;
                    prd.radiance = float3(1000000.0f, 0.0f, 0.0f);
                    prd.throughput = float3(0.0f);
                    break;
                }
                if (evalData.pdf > 0.0f)
                {
                    const float misWeight = misWeightBalance(lightPdf, evalData.pdf);
                    const float3 w = prd.throughput * radianceOverPdf * misWeight;
                    prd.radiance += w * evalData.bsdf_diffuse;
                }
            }

            const float z1 = random<SampleDimension::eBSDF0>(prd.sampler, uniforms.samplerType);
            const float z2 = random<SampleDimension::eBSDF1>(prd.sampler, uniforms.samplerType);
            const float z3 = random<SampleDimension::eBSDF2>(prd.sampler, uniforms.samplerType);
            const float z4 = random<SampleDimension::eBSDF3>(prd.sampler, uniforms.samplerType);

            MaterialSample sampleData {};
            // sampleData.ior1 = ior1;
            // sampleData.ior2 = ior2;
            sampleData.k1 = -prd.direction;
            sampleData.xi = float4(z1, z2, z3, z4);

            materialSample(sampleData, matState);

            prd.origin = offset_ray(worldPosition, matState.normal * (prd.inside ? -1.0f : 1.0f));
            prd.direction = normalize(sampleData.k2);
            prd.throughput *= sampleData.bsdf_over_pdf;
            prd.lastBsdfPdf = (prd.specularBounce) ? 1.0f : sampleData.pdf;

            if (dot(prd.throughput, prd.throughput) < 1e-4f)
            {
                break;
            }

            if (prd.depth > 3)
            {
                const float p = max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z));
                if (random<SampleDimension::eRussianRoulette>(prd.sampler, uniforms.samplerType) > p)
                {
                    break;
                }
                prd.throughput *= 1.0f / (p + 1e-5f);
            }
        }
        ++prd.depth;
        ++prd.sampler.depth;
    }

    float3 result = prd.radiance;

    if (uniforms.enableAccumulation)
    {
        float3 accum_color = result / static_cast<float>(uniforms.samples_per_launch);

        if (uniforms.subframeIndex > 0)
        {
            const float a = 1.0f / static_cast<float>(uniforms.subframeIndex + 1);
            const float3 accum_color_prev = float3(accum[linearPixelIndex]);
            accum_color = mix(accum_color_prev, accum_color, a);
        }
        accum[linearPixelIndex] = float4(accum_color, 1.0f);
        result = accum_color;
    }

    res[linearPixelIndex] = float4(result, 1.0f);
}
