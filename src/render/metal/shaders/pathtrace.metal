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
    uint32_t rndSeed;
    uint32_t depth;
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
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
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) / 511.99999f * 2.0f - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) / 511.99999f * 2.0f - 1.0f;
    normal.x = (val & 0x000003ff) / 511.99999f * 2.0f - 1.0f;
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

static __attribute__((always_inline)) float3 interpolateAttrib(thread const float3& attr1, thread const float3& attr2, thread const float3& attr3, thread const float2& bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) float2 interpolateAttrib(thread const float2& attr1, thread const float2& attr2, thread const float2& attr3, thread const float2& bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) bool all(thread const float3& v)
{
    return v.x != 0.0f && v.y != 0.0f && v.z != 0.0f;
}

void generateCameraRay(uint2 pixelIndex,
                        uint seed,
                        thread float3& origin,
                        thread float3& direction,
                        const constant Uniforms& params)
{
    const float2 subpixel_jitter = float2(rnd(seed), rnd(seed));
    float2 pixelPos {pixelIndex.x + subpixel_jitter.x, pixelIndex.y + subpixel_jitter.y};

    float2 dimension {(float)params.width, (float)params.height};
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    float4 viewSpace = params.clipToView * clip;

    float4 wdir = params.viewToWorld * float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

    origin = (params.viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    direction = normalize(wdir.xyz);
}

struct MaterialState
{
    float3 position;
    float3 normal; // shading normal (normal mapping)
    float3 geom_normal; // triangle normal
    float3 diffuse;
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
    int event_type;
};

void materialEvaluate(thread MaterialEval& data, thread const MaterialState& state)
{
    // const float M_PIf = 3.1415926f;
    data.bsdf_diffuse = state.diffuse * dot(state.normal, data.inDir) / M_PIf; 
    data.bsdf_glossy = float3(0.0f);
    data.pdf = dot(state.normal, data.inDir) / M_PIf;
}

void materialSample(thread MaterialSample& data, thread MaterialState& state)
{
    // Source: https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_16.pdf
    {
        float a = 1.0f - 2.0f * data.xi.x;
        float b = sqrt(1.0f - a * a);
        float phi = data.xi.y * 2.0f * M_PIf;
        data.k2.x = state.normal.x + b * cos(phi);
        data.k2.y = state.normal.y + b * sin(phi);
        data.k2.z = state.normal.z + a;
        // data.pdf = a / M_PIf;
    }

    data.bsdf_over_pdf = state.diffuse;
}

bool traceOcclusion(
    instance_acceleration_structure accelerationStructure,
    thread intersector<triangle_data, instancing>& isect, 
    thread float3 origin, 
    thread float3 direction,
    float tMin,
    float tMax)
{
    struct ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = direction;
    shadowRay.min_distance = tMin;
    shadowRay.max_distance = tMax;
    isect.accept_any_intersection(true);

    bool res = true;
    typename intersector<triangle_data, instancing>::result_type intersection;
    intersection = isect.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW);
    if (intersection.type == intersection_type::none)
    {
        res = false;
    }
    isect.accept_any_intersection(false);
    return res;
}

float3 sampleLight(
    instance_acceleration_structure accelerationStructure,
    thread intersector<triangle_data, instancing>& isect, 
    thread uint32_t& rngState, 
    device const UniformLight& light, 
    thread MaterialState& state, 
    thread float3& toLight, 
    thread float& lightPdf)
{
    LightSampleData lightSampleData = {};
    switch (light.type)
    {
    case 0:
    {
        float2 u = float2(rnd(rngState), rnd(rngState));
        lightSampleData = SampleRectLight(light, u, state.position);
        break;
    }
        // case 1:
        //     lightSampleData = SampleDiscLight(light, float2(rand(rngState), rand(rngState)), state.position);
        //     break;
        // case 2:
        //     lightSampleData = SampleSphereLight(light, state.normal, state.position, float2(rand(rngState),
        //     rand(rngState))); break;
    }

    toLight = lightSampleData.L;
    float3 Li = float3(light.color);

    if (dot(state.normal, lightSampleData.L) > 0.0f && -dot(lightSampleData.L, lightSampleData.normal) > 0.0 && all(Li))
    {
        // Ray shadowRay;
        // shadowRay.d = float4(lightSampleData.L, 0.0f);
        // shadowRay.o = float4(offset_ray(state.position, state.geom_normal), lightSampleData.distToLight); // need to
        // set
        const bool occluded = traceOcclusion(accelerationStructure, isect, state.position, lightSampleData.L,
                                             0.001f, // tmin
                                             lightSampleData.distToLight - 1e-5f // tmax
        );

        // bool occluded = false;
        float visibility = occluded ? 0.0f : 1.0f;
        // TODO: skip light hit

        // if (visibility == 0.0f)
        // {
        //     // check if it was light hit?
        //     InstanceConstants instConst = accel.instanceConstants[NonUniformResourceIndex(shadowHit.instId)];
        //     if (instConst.lightId != -1)
        //     {
        //         // light hit => visible
        //         visibility = 1.0f;
        //     }
        // }
        lightPdf = lightSampleData.pdf;
        return visibility * Li * saturate(dot(state.normal, lightSampleData.L));
    }

    return float3(0.0f, 0.0f, 0.0f);
}

float3 estimateDirectLighting(
    instance_acceleration_structure accelerationStructure,
    thread intersector<triangle_data, instancing>& isect, 
    const uint32_t numLights,
    device UniformLight* lights,
    thread uint32_t& rngSeed, 
    thread MaterialState& state, 
    thread float3& toLight, 
    thread float& lightPdf)
{
    const uint32_t lightId = (uint32_t)(numLights * rnd(rngSeed));
    const float lightSelectionPdf = 1.0f / numLights;
    device const UniformLight& currLight = lights[lightId];
    const float3 r = sampleLight(accelerationStructure, isect, rngSeed, currLight, state, toLight, lightPdf);
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

static float3 offset_ray(thread const float3& p, thread const float3& n)
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
    uint2                                                  tid                       [[thread_position_in_grid]],
    constant Uniforms&                                     uniforms                  [[buffer(0)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor*   instances           [[buffer(1)]],
    instance_acceleration_structure                        accelerationStructure     [[buffer(2)]],
    device UniformLight* lights                                                      [[buffer(3)]],
    device Material* materials                                                       [[buffer(4)]],
    device float4* res                                                               [[buffer(5)]],
    device float4* accum                                                             [[buffer(6)]]
     )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height) 
    {
        return;
    }
    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    // uint32_t rndSeed = tea<4>(linearPixelIndex, uniforms.subframeIndex);
    uint32_t rndSeed = initRNG(tid.xy, uint2(uniforms.width, uniforms.height), uniforms.subframeIndex);

    PerRayData prd;
    prd.rndSeed = rndSeed;
    prd.radiance = float3(0.0f);
    prd.throughput = float3(1.0f);
    prd.inside = false;
    prd.depth = 0;
    prd.specularBounce = false;

    generateCameraRay(tid, prd.rndSeed, prd.origin, prd.direction, uniforms);

    while (prd.depth < uniforms.maxDepth)
    {
        ray ray;
        ray.min_distance = 0.0f;
        ray.max_distance = INFINITY;
        ray.origin = prd.origin;
        ray.direction = prd.direction;

        // Create an intersector to test for intersection between the ray and the geometry in the scene.
        intersector<triangle_data, instancing> i;
        i.assume_geometry_type(geometry_type::triangle);
        i.force_opacity(forced_opacity::opaque);

        typename intersector<triangle_data, instancing>::result_type intersection;

        i.accept_any_intersection(false);
        intersection = i.intersect(ray, accelerationStructure, RAY_MASK_PRIMARY);

        // Stop if the ray didn't hit anything and has bounced out of the scene.
        if (intersection.type == intersection_type::none)
        {
            prd.radiance += prd.throughput * uniforms.missColor;
            prd.throughput = float3(0.0f);
            break;
        }
        else
        {
            const uint32_t instanceIndex = intersection.instance_id;
            const uint32_t mask = instances[instanceIndex].mask;
            if (mask == GEOMETRY_MASK_LIGHT)
            {
                if (prd.depth == 0 || prd.specularBounce)
                {
                    // TODO: extract light colors
                    prd.radiance += prd.throughput * float3(1.0f);
                }
                prd.throughput = float3(0.0f);
                // stop tracing
                break;
            }

            const Triangle triangle = *(const device Triangle*)intersection.primitive_data;
            const float3 p0 = triangle.positions[0];
            const float3 p1 = triangle.positions[1];
            const float3 p2 = triangle.positions[2];

            const float3 n0 = unpackNormal(triangle.normals[0]);
            const float3 n1 = unpackNormal(triangle.normals[1]);
            const float3 n2 = unpackNormal(triangle.normals[2]);

            // The ray hit something. Look up the transformation matrix for this instance.
            float4x4 objectToWorldSpaceTransform(1.0f);

            for (int column = 0; column < 4; column++)
                for (int row = 0; row < 3; row++)
                    objectToWorldSpaceTransform[column][row] = instances[instanceIndex].transformationMatrix[column][row];

            const float2 barycentrics = intersection.triangle_barycentric_coord;
            // // Compute the intersection point in world space.
            // float3 worldPosition = ray.origin + ray.direction * intersection.distance;
            float3 worldPosition = transformPoint(interpolateAttrib(p0, p1, p2, barycentrics), objectToWorldSpaceTransform);

            // unsigned primitiveIndex = intersection.primitive_id;
            // unsigned int geometryIndex = instances[instanceIndex].accelerationStructureIndex;

            const float3 objectNormal = normalize(interpolateAttrib(n0, n1, n2, barycentrics));
            const float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldSpaceTransform));

            float3 geomNormal = normalize(cross(p1 - p0, p2 - p0));
            geomNormal = transformDirection(geomNormal, objectToWorldSpaceTransform); 

            MaterialState matState;

            matState.position = worldPosition;
            matState.normal = worldNormal;
            matState.geom_normal = geomNormal;

            const uint32_t materialId = instances[instanceIndex].userID;
            matState.diffuse = materials[materialId].diffuse;

            float3 toLight; // return value for estimateDirectLighting()
            float lightPdf = 0.0f; // return value for estimateDirectLighting()
            const float3 radiance = estimateDirectLighting(accelerationStructure, i,
                uniforms.numLights, lights, rndSeed, matState, toLight, lightPdf);

            const bool isNextEventValid = ((dot(toLight, matState.geom_normal) > 0.0f) != prd.inside) && lightPdf != 0.0f;
            if (isNextEventValid)
            {
                const float3 radianceOverPdf = radiance / lightPdf;
                
                MaterialEval evalData {};
                // evalData.ior1 = ior1;
                // evalData.ior2 = ior2;
                evalData.outDir = -prd.direction;
                evalData.inDir = toLight;

                materialEvaluate(evalData, matState);
                if (evalData.pdf > 1e-6f)
                {
                    const float misWeight = (lightPdf == 0.0f) ? 1.0f : lightPdf / (lightPdf + evalData.pdf);
                    const float3 w = prd.throughput * radianceOverPdf * misWeight;
                    prd.radiance += w * evalData.bsdf_diffuse;
                    prd.radiance += w * evalData.bsdf_glossy;
                }
            }

            const float z1 = rnd(prd.rndSeed);
            const float z2 = rnd(prd.rndSeed);
            const float z3 = rnd(prd.rndSeed);
            const float z4 = rnd(prd.rndSeed);

            MaterialSample sampleData {};
            // sampleData.ior1 = ior1;
            // sampleData.ior2 = ior2;
            sampleData.k1 = -prd.direction;
            sampleData.xi = float4(z1, z2, z3, z4);

            materialSample(sampleData, matState);

            prd.origin = offset_ray(worldPosition, matState.geom_normal * (prd.inside ? -1.0 : 1.0));
            prd.direction = sampleData.k2;
            prd.throughput *= sampleData.bsdf_over_pdf;

            if (dot(prd.throughput, prd.throughput) < 1e-4f)
            {
                break;
            }

            if (prd.depth > 3)
            {
                const float p = max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z));
                if (rnd(prd.rndSeed) > p)
                {
                    break;
                }
                prd.throughput *= 1.0 / (p + 1e-5f);
            }
        }
        ++prd.depth;
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

    switch ((ToneMapperType) uniforms.tonemapperType)
    {
        case ToneMapperType::eReinhard:
        {
            result = reinhard(result);
            break;
        }
        case ToneMapperType::eACES:
        {
            result = ACESFitted(result);
            break;
        }
        case ToneMapperType::eFilmic: 
        {
            result = ACESFilm(result);
            break;
        }
        case ToneMapperType::eNone:
        {
            break;
        }
    }

    if (uniforms.gamma > 0.0f)
    {
        result = pow(result, float3(1.0f / uniforms.gamma));
    }

    res[linearPixelIndex] = float4(result, 1.0f);
}
