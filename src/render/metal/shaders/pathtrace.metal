#include <metal_stdlib>
#include <simd/simd.h>

#include "random.h"
#include "lights.h"
#include "ShaderTypes.h"

using namespace metal;

using namespace raytracing;

constant unsigned int primes[] = {
    2,   3,  5,  7,
    11, 13, 17, 19,
    23, 29, 31, 37,
    41, 43, 47, 53,
    59, 61, 67, 71,
    73, 79, 83, 89
};


// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a low discrepency sequence: the values appear
// random, but are more evenly distributed than a purely random sequence. Each random
// value used to render the image uses a different independent dimension, `d`,
// and each sample (frame) uses a different index `i`. To decorrelate each pixel,
// you can apply a random offset to `i`.
float halton(unsigned int i, unsigned int d) {
    unsigned int b = primes[d];

    float f = 1.0f;
    float invB = 1.0f / b;

    float r = 0;

    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }

    return r;
}

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

// Resources for a piece of triangle geometry.
struct TriangleResources {
    device uint16_t *indices;
    device float3 *vertexNormals;
    device float3 *vertexColors;
};

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
    // const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
    const float2 subpixel_jitter {0.0f, 0.0f};
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
    float3 normal;
};

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
        lightSampleData = SampleRectLight(light, float2(rnd(rngState), rnd(rngState)), state.position);
        break;
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
                                             0.01f, // tmin
                                             lightSampleData.distToLight // tmax
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

    return float3(0.0f);
    // if (!all(Li))
    // {
    //     return float3(1.0);
    // }
    // return Li;
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
    // const uint32_t lightId = (uint32_t)(numLights * rnd(rngSeed));
    const uint32_t lightId = 0;
    const float lightSelectionPdf = 1.0f / numLights;
    device const UniformLight& currLight = lights[lightId];
    const float3 r = sampleLight(accelerationStructure, isect, rngSeed, currLight, state, toLight, lightPdf);
    lightPdf *= lightSelectionPdf;
    return r;
}

// Main ray tracing kernel.
kernel void raytracingKernel(
     uint2                                                  tid                       [[thread_position_in_grid]],
     constant Uniforms&                                     uniforms                  [[buffer(0)]],
     constant MTLAccelerationStructureInstanceDescriptor*   instances                 [[buffer(1)]],
     instance_acceleration_structure                        accelerationStructure     [[buffer(2)]],
     device UniformLight* lights                                                      [[buffer(3)]],
     device float4* res                                                               [[buffer(4)]]
     )
{
    // The sample aligns the thread count to the threadgroup size, which means the thread count
    // may be different than the bounds of the texture. Test to make sure this thread
    // is referencing a pixel within the bounds of the texture.
    if (tid.x < uniforms.width && tid.y < uniforms.height) {

        uint32_t rndSeed = tea<4>(tid.y * uniforms.width + tid.x, uniforms.subframeIndex);

        // The ray to cast.
        ray ray;

        // Pixel coordinates for this thread.
        // float2 pixel = (float2)tid;

        // // Apply a random offset to the random number index to decorrelate pixels.
        // unsigned int offset = randomTex.read(tid).x;

        // // Add a random offset to the pixel coordinates for antialiasing.
        // float2 r = float2(halton(offset + uniforms.frameIndex, 0),
        //                   halton(offset + uniforms.frameIndex, 1));

        generateCameraRay(tid, 0, ray.origin, ray.direction, uniforms);
        // Don't limit intersection distance.
        ray.max_distance = INFINITY;

        // Start with a fully white color. The kernel scales the light each time the
        // ray bounces off of a surface, based on how much of each light component
        // the surface absorbs.
        float3 color = float3(1.0f, 1.0f, 1.0f);

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
            color = float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            Triangle triangle = *(const device Triangle*)intersection.primitive_data;

            const float3 n0 = unpackNormal(triangle.normals[0]);
            const float3 n1 = unpackNormal(triangle.normals[1]);
            const float3 n2 = unpackNormal(triangle.normals[2]);

            unsigned int instanceIndex = intersection.instance_id;

            // // Look up the mask for this instance, which indicates what type of geometry the ray hit.
            // unsigned int mask = instances[instanceIndex].mask;

            // The ray hit something. Look up the transformation matrix for this instance.
            float4x4 objectToWorldSpaceTransform(1.0f);

            for (int column = 0; column < 4; column++)
                for (int row = 0; row < 3; row++)
                    objectToWorldSpaceTransform[column][row] = instances[instanceIndex].transformationMatrix[column][row];

            // // Compute the intersection point in world space.
            float3 worldSpaceIntersectionPoint = ray.origin + ray.direction * intersection.distance;

            // unsigned primitiveIndex = intersection.primitive_id;
            // unsigned int geometryIndex = instances[instanceIndex].accelerationStructureIndex;
            const float2 barycentrics = intersection.triangle_barycentric_coord;

            // float3 objectSpaceSurfaceNormal = interpolateVertexAttribute(triangle.normals, barycentric_coords);
            const float3 objectNormal = normalize(interpolateAttrib(n0, n1, n2, barycentrics));
            const float3 worldNormal = normalize(transformDirection(objectNormal, objectToWorldSpaceTransform));

            MaterialState matState;

            matState.position = worldSpaceIntersectionPoint;
            matState.normal = worldNormal;

            float3 toLight; // return value for sampleLights()
            float lightPdf = 0.0f; // return value for sampleLights()
            const float3 radiance = estimateDirectLighting(accelerationStructure, i,
                uniforms.numLights, lights, rndSeed, matState, toLight, lightPdf);


            // device const UniformLight& currLight = lights[0];
            // color = lights[0].color.xyz;
            color = radiance;
            // color = float3(1.0f - barycentric_coords.x - barycentric_coords.y, barycentric_coords.x, barycentric_coords.y);
        }
        res[tid.y * uniforms.width + tid.x] = float4(color, 1.0f);
        // res[tid.y * uniforms.width + tid.x] = float4(1.0f);
        // dstTex.write(float4(color, 1.0f), tid);
    }
}
