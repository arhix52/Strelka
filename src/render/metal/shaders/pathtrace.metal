#include <metal_stdlib>
#include <simd/simd.h>

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

// Main ray tracing kernel.
kernel void raytracingKernel(
     uint2                                                  tid                       [[thread_position_in_grid]],
     constant Uniforms&                                     uniforms                  [[buffer(0)]],
     constant MTLAccelerationStructureInstanceDescriptor*   instances                 [[buffer(1)]],
     instance_acceleration_structure                        accelerationStructure     [[buffer(2)]],
     device Vertex* vb                                                                [[buffer(3)]],
     device uint32_t* ib                                                              [[buffer(4)]],
     device float4* res                                                               [[buffer(5)]]
     )
{
    // The sample aligns the thread count to the threadgroup size, which means the thread count
    // may be different than the bounds of the texture. Test to make sure this thread
    // is referencing a pixel within the bounds of the texture.
    if (tid.x < uniforms.width && tid.y < uniforms.height) {
        // The ray to cast.
        ray ray;

        // Pixel coordinates for this thread.
        // float2 pixel = (float2)tid;

        // // Apply a random offset to the random number index to decorrelate pixels.
        // unsigned int offset = randomTex.read(tid).x;

        // // Add a random offset to the pixel coordinates for antialiasing.
        // float2 r = float2(halton(offset + uniforms.frameIndex, 0),
        //                   halton(offset + uniforms.frameIndex, 1));

        // pixel += r;

        // Map pixel coordinates to -1..1.
        // float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        // uv = uv * 2.0f - 1.0f;

        // constant Camera & camera = uniforms.camera;

        // // Rays start at the camera position.
        // ray.origin = camera.position;

        // // Map normalized pixel coordinates into camera's coordinate system.
        // ray.direction = normalize(uv.x * camera.right +
        //                           uv.y * camera.up +
        //                           camera.forward);

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
            // unsigned int instanceIndex = intersection.instance_id;

            // // Look up the mask for this instance, which indicates what type of geometry the ray hit.
            // unsigned int mask = instances[instanceIndex].mask;

            // // The ray hit something. Look up the transformation matrix for this instance.
            // float4x4 objectToWorldSpaceTransform(1.0f);

            // for (int column = 0; column < 4; column++)
            //     for (int row = 0; row < 3; row++)
            //         objectToWorldSpaceTransform[column][row] = instances[instanceIndex].transformationMatrix[column][row];

            // // Compute the intersection point in world space.
            // float3 worldSpaceIntersectionPoint = ray.origin + ray.direction * intersection.distance;

            // unsigned primitiveIndex = intersection.primitive_id;
            // unsigned int geometryIndex = instances[instanceIndex].accelerationStructureIndex;
            float2 barycentric_coords = intersection.triangle_barycentric_coord;
            color = float3(1.0f - barycentric_coords.x - barycentric_coords.y, barycentric_coords.x, barycentric_coords.y);
        }
        res[tid.y * uniforms.width + tid.x] = float4(color, 1.0f);
        // res[tid.y * uniforms.width + tid.x] = float4(1.0f);
        // dstTex.write(float4(color, 1.0f), tid);
    }
}
