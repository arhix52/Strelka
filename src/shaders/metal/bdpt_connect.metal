// ============================================================================
// bdpt_connect.metal -- BDPT Connection Kernel
//
// Connects camera and light subpath vertices and computes contributions
// with MIS weighting. Handles all (s,t) connection strategies:
//   s=0,t>=2: Camera path hits light directly
//   s=1,t>=1: NEE (sample light, connect to camera vertex)
//   s>=1,t=1: Light tracing (connect light vertex to camera lens -> splat)
//   s>=2,t>=2: General connection via shadow ray
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "env_light_metal.h"
#include "bdpt_common.h"

#include "ShaderTypes.h"

using namespace metal;
using namespace raytracing;

__attribute__((always_inline))
static int __float_as_int_co(float x) { return as_type<int>(x); }
__attribute__((always_inline))
static float __int_as_float_co(int x) { return as_type<float>(x); }

static float3 offset_ray_co(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float_co(__float_as_int_co(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        __int_as_float_co(__float_as_int_co(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        __int_as_float_co(__float_as_int_co(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

static bool traceOcclusion_co(
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    const float3 origin,
    const float3 direction,
    const float tMin,
    const float tMax)
{
    struct ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = direction;
    shadowRay.min_distance = tMin;
    shadowRay.max_distance = tMax;
    isect.accept_any_intersection(true);

    auto intersection = isect.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, 1.0f);
    bool occluded = (intersection.type != intersection_type::none);
    isect.accept_any_intersection(false);
    return occluded;
}

kernel void bdpt_connect(
    uint2                                                      tid                   [[thread_position_in_grid]],
    constant Uniforms&                                         uniforms              [[buffer(0)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor*  instances             [[buffer(1)]],
    acceleration_structure<instancing, primitive_motion>        accelerationStructure [[buffer(2)]],
    device UniformLight*                                       lights                [[buffer(3)]],
    device Material*                                           materials             [[buffer(4)]],
    device float4*                                             outputBuffer          [[buffer(5)]],
    device float4*                                             accumBuffer           [[buffer(6)]],
    device const BDPTVertex*                                   cameraVertices       [[buffer(7)]],
    device const uint32_t*                                     cameraPathLengths    [[buffer(8)]],
    device const BDPTVertex*                                   lightVertices        [[buffer(9)]],
    device const uint32_t*                                     lightPathLengths     [[buffer(10)]],
    device atomic_uint*                                        splatBuffer          [[buffer(11)]],
    device const float*                                        envCdfX              [[buffer(12)]],
    device const float*                                        envCdfY              [[buffer(13)]],
    texture2d<float>                                           envMapTexture        [[texture(0)]]
    )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    const uint32_t cameraLen = cameraPathLengths[linearPixelIndex];
    const uint32_t lightLen  = lightPathLengths[linearPixelIndex];

    float3 result = float3(0.0f);

    // Create intersector for shadow rays
    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    // ===================================================================
    // Strategy (s=0, t>=2): Camera path hits light directly
    // This is the "unidirectional PT" contribution.
    // Already handled by camera subpath recording is_on_light vertices.
    // ===================================================================
    for (uint32_t t = 1; t < cameraLen; ++t)
    {
        device const BDPTVertex& cv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t];
        if (cv.is_on_light)
        {
            if (cv.light_index == 0xFFFFFFFF)
            {
                // Environment map hit -- throughput already includes Le
                float misWeight = 1.0f;
                if (t > 1)
                {
                    device const BDPTVertex& prevCv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t - 1];
                    if (!prevCv.is_delta)
                    {
                        // Compute env map PDF for MIS against NEE
                        float3 hitDir = -float3(cv.wo); // direction toward env
                        float envPdf = envMapPdf(hitDir, envCdfX, envCdfY,
                                                  uniforms.envMapWidth, uniforms.envMapHeight,
                                                  uniforms.envMapRotation);
                        float envSelectionPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
                        envPdf *= envSelectionPdf;
                        float bsdfPdf = prevCv.pdf_fwd;
                        if (bsdfPdf > 0.0f && envPdf > 0.0f)
                            misWeight = misWeightBalance(bsdfPdf, envPdf);
                    }
                }
                result += float3(cv.throughput) * misWeight;
            }
            else
            {
                // Area light hit
                device const UniformLight& light = lights[cv.light_index];
                float3 Le = float3(light.color);

                // MIS weight: balance between direct hit (s=0) and NEE (s=1)
                float misWeight = 1.0f;
                if (t > 1)
                {
                    device const BDPTVertex& prevCv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t - 1];
                    if (prevCv.is_delta)
                    {
                        misWeight = 1.0f;
                    }
                    else
                    {
                        float lightPdf = getLightPdf(light, float3(cv.position), float3(prevCv.position));
                        float lightSelectionPdf = uniforms.hasEnvMap
                            ? 0.5f / float(uniforms.numLights)
                            : 1.0f / float(uniforms.numLights);
                        lightPdf *= lightSelectionPdf;
                        float bsdfPdf = prevCv.pdf_fwd;
                        if (bsdfPdf > 0.0f && lightPdf > 0.0f)
                            misWeight = misWeightBalance(bsdfPdf, lightPdf);
                    }
                }
                result += float3(cv.throughput) * Le * misWeight;
            }
        }
    }

    // ===================================================================
    // Strategy (s>=1, t=1): Light tracing -- connect light vertex to camera
    //
    // TODO: Implement light tracing with pixel splatting.
    // This requires a two-pass approach:
    //   Pass 1 (this kernel): For each non-delta light subpath vertex,
    //     project to image plane, evaluate BSDF + camera response (We),
    //     compute MIS weight, and atomically add to splatBuffer.
    //   Pass 2 (separate resolve kernel): Read splatBuffer per pixel and
    //     merge into the accumulation buffer.
    // The global barrier between writes (from all threadgroups) and reads
    // cannot be done within a single compute dispatch.
    // This strategy mainly improves caustics (SDS paths); wall noise is
    // addressed by the NEE (s=1, t>=1) strategy below.
    // ===================================================================

    // ===================================================================
    // Strategy (s>=2, t>=2): General connection
    // Connect each non-delta camera vertex with each non-delta light vertex
    // via a shadow ray.
    // ===================================================================
    for (uint32_t t = 1; t < cameraLen; ++t)
    {
        device const BDPTVertex& cv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t];

        // Skip delta camera vertices (can't connect) and light-hit vertices
        if (cv.is_delta || cv.is_on_light || cv.is_on_camera)
            continue;

        float3 camPos = float3(cv.position);
        float3 camNormal = float3(cv.shading_normal);
        float3 camGeomNormal = float3(cv.geometry_normal);

        for (uint32_t s = 1; s < lightLen; ++s)
        {
            device const BDPTVertex& lv = lightVertices[linearPixelIndex * BDPT_MAX_DEPTH + s];

            // Skip delta light vertices
            if (lv.is_delta)
                continue;

            float3 lightPos = float3(lv.position);
            float3 lightNormal = float3(lv.shading_normal);
            // Direction from camera vertex to light vertex
            float3 connDir = lightPos - camPos;
            float dist2 = dot(connDir, connDir);
            if (dist2 < 1e-8f) continue;
            float dist = sqrt(dist2);
            connDir /= dist;

            // Check geometry: both vertices should face toward each other
            float cosAtCamera = dot(camNormal, connDir);
            float cosAtLight  = dot(lightNormal, -connDir);
            if (cosAtCamera <= 0.0f || cosAtLight <= 0.0f)
                continue;

            // Shadow ray
            float3 shadowOrigin = offset_ray_co(camPos, camGeomNormal);
            bool occluded = traceOcclusion_co(accelerationStructure, isect,
                                               shadowOrigin, connDir, 0.001f, dist - 0.001f);
            if (occluded)
                continue;

            // Evaluate BSDF at camera vertex
            SurfaceInteraction si_cam = vertexToSI(cv, materials);
            BsdfEvalResult evalCam = bsdf_eval(si_cam, connDir);
            if (evalCam.pdf <= 0.0f)
                continue;

            // Evaluate BSDF at light vertex (reversed direction)
            SurfaceInteraction si_light = vertexToSI(lv, materials);
            if (lv.is_on_light)
            {
                // Light surface vertex (s=1): emission angular profile + geometry term
                float G = cosAtCamera * cosAtLight / dist2;

                // Lambertian emission angular profile for connection direction
                float emissionProfile = cosAtLight * M_1_PI_F;

                // Reverse PDF at camera vertex for SmallVCM MIS
                float cameraRevPdfW = bsdf_pdf_reverse(si_cam, connDir);

                // Convert forward PDFs to area measure
                float cameraDirPdfA = evalCam.pdf * cosAtLight / dist2;
                // Light emission directional PDF for direction toward camera
                float lightDirPdfW = cosAtLight * M_1_PI_F; // Lambertian
                float lightDirPdfA = lightDirPdfW * cosAtCamera / dist2;

                // SmallVCM MIS weight (lv.dVC = 0 for vertex 0)
                // VCM: include dVM merge term when integratorType == 2
                float wLight  = cameraDirPdfA * lv.dVCM;
                float wCamera = lightDirPdfA * (cv.dVCM + cv.dVC * cameraRevPdfW + cv.dVM * cameraRevPdfW);
                float misWeight = 1.0f / (wLight + 1.0f + wCamera + 1e-10f);

                // lv.throughput = Le / pdf_pos; emissionProfile provides the angular distribution
                float3 contrib = float3(cv.throughput) * evalCam.bsdf * G * emissionProfile * float3(lv.throughput) * misWeight;
                result += contrib;
            }
            else
            {
                // General surface vertex (s>=2): evaluate BSDF at both endpoints
                BsdfEvalResult evalLight = bsdf_eval(si_light, -connDir);
                if (evalLight.pdf <= 0.0f)
                    continue;

                float G = cosAtCamera * cosAtLight / dist2;

                // Reverse PDFs for SmallVCM MIS
                float cameraRevPdfW = bsdf_pdf_reverse(si_cam, connDir);
                float lightRevPdfW  = bsdf_pdf_reverse(si_light, -connDir);

                // Convert forward PDFs to area measure
                float cameraDirPdfA = evalCam.pdf * cosAtLight / dist2;
                float lightDirPdfA  = evalLight.pdf * cosAtCamera / dist2;

                // SmallVCM MIS weight (VCM: include dVM merge terms)
                float wLight  = cameraDirPdfA * (lv.dVCM + lv.dVC * lightRevPdfW + lv.dVM * lightRevPdfW);
                float wCamera = lightDirPdfA  * (cv.dVCM + cv.dVC * cameraRevPdfW + cv.dVM * cameraRevPdfW);
                float misWeight = 1.0f / (wLight + 1.0f + wCamera + 1e-10f);

                float3 contrib = float3(cv.throughput) * evalCam.bsdf * G * evalLight.bsdf * float3(lv.throughput) * misWeight;
                result += contrib;
            }
        }
    }

    // ===================================================================
    // Strategy (s=1, t>=1): NEE -- sample a FRESH light point per camera vertex
    //
    // For each non-delta camera vertex, importance-sample a point on a light
    // source and connect. This is the classic next-event estimation and is
    // critical for low-variance direct illumination. Unlike the (s>=2,t>=2)
    // connections which reuse a single random light subpath vertex, NEE
    // picks a light point specifically optimized for each camera vertex.
    // ===================================================================
    {
        // Initialize a sampler for NEE random numbers
        SamplerState neeSampler = initSampler(linearPixelIndex, uniforms.subframeIndex, 2u);

        for (uint32_t t = 1; t < cameraLen; ++t)
        {
            device const BDPTVertex& cv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t];

            // Skip delta, light-hit, or camera-origin vertices
            if (cv.is_delta || cv.is_on_light || cv.is_on_camera)
                continue;

            // Reconstruct SurfaceInteraction at camera vertex for BSDF eval
            SurfaceInteraction si_cam = vertexToSI(cv, materials);

            float3 toLight = float3(0.0f);
            float lightPdf = 0.0f;
            float3 radiance = float3(0.0f);
            float lightSelectionPdf = 1.0f;

            // Advance sampler depth to get unique random numbers per camera vertex
            neeSampler.depth = t;

            float uLightId = random<SampleDimension::eLightId>(neeSampler, uniforms.samplerType);
            float2 uLightPt = float2(
                random<SampleDimension::eLightPointX>(neeSampler, uniforms.samplerType),
                random<SampleDimension::eLightPointY>(neeSampler, uniforms.samplerType));

            if (uniforms.hasEnvMap && (uniforms.numLights == 0 || uLightId >= 0.5f))
            {
                // Sample environment map
                float envSelPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
                float envPdf = 0.0f;
                float3 dir = sampleEnvMap(uLightPt, envCdfX, envCdfY,
                                          uniforms.envMapWidth, uniforms.envMapHeight,
                                          uniforms.envMapRotation, envPdf);
                if (envPdf > 0.0f && dot(si_cam.shading_normal, dir) > 0.0f)
                {
                    float3 shadowOrigin = offset_ray_co(si_cam.position, si_cam.geometry_normal);
                    bool occluded = traceOcclusion_co(accelerationStructure, isect,
                                                      shadowOrigin, dir, 0.001f, 1e16f);
                    if (!occluded)
                    {
                        constexpr sampler envSampler(mag_filter::linear, min_filter::linear,
                                                     address::repeat, coord::normalized);
                        float2 envUV = dirToEnvUV(dir, uniforms.envMapRotation);
                        float4 envSample = envMapTexture.sample(envSampler, envUV);
                        float3 Li = envSample.xyz * uniforms.envMapIntensity
                                  * float3(uniforms.envMapColorTint);
                        toLight = dir;
                        lightPdf = envPdf * envSelPdf;
                        radiance = Li * max(dot(si_cam.shading_normal, dir), 0.0f);
                    }
                }
            }
            else if (uniforms.numLights > 0)
            {
                // Sample local light
                float remappedU = uniforms.hasEnvMap ? (uLightId * 2.0f) : uLightId;
                uint32_t lightId = min((uint32_t)(uniforms.numLights * remappedU),
                                       uniforms.numLights - 1);
                lightSelectionPdf = uniforms.hasEnvMap
                    ? 0.5f / float(uniforms.numLights)
                    : 1.0f / float(uniforms.numLights);

                device const UniformLight& light = lights[lightId];
                LightSampleData lsd = {};

                switch (light.type)
                {
                case 0: // Rect
                    if (uniforms.rectLightSamplingMethod == 0)
                        lsd = SampleRectLightUniform(light, uLightPt, si_cam.position);
                    else
                        lsd = SampleRectLight(light, uLightPt, si_cam.position);
                    break;
                case 2: // Sphere
                    lsd = SampleSphereLight(light, uLightPt, si_cam.position);
                    break;
                case 3: // Distant
                    lsd = SampleDistantLight(light, uLightPt, si_cam.position);
                    break;
                }

                float3 Li = float3(light.color);
                toLight = lsd.L;
                float cosAtSurface = dot(si_cam.shading_normal, lsd.L);
                float cosAtLightN = -dot(lsd.L, lsd.normal);

                if (cosAtSurface > 0.0f && cosAtLightN > 0.001f && any(Li > float3(0.0f)))
                {
                    float3 shadowOrigin = offset_ray_co(si_cam.position, si_cam.geometry_normal);
                    bool occluded = traceOcclusion_co(accelerationStructure, isect,
                                                      shadowOrigin, lsd.L,
                                                      0.001f,
                                                      lsd.distToLight - 1e-5f);
                    if (!occluded)
                    {
                        lightPdf = lsd.pdf * lightSelectionPdf;
                        radiance = Li * cosAtSurface;
                    }
                }
            }

            // Evaluate BSDF at camera vertex toward the light sample
            if (lightPdf > 0.0f)
            {
                BsdfEvalResult evalCam = bsdf_eval(si_cam, toLight);
                if (evalCam.pdf > 0.0f)
                {
                    // MIS weight: balance between NEE (s=1) and BSDF hit (s=0)
                    float misWeight = bdptMISWeightNEE(lightPdf, evalCam.pdf);

                    float3 contrib = float3(cv.throughput) * evalCam.bsdf
                                   * radiance / lightPdf * misWeight;
                    result += contrib;
                }
            }
        }
    }

    // ===================================================================
    // Output: combine with accumulation
    // For VCM (integratorType == 2), the merge kernel handles accumulation
    // after adding its contribution. Write raw result for merge to read.
    // ===================================================================
    if (uniforms.integratorType == 2)
    {
        // Raw result -- merge kernel will add merge contribution and accumulate
        outputBuffer[linearPixelIndex] = float4(result, 1.0f);
    }
    else if (uniforms.enableAccumulation)
    {
        float3 accum_color = result / float(uniforms.samples_per_launch);

        if (uniforms.subframeIndex > 0)
        {
            float a = 1.0f / float(uniforms.subframeIndex + 1);
            float3 prev = float3(accumBuffer[linearPixelIndex]);
            accum_color = mix(prev, accum_color, a);
        }
        accumBuffer[linearPixelIndex] = float4(accum_color, 1.0f);
        result = accum_color;
        outputBuffer[linearPixelIndex] = float4(result, 1.0f);
    }
    else
    {
        outputBuffer[linearPixelIndex] = float4(result, 1.0f);
    }
}
