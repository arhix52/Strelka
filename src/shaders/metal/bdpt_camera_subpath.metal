// ============================================================================
// bdpt_camera_subpath.metal -- BDPT Camera Subpath Tracing Kernel
//
// Traces camera subpaths and records BDPTVertex at each bounce.
// Computes and stores partial MIS weights (dVCM, dVC) per vertex.
// Does NOT perform NEE -- the connection kernel handles that.
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "env_light_metal.h"
#include "bdpt_common.h"

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h>
#include <strelka/material/bsdf.h>

using namespace metal;
using namespace raytracing;

// Shared utilities from pathtrace.metal (duplicated here to avoid multi-metallib linking)
__attribute__((always_inline))
static float3 transformPoint_cs(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 1.0f)).xyz;
}

__attribute__((always_inline))
static float3 transformDirection_cs(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 0.0f)).xyz;
}

static float3 unpackNormal_cs(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

static float2 unpackUV_cs(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

static float3 interpolateAttrib3_cs(const float3 a1, const float3 a2, const float3 a3, const float2 bary)
{
    return a1 * (1.0f - bary.x - bary.y) + a2 * bary.x + a3 * bary.y;
}

static float2 interpolateAttrib2_cs(const float2 a1, const float2 a2, const float2 a3, const float2 bary)
{
    return a1 * (1.0f - bary.x - bary.y) + a2 * bary.x + a3 * bary.y;
}

__attribute__((always_inline))
static int __float_as_int_cs(float x) { return as_type<int>(x); }
__attribute__((always_inline))
static float __int_as_float_cs(int x) { return as_type<float>(x); }

static float3 offset_ray_cs(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float_cs(__float_as_int_cs(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        __int_as_float_cs(__float_as_int_cs(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        __int_as_float_cs(__float_as_int_cs(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Fill SurfaceInteraction from hit geometry and sample Material textures
static void initSurfaceInteraction_cs(
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

    float3 baseColor = float3(material.base_color);
    if (!is_null_texture(material.baseColorTexture))
    {
        float4 texVal = material.baseColorTexture.sample(texSampler, uv);
        baseColor *= texVal.rgb;
    }
    si.albedo = baseColor;

    float resolvedRoughness = material.roughness;
    float resolvedMetallic = material.metallic;
    if (!is_null_texture(material.metallicRoughnessTexture))
    {
        float4 mrTex = material.metallicRoughnessTexture.sample(texSampler, uv);
        resolvedRoughness *= mrTex.g;
        resolvedMetallic *= mrTex.b;
    }

    if (!is_null_texture(material.normalTexture))
    {
        float3 bumpNormal = material.normalTexture.sample(texSampler, uv).xyz * 2.0f - 1.0f;
        bumpNormal.xy *= material.normal_scale;
        float3x3 TBN = float3x3(worldTangent, worldBinormal, worldNormal);
        si.shading_normal = normalize(TBN * bumpNormal);
    }

    float3 emissionColor = float3(material.emission);
    if (!is_null_texture(material.emissionTexture))
    {
        float4 emTex = material.emissionTexture.sample(texSampler, uv);
        emissionColor *= emTex.rgb;
    }
    si.emission = emissionColor * material.emission_strength;

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

    bsdf_init(si, matParams);
    si.roughness = max(resolvedRoughness, 0.0001f);
    si.metallic = saturate(resolvedMetallic);
}

kernel void bdpt_camera_subpath(
    uint2                                                      tid                   [[thread_position_in_grid]],
    constant Uniforms&                                         uniforms              [[buffer(0)]],
    constant MTLAccelerationStructureUserIDInstanceDescriptor*  instances             [[buffer(1)]],
    acceleration_structure<instancing, primitive_motion>        accelerationStructure [[buffer(2)]],
    device UniformLight*                                       lights                [[buffer(3)]],
    device Material*                                           materials             [[buffer(4)]],
    device const char*                                         prevVertexBuffer      [[buffer(5)]],
    device const uint32_t*                                     indexBuffer           [[buffer(6)]],
    device const InstanceData*                                 instanceDataBuffer    [[buffer(7)]],
    device const float*                                        envCdfX              [[buffer(8)]],
    device const float*                                        envCdfY              [[buffer(9)]],
    device BDPTVertex*                                         cameraVertices       [[buffer(10)]],
    device uint32_t*                                           cameraPathLengths    [[buffer(11)]],
    texture2d<float>                                           envMapTexture        [[texture(0)]]
    )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    const uint32_t maxDepth = min(uniforms.maxCameraSubpathDepth, uniforms.bdptStride - 1u);
    // Initialize sampler
    SamplerState sampler = initSampler(linearPixelIndex, uniforms.subframeIndex, 0u);

    // Generate camera ray
    const float2 subpixel_jitter = {
        random<SampleDimension::ePixelX>(sampler, uniforms.samplerType),
        random<SampleDimension::ePixelY>(sampler, uniforms.samplerType)};
    float2 pixelPos {tid.x + subpixel_jitter.x, uniforms.height - (tid.y + subpixel_jitter.y)};
    float2 dimension {(float)uniforms.width, (float)uniforms.height};
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;
    pixelNDC.x += uniforms.shiftX * 2.0f;
    pixelNDC.y += uniforms.shiftY * 2.0f;

    float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
    float4 viewSpace = uniforms.clipToView * clip;

    float4 wdir = uniforms.viewToWorld * float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);
    float3 origin = (uniforms.viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
    float3 direction = normalize(wdir.xyz);

    // Camera vertex (vertex 0) -- on the lens
    // The image plane PDF = 1 / (W * H) in pixel area
    // Convert to solid angle: imagePdf * cosAtCamera^3 * dist^2 / A_pixel
    // For simplicity, use the standard SmallVCM formulation:
    float cosAtCamera = fabs(dot(direction, float3(-uniforms.viewToWorld[0][2],
                                                    -uniforms.viewToWorld[1][2],
                                                    -uniforms.viewToWorld[2][2])));
    cosAtCamera = max(cosAtCamera, 1e-6f);

    // SmallVCM: dVCM = numPixels / cameraPdfW for light tracing (s>=1,t=1).
    // Since we do NOT implement light tracing, set dVCM = 0 to remove
    // the non-existent strategy from MIS weights. This causes the merge
    // at camera vertex 1 to get slightly more weight than optimal (no
    // camera-side competition), but setting dVCM non-zero without light
    // tracing to compensate makes the result too dark. The small merge
    // bias (~5-10%) decreases with progressive radius shrinkage.
    float3 throughput = float3(1.0f);
    float dVCM = 0.0f;
    float dVC  = 0.0f;
    float dVM  = 0.0f;

    // Store camera vertex 0 (lens point)
    {
        device BDPTVertex& v = cameraVertices[linearPixelIndex * uniforms.bdptStride + 0];
        v.position        = packed_float3(origin);
        v.geometry_normal  = packed_float3(float3(-uniforms.viewToWorld[0][2],
                                                   -uniforms.viewToWorld[1][2],
                                                   -uniforms.viewToWorld[2][2]));
        v.shading_normal   = v.geometry_normal;
        v.throughput       = packed_float3(throughput);
        v.wo               = packed_float3(direction);
        v.pdf_fwd          = 1.0f; // camera pixel PDF (area)
        v.pdf_rev          = 0.0f;
        v.dVCM             = dVCM;
        v.dVC              = dVC;
        v.dVM              = dVM;
        v.uv_x             = (float)tid.x;
        v.uv_y             = (float)tid.y;
        v.material_index   = 0;
        v.event_type       = 0;
        v.is_delta         = 0;
        v.is_on_light      = 0;
        v.is_on_camera     = 1;
        v.light_index      = 0;
        v.exterior_ior     = 1.0f; // camera is in air
    }

    // IOR stack for nested dielectrics
    IorStack iorStack;
    ior_stack_init(iorStack);

    // Create intersector
    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    uint32_t pathLength = 1; // vertex 0 is camera

    for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)
    {
        ray r;
        r.min_distance = 0.0f;
        r.max_distance = INFINITY;
        r.origin = origin;
        r.direction = direction;

        isect.accept_any_intersection(false);
        auto intersection = isect.intersect(r, accelerationStructure, RAY_MASK_PRIMARY, 1.0f);

        if (intersection.type == intersection_type::none)
        {
            // Miss -- record env map hit as a special light vertex
            if (uniforms.hasEnvMap && pathLength < uniforms.bdptStride)
            {
                // Evaluate env map radiance
                constexpr struct sampler envSampler(mag_filter::linear, min_filter::linear,
                                                    address::repeat, coord::normalized);
                float2 envUV = dirToEnvUV(direction, uniforms.envMapRotation);
                float4 envSample = envMapTexture.sample(envSampler, envUV);
                float3 envLe = envSample.xyz * uniforms.envMapIntensity
                             * float3(uniforms.envMapColorTint);

                device BDPTVertex& v = cameraVertices[linearPixelIndex * uniforms.bdptStride + pathLength];
                v.position        = packed_float3(float3(0.0f)); // no position for env
                v.geometry_normal  = packed_float3(float3(0.0f));
                v.shading_normal   = packed_float3(float3(0.0f));
                v.throughput       = packed_float3(throughput * envLe); // pre-multiply Le into throughput
                v.wo               = packed_float3(-direction);
                v.pdf_fwd          = 0.0f;
                v.pdf_rev          = 0.0f;
                v.dVCM             = dVCM;
                v.dVC              = dVC;
                v.dVM              = dVM;
                v.uv_x             = envUV.x;
                v.uv_y             = envUV.y;
                v.material_index   = 0;
                v.event_type       = BSDF_EVENT_DIFFUSE_REFLECTION;
                v.is_delta         = 0;
                v.is_on_light      = 1;
                v.is_on_camera     = 0;
                v.light_index      = 0xFFFFFFFF; // sentinel for env map
                v.exterior_ior     = 1.0f;
                pathLength++;
            }
            break;
        }

        const uint32_t instanceIndex = intersection.instance_id;
        const auto inst = instances[instanceIndex];

        if (inst.mask == GEOMETRY_MASK_LIGHT)
        {
            // Hit a light -- record this as the last vertex with is_on_light = 1
            const float3 hitPoint = r.origin + r.direction * intersection.distance;
            device const UniformLight& currLight = lights[inst.userID];

            if (pathLength < uniforms.bdptStride)
            {
                device BDPTVertex& v = cameraVertices[linearPixelIndex * uniforms.bdptStride + pathLength];
                v.position        = packed_float3(hitPoint);
                v.geometry_normal  = packed_float3(calcLightNormal(currLight, hitPoint));
                v.shading_normal   = v.geometry_normal;
                v.throughput       = packed_float3(throughput);
                v.wo               = packed_float3(-direction);
                v.pdf_fwd          = 0.0f; // filled by forward path
                v.pdf_rev          = 0.0f;
                v.dVCM             = dVCM;
                v.dVC              = dVC;
                v.dVM              = dVM;
                v.uv_x             = 0.0f;
                v.uv_y             = 0.0f;
                v.material_index   = 0;
                v.event_type       = BSDF_EVENT_DIFFUSE_REFLECTION;
                v.is_delta         = 0;
                v.is_on_light      = 1;
                v.is_on_camera     = 0;
                v.light_index      = inst.userID;
                v.exterior_ior     = 1.0f;
                pathLength++;
            }
            break;
        }

        // Triangle hit -- extract geometry
        const Triangle triangle = *(const device Triangle*)intersection.primitive_data;
        float3 p0 = triangle.positions[0];
        float3 p1 = triangle.positions[1];
        float3 p2 = triangle.positions[2];
        float3 n0 = unpackNormal_cs(triangle.normals[0]);
        float3 n1 = unpackNormal_cs(triangle.normals[1]);
        float3 n2 = unpackNormal_cs(triangle.normals[2]);
        float3 t0 = unpackNormal_cs(triangle.tangent[0]);
        float3 t1 = unpackNormal_cs(triangle.tangent[1]);
        float3 t2 = unpackNormal_cs(triangle.tangent[2]);
        float2 uv0 = unpackUV_cs(triangle.uv[0]);
        float2 uv1 = unpackUV_cs(triangle.uv[1]);
        float2 uv2 = unpackUV_cs(triangle.uv[2]);

        const float4x4 objectToWorld = float4x4(
            float4(float3(inst.transformationMatrix[0]), 0.0f),
            float4(float3(inst.transformationMatrix[1]), 0.0f),
            float4(float3(inst.transformationMatrix[2]), 0.0f),
            float4(float3(inst.transformationMatrix[3]), 1.0f));

        const float2 barycentrics = intersection.triangle_barycentric_coord;
        const float3 worldPosition = r.origin + r.direction * intersection.distance;
        const float2 texUV = interpolateAttrib2_cs(uv0, uv1, uv2, barycentrics);
        const float3 objectNormal = normalize(interpolateAttrib3_cs(n0, n1, n2, barycentrics));
        const float3 worldNormal = normalize(transformDirection_cs(objectNormal, objectToWorld));
        const float3 worldTangent = normalize(transformDirection_cs(
            normalize(interpolateAttrib3_cs(t0, t1, t2, barycentrics)), objectToWorld));
        const float3 worldBinormal = cross(worldNormal, worldTangent);
        float3 geomNormal = cross(p1 - p0, p2 - p0);
        geomNormal = normalize(transformDirection_cs(geomNormal, objectToWorld));

        const uint32_t materialId = inst.userID;
        SurfaceInteraction si;
        initSurfaceInteraction_cs(si, materials[materialId],
            worldPosition, worldNormal, geomNormal,
            worldTangent, worldBinormal, texUV,
            direction);

        // Set exterior IOR from IOR stack
        bool entering = si.front_face;
        if (entering)
            si.exterior_ior = ior_stack_current_ior(iorStack);
        else
            si.exterior_ior = ior_stack_peek_after_pop(iorStack, si.dielectric_priority);

        // Compute pdf_fwd: the forward pdf of reaching this vertex from the previous one
        float dist2 = intersection.distance * intersection.distance;
        float cosIn = fabs(dot(si.shading_normal, -direction));
        cosIn = max(cosIn, 1e-6f);

        // Update MIS partial weights
        // dVCM = (dist^2 / cosIn) from previous vertex's dVC_prev
        // Here we update after computing the PDF
        dVCM *= dist2;
        dVCM /= cosIn;
        dVC  /= cosIn;
        dVM  /= cosIn;

        // Sample BSDF
        const float z1 = random<SampleDimension::eBSDF0>(sampler, uniforms.samplerType);
        const float z2 = random<SampleDimension::eBSDF1>(sampler, uniforms.samplerType);
        const float z3 = random<SampleDimension::eBSDF2>(sampler, uniforms.samplerType);
        const float z4 = random<SampleDimension::eBSDF3>(sampler, uniforms.samplerType);
        float4 xi = float4(z1, z2, z3, z4);

        BsdfSampleResult sampleResult = bsdf_sample(si, xi);
        if (sampleResult.event_type == BSDF_EVENT_ABSORB)
            break;

        bool isDelta = isDeltaEvent(sampleResult.event_type);
        float pdf_fwd = sampleResult.pdf;
        float pdf_rev = isDelta ? 0.0f : bsdf_pdf_reverse(si, sampleResult.wi);

        // Store this vertex
        if (pathLength < uniforms.bdptStride)
        {
            device BDPTVertex& v = cameraVertices[linearPixelIndex * uniforms.bdptStride + pathLength];
            v.position        = packed_float3(si.position);
            v.geometry_normal  = packed_float3(si.geometry_normal);
            v.shading_normal   = packed_float3(si.shading_normal);
            v.throughput       = packed_float3(throughput);
            v.wo               = packed_float3(si.wo);
            v.pdf_fwd          = pdf_fwd;
            v.pdf_rev          = pdf_rev;
            v.dVCM             = isDelta ? 0.0f : dVCM;
            v.dVC              = dVC;
            v.dVM              = isDelta ? 0.0f : dVM;
            v.uv_x             = texUV.x;
            v.uv_y             = texUV.y;
            v.material_index   = materialId;
            v.event_type       = sampleResult.event_type;
            v.is_delta         = isDelta ? 1 : 0;
            v.is_on_light      = 0;
            v.is_on_camera     = 0;
            v.light_index      = 0;
            v.exterior_ior     = si.exterior_ior;
            pathLength++;
        }

        // Update throughput
        throughput *= sampleResult.bsdf_over_pdf;

        // Russian roulette after depth 3
        if (bounce > 3)
        {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (random<SampleDimension::eRussianRoulette>(sampler, uniforms.samplerType) > p)
                break;
            throughput /= (p + 1e-5f);
        }

        if (dot(throughput, throughput) < 1e-4f)
            break;

        // Update MIS weights for next vertex
        if (isDelta)
        {
            // Delta: cannot connect or merge, so dVCM = 0
            float cosFactor = fabs(dot(si.shading_normal, sampleResult.wi)) / (pdf_fwd + 1e-10f);
            dVCM = 0.0f;
            dVC *= cosFactor;
            dVM *= cosFactor;
        }
        else
        {
            float cosOut = fabs(dot(si.shading_normal, sampleResult.wi));
            float factor = cosOut / (pdf_fwd + 1e-10f);
            // Split recursion for VCM without light tracing:
            //   dVC  = BDPT mode (no etaVCM) — used by connection kernel only
            //   dVM  = vcWeightFactor=1 to match BDPT-strength connections
            //          (connections use vmWF=0, so merge must see full connection weight)
            float vcmNvm = uniforms.vcmNvm;
            float vcmOn = (vcmNvm > 0.0f) ? 1.0f : 0.0f;
            dVM = factor * (dVM * pdf_rev + dVCM * vcmOn + vcmOn);
            dVC = factor * (dVC * pdf_rev + dVCM);
            dVCM = 1.0f / (pdf_fwd + 1e-10f);
        }

        // Setup next ray
        float3 faceNg = (dot(si.geometry_normal, si.wo) > 0.0f)
                      ? si.geometry_normal : -si.geometry_normal;
        if ((sampleResult.event_type & BSDF_EVENT_TRANSMISSION) != 0)
        {
            if (entering)
                ior_stack_push(iorStack, si.dielectric_priority, si.ior);
            else
                ior_stack_pop(iorStack, si.dielectric_priority);
            origin = offset_ray_cs(si.position, -faceNg);
        }
        else
        {
            origin = offset_ray_cs(si.position, faceNg);
        }
        direction = normalize(sampleResult.wi);

        ++sampler.depth;
    }

    cameraPathLengths[linearPixelIndex] = pathLength;
}
