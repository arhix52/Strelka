// ============================================================================
// bdpt_light_subpath.metal -- BDPT Light Subpath Tracing Kernel
//
// Traces light subpaths starting from emission sampling.
// Records BDPTVertex at each bounce with MIS partial weights.
// Applies adjoint correction at each bounce.
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "light_emission_metal.h"
#include "bdpt_common.h"

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h>
#include <strelka/material/bsdf.h>

using namespace metal;
using namespace raytracing;

// Shared utilities (same as camera subpath -- duplicated for separate metallib)
__attribute__((always_inline))
static float3 transformDirection_ls(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 0.0f)).xyz;
}

static float3 unpackNormal_ls(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0xfff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

static float2 unpackUV_ls(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

static float3 interpolateAttrib3_ls(const float3 a1, const float3 a2, const float3 a3, const float2 bary)
{
    return a1 * (1.0f - bary.x - bary.y) + a2 * bary.x + a3 * bary.y;
}

static float2 interpolateAttrib2_ls(const float2 a1, const float2 a2, const float2 a3, const float2 bary)
{
    return a1 * (1.0f - bary.x - bary.y) + a2 * bary.x + a3 * bary.y;
}

__attribute__((always_inline))
static int __float_as_int_ls(float x) { return as_type<int>(x); }
__attribute__((always_inline))
static float __int_as_float_ls(int x) { return as_type<float>(x); }

static float3 offset_ray_ls(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float_ls(__float_as_int_ls(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        __int_as_float_ls(__float_as_int_ls(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        __int_as_float_ls(__float_as_int_ls(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

static void initSurfaceInteraction_ls(
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

kernel void bdpt_light_subpath(
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
    device BDPTVertex*                                         lightVertices        [[buffer(10)]],
    device uint32_t*                                           lightPathLengths     [[buffer(11)]],
    texture2d<float>                                           envMapTexture        [[texture(0)]]
    )
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    const uint32_t maxDepth = min(uniforms.maxLightSubpathDepth, (uint32_t)BDPT_MAX_DEPTH);

    // Initialize sampler with different seed for light paths
    SamplerState sampler = initSampler(linearPixelIndex + uniforms.width * uniforms.height,
                                       uniforms.subframeIndex, 1u);

    // Sample light emission
    LightEmissionSample emission = sampleLightEmission(
        uniforms, lights, sampler, envCdfX, envCdfY, envMapTexture);

    if (emission.pdf_pos <= 0.0f || emission.pdf_dir <= 0.0f ||
        (emission.Le.x <= 0.0f && emission.Le.y <= 0.0f && emission.Le.z <= 0.0f))
    {
        lightPathLengths[linearPixelIndex] = 0;
        return;
    }

    // Light vertex 0 (on light surface)
    // Vertex 0 throughput for connections: Le / pdf_pos (no pdf_dir -- the emission
    // angular profile is applied at connection time for the actual connection direction).
    // Working throughput for continuing the light path includes pdf_dir.
    float3 vertex0Throughput = emission.Le / (emission.pdf_pos + 1e-10f);
    float3 throughput = emission.Le / (emission.pdf_pos * emission.pdf_dir);

    // SmallVCM: dVCM = pdf_dir / pdf_pos, dVC = 0, dVM = 0
    float dVCM = emission.pdf_dir / (emission.pdf_pos + 1e-10f);
    float dVC  = 0.0f;
    float dVM  = 0.0f;

    uint32_t pathLength = 0;

    // Store light vertex 0
    {
        device BDPTVertex& v = lightVertices[linearPixelIndex * BDPT_MAX_DEPTH + 0];
        v.position        = packed_float3(emission.position);
        v.geometry_normal  = packed_float3(emission.normal);
        v.shading_normal   = packed_float3(emission.normal);
        v.throughput       = packed_float3(vertex0Throughput);
        v.wo               = packed_float3(emission.direction);
        v.pdf_fwd          = emission.pdf_pos;
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
        v.light_index      = (uint32_t)max(emission.light_index, 0);
        pathLength = 1;
    }

    float3 origin = offset_ray_ls(emission.position, emission.normal);
    float3 direction = emission.direction;

    IorStack iorStack;
    ior_stack_init(iorStack);

    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);

    for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)
    {
        ray r;
        r.min_distance = 0.0f;
        r.max_distance = INFINITY;
        r.origin = origin;
        r.direction = direction;

        isect.accept_any_intersection(false);
        auto intersection = isect.intersect(r, accelerationStructure, RAY_MASK_SECONDARY, 1.0f);

        if (intersection.type == intersection_type::none)
            break;

        const uint32_t instanceIndex = intersection.instance_id;
        const auto inst = instances[instanceIndex];

        // Skip light geometry hits for light subpaths
        if (inst.mask == GEOMETRY_MASK_LIGHT)
            break;

        // Extract triangle geometry
        const Triangle triangle = *(const device Triangle*)intersection.primitive_data;
        float3 p0 = triangle.positions[0];
        float3 p1 = triangle.positions[1];
        float3 p2 = triangle.positions[2];
        float3 n0 = unpackNormal_ls(triangle.normals[0]);
        float3 n1 = unpackNormal_ls(triangle.normals[1]);
        float3 n2 = unpackNormal_ls(triangle.normals[2]);
        float3 t0 = unpackNormal_ls(triangle.tangent[0]);
        float3 t1 = unpackNormal_ls(triangle.tangent[1]);
        float3 t2 = unpackNormal_ls(triangle.tangent[2]);
        float2 uv0 = unpackUV_ls(triangle.uv[0]);
        float2 uv1 = unpackUV_ls(triangle.uv[1]);
        float2 uv2 = unpackUV_ls(triangle.uv[2]);

        const float4x4 objectToWorld = float4x4(
            float4(float3(inst.transformationMatrix[0]), 0.0f),
            float4(float3(inst.transformationMatrix[1]), 0.0f),
            float4(float3(inst.transformationMatrix[2]), 0.0f),
            float4(float3(inst.transformationMatrix[3]), 1.0f));

        const float2 barycentrics = intersection.triangle_barycentric_coord;
        const float3 worldPosition = r.origin + r.direction * intersection.distance;
        const float2 texUV = interpolateAttrib2_ls(uv0, uv1, uv2, barycentrics);
        const float3 objectNormal = normalize(interpolateAttrib3_ls(n0, n1, n2, barycentrics));
        const float3 worldNormal = normalize(transformDirection_ls(objectNormal, objectToWorld));
        const float3 worldTangent = normalize(transformDirection_ls(
            normalize(interpolateAttrib3_ls(t0, t1, t2, barycentrics)), objectToWorld));
        const float3 worldBinormal = cross(worldNormal, worldTangent);
        float3 geomNormal = cross(p1 - p0, p2 - p0);
        geomNormal = normalize(transformDirection_ls(geomNormal, objectToWorld));

        const uint32_t materialId = inst.userID;
        SurfaceInteraction si;
        initSurfaceInteraction_ls(si, materials[materialId],
            worldPosition, worldNormal, geomNormal,
            worldTangent, worldBinormal, texUV,
            direction);

        bool entering = si.front_face;
        if (entering)
            si.exterior_ior = ior_stack_current_ior(iorStack);
        else
            si.exterior_ior = ior_stack_peek_after_pop(iorStack, si.dielectric_priority);

        // Update MIS weights with distance and cos
        float dist2 = intersection.distance * intersection.distance;
        float cosIn = fabs(dot(si.shading_normal, -direction));
        cosIn = max(cosIn, 1e-6f);

        dVCM *= dist2;
        dVCM /= cosIn;
        dVC  /= cosIn;
        dVM  /= cosIn;

        // Sample BSDF
        ++sampler.depth;
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

        // Apply adjoint correction (light tracing uses non-symmetric BSDF)
        float adj = adjoint_correction(si, sampleResult.wi);

        // Store this vertex
        if (pathLength < BDPT_MAX_DEPTH)
        {
            device BDPTVertex& v = lightVertices[linearPixelIndex * BDPT_MAX_DEPTH + pathLength];
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
            pathLength++;
        }

        // Update throughput with adjoint correction
        throughput *= sampleResult.bsdf_over_pdf * adj;

        // Russian roulette
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
            float cosFactor = fabs(dot(si.shading_normal, sampleResult.wi)) / (pdf_fwd + 1e-10f);
            dVCM = 0.0f;
            dVC *= cosFactor;
            dVM *= cosFactor;
        }
        else
        {
            float cosOut = fabs(dot(si.shading_normal, sampleResult.wi));
            float factor = cosOut / (pdf_fwd + 1e-10f);
            dVM = factor * (dVCM * uniforms.vcmNvm + dVM * pdf_rev);
            dVC = factor * (dVCM + dVC * pdf_rev);
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
            origin = offset_ray_ls(si.position, -faceNg);
        }
        else
        {
            origin = offset_ray_ls(si.position, faceNg);
        }
        direction = normalize(sampleResult.wi);
    }

    lightPathLengths[linearPixelIndex] = pathLength;
}
