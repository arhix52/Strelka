// ============================================================================
// vcm_merge.metal -- VCM Vertex Merging Kernel
//
// For each non-delta camera vertex, queries the spatial hash grid for
// nearby light vertices and accumulates the kernel density estimate
// contribution with VCM MIS weighting.
//
// The merge result is added to the output buffer which already contains
// the connection kernel's result. Accumulation is handled here for VCM
// (the connect kernel skips accumulation when integratorType == 2).
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "bdpt_common.h"
#include "ShaderTypes.h"
#include "vcm_types.h"

using namespace metal;

// Teschner spatial hash (same as build kernel)
static inline uint32_t teschnerHash_m(int3 cell)
{
    uint32_t h = (uint32_t)(cell.x * 73856093) ^
                 (uint32_t)(cell.y * 19349663) ^
                 (uint32_t)(cell.z * 83492791);
    return h & (VCM_HASH_SIZE - 1);
}

// Maximum merges per camera vertex — caps worst-case in dense regions
#define VCM_MAX_MERGES_PER_VERTEX 64

kernel void vcm_merge(
    uint2                       tid              [[thread_position_in_grid]],
    constant Uniforms&          uniforms         [[buffer(0)]],
    device const BDPTVertex*    cameraVertices   [[buffer(1)]],
    device const uint32_t*      cameraPathLengths [[buffer(2)]],
    device const BDPTVertex*    lightVertices    [[buffer(3)]],
    device const uint32_t*      lightPathLengths [[buffer(4)]],
    device const uint32_t*      hashHeads        [[buffer(5)]],
    device const VCMHashEntry*  hashEntries      [[buffer(6)]],
    device Material*            materials        [[buffer(7)]],
    device float4*              outputBuffer     [[buffer(8)]],
    device float4*              accumBuffer      [[buffer(9)]]
)
{
    if (tid.x >= uniforms.width || tid.y >= uniforms.height)
        return;

    const uint32_t linearPixelIndex = tid.y * uniforms.width + tid.x;
    const uint32_t cameraLen = cameraPathLengths[linearPixelIndex];

    const float radiusSqr = uniforms.vcmMergeRadiusSqr;
    const float cellSize = uniforms.vcmHashCellSize;

    float3 mergeResult = float3(0.0f);

    if (cellSize > 0.0f && radiusSqr > 0.0f)
    {
        const float invCellSize = 1.0f / cellSize;
        const float invNvm = 1.0f / (uniforms.vcmNvm + 1e-10f);
        const float radius = uniforms.vcmMergeRadius;
        const float vcWeightFactor = (uniforms.vcmNvm > 0.0f) ? (1.0f / uniforms.vcmNvm) : 0.0f;

        // Iterate over non-delta camera vertices (skip vertex 0 = on lens)
        for (uint32_t t = 1; t < cameraLen; ++t)
        {
            device const BDPTVertex& cv = cameraVertices[linearPixelIndex * BDPT_MAX_DEPTH + t];

            // Skip delta, light-hit, or camera vertices
            if (cv.is_delta || cv.is_on_light || cv.is_on_camera)
                continue;

            float3 camPos = float3(cv.position);
            float3 camNormal = float3(cv.shading_normal);
            float3 camThroughput = float3(cv.throughput);

            // Reconstruct camera surface interaction (once per camera vertex)
            SurfaceInteraction si_cam = vertexToSI(cv, materials);

            // Compute the actual cell range the radius covers (may be less than 3x3x3)
            int3 minCell = int3(floor((camPos - radius) * invCellSize));
            int3 maxCell = int3(floor((camPos + radius) * invCellSize));

            uint32_t mergeCount = 0;

            for (int cz = minCell.z; cz <= maxCell.z; ++cz)
            for (int cy = minCell.y; cy <= maxCell.y; ++cy)
            for (int cx = minCell.x; cx <= maxCell.x; ++cx)
            {
                uint32_t bucket = teschnerHash_m(int3(cx, cy, cz));
                uint32_t entryIdx = hashHeads[bucket];

                // Walk linked list (capped)
                uint32_t listIter = 0;
                while (entryIdx != VCM_HASH_INVALID && listIter < 128)
                {
                    ++listIter;
                    device const VCMHashEntry& entry = hashEntries[entryIdx];
                    device const BDPTVertex& lv = lightVertices[entry.vertexIndex];

                    // Distance check first (cheapest rejection)
                    float3 lightPos = float3(lv.position);
                    float3 diff = camPos - lightPos;
                    float dist2 = dot(diff, diff);

                    if (dist2 < radiusSqr)
                    {
                        // Normal compatibility: skip if normals face away
                        // (merging across opposite hemispheres is physically invalid)
                        float3 lightNormal = float3(lv.shading_normal);
                        float normalDot = dot(camNormal, lightNormal);
                        if (normalDot > -0.1f)  // allow coplanar merges
                        {
                            float3 lightWo = float3(lv.wo);

                            // Evaluate BSDF at camera vertex for the light arrival direction.
                            // lightWo points from merge point toward previous light vertex
                            // (toward the light source) — correct BSDF wi convention.
                            BsdfEvalResult evalCam = bsdf_eval(si_cam, lightWo);
                            if (evalCam.pdf > 0.0f)
                            {
                                float3 lightThroughput = float3(lv.throughput);

                                // Reverse PDF at camera vertex (SmallVCM: cameraBsdfRevPdfW)
                                float cameraBsdfRevPdf = bsdf_pdf_reverse(si_cam, lightWo);

                                // Epanechnikov kernel weight (2D normalization: 2/π per unit disk)
                                float kernel_weight = 2.0f * (1.0f - dist2 / radiusSqr) * invNvm;

                                // VCM MIS weight for merging (SmallVCM formula)
                                float misWeight = vcmMergeMISWeight(
                                    cv.dVCM, cv.dVM,
                                    lv.dVCM, lv.dVM,
                                    evalCam.pdf, cameraBsdfRevPdf,
                                    vcWeightFactor);

                                mergeResult += camThroughput * evalCam.bsdf * kernel_weight * lightThroughput * misWeight;
                            }
                        }
                        ++mergeCount;
                        if (mergeCount >= VCM_MAX_MERGES_PER_VERTEX)
                            break;
                    }

                    entryIdx = entry.next;
                }
                if (mergeCount >= VCM_MAX_MERGES_PER_VERTEX)
                    break;
            }
        }
    }

    // Read connection result from output buffer (written by connect kernel)
    float3 connectResult = float3(outputBuffer[linearPixelIndex]);
    float3 result = connectResult + mergeResult;

    // Accumulation (VCM handles it here since connect kernel writes raw result)
    if (uniforms.enableAccumulation)
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
    }

    outputBuffer[linearPixelIndex] = float4(result, 1.0f);
}
