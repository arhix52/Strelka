#include "shading_common.h"


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
    device const GeometryEntry* geometryEntries                                      [[buffer(9)]],
    device const EnvAliasEntry* envAliasTable                                        [[buffer(10)]],
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
    prd.neeDone = false;
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
        intersection = i.intersect(ray, accelerationStructure, uniforms.primaryRayMask, motionTime);

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

                if (prd.depth == 0 || prd.specularBounce || !prd.neeDone)
                {
                    prd.radiance += prd.throughput * envColor;
                }
                else
                {
                    const float envPdf = envMapPdf(prd.direction,
                                                   envMapTexture,
                                                   uniforms.envMapWidth, uniforms.envMapHeight,
                                                   uniforms.envMapRotation,
                                                   uniforms.envPdfScale);
                    const float envSelectionPdf = (uniforms.numLights > 0) ? 0.5f : 1.0f;
                    const float effectiveEnvPdf = envPdf * envSelectionPdf;
                    // A texel of zero luminance has zero sampling density, so
                    // light sampling could never have produced this direction and
                    // the BSDF strategy owns it outright. Dropping it instead --
                    // which the guard used to do -- loses energy exactly along the
                    // edges of dark regions, where the bilinear radiance is still
                    // non-zero.
                    const float misWeight = (effectiveEnvPdf > 0.0f)
                                                ? misWeightBalance(prd.lastBsdfPdf, effectiveEnvPdf)
                                                : 1.0f;
                    prd.radiance += prd.throughput * envColor * misWeight;
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
                // One-sided emitter: only the front face radiates. The cosine at
                // the light must not scale the result — light.color is radiance,
                // and NEE (sampleLight) adds no such factor either. Scaling it
                // here made the two strategies estimate different quantities, so
                // MIS was blending inconsistent estimators.
                if (-dot(prd.direction, lightNormal) > 0.0f)
                {
                    const float3 Le = float3(currLight.color);
                    if (prd.depth == 0 || prd.specularBounce || !prd.neeDone)
                    {
                        prd.radiance += prd.throughput * Le;
                    }
                    else
                    {
                        // Account for env map selection probability in light PDF
                        const float lightSelectionPdf = uniforms.hasEnvMap
                            ? 0.5f / (float)uniforms.numLights
                            : 1.0f / (float)uniforms.numLights;
                        const float lightPdf = getLightPdf(currLight, hitPoint, ray.origin) * lightSelectionPdf;
                        const float misWeight = misWeightBalance(prd.lastBsdfPdf, lightPdf);
                        prd.radiance += prd.throughput * Le * misWeight;
                    }
                }
                prd.throughput = float3(0.0f);
                // stop tracing

                break;
            }

            // One acceleration structure holds many geometries, so the instance's
            // userID is a base offset into the per-geometry table and the
            // geometry index within the structure selects the entry. This is what
            // lets every primitive of a mesh share a single BLAS while keeping
            // its own material and vertex-buffer offsets.
            const uint32_t geomEntryIndex = inst.userID + intersection.geometry_id;

            // Reference the per-primitive record in place instead of copying all
            // 72 bytes into registers up front. Its fields are consumed at
            // different points — positions only to build the geometric normal,
            // then normals/tangents for the shading frame, then uvs — so loading
            // them lazily lets the compiler retire each group early. This kernel
            // is occupancy-limited by register pressure.
            device const Triangle* triangle = (device const Triangle*)intersection.primitive_data;

            // Per-primitive positions are from keyframe 1 (current VB) only.
            // For motion blur, we also need keyframe 0 positions to:
            //  (a) compute correct worldPosition via interpolated positions + barycentrics
            //  (b) compute correct geomNormal from the interpolated triangle
            float3 p0, p1, p2;
            float3 n0, n1, n2;
            float3 t0, t1, t2;

            if (uniforms.enableMotionBlur && motionTime < 1.0f &&
                prevVertexBuffer && indexBuffer && geometryEntries)
            {
                const uint32_t primitiveId = intersection.primitive_id;
                const GeometryEntry instData = geometryEntries[geomEntryIndex];

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
                p0 = mix(p0_prev, float3(triangle->positions[0]), motionTime);
                p1 = mix(p1_prev, float3(triangle->positions[1]), motionTime);
                p2 = mix(p2_prev, float3(triangle->positions[2]), motionTime);

                // Previous frame normals
                const float3 n0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + normalOff));
                const float3 n1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + normalOff));
                const float3 n2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + normalOff));

                // Interpolate normals: t=0 → prev (matches kf0=prevVB), t=1 → current (matches kf1=VB)
                n0 = mix(n0_prev, unpackNormal(triangle->normals[0]), motionTime);
                n1 = mix(n1_prev, unpackNormal(triangle->normals[1]), motionTime);
                n2 = mix(n2_prev, unpackNormal(triangle->normals[2]), motionTime);

                // Previous frame tangents
                const float3 t0_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i0) * vtxStride + tangentOff));
                const float3 t1_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i1) * vtxStride + tangentOff));
                const float3 t2_prev = unpackNormal(*(device const uint32_t*)(prevVertexBuffer + (instData.vbOffset + i2) * vtxStride + tangentOff));

                // Interpolate tangents: same direction as normals
                t0 = mix(t0_prev, unpackNormal(triangle->tangent[0]), motionTime);
                t1 = mix(t1_prev, unpackNormal(triangle->tangent[1]), motionTime);
                t2 = mix(t2_prev, unpackNormal(triangle->tangent[2]), motionTime);
            }
            else
            {
                p0 = float3(triangle->positions[0]);
                p1 = float3(triangle->positions[1]);
                p2 = float3(triangle->positions[2]);
                n0 = unpackNormal(triangle->normals[0]);
                n1 = unpackNormal(triangle->normals[1]);
                n2 = unpackNormal(triangle->normals[2]);
                t0 = unpackNormal(triangle->tangent[0]);
                t1 = unpackNormal(triangle->tangent[1]);
                t2 = unpackNormal(triangle->tangent[2]);
            }

            const float2 uv0 = unpackUV(triangle->uv[0]);
            const float2 uv1 = unpackUV(triangle->uv[1]);
            const float2 uv2 = unpackUV(triangle->uv[2]);

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

            const uint32_t materialId = geometryEntries[geomEntryIndex].materialId;

            SurfaceInteraction si;
            initSurfaceInteraction(si, materials[materialId],
                worldPosition, worldNormal, geomNormal,
                worldTangent, worldBinormal, uv,
                prd.direction);

            if (debugMode == DebugMode::eMotionBlur)
            {
                float3 nDelta = n0 - unpackNormal(triangle->normals[0]);
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
            prd.neeDone = (uniforms.estimatorMode == 0) &&
                          (sampleResult.event_type & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)) &&
                          (uniforms.numLights > 0 || uniforms.hasEnvMap);
            if (prd.neeDone)
            {
                float3 toLight;
                float lightPdf = 0.0f;
                const float3 radiance = estimateDirectLighting(uniforms, accelerationStructure, i,
                    uniforms.numLights, lights,
                    prd.sampler, si, toLight, lightPdf,
                    envAliasTable, envMapTexture, motionTime);

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

            // Narrow "NEE ran here" to "NEE could have generated a direction in
            // the hemisphere this ray is heading into". Light sampling only ever
            // returns directions above the shading normal of a front face — see
            // the guard in sampleEnvLightNEE and isNextEventValid above — so its
            // effective PDF elsewhere is zero. Applying a balance-heuristic
            // weight to such a hit would discount it against a strategy that
            // could never have produced it, and the two weights would sum to
            // less than one. Those hits must take full weight instead.
            prd.neeDone = prd.neeDone && si.front_face &&
                          dot(si.shading_normal, prd.direction) > 0.0f;

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
