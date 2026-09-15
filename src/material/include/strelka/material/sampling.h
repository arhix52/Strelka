#ifndef STRELKA_SAMPLING_H
#define STRELKA_SAMPLING_H

// ============================================================================
// sampling.h -- Hemisphere sampling utilities and tangent-space transforms
// ============================================================================

#include "material_math.h"

DEVICE_FUNC float3 cosine_hemisphere_sample(float u1, float u2)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI_F * u2;

    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    return make_float3(x, y, z);
}

DEVICE_FUNC float cosine_hemisphere_pdf(float cos_theta)
{
    return fmaxf(cos_theta, 0.0f) * M_1_PI_F;
}

// ---------------------------------------------------------------------------
// Uniform hemisphere sampling
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 uniform_hemisphere_sample(float u1, float u2)
{
    const float z = u1;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * M_PI_F * u2;

    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

DEVICE_FUNC float uniform_hemisphere_pdf()
{
    return M_1_PI_F * 0.5f; // 1 / (2*pi)
}

DEVICE_FUNC void build_onb(float3 N, THREAD_REF float3& T, THREAD_REF float3& B)
{
    if (N.z < -0.9999999f)
    {
        T = make_float3( 0.0f, -1.0f, 0.0f);
        B = make_float3(-1.0f,  0.0f, 0.0f);
        return;
    }
    const float a = 1.0f / (1.0f + N.z);
    const float b = -N.x * N.y * a;
    T = make_float3(1.0f - N.x * N.x * a, b, -N.x);
    B = make_float3(b, 1.0f - N.y * N.y * a, -N.y);
}

DEVICE_FUNC float3 local_to_world(float3 local, float3 T, float3 B, float3 N)
{
    return local.x * T + local.y * B + local.z * N;
}

DEVICE_FUNC float3 world_to_local(float3 world, float3 T, float3 B, float3 N)
{
    return make_float3(dot(world, T), dot(world, B), dot(world, N));
}

// ---------------------------------------------------------------------------
// Power-heuristic for MIS (beta = 2)
// ---------------------------------------------------------------------------
DEVICE_FUNC float power_heuristic(float pdf_a, float pdf_b)
{
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-10f);
}

#endif // STRELKA_SAMPLING_H
