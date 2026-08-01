#ifndef STRELKA_SAMPLING_H
#define STRELKA_SAMPLING_H

// ============================================================================
// sampling.h -- Hemisphere sampling utilities and tangent-space transforms
// ============================================================================

#include "material_math.h"

// ---------------------------------------------------------------------------
// Cosine-weighted hemisphere sampling (Malley's method)
//
// u1, u2 in [0,1) -- uniform random numbers
// Returns direction in tangent space where Z = up (normal direction)
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 cosine_hemisphere_sample(float u1, float u2)
{
    float r   = sqrtf(u1);
    float phi = 2.0f * M_PI_F * u2;

    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    return make_float3(x, y, z);
}

// ---------------------------------------------------------------------------
// PDF of cosine-weighted hemisphere sampling
// cos_theta = dot(N, sampled_dir)
// ---------------------------------------------------------------------------
DEVICE_FUNC float cosine_hemisphere_pdf(float cos_theta)
{
    return fmaxf(cos_theta, 0.0f) * M_1_PI_F;
}

// ---------------------------------------------------------------------------
// Uniform hemisphere sampling
// ---------------------------------------------------------------------------
DEVICE_FUNC float3 uniform_hemisphere_sample(float u1, float u2)
{
    float z   = u1;
    float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI_F * u2;

    return make_float3(r * cosf(phi), r * sinf(phi), z);
}

DEVICE_FUNC float uniform_hemisphere_pdf()
{
    return M_1_PI_F * 0.5f; // 1 / (2*pi)
}

// ---------------------------------------------------------------------------
// Build an orthonormal basis from a normal vector (Frisvad / Duff et al.)
// Returns tangent T and bitangent B such that (T, B, N) is right-handed.
// ---------------------------------------------------------------------------
DEVICE_FUNC void build_onb(float3 N, THREAD_REF float3& T, THREAD_REF float3& B)
{
    if (N.z < -0.9999999f)
    {
        T = make_float3( 0.0f, -1.0f, 0.0f);
        B = make_float3(-1.0f,  0.0f, 0.0f);
        return;
    }
    float a = 1.0f / (1.0f + N.z);
    float b = -N.x * N.y * a;
    T = make_float3(1.0f - N.x * N.x * a, b, -N.x);
    B = make_float3(b, 1.0f - N.y * N.y * a, -N.y);
}

// ---------------------------------------------------------------------------
// Tangent-space <-> World-space transforms
//
// local  = tangent-space direction (Z = normal)
// T, B, N = orthonormal basis vectors in world space
// ---------------------------------------------------------------------------
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
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(a2 + b2, 1e-10f);
}

#endif // STRELKA_SAMPLING_H
