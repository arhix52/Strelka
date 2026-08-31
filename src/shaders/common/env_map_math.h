#ifndef STRELKA_ENV_MAP_MATH_H
#define STRELKA_ENV_MAP_MATH_H

// ============================================================================
// env_map_math.h -- the equirectangular parametrisation and the density built
// on it, shared by both backends and by the host tests.
//
// Four small functions, and the reason they are here rather than in each
// backend is the same reason light_pdf.h exists: every one of them is used
// twice per environment sample, once to *draw* a direction and once to state
// its density for the MIS weight. Two hand-maintained copies of that pair is
// two chances for the halves to describe different maps.
//
// They were exactly that -- src/shaders/common/env_light.h and
// src/shaders/metal/env_light_metal.h held line-for-line duplicates, with a
// comment on each saying the two must stay identical. Nothing enforced it, and
// nothing tested either: the test suite could not compile a CUDA header or a
// Metal one. It can compile this.
//
// Deliberately free of CUDA, Metal and any texture type: directions and
// luminances in, uv and density out. Fetching the texel is the backend's job.
// ============================================================================

#include <strelka/material/material_math.h>

/// World-space direction to equirectangular uv.
///
/// `rotation` is a Y-axis rotation of the map, applied inverted here and
/// forwards in envUVToDir() so the two remain inverses of each other.
DEVICE_FUNC float2 dirToEnvUV(float3 dir, float rotation)
{
    const float cosR = cosf(-rotation);
    const float sinR = sinf(-rotation);
    const float rx = cosR * dir.x + sinR * dir.z;
    const float rz = -sinR * dir.x + cosR * dir.z;

    const float phi = atan2f(rx, rz); // [-pi, pi]
    const float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f)); // [0, pi]

    return make_float2((phi + M_PI_F) / (2.0f * M_PI_F), theta / M_PI_F);
}

/// Equirectangular uv back to a world-space direction.
DEVICE_FUNC float3 envUVToDir(float2 uv, float rotation)
{
    const float phi = uv.x * 2.0f * M_PI_F - M_PI_F;
    const float theta = uv.y * M_PI_F;

    const float sinTheta = sinf(theta);
    const float cosTheta = cosf(theta);

    const float x = sinTheta * sinf(phi);
    const float y = cosTheta;
    const float z = sinTheta * cosf(phi);

    const float cosR = cosf(rotation);
    const float sinR = sinf(rotation);
    return make_float3(cosR * x + sinR * z, y, -sinR * x + cosR * z);
}

/// The luminance the sampling distribution is built from.
///
/// Must match buildIblAliasTable() in render/host/ibl_alias_table.h exactly:
/// the host weights texels by this and the device divides by the result, so a
/// different set of coefficients on either side is a density for a map that was
/// never sampled.
DEVICE_FUNC float envLuminance(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

/// Solid-angle density of the texel a direction falls into.
///
/// The discrete probability of texel i is w_i / W with w_i = lum_i *
/// sin(theta_row), and the texel subtends dOmega = 2*pi^2*sin(theta_row)/(w*h).
/// Dividing them cancels sin(theta) outright, so the whole density collapses to
/// the texel's luminance times one precomputed constant:
///
///     envPdfScale = (w*h) / (2*pi^2 * totalPower)
///
/// which is why no CDF and no per-texel pdf array has to be stored or searched.
/// Integrating this over the texels' true solid angles comes to 1.000 to five
/// digits -- tests/render/test_env_map_math.cpp measures it.
DEVICE_FUNC float envTexelPdf(float3 radiance, float envPdfScale)
{
    return envLuminance(radiance) * envPdfScale;
}

#endif // STRELKA_ENV_MAP_MATH_H
