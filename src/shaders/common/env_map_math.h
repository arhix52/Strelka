#ifndef STRELKA_ENV_MAP_MATH_H
#define STRELKA_ENV_MAP_MATH_H

// ============================================================================
// env_map_math.h -- the equirectangular parametrisation and the density built
// on it, shared by both backends and by the host tests.
//
// Small functions, and the reason they are here rather than in each
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
    // atan2(length(xz), y) remains resolvable next to the poles, where a
    // normalized float direction's y component can already have rounded to
    // +/-1 while x and z still carry its nonzero polar angle.
    const float radial = sqrtf(fmaxf(rx * rx + rz * rz, 0.0f));
    const float theta = atan2f(radial, fminf(fmaxf(dir.y, -1.0f), 1.0f)); // [0, pi]

    return make_float2((phi + M_PI_F) / (2.0f * M_PI_F), theta / M_PI_F);
}

/// Move the renderer's 23-bit [0,1) random lattice to cell centres. This keeps
/// finite-probability samples away from UV bin boundaries and, in particular,
/// away from the lat-long coordinate singularities at the two poles.
DEVICE_FUNC float envOpenUnitInterval(float xi)
{
    const float centred = xi + 0x1p-24f;
    return fminf(fmaxf(centred, 0x1p-24f), 0x1.fffffep-1f);
}

/// A representable normalized coordinate whose lookup remains in texel `x`.
/// If the first rounded value crosses a boundary, retain the input coordinate
/// and project it monotonically to the nearest representable interior value.
DEVICE_FUNC float envSampleTexelU(int x, int width, float xi)
{
    const float w = (float)width;
    const float t = envOpenUnitInterval(xi);
    float u = ((float)x + t) / w;
    if ((int)(u * w) != x)
    {
        float accepted = 0.5f;
        float rejected = t;
        u = ((float)x + accepted) / w;
        for (int i = 0; i < 24; ++i)
        {
            const float trial = accepted + 0.5f * (rejected - accepted);
            const float trialU = ((float)x + trial) / w;
            if ((int)(trialU * w) == x)
            {
                accepted = trial;
                u = trialU;
            }
            else
            {
                rejected = trial;
            }
        }
    }
    return u;
}

DEVICE_FUNC float envSolidAngleRowV(int y, int height, float t)
{
    const float theta0 = M_PI_F * (float)y / (float)height;
    const float theta1 = M_PI_F * (float)(y + 1) / (float)height;
    const float approximateCosTheta = cosf(theta0) + (cosf(theta1) - cosf(theta0)) * t;

    // Inverting cos(theta) directly loses the first representable samples next
    // to either pole: 1-cos(theta) (or 1+cos(theta)) rounds away. Work in the
    // nearer half-angle cap, whose squared sine is linear in cos(theta).
    float theta;
    if (approximateCosTheta >= 0.0f)
    {
        const float sinHalf0 = sinf(0.5f * theta0);
        const float sinHalf1 = sinf(0.5f * theta1);
        const float sinHalfSquared = sinHalf0 * sinHalf0 + (sinHalf1 * sinHalf1 - sinHalf0 * sinHalf0) * t;
        theta = 2.0f * asinf(sqrtf(fminf(fmaxf(sinHalfSquared, 0.0f), 1.0f)));
    }
    else
    {
        const float southHalf0 = sinf(0.5f * (M_PI_F - theta0));
        const float southHalf1 = sinf(0.5f * (M_PI_F - theta1));
        const float southHalfSquared = southHalf0 * southHalf0 + (southHalf1 * southHalf1 - southHalf0 * southHalf0) * t;
        theta = M_PI_F - 2.0f * asinf(sqrtf(fminf(fmaxf(southHalfSquared, 0.0f), 1.0f)));
    }
    return theta / M_PI_F;
}

/// Sample v within lat-long row `y` uniformly in solid angle.
///
/// Uniform v would make theta uniform and induce a 1/sin(theta) directional
/// density. Interpolating cos(theta) instead makes the conditional density
/// constant over the row's exact solid angle.
DEVICE_FUNC float envSampleSolidAngleV(int y, int height, float xi)
{
    return envSolidAngleRowV(y, height, envOpenUnitInterval(xi));
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

DEVICE_FUNC uint32_t envJitterWord(float xi)
{
    const uint32_t word = (uint32_t)(fminf(fmaxf(xi, 0.0f), 0x1.fffffep-1f) * 8388608.0f);
    return word < 8388608u ? word : 8388607u;
}

DEVICE_FUNC float envWordJitter(uint32_t word)
{
    return (float)(word >> 9u) * 0x1p-23f;
}

/// Sample a direction whose finite-precision inverse mapping still belongs to
/// the selected texel. A mismatch is a numerical rejection, not a new event:
/// retry from independent random dimensions rather than concentrating all such
/// samples at one midpoint.
DEVICE_FUNC float3 envSampleTexelDirection(
    int x, int y, int width, int height, float xiU, float xiV, uint32_t retryU, uint32_t retryV, float rotation)
{
    const uint32_t originalU = envJitterWord(xiU);
    const uint32_t originalV = envJitterWord(xiV);
    uint32_t stateU = retryU ^ originalV;
    uint32_t stateV = retryV ^ originalU ^ originalV * 0x9e3779b9u;
    float3 direction = make_float3(0.0f, 1.0f, 0.0f);
    for (int attempt = 0; attempt < 9; ++attempt)
    {
        const float u = envSampleTexelU(x, width, xiU);
        const float v = envSampleSolidAngleV(y, height, xiV);
        direction = envUVToDir(make_float2(u, v), rotation);
        const float2 evaluated = dirToEnvUV(direction, rotation);
        int evaluatedX = (int)(evaluated.x * (float)width);
        int evaluatedY = (int)(evaluated.y * (float)height);
        evaluatedX = evaluatedX < 0 ? 0 : (evaluatedX >= width ? width - 1 : evaluatedX);
        evaluatedY = evaluatedY < 0 ? 0 : (evaluatedY >= height ? height - 1 : evaluatedY);
        if (evaluatedX == x && evaluatedY == y)
        {
            return direction;
        }

        stateU = stateU * 1664525u + 1013904223u;
        stateV = stateV * 1664525u + 1013904223u;
        xiU = envWordJitter(stateU);
        xiV = envWordJitter(stateV);
    }

    const float u = ((float)x + 0.5f) / (float)width;
    const float v = envSolidAngleRowV(y, height, 0.5f);
    return envUVToDir(make_float2(u, v), rotation);
}

/// Sanitized environment luminance. The host uses these coefficients before
/// constructing its bilinear-footprint proposal envelope.
DEVICE_FUNC float envLuminance(float3 rgb)
{
    constexpr float maxFinite = 3.402823466e38f;
    // Comparisons reject NaN as well as infinity. This mirrors the host alias
    // builder: an invalid texel has zero mass and therefore must also report
    // zero density when a BSDF direction happens to query it.
    if (!(rgb.x >= -maxFinite && rgb.x <= maxFinite && rgb.y >= -maxFinite && rgb.y <= maxFinite &&
          rgb.z >= -maxFinite && rgb.z <= maxFinite))
    {
        return 0.0f;
    }

    // Scaling first avoids overflowing an intermediate for finite HDR inputs.
    const float magnitude = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
    if (!(magnitude > 0.0f))
    {
        return 0.0f;
    }
    const float lum = (0.2126f * fmaxf(rgb.x / magnitude, 0.0f) + 0.7152f * fmaxf(rgb.y / magnitude, 0.0f) +
                       0.0722f * fmaxf(rgb.z / magnitude, 0.0f)) *
                      magnitude;
    return (lum > 0.0f && lum <= maxFinite) ? lum : 0.0f;
}

/// Compatibility helper for standalone audit kernels. Production map PDFs
/// come directly from EnvAliasEntry::solidAnglePdf.
DEVICE_FUNC float envTexelPdf(float3 radiance, float normalization)
{
    return envLuminance(radiance) * normalization;
}

#endif // STRELKA_ENV_MAP_MATH_H
