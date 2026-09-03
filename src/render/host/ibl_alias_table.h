#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif


namespace oka::metal
{

// Host-side mirror of ShaderTypes.h EnvAliasEntry (layout must stay identical).
struct EnvAliasEntry
{
    float prob = 1.0f;
    uint32_t alias = 0;
};

struct IblAliasTableResult
{
    std::vector<EnvAliasEntry> alias;
    float envPdfScale = 0.0f;
    double totalPower = 0.0;
    double averageWeightedLuminance = 0.0;
};

namespace detail
{

enum class IblRowMeasure
{
    CentreJacobian,
    ExactSolidAngle,
};

inline IblAliasTableResult buildIblAliasTable(const float* pixelRgba, int width, int height, IblRowMeasure rowMeasure)
{
    IblAliasTableResult out;
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return out;
    }

    const size_t texelCount = (size_t)width * (size_t)height;
    std::vector<double> weights(texelCount);
    double totalPower = 0.0;
    double calibrationPower = 0.0;
    const double deltaPhi = 2.0 * M_PI / (double)width;

    for (int y = 0; y < height; ++y)
    {
        const double theta0 = M_PI * (double)y / (double)height;
        const double theta1 = M_PI * (double)(y + 1) / (double)height;
        const double thetaCentre = 0.5 * (theta0 + theta1);
        const double centreJacobian = std::sin(thetaCentre);
        const double rowSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
        const double rowWeight = rowMeasure == IblRowMeasure::ExactSolidAngle ? rowSolidAngle : centreJacobian;
        for (int x = 0; x < width; ++x)
        {
            const size_t i = (size_t)y * (size_t)width + (size_t)x;
            const float* px = pixelRgba + i * 4;
            const double lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            // A NaN texel is not a rare thing in a downloaded HDRI, and without
            // this test one of them takes the whole map with it: totalPower
            // becomes NaN, envPdfScale becomes NaN, and every density the
            // shaders compute -- for sampling and for the MIS weight alike --
            // is NaN. std::max propagates it rather than clamping it, so the
            // guard has to be an explicit finiteness test.
            const double clean = std::isfinite(lum) ? std::max(lum, 0.0) : 0.0;
            const double w = clean * rowWeight;
            weights[i] = w;
            totalPower += w;
            calibrationPower += clean * centreJacobian;
        }
    }

    out.alias.resize(texelCount);
    out.totalPower = totalPower;
    out.averageWeightedLuminance = calibrationPower / (double)texelCount;

    if (totalPower > 0.0)
    {
        const double scale = (double)texelCount / totalPower;
        std::vector<double> p(texelCount);
        std::vector<uint32_t> small;
        std::vector<uint32_t> large;
        small.reserve(texelCount / 2);
        large.reserve(texelCount / 2);
        for (size_t i = 0; i < texelCount; ++i)
        {
            p[i] = weights[i] * scale;
            (p[i] < 1.0 ? small : large).push_back((uint32_t)i);
        }

        while (!small.empty() && !large.empty())
        {
            const uint32_t l = small.back();
            small.pop_back();
            const uint32_t g = large.back();
            large.pop_back();

            out.alias[l].prob = (float)p[l];
            out.alias[l].alias = g;

            p[g] = (p[g] + p[l]) - 1.0;
            (p[g] < 1.0 ? small : large).push_back(g);
        }
        for (const uint32_t i : large)
        {
            out.alias[i].prob = 1.0f;
            out.alias[i].alias = i;
        }
        for (const uint32_t i : small)
        {
            out.alias[i].prob = 1.0f;
            out.alias[i].alias = i;
        }
    }
    else
    {
        for (size_t i = 0; i < texelCount; ++i)
        {
            out.alias[i].prob = 1.0f;
            out.alias[i].alias = (uint32_t)i;
        }
    }

    if (totalPower > 0.0)
    {
        out.envPdfScale = rowMeasure == IblRowMeasure::ExactSolidAngle ?
                              (float)(1.0 / totalPower) :
                              (float)((double)texelCount / (2.0 * M_PI * M_PI * totalPower));
    }
    return out;
}

} // namespace detail

// Legacy centre-Jacobian distribution used by the OptiX uniform-v sampler.
// Keep this entry point until that backend is explicitly migrated: pairing an
// exact-solid-angle alias table with its current uniform-v jitter would make
// sample() and pdf() disagree by a row-dependent Jacobian.
inline IblAliasTableResult buildIblAliasTable(const float* pixelRgba, int width, int height)
{
    return detail::buildIblAliasTable(pixelRgba, width, height, detail::IblRowMeasure::CentreJacobian);
}

// Walker/Vose alias table for a piecewise-constant lat-long environment in the
// continuous solid-angle measure. Texel i receives mass
//
//     P_i = luminance_i * DeltaOmega_i / sum_j(luminance_j * DeltaOmega_j)
//
// where DeltaOmega_i is the exact solid angle of its row segment. The Metal
// sampler draws cos(theta) uniformly between the selected row's boundaries, so
// its directional density is simply luminance_i * envPdfScale.
inline IblAliasTableResult buildSolidAngleIblAliasTable(const float* pixelRgba, int width, int height)
{
    return detail::buildIblAliasTable(pixelRgba, width, height, detail::IblRowMeasure::ExactSolidAngle);
}

} // namespace oka::metal
