#pragma once

#include <host/light_selection.h>

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
    float solidAnglePdf = 0.0f;
};

struct IblAliasTableResult
{
    std::vector<EnvAliasEntry> alias;
    float envPdfScale = 0.0f;
    double totalPower = 0.0;
    double averageWeightedLuminance = 0.0;
};

// Walker/Vose alias table for a piecewise-constant lat-long environment in the
// continuous solid-angle measure. Texel i receives mass
//
//     P_i = luminance_i * DeltaOmega_i / sum_j(luminance_j * DeltaOmega_j)
//
// where DeltaOmega_i is the exact solid angle of its row segment. Both GPU
// samplers draw cos(theta) uniformly inside the selected row, so the directional
// density is its represented texel PMF divided by DeltaOmega_i.
inline IblAliasTableResult buildSolidAngleIblAliasTable(const float* pixelRgba, int width, int height)
{
    IblAliasTableResult out;
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return out;
    }

    const size_t texelCount = (size_t)width * (size_t)height;
    std::vector<double> weights(texelCount);
    double calibrationPower = 0.0;
    const double deltaPhi = 2.0 * M_PI / (double)width;

    for (int y = 0; y < height; ++y)
    {
        const double theta0 = M_PI * (double)y / (double)height;
        const double theta1 = M_PI * (double)(y + 1) / (double)height;
        const double rowSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < width; ++x)
        {
            const size_t i = (size_t)y * (size_t)width + (size_t)x;
            const float* px = pixelRgba + i * 4;
            const bool finite = std::isfinite(px[0]) && std::isfinite(px[1]) && std::isfinite(px[2]);
            const double lum = finite ? 0.2126 * std::max(px[0], 0.0f) + 0.7152 * std::max(px[1], 0.0f) +
                                            0.0722 * std::max(px[2], 0.0f) :
                                        0.0;
            // A NaN texel is not a rare thing in a downloaded HDRI, and without
            // this test one of them takes the whole map with it: totalPower
            // becomes NaN, envPdfScale becomes NaN, and every density the
            // shaders compute -- for sampling and for the MIS weight alike --
            // is NaN. std::max propagates it rather than clamping it, so the
            // guard has to be an explicit finiteness test.
            const double clean = std::isfinite(lum) ? lum : 0.0;
            const double w = clean * rowSolidAngle;
            weights[i] = w;
            calibrationPower += clean * std::sin(0.5 * (theta0 + theta1));
        }
    }

    const LightSelectionTable selection = buildLightSelectionAlias(weights);
    out.alias.resize(texelCount);
    out.totalPower = selection.totalPower;
    out.averageWeightedLuminance = calibrationPower / (double)texelCount;

    if (selection.totalPower > 0.0)
    {
        for (size_t i = 0; i < texelCount; ++i)
        {
            const LightSelectionEntry& entry = selection.entries[i];
            const int y = static_cast<int>(i / static_cast<size_t>(width));
            const double theta0 = M_PI * static_cast<double>(y) / static_cast<double>(height);
            const double theta1 = M_PI * static_cast<double>(y + 1) / static_cast<double>(height);
            const double solidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
            out.alias[i].prob = entry.aliasProbability;
            out.alias[i].alias = entry.alias;
            out.alias[i].solidAnglePdf = static_cast<float>(static_cast<double>(entry.pdf) / solidAngle);
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

    if (selection.totalPower > 0.0)
    {
        out.envPdfScale = static_cast<float>(1.0 / selection.totalPower);
    }
    return out;
}

} // namespace oka::metal
