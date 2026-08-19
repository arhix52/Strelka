#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace oka
{
namespace metal
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
};

// Walker/Vose alias table over equirectangular HDR texels. Weight is luminance
// times sin(theta) of the row (solid-angle Jacobian). pixelRgba is tightly packed
// RGBA float rows, width * height * 4 floats.
inline IblAliasTableResult buildIblAliasTable(const float* pixelRgba, int width, int height)
{
    IblAliasTableResult out;
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return out;
    }

    const size_t texelCount = (size_t)width * (size_t)height;
    std::vector<double> weights(texelCount);
    double totalPower = 0.0;

    for (int y = 0; y < height; ++y)
    {
        const double v = ((double)y + 0.5) / (double)height;
        const double sinTheta = std::sin(v * M_PI);
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
            const double w = clean * sinTheta;
            weights[i] = w;
            totalPower += w;
        }
    }

    out.alias.resize(texelCount);
    out.totalPower = totalPower;

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

    out.envPdfScale = (totalPower > 0.0) ? (float)((double)texelCount / (2.0 * M_PI * M_PI * totalPower)) : 0.0f;
    return out;
}

} // namespace metal
} // namespace oka
