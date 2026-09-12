#pragma once

#include <host/light_selection.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace oka::metal
{

// Host-side mirror of ShaderTypes.h EnvAliasEntry (layout must stay identical).
struct EnvAliasEntry
{
    uint32_t threshold = 0u;
    uint32_t alias = 0;
    float solidAnglePdf = 0.0f;
};

struct IblAliasTableResult
{
    std::vector<EnvAliasEntry> alias;
    float envPdfScale = 0.0f;
    // Integral of the sanitized radiance luminance over the sphere. This is
    // distinct from the footprint-envelope normalizer used by the proposal.
    double totalPower = 0.0;
    double averageWeightedLuminance = 0.0;
};

inline float sanitizeEnvironmentChannel(float value)
{
    return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

inline void sanitizeEnvironmentPixels(float* pixelRgba, int width, int height)
{
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return;
    }
    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < texelCount; ++i)
    {
        float* pixel = pixelRgba + i * 4u;
        pixel[0] = sanitizeEnvironmentChannel(pixel[0]);
        pixel[1] = sanitizeEnvironmentChannel(pixel[1]);
        pixel[2] = sanitizeEnvironmentChannel(pixel[2]);
        if (!std::isfinite(pixel[3]))
        {
            pixel[3] = 1.0f;
        }
    }
}

namespace detail
{
struct IblMapStatistics
{
    double totalPower = 0.0;
    double averageWeightedLuminance = 0.0;
};

inline double environmentLuminance(const float* pixel)
{
    return 0.2126 * sanitizeEnvironmentChannel(pixel[0]) + 0.7152 * sanitizeEnvironmentChannel(pixel[1]) +
           0.0722 * sanitizeEnvironmentChannel(pixel[2]);
}

inline IblMapStatistics measureIblMap(const float* pixelRgba, int width, int height, std::vector<double>* luminance = nullptr)
{
    IblMapStatistics statistics;
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return statistics;
    }

    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (luminance)
    {
        luminance->resize(texelCount);
    }

    double calibrationPower = 0.0;
    const double deltaPhi = 2.0 * std::numbers::pi_v<double> / static_cast<double>(width);
    for (int y = 0; y < height; ++y)
    {
        const double theta0 = std::numbers::pi_v<double> * static_cast<double>(y) / static_cast<double>(height);
        const double theta1 = std::numbers::pi_v<double> * static_cast<double>(y + 1) / static_cast<double>(height);
        const double rowSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < width; ++x)
        {
            const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double clean = environmentLuminance(pixelRgba + i * 4u);
            if (luminance)
            {
                (*luminance)[i] = clean;
            }
            statistics.totalPower += clean * rowSolidAngle;
            calibrationPower += clean * std::sin(0.5 * (theta0 + theta1));
        }
    }
    statistics.averageWeightedLuminance = calibrationPower / static_cast<double>(texelCount);
    return statistics;
}
} // namespace detail

// Walker/Vose alias table for a piecewise-constant lat-long environment in the
// continuous solid-angle measure. Texel i receives mass
//
//     P_i = envelope_i * DeltaOmega_i / sum_j(envelope_j * DeltaOmega_j)
//
// where envelope_i covers the bilinear reconstruction footprint and
// DeltaOmega_i is the exact solid angle of its row segment. Both GPU samplers
// draw cos(theta) uniformly inside the selected row, so the directional density
// is its represented texel PMF divided by DeltaOmega_i.
inline IblAliasTableResult buildSolidAngleIblAliasTable(const float* pixelRgba, int width, int height)
{
    IblAliasTableResult out;
    if (!pixelRgba || width <= 0 || height <= 0)
    {
        return out;
    }

    const size_t texelCount = (size_t)width * (size_t)height;
    std::vector<double> luminance;
    std::vector<double> weights(texelCount);
    const detail::IblMapStatistics statistics = detail::measureIblMap(pixelRgba, width, height, &luminance);
    const double deltaPhi = 2.0 * std::numbers::pi_v<double> / (double)width;

    // A normalized linear texture lookup from inside texel bin (x,y) can use
    // x-1/x/x+1 and y-1/y/y+1 because texel centres are half a pixel from the
    // bin edges. Use the maximum luminance over that exact footprint as a
    // piecewise-constant proposal envelope. It leaves a constant map uniform
    // and, unlike centre-only weights, cannot assign PDF zero to positive
    // bilinear radiance.
    for (int y = 0; y < height; ++y)
    {
        const double theta0 = std::numbers::pi_v<double> * (double)y / (double)height;
        const double theta1 = std::numbers::pi_v<double> * (double)(y + 1) / (double)height;
        const double rowSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < width; ++x)
        {
            double envelope = 0.0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                const int sy = std::clamp(y + dy, 0, height - 1);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int sx = (x + dx + width) % width;
                    envelope = std::max(envelope, luminance[(size_t)sy * (size_t)width + (size_t)sx]);
                }
            }
            weights[(size_t)y * (size_t)width + (size_t)x] = envelope * rowSolidAngle;
        }
    }

    const LightSelectionTable selection = buildLightSelectionAlias(weights);
    out.alias.resize(texelCount);
    out.totalPower = statistics.totalPower;
    out.averageWeightedLuminance = statistics.averageWeightedLuminance;

    if (selection.totalPower > 0.0)
    {
        for (size_t i = 0; i < texelCount; ++i)
        {
            const LightSelectionEntry& entry = selection.entries[i];
            const int y = static_cast<int>(i / static_cast<size_t>(width));
            const double theta0 = std::numbers::pi_v<double> * static_cast<double>(y) / static_cast<double>(height);
            const double theta1 = std::numbers::pi_v<double> * static_cast<double>(y + 1) / static_cast<double>(height);
            const double solidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
            out.alias[i].threshold = entry.aliasThreshold;
            out.alias[i].alias = entry.alias;
            out.alias[i].solidAnglePdf = static_cast<float>(static_cast<double>(entry.pdf) / solidAngle);
        }
    }
    else
    {
        for (size_t i = 0; i < texelCount; ++i)
        {
            out.alias[i].threshold = 0u;
            out.alias[i].alias = (uint32_t)i;
        }
    }

    if (selection.totalPower > 0.0)
    {
        out.envPdfScale = static_cast<float>(
            std::min(1.0 / selection.totalPower, static_cast<double>(std::numeric_limits<float>::max())));
    }

    return out;
}

// Build a smaller Metal sampling table without changing the environment
// texture or its energy calibration. Each proposal texel stores the maximum
// source luminance in its footprint; buildSolidAngleIblAliasTable then expands
// that by the neighboring bilinear footprint. This keeps positive texture
// reconstruction supported while reducing the random alias-table working set.
inline IblAliasTableResult buildDownsampledSolidAngleIblAliasTable(
    const float* pixelRgba, int sourceWidth, int sourceHeight, int requestedWidth, int requestedHeight)
{
    if (!pixelRgba || sourceWidth <= 0 || sourceHeight <= 0 || requestedWidth <= 0 || requestedHeight <= 0)
    {
        return {};
    }

    const int tableWidth = std::min(sourceWidth, requestedWidth);
    const int tableHeight = std::min(sourceHeight, requestedHeight);
    if (tableWidth == sourceWidth && tableHeight == sourceHeight)
    {
        return buildSolidAngleIblAliasTable(pixelRgba, sourceWidth, sourceHeight);
    }

    std::vector<float> proxy(static_cast<size_t>(tableWidth) * static_cast<size_t>(tableHeight) * 4u, 0.0f);
    for (int y = 0; y < tableHeight; ++y)
    {
        const int sourceY0 = static_cast<int>(static_cast<int64_t>(y) * sourceHeight / tableHeight);
        const int sourceY1 = static_cast<int>(static_cast<int64_t>(y + 1) * sourceHeight / tableHeight);
        for (int x = 0; x < tableWidth; ++x)
        {
            const int sourceX0 = static_cast<int>(static_cast<int64_t>(x) * sourceWidth / tableWidth);
            const int sourceX1 = static_cast<int>(static_cast<int64_t>(x + 1) * sourceWidth / tableWidth);
            double envelope = 0.0;
            for (int sourceY = sourceY0; sourceY < sourceY1; ++sourceY)
            {
                for (int sourceX = sourceX0; sourceX < sourceX1; ++sourceX)
                {
                    const size_t sourceIndex =
                        static_cast<size_t>(sourceY) * static_cast<size_t>(sourceWidth) + static_cast<size_t>(sourceX);
                    envelope = std::max(envelope, detail::environmentLuminance(pixelRgba + sourceIndex * 4u));
                }
            }

            float* destination =
                proxy.data() + 4u * (static_cast<size_t>(y) * static_cast<size_t>(tableWidth) + static_cast<size_t>(x));
            destination[0] = destination[1] = destination[2] = static_cast<float>(envelope);
            destination[3] = 1.0f;
        }
    }

    IblAliasTableResult out = buildSolidAngleIblAliasTable(proxy.data(), tableWidth, tableHeight);
    const detail::IblMapStatistics sourceStatistics = detail::measureIblMap(pixelRgba, sourceWidth, sourceHeight);
    out.totalPower = sourceStatistics.totalPower;
    out.averageWeightedLuminance = sourceStatistics.averageWeightedLuminance;
    return out;
}

} // namespace oka::metal
