#pragma once

#include <strelka/scene/light_desc.h>
#include <strelka/scene/scene.h>
#include <analytic_light.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace oka::metal
{

struct LightSelectionEntry
{
    float aliasProbability = 0.0f;
    uint32_t alias = std::numeric_limits<uint32_t>::max();
    float pdf = 0.0f;
};

struct LightSelectionTable
{
    std::vector<LightSelectionEntry> entries;
    double totalPower = 0.0;
};

inline double cleanLightPower(double power)
{
    return std::isfinite(power) ? std::max(power, 0.0) : 0.0;
}

inline float binaryPowerProbability(double firstPower, double secondPower)
{
    const double first = cleanLightPower(firstPower);
    const double second = cleanLightPower(secondPower);
    if (!(first > 0.0))
    {
        return 0.0f;
    }
    if (!(second > 0.0))
    {
        return 1.0f;
    }
    const double scale = std::max(first, second);
    const double probability = (first / scale) / (first / scale + second / scale);
    constexpr float minimumProbability = 0x1p-22f;
    return std::clamp(static_cast<float>(probability), minimumProbability, 1.0f - minimumProbability);
}

// Scene-wide emitted-power proxy. It need not know the shading point: RIS still
// makes the point-dependent choice. This proposal only stops spending equal
// probability on lights whose total output differs by orders of magnitude.
inline double analyticLightPower(const Scene::Light& light)
{
    const double luminance = cleanLightPower(0.2126 * light.color.r + 0.7152 * light.color.g + 0.0722 * light.color.b);
    constexpr double pi = std::numbers::pi_v<double>;
    double measure = 0.0;
    switch (light.type)
    {
    case LIGHT_TYPE_RECT: {
        const glm::float3 e1 = glm::float3(light.points[1] - light.points[0]);
        const glm::float3 e2 = glm::float3(light.points[3] - light.points[0]);
        measure = pi * glm::length(glm::cross(e1, e2));
        break;
    }
    case LIGHT_TYPE_DISC:
        measure = pi * pi * glm::length(glm::cross(glm::dvec3(light.points[2]), glm::dvec3(light.points[3])));
        break;
    case LIGHT_TYPE_SPHERE: {
        // Same deterministic equal-solid-angle quadrature as
        // analyticEllipsoidSurfaceArea(), evaluated in host double precision.
        // Keeping this in GLM avoids making the host proposal depend on whether
        // material_math.h names float3 as GLM (Metal/CPU) or CUDA (OptiX).
        constexpr size_t sampleCount = 256u;
        constexpr double goldenAngle = 2.39996322972865332;
        const glm::dvec3 axisX(light.points[0]);
        const glm::dvec3 axisY(light.points[2]);
        const glm::dvec3 axisZ(light.points[3]);
        double jacobianSum = 0.0;
        for (size_t i = 0u; i < sampleCount; ++i)
        {
            const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(sampleCount);
            const double radial = std::sqrt(std::max(1.0 - z * z, 0.0));
            const double phi = goldenAngle * static_cast<double>(i);
            const glm::dvec3 n(radial * std::cos(phi), radial * std::sin(phi), z);
            const glm::dvec3 cofactor =
                n.x * glm::cross(axisY, axisZ) + n.y * glm::cross(axisZ, axisX) + n.z * glm::cross(axisX, axisY);
            jacobianSum += glm::length(cofactor);
        }
        const double area = 4.0 * pi * jacobianSum / static_cast<double>(sampleCount);
        measure = pi * area;
        break;
    }
    case LIGHT_TYPE_POINT:
        measure = 4.0 * pi;
        break;
    case LIGHT_TYPE_SPOT:
        measure = coneSolidAngle(light.halfAngle);
        break;
    case LIGHT_TYPE_PROJECTOR:
        measure = projectorSolidAngleFromFov(light.halfAngle, light.points[0].w);
        break;
    case LIGHT_TYPE_DISTANT:
        measure = coneSolidAngle(light.halfAngle);
        break;
    case LIGHT_TYPE_DOME:
        measure = 4.0 * pi;
        break;
    default:
        break;
    }
    return cleanLightPower(luminance * measure);
}

inline LightSelectionTable buildLightSelectionAlias(const std::vector<double>& powers)
{
    LightSelectionTable table;
    table.entries.resize(powers.size());
    std::vector<double> cleanPowers(powers.size());
    for (size_t i = 0; i < powers.size(); ++i)
    {
        cleanPowers[i] = cleanLightPower(powers[i]);
        table.totalPower += cleanPowers[i];
    }
    if (powers.empty() || !(table.totalPower > 0.0))
    {
        return table;
    }

    const size_t count = powers.size();
    // A positive input must remain a positive float PMF even after division by
    // N in the alias representation. This floor is many orders below any
    // meaningful scene-light probability, changes only the proposal, and the
    // represented (post-rounding) PMF below is what the estimator actually uses.
    // The shipped samplers expose at least 23 useful fractional bits. Keep a
    // non-empty alias branch wider than one such step so strict `u < q` can
    // actually take it, rather than preserving a merely symbolic float mass
    // that no generated variate can reach.
    constexpr double minimumAliasProbability = 0x1p-22;
    const double minimumProbability =
        std::max(static_cast<double>(std::numeric_limits<float>::min()) * static_cast<double>(count),
                 minimumAliasProbability / static_cast<double>(count));
    std::vector<double> probabilities(count, 0.0);
    double adjustedTotal = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        if (cleanPowers[i] > 0.0)
        {
            probabilities[i] = std::max(cleanPowers[i] / table.totalPower, minimumProbability);
            adjustedTotal += probabilities[i];
        }
    }
    for (double& probability : probabilities)
    {
        probability /= adjustedTotal;
    }

    std::vector<double> scaled(count);
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(count / 2);
    large.reserve(count / 2);
    uint32_t fallback = 0u;
    for (size_t i = 0; i < count; ++i)
    {
        if (probabilities[i] > probabilities[fallback])
        {
            fallback = static_cast<uint32_t>(i);
        }
        scaled[i] = probabilities[i] * static_cast<double>(count);
        (scaled[i] < 1.0 ? small : large).push_back(static_cast<uint32_t>(i));
    }

    while (!small.empty() && !large.empty())
    {
        const uint32_t little = small.back();
        small.pop_back();
        const uint32_t great = large.back();
        large.pop_back();
        table.entries[little].aliasProbability = static_cast<float>(std::clamp(scaled[little], 0.0, 1.0));
        table.entries[little].alias = great;
        scaled[great] = (scaled[great] + scaled[little]) - 1.0;
        (scaled[great] < 1.0 ? small : large).push_back(great);
    }
    for (const uint32_t i : large)
    {
        table.entries[i].aliasProbability = 1.0f;
        table.entries[i].alias = i;
    }
    // Round-off can leave a nominally-small bucket after the last large one.
    // Keep its own remaining mass and send the rest to a known-positive target;
    // never turn a zero-weight bucket into a self alias.
    for (const uint32_t i : small)
    {
        if (i == fallback)
        {
            table.entries[i].aliasProbability = 1.0f;
            table.entries[i].alias = i;
        }
        else
        {
            table.entries[i].aliasProbability = static_cast<float>(std::clamp(scaled[i], 0.0, 1.0));
            table.entries[i].alias = fallback;
        }
    }

    // The float thresholds are the GPU distribution. Reconstruct its marginal
    // PMF rather than uploading the unrounded target and silently disagreeing
    // with it in MIS.
    std::vector<double> represented(count, 0.0);
    const double bucketMass = 1.0 / static_cast<double>(count);
    for (size_t bucket = 0; bucket < count; ++bucket)
    {
        const LightSelectionEntry& entry = table.entries[bucket];
        const double own = std::clamp(static_cast<double>(entry.aliasProbability), 0.0, 1.0);
        represented[bucket] += bucketMass * own;
        represented[entry.alias] += bucketMass * (1.0 - own);
    }
    for (size_t i = 0; i < count; ++i)
    {
        table.entries[i].pdf = static_cast<float>(represented[i]);
    }
    return table;
}

} // namespace oka::metal
