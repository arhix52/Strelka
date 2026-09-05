#pragma once

#include <strelka/scene/light_desc.h>
#include <strelka/scene/scene.h>
#include <analytic_light.h>
#include <discrete_sampling.h>
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
    uint32_t aliasThreshold = 0u;
    uint32_t alias = std::numeric_limits<uint32_t>::max();
    float pdf = 0.0f;
};

struct LightSelectionTable
{
    std::vector<LightSelectionEntry> entries;
    double totalPower = 0.0;
};

struct EmitterSelectionProbabilities
{
    float environment = 0.0f;
    float local = 0.0f;
    float meshGivenLocal = 0.0f;
    float analyticGivenLocal = 0.0f;
};

inline double cleanLightPower(double power)
{
    return std::isfinite(power) ? std::max(power, 0.0) : 0.0;
}

inline double addLightPowers(double firstPower, double secondPower)
{
    const double first = cleanLightPower(firstPower);
    const double second = cleanLightPower(secondPower);
    return first > std::numeric_limits<double>::max() - second ? std::numeric_limits<double>::max() : first + second;
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
    const float clamped = std::clamp(static_cast<float>(probability), minimumProbability, 1.0f - minimumProbability);
    const uint32_t threshold = discreteProbabilityThreshold(clamped);
    return discreteThresholdProbability(threshold);
}

inline uint32_t aliasThreshold(double probability)
{
    if (!(probability > 0.0))
    {
        return 0u;
    }
    if (!(probability < 1.0))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    constexpr double stateCount = 4294967296.0;
    const uint64_t rounded = static_cast<uint64_t>(std::llround(probability * stateCount));
    return static_cast<uint32_t>(std::clamp<uint64_t>(rounded, 1u, uint64_t{ 0xffffffffu }));
}

inline double environmentLightPower(double mapIntegral, double sceneExtent, double intensity, double tintLuminance)
{
    constexpr double pi = std::numbers::pi_v<double>;
    const double radius = std::isfinite(sceneExtent) && sceneExtent < 1e15 ? 0.5 * std::max(sceneExtent, 0.0) : 1.0;
    double result = pi;
    for (const double factor :
         { radius, radius, cleanLightPower(mapIntegral), cleanLightPower(intensity), cleanLightPower(tintLuminance) })
    {
        if (!(factor > 0.0))
        {
            return 0.0;
        }
        if (result > std::numeric_limits<double>::max() / factor)
        {
            return std::numeric_limits<double>::max();
        }
        result *= factor;
    }
    return result;
}

inline EmitterSelectionProbabilities emitterSelectionProbabilities(
    bool hasEnvironment, double environmentPower, bool hasAnalytic, double analyticPower, bool hasMesh, double meshPower)
{
    EmitterSelectionProbabilities result;
    const bool hasLocal = hasAnalytic || hasMesh;
    if (hasEnvironment)
    {
        result.environment =
            hasLocal ? binaryPowerProbability(environmentPower, addLightPowers(analyticPower, meshPower)) : 1.0f;
    }
    if (hasLocal)
    {
        result.local = hasEnvironment ? 1.0f - result.environment : 1.0f;
    }
    if (hasMesh)
    {
        result.meshGivenLocal = hasAnalytic ? binaryPowerProbability(meshPower, analyticPower) : 1.0f;
    }
    if (hasAnalytic)
    {
        result.analyticGivenLocal = hasMesh ? 1.0f - result.meshGivenLocal : 1.0f;
    }
    return result;
}

// Scene-wide emitted-power proxy. It need not know the shading point: RIS still
// makes the point-dependent choice. This proposal only stops spending equal
// probability on lights whose total output differs by orders of magnitude.
inline double analyticLightPower(const Scene::Light& light)
{
    const double luminance =
        cleanLightPower(0.2126 * std::max(light.color.r, 0.0f) + 0.7152 * std::max(light.color.g, 0.0f) +
                        0.0722 * std::max(light.color.b, 0.0f));
    constexpr double pi = std::numbers::pi_v<double>;
    const glm::dvec3 packedDirection(light.normal);
    const double packedDirectionLengthSquared = glm::dot(packedDirection, packedDirection);
    const bool hasFiniteDirection = packedDirectionLengthSquared > 0.0 && std::isfinite(packedDirectionLengthSquared);
    const bool hasFinitePosition = affineVectorIsFinite(glm::float3(light.points[1]));
    const OrthonormalLightFrame profileFrame =
        makeOrthonormalLightFrame(glm::float3(light.points[2]), glm::float3(light.points[3]), glm::float3(light.normal));
    double measure = 0.0;
    switch (light.type)
    {
    case LIGHT_TYPE_RECT: {
        const glm::float3 e1 = glm::float3(light.points[1] - light.points[0]);
        const glm::float3 e2 = glm::float3(light.points[3] - light.points[0]);
        if (glm::dot(glm::dvec3(light.normal), glm::dvec3(light.normal)) > 0.0 &&
            affineSamplePointRangeIsFinite(glm::float3(light.points[0]), e1, e2, glm::float3(0.0f)) &&
            inverseFiniteCrossLength(e1, e2) > 0.0f)
        {
            measure = pi * glm::length(glm::cross(glm::dvec3(e1), glm::dvec3(e2)));
        }
        break;
    }
    case LIGHT_TYPE_DISC:
        if (glm::dot(glm::dvec3(light.normal), glm::dvec3(light.normal)) > 0.0 &&
            affineSamplePointRangeIsFinite(glm::float3(light.points[1]), glm::float3(light.points[2]),
                                           glm::float3(light.points[3]), glm::float3(0.0f)) &&
            analyticDiscAreaPdf(glm::float3(light.points[2]), glm::float3(light.points[3])) > 0.0f)
        {
            measure = pi * pi * glm::length(glm::cross(glm::dvec3(light.points[2]), glm::dvec3(light.points[3])));
        }
        break;
    case LIGHT_TYPE_SPHERE: {
        // Same deterministic equal-solid-angle quadrature as
        // analyticEllipsoidSurfaceArea(), evaluated in host double precision.
        // Keeping this in GLM avoids making the host proposal depend on whether
        // material_math.h names float3 as GLM (Metal/CPU) or CUDA (OptiX).
        constexpr size_t sampleCount = 256u;
        constexpr double goldenAngle = 2.39996322972865332;
        const glm::float3 deviceAxisX(light.points[0]);
        const glm::float3 deviceAxisY(light.points[2]);
        const glm::float3 deviceAxisZ(light.points[3]);
        if (!analyticEllipsoidIsRepresentable(glm::float3(light.points[1]), deviceAxisX, deviceAxisY, deviceAxisZ))
        {
            break;
        }
        const glm::dvec3 axisX(deviceAxisX);
        const glm::dvec3 axisY(deviceAxisY);
        const glm::dvec3 axisZ(deviceAxisZ);
        if (!(std::abs(glm::dot(axisX, glm::cross(axisY, axisZ))) > 0.0))
        {
            break;
        }
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
        if (hasFinitePosition && (light.points[0].y < 0.0f || profileFrame.valid))
        {
            measure = 4.0 * pi;
        }
        break;
    case LIGHT_TYPE_SPOT:
        if (hasFinitePosition && hasFiniteDirection && (light.points[0].y < 0.0f || profileFrame.valid))
        {
            measure = coneSolidAngle(light.halfAngle);
        }
        break;
    case LIGHT_TYPE_PROJECTOR:
        if (hasFinitePosition && profileFrame.valid)
        {
            measure = projectorSolidAngleFromFov(light.halfAngle, light.points[0].w);
        }
        break;
    case LIGHT_TYPE_DISTANT:
        // A zero-angle distant light is a Dirac mass, not a vanishing spherical
        // cap. Its conditional continuous PDF remains zero, but the outer
        // categorical proposal still needs a finite positive variance proxy.
        if (hasFiniteDirection)
        {
            measure = distantLightUsesDeltaMeasure(light.halfAngle) ? 1.0 : distantLightSolidAngle(light.halfAngle);
        }
        break;
    case LIGHT_TYPE_DOME:
        measure = 4.0 * pi;
        break;
    default:
        break;
    }
    return cleanLightPower(luminance * measure);
}

inline uint32_t temporalLightMapping(const Scene::Light* source, const Scene::Light* destination, uint32_t stableId)
{
    if (!source || !destination)
        return std::numeric_limits<uint32_t>::max();
    if (source->type != destination->type)
        return std::numeric_limits<uint32_t>::max() - 1u;
    if (!(analyticLightPower(*destination) > 0.0))
        return std::numeric_limits<uint32_t>::max();
    return stableId;
}

inline LightSelectionTable buildLightSelectionAlias(const std::vector<double>& powers)
{
    LightSelectionTable table;
    table.entries.resize(powers.size());
    std::vector<double> cleanPowers(powers.size());
    double powerScale = 0.0;
    for (size_t i = 0; i < powers.size(); ++i)
    {
        cleanPowers[i] = cleanLightPower(powers[i]);
        powerScale = std::max(powerScale, cleanPowers[i]);
    }
    if (powers.empty() || !(powerScale > 0.0))
    {
        return table;
    }

    double scaledTotal = 0.0;
    for (const double power : cleanPowers)
    {
        scaledTotal += power / powerScale;
    }
    table.totalPower = powerScale > std::numeric_limits<double>::max() / scaledTotal ?
                           std::numeric_limits<double>::max() :
                           powerScale * scaledTotal;

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
            probabilities[i] = std::max((cleanPowers[i] / powerScale) / scaledTotal, minimumProbability);
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
        table.entries[little].aliasThreshold = aliasThreshold(std::clamp(scaled[little], 0.0, 1.0));
        table.entries[little].alias = great;
        scaled[great] = (scaled[great] + scaled[little]) - 1.0;
        (scaled[great] < 1.0 ? small : large).push_back(great);
    }
    for (const uint32_t i : large)
    {
        table.entries[i].aliasThreshold = 0u;
        table.entries[i].alias = i;
    }
    // Round-off can leave a nominally-small bucket after the last large one.
    // Keep its own remaining mass and send the rest to a known-positive target;
    // never turn a zero-weight bucket into a self alias.
    for (const uint32_t i : small)
    {
        if (i == fallback)
        {
            table.entries[i].aliasThreshold = 0u;
            table.entries[i].alias = i;
        }
        else
        {
            table.entries[i].aliasThreshold = aliasThreshold(std::clamp(scaled[i], 0.0, 1.0));
            table.entries[i].alias = fallback;
        }
    }

    // The float thresholds are the GPU distribution. Reconstruct its marginal
    // PMF rather than uploading the unrounded target and silently disagreeing
    // with it in MIS.
    std::vector<double> represented(count, 0.0);
    constexpr double integerStateCount = 4294967296.0;
    for (size_t bucket = 0; bucket < count; ++bucket)
    {
        const LightSelectionEntry& entry = table.entries[bucket];
        const double bucketMass =
            static_cast<double>(discreteBucketStateCount(static_cast<uint32_t>(count), static_cast<uint32_t>(bucket))) /
            integerStateCount;
        const double own = entry.alias == bucket ? 1.0 : static_cast<double>(entry.aliasThreshold) / integerStateCount;
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
