#pragma once

#include <strelka/scene/light_desc.h>
#include <strelka/scene/scene.h>
#include <analytic_light.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace oka::metal
{

// Keep a small uniform component so a weak light near a surface cannot acquire
// an enormous 1/pdf weight merely because its scene-wide power is small.
inline constexpr double kLightSelectionUniformMix = 0.05;

struct LightSelectionEntry
{
    float cdf = 1.0f;
    float pdf = 1.0f;
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

inline float regularizedPowerProbability(double power, double totalPower, size_t count)
{
    if (count == 0)
    {
        return 0.0f;
    }
    const double uniform = 1.0 / static_cast<double>(count);
    const double importance = totalPower > 0.0 ? cleanLightPower(power) / totalPower : uniform;
    return static_cast<float>((1.0 - kLightSelectionUniformMix) * importance + kLightSelectionUniformMix * uniform);
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
        measure = pi * static_cast<double>(analyticDiscArea(float3(light.points[2]), float3(light.points[3])));
        break;
    case LIGHT_TYPE_SPHERE:
        measure = pi * static_cast<double>(analyticEllipsoidSurfaceArea(
                           float3(light.points[0]), float3(light.points[2]), float3(light.points[3])));
        break;
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

inline LightSelectionTable buildLightSelectionCdf(const std::vector<double>& powers)
{
    LightSelectionTable table;
    table.entries.resize(powers.size());
    for (const double power : powers)
    {
        table.totalPower += cleanLightPower(power);
    }

    double cdf = 0.0;
    for (size_t i = 0; i < powers.size(); ++i)
    {
        const float pdf = regularizedPowerProbability(powers[i], table.totalPower, powers.size());
        cdf += pdf;
        table.entries[i] = { static_cast<float>(cdf), pdf };
    }
    if (!table.entries.empty())
    {
        table.entries.back().cdf = 1.0f;
    }
    return table;
}

} // namespace oka::metal
