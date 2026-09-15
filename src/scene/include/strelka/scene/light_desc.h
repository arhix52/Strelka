#pragma once

#include <light_types.h>

// glm_wrapper, not <glm/glm.hpp>: glm::float3 is an alias defined in the wrapper.
#include <strelka/scene/glm_wrapper.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <numbers>

namespace oka
{

/// Human-readable light type for the outliner and the light JSON sidecar.
inline const char* lightTypeName(int type)
{
    switch (type)
    {
    case LIGHT_TYPE_RECT:
        return "rect";
    case LIGHT_TYPE_DISC:
        return "disc";
    case LIGHT_TYPE_SPHERE:
        return "sphere";
    case LIGHT_TYPE_DISTANT:
        return "distant";
    case LIGHT_TYPE_POINT:
        return "point";
    case LIGHT_TYPE_SPOT:
        return "spot";
    case LIGHT_TYPE_PROJECTOR:
        return "projector";
    case LIGHT_TYPE_DOME:
        return "dome";
    default:
        return "unknown";
    }
}

inline const char* lightUnitName(int unit)
{
    switch (unit)
    {
    case LIGHT_UNIT_POWER:
        return "power";
    case LIGHT_UNIT_INTENSITY:
        return "intensity";
    case LIGHT_UNIT_IRRADIANCE:
        return "irradiance";
    case LIGHT_UNIT_RADIANCE:
    default:
        return "radiance";
    }
}

inline int lightTypeFromName(const std::string& name)
{
    if (name == "disc")
        return LIGHT_TYPE_DISC;
    if (name == "sphere")
        return LIGHT_TYPE_SPHERE;
    if (name == "distant" || name == "directional" || name == "sun")
        return LIGHT_TYPE_DISTANT;
    if (name == "dome")
        return LIGHT_TYPE_DOME;
    if (name == "point")
        return LIGHT_TYPE_POINT;
    if (name == "spot")
        return LIGHT_TYPE_SPOT;
    if (name == "projector" || name == "gobo")
        return LIGHT_TYPE_PROJECTOR;
    return LIGHT_TYPE_RECT;
}

inline bool lightTypeIsPunctual(int type)
{
    return type == LIGHT_TYPE_POINT || type == LIGHT_TYPE_SPOT || type == LIGHT_TYPE_PROJECTOR;
}

inline int lightUnitFromName(const std::string& name)
{
    if (name == "power" || name == "watt" || name == "W")
        return LIGHT_UNIT_POWER;
    if (name == "intensity" || name == "candela" || name == "cd")
        return LIGHT_UNIT_INTENSITY;
    if (name == "irradiance" || name == "lux" || name == "lx")
        return LIGHT_UNIT_IRRADIANCE;
    return LIGHT_UNIT_RADIANCE;
}

inline constexpr float kLuminousEfficacyD65 = 177.83f;
inline constexpr float kCandelaToRadiantIntensity = 1.0f / kLuminousEfficacyD65;

inline float coneSolidAngle(float halfAngleRad)
{
    const float s = std::sin(0.5f * halfAngleRad);
    return 4.0f * std::numbers::pi_v<float> * s * s;
}

inline constexpr float kMinContinuousDistantHalfAngle = 2.168404344971009e-19f;

inline float distantLightHalfAngleForMeasure(float halfAngleRad)
{
    return halfAngleRad > 0.0f ? std::min(halfAngleRad, std::numbers::pi_v<float>) : 0.0f;
}

inline bool distantLightUsesDeltaMeasure(float halfAngleRad)
{
    return distantLightHalfAngleForMeasure(halfAngleRad) < kMinContinuousDistantHalfAngle;
}

inline float distantLightSolidAngle(float halfAngleRad)
{
    return distantLightUsesDeltaMeasure(halfAngleRad) ? 0.0f :
                                                        coneSolidAngle(distantLightHalfAngleForMeasure(halfAngleRad));
}

inline float projectorSolidAngleFromFov(float halfFovX, float aspect)
{
    // The same clamp projectorTanHalfX() applies: at 90 degrees the tangent is
    // infinite and the pyramid is a half space, which is not a projector.
    const float ax = std::min(std::max(halfFovX, 1e-4f), 1.55334f);
    const float tanX = std::tan(ax);
    const float tanY = tanX / std::max(aspect, 1e-4f);
    const float sinX = tanX / std::sqrt(1.0f + tanX * tanX);
    const float sinY = tanY / std::sqrt(1.0f + tanY * tanY);
    return 4.0f * std::asin(std::min(sinX * sinY, 1.0f));
}

/// Area of the light's emissive surface in world units squared. Zero for
/// punctual and distant lights.
inline float lightSurfaceArea(int type, float width, float height, float radius)
{
    switch (type)
    {
    case LIGHT_TYPE_RECT:
        return std::max(width, 0.0f) * std::max(height, 0.0f);
    case LIGHT_TYPE_DISC:
        return std::numbers::pi_v<float> * radius * radius;
    case LIGHT_TYPE_SPHERE:
        return 4.0f * std::numbers::pi_v<float> * radius * radius;
    default:
        return 0.0f;
    }
}

inline glm::float3 bakeAreaLightPower(const glm::float3& color, float power, float surfaceArea)
{
    return color * std::max(power, 0.0f) /
           (std::numbers::pi_v<float> * std::max(surfaceArea, 1e-8f));
}

inline glm::float3 bakeLightRadiometric(int type,
                                        int unit,
                                        const glm::float3& color,
                                        float intensity,
                                        float width,
                                        float height,
                                        float radius,
                                        float halfAngleRad,
                                        float outerConeAngleRad,
                                        // Projector only: the frame's width / height. Trailing and
                                        // defaulted because every other light type has no frame, and
                                        // the callers that predate the projector say nothing about one.
                                        float projectorAspect = 1.0f)
{
    const glm::float3 tint = color * std::max(intensity, 0.0f);
    if (intensity <= 0.0f)
    {
        return glm::float3(0.0f);
    }

    switch (unit)
    {
    case LIGHT_UNIT_POWER: {
        // Φ (W). Lambertian area: L = Φ / (π A). Point: I = Φ / 4π.
        // Spot: I = Φ / Ω_outer so the integral over the cone recovers Φ.
        if (type == LIGHT_TYPE_POINT)
        {
            return tint / (4.0f * std::numbers::pi_v<float>);
        }
        if (type == LIGHT_TYPE_SPOT)
        {
            const float omega = std::max(coneSolidAngle(outerConeAngleRad), 1e-8f);
            return tint / omega;
        }
        if (type == LIGHT_TYPE_PROJECTOR)
        {
            const float omega = std::max(projectorSolidAngleFromFov(outerConeAngleRad, projectorAspect), 1e-8f);
            return tint / omega;
        }
        if (type == LIGHT_TYPE_DISTANT)
        {
            // Treat power as irradiance for a distant light — there is no area.
            if (distantLightUsesDeltaMeasure(halfAngleRad))
            {
                return tint;
            }
            const float omega = distantLightSolidAngle(halfAngleRad);
            return tint / omega;
        }
        return bakeAreaLightPower(color, intensity, lightSurfaceArea(type, width, height, radius));
    }
    case LIGHT_UNIT_INTENSITY:
        (void)type;
        return tint;
    case LIGHT_UNIT_IRRADIANCE: {
        // E (W/m²). Distant: L = E / Ω.
        if (type == LIGHT_TYPE_DISTANT && distantLightUsesDeltaMeasure(halfAngleRad))
        {
            return tint;
        }
        const float omega = type == LIGHT_TYPE_DISTANT ? distantLightSolidAngle(halfAngleRad) :
                                                         std::max(coneSolidAngle(halfAngleRad), 1e-8f);
        return tint / omega;
    }
    case LIGHT_UNIT_RADIANCE:
    default:
        return tint;
    }
}

} // namespace oka
