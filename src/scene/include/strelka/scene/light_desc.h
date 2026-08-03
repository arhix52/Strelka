#pragma once

#include <light_types.h>

#include <glm/glm.hpp>

#include <cmath>
#include <string>

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
    if (name == "point")
        return LIGHT_TYPE_POINT;
    if (name == "spot")
        return LIGHT_TYPE_SPOT;
    return LIGHT_TYPE_RECT;
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

/// Solid angle of a cone with the given half-angle (radians).
///
/// 4pi sin^2(x/2) rather than 2pi (1 - cos x). They agree exactly in real
/// arithmetic and not at all in floats at the angles that matter: the sun is
/// 0.0046 rad, where 1 - cos loses three digits to cancellation, and anything
/// narrower rounds to zero. The shader computes the sampling pdf from the same
/// quantity, so a discrepancy here does not cancel against the baked radiance --
/// it is a multiplier on the light, and it was 73x.
inline float coneSolidAngle(float halfAngleRad)
{
    const float s = std::sin(0.5f * halfAngleRad);
    return 4.0f * float(M_PI) * s * s;
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
        return float(M_PI) * radius * radius;
    case LIGHT_TYPE_SPHERE:
        return 4.0f * float(M_PI) * radius * radius;
    default:
        return 0.0f;
    }
}

/// Convert the authored intensity into the quantity the shader expects in
/// UniformLight::color:
///   area / distant → radiance (W/sr/m²)
///   point / spot    → radiant intensity (W/sr), divided by r² in the shader
inline glm::float3 bakeLightRadiometric(int type,
                                        int unit,
                                        const glm::float3& color,
                                        float intensity,
                                        float width,
                                        float height,
                                        float radius,
                                        float halfAngleRad,
                                        float outerConeAngleRad)
{
    const glm::float3 tint = color * std::max(intensity, 0.0f);
    if (intensity <= 0.0f)
    {
        return glm::float3(0.0f);
    }

    switch (unit)
    {
    case LIGHT_UNIT_POWER:
    {
        // Φ (W). Lambertian area: L = Φ / (π A). Point: I = Φ / 4π.
        // Spot: I = Φ / Ω_outer so the integral over the cone recovers Φ.
        if (type == LIGHT_TYPE_POINT)
        {
            return tint / (4.0f * float(M_PI));
        }
        if (type == LIGHT_TYPE_SPOT)
        {
            const float omega = std::max(coneSolidAngle(outerConeAngleRad), 1e-8f);
            return tint / omega;
        }
        if (type == LIGHT_TYPE_DISTANT)
        {
            // Treat power as irradiance for a distant light — there is no area.
            const float omega = std::max(coneSolidAngle(halfAngleRad), 1e-8f);
            return tint / omega;
        }
        const float area = std::max(lightSurfaceArea(type, width, height, radius), 1e-8f);
        return tint / (float(M_PI) * area);
    }
    case LIGHT_UNIT_INTENSITY:
        // Candela. Meaningful for point/spot; for anything else fall through to
        // radiance so a mis-tagged area light still lights something.
        if (type == LIGHT_TYPE_POINT || type == LIGHT_TYPE_SPOT)
        {
            return tint;
        }
        return tint;
    case LIGHT_UNIT_IRRADIANCE:
    {
        // E (W/m²). Distant: L = E / Ω.
        const float omega = std::max(coneSolidAngle(std::max(halfAngleRad, 1e-6f)), 1e-8f);
        return tint / omega;
    }
    case LIGHT_UNIT_RADIANCE:
    default:
        return tint;
    }
}

} // namespace oka
