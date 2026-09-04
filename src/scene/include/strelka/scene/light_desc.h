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

/// True when a light is a lamp at a point rather than a surface: point, spot or
/// projector.
///
/// The host mirror of lightIsPunctual() in shaders/common/light_pdf.h, which is
/// the same question asked on the device. The two are separate because that
/// header is written against whichever vector spellings the device path
/// installs, and a scene header reached from the editor, the loaders and both
/// backends' host code cannot drag those in. Anything that changes on one side
/// changes on the other.
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

/// Solid angle of a cone with the given half-angle (radians).
///
/// 4pi sin^2(x/2) rather than 2pi (1 - cos x). They agree exactly in real
/// arithmetic and not at all in floats at the angles that matter: the sun is
/// 0.0046 rad, where 1 - cos loses three digits to cancellation, and anything
/// narrower rounds to zero. The shader computes the sampling pdf from the same
/// quantity, so a discrepancy here does not cancel against the baked radiance --
/// it is a multiplier on the light, and it was 73x.
/// Luminous efficacy assumed when a photometric measurement has to become a
/// radiometric one.
///
/// 177.83 lm/W is D65's, and it is the figure Cycles assumes for exactly this
/// conversion in cycles/src/util/ies.cpp. A photometric file carries no spectrum
/// of its own, so some illuminant has to be assumed, and picking the reference's
/// is what lets the `27_ies` ladder row compare two angular distributions rather
/// than two guesses at an absolute scale.
///
/// Note that this is NOT the number the glTF loader divides by. A candela in a
/// glTF file is whatever the exporter put there, and Blender's glTF exporter
/// converts its own watts with a flat 683 lm/W -- so undoing that conversion
/// needs 683, not this. The two constants describe two different things: this
/// one is a physical assumption about an unknown spectrum, and 683 is the
/// inverse of a specific tool's bookkeeping. Collapsing them into one number
/// necessarily breaks agreement with one reference or the other -- with Cycles
/// on IES profiles, or with Blender on round-tripped lamps. See
/// tests/scene/test_light_units.cpp, which pins both.
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

/// Solid angle of the rectangular pyramid a projector throws into, from half of
/// its horizontal field of view and the frame's aspect (width / height).
///
/// Omega = 4 asin(sin a sin b), and the host mirror of projectorSolidAngle() in
/// shaders/common/projector.h: the bake below divides a projector's Watts by
/// this number and the shader spreads the image back out over exactly that
/// pyramid, so the two have to agree to the last bit. tests/render/
/// test_projector.cpp pins them against each other for that reason.
///
/// Copied rather than included for the same reason coneSolidAngle() above is a
/// copy of coneSolidAngleFromHalfAngle(). The shader header is written against
/// whichever vector spellings the device path installs, and dragging it into a
/// scene header that every target on three platforms includes would put
/// material_math.h ahead of sutil in translation units that have no reason to
/// know about either.
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

/// Convert the authored intensity into the quantity the shader expects in
/// UniformLight::color:
///   area / distant           → radiance (W/sr/m²)
///   point / spot / projector → radiant intensity (W/sr), divided by r² in the shader
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
            // Same idea as the spot, over a rectangular pyramid instead of a
            // cone: I = Phi / Omega, so that integrating the light over the frame
            // it actually fills gives back the Watts that were typed in. The
            // image on top averages whatever it averages -- a slide that is half
            // black throws half the light, which is what a real projector does
            // with the same lamp.
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
        const float area = std::max(lightSurfaceArea(type, width, height, radius), 1e-8f);
        return tint / (std::numbers::pi_v<float> * area);
    }
    case LIGHT_UNIT_INTENSITY:
        // Radiant intensity, W/sr, already. The name says candela and the
        // sidecar spells the unit "intensity", but nothing here converts:
        // whoever fills this in has done the photometry. The glTF loader divides
        // by its exporter's lm/W before handing the value over, and a sidecar
        // written by scripts/blend2strelka.py uses "power" and never reaches
        // this branch at all.
        //
        // Meaningful for point/spot; anything else falls through to the same
        // value so a mis-tagged area light still lights something.
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
