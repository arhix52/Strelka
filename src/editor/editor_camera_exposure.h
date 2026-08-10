#pragma once

#include <algorithm>
#include <cmath>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace oka
{
namespace editor_camera_exposure
{

/// Thin-lens radius in metres: focalLengthMm / (2 * fStop * 1000).
inline float lensRadiusMetres(float focalLengthMm, float fStop)
{
    if (!(fStop > 0.0f) || !(focalLengthMm > 0.0f))
    {
        return 0.0f;
    }
    return focalLengthMm / (2.0f * fStop * 1000.0f);
}

/// Vertical FOV in degrees from focal length and sensor height (both mm).
inline float verticalFovDegrees(float focalLengthMm, float sensorHeightMm)
{
    if (!(focalLengthMm > 0.0f) || !(sensorHeightMm > 0.0f))
    {
        return 0.0f;
    }
    return static_cast<float>(2.0 * std::atan(static_cast<double>(sensorHeightMm) / (2.0 * focalLengthMm)) * 180.0 / M_PI);
}

/// Same scale the Metal/OptiX tonemapper applies when filmIso > 0; when ISO is 0
/// the pipeline uses cm2Factor alone (arbitrary / multiplier mode).
inline float photographicLinearScale(float filmIso, float fStop, float shutterReciprocal, float cm2Factor)
{
    if (!(filmIso > 0.0f))
    {
        return cm2Factor;
    }
    const float denom = shutterReciprocal * fStop * fStop;
    if (!(denom > 0.0f))
    {
        return 0.0f;
    }
    return cm2Factor * filmIso / denom / 100.0f;
}

/// EV at ISO 100 for a reciprocal shutter S (meaning 1/S seconds): log2(N^2 * S / ISO).
inline float ev100(float filmIso, float fStop, float shutterReciprocal)
{
    if (!(filmIso > 0.0f) || !(fStop > 0.0f) || !(shutterReciprocal > 0.0f))
    {
        return 0.0f;
    }
    return static_cast<float>(std::log2(static_cast<double>(fStop * fStop) * shutterReciprocal / filmIso));
}

inline float radiansFromDegrees(float degrees)
{
    return degrees * static_cast<float>(M_PI / 180.0);
}

inline float degreesFromRadians(float radians)
{
    return radians * static_cast<float>(180.0 / M_PI);
}

/// Carry brightness when switching Photographic <-> Multiplier.
/// On enter Multiplier: fold the photographic scale into cm2Factor and zero ISO.
/// On enter Photographic: restore daylight defaults and back-solve cm2Factor.
inline void carryExposureAcrossModeSwitch(
    bool toMultiplier, float& filmIso, float& fStop, float& shutterReciprocal, float& cm2Factor)
{
    if (toMultiplier)
    {
        cm2Factor = photographicLinearScale(filmIso, fStop, shutterReciprocal, cm2Factor);
        filmIso = 0.0f;
        return;
    }
    filmIso = 100.0f;
    fStop = 4.0f;
    shutterReciprocal = 100.0f;
    const float target = cm2Factor; // already the linear scale from multiplier mode
    cm2Factor = target * (shutterReciprocal * fStop * fStop) * 100.0f / filmIso;
}

} // namespace editor_camera_exposure
} // namespace oka
