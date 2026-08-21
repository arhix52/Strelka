#pragma once

#include <algorithm>
#include <cmath>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif


namespace oka::editor_camera_exposure
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

/// Mean luminance of an RGBA float frame, over the pixels that caught light.
///
/// Pixels that are exactly zero received no light along any path, and they are
/// not evidence about how bright the scene is -- they are evidence about how
/// much empty space the camera happens to see. A scene with no environment
/// renders its background as exact zero, so averaging the whole frame meters
/// the framing instead of the lighting: on the Open Chess Set, whose own camera
/// leaves 98% of the frame black, the full-frame mean reads 0.0127 against the
/// subject's 0.62 and asks for +3.8 EV -- which blows out the only thing in
/// shot. The same scene framed tight asks for -1.3 EV. One scene under one
/// lighting rig must not meter 35x apart because the camera moved back.
///
/// Where an environment exists no pixel is zero and this excludes nothing.
///
/// \param litPixels receives the count metered, \param totalPixels the count seen.
/// Returns 0 when nothing caught light, which the caller must treat as "do not
/// expose" rather than as a dark scene.
inline double meteredMeanLuminance(const float* rgba, size_t floatCount, size_t& litPixels, size_t& totalPixels)
{
    litPixels = 0;
    totalPixels = 0;
    if (rgba == nullptr)
    {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i + 3 < floatCount; i += 4)
    {
        // Rec.709 luma of the linear radiance, which is what the eye weights.
        const double luma = 0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2];
        ++totalPixels;
        if (luma > 0.0)
        {
            sum += luma;
            ++litPixels;
        }
    }
    return litPixels > 0 ? sum / double(litPixels) : 0.0;
}

} // namespace oka::editor_camera_exposure

