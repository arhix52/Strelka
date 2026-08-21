#include <doctest/doctest.h>

#include "editor_camera_exposure.h"

#include <vector>

using namespace oka::editor_camera_exposure;

TEST_CASE("lensRadiusMetres matches the Metal thin-lens formula")
{
    // 50 mm at f/2.8 -> 50 / (2 * 2.8 * 1000) = 0.00892857 m
    CHECK(lensRadiusMetres(50.0f, 2.8f) == doctest::Approx(0.00892857f).epsilon(1e-5));
    CHECK(lensRadiusMetres(50.0f, 0.0f) == 0.0f);
    CHECK(lensRadiusMetres(0.0f, 2.8f) == 0.0f);
}

TEST_CASE("verticalFovDegrees for a full-frame 50 mm lens")
{
    // 2 * atan(24/100) in degrees ≈ 26.991 deg
    CHECK(verticalFovDegrees(50.0f, 24.0f) == doctest::Approx(26.991f).epsilon(1e-3));
    CHECK(verticalFovDegrees(0.0f, 24.0f) == 0.0f);
}

TEST_CASE("photographicLinearScale and EV100 for daylight defaults")
{
    // ISO 100, f/4, 1/100 s, cm2=1 -> 100 / (100 * 16) / 100 = 0.000625
    CHECK(photographicLinearScale(100.0f, 4.0f, 100.0f, 1.0f) == doctest::Approx(0.000625f).epsilon(1e-6));
    CHECK(photographicLinearScale(0.0f, 4.0f, 100.0f, 2.5f) == doctest::Approx(2.5f));

    // EV100 = log2(16 * 100 / 100) = log2(16) = 4
    CHECK(ev100(100.0f, 4.0f, 100.0f) == doctest::Approx(4.0f).epsilon(1e-5));
}

TEST_CASE("carryExposureAcrossModeSwitch keeps linear brightness")
{
    float iso = 100.0f;
    float fStop = 4.0f;
    float shutter = 100.0f;
    float cm2 = 1.0f;
    const float before = photographicLinearScale(iso, fStop, shutter, cm2);

    carryExposureAcrossModeSwitch(true, iso, fStop, shutter, cm2);
    CHECK(iso == 0.0f);
    CHECK(photographicLinearScale(iso, fStop, shutter, cm2) == doctest::Approx(before).epsilon(1e-6));

    carryExposureAcrossModeSwitch(false, iso, fStop, shutter, cm2);
    CHECK(iso == 100.0f);
    CHECK(fStop == 4.0f);
    CHECK(shutter == 100.0f);
    CHECK(photographicLinearScale(iso, fStop, shutter, cm2) == doctest::Approx(before).epsilon(1e-5));
}

TEST_CASE("degrees and radians convert both ways")
{
    CHECK(degreesFromRadians(radiansFromDegrees(90.0f)) == doctest::Approx(90.0f).epsilon(1e-5));
    CHECK(radiansFromDegrees(180.0f) == doctest::Approx(3.14159265f).epsilon(1e-5));
}

namespace
{
// Builds an RGBA frame of `pixels` pixels of which `lit` carry luminance `value`.
std::vector<float> frameWith(size_t pixels, size_t lit, float value)
{
    std::vector<float> f(pixels * 4, 0.0f);
    for (size_t i = 0; i < pixels; ++i)
    {
        const float v = i < lit ? value : 0.0f;
        f[i * 4 + 0] = v;
        f[i * 4 + 1] = v;
        f[i * 4 + 2] = v;
        f[i * 4 + 3] = 1.0f;
    }
    return f;
}
} // namespace

TEST_CASE("meteredMeanLuminance ignores a frame's empty background")
{
    // The Open Chess Set as its own camera frames it: 2% subject, 98% black
    // because the scene has no environment.
    const std::vector<float> frame = frameWith(10000, 200, 0.62f);
    size_t lit = 0;
    size_t total = 0;
    const double mean = oka::editor_camera_exposure::meteredMeanLuminance(frame.data(), frame.size(), lit, total);

    CHECK(lit == 200);
    CHECK(total == 10000);
    // The subject's brightness, not the subject diluted by the void -- which
    // would read 0.0124 and buy the scene +3.8 EV it must not have.
    CHECK(mean == doctest::Approx(0.62).epsilon(1e-4));
}

TEST_CASE("meteredMeanLuminance does not move when only the framing does")
{
    // One scene, one lighting rig, two framings. A meter that disagrees here is
    // metering the camera instead of the light.
    size_t litWide = 0, totalWide = 0, litTight = 0, totalTight = 0;
    const std::vector<float> wide = frameWith(10000, 200, 0.62f);
    const std::vector<float> tight = frameWith(10000, 9000, 0.62f);

    const double meanWide =
        oka::editor_camera_exposure::meteredMeanLuminance(wide.data(), wide.size(), litWide, totalWide);
    const double meanTight =
        oka::editor_camera_exposure::meteredMeanLuminance(tight.data(), tight.size(), litTight, totalTight);

    CHECK(meanWide == doctest::Approx(meanTight).epsilon(1e-6));
}

TEST_CASE("meteredMeanLuminance keeps every pixel when there is an environment")
{
    // With an environment nothing is exactly zero, so the exclusion must not
    // fire and the result must be the plain full-frame mean.
    std::vector<float> frame = frameWith(1000, 1000, 0.5f);
    for (size_t i = 0; i < 900; ++i)
    {
        frame[i * 4 + 0] = frame[i * 4 + 1] = frame[i * 4 + 2] = 0.05f; // sky
    }
    size_t lit = 0;
    size_t total = 0;
    const double mean = oka::editor_camera_exposure::meteredMeanLuminance(frame.data(), frame.size(), lit, total);

    CHECK(lit == 1000);
    CHECK(total == 1000);
    CHECK(mean == doctest::Approx(0.9 * 0.05 + 0.1 * 0.5).epsilon(1e-4));
}

TEST_CASE("meteredMeanLuminance reports nothing metered on a black frame")
{
    const std::vector<float> frame = frameWith(64, 0, 0.0f);
    size_t lit = 0;
    size_t total = 0;
    const double mean = oka::editor_camera_exposure::meteredMeanLuminance(frame.data(), frame.size(), lit, total);

    CHECK(lit == 0);
    CHECK(total == 64);
    CHECK(mean == 0.0);

    size_t l2 = 0, t2 = 0;
    CHECK(oka::editor_camera_exposure::meteredMeanLuminance(nullptr, 400, l2, t2) == 0.0);
    CHECK(l2 == 0);
    CHECK(t2 == 0);
}
