#include <doctest/doctest.h>

#include "editor_camera_exposure.h"

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
