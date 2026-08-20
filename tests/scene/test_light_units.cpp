#include <doctest/doctest.h>

// glm_wrapper.hpp is what defines the glm::floatN aliases that light_desc.h uses in its signatures.
#include <strelka/scene/glm_wrapper.hpp>
#include <strelka/scene/light_desc.h>
#include <light_types.h>

#include <cmath>
#include <numbers>

using namespace oka;

// Photometric -> radiometric conversion factor used by the glTF loader
// (KHR_lights_punctual intensity is candela / lux, everything downstream of
// UniformLightDesc is watts). Mirrored here on purpose: gltfloader.cpp keeps it
// file-static, so this is the contract the test pins, not the symbol.
static constexpr float kLumensPerWatt = 683.0f;

static constexpr float kPi = std::numbers::pi_v<float>;

static constexpr glm::float3 kWhite(1.0f, 1.0f, 1.0f);

// bakeLightRadiometric takes nine positional arguments and most tests only care
// about two or three of them; these wrappers keep the intent readable.
namespace
{
glm::float3 bakePoint(int unit, float intensity, const glm::float3& color = kWhite)
{
    return bakeLightRadiometric(LIGHT_TYPE_POINT, unit, color, intensity, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

glm::float3 bakeSpot(int unit, float intensity, float outerConeAngleRad, const glm::float3& color = kWhite)
{
    return bakeLightRadiometric(
        LIGHT_TYPE_SPOT, unit, color, intensity, 0.0f, 0.0f, 0.0f, 0.0f, outerConeAngleRad);
}

glm::float3 bakeProjector(int unit, float intensity, float halfFovX, float aspect, const glm::float3& color = kWhite)
{
    return bakeLightRadiometric(LIGHT_TYPE_PROJECTOR, unit, color, intensity, 0.0f, 0.0f, 0.0f, 0.0f, halfFovX, aspect);
}

glm::float3 bakeRect(int unit, float intensity, float width, float height, const glm::float3& color = kWhite)
{
    return bakeLightRadiometric(LIGHT_TYPE_RECT, unit, color, intensity, width, height, 0.0f, 0.0f, 0.0f);
}

glm::float3 bakeDistant(int unit, float intensity, float halfAngleRad, const glm::float3& color = kWhite)
{
    return bakeLightRadiometric(LIGHT_TYPE_DISTANT, unit, color, intensity, 0.0f, 0.0f, 0.0f, halfAngleRad, 0.0f);
}
} // namespace

// --- LIGHT_UNIT_POWER: watts in, the quantity the shader wants out ---

TEST_CASE("power on a point light is spread over the full sphere")
{
    // I = Phi / 4pi.
    const glm::float3 baked = bakePoint(LIGHT_UNIT_POWER, 1000.0f);
    const float expected = 1000.0f / (4.0f * kPi);
    CHECK(baked.x == doctest::Approx(expected).epsilon(1e-5));
    CHECK(baked.y == doctest::Approx(expected).epsilon(1e-5));
    CHECK(baked.z == doctest::Approx(expected).epsilon(1e-5));

    // Integrating the (isotropic) intensity over the sphere gives the watts back.
    CHECK(baked.x * 4.0f * kPi == doctest::Approx(1000.0f).epsilon(1e-4));
}

TEST_CASE("power on a point light scales linearly and keeps the colour tint")
{
    const glm::float3 tinted = bakePoint(LIGHT_UNIT_POWER, 100.0f, glm::float3(1.0f, 0.5f, 0.25f));
    const float expected = 100.0f / (4.0f * kPi);
    CHECK(tinted.x == doctest::Approx(expected).epsilon(1e-5));
    CHECK(tinted.y == doctest::Approx(expected * 0.5f).epsilon(1e-5));
    CHECK(tinted.z == doctest::Approx(expected * 0.25f).epsilon(1e-5));

    CHECK(bakePoint(LIGHT_UNIT_POWER, 200.0f).x == doctest::Approx(2.0f * bakePoint(LIGHT_UNIT_POWER, 100.0f).x));
}

TEST_CASE("power on a rect light becomes Lambertian radiance L = Phi / (pi A)")
{
    // 2 x 3 m emitter, 600 W.
    const glm::float3 baked = bakeRect(LIGHT_UNIT_POWER, 600.0f, 2.0f, 3.0f);
    const float area = 6.0f;
    const float expected = 600.0f / (kPi * area);
    CHECK(baked.x == doctest::Approx(expected).epsilon(1e-5));

    // Doubling the area at fixed power halves the radiance.
    const glm::float3 twiceTheArea = bakeRect(LIGHT_UNIT_POWER, 600.0f, 4.0f, 3.0f);
    CHECK(twiceTheArea.x == doctest::Approx(0.5f * baked.x).epsilon(1e-5));

    // A unit-area white rect fed pi watts is exactly radiance 1 -- the sanity
    // value used when eyeballing EXR output.
    CHECK(bakeRect(LIGHT_UNIT_POWER, kPi, 1.0f, 1.0f).x == doctest::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("power on a spot light integrates back to the input power over its cone")
{
    const float watts = 750.0f;
    for (float outer : { 0.1f, kPi / 8.0f, kPi / 6.0f, kPi / 4.0f, kPi / 3.0f, kPi / 2.0f })
    {
        CAPTURE(outer);
        const glm::float3 baked = bakeSpot(LIGHT_UNIT_POWER, watts, outer);
        const float omega = coneSolidAngle(outer);
        // I is constant inside the cone, so the integral is just I * Omega.
        CHECK(baked.x * omega == doctest::Approx(watts).epsilon(1e-4));
    }
}

TEST_CASE("power on a projector integrates back over its rectangular frame")
{
    // I = Phi / Omega, with Omega the pyramid the image fills. Six 90-degree
    // square pyramids tile the sphere, so this one takes a sixth of the watts of
    // the equivalent point lamp -- and multiplying the intensity back by the
    // solid angle has to return the watts that went in.
    const float halfFov = 0.25f * kPi;
    const float aspect = 1.0f;
    const glm::float3 baked = bakeProjector(LIGHT_UNIT_POWER, 60.0f, halfFov, aspect);
    const float omega = projectorSolidAngleFromFov(halfFov, aspect);
    CHECK(baked.x * omega == doctest::Approx(60.0f).epsilon(1e-4));
    CHECK(omega == doctest::Approx(4.0f * kPi / 6.0f).epsilon(1e-5));
}

TEST_CASE("a wider projector frame spreads the same watts thinner")
{
    // Not the cone the same angle would give: a 16:9 frame is the narrower of
    // the two, so the same watts land brighter in it than in a square one.
    const float halfFov = 0.5f;
    const glm::float3 square = bakeProjector(LIGHT_UNIT_POWER, 100.0f, halfFov, 1.0f);
    const glm::float3 wide = bakeProjector(LIGHT_UNIT_POWER, 100.0f, halfFov, 16.0f / 9.0f);
    CHECK(wide.x > square.x);
    CHECK(wide.x / square.x ==
          doctest::Approx(projectorSolidAngleFromFov(halfFov, 1.0f) / projectorSolidAngleFromFov(halfFov, 16.0f / 9.0f))
              .epsilon(1e-4));
}

TEST_CASE("intensity on a projector is candela straight through, like a spot")
{
    // The image multiplies this, so the number the user types is the intensity
    // along a fully lit texel and not an average over the frame.
    CHECK(bakeProjector(LIGHT_UNIT_INTENSITY, 250.0f, 0.4f, 16.0f / 9.0f).x == doctest::Approx(250.0f));
}

TEST_CASE("a spot opened to the full sphere matches the point light conversion")
{
    // Omega(pi) = 4pi, so a hemisphere-and-then-some spot degenerates to I = Phi/4pi.
    const glm::float3 spot = bakeSpot(LIGHT_UNIT_POWER, 1000.0f, kPi);
    const glm::float3 point = bakePoint(LIGHT_UNIT_POWER, 1000.0f);
    CHECK(spot.x == doctest::Approx(point.x).epsilon(1e-4));
}

TEST_CASE("a narrower spot cone concentrates the same watts into more intensity")
{
    const glm::float3 wide = bakeSpot(LIGHT_UNIT_POWER, 100.0f, kPi / 4.0f);
    const glm::float3 narrow = bakeSpot(LIGHT_UNIT_POWER, 100.0f, kPi / 16.0f);
    CHECK(narrow.x > wide.x);
    CHECK(narrow.x * coneSolidAngle(kPi / 16.0f) == doctest::Approx(wide.x * coneSolidAngle(kPi / 4.0f)).epsilon(1e-4));
}

TEST_CASE("a degenerate spot cone does not produce infinity")
{
    // Omega is floored at 1e-8 rather than allowed to hit zero.
    const glm::float3 baked = bakeSpot(LIGHT_UNIT_POWER, 1.0f, 0.0f);
    CHECK(std::isfinite(baked.x));
    CHECK(baked.x > 0.0f);
}

// --- LIGHT_UNIT_INTENSITY: candela straight through for the delta lights ---

TEST_CASE("intensity passes candela through unchanged for point and spot")
{
    CHECK(bakePoint(LIGHT_UNIT_INTENSITY, 40.0f).x == doctest::Approx(40.0f));
    CHECK(bakeSpot(LIGHT_UNIT_INTENSITY, 40.0f, kPi / 4.0f).x == doctest::Approx(40.0f));

    // ...and the cone angle must not enter into it: candela is already per-steradian.
    CHECK(bakeSpot(LIGHT_UNIT_INTENSITY, 40.0f, kPi / 32.0f).x ==
          doctest::Approx(bakeSpot(LIGHT_UNIT_INTENSITY, 40.0f, kPi / 2.0f).x));

    const glm::float3 tinted = bakePoint(LIGHT_UNIT_INTENSITY, 40.0f, glm::float3(1.0f, 0.5f, 0.25f));
    CHECK(tinted.x == doctest::Approx(40.0f));
    CHECK(tinted.y == doctest::Approx(20.0f));
    CHECK(tinted.z == doctest::Approx(10.0f));
}

TEST_CASE("intensity on a mis-tagged area light degrades to radiance, not to zero")
{
    // Documented fallback: a rect authored in candela still lights the scene
    // instead of silently vanishing or being divided by an area it does not use.
    const glm::float3 baked = bakeRect(LIGHT_UNIT_INTENSITY, 7.0f, 2.0f, 3.0f);
    CHECK(baked.x == doctest::Approx(7.0f));
}

// --- LIGHT_UNIT_IRRADIANCE: lux on a distant light ---

TEST_CASE("irradiance on a distant light divides by the cone solid angle")
{
    const float lux = 10.0f;
    for (float halfAngle : { 0.1f, 0.05f, 0.01f })
    {
        CAPTURE(halfAngle);
        const glm::float3 baked = bakeDistant(LIGHT_UNIT_IRRADIANCE, lux, halfAngle);
        const float omega = coneSolidAngle(halfAngle);
        CHECK(baked.x == doctest::Approx(lux / omega).epsilon(1e-4));
        // Round trip: L * Omega is the irradiance arriving on a facing surface.
        CHECK(baked.x * omega == doctest::Approx(lux).epsilon(1e-4));
    }
}

TEST_CASE("a smaller sun disc concentrates the same irradiance into more radiance")
{
    const glm::float3 wide = bakeDistant(LIGHT_UNIT_IRRADIANCE, 10.0f, 0.1f);
    const glm::float3 tight = bakeDistant(LIGHT_UNIT_IRRADIANCE, 10.0f, 0.01f);
    CHECK(tight.x > wide.x);
    // Omega ~ pi*theta^2 for small angles, so a 10x smaller disc is ~100x brighter.
    CHECK(tight.x / wide.x == doctest::Approx(100.0f).epsilon(0.02));
}

TEST_CASE("a zero-width sun disc does not produce infinity")
{
    // halfAngle is floored at 1e-6 before the solid angle is taken.
    const glm::float3 baked = bakeDistant(LIGHT_UNIT_IRRADIANCE, 10.0f, 0.0f);
    CHECK(std::isfinite(baked.x));
    CHECK(baked.x > 0.0f);
}

TEST_CASE("power on a distant light is treated as irradiance")
{
    // A distant light has no area, so watts and lux take the same path.
    const glm::float3 asPower = bakeDistant(LIGHT_UNIT_POWER, 10.0f, 0.05f);
    const glm::float3 asIrradiance = bakeDistant(LIGHT_UNIT_IRRADIANCE, 10.0f, 0.05f);
    CHECK(asPower.x == doctest::Approx(asIrradiance.x).epsilon(1e-5));
}

// --- non-positive intensity ---

TEST_CASE("zero or negative intensity bakes to exactly zero for every type and unit")
{
    const int types[] = { LIGHT_TYPE_RECT,    LIGHT_TYPE_DISC,  LIGHT_TYPE_SPHERE, LIGHT_TYPE_DISTANT,
                          LIGHT_TYPE_DOME,    LIGHT_TYPE_POINT, LIGHT_TYPE_SPOT };
    const int units[] = { LIGHT_UNIT_RADIANCE, LIGHT_UNIT_POWER, LIGHT_UNIT_INTENSITY, LIGHT_UNIT_IRRADIANCE };
    const float intensities[] = { 0.0f, -0.0f, -1.0f, -1e9f };

    for (int type : types)
    {
        for (int unit : units)
        {
            for (float intensity : intensities)
            {
                CAPTURE(type);
                CAPTURE(unit);
                CAPTURE(intensity);
                const glm::float3 baked = bakeLightRadiometric(
                    type, unit, glm::float3(1.0f, 2.0f, 3.0f), intensity, 2.0f, 3.0f, 0.5f, 0.1f, kPi / 4.0f);
                // Exactly zero, not "small": the renderer skips lights on == 0.
                CHECK(baked.x == 0.0f);
                CHECK(baked.y == 0.0f);
                CHECK(baked.z == 0.0f);
            }
        }
    }
}

// --- the field regression: Blender -> glTF -> Strelka ---

TEST_CASE("a 500 W Blender point lamp round-trips through 27175.7 cd")
{
    // Blender exports watts * 683 / (4pi) as candela; the loader divides the
    // photometric value by 683 and hands the result to the INTENSITY path.
    const float exportedCandela = 27175.7f;
    const glm::float3 fromGltf = bakePoint(LIGHT_UNIT_INTENSITY, exportedCandela / kLumensPerWatt);

    const float expected = 500.0f / (4.0f * kPi); // 39.789 W/sr
    CHECK(expected == doctest::Approx(39.789f).epsilon(1e-4));
    // The tolerance the field bug needs: 683x off would be ~27000 away.
    CHECK(std::abs(fromGltf.x - expected) < 0.01f);

    // Authoring the same lamp in watts directly must land on the same intensity.
    const glm::float3 fromWatts = bakePoint(LIGHT_UNIT_POWER, 500.0f);
    CHECK(std::abs(fromGltf.x - fromWatts.x) < 0.01f);
}

TEST_CASE("forgetting the 683 lm/W divide makes the lamp 683x too bright")
{
    // This is the bug that shipped: the photometric intensity went in raw.
    const float exportedCandela = 27175.7f;
    const glm::float3 correct = bakePoint(LIGHT_UNIT_INTENSITY, exportedCandela / kLumensPerWatt);
    const glm::float3 buggy = bakePoint(LIGHT_UNIT_INTENSITY, exportedCandela);
    CHECK(buggy.x / correct.x == doctest::Approx(kLumensPerWatt).epsilon(1e-4));
}

TEST_CASE("a 1000 lux Blender sun round-trips through 683000 lx")
{
    const float halfAngle = 0.53f * 0.5f * (kPi / 180.0f); // the loader's sun disc
    const float exportedLux = 1000.0f * kLumensPerWatt;
    const glm::float3 baked = bakeDistant(LIGHT_UNIT_IRRADIANCE, exportedLux / kLumensPerWatt, halfAngle);
    CHECK(baked.x * coneSolidAngle(halfAngle) == doctest::Approx(1000.0f).epsilon(1e-3));
}

// --- the two photometric constants, and why they are two ---

TEST_CASE("the renderer carries two lm/W figures, for two different jobs")
{
    // This looks like an inconsistency and is regularly reported as one, so it
    // is pinned here with the reason.
    //
    //   kLuminousEfficacyD65 = 177.83 converts a *measurement*. An IES file
    //   holds candela produced by a real luminaire with an unknown spectrum, and
    //   turning that into watts needs an assumed illuminant. Cycles assumes D65
    //   for the same conversion, which is what lets the 27_ies ladder row
    //   compare two angular distributions instead of two guesses at a scale.
    //
    //   683 lm/W, in gltfloader.cpp, undoes a *bookkeeping step*. Blender's glTF
    //   exporter writes candela as watts * 683 / (4 pi), so recovering the watts
    //   the artist typed means dividing by 683 -- the same 683, whatever the
    //   lamp's spectrum is or is not.
    //
    // Collapsing them into one number necessarily breaks agreement with one
    // reference or the other: with Cycles on IES profiles, or with Blender on a
    // round-tripped lamp. The round trip is pinned above; this pins the pair.
    CHECK(oka::kLuminousEfficacyD65 == doctest::Approx(177.83f));
    CHECK(oka::kCandelaToRadiantIntensity == doctest::Approx(1.0f / 177.83f));
    CHECK(kLumensPerWatt == doctest::Approx(683.0f));

    // A luminaire measured at 1000 cd and a Blender lamp exported at 1000 cd are
    // therefore not the same light, and differ by this much. Anyone mixing the
    // two in one scene is looking at a 3.84x imbalance that is inherited, not
    // introduced.
    CHECK(kLumensPerWatt / oka::kLuminousEfficacyD65 == doctest::Approx(3.841f).epsilon(1e-3));
}

// --- helpers the conversions are built on ---

TEST_CASE("coneSolidAngle spans the sphere correctly")
{
    CHECK(coneSolidAngle(0.0f) == doctest::Approx(0.0f));
    CHECK(coneSolidAngle(kPi / 2.0f) == doctest::Approx(2.0f * kPi).epsilon(1e-5)); // hemisphere
    CHECK(coneSolidAngle(kPi) == doctest::Approx(4.0f * kPi).epsilon(1e-5)); // full sphere
    // Small-angle limit: Omega -> pi * theta^2.
    CHECK(coneSolidAngle(0.001f) == doctest::Approx(kPi * 1e-6f).epsilon(1e-3));
}

TEST_CASE("lightSurfaceArea matches the emitter geometry and is zero for punctual lights")
{
    CHECK(lightSurfaceArea(LIGHT_TYPE_RECT, 2.0f, 3.0f, 0.0f) == doctest::Approx(6.0f));
    CHECK(lightSurfaceArea(LIGHT_TYPE_DISC, 0.0f, 0.0f, 2.0f) == doctest::Approx(kPi * 4.0f).epsilon(1e-5));
    CHECK(lightSurfaceArea(LIGHT_TYPE_SPHERE, 0.0f, 0.0f, 2.0f) == doctest::Approx(4.0f * kPi * 4.0f).epsilon(1e-5));
    CHECK(lightSurfaceArea(LIGHT_TYPE_POINT, 1.0f, 1.0f, 1.0f) == 0.0f);
    CHECK(lightSurfaceArea(LIGHT_TYPE_SPOT, 1.0f, 1.0f, 1.0f) == 0.0f);
    CHECK(lightSurfaceArea(LIGHT_TYPE_DISTANT, 1.0f, 1.0f, 1.0f) == 0.0f);
    // Negative extents must not produce negative area (which would flip the sign
    // of the baked radiance).
    CHECK(lightSurfaceArea(LIGHT_TYPE_RECT, -2.0f, 3.0f, 0.0f) == 0.0f);
}

TEST_CASE("a zero-area rect does not produce infinity")
{
    const glm::float3 baked = bakeRect(LIGHT_UNIT_POWER, 1.0f, 0.0f, 0.0f);
    CHECK(std::isfinite(baked.x));
}

TEST_CASE("unit names round-trip through the sidecar spellings")
{
    const int units[] = { LIGHT_UNIT_RADIANCE, LIGHT_UNIT_POWER, LIGHT_UNIT_INTENSITY, LIGHT_UNIT_IRRADIANCE };
    for (int unit : units)
    {
        CAPTURE(unit);
        CHECK(lightUnitFromName(lightUnitName(unit)) == unit);
    }
    // The aliases Blender and glTF users actually type.
    CHECK(lightUnitFromName("watt") == LIGHT_UNIT_POWER);
    CHECK(lightUnitFromName("W") == LIGHT_UNIT_POWER);
    CHECK(lightUnitFromName("candela") == LIGHT_UNIT_INTENSITY);
    CHECK(lightUnitFromName("cd") == LIGHT_UNIT_INTENSITY);
    CHECK(lightUnitFromName("lux") == LIGHT_UNIT_IRRADIANCE);
    CHECK(lightUnitFromName("lx") == LIGHT_UNIT_IRRADIANCE);
    // An unknown spelling must fall back to radiance, the legacy behaviour, so
    // old sidecar JSON without a unit field renders as it always did.
    CHECK(lightUnitFromName("lumens") == LIGHT_UNIT_RADIANCE);
    CHECK(lightUnitFromName("") == LIGHT_UNIT_RADIANCE);
}

TEST_CASE("cone solid angle survives sun-sized half-angles")
{
    // 2pi (1 - cos x) is exact in real arithmetic and worthless in floats here:
    // at the sun's 0.00459 rad, cos rounds to within 6e-8 of 1 while the true
    // 1 - cos is 1.05e-5, so three of the five significant digits are gone, and
    // a few times narrower it collapses to zero outright.
    //
    // This matters more than a precision note usually would. The host bakes a
    // distant light's radiance as irradiance / solid angle and the shader
    // divides it back out by the sampling pdf; they only cancel while both are
    // computing the same number. Once they stopped, the sun came out 73 times
    // too bright.
    struct Case { double halfAngle; };
    const Case cases[] = { { 0.5 }, { 0.05 }, { 0.00459216 }, { 1e-3 }, { 1e-4 }, { 1e-5 } };
    for (const Case& c : cases)
    {
        CAPTURE(c.halfAngle);
        // Reference in double, where the cancellation is survivable.
        const double reference = 2.0 * std::numbers::pi * (1.0 - std::cos(c.halfAngle));
        const double got = coneSolidAngle((float)c.halfAngle);
        CHECK(got == doctest::Approx(reference).epsilon(1e-4));
        CHECK(got > 0.0);
    }
}

TEST_CASE("a distant light's baked radiance times its solid angle is its irradiance")
{
    // What the renderer actually does: bake E / omega on the host, multiply by
    // omega again in the estimator. The product has to come back as E for every
    // sun anyone would author, which is the invariant that broke.
    const float irradiance = 5.0f;
    const double halfAngles[] = { 0.5, 0.05, 0.00459216, 1e-3, 1e-4 };
    for (double halfAngle : halfAngles)
    {
        CAPTURE(halfAngle);
        const glm::float3 radiance =
            bakeLightRadiometric(LIGHT_TYPE_DISTANT, LIGHT_UNIT_IRRADIANCE, glm::float3(1.0f), irradiance,
                                 0.0f, 0.0f, 0.0f, (float)halfAngle, 0.0f);
        const float recovered = radiance.x * coneSolidAngle((float)halfAngle);
        CHECK(recovered == doctest::Approx(irradiance).epsilon(1e-3));
    }
}
