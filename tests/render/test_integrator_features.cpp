#include <doctest/doctest.h>

#include "integrator_features.h"

using oka::metal::IntegratorFeatureInputs;
using oka::metal::packWavefrontFeatures;
using oka::metal::WavefrontFeatures;

TEST_CASE("packWavefrontFeatures empty is zero")
{
    CHECK(packWavefrontFeatures({}).bits() == 0u);
}

TEST_CASE("packWavefrontFeatures sets each independent flag")
{
    IntegratorFeatureInputs in;
    in.hasEnvMap = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kEnvMap));

    in = {};
    in.hasLights = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kLights));

    in = {};
    in.hasAlphaMaterials = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kAlpha));

    in = {};
    in.useDof = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kDof));

    in = {};
    in.debug = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kDebug));

    in = {};
    in.hasFog = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kFog));

    in = {};
    in.hasSharc = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kSharc));

    in = {};
    in.hasSubsurface = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kSubsurface));

    in = {};
    in.hasCurves = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kCurves));

    in = {};
    in.useMetal4 = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kMetal4));
}

TEST_CASE("packWavefrontFeatures motion blur needs enable and a mover")
{
    IntegratorFeatureInputs in;
    in.enableMotionBlur = true;
    CHECK_FALSE(packWavefrontFeatures(in).has(WavefrontFeatures::kMotionBlur));

    in.motionBlasBuilt = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kMotionBlur));

    in.motionBlasBuilt = false;
    in.enableCameraMotionBlur = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kMotionBlur));

    in.enableMotionBlur = false;
    CHECK_FALSE(packWavefrontFeatures(in).has(WavefrontFeatures::kMotionBlur));
}

TEST_CASE("WavefrontFeatures with() accumulates bits")
{
    const auto f = WavefrontFeatures()
                       .with(WavefrontFeatures::kEnvMap)
                       .with(WavefrontFeatures::kSharc)
                       .with(WavefrontFeatures::kSharcUpdate)
                       .with(WavefrontFeatures::kMetal4);
    CHECK(f.bits() == (WavefrontFeatures::kEnvMap | WavefrontFeatures::kSharc | WavefrontFeatures::kSharcUpdate |
                       WavefrontFeatures::kMetal4));
}
