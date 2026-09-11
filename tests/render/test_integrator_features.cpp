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
    in.hasEmissiveMeshLights = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kEmissiveMeshLights));
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kLights));

    in = {};
    in.allAnalyticLightsRect = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kAllAnalyticLightsRect));

    in = {};
    in.uniformRectLightSampling = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kUniformRectLightSampling));

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

    in = {};
    in.restir = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kRestir));

    in = {};
    in.risOne = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kRisOne));

    in = {};
    in.writeAov = true;
    CHECK(packWavefrontFeatures(in).has(WavefrontFeatures::kAov));

    in = {};
    in.allOpenPBR = true;
    const WavefrontFeatures allOpenPBR = packWavefrontFeatures(in);
    CHECK(allOpenPBR.has(WavefrontFeatures::kAllOpenPBR));
    CHECK(allOpenPBR.has(WavefrontFeatures::kOpenPBR));

    in = {};
    in.allNativeOpenPBR = true;
    const WavefrontFeatures allNativeOpenPBR = packWavefrontFeatures(in);
    CHECK(allNativeOpenPBR.has(WavefrontFeatures::kAllNativeOpenPBR));
    CHECK(allNativeOpenPBR.has(WavefrontFeatures::kAllOpenPBR));
    CHECK(allNativeOpenPBR.has(WavefrontFeatures::kOpenPBR));

    in = {};
    in.samplerType = 4u;
    const uint32_t sampler =
        (packWavefrontFeatures(in).bits() & WavefrontFeatures::kSamplerMask) >> WavefrontFeatures::kSamplerShift;
    CHECK(sampler == 4u);
}

TEST_CASE("ReSTIR disables the plain one-candidate RIS specialization")
{
    IntegratorFeatureInputs in;
    in.risOne = true;
    in.restir = true;
    const WavefrontFeatures features = packWavefrontFeatures(in);
    CHECK_FALSE(features.has(WavefrontFeatures::kRisOne));
    CHECK(features.has(WavefrontFeatures::kRestir));
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
