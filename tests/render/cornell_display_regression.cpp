// Display-transform regression on a real frame.
//
// The unit suite exercises the tone curves on synthetic ramps, which is where
// monotonicity and the headroom ceiling are easiest to pin down. It is not where
// the interesting failures have been. Scaling both of a curve's axes to the
// display headroom looked correct on a grey ramp and on the Cornell box, and was
// wrong by a stop and a half on a frame that actually had content above white --
// the defect lived in the *distribution* of a scene, not in the curve's algebra.
//
// So this renders one, at a resolution and sample count chosen to be quick
// rather than converged, and asserts the properties the display path promises
// over every pixel of it.
//
// Skips with 77 rather than failing when there is no StrelkaCLI beside it, no
// scene, or no device to render with: a build host without a GPU should not have
// a red test.

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#include <tinyexr.h>

#include <tonemappers.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
constexpr int kSkipExitCode = 77;

using oka::tonemap::ToneMapperType;
using oka::tonemap::float3;
using oka::tonemap::make_float3;

/// Records rather than aborts, so one run reports every property that broke
/// instead of the first.
class Checks
{
public:
    void operator()(bool condition, const std::string& what)
    {
        if (!condition)
        {
            std::printf("FAIL: %s\n", what.c_str());
            ++mFailed;
        }
    }

    int failed() const
    {
        return mFailed;
    }

private:
    int mFailed = 0;
};

float3 applyCurve(ToneMapperType type, const float3 color, float maxOutput)
{
    switch (type)
    {
    case ToneMapperType::eReinhard:
        return oka::tonemap::reinhard(color, maxOutput);
    case ToneMapperType::eACES:
        return oka::tonemap::ACESFitted(color, maxOutput);
    case ToneMapperType::eFilmic:
        return oka::tonemap::ACESFilm(color, maxOutput);
    case ToneMapperType::eNone:
        break;
    }
    return color;
}

float peakChannel(const float3 c)
{
    return std::max(std::max(c.x, c.y), c.z);
}

const char* curveName(ToneMapperType type)
{
    switch (type)
    {
    case ToneMapperType::eReinhard:
        return "Reinhard";
    case ToneMapperType::eACES:
        return "ACES";
    case ToneMapperType::eFilmic:
        return "Filmic";
    case ToneMapperType::eNone:
        break;
    }
    return "None";
}
} // namespace

int main()
{
    const std::filesystem::path exeDir = std::filesystem::path(STRELKA_TEST_BINARY_DIR);
    const std::filesystem::path cli = exeDir / "StrelkaCLI";
    const std::filesystem::path scene =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR) / "cornell_box" / "cornell_box.glb";
    std::error_code ec;

    if (!std::filesystem::exists(cli, ec) || !std::filesystem::exists(scene, ec))
    {
        std::printf("SKIP: StrelkaCLI or the Cornell box scene is not present\n");
        return kSkipExitCode;
    }

    const std::filesystem::path out =
        std::filesystem::temp_directory_path() / "strelka_cornell_display_regression.exr";
    std::filesystem::remove(out, ec);

    // Linear radiance, which is what the display transform takes as input. Small
    // and noisy on purpose: the assertions below are about the transform, and
    // noise only widens the distribution it has to hold for.
    const std::string command = "\"" + cli.string() + "\" \"" + scene.string() + "\" -o \"" + out.string() +
                                "\" -w 256 --height 192 --spp 16 > /dev/null 2>&1";
    // A fixed command line built from compile-time paths, run once from a
    // single-threaded test binary: neither the injection nor the reentrancy the
    // two checks exist to catch is reachable here, and rendering a frame is the
    // point of this test.
    // NOLINTNEXTLINE(bugprone-command-processor,concurrency-mt-unsafe)
    if (std::system(command.c_str()) != 0 || !std::filesystem::exists(out, ec))
    {
        std::printf("SKIP: StrelkaCLI could not render the Cornell box (no device?)\n");
        return kSkipExitCode;
    }

    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    const char* err = nullptr;
    if (LoadEXR(&pixels, &width, &height, out.string().c_str(), &err) != TINYEXR_SUCCESS)
    {
        std::printf("FAIL: could not read %s: %s\n", out.string().c_str(), err ? err : "unknown");
        FreeEXRErrorMessage(err);
        return 1;
    }
    // Success is supposed to imply all three, and the analyzer cannot see that
    // through tinyexr; asserting it here is cheaper than a NOLINT and catches a
    // reader that ever stops honouring its own contract.
    if (pixels == nullptr || width <= 0 || height <= 0)
    {
        std::printf("FAIL: %s decoded to nothing\n", out.string().c_str());
        return 1;
    }

    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    // Copied out at once so the malloc'd block does not have to stay alive across
    // everything below, and its free sits next to its load where the pairing can
    // be checked by eye.
    const std::vector<float> frame(pixels, pixels + count * 4);
    // tinyexr hands back a malloc'd block; free is the matching call, not a
    // choice this test gets to make.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    free(pixels);
    pixels = nullptr;

    Checks check;

    // The frame has to be worth asserting over before the assertions mean
    // anything: a black or a NaN-filled render would satisfy most of them.
    size_t aboveWhite = 0;
    size_t nonFinite = 0;
    float sceneMax = 0.0f;
    double sceneMean = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const float3 c = make_float3(frame[i * 4], frame[i * 4 + 1], frame[i * 4 + 2]);
        if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z))
        {
            ++nonFinite;
            continue;
        }
        const float peak = peakChannel(c);
        sceneMax = std::max(sceneMax, peak);
        sceneMean += peak;
        if (peak > 1.0f)
        {
            ++aboveWhite;
        }
    }
    sceneMean /= static_cast<double>(count);
    std::printf("Cornell box %dx%d: mean %.4f, max %.3f, %.2f%% above linear white\n", width, height, sceneMean,
                sceneMax, 100.0 * static_cast<double>(aboveWhite) / static_cast<double>(count));

    check(nonFinite == 0, "the render contains no NaN or infinite radiance");
    check(sceneMean > 1e-3, "the render is not black");
    check(aboveWhite > count / 1000, "the render has content above linear white for the headroom to work on");

    const ToneMapperType curves[] = { ToneMapperType::eReinhard, ToneMapperType::eACES, ToneMapperType::eFilmic };
    const float headrooms[] = { 1.5f, 3.54f, 16.0f };

    for (const ToneMapperType curve : curves)
    {
        for (const float headroom : headrooms)
        {
            size_t darker = 0;
            size_t overCeiling = 0;
            size_t movedUnderWhite = 0;
            size_t lifted = 0;
            for (size_t i = 0; i < count; ++i)
            {
                const float3 linear = make_float3(frame[i * 4], frame[i * 4 + 1], frame[i * 4 + 2]);
                const float3 sdr = applyCurve(curve, linear, 1.0f);
                const float3 hdr = applyCurve(curve, linear, headroom);
                const float sdrPeak = peakChannel(sdr);
                const float hdrPeak = peakChannel(hdr);

                if (hdrPeak < sdrPeak - 1e-6f)
                {
                    ++darker;
                }
                // The curve's own output is the floor here, not white: Reinhard
                // divides by luminance, so a saturated channel can land above the
                // display peak on its own, and pulling that back down would be a
                // pixel darker than the SDR curve rather than a brighter one.
                if (hdrPeak > std::max(headroom, sdrPeak) + 1e-4f)
                {
                    ++overCeiling;
                }
                if (hdrPeak > sdrPeak + 1e-6f)
                {
                    ++lifted;
                }
                // Nothing the scene put at or below diffuse white may move: the
                // display mode is meant to add highlights, not to regrade.
                if (peakChannel(linear) <= 1.0f && std::abs(hdrPeak - sdrPeak) > 1e-6f)
                {
                    ++movedUnderWhite;
                }
            }
            std::printf("  %-8s at %5.2fx: lifted %5.2f%%, darker %zu, over ceiling %zu, sub-white moved %zu\n",
                        curveName(curve), static_cast<double>(headroom),
                        100.0 * static_cast<double>(lifted) / static_cast<double>(count), darker, overCeiling,
                        movedUnderWhite);

            const std::string label = std::string(curveName(curve)) + " at " + std::to_string(headroom) + "x";
            check(darker == 0, label + ": no pixel is darker than the SDR curve");
            check(overCeiling == 0, label + ": no pixel exceeds the display headroom");
            check(movedUnderWhite == 0, label + ": nothing at or below diffuse white moves");
            check(lifted > 0, label + ": the headroom is actually spent on something");
        }
    }

    std::filesystem::remove(out, ec);
    if (check.failed() != 0)
    {
        std::printf("%d check(s) failed\n", check.failed());
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
