#include <doctest/doctest.h>

#include "integrator_buffer_sizes.h"

using oka::metal::kWavefrontControlUints;
using oka::metal::wavefrontBufferLayout;
using oka::metal::WavefrontElementSizes;

TEST_CASE("wavefrontBufferLayout scales with pixel count")
{
    WavefrontElementSizes sz;
    sz.pathState = 64;
    sz.pathRay = 32;
    sz.hitRecord = 48;
    sz.iorStack = 16;
    sz.radiance = 16;
    sz.shadowRay = 40;
    sz.aovSample = 80;

    const auto a = wavefrontBufferLayout(64, 48, sz);
    CHECK(a.pixels == 64u * 48u);
    CHECK(a.pathStateBytes == (size_t)a.pixels * 64);
    CHECK(a.pathRayBytes == (size_t)a.pixels * 32);
    CHECK(a.hitBytes == (size_t)a.pixels * 48);
    CHECK(a.iorStackBytes == (size_t)a.pixels * 16);
    CHECK(a.radianceBytes == (size_t)a.pixels * 16);
    CHECK(a.guideRadianceBytes == a.radianceBytes);
    CHECK(a.pathQueueBytes == (size_t)a.pixels * sizeof(uint32_t));
    CHECK(a.controlBytes == (size_t)kWavefrontControlUints * sizeof(uint32_t));
    CHECK(a.shadowRayBytes == (size_t)a.pixels * 40);
    CHECK(a.aovBytes == (size_t)a.pixels * 80);
    CHECK(a.hitQueueBytes == a.pathQueueBytes);
    CHECK(a.missQueueBytes == a.pathQueueBytes);

    const auto b = wavefrontBufferLayout(128, 96, sz);
    CHECK(b.pixels == 4 * a.pixels);
    CHECK(b.pathStateBytes == 4 * a.pathStateBytes);
    CHECK(b.controlBytes == a.controlBytes);
}

TEST_CASE("preview presets make wavefront memory growth explicit")
{
    WavefrontElementSizes sz;
    sz.pathState = 60;
    sz.pathRay = 24;
    sz.hitRecord = 32;
    sz.iorStack = 52;
    sz.radiance = 16;
    sz.shadowRay = 48;
    sz.aovSample = 64;

    const auto preview = wavefrontBufferLayout(960, 540, sz);
    const auto fullHd = wavefrontBufferLayout(1920, 1080, sz);

    CHECK(preview.pixels == 518400);
    CHECK(fullHd.pixels == 4 * preview.pixels);
    CHECK(fullHd.pathStateBytes == 4 * preview.pathStateBytes);
    CHECK(fullHd.shadowRayBytes == 4 * preview.shadowRayBytes);
    CHECK(fullHd.aovBytes == 4 * preview.aovBytes);
    CHECK(fullHd.controlBytes == preview.controlBytes);
}
