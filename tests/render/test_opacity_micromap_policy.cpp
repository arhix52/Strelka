#include <doctest/doctest.h>

#include "opacity_micromap_policy.h"
#include "texture_compress.h"

#include <vector>

using namespace oka::optix_omm;

namespace
{
/// resolveOpacity() from src/shaders/optix/alpha.h, restated on the host. The
/// micromap is only correct if it agrees with this at every point it claims to
/// know, so the tests compare against this rather than against themselves.
float resolveOpacityReference(const AlphaRule& rule, float texAlpha)
{
    if (rule.alphaMode == kAlphaOpaque)
    {
        return 1.0f;
    }
    float alpha = rule.baseAlpha;
    if (rule.hasTexture)
    {
        alpha *= texAlpha;
    }
    if (rule.alphaMode == kAlphaMask)
    {
        return alpha >= rule.cutoff ? 1.0f : 0.0f;
    }
    return std::min(1.0f, std::max(0.0f, alpha));
}
} // namespace

TEST_CASE("a classified region agrees with resolveOpacity everywhere inside it")
{
    // The whole contract. Sweep alpha ranges and cutoffs, and for every range
    // the classifier calls Opaque or Transparent, check that no alpha in the
    // range resolves the other way.
    AlphaRule rule;
    rule.alphaMode = kAlphaMask;
    rule.hasTexture = true;

    for (int cutoffStep = 0; cutoffStep <= 20; ++cutoffStep)
    {
        rule.cutoff = (float)cutoffStep / 20.0f;
        for (int baseStep = 0; baseStep <= 4; ++baseStep)
        {
            rule.baseAlpha = (float)baseStep / 4.0f;
            for (int loStep = 0; loStep <= 16; ++loStep)
            {
                for (int hiStep = loStep; hiStep <= 16; ++hiStep)
                {
                    const float lo = (float)loStep / 16.0f;
                    const float hi = (float)hiStep / 16.0f;
                    const Coverage c = classifyCoverage(rule, lo, hi, kExactAlphaTolerance);
                    if (c == Coverage::Mixed)
                    {
                        continue;
                    }
                    const float expected = (c == Coverage::Opaque) ? 1.0f : 0.0f;
                    for (int s = 0; s <= 8; ++s)
                    {
                        const float a = lo + (hi - lo) * (float)s / 8.0f;
                        CHECK(resolveOpacityReference(rule, a) == expected);
                    }
                }
            }
        }
    }
}

TEST_CASE("a blend material is opaque only where its alpha saturates")
{
    AlphaRule rule;
    rule.alphaMode = kAlphaBlend;
    rule.hasTexture = true;
    rule.baseAlpha = 1.0f;

    CHECK(classifyCoverage(rule, 1.0f, 1.0f, kExactAlphaTolerance) == Coverage::Opaque);
    CHECK(classifyCoverage(rule, 0.0f, 0.0f, kExactAlphaTolerance) == Coverage::Transparent);
    // 08_alpha_blend: a constant 0.45 with no texture. Nothing about it is
    // resolvable, so every triangle falls through to the shader and the micromap
    // has nothing to say.
    rule.baseAlpha = 0.45f;
    rule.hasTexture = false;
    CHECK(classifyCoverage(rule, 1.0f, 1.0f, kExactAlphaTolerance) == Coverage::Mixed);
    // Nearly opaque is not opaque: the stochastic test still lets one path in
    // two hundred through, and a micromap that swallowed it would be a bias
    // rather than noise.
    rule.baseAlpha = 0.995f;
    rule.hasTexture = true;
    CHECK(classifyCoverage(rule, 1.0f, 1.0f, kExactAlphaTolerance) == Coverage::Mixed);
    // Nor is nearly transparent transparent.
    rule.baseAlpha = 0.001f;
    CHECK(classifyCoverage(rule, 1.0f, 1.0f, kExactAlphaTolerance) == Coverage::Mixed);
}

TEST_CASE("an opaque material needs no test at all")
{
    AlphaRule rule;
    rule.alphaMode = kAlphaOpaque;
    rule.baseAlpha = 0.1f;
    CHECK(classifyCoverage(rule, 0.0f, 0.0f, kExactAlphaTolerance) == Coverage::Opaque);
}

TEST_CASE("the tolerance only ever widens the band nothing is classified in")
{
    AlphaRule rule;
    rule.alphaMode = kAlphaMask;
    rule.hasTexture = true;
    rule.baseAlpha = 1.0f;
    rule.cutoff = 0.5f;

    for (int loStep = 0; loStep <= 32; ++loStep)
    {
        for (int hiStep = loStep; hiStep <= 32; ++hiStep)
        {
            const float lo = (float)loStep / 32.0f;
            const float hi = (float)hiStep / 32.0f;
            const Coverage exact = classifyCoverage(rule, lo, hi, kExactAlphaTolerance);
            const Coverage loose = classifyCoverage(rule, lo, hi, kBlockCompressedAlphaTolerance);
            if (loose != Coverage::Mixed)
            {
                CHECK(loose == exact);
            }
        }
    }
}

TEST_CASE("a degenerate rule is never claimed to be known")
{
    AlphaRule rule;
    rule.alphaMode = kAlphaMask;
    rule.hasTexture = true;
    rule.baseAlpha = 1.0f;
    rule.cutoff = 0.5f;
    // Inverted range: whoever produced it did not measure what they thought.
    CHECK(classifyCoverage(rule, 1.0f, 0.0f, kExactAlphaTolerance) == Coverage::Mixed);
    // A negative factor would flip the bound below, so the bound is not one.
    rule.baseAlpha = -1.0f;
    CHECK(classifyCoverage(rule, 0.0f, 1.0f, kExactAlphaTolerance) == Coverage::Mixed);
}

TEST_CASE("microtriangle states pack four to a byte, least significant first")
{
    constexpr uint32_t level = 3;
    const uint32_t count = microTriangleCount(level);
    CHECK(count == 64u);
    CHECK(microMapBytes(level) == 16u);

    std::vector<uint8_t> states(microMapBytes(level), 0u);
    for (uint32_t i = 0; i < count; ++i)
    {
        setMicroState(states.data(), i, i & 3u);
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        CHECK(getMicroState(states.data(), i) == (i & 3u));
    }
    // The layout the format reads, spelled out once so a change to setMicroState
    // has to face it: microtriangles 0..3 are the four state pairs of byte 0.
    CHECK(states[0] == (uint8_t)(0u | (1u << 2) | (2u << 4) | (3u << 6)));

    // Overwriting one pair leaves its neighbours alone -- the classifier fills
    // the array in one pass, but a partial rewrite must not smear.
    setMicroState(states.data(), 1u, kStateTransparent);
    CHECK(getMicroState(states.data(), 0u) == 0u);
    CHECK(getMicroState(states.data(), 1u) == kStateTransparent);
    CHECK(getMicroState(states.data(), 2u) == 2u);
}

TEST_CASE("subdivision follows the budget rather than the mesh")
{
    // Level 4 is 64 bytes a triangle. Two triangles of a cutout card fit
    // anywhere; a million of them do not.
    CHECK(chooseSubdivisionLevel(2, 64u << 20, 4) == 4u);
    CHECK(microMapBytes(4) == 64u);

    const uint32_t forest = chooseSubdivisionLevel(1u << 20, 64u << 20, 4);
    CHECK(forest == 4u); // exactly 64 MB, which fits
    CHECK(chooseSubdivisionLevel((1u << 20) + 1u, 64u << 20, 4) == 3u);

    // No level at all rather than a level that cannot pay for itself.
    CHECK(chooseSubdivisionLevel(1u << 30, 1024, 4) == kNoSubdivision);
    CHECK(chooseSubdivisionLevel(0, 64u << 20, 4) == kNoSubdivision);
    // Level 0 is one microtriangle for the whole triangle, which a predefined
    // index already expresses for free, so the search never returns it.
    CHECK(chooseSubdivisionLevel(1, 64u << 20, 0) == kNoSubdivision);
}

TEST_CASE("a texel span covers every texel a bilinear fetch inside the box can read")
{
    constexpr int size = 8;
    // The centre of texel 3 is at (3 + 0.5) / 8. A point there reads texels 3
    // and, at the boundary of the filter, its neighbour either side.
    const TexelSpan point = bilinearTexelSpan(3.5f / 8.0f, 3.5f / 8.0f, size, 0.0f);
    CHECK_FALSE(point.full);
    CHECK(point.lo <= 3);
    CHECK(point.hi >= 3);

    // Brute force: for every sample in a box, the two texels CUDA's linear
    // filter reads must be inside the span.
    for (int a = 0; a <= 12; ++a)
    {
        for (int b = a; b <= 12; ++b)
        {
            const float c0 = (float)a / 12.0f;
            const float c1 = (float)b / 12.0f;
            const TexelSpan span = bilinearTexelSpan(c0, c1, size, 0.0f);
            if (span.full)
            {
                continue;
            }
            for (int s = 0; s <= 64; ++s)
            {
                const float u = c0 + (c1 - c0) * (float)s / 64.0f;
                const float x = u * (float)size - 0.5f;
                const int i0 = (int)std::floor(x);
                const int i1 = i0 + 1;
                CHECK(i0 >= span.lo);
                CHECK(i1 <= span.hi);
            }
        }
    }
}

TEST_CASE("a box wider than the texture is reported as the whole axis")
{
    CHECK(bilinearTexelSpan(0.0f, 2.0f, 8, 0.0f).full);
    CHECK(bilinearTexelSpan(-1000.0f, 1000.0f, 4, 0.0f).full);
    CHECK(bilinearTexelSpan(0.0f, 1.0f, 4, 0.0f).full);
}

TEST_CASE("texel indices wrap the way cudaAddressModeWrap does")
{
    CHECK(wrapTexel(0, 8) == 0);
    CHECK(wrapTexel(7, 8) == 7);
    CHECK(wrapTexel(8, 8) == 0);
    CHECK(wrapTexel(-1, 8) == 7);
    CHECK(wrapTexel(-9, 8) == 7);
}

TEST_CASE("BC4 alpha decodes to what the compressor was handed, within the palette")
{
    // Round trip through the encoder this tree ships. A micromap built from the
    // uncompressed file would describe a texture the renderer does not have, so
    // the decode has to be the compressed one's -- and the tolerance has to
    // cover whatever the palette cost.
    auto roundTrip = [](const std::vector<uint8_t>& alphas) {
        REQUIRE(alphas.size() == 16u);
        std::vector<uint8_t> rgba(size_t{ 16 } * 4, 0u);
        for (size_t i = 0; i < 16; ++i)
        {
            rgba[i * 4 + 3] = alphas[i];
        }
        uint8_t block[8] = {};
        oka::bc::compressBlockBC4(rgba.data(), 4, 4, 0, 0, size_t{ 4 } * 4, 3, block);
        uint8_t decoded[16] = {};
        decodeBc4AlphaBlock(block, decoded);
        return std::vector<uint8_t>(decoded, decoded + 16);
    };

    SUBCASE("a flat block is exact")
    {
        for (int v : { 0, 1, 127, 254, 255 })
        {
            const std::vector<uint8_t> in(16, (uint8_t)v);
            const std::vector<uint8_t> out = roundTrip(in);
            for (size_t i = 0; i < 16; ++i)
            {
                CHECK((int)out[i] == v);
            }
        }
    }

    SUBCASE("a two-valued block -- what a cutout mask actually is -- is exact")
    {
        std::vector<uint8_t> in(16, 0u);
        for (size_t i = 0; i < 16; ++i)
        {
            in[i] = (i % 3 == 0) ? 255u : 0u;
        }
        const std::vector<uint8_t> out = roundTrip(in);
        for (size_t i = 0; i < 16; ++i)
        {
            CHECK((int)out[i] == (int)in[i]);
        }
    }

    SUBCASE("a ramp stays inside the palette's own error")
    {
        std::vector<uint8_t> in(16, 0u);
        for (size_t i = 0; i < 16; ++i)
        {
            in[i] = (uint8_t)(i * 17);
        }
        const std::vector<uint8_t> out = roundTrip(in);
        int worst = 0;
        for (size_t i = 0; i < 16; ++i)
        {
            worst = std::max(worst, std::abs((int)out[i] - (int)in[i]));
        }
        // Sixteen values over a range of 255 through an eight-entry palette:
        // half a step is 255/14, and that is what the encoder achieves.
        CHECK(worst <= 255 / 14 + 1);
    }
}

TEST_CASE("a uniform triangle names a predefined index instead of storing a map")
{
    CHECK(predefinedIndexFor(Coverage::Opaque) == kIndexFullyOpaque);
    CHECK(predefinedIndexFor(Coverage::Transparent) == kIndexFullyTransparent);
    CHECK(predefinedIndexFor(Coverage::Mixed) == kIndexFullyUnknownOpaque);

    CHECK(microStateFor(Coverage::Opaque) == kStateOpaque);
    CHECK(microStateFor(Coverage::Transparent) == kStateTransparent);
    // Mixed must be one of the two unknown states, or the shader never runs.
    CHECK((microStateFor(Coverage::Mixed) == kStateUnknownOpaque ||
           microStateFor(Coverage::Mixed) == kStateUnknownTransparent));
}

TEST_CASE("a micromap that resolves nothing is not worth building")
{
    BuildSummary all_unknown;
    all_unknown.triangles = 100;
    all_unknown.uniformUnknown = 100;
    CHECK(all_unknown.isPointless());

    BuildSummary useful;
    useful.triangles = 100;
    useful.uniformUnknown = 98;
    useful.uniformOpaque = 2;
    CHECK_FALSE(useful.isPointless());

    BuildSummary subdivided;
    subdivided.triangles = 100;
    subdivided.subdivided = 1;
    subdivided.uniformUnknown = 99;
    CHECK_FALSE(subdivided.isPointless());
}
