// Tests for src/scene/include/strelka/scene/vertex_packing.h
//
// The tangent handedness sign lives in bit 30 of the packed tangent. That only
// works because packNormal fills bits 0..29 and nothing above, and because the
// unpacked z mask is 0x3ff00000 (10 bits) rather than the older 0xfff00000
// (12 bits). The same narrowing had to be repeated in the CUDA and Metal
// device-side copies, so the invariants below are the contract all three share.

#include <doctest/doctest.h>

// glm_wrapper.hpp is what defines the glm::floatN aliases that vertex_packing.h uses.
#include <strelka/scene/glm_wrapper.hpp>
#include <strelka/scene/vertex_packing.h>

#include <cmath>
#include <vector>

using oka::kTangentSignBit;
using oka::packNormal;
using oka::packTangent;
using oka::packUV;
using oka::unpackNormal;
using oka::unpackTangentSign;
using oka::unpackUV;

namespace
{

// packNormal biases by +1 and scales by 256, so one quantisation step is 1/256
// of the [-1, 1] range. The conversion truncates, so the unpacked value is at
// or below the original, up to one step away.
constexpr float kNormalStep = 1.0f / 256.0f;
// (n + 1.0f) is rounded once in float before the scale; that can nudge a value
// across an integer boundary by ~1.5e-5 packed units. Slack for exactly that.
constexpr float kFloatSlack = 1.0e-4f;

// packUV maps [-10, 10] onto [0, 16383] per component.
constexpr float kUVStep = 20.0f / 16383.99999f;

std::vector<glm::float3> unitVectors()
{
    std::vector<glm::float3> v = {
        // axis aligned, both signs -- these hit the extremes of the packed range
        { 1.0f, 0.0f, 0.0f },  { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f },
        { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f },  { 0.0f, 0.0f, -1.0f },
        // mixed signs
        { 0.0f, 0.0f, 0.0f },
    };
    v.back() = glm::normalize(glm::float3(1.0f, 1.0f, 1.0f));
    v.push_back(glm::normalize(glm::float3(-1.0f, 1.0f, -1.0f)));
    v.push_back(glm::normalize(glm::float3(1.0f, -1.0f, -1.0f)));
    v.push_back(glm::normalize(glm::float3(-1.0f, -1.0f, 1.0f)));
    v.push_back(glm::normalize(glm::float3(0.3f, -0.7f, 0.65f)));
    v.push_back(glm::normalize(glm::float3(-0.02f, 0.999f, 0.04f)));

    // a lat/long sweep so the "no high bits" claim is tested densely, not just
    // on hand-picked directions
    for (int i = 0; i <= 16; ++i)
    {
        const float theta = float(i) / 16.0f * 3.14159265358979f;
        for (int j = 0; j < 32; ++j)
        {
            const float phi = float(j) / 32.0f * 2.0f * 3.14159265358979f;
            v.emplace_back(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
        }
    }
    return v;
}

void checkComponentRoundTrip(float original, float unpacked)
{
    const float err = original - unpacked;
    // truncation: never overshoot (beyond float slack)...
    CHECK(err >= -kFloatSlack);
    // ...and never fall more than one quantisation step short
    CHECK(err <= kNormalStep + kFloatSlack);
    // and stay inside the representable range
    CHECK(unpacked >= -1.0f);
    CHECK(unpacked <= 1.0f);
}

} // namespace

TEST_CASE("packNormal/unpackNormal round-trips within 10-bit quantisation error")
{
    for (const glm::float3& n : unitVectors())
    {
        const glm::float3 r = unpackNormal(packNormal(n));
        CAPTURE(n.x);
        CAPTURE(n.y);
        CAPTURE(n.z);
        checkComponentRoundTrip(n.x, r.x);
        checkComponentRoundTrip(n.y, r.y);
        checkComponentRoundTrip(n.z, r.z);
    }
}

TEST_CASE("packNormal round-trips axis-aligned vectors exactly")
{
    // +-1 and 0 land on exact multiples of 1/256, so no error at all is allowed.
    const glm::float3 axes[] = {
        { 1.0f, 0.0f, 0.0f },  { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f },
        { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f },  { 0.0f, 0.0f, -1.0f },
    };
    for (const glm::float3& n : axes)
    {
        const glm::float3 r = unpackNormal(packNormal(n));
        CHECK(r.x == n.x);
        CHECK(r.y == n.y);
        CHECK(r.z == n.z);
    }
}

TEST_CASE("packNormal never sets bit 30 or bit 31")
{
    // This is the invariant the tangent sign bit is built on. If packNormal ever
    // spilled into bit 30, packTangent would corrupt the direction and
    // unpackTangentSign would read garbage.
    for (const glm::float3& n : unitVectors())
    {
        const uint32_t p = packNormal(n);
        CAPTURE(n.x);
        CAPTURE(n.y);
        CAPTURE(n.z);
        CHECK((p & 0xc0000000u) == 0u);
        CHECK((p & kTangentSignBit) == 0u);
        // each 10-bit field tops out at 512 == (1 + 1) * 256
        CHECK(((p >> 20) & 0x3ffu) <= 512u);
        CHECK(((p >> 10) & 0x3ffu) <= 512u);
        CHECK((p & 0x3ffu) <= 512u);
    }
}

TEST_CASE("packNormal leaves bit 30 clear even for slightly over-unit input")
{
    // Normalisation in the loader is not exact; a component can land a few ulps
    // past 1.0. That must still not reach bit 30.
    const glm::float3 nudged[] = {
        { 1.0f + 1e-6f, 0.0f, 0.0f },
        { 0.0f, 1.0f + 1e-6f, 0.0f },
        { 0.0f, 0.0f, 1.0f + 1e-6f },
        { 1.0f + 1e-6f, 1.0f + 1e-6f, 1.0f + 1e-6f },
    };
    for (const glm::float3& n : nudged)
    {
        CHECK((packNormal(n) & 0xc0000000u) == 0u);
    }
}

TEST_CASE("packTangent encodes handedness without disturbing the direction")
{
    for (const glm::float3& t : unitVectors())
    {
        const uint32_t pos = packTangent(t, 1.0f);
        const uint32_t neg = packTangent(t, -1.0f);

        CAPTURE(t.x);
        CAPTURE(t.y);
        CAPTURE(t.z);

        // the sign is the only difference between the two encodings
        CHECK((pos ^ neg) == kTangentSignBit);
        CHECK((pos & kTangentSignBit) == 0u);
        CHECK((neg & kTangentSignBit) == kTangentSignBit);

        CHECK(unpackTangentSign(pos) == 1.0f);
        CHECK(unpackTangentSign(neg) == -1.0f);

        // positive handedness must be byte-identical to a plain packed normal
        CHECK(pos == packNormal(t));

        // and both unpack to the same direction, bit for bit
        const glm::float3 dp = unpackNormal(pos);
        const glm::float3 dn = unpackNormal(neg);
        CHECK(dp.x == dn.x);
        CHECK(dp.y == dn.y);
        CHECK(dp.z == dn.z);
        CHECK(dn.x == unpackNormal(packNormal(t)).x);
        CHECK(dn.y == unpackNormal(packNormal(t)).y);
        CHECK(dn.z == unpackNormal(packNormal(t)).z);
    }
}

TEST_CASE("packTangent treats zero handedness as positive")
{
    // glTF only ever emits +-1, but the comparison is `< 0.0f`, so 0 is positive.
    const glm::float3 t = glm::normalize(glm::float3(0.5f, -0.25f, 0.8f));
    CHECK(unpackTangentSign(packTangent(t, 0.0f)) == 1.0f);
    CHECK(packTangent(t, 0.0f) == packNormal(t));
}

TEST_CASE("the tangent sign bit does not perturb the unpacked z component")
{
    // The regression: a 12-bit z mask (0xfff00000) folds bit 30 into z, adding
    // 1024 * (1/256) == 4.0 to it. The mask must be 0x3ff00000.
    for (const glm::float3& n : unitVectors())
    {
        const uint32_t plain = packNormal(n);
        const uint32_t signed_ = plain | kTangentSignBit;

        CAPTURE(n.z);
        // exact equality, not approximate -- the mask either covers bit 30 or not
        CHECK(unpackNormal(signed_).z == unpackNormal(plain).z);
        CHECK(unpackNormal(signed_).y == unpackNormal(plain).y);
        CHECK(unpackNormal(signed_).x == unpackNormal(plain).x);
        CHECK(unpackNormal(signed_).z <= 1.0f);
        CHECK(unpackNormal(signed_).z >= -1.0f);

        // spell out what the old wide mask would have produced, so that
        // re-widening the mask fails here loudly
        const float wideZ = float((signed_ & 0xfff00000u) >> 20) * (1.0f / 256.0f) - 1.0f;
        CHECK(wideZ == doctest::Approx(unpackNormal(signed_).z + 4.0f));
    }
}

TEST_CASE("packUV/unpackUV round-trips over [-10, 10]")
{
    std::vector<float> samples = { -10.0f, -9.999f, -7.5f, -1.0f,  -0.5f, -0.001f, 0.0f,
                                  0.001f, 0.5f,    1.0f,  3.3333f, 7.5f,  9.999f,  10.0f };
    for (int i = 0; i <= 40; ++i)
    {
        samples.push_back(-10.0f + float(i) * 0.5f);
    }

    for (float u : samples)
    {
        for (float v : samples)
        {
            const glm::float2 r = unpackUV(packUV(glm::float2(u, v)));
            CAPTURE(u);
            CAPTURE(v);
            CHECK(u - r.x >= -kFloatSlack);
            CHECK(u - r.x <= kUVStep + kFloatSlack);
            CHECK(v - r.y >= -kFloatSlack);
            CHECK(v - r.y <= kUVStep + kFloatSlack);
            CHECK(r.x >= -10.0f);
            CHECK(r.x <= 10.0f);
            CHECK(r.y >= -10.0f);
            CHECK(r.y <= 10.0f);
        }
    }
}

TEST_CASE("packUV keeps the two components independent")
{
    // x lives in the low 16 bits, y in the high 16; a carry between them would
    // show up as one component moving when only the other changes.
    const glm::float2 base(-10.0f, 10.0f);
    const uint32_t packed = packUV(base);
    CHECK((packed & 0x0000ffffu) == 0u); // x == -10 -> 0
    // 16383.99999f rounds to exactly 16384.0f in single precision, so the top of
    // the range packs to 16384, not 16383. Harmless -- the field is 16 bits wide
    // and unpackUV divides by the same rounded constant, so 10.0 still
    // round-trips exactly -- but it is load-bearing enough to pin down.
    CHECK(((packed >> 16) & 0xffffu) == 16384u);

    const glm::float2 swapped = unpackUV(packUV(glm::float2(10.0f, -10.0f)));
    CHECK(swapped.x == 10.0f);
    CHECK(swapped.y == -10.0f);

    // and neither component's encoding depends on the other
    for (const float other : { -10.0f, -3.0f, 0.0f, 4.5f, 10.0f })
    {
        CHECK((packUV(glm::float2(2.5f, other)) & 0x0000ffffu) ==
              (packUV(glm::float2(2.5f, 0.0f)) & 0x0000ffffu));
        CHECK((packUV(glm::float2(other, 2.5f)) >> 16) == (packUV(glm::float2(0.0f, 2.5f)) >> 16));
    }
}
