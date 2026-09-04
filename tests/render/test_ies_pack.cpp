#include <doctest/doctest.h>

#include <host/ies_pack.h>
#include <strelka/scene/scene.h>

#include <cmath>
#include <cstring>
#include <limits>

using namespace oka::ies_pack;
using Profile = oka::Scene::IesProfile;

namespace
{

/// Read the packed buffer back exactly the way lights.h::sampleIesCandela does:
/// header, then profile headers at a fixed stride, then float indices relative
/// to the blob. The point of duplicating the arithmetic here rather than calling
/// a shared helper is that the device side cannot be linked into these tests, so
/// what is under test is that the two independent readers agree on the layout.
struct Reader
{
    const uint8_t* base;

    const IesBufferHeader& buffer() const
    {
        return *(const IesBufferHeader*)base;
    }

    const IesProfileHeader& profile(size_t i) const
    {
        return ((const IesProfileHeader*)(base + sizeof(IesBufferHeader)))[i];
    }

    const float* floats() const
    {
        return (const float*)(base + buffer().floatOffset);
    }

    float vertical(size_t p, int i) const
    {
        return floats()[profile(p).anglesOffset + i];
    }

    float horizontal(size_t p, int i) const
    {
        return floats()[profile(p).anglesOffset + profile(p).nVertical + i];
    }

    /// candela[v + h * nVertical], which is the indexing both the loader writes
    /// and the shader reads.
    float candela(size_t p, int v, int h) const
    {
        return floats()[profile(p).candelaOffset + v + h * (int)profile(p).nVertical];
    }
};

/// The 27_ies ladder scene's synthetic luminaire: a cosine^4 hotspot, 1000 cd on
/// axis, 11 vertical angles and one horizontal.
Profile ladderProfile()
{
    Profile p;
    p.verticalAngles = { 0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 75.0f, 90.0f };
    p.horizontalAngles = { 0.0f };
    p.candela = { 1000.0f, 984.865f, 940.602f, 870.513f, 779.728f, 562.5f, 344.363f, 170.714f, 62.5f, 10.0f, 10.0f };
    p.maxCandela = 1000.0f;
    return p;
}

Profile grid(uint32_t nV, uint32_t nH, float first)
{
    Profile p;
    for (uint32_t v = 0; v < nV; ++v)
        p.verticalAngles.push_back(static_cast<float>(v) * (90.0f / (float)(nV - 1)));
    for (uint32_t h = 0; h < nH; ++h)
        p.horizontalAngles.push_back(static_cast<float>(h) * (360.0f / (float)nH));
    for (uint32_t i = 0; i < nV * nH; ++i)
        p.candela.push_back(first + (float)i);
    p.maxCandela = first + (float)(nV * nH - 1);
    return p;
}

} // namespace

TEST_CASE("a scene with no IES profile still gets a buffer the shader can read")
{
    // The alternative is a null pointer in SceneData and a branch per light in
    // connectLight. A zero-count header is a valid empty table and costs 16
    // bytes, so every scene takes the same path.
    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{});
    REQUIRE(bytes.size() == sizeof(IesBufferHeader));

    const Reader r{ bytes.data() };
    CHECK(r.buffer().profileCount == 0u);
    CHECK(r.buffer().floatOffset == sizeof(IesBufferHeader));
}

TEST_CASE("the packed layout is what the device-side reader indexes")
{
    const Profile p = ladderProfile();
    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{ p });

    const Reader r{ bytes.data() };
    REQUIRE(r.buffer().profileCount == 1u);
    CHECK(r.buffer().floatOffset == sizeof(IesBufferHeader) + sizeof(IesProfileHeader));

    CHECK(r.profile(0).nVertical == 11u);
    CHECK(r.profile(0).nHorizontal == 2u);
    // Angles come first in the blob, vertical then horizontal, candela after.
    CHECK(r.profile(0).anglesOffset == 0u);
    CHECK(r.profile(0).candelaOffset == 13u); // 11 vertical + 0/360 horizontal

    CHECK(r.vertical(0, 0) == doctest::Approx(0.0f));
    CHECK(r.vertical(0, 10) == doctest::Approx(90.0f));
    CHECK(r.horizontal(0, 0) == doctest::Approx(0.0f));
    CHECK(r.horizontal(0, 1) == doctest::Approx(360.0f));
    CHECK(r.candela(0, 0, 1) == doctest::Approx(1000.0f * kCandelaToRadiantIntensity));

    // Exactly the bytes the blob should occupy, with nothing padded between the
    // two sections: a stray float here would shift every candela read by one.
    CHECK(bytes.size() == r.buffer().floatOffset + (11 + 2 + 22) * sizeof(float));
}

TEST_CASE("candela is divided by the D65 luminous efficacy Cycles assumes")
{
    // An IES file is photometric (candela) and a light's colour here is radiant
    // intensity (W/sr). 177.83 lm/W is the D65 figure Cycles uses for the same
    // conversion. This test exists to stop it being quietly replaced by a
    // constant fitted to the 27_ies row: a fit that lands that row would leave
    // every IES scene outside the ladder wrong by whatever it absorbed.
    CHECK(kLuminousEfficacyD65 == doctest::Approx(177.83f));

    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{ ladderProfile() });
    const Reader r{ bytes.data() };

    CHECK(r.candela(0, 0, 0) == doctest::Approx(1000.0f / 177.83f));
    CHECK(r.candela(0, 8, 0) == doctest::Approx(62.5f / 177.83f));
    CHECK(r.candela(0, 10, 0) == doctest::Approx(10.0f / 177.83f));
    CHECK(r.profile(0).maxCandela == doctest::Approx(1000.0f / 177.83f));

    // The whole table is scaled by one constant, so the shape survives: the
    // ratio of any two entries is what the row is actually comparing.
    CHECK(r.candela(0, 0, 0) / r.candela(0, 8, 0) == doctest::Approx(1000.0f / 62.5f));
}

TEST_CASE("row-major candela indexing survives more than one horizontal plane")
{
    // candela[v + h * nVertical]. Getting this transposed reads a plausible
    // number from the wrong plane, which is invisible on a rotationally
    // symmetric luminaire like the ladder's and wrong on every real one.
    const Profile p = grid(4, 3, 100.0f);
    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{ p });
    const Reader r{ bytes.data() };

    for (int h = 0; h < 3; ++h)
    {
        for (int v = 0; v < 4; ++v)
        {
            CHECK(r.candela(0, v, h) == doctest::Approx((100.0f + (float)(v + h * 4)) * kCandelaToRadiantIntensity));
        }
    }
}

TEST_CASE("a second profile's blob indices follow the first")
{
    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{ grid(4, 1, 1.0f), grid(6, 2, 50.0f) });
    const Reader r{ bytes.data() };
    REQUIRE(r.buffer().profileCount == 2u);

    // A single rotationally symmetric plane becomes addressable 0/360 columns.
    CHECK(r.profile(0).anglesOffset == 0u);
    CHECK(r.profile(0).candelaOffset == 6u);
    // Profile 1 starts after all fourteen of profile 0's floats.
    CHECK(r.profile(1).anglesOffset == 14u);
    CHECK(r.profile(1).candelaOffset == 14u + 8u); // 6 vertical + 2 horizontal

    CHECK(r.candela(0, 0, 0) == doctest::Approx(1.0f * kCandelaToRadiantIntensity));
    CHECK(r.candela(1, 0, 0) == doctest::Approx(50.0f * kCandelaToRadiantIntensity));
    CHECK(r.candela(1, 5, 1) == doctest::Approx((50.0f + 11.0f) * kCandelaToRadiantIntensity));
}

TEST_CASE("a degenerate profile is packed empty rather than dropped")
{
    // The light records carry profile indices assigned at load time, so
    // dropping a malformed profile would silently repoint every later light at
    // its neighbour's distribution. An empty entry is read as "no data" by
    // sampleIesCandela's nVertical < 2 guard instead.
    Profile broken;
    broken.verticalAngles = { 0.0f }; // one angle cannot be interpolated
    broken.horizontalAngles = { 0.0f };
    broken.candela = { 500.0f };

    Profile truncated = grid(4, 2, 7.0f);
    truncated.candela.pop_back(); // shorter than the grid it declares

    const std::vector<uint8_t> bytes = packProfiles(std::vector<Profile>{ broken, ladderProfile(), truncated });
    const Reader r{ bytes.data() };
    REQUIRE(r.buffer().profileCount == 3u);

    CHECK(r.profile(0).nVertical == 0u);
    CHECK(r.profile(2).nVertical == 0u);
    // The good profile keeps index 1, which is what its light records point at.
    CHECK(r.profile(1).nVertical == 11u);
    CHECK(r.candela(1, 0, 0) == doctest::Approx(1000.0f * kCandelaToRadiantIntensity));
}

TEST_CASE("malformed IES grids are empty and cannot expose device lookups")
{
    Profile nonFiniteVertical = grid(3, 2, 1.0f);
    nonFiniteVertical.verticalAngles[1] = std::numeric_limits<float>::quiet_NaN();

    Profile descendingVertical = grid(3, 2, 2.0f);
    descendingVertical.verticalAngles = { 0.0f, 90.0f, 45.0f };

    Profile nonFiniteHorizontal = grid(3, 2, 3.0f);
    nonFiniteHorizontal.horizontalAngles[1] = std::numeric_limits<float>::infinity();

    Profile descendingHorizontal = grid(3, 2, 4.0f);
    descendingHorizontal.horizontalAngles = { 90.0f, 0.0f };

    Profile negativeCandela = grid(3, 2, 5.0f);
    negativeCandela.candela[2] = -1.0f;

    Profile nonFiniteCandela = grid(3, 2, 6.0f);
    nonFiniteCandela.candela[4] = std::numeric_limits<float>::infinity();

    Profile staleMaximum = grid(3, 2, 7.0f);
    staleMaximum.maxCandela = std::numeric_limits<float>::quiet_NaN();

    const std::vector<uint8_t> bytes =
        packProfiles(std::vector<Profile>{ nonFiniteVertical, descendingVertical, nonFiniteHorizontal,
                                           descendingHorizontal, negativeCandela, nonFiniteCandela, staleMaximum });
    const Reader r{ bytes.data() };
    REQUIRE(r.buffer().profileCount == 7u);
    for (size_t i = 0; i < 6; ++i)
    {
        CHECK(r.profile(i).nVertical == 0u);
        CHECK(r.profile(i).nHorizontal == 0u);
    }

    // maxCandela is derived data. A stale or non-finite cached value must not
    // invalidate an otherwise usable table or leak into shader radiance.
    CHECK(r.profile(6).nVertical == 3u);
    CHECK(r.profile(6).nHorizontal == 2u);
    CHECK(r.profile(6).maxCandela == doctest::Approx(12.0f * kCandelaToRadiantIntensity));
}

TEST_CASE("the float blob starts immediately after the profile headers")
{
    for (const size_t n : { size_t(0), size_t(1), size_t(2), size_t(7) })
    {
        const std::vector<Profile> profiles(n, grid(3, 1, 1.0f));
        const std::vector<uint8_t> bytes = packProfiles(profiles);
        const Reader r{ bytes.data() };
        CHECK(r.buffer().floatOffset == floatBlobOffset(n));
        CHECK(r.buffer().floatOffset == sizeof(IesBufferHeader) + n * sizeof(IesProfileHeader));
        CHECK(bytes.size() >= r.buffer().floatOffset);
    }
}
