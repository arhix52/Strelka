// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#define __device__
#define __inline__ inline
#define __constant__
#define __forceinline__ inline
#define __host__
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)

#include <algorithm>
#include <cmath>

using std::min;

inline float clamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

#include <random.h>

#include <doctest/doctest.h>

#include <cstdint>

namespace
{

// Columns 0 and 1 of the Joe-Kuo table (new-joe-kuo-6.21201), verbatim from the
// table random.h used to carry.
constexpr uint32_t kDim0[32] = {
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u,
    0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u,
    0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u,
    0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u,
    0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u,
};

constexpr uint32_t kDim1[32] = {
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
    0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
    0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
    0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
    0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
};

/// The definition the closed forms have to agree with: XOR the direction number
/// of every set bit of the index.
uint32_t walk(const uint32_t (&v)[32], uint32_t index)
{
    uint32_t x = 0;
    for (uint32_t bit = 0; bit < 32u; ++bit)
    {
        if ((index >> bit) & 1u)
        {
            x ^= v[bit];
        }
    }
    return x;
}

} // namespace

TEST_CASE("the closed-form Sobol' dimensions are the tabulated ones")
{
    // Linear over GF(2), so agreeing on the 32 basis vectors settles every
    // index. The sweep afterwards is belt and braces against a transcription
    // slip in `walk` itself.
    for (uint32_t bit = 0; bit < 32u; ++bit)
    {
        const uint32_t index = 1u << bit;
        CHECK(sobol_dim0(index) == kDim0[bit]);
        CHECK(sobol_dim1(index) == kDim1[bit]);
    }

    for (uint32_t index = 0; index < 4096u; ++index)
    {
        CHECK(sobol_dim0(index) == walk(kDim0, index));
        CHECK(sobol_dim1(index) == walk(kDim1, index));
    }

    // A few by hand, so a failure says which end it is wrong at.
    CHECK(sobol_dim0(0) == 0u);
    CHECK(sobol_dim0(1) == 0x80000000u);
    CHECK(sobol_dim0(2) == 0x40000000u);
    CHECK(sobol_dim1(1) == 0x80000000u);
    CHECK(sobol_dim1(2) == 0xc0000000u);
    CHECK(sobol_dim1(3) == 0x40000000u);

    // Dimension 0 is the van der Corput sequence: the index reversed, bit for
    // bit.
    for (uint32_t index = 1; index < 64u; ++index)
    {
        uint32_t reversed = 0;
        for (uint32_t bit = 0; bit < 32u; ++bit)
        {
            reversed |= ((index >> bit) & 1u) << (31u - bit);
        }
        CHECK(sobol_dim0(index) == reversed);
    }
}

namespace
{

double gridCoverage(SampleSlot a, SampleSlot b, uint32_t count)
{
    bool seen[256] = {};
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t x = sobol_padded_bits(index, a, 52u) >> 28u;
        const uint32_t y = sobol_padded_bits(index, b, 52u) >> 28u;
        seen[y * 16u + x] = true;
    }
    int filled = 0;
    for (const bool cell : seen)
    {
        filled += cell ? 1 : 0;
    }
    return filled / 256.0;
}

} // namespace

TEST_CASE("padding gives every pair of draws a two-dimensional projection")
{
    CHECK(gridCoverage({ 9u, 0u }, { 12u, 0u }, 4096u) > 0.95);
    CHECK(gridCoverage({ 3u, 0u }, { 4u, 0u }, 4096u) > 0.95);
    CHECK(gridCoverage({ 1u, 0u }, { 2u, 0u }, 4096u) > 0.95);

    // The two axes of one decision are the pair the (0,2)-sequence stratifies,
    // so they should cover it exactly: 256 samples, 256 cells, one each.
    CHECK(gridCoverage({ 0u, 0u }, { 0u, 1u }, 256u) == doctest::Approx(1.0));
    CHECK(gridCoverage({ 8u, 0u }, { 8u, 1u }, 256u) == doctest::Approx(1.0));
}
