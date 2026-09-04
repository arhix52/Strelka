// The Sobol' direction numbers, pinned.
//
// random.h is device code -- it is one of the three headers under shaders/common
// that are CUDA-only -- so the table it holds had no test at all, and the layout
// of that table is not a free choice: it is read in the sampler's inner loop,
// once per random number, by every path. Storing it [bit][dimension] rather than
// [dimension][bit] is worth 3-4% of the frame on iso_bathroom and chess_set,
// because a warp whose threads sit at different bounce depths then reads
// consecutive words instead of one 128-byte stride apart.
//
// Nothing about that changes a single value, and this is what says so. A layout
// or indexing mistake there does not crash and does not look wrong: it silently
// substitutes one Sobol' dimension for another, which is the failure the note
// above the table describes -- two dimensions that alias produce the *same
// number*, and the image is merely biased.
//
// The shim is what lets a host test read a device header. It is deliberately
// confined to this file, and clang-tidy is told so once rather than four times:
// these names are reserved because they belong to the CUDA compiler, which is
// exactly why they are the ones that have to be defined away here.
//
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

TEST_CASE("the Sobol' direction numbers survive the table's layout")
{
    // FNV-1a over the first 512 sample indices of all 256 dimensions: 131072
    // values, which is every entry of the table reachable from the sampler.
    // Taken from the layout this replaced, so it pins the values and not the
    // arrangement.
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t dimension = 0; dimension < 256u; ++dimension)
    {
        for (uint32_t index = 0; index < 512u; ++index)
        {
            hash = (hash ^ sobol_uint(index, dimension)) * 1099511628211ull;
        }
    }
    CHECK(hash == 0xfeff64cb2cb50383ull);

    // A few by hand, so a wrong hash says which end it is wrong at.
    CHECK(sobol_uint(1, 0) == 0x80000000u);
    CHECK(sobol_uint(2, 0) == 0x40000000u);
    CHECK(sobol_uint(2, 1) == 0xc0000000u);
    CHECK(sobol_uint(12345, 255) == 0xd1840000u);

    // Dimension 0 is the van der Corput sequence: index reversed, bit for bit.
    for (uint32_t index = 1; index < 64u; ++index)
    {
        uint32_t reversed = 0;
        for (uint32_t bit = 0; bit < 32u; ++bit)
        {
            reversed |= ((index >> bit) & 1u) << (31u - bit);
        }
        CAPTURE(index);
        CHECK(sobol_uint(index, 0) == reversed);
    }
}
