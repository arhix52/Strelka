#pragma once

#include <bit>
#include <cstdint>


namespace oka::metal
{

// Invariant unsigned division as used by libdivide: the host derives one
// multiplier/shift pair when the render size changes, and the GPU replaces a
// dynamic divide with mulhi, shifts and (for difficult divisors) one add.
struct FastUnsignedDivisor
{
    uint32_t multiplier = 0;
    uint32_t shiftAdd = 0;
};

inline constexpr uint32_t kFastUnsignedDivisorAdd = 0x40u;

inline FastUnsignedDivisor makeFastUnsignedDivisor(uint32_t divisor)
{
    if (divisor == 0u)
    {
        return {};
    }

    const uint32_t floorLog2 = 31u - std::countl_zero(divisor);
    if ((divisor & (divisor - 1u)) == 0u)
    {
        return { 0u, floorLog2 };
    }

    const uint64_t numerator = uint64_t{ 1 } << (32u + floorLog2);
    uint64_t proposedMultiplier = numerator / divisor;
    const uint32_t remainder = static_cast<uint32_t>(numerator - proposedMultiplier * divisor);
    uint32_t shiftAdd = floorLog2;
    if (divisor - remainder >= (uint32_t{ 1 } << floorLog2))
    {
        proposedMultiplier *= 2u;
        if (uint64_t{ remainder } * 2u >= divisor)
        {
            ++proposedMultiplier;
        }
        shiftAdd |= kFastUnsignedDivisorAdd;
    }
    return { static_cast<uint32_t>(proposedMultiplier + 1u), shiftAdd };
}

inline uint32_t fastUnsignedDivide(uint32_t numerator, FastUnsignedDivisor divisor)
{
    if (divisor.multiplier == 0u)
    {
        return numerator >> divisor.shiftAdd;
    }
    uint32_t quotient = static_cast<uint32_t>((uint64_t{ numerator } * divisor.multiplier) >> 32u);
    if ((divisor.shiftAdd & kFastUnsignedDivisorAdd) != 0u)
    {
        quotient = ((numerator - quotient) >> 1u) + quotient;
    }
    return quotient >> (divisor.shiftAdd & 31u);
}

} // namespace oka::metal
