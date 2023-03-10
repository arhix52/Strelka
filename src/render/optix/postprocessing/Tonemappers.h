#pragma once
#include <sutil/vec_math.h>
#include <stdint.h>

enum class ToneMapperType : uint32_t
{
    eNone = 0,
    eReinhard,
    eACES,
    eFilmic,
};

extern "C" void tonemap(
    const ToneMapperType type, const float gamma, float4* image, const uint32_t width, const uint32_t height);
