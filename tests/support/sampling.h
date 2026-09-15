#pragma once

namespace oka::test
{

constexpr float stratum(int index, int count)
{
    return (static_cast<float>(index) + 0.5f) / static_cast<float>(count);
}

/// The value a UNORM8 byte encodes: 0 maps to 0.0, 255 to 1.0.
constexpr float unorm8(int value)
{
    return static_cast<float>(value) / 255.0f;
}

} // namespace oka::test
