#pragma once

namespace oka::test
{

/// Centre of the index-th of `count` equal strata of [0, 1).
///
/// The suites drive samplers with a stratified sweep rather than a random
/// stream, so that a failure reproduces exactly. It is spelled once, here,
/// because the int-to-float conversion inside it has to be explicit -- an
/// implicit one is a narrowing conversion, which is the shape a real precision
/// bug takes, so it is worth keeping the analyzer able to complain about the
/// others -- and forty copies of a static_cast say less about what the number
/// means than the name does.
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
