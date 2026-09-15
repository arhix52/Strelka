#pragma once

#include <log.h>

#include <cstdint>
#include <cstdlib>

namespace oka
{

/// below happens while the process is still setting itself up, so the NOLINTs

/// True when the variable is present, whatever its value.
inline bool envFlag(const char* name)
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    return std::getenv(name) != nullptr;
}

inline uint32_t envUint(const char* name, uint32_t fallback)
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
    // Not const: strtoll's out-parameter is char**, so a const char* here does
    // not convert. misc-const-correctness offers the fix anyway.
    // NOLINTNEXTLINE(misc-const-correctness)
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0)
    {
        STRELKA_WARNING("{}='{}' is not a non-negative integer, using {}", name, raw, fallback);
        return fallback;
    }
    return static_cast<uint32_t>(parsed);
}

inline double envDouble(const char* name, double fallback)
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
    // strtod's out-parameter is char**; see envUint above.
    // NOLINTNEXTLINE(misc-const-correctness)
    char* end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || *end != '\0')
    {
        STRELKA_WARNING("{}='{}' is not a number, using {}", name, raw, fallback);
        return fallback;
    }
    return parsed;
}

inline float envFloat(const char* name, float fallback)
{
    return static_cast<float>(envDouble(name, static_cast<double>(fallback)));
}

inline bool envBool(const char* name, bool fallback)
{
    return envUint(name, fallback ? 1u : 0u) != 0u;
}

} // namespace oka
