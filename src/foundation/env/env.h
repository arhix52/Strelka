#pragma once

#include <log.h>

#include <cstdint>
#include <cstdlib>

namespace oka
{

/// Environment overrides, parsed in one place.
///
/// The benchmarks, audits and backend switches are all driven by STRELKA_*
/// variables, and every one of them used to be read with atoi(getenv(...)):
/// undefined behaviour when the variable is unset, and silently zero when it is
/// misspelled. Zero is a meaningful value for most of these knobs -- upscaling
/// off, depth 0, Metal 3 -- so a typo produced a measurement quietly taken with
/// settings nobody asked for. Parsing here reports it instead.

/// True when the variable is present, whatever its value. For the knobs that
/// select a mode by existing at all (STRELKA_BENCH, STRELKA_REF, ...).
inline bool envFlag(const char* name)
{
    return std::getenv(name) != nullptr;
}

inline uint32_t envUint(const char* name, uint32_t fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
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
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
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
