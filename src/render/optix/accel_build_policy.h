#pragma once

#include <cstdint>

namespace oka::optix_accel
{

/// Mirrors of OptixBuildFlags. Checked against the SDK in OptixRender.cpp.
enum BuildFlag : uint32_t
{
    kFlagNone = 0u,
    kFlagAllowUpdate = 1u << 0,
    kFlagAllowCompaction = 1u << 1,
    kFlagPreferFastTrace = 1u << 2,
    kFlagPreferFastBuild = 1u << 3,
    kFlagAllowRandomVertexAccess = 1u << 4,
};

/// What is being built, which is what decides whether it is ever refit and
/// whether the shading path reads back its vertices.
enum class Geometry : int
{
    /// Triangles that never move. Built once, traced for the whole render.
    StaticMesh = 0,
    /// Triangles behind a skeleton. Refit every animated frame, with a bounded
    /// round-robin subset rebuilt to prevent quality drift.
    SkinnedMesh = 1,
    Curve = 2,
    /// The instance structure of a scene nothing animates.
    StaticTlas = 3,
    /// The instance structure of a scene that does, refit as transforms move.
    RefittableTlas = 4,
};

/// The flags for a full build of `geometry`.
inline uint32_t buildFlags(Geometry geometry)
{
    switch (geometry)
    {
    case Geometry::StaticMesh:
        return kFlagAllowCompaction | kFlagPreferFastTrace;
    case Geometry::SkinnedMesh:
        // Fast build because bounded round-robin rebuilds make build time a
        // recurring cost. No compaction: this structure is refit in place.
        return kFlagPreferFastBuild | kFlagAllowUpdate;
    case Geometry::Curve:
        return kFlagAllowCompaction | kFlagPreferFastTrace | kFlagAllowRandomVertexAccess;
    case Geometry::StaticTlas:
        return kFlagAllowCompaction | kFlagPreferFastTrace;
    case Geometry::RefittableTlas:
        return kFlagPreferFastTrace | kFlagAllowUpdate;
    }
    return kFlagNone;
}

/// The flags for an update of `geometry`. Identical to the build's by
/// construction -- this exists so the call site cannot spell them differently,
/// which is what it used to do.
inline uint32_t updateFlags(Geometry geometry)
{
    return buildFlags(geometry);
}

/// Whether `geometry` may be refit at all.
inline bool isRefittable(Geometry geometry)
{
    return (buildFlags(geometry) & kFlagAllowUpdate) != 0u;
}

inline bool shouldCompact(Geometry geometry)
{
    const uint32_t flags = buildFlags(geometry);
    return (flags & kFlagAllowCompaction) != 0u && (flags & kFlagAllowUpdate) == 0u;
}

} // namespace oka::optix_accel

