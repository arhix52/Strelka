#pragma once

// Which acceleration-structure build flags each class of geometry gets, and the
// two rules that stop the combinations from contradicting each other.
//
// The rules are not style. They come out of what optixAccelBuild documents:
//
//   1. An update reads the output buffer as "the result of a full build with
//      OPTIX_BUILD_FLAG_ALLOW_UPDATE set". So the flags handed to an update must
//      be the flags the structure was built with -- one table, read by both --
//      and a structure that is going to be updated must not have been compacted
//      first, because after optixAccelCompact the buffer holds a compacted copy
//      rather than the build's own output, at the compacted size.
//
//   2. ALLOW_UPDATE is not free. A refittable structure is built with a
//      topology that can survive being moved, which makes it larger and slower
//      to trace than one built for tracing. Asking for it on geometry nothing
//      ever refits is a permanent cost for a capability the renderer does not
//      use, so the flag follows the geometry class rather than being set
//      everywhere in case.
//
// The flag values are mirrored here rather than included from <optix_types.h>
// so that this file, and its tests, need no CUDA and no OptiX SDK.
// OptixRender.cpp static_asserts each mirror against the real enumerator, so a
// header change upstream is a build failure and not a silently wrong flag.

#include <cstdint>

namespace oka
{
namespace optix_accel
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
    /// Triangles behind a skeleton. Refit every animated frame, rebuilt every
    /// tenth to stop the refit's quality drifting.
    SkinnedMesh = 1,
    /// Curve segments. Never deform (skinning does not reach them) and the
    /// closest-hit program reads their control points back with
    /// optixGetCubicBSplineVertexData, which is what the random-vertex-access
    /// flag is for.
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
        // Fast build rather than fast trace: this one is rebuilt from scratch
        // every tenth animated frame, so its build time is a per-frame cost in a
        // way a static mesh's is not. No compaction -- it is refit in place.
        return kFlagPreferFastBuild | kFlagAllowUpdate;
    case Geometry::Curve:
        return kFlagAllowCompaction | kFlagPreferFastTrace | kFlagAllowRandomVertexAccess;
    case Geometry::StaticTlas:
        return kFlagAllowCompaction | kFlagPreferFastTrace;
    case Geometry::RefittableTlas:
        // Not compactable, and that is the point: an instance structure that is
        // refit cannot be compacted first. An IAS is a few dozen bytes per
        // instance, so what compaction would have saved here is small next to
        // rebuilding it whenever a transform moves.
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

/// Whether the renderer should compact `geometry` after building it.
///
/// Never for anything refittable: optixAccelCompact replaces the buffer with a
/// compacted copy, and an update expects the build's own output at the build's
/// own size.
inline bool shouldCompact(Geometry geometry)
{
    const uint32_t flags = buildFlags(geometry);
    return (flags & kFlagAllowCompaction) != 0u && (flags & kFlagAllowUpdate) == 0u;
}

} // namespace optix_accel
} // namespace oka
