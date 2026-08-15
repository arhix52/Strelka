#include <doctest/doctest.h>

#include "accel_build_policy.h"

using namespace oka::optix_accel;

namespace
{
constexpr Geometry kAll[] = { Geometry::StaticMesh, Geometry::SkinnedMesh, Geometry::Curve, Geometry::StaticTlas,
                              Geometry::RefittableTlas };
}

TEST_CASE("an update uses the flags its build used")
{
    // The defect this table exists to prevent: the instance structure was built
    // with PREFER_FAST_TRACE | ALLOW_COMPACTION | ALLOW_UPDATE and then refit
    // with PREFER_FAST_BUILD | ALLOW_UPDATE, and a skinned mesh did the same.
    // optixAccelBuild reads an update's output buffer as the result of a full
    // build with the flags it is handed.
    for (const Geometry g : kAll)
    {
        CHECK(updateFlags(g) == buildFlags(g));
    }
}

TEST_CASE("nothing refittable is compacted")
{
    // optixAccelCompact leaves a compacted copy in a new buffer at a new size;
    // an update expects the build's own output at the build's own size. Wanting
    // both of a structure is the contradiction, not either one on its own.
    for (const Geometry g : kAll)
    {
        if (isRefittable(g))
        {
            CHECK_FALSE(shouldCompact(g));
            CHECK((buildFlags(g) & kFlagAllowCompaction) == 0u);
        }
    }
}

TEST_CASE("compaction is only asked for where it was allowed")
{
    for (const Geometry g : kAll)
    {
        if (shouldCompact(g))
        {
            CHECK((buildFlags(g) & kFlagAllowCompaction) != 0u);
        }
    }
}

TEST_CASE("only the geometry something refits pays for ALLOW_UPDATE")
{
    // A refittable structure is built with a topology that survives being
    // moved, rather than one built to be traced. Geometry that never deforms
    // should not carry that.
    CHECK_FALSE(isRefittable(Geometry::StaticMesh));
    CHECK_FALSE(isRefittable(Geometry::Curve));
    CHECK_FALSE(isRefittable(Geometry::StaticTlas));

    CHECK(isRefittable(Geometry::SkinnedMesh));
    CHECK(isRefittable(Geometry::RefittableTlas));
}

TEST_CASE("every class states a build preference, and only one")
{
    for (const Geometry g : kAll)
    {
        const uint32_t flags = buildFlags(g);
        const bool fastTrace = (flags & kFlagPreferFastTrace) != 0u;
        const bool fastBuild = (flags & kFlagPreferFastBuild) != 0u;
        CHECK(fastTrace != fastBuild);
    }
}

TEST_CASE("curves keep random vertex access, because the shader reads them back")
{
    // OptixRender_closest_hit.cu calls optixGetCubicBSplineVertexData to rebuild
    // the strand's frame. Dropping the flag to save memory would return garbage
    // control points rather than fail to compile.
    CHECK((buildFlags(Geometry::Curve) & kFlagAllowRandomVertexAccess) != 0u);

    // Nothing else reads vertices back, and the flag is not free.
    CHECK((buildFlags(Geometry::StaticMesh) & kFlagAllowRandomVertexAccess) == 0u);
    CHECK((buildFlags(Geometry::SkinnedMesh) & kFlagAllowRandomVertexAccess) == 0u);
}
