#include <doctest/doctest.h>

#include "static_blas_grouping.h"

using oka::metal::groupStaticBlasItems;
using oka::metal::shouldBakeStaticMesh;
using oka::metal::StaticBlasItem;

TEST_CASE("only unique rigid static meshes are baked")
{
    CHECK(shouldBakeStaticMesh(1u, false, false));
    CHECK_FALSE(shouldBakeStaticMesh(2u, false, false));
    CHECK_FALSE(shouldBakeStaticMesh(1u, true, false));
    CHECK_FALSE(shouldBakeStaticMesh(1u, false, true));
}

TEST_CASE("unique static geometry is packed by triangle budget")
{
    const std::vector<StaticBlasItem> items = {
        { 0u, 4u, 1u, { 0.0f, 0.0f, 0.0f } },
        { 1u, 4u, 1u, { 1.0f, 0.0f, 0.0f } },
        { 2u, 4u, 1u, { 2.0f, 0.0f, 0.0f } },
    };

    const auto groups = groupStaticBlasItems(items, 8u);
    REQUIRE(groups.size() == 2u);
    CHECK(groups[0].size() == 2u);
    CHECK(groups[1].size() == 1u);
}

TEST_CASE("static BLAS groups never mix TLAS masks")
{
    const std::vector<StaticBlasItem> items = {
        { 7u, 1u, 1u, { 0.0f, 0.0f, 0.0f } },
        { 8u, 1u, 2u, { 0.1f, 0.0f, 0.0f } },
        { 9u, 1u, 1u, { 0.2f, 0.0f, 0.0f } },
    };

    const auto groups = groupStaticBlasItems(items, 100u);
    REQUIRE(groups.size() == 2u);
    CHECK(groups[0] == std::vector<uint32_t>{ 7u, 9u });
    CHECK(groups[1] == std::vector<uint32_t>{ 8u });
}

TEST_CASE("static BLAS ordering keeps nearby geometry together")
{
    const std::vector<StaticBlasItem> items = {
        { 0u, 1u, 1u, { 0.0f, 0.0f, 0.0f } },
        { 1u, 1u, 1u, { 100.0f, 0.0f, 0.0f } },
        { 2u, 1u, 1u, { 1.0f, 0.0f, 0.0f } },
        { 3u, 1u, 1u, { 101.0f, 0.0f, 0.0f } },
    };

    const auto groups = groupStaticBlasItems(items, 2u);
    REQUIRE(groups.size() == 2u);
    CHECK(groups[0] == std::vector<uint32_t>{ 0u, 2u });
    CHECK(groups[1] == std::vector<uint32_t>{ 1u, 3u });
}
