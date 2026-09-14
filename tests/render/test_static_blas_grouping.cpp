#include <doctest/doctest.h>

#include "static_blas_grouping.h"

using oka::metal::groupStaticBlasItems;
using oka::metal::hasUniformOrthogonalLinearPart;
using oka::metal::shouldBakeStaticMesh;
using oka::metal::StaticBlasItem;

TEST_CASE("only unique rigid static meshes are baked")
{
    CHECK(shouldBakeStaticMesh(1u, false, false));
    CHECK_FALSE(shouldBakeStaticMesh(2u, false, false));
    CHECK_FALSE(shouldBakeStaticMesh(1u, true, false));
    CHECK_FALSE(shouldBakeStaticMesh(1u, false, true));
}

TEST_CASE("uniform orthogonal transforms use the fast normal path")
{
    glm::mat4 rotatedScale(1.0f);
    rotatedScale[0] = glm::vec4(0.0f, 2.0f, 0.0f, 0.0f);
    rotatedScale[1] = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
    rotatedScale[2] = glm::vec4(0.0f, 0.0f, 2.0f, 0.0f);
    rotatedScale[3] = glm::vec4(3.0f, 4.0f, 5.0f, 1.0f);
    CHECK(hasUniformOrthogonalLinearPart(rotatedScale));

    rotatedScale[0].x = -rotatedScale[0].x;
    rotatedScale[0].y = -rotatedScale[0].y;
    CHECK(hasUniformOrthogonalLinearPart(rotatedScale));
}

TEST_CASE("general affine transforms retain the inverse transpose normal path")
{
    glm::mat4 nonUniform(1.0f);
    nonUniform[0].x = 2.0f;
    CHECK_FALSE(hasUniformOrthogonalLinearPart(nonUniform));

    glm::mat4 shear(1.0f);
    shear[1].x = 0.1f;
    CHECK_FALSE(hasUniformOrthogonalLinearPart(shear));

    glm::mat4 singular(1.0f);
    singular[2] = glm::vec4(0.0f);
    CHECK_FALSE(hasUniformOrthogonalLinearPart(singular));
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
