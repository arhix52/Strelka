#include <doctest/doctest.h>

#include <strelka/scene/transform.h>

#include <glm/gtx/matrix_decompose.hpp>

#include <cmath>

using namespace oka;

namespace
{

void checkRoundTrip(const glm::float4x4& matrix, float tolerance = 1e-4f)
{
    glm::float3 translation, scale;
    glm::quat rotation;
    decomposeTrs(matrix, translation, rotation, scale);

    const glm::float4x4 rebuilt = composeTrs(translation, rotation, scale);
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            INFO("column " << col << " row " << row);
            REQUIRE(std::isfinite(rebuilt[col][row]));
            CHECK(std::fabs(rebuilt[col][row] - matrix[col][row]) <= tolerance);
        }
    }
}

} // namespace

TEST_CASE("decomposeTrs round-trips a plain transform")
{
    const glm::float4x4 matrix = composeTrs(glm::float3(3.0f, -2.0f, 0.5f),
                                            glm::angleAxis(glm::radians(35.0f), glm::normalize(glm::float3(1, 2, 3))),
                                            glm::float3(2.0f, 0.5f, 1.5f));
    checkRoundTrip(matrix);
}

// The vespa asset scales its node by 0.003, which puts the determinant of the
// upper 3x3 at 2.7e-8 -- below glm's epsilon, so glm::decompose() refuses the
// matrix and writes nothing. This is the case that turned a gizmo drag into NaN.
TEST_CASE("decomposeTrs handles a scale glm::decompose refuses")
{
    const glm::float3 translation(1.2437f, -0.07f, -0.0804f);
    const glm::float4x4 matrix =
        composeTrs(translation, glm::angleAxis(glm::radians(20.0f), glm::float3(0, 1, 0)), glm::float3(0.003f));

    glm::float3 outTranslation, outScale;
    glm::quat outRotation;
    decomposeTrs(matrix, outTranslation, outRotation, outScale);

    CHECK(outTranslation.x == doctest::Approx(translation.x));
    CHECK(outTranslation.y == doctest::Approx(translation.y));
    CHECK(outTranslation.z == doctest::Approx(translation.z));
    CHECK(outScale.x == doctest::Approx(0.003f));
    CHECK(outScale.y == doctest::Approx(0.003f));
    CHECK(outScale.z == doctest::Approx(0.003f));
    CHECK(glm::degrees(glm::angle(glm::normalize(outRotation))) == doctest::Approx(20.0f).epsilon(0.001));
    checkRoundTrip(matrix, 1e-5f);

    // Pin the reason this helper exists: glm gives up on the same matrix.
    glm::float3 glmScale, glmTranslation, glmSkew;
    glm::quat glmRotation;
    glm::float4 glmPerspective;
    CHECK_FALSE(glm::decompose(matrix, glmScale, glmRotation, glmTranslation, glmSkew, glmPerspective));
}

TEST_CASE("decomposeTrs agrees with glm on a well conditioned matrix")
{
    const glm::float4x4 matrix =
        composeTrs(glm::float3(1.0f, 2.0f, 3.0f),
                   glm::angleAxis(glm::radians(50.0f), glm::normalize(glm::float3(0, 1, 1))), glm::float3(1.0f));

    glm::float3 translation, scale;
    glm::quat rotation;
    decomposeTrs(matrix, translation, rotation, scale);

    glm::float3 glmScale, glmTranslation, glmSkew;
    glm::quat glmRotation;
    glm::float4 glmPerspective;
    REQUIRE(glm::decompose(matrix, glmScale, glmRotation, glmTranslation, glmSkew, glmPerspective));

    // Same rotation convention, so the camera sites that conjugate the result keep
    // behaving as they did.
    CHECK(std::fabs(glm::dot(rotation, glmRotation)) == doctest::Approx(1.0f).epsilon(1e-4));
    CHECK(scale.x == doctest::Approx(glmScale.x).epsilon(1e-4));
    CHECK(translation.z == doctest::Approx(glmTranslation.z).epsilon(1e-4));
}

TEST_CASE("decomposeTrs reports a mirrored basis as negative scale")
{
    const glm::float4x4 matrix =
        composeTrs(glm::float3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::float3(-1.0f, 1.0f, 1.0f));

    glm::float3 translation, scale;
    glm::quat rotation;
    decomposeTrs(matrix, translation, rotation, scale);

    CHECK(scale.x < 0.0f);
    CHECK(scale.y == doctest::Approx(1.0f));
    // The rotation stays a rotation instead of absorbing the flip.
    CHECK(glm::determinant(glm::float3x3(glm::float4x4(rotation))) == doctest::Approx(1.0f).epsilon(1e-4));
    checkRoundTrip(matrix);
}

TEST_CASE("decomposeTrs stays finite for a collapsed axis")
{
    const glm::float4x4 matrix =
        composeTrs(glm::float3(1.0f, 1.0f, 1.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::float3(1.0f, 0.0f, 1.0f));

    glm::float3 translation, scale;
    glm::quat rotation;
    decomposeTrs(matrix, translation, rotation, scale);

    CHECK(scale.y == doctest::Approx(0.0f));
    CHECK(std::isfinite(rotation.w));
    CHECK(std::isfinite(rotation.x));
    CHECK(std::isfinite(rotation.y));
    CHECK(std::isfinite(rotation.z));
    CHECK(translation.x == doctest::Approx(1.0f));
}

// glTF stores [x,y,z,w]; GLM's constructor is (w,x,y,z). The conversion must
// not memcpy through glm::make_quat, whose layout changed between 0.9.9 and 1.0.
TEST_CASE("quatFromGltf maps glTF xyzw onto GLM's constructor")
{
    const glm::quat identity = quatFromGltf(0.0f, 0.0f, 0.0f, 1.0f);
    CHECK(identity.w == doctest::Approx(1.0f));
    CHECK(identity.x == doctest::Approx(0.0f));
    CHECK(identity.y == doctest::Approx(0.0f));
    CHECK(identity.z == doctest::Approx(0.0f));

    const float s = 0.70710678f;
    const glm::quat yaw = glm::normalize(quatFromGltf(0.0f, s, 0.0f, s));
    const glm::float3 forward = glm::normalize(yaw * glm::float3(0.0f, 0.0f, -1.0f));
    CHECK(forward.x == doctest::Approx(-1.0f).epsilon(1e-4));
    CHECK(forward.z == doctest::Approx(0.0f).epsilon(1e-4));
}
