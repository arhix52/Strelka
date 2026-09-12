#include <doctest/doctest.h>

#include <host/camera_ray_basis.h>

#include <glm/gtc/matrix_transform.hpp>

using oka::metal::perspectiveCameraRayBasis;

TEST_CASE("expanded perspective camera basis matches matrix unprojection")
{
    const glm::float4x4 view =
        glm::lookAt(glm::float3(2.0f, 3.0f, 5.0f), glm::float3(-1.0f, 0.5f, 0.0f), glm::float3(0.0f, 1.0f, 0.0f));
    const glm::float4x4 viewToWorld = glm::inverse(view);
    const glm::float4x4 clipToView = glm::inverse(glm::perspective(glm::radians(53.0f), 16.0f / 9.0f, 0.05f, 700.0f));
    const auto basis = perspectiveCameraRayBasis(viewToWorld, clipToView);

    for (const glm::float2 ndc :
         { glm::float2(-1.0f, -1.0f), glm::float2(0.0f), glm::float2(0.37f, -0.61f), glm::float2(1.0f) })
    {
        const glm::float4 viewDirection = clipToView * glm::float4(ndc.x, ndc.y, 1.0f, 1.0f);
        const glm::float3 matrixDirection =
            glm::normalize(glm::float3(viewToWorld * glm::float4(glm::float3(viewDirection), 0.0f)));
        const glm::float3 basisDirection = glm::normalize(basis.forward + ndc.x * basis.right + ndc.y * basis.up);
        CHECK(glm::length(matrixDirection - basisDirection) == doctest::Approx(0.0f).epsilon(1e-6));
    }
}
