
#include <doctest/doctest.h>

#include <strelka/material/bsdf.h>

#include <cmath>

namespace
{

double directionalAlbedo(float roughness, float specular, float base, float cosV)
{
    SurfaceInteraction si = {};
    si.material_type = MATERIAL_TYPE_STANDARD_PBR;
    si.albedo = make_float3(base, base, base);
    si.roughness = roughness;
    si.metallic = 0.0f;
    si.specular = specular;
    si.specular_color = make_float3(1.0f, 1.0f, 1.0f);
    si.ior = 1.5f;
    si.exterior_ior = 1.0f;
    si.shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    si.geometry_normal = si.shading_normal;
    si.tangent = make_float3(1.0f, 0.0f, 0.0f);
    si.front_face = true;

    const float sinV = std::sqrt(std::max(0.0f, 1.0f - cosV * cosV));
    si.wo = make_float3(sinV, 0.0f, cosV);

    constexpr int kTheta = 400;
    constexpr int kPhi = 400;
    double sum = 0.0;
    for (int i = 0; i < kTheta; ++i)
    {
        const double theta = (i + 0.5) * (M_PI / 2) / kTheta;
        for (int j = 0; j < kPhi; ++j)
        {
            const double phi = (j + 0.5) * (2 * M_PI) / kPhi;
            const float3 wi = make_float3(static_cast<float>(std::sin(theta) * std::cos(phi)),
                                          static_cast<float>(std::sin(theta) * std::sin(phi)),
                                          static_cast<float>(std::cos(theta)));
            sum += bsdf_eval(si, wi).bsdf.x * std::cos(theta) * std::sin(theta);
        }
    }
    return sum * (M_PI / 2) / kTheta * (2 * M_PI) / kPhi;
}

} // namespace

TEST_CASE("the specular lobe is layered over the base rather than added to it")
{
    // Head on, where Schlick's grazing tail is zero and conservation is therefore
    // exactly testable. Before specular_base_scale() these read 1.042, 1.074 and
    // 1.080.
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        CHECK(directionalAlbedo(roughness, 1.0f, 1.0f, 1.0f) == doctest::Approx(1.0).epsilon(0.001));
    }
}

TEST_CASE("layering the base costs nothing where there is no lobe to layer under")
{
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        CHECK(directionalAlbedo(roughness, 0.0f, 1.0f, 1.0f) == doctest::Approx(1.0).epsilon(0.001));
    }
}

TEST_CASE("the specular weight no longer decides whether energy is conserved")
{
    // The split that named the defect: with it fixed, turning the lobe on and off
    // moves the albedo by well under a percent at every angle, instead of by 4-23%.
    for (const float roughness : { 0.85f, 0.50f, 0.20f })
    {
        for (const float cosV : { 1.0f, 0.7f, 0.3f })
        {
            const double on = directionalAlbedo(roughness, 1.0f, 1.0f, cosV);
            const double off = directionalAlbedo(roughness, 0.0f, 1.0f, cosV);
            CHECK(on == doctest::Approx(off).epsilon(0.01));
        }
    }
}

TEST_CASE("a grazing view creates energy even with the specular lobe switched off")
{
    CHECK(directionalAlbedo(0.85f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.015).epsilon(0.002));
    CHECK(directionalAlbedo(0.50f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.065).epsilon(0.002));
    CHECK(directionalAlbedo(0.20f, 0.0f, 1.0f, 0.3f) == doctest::Approx(1.161).epsilon(0.002));
}
