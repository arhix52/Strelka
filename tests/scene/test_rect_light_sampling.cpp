#include <doctest/doctest.h>

#include <strelka/scene/rect_light_sampling.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace oka::rect_light_sampling;

namespace
{

// Ceiling light above the origin, facing down (normal -Y).
RectCorners ceilingLight(float width = 2.0f, float height = 1.0f, float y = 2.0f)
{
    return fromWidthHeight(glm::float3(0.0f, y, 0.0f), width, height);
}

bool pointOnRect(const RectCorners& c, const glm::float3& p, float eps = 1e-4f)
{
    const glm::float3 e1 = c.p1 - c.p0;
    const glm::float3 e2 = c.p3 - c.p0;
    const float a = glm::dot(e1, e1);
    const float b = glm::dot(e1, e2);
    const float d = glm::dot(e2, e2);
    const glm::float3 rel = p - c.p0;
    const float e = glm::dot(rel, e1);
    const float f = glm::dot(rel, e2);
    const float det = a * d - b * b;
    if (std::abs(det) < 1e-12f)
    {
        return false;
    }
    const float u = (d * e - b * f) / det;
    const float v = (a * f - b * e) / det;
    const glm::float3 closest = c.p0 + e1 * u + e2 * v;
    return u >= -eps && u <= 1.0f + eps && v >= -eps && v <= 1.0f + eps && length3(p - closest) < eps;
}

struct EstimatorStats
{
    double mean = 0.0;
    double variance = 0.0;
};

template <typename Integrand>
EstimatorStats estimate(const RectCorners& c, const glm::float3& hit, bool solidAngle, int samples, Integrand&& integrand)
{
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    double sum = 0.0;
    double sumSq = 0.0;
    int used = 0;
    for (int i = 0; i < samples; ++i)
    {
        const glm::float2 uv(dist(rng), dist(rng));
        const LightSample s = solidAngle ? sampleRectSolidAngle(c, uv, hit) : sampleRectUniform(c, uv, hit);
        if (s.pdf <= 0.0f)
        {
            continue;
        }
        const double w = double(integrand(s)) / double(s.pdf);
        sum += w;
        sumSq += w * w;
        ++used;
    }
    EstimatorStats out;
    if (used == 0)
    {
        return out;
    }
    out.mean = sum / used;
    out.variance = std::max(0.0, sumSq / used - out.mean * out.mean);
    return out;
}

} // namespace

TEST_CASE("SphQuad solid angle matches Girard's theorem")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);

    const SphQuad squad = initSphQuad(c, hit);
    const float girard = solidAngleGirard(c, hit);

    REQUIRE(squad.S > 0.0f);
    CHECK(squad.S == doctest::Approx(girard).epsilon(1e-4));
    CHECK_FALSE(squad.useAreaFallback);
}

TEST_CASE("SphQuad solid angle shrinks with distance and grows with size")
{
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const float nearS = initSphQuad(ceilingLight(2.0f, 1.0f, 1.0f), hit).S;
    const float farS = initSphQuad(ceilingLight(2.0f, 1.0f, 4.0f), hit).S;
    const float wideS = initSphQuad(ceilingLight(4.0f, 2.0f, 2.0f), hit).S;
    const float narrowS = initSphQuad(ceilingLight(0.5f, 0.25f, 2.0f), hit).S;

    CHECK(nearS > farS);
    CHECK(wideS > narrowS);
}

TEST_CASE("SphQuad samples land on the rectangle plane and inside its bounds")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    const glm::float3 hit(0.2f, 0.0f, -0.3f);
    const SphQuad squad = initSphQuad(c, hit);
    REQUIRE(squad.S > 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const glm::float3 n = rectNormal(c);
    for (int i = 0; i < 256; ++i)
    {
        const glm::float3 p = sampleSphQuad(squad, dist(rng), dist(rng));
        CHECK(pointOnRect(c, p));
        // On the light plane.
        CHECK(std::abs(glm::dot(p - c.p0, n)) == doctest::Approx(0.0f).epsilon(1e-4));
    }
}

TEST_CASE("Solid-angle PDF is constant 1/S over the rectangle")
{
    const RectCorners c = ceilingLight(1.5f, 1.5f, 2.0f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const SphQuad squad = initSphQuad(c, hit);
    REQUIRE(squad.S > 1e-4f);

    for (float u : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
    {
        for (float v : { 0.0f, 0.5f, 1.0f })
        {
            const LightSample s = sampleRectSolidAngle(c, glm::float2(u, v), hit);
            CHECK(s.pdf == doctest::Approx(1.0f / squad.S).epsilon(1e-5));
            CHECK(rectLightPdf(c, s.pointOnLight, hit, true) == doctest::Approx(s.pdf).epsilon(1e-5));
        }
    }
}

TEST_CASE("Area-uniform PDF matches the area-to-solid-angle conversion")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const LightSample s = sampleRectUniform(c, glm::float2(0.3f, 0.7f), hit);

    const float expected = areaToSolidAnglePdf(s.pointOnLight, hit, s.normal, s.area);
    CHECK(s.pdf == doctest::Approx(expected).epsilon(1e-5));
    CHECK(rectLightPdf(c, s.pointOnLight, hit, false) == doctest::Approx(expected).epsilon(1e-5));
}

// The defining property of solid-angle sampling: the unbiased estimator of the
// solid angle itself is 1 with zero variance (pdf = 1/S, integrand = 1).
TEST_CASE("Solid-angle sampling estimates its own measure with near-zero variance")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const float S = initSphQuad(c, hit).S;

    const auto one = [](const LightSample&) { return 1.0f; };
    const EstimatorStats solid = estimate(c, hit, true, 4096, one);
    const EstimatorStats area = estimate(c, hit, false, 4096, one);

    CHECK(solid.mean == doctest::Approx(S).epsilon(1e-3));
    CHECK(solid.variance < 1e-6);

    // Area sampling is also unbiased for ∫ 1 dω = S, but noisy.
    CHECK(area.mean == doctest::Approx(S).epsilon(0.05));
    CHECK(area.variance > solid.variance);
}

// Classic stress case from the paper: shading point close to a large light.
// Area sampling's pdf ~ r²/(cos θ A) blows up near the surface; solid-angle
// stays flat. Estimating irradiance ∫ cosθ_surface dω should therefore be much
// quieter with SphQuad.
TEST_CASE("Solid-angle sampling beats area sampling on a close large light")
{
    const RectCorners c = ceilingLight(4.0f, 4.0f, 0.5f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const glm::float3 n(0.0f, 1.0f, 0.0f);

    const auto irradiance = [&](const LightSample& s) { return std::max(0.0f, glm::dot(s.L, n)); };

    const EstimatorStats solid = estimate(c, hit, true, 8192, irradiance);
    const EstimatorStats area = estimate(c, hit, false, 8192, irradiance);

    REQUIRE(solid.mean > 0.0);
    REQUIRE(area.mean > 0.0);
    // Both unbiased — means should agree.
    CHECK(solid.mean == doctest::Approx(area.mean).epsilon(0.05));
    // Variance is the whole point of the advanced method.
    CHECK(solid.variance < 0.5 * area.variance);
}

TEST_CASE("Paper appendix form and Cycles asin form agree on S for moderate sizes")
{
    // Cross-check: recompute S with the paper's acos path (Girard) vs Cycles asin.
    const RectCorners c = ceilingLight(1.0f, 1.0f, 2.0f);
    for (const glm::float3& hit :
         { glm::float3(0.0f, 0.0f, 0.0f), glm::float3(0.4f, 0.0f, 0.2f), glm::float3(-0.8f, 0.1f, 0.5f) })
    {
        const SphQuad squad = initSphQuad(c, hit);
        const float girard = solidAngleGirard(c, hit);
        if (girard > 1e-4f)
        {
            CHECK(squad.S == doctest::Approx(girard).epsilon(1e-4));
        }
    }
}

TEST_CASE("Edge-on and behind-the-plane lights are rejected or fall back")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    // Exactly on the light plane: z0 = 0 makes the acos form report ~2π and the
    // asin form trip the grazing fallback. Either way the sample path must not
    // trust 1/S.
    const SphQuad onPlane = initSphQuad(c, glm::float3(0.0f, 2.0f, 0.0f));
    CHECK((onPlane.S <= 1e-4f || onPlane.useAreaFallback));

    const SphQuad above = initSphQuad(c, glm::float3(0.0f, 3.0f, 0.0f));
    CHECK(std::isfinite(above.S));
}

TEST_CASE("MIS light PDF matches the NEE sample PDF for both strategies")
{
    const RectCorners c = ceilingLight(2.0f, 1.0f, 2.0f);
    const glm::float3 hit(0.1f, 0.0f, -0.2f);

    const LightSample solid = sampleRectSolidAngle(c, glm::float2(0.3f, 0.6f), hit);
    CHECK(rectLightPdf(c, solid.pointOnLight, hit, true) == doctest::Approx(solid.pdf).epsilon(1e-5));

    const LightSample area = sampleRectUniform(c, glm::float2(0.3f, 0.6f), hit);
    CHECK(rectLightPdf(c, area.pointOnLight, hit, false) == doctest::Approx(area.pdf).epsilon(1e-5));

    // Crossed strategies must disagree: that is the MIS bug Advanced used to hit.
    const float areaPdfOfSolidSample = rectLightPdf(c, solid.pointOnLight, hit, false);
    CHECK(std::abs(areaPdfOfSolidSample - solid.pdf) > 1e-3f * std::max(solid.pdf, 1e-6f));
}

TEST_CASE("Tiny lights fall back to area sampling rather than a bogus 1/S")
{
    // A millimetre-scale light at room distance: float32 SphQuad S cancels.
    const RectCorners c = ceilingLight(1e-4f, 1e-4f, 2.0f);
    const glm::float3 hit(0.0f, 0.0f, 0.0f);
    const SphQuad squad = initSphQuad(c, hit);
    CHECK(squad.useAreaFallback);

    const LightSample s = sampleRectSolidAngle(c, glm::float2(0.5f, 0.5f), hit);
    // Must look like the uniform path, not 1/S with a destroyed S.
    const LightSample u = sampleRectUniform(c, glm::float2(0.5f, 0.5f), hit);
    CHECK(s.pdf == doctest::Approx(u.pdf).epsilon(1e-4));
}
