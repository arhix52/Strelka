#include <doctest/doctest.h>

#include <rect_sampling.h>
#include <strelka/scene/glm_wrapper.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace
{

struct RectCorners
{
    float3 p0{};
    float3 p1{};
    float3 p3{};
};

struct LightSample
{
    float3 pointOnLight{};
    float3 L{};
    float3 normal{};
    float distToLight = 0.0f;
    float area = 0.0f;
    float pdf = 0.0f;
};

float length3(const float3& v)
{
    return std::sqrt(glm::dot(v, v));
}

RectCorners fromWidthHeight(const float3& center, float width, float height)
{
    const float halfWidth = 0.5f * width;
    const float halfHeight = 0.5f * height;
    return {
        center + float3(-halfWidth, 0.0f, halfHeight),
        center + float3(halfWidth, 0.0f, halfHeight),
        center + float3(-halfWidth, 0.0f, -halfHeight),
    };
}

float rectArea(const RectCorners& c)
{
    return length3(glm::cross(c.p1 - c.p0, c.p3 - c.p0));
}

float3 rectNormal(const RectCorners& c)
{
    return -glm::normalize(glm::cross(c.p1 - c.p0, c.p3 - c.p0));
}

float solidAngleGirard(const RectCorners& c, const float3& origin)
{
    const float3 v0 = glm::normalize(c.p0 - origin);
    const float3 v1 = glm::normalize(c.p1 - origin);
    const float3 v2 = glm::normalize(c.p1 + c.p3 - c.p0 - origin);
    const float3 v3 = glm::normalize(c.p3 - origin);
    const auto edgeNormal = [](const float3& a, const float3& b) { return glm::normalize(glm::cross(a, b)); };
    const float3 n0 = edgeNormal(v0, v1);
    const float3 n1 = edgeNormal(v1, v2);
    const float3 n2 = edgeNormal(v2, v3);
    const float3 n3 = edgeNormal(v3, v0);
    const auto angle = [](const float3& a, const float3& b) {
        return std::acos(std::clamp(-glm::dot(a, b), -1.0f, 1.0f));
    };
    return angle(n0, n1) + angle(n1, n2) + angle(n2, n3) + angle(n3, n0) - 2.0f * std::numbers::pi_v<float>;
}

SphQuad initSphQuad(const RectCorners& c, const float3& origin)
{
    return sphQuadInit(c.p0, c.p1 - c.p0, c.p3 - c.p0, origin);
}

float3 sampleSphQuad(const SphQuad& squad, float u, float v)
{
    return sphQuadSample(squad, u, v);
}

float areaToSolidAnglePdf(const float3& pointOnLight, const float3& hitPoint, const float3& lightNormal, float area)
{
    const float3 toLight = pointOnLight - hitPoint;
    const float distance = length3(toLight);
    if (distance < 1e-8f || area <= 0.0f)
    {
        return 0.0f;
    }
    const float3 direction = toLight / distance;
    const float cosine = glm::dot(-direction, lightNormal);
    return cosine > 0.0f ? distance * distance / (cosine * area) : 0.0f;
}

LightSample sampleRectUniform(const RectCorners& c, const float2& uv, const float3& hitPoint)
{
    LightSample sample;
    sample.pointOnLight = c.p0 + (c.p1 - c.p0) * uv.x + (c.p3 - c.p0) * uv.y;
    sample.area = rectArea(c);
    sample.normal = rectNormal(c);
    const float3 toLight = sample.pointOnLight - hitPoint;
    sample.distToLight = length3(toLight);
    sample.L = sample.distToLight > 1e-8f ? toLight / sample.distToLight : float3(0.0f);
    sample.pdf = areaToSolidAnglePdf(sample.pointOnLight, hitPoint, sample.normal, sample.area);
    return sample;
}

LightSample sampleRectSolidAngle(const RectCorners& c, const float2& uv, const float3& hitPoint)
{
    const SphQuad squad = initSphQuad(c, hitPoint);
    if (squad.S <= 0.0f)
    {
        LightSample sample = sampleRectUniform(c, uv, hitPoint);
        sample.pdf = 0.0f;
        return sample;
    }
    if (squad.useAreaFallback)
    {
        return sampleRectUniform(c, uv, hitPoint);
    }

    LightSample sample;
    sample.pointOnLight = sampleSphQuad(squad, uv.x, uv.y);
    sample.area = rectArea(c);
    sample.normal = rectNormal(c);
    const float3 toLight = sample.pointOnLight - hitPoint;
    sample.distToLight = length3(toLight);
    sample.L = sample.distToLight > 1e-8f ? toLight / sample.distToLight : float3(0.0f);
    sample.pdf = 1.0f / squad.S;
    return sample;
}

float rectLightPdf(const RectCorners& c, const float3& lightHitPoint, const float3& surfaceHitPoint, bool solidAngle)
{
    if (!solidAngle)
    {
        return areaToSolidAnglePdf(lightHitPoint, surfaceHitPoint, rectNormal(c), rectArea(c));
    }
    const SphQuad squad = initSphQuad(c, surfaceHitPoint);
    if (squad.S <= 0.0f)
    {
        return 0.0f;
    }
    return squad.useAreaFallback ? areaToSolidAnglePdf(lightHitPoint, surfaceHitPoint, rectNormal(c), rectArea(c)) :
                                   1.0f / squad.S;
}

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
EstimatorStats estimate(const RectCorners& c, const glm::float3& hit, bool solidAngle, int samples, const Integrand& integrand)
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

    for (const float u : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
    {
        for (const float v : { 0.0f, 0.5f, 1.0f })
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

TEST_CASE("large finite rectangles choose the exact area fallback")
{
    const float3 p0 = make_float3(0.0f, 0.0f, 0.0f);
    const float3 ex = make_float3(1e20f, 0.0f, 0.0f);
    const float3 ey = make_float3(0.0f, 1.0f, 0.0f);
    const float3 origin = make_float3(0.0f, 0.0f, 1.0f);
    const SphQuad squad = sphQuadInit(p0, ex, ey, origin);
    CHECK(squad.useAreaFallback);
    CHECK(squad.S > 0.0f);
    CHECK(std::isfinite(squad.S));
    CHECK_FALSE(std::isfinite(length(ex)));
    CHECK(finiteVectorLength(ex) == doctest::Approx(1e20f));
}

TEST_CASE("any represented rectangle shear uses the affine area sampler")
{
    const float3 p0 = make_float3(0.0f, 0.0f, 1.0f);
    const float3 ex = make_float3(1.0f, 0.0f, 0.0f);
    const float3 ey = make_float3(5e-7f, 1.0f, 0.0f);
    const SphQuad squad = sphQuadInit(p0, ex, ey, make_float3(0.0f));
    CHECK(squad.useAreaFallback);
    CHECK(dot(normalizeFiniteVectorOrZero(ex), normalizeFiniteVectorOrZero(ey)) != 0.0f);
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

TEST_CASE("A sheared affine rectangle uses its exact parallelogram sampler")
{
    const RectCorners c{ float3(0.0f, 0.0f, 1.0f), float3(1.0f, 0.0f, 1.0f), float3(1.0f, 1.0f, 1.0f) };
    const float3 hit(0.0f);
    const SphQuad squad = initSphQuad(c, hit);
    REQUIRE(squad.useAreaFallback);

    for (const float u : { 0.0f, 0.25f, 0.75f, 1.0f })
    {
        for (const float v : { 0.0f, 0.5f, 1.0f })
        {
            const LightSample sample = sampleRectSolidAngle(c, float2(u, v), hit);
            CHECK(pointOnRect(c, sample.pointOnLight));
            CHECK(sample.pointOnLight.z == doctest::Approx(1.0f));
            CHECK(rectLightPdf(c, sample.pointOnLight, hit, true) == doctest::Approx(sample.pdf).epsilon(1e-5));
        }
    }

    // Mutation: the old spherical-rectangle frame normalizes the sheared
    // edges independently and reconstructs a point at z=0.5, off the proxy.
    const float3 oldX = normalize(c.p1 - c.p0);
    const float3 oldY = normalize(c.p3 - c.p0);
    const float3 oldZ = cross(oldX, oldY);
    const float oldZ0 = dot(c.p0 - hit, oldZ);
    const float3 oldPoint = hit + oldZ0 * oldZ;
    CHECK(oldPoint.z == doctest::Approx(0.5f));
    CHECK_FALSE(pointOnRect(c, oldPoint));
}
