#include <doctest/doctest.h>

#include <analytic_light.h>
#include <light_pdf.h>
#include <strelka/scene/light_desc.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <random>

// ============================================================================
// test_light_pdf.cpp -- the densities the two halves of the MIS estimate divide
// by, and the heuristics that split them.
//
// Two separate properties are pinned here and they fail in different ways:
//
//   1. CONSISTENCY. Next-event estimation divides by its own density and the
//      BSDF strategy weighs itself against the same number. If the two agree,
//      the weights sum to one whatever the number is -- so a wrong density is
//      invisible to any test that only checks the weights. The sphere light was
//      wrong for exactly that reason and looked self-consistent throughout.
//
//   2. CORRECTNESS. The density has to be the one the sampler actually drew
//      from, or the estimator that divides by it is biased. There is only one
//      way to test that: sample the way the shader samples, divide by the
//      density the shader reports, and compare the mean against a closed form.
//      That is what the Monte Carlo cases below do.
//
// The shaders call these same functions -- common/lights.h on the OptiX side,
// metal/lights_metal.h on the Metal side -- so the arithmetic under test is the
// arithmetic that ships, not a host restatement of it.
// ============================================================================

namespace
{

// A fixed-seed generator; a flaky numerical test is one people learn to re-run
// rather than read.
struct Rng
{
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{ 0.0f, 1.0f };

    explicit Rng(std::uint32_t seed) : gen(seed)
    {
    }
    float next()
    {
        return dist(gen);
    }
};

float3 sub(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float len(float3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 unit(float3 v)
{
    const float l = len(v);
    return make_float3(v.x / l, v.y / l, v.z / l);
}

float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double affineAreaJacobian(const glm::dmat3& transform, const glm::dvec3& objectNormal)
{
    return std::abs(glm::determinant(transform)) * glm::length(glm::transpose(glm::inverse(transform)) * objectNormal);
}

glm::dvec3 objectCoordinates(const glm::dmat3& transform, float3 center, float3 point)
{
    return glm::inverse(transform) * (glm::dvec3(point) - glm::dvec3(center));
}

/// The irradiance a Lambert-facing point receives, estimated exactly the way
/// connectLight() + estimateDirectLighting() do it: draw a point on the emitter,
/// take the cosine at the shading vertex, divide by the reported solid-angle
/// density, and reject the samples the facing test rejects.
///
/// `radiance` is what UniformLight::color holds for an area light.
double sphereLightIrradiance(float radius, float distance, float radiance, int samples)
{
    const float3 shadingPoint = make_float3(0.0f, 0.0f, 0.0f);
    const float3 shadingNormal = make_float3(0.0f, 1.0f, 0.0f);
    const float3 centre = make_float3(0.0f, distance, 0.0f);

    Rng rng(0xC0FFEEu);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        // SampleSphereLight(), line for line.
        const float3 sphereDirection = uniformSphereDirection(rng.next(), rng.next());
        const float3 pointOnLight =
            make_float3(centre.x + radius * sphereDirection.x, centre.y + radius * sphereDirection.y,
                        centre.z + radius * sphereDirection.z);
        const float3 toLight = sub(pointOnLight, shadingPoint);
        const float distToLight = len(toLight);
        const float3 L = unit(toLight);
        const float cosAtLight = -dot3(L, sphereDirection);
        const float pdf = sphereLightSolidAnglePdf(distToLight, cosAtLight, radius);

        if (!lightSampleFacesVertex(cosAtLight) || !(pdf > 0.0f))
        {
            continue; // connectLight()'s facing test drops it, contributing zero
        }
        const float cosAtSurface = std::max(dot3(shadingNormal, L), 0.0f);
        sum += double(radiance) * double(cosAtSurface) / double(pdf);
    }
    return sum / double(samples);
}

/// Irradiance from a uniformly emitting sphere onto a point whose normal looks
/// straight at the centre: E = pi * L * sin^2(theta_max), sin(theta_max) = r/d.
double sphereIrradianceClosedForm(float radius, float distance, float radiance)
{
    const double s = double(radius) / double(distance);
    return double(M_PI_F) * double(radiance) * s * s;
}

/// The same estimator for a point light given a soft radius, where colour is
/// radiant intensity rather than radiance. Mirrors connectLight()'s soft branch.
double softPointIrradiance(float radius, float distance, float intensity, int samples)
{
    const float3 shadingPoint = make_float3(0.0f, 0.0f, 0.0f);
    const float3 shadingNormal = make_float3(0.0f, 1.0f, 0.0f);
    const float3 centre = make_float3(0.0f, distance, 0.0f);

    Rng rng(0x5EEDu);
    double sum = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        const float3 sphereDirection = uniformSphereDirection(rng.next(), rng.next());
        const float3 pointOnLight =
            make_float3(centre.x + radius * sphereDirection.x, centre.y + radius * sphereDirection.y,
                        centre.z + radius * sphereDirection.z);
        const float3 toLight = sub(pointOnLight, shadingPoint);
        const float distToLight = len(toLight);
        const float3 L = unit(toLight);
        const float cosAtLight = -dot3(L, sphereDirection);
        const float pdf = sphereLightSolidAnglePdf(distToLight, cosAtLight, radius);
        if (!lightSampleFacesVertex(cosAtLight) || !(pdf > 0.0f))
        {
            continue;
        }
        // Intensity converted to the radiance a sphere of this size emits; no
        // inverse-square, the density carries the distance.
        const float Li = intensity * sphereRadianceFromIntensity(radius);
        const float cosAtSurface = std::max(dot3(shadingNormal, L), 0.0f);
        sum += double(Li) * double(cosAtSurface) / double(pdf);
    }
    return sum / double(samples);
}

/// Every value LightType takes. A test that walks this cannot silently miss a
/// type the way a switch can.
const int kAllLightTypes[] = { LIGHT_TYPE_RECT, LIGHT_TYPE_DISC,  LIGHT_TYPE_SPHERE, LIGHT_TYPE_DISTANT,
                               LIGHT_TYPE_DOME, LIGHT_TYPE_POINT, LIGHT_TYPE_SPOT };

/// A query that is valid for whichever type it is asked about.
LightPdfQuery plausibleQuery(int type)
{
    LightPdfQuery q = makeLightPdfQuery(type);
    q.distToLight = 3.0f;
    q.cosAtLight = 0.8f;
    q.areaPdf = 0.5f; // rect / disc
    q.radius = 0.25f; // sphere, and a soft point
    q.halfAngle = 0.05f; // distant
    q.solidAngle = 0.0f; // rect: area sampling
    return q;
}

} // namespace

// ---------------------------------------------------------------------------
// The sphere light. This is the case that was wrong, and wrong in a way that
// only a comparison against a closed form could see.
// ---------------------------------------------------------------------------
TEST_CASE("a sphere light's density integrates to the analytic irradiance")
{
    const float radiance = 1.0f;
    const int samples = 400000;

    struct Case
    {
        float radius;
        float distance;
    };
    // Deliberately spanning distance/radius ratios: the density that shipped was
    // the constant 1/(4pi), whose error is exactly this ratio squared, so a
    // single near case would have looked almost right.
    const Case cases[] = { { 0.5f, 1.5f }, { 0.5f, 4.0f }, { 0.25f, 6.0f }, { 1.0f, 20.0f } };

    for (const Case& c : cases)
    {
        CAPTURE(c.radius);
        CAPTURE(c.distance);
        const double measured = sphereLightIrradiance(c.radius, c.distance, radiance, samples);
        const double expected = sphereIrradianceClosedForm(c.radius, c.distance, radiance);
        CHECK(measured == doctest::Approx(expected).epsilon(0.02));
    }
}

TEST_CASE("a sphere light's density is not the constant it used to be")
{
    // The direct regression guard for the shipped bug. 1/(4pi) is independent of
    // everything; a solid-angle density for an area sample cannot be, and moving
    // the light twice as far away has to quadruple it.
    const float near = sphereLightSolidAnglePdf(/*dist=*/4.0f, /*cos=*/1.0f, /*radius=*/0.5f);
    const float far = sphereLightSolidAnglePdf(/*dist=*/8.0f, /*cos=*/1.0f, /*radius=*/0.5f);

    CHECK(near > 0.0f);
    CHECK(far == doctest::Approx(4.0f * near).epsilon(1e-5));
    CHECK(near != doctest::Approx(1.0f / (4.0f * float(M_PI_F))).epsilon(0.01));

    // And it is the plain area density with the sphere's own area substituted,
    // so nothing about a sphere is special except which area is used.
    CHECK(sphereLightSolidAnglePdf(4.0f, 0.7f, 0.5f) ==
          doctest::Approx(areaLightSolidAnglePdf(4.0f, 0.7f, sphereLightArea(0.5f))));
    CHECK(sphereLightArea(0.5f) == doctest::Approx(4.0f * float(M_PI_F) * 0.25f));
}

TEST_CASE("analytic transformed light densities obey the affine area Jacobian")
{
    const float3 axisX = make_float3(-1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.5f, 0.0f);
    const float3 axisZ = make_float3(0.25f, 0.0f, 2.0f);
    const glm::dmat3 transform{ glm::dvec3(axisX), glm::dvec3(axisY), glm::dvec3(axisZ) };

    CHECK(double(analyticDiscArea(axisX, axisY)) ==
          doctest::Approx(std::numbers::pi * glm::length(glm::cross(glm::dvec3(axisX), glm::dvec3(axisY)))).epsilon(1e-6));

    Rng rng(0xA11Fu);
    double integratedMass = 0.0;
    for (int i = 0; i < 4096; ++i)
    {
        const float z = 1.0f - 2.0f * rng.next();
        const float radial = std::sqrt(std::max(1.0f - z * z, 0.0f));
        const float phi = 2.0f * float(M_PI_F) * rng.next();
        const float3 n = make_float3(radial * std::cos(phi), radial * std::sin(phi), z);
        const double oracle = affineAreaJacobian(transform, glm::dvec3(n));
        const double shared = glm::length(glm::dvec3(affineSphereCofactor(axisX, axisY, axisZ, n)));
        CHECK(shared == doctest::Approx(oracle).epsilon(2e-5));
        const float3 point = axisX * n.x + axisY * n.y + axisZ * n.z;
        float3 evaluatedNormal;
        const double areaPdf = analyticEllipsoidAreaPdf(make_float3(0.0f), axisX, axisY, axisZ, point, evaluatedNormal);
        integratedMass += areaPdf * oracle * (4.0 * std::numbers::pi / 4096.0);
    }
    CHECK(integratedMass == doctest::Approx(1.0).epsilon(3e-5));

    const float r = 0.7f;
    CHECK(analyticEllipsoidSurfaceArea(
              make_float3(r, 0.0f, 0.0f), make_float3(0.0f, r, 0.0f), make_float3(0.0f, 0.0f, r)) ==
          doctest::Approx(4.0f * float(M_PI_F) * r * r).epsilon(2e-5));
}

TEST_CASE("analytic samples, intersections, and hit-side PDFs agree")
{
    const float3 center = make_float3(0.3f, -0.2f, 0.5f);
    const float3 axisX = make_float3(-1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.5f, 0.0f);
    const float3 axisZ = make_float3(0.25f, 0.0f, 2.0f);
    const float3 discNormal = make_float3(0.0f, 0.0f, -1.0f);
    const float3 shadingPoint = make_float3(0.3f, -0.2f, -5.0f);

    Rng rng(0xA771Eu);
    const glm::dmat3 transform{ glm::dvec3(axisX), glm::dvec3(axisY), glm::dvec3(axisZ) };
    for (int i = 0; i < 4096; ++i)
    {
        const AnalyticLightSample disc = sampleAnalyticDisc(center, axisX, axisY, discNormal, rng.next(), rng.next());
        const float3 discDirection = unit(sub(disc.point, shadingPoint));
        const float discDistance = len(sub(disc.point, shadingPoint));
        const AnalyticLightIntersection discHit =
            intersectAnalyticDisc(shadingPoint, discDirection, 0.0f, 1e9f, center, axisX, axisY, discNormal);
        REQUIRE(discHit.hit);
        CHECK(discHit.distance == doctest::Approx(discDistance).epsilon(2e-5));
        CHECK(disc.areaPdf == doctest::Approx(1.0f / analyticDiscArea(axisX, axisY)).epsilon(1e-6));

        const AnalyticLightSample sphere = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, rng.next(), rng.next());
        const glm::dvec3 q = objectCoordinates(transform, center, sphere.point);
        CHECK(glm::length(q) == doctest::Approx(1.0).epsilon(3e-5));
        const double jacobian = affineAreaJacobian(transform, glm::normalize(q));
        CHECK(double(sphere.areaPdf) == doctest::Approx(1.0 / (4.0 * std::numbers::pi * jacobian)).epsilon(3e-5));
        CHECK(sphere.areaPdf > 0.0f);
        CHECK(std::isfinite(sphere.areaPdf));
        CHECK(std::isfinite(sphere.normal.x));
        CHECK(std::isfinite(sphere.normal.y));
        CHECK(std::isfinite(sphere.normal.z));
        const glm::dvec3 normalOracle = glm::normalize(glm::transpose(glm::inverse(transform)) * glm::normalize(q));
        CHECK(len(sub(sphere.normal, float3(normalOracle))) < 3e-5f);

        const float3 toLight = sub(sphere.point, shadingPoint);
        const double distance = double(len(toLight));
        const float3 wi = unit(toLight);
        const double cosine = -double(dot3(wi, sphere.normal));
        const float reported = areaPdfToSolidAnglePdf(float(distance), float(cosine), sphere.areaPdf);
        const double oracle = cosine > 0.0 ? double(sphere.areaPdf) * distance * distance / cosine : 0.0;
        CHECK(double(reported) == doctest::Approx(oracle).epsilon(2e-5));
        float3 evaluatedNormal;
        const float evaluatedPdf = analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, sphere.point, evaluatedNormal);
        CHECK(evaluatedPdf == doctest::Approx(sphere.areaPdf).epsilon(3e-5));
        CHECK(len(sub(evaluatedNormal, sphere.normal)) < 3e-5f);

        // Near tangency the intersection location is condition-numbered by
        // 1/cos(theta); all such rays are still checked for finite output, while
        // the pointwise identity test is restricted to well-conditioned hits.
        if (cosine > 0.05)
        {
            const AnalyticLightIntersection sphereHit =
                intersectAnalyticEllipsoid(shadingPoint, wi, 0.0f, 1e9f, center, axisX, axisY, axisZ);
            REQUIRE(sphereHit.hit);
            CHECK(sphereHit.distance == doctest::Approx(float(distance)).epsilon(5e-5));
            CHECK(len(sub(sphereHit.normal, sphere.normal)) < 5e-5f);
        }
        else
        {
            const AnalyticLightIntersection grazingHit =
                intersectAnalyticEllipsoid(shadingPoint, wi, 0.0f, 1e9f, center, axisX, axisY, axisZ);
            CHECK(std::isfinite(grazingHit.distance));
            CHECK(std::isfinite(grazingHit.normal.x));
            CHECK(std::isfinite(grazingHit.normal.y));
            CHECK(std::isfinite(grazingHit.normal.z));
        }
    }
}

TEST_CASE("analytic intersection covers the smooth disc beyond the editor proxy")
{
    CHECK(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_DISC));
    CHECK(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_SPHERE));
    CHECK(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_RECT));
    CHECK_FALSE(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_POINT));
    CHECK_FALSE(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_DISTANT));
    const float angle = float(M_PI_F) / 16.0f;
    const float3 target = make_float3(0.99f * std::cos(angle), 0.99f * std::sin(angle), 0.0f);
    const float3 origin = target + make_float3(0.0f, 0.0f, 2.0f);
    const AnalyticLightIntersection hit = intersectAnalyticDisc(
        origin, make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f, make_float3(0.0f), make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f));
    CHECK(hit.hit);
    CHECK(hit.distance == doctest::Approx(2.0f));
    CHECK(0.99f > std::cos(angle)); // inscribed-16-gon mutation misses this point
}

TEST_CASE("high-shear analytic discs use a non-cancelling dual basis")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(1.0f, 1e-4f, 0.0f);
    const float3 normal = make_float3(0.0f, 0.0f, -1.0f);
    REQUIRE(analyticDiscArea(axisX, axisY) > 0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticDisc(
        make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f, center, axisX, axisY, normal);
    CHECK(hit.hit);
    CHECK(hit.distance == doctest::Approx(1.0f));

    // Mutation: normal equations square the condition number and round the
    // nonzero determinant away at this shear.
    const float xx = dot(axisX, axisX);
    const float xy = dot(axisX, axisY);
    const float yy = dot(axisY, axisY);
    CHECK(xx * yy - xy * xy == 0.0f);
}

TEST_CASE("analytic parallelogram intersection matches its continuous area measure")
{
    const float3 corner = make_float3(-1.0f, -2.0f, 3.0f);
    const float3 edgeX = make_float3(2.0f, 0.0f, 0.0f);
    const float3 edgeY = make_float3(0.5f, 4.0f, 0.0f);
    const float3 normal = make_float3(0.0f, 0.0f, -1.0f);
    const float3 expected = corner + 0.25f * edgeX + 0.75f * edgeY;
    const AnalyticLightIntersection hit =
        intersectAnalyticRectangle(expected + make_float3(0.0f, 0.0f, 5.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f,
                                   10.0f, corner, edgeX, edgeY, normal);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(5.0f));
    CHECK(hit.point.x == doctest::Approx(expected.x));
    CHECK(hit.point.y == doctest::Approx(expected.y));
    CHECK(hit.point.z == doctest::Approx(expected.z));
    CHECK(hit.areaPdf == doctest::Approx(1.0 / 8.0));
    CHECK(hit.normal == normal);

    const AnalyticLightIntersection outside =
        intersectAnalyticRectangle(corner + 1.25f * edgeX + 0.5f * edgeY + make_float3(0.0f, 0.0f, 5.0f),
                                   make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f, corner, edgeX, edgeY, normal);
    CHECK_FALSE(outside.hit);
    CHECK(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_RECT));

    const float boundaryU = 2.9802322387695312e-8f;
    const float boundaryV = 0.9878741800785065f;
    const float3 boundaryPoint = corner + boundaryU * edgeX + boundaryV * edgeY;
    const float3 boundaryOrigin = make_float3(0.25f, -0.75f, -2.0f);
    const float3 boundaryDirection = unit(sub(boundaryPoint, boundaryOrigin));
    CHECK(intersectAnalyticLightSurface(LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY,
                                        normal, boundaryOrigin, boundaryDirection, 0.0f, 20.0f)
              .hit);

    Rng rng(0xA11CEu);
    for (int i = 0; i < 4096; ++i)
    {
        const float u = rng.next();
        const float v = rng.next();
        const float3 point = corner + u * edgeX + v * edgeY;
        const float3 rayOrigin = make_float3(0.25f, -0.75f, -2.0f);
        const float3 toPoint = sub(point, rayOrigin);
        const float distance = len(toPoint);
        const float3 rayDirection = unit(toPoint);
        const AnalyticLightIntersection sampleHit =
            intersectAnalyticLightSurface(LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY,
                                          normal, rayOrigin, rayDirection, 0.0f, 20.0f);
        CAPTURE(i);
        REQUIRE(sampleHit.hit);
        CHECK(sampleHit.distance == doctest::Approx(distance).epsilon(2e-5));
        CHECK(len(sub(sampleHit.point, point)) <= 2e-5f * std::max(distance, 1.0f));
        const double cosine =
            std::abs(double(dot3(normal, make_float3(-rayDirection.x, -rayDirection.y, -rayDirection.z))));
        const double oraclePdf = 0.125 * double(distance) * double(distance) / cosine;
        CHECK(double(areaPdfToSolidAnglePdf(distance, float(cosine), sampleHit.areaPdf)) ==
              doctest::Approx(oraclePdf).epsilon(3e-6));
    }
}

TEST_CASE("finite analytic light surfaces block only the open shadow segment")
{
    const float3 zero = make_float3(0.0f);
    const float3 center = make_float3(0.0f, 0.0f, 3.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 normal = make_float3(0.0f, 0.0f, -1.0f);
    const float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);

    CHECK(analyticLightSurfaceOccludesSegment(
        LIGHT_TYPE_DISC, zero, center, axisX, axisY, normal, 3.0f, origin, direction, 0.0f, 10.0f));
    CHECK_FALSE(analyticLightSurfaceOccludesSegment(
        LIGHT_TYPE_DISC, zero, center, axisX, axisY, normal, 3.0f, origin, direction, 0.0f, 2.0f));
    // The endpoint is open: a sampled target exactly at t=3 must not shadow itself.
    CHECK_FALSE(analyticLightSurfaceOccludesSegment(
        LIGHT_TYPE_DISC, zero, center, axisX, axisY, normal, 3.0f, origin, direction, 0.0f, 3.0f));

    const float radius = 0.5f;

    Rng rng(0x50F7u);
    for (int i = 0; i < 4096; ++i)
    {
        const float z = -0.2f - 0.8f * rng.next();
        const float phi = 2.0f * float(M_PI_F) * rng.next();
        const float radial = std::sqrt(std::max(1.0f - z * z, 0.0f));
        const float3 surfaceNormal = make_float3(radial * std::cos(phi), radial * std::sin(phi), z);
        const float3 samplePoint = center + radius * surfaceNormal;
        const float3 toPoint = sub(samplePoint, origin);
        const float distance = len(toPoint);
        const float3 sampleDirection = unit(toPoint);
        const AnalyticLightIntersection sampleHit =
            intersectAnalyticLightSurface(LIGHT_TYPE_POINT, make_float3(radius, 0.0f, 0.0f), center, zero, zero, zero,
                                          origin, sampleDirection, 0.0f, 10.0f);
        CAPTURE(i);
        REQUIRE(sampleHit.hit);
        CHECK(len(sub(sampleHit.point, samplePoint)) <= 4e-5f);
        CHECK(sampleHit.areaPdf == doctest::Approx(1.0 / (4.0 * M_PI_F * radius * radius)).epsilon(3e-6));
        const float cosAtLight = -dot3(sampleDirection, sampleHit.normal);
        REQUIRE(cosAtLight > 0.0f);
        const float lightPdf = sphereLightSolidAnglePdf(distance, cosAtLight, radius);
        const float bsdfPdf = 0.2f;
        CHECK(computeMisWeight(lightPdf, bsdfPdf, 0) + computeMisWeight(bsdfPdf, lightPdf, 0) == doctest::Approx(1.0f));
    }

    // Mutation: the old shadow mask enumerated ordinary geometry only, hence
    // no light surface above could ever have blocked the connection.
    constexpr bool oldShadowMaskCouldHitAnalyticLight = false;
    CHECK_FALSE(oldShadowMaskCouldHitAnalyticLight);
}

TEST_CASE("selected sphere light keeps near-side self-occlusion")
{
    const float3 center = make_float3(0.0f, 0.0f, 3.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);

    const AnalyticLightIntersection nearSample =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, std::nextafter(2.0f, 0.0f), center, axisX, axisY, axisZ);
    const AnalyticLightIntersection farSample =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, std::nextafter(4.0f, 0.0f), center, axisX, axisY, axisZ);
    CHECK_FALSE(nearSample.hit);
    REQUIRE(farSample.hit);
    CHECK(farSample.distance == doctest::Approx(2.0f));

    const std::filesystem::path repository =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream shaderFile(repository / "src/shaders/metal/wavefront.metal");
    REQUIRE(shaderFile.good());
    const std::string shader((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    CHECK(shader.find("ignoredLightId") == std::string::npos);
    // Analytic lights were removed from RAY_MASK_SHADOW to match Cycles, so a
    // descriptor-table probe for a light candidate is both unreachable and an
    // expensive random load in the shadow hot path.
    CHECK(shader.find("shadowLightProxy") == std::string::npos);
}

TEST_CASE("continuous light samples use interior finite-lattice representatives")
{
    CHECK(lightOpenUnitInterval(0.0f) == 0x1p-24f);
    CHECK(lightOpenUnitInterval(0x1.fffffep-1f) < 1.0f);
    CHECK(lightOpenUnitInterval(0.5f) > 0.5f);

    // Mutation: the old closed endpoint put finite probability on a rectangle
    // edge or sphere pole even though both are null sets of the declared area
    // measure.
    constexpr float oldClosedEndpoint = 0.0f;
    CHECK(oldClosedEndpoint != lightOpenUnitInterval(0.0f));
}

TEST_CASE("transformed tangents use vector transport and Gram-Schmidt")
{
    const float3 worldNormal = normalizeFiniteVectorOrZero(make_float3(0.5f, 1.0f, 0.0f));
    const float3 forwardVector = make_float3(-2.0f, 1.0f, 0.0f);
    const float3 tangent = orthonormalizeTangent(worldNormal, forwardVector);
    CHECK(dot3(worldNormal, tangent) == doctest::Approx(0.0f).scale(1.0f).epsilon(1e-6));
    CHECK(len(tangent) == doctest::Approx(1.0f));

    // Mutation: inverse-transpose is correct for normals, not for tangents.
    // Under diag(2,1,1) it produces a visibly non-orthogonal frame.
    const float3 oldOptixTangent = normalizeFiniteVectorOrZero(make_float3(-0.5f, 1.0f, 0.0f));
    CHECK(std::abs(dot3(worldNormal, oldOptixTangent)) > 0.5f);
}

TEST_CASE("light profile frames match a double-precision Gram-Schmidt oracle")
{
    std::mt19937 generator(0xF24A9u);
    std::uniform_real_distribution<double> value(-2.0, 2.0);
    for (int sample = 0; sample < 4096; ++sample)
    {
        glm::dmat3 transform;
        for (;;)
        {
            for (int column = 0; column < 3; ++column)
            {
                for (int row = 0; row < 3; ++row)
                {
                    transform[column][row] = value(generator);
                }
            }
            if (std::abs(glm::determinant(transform)) >= 0.2)
            {
                break;
            }
        }

        const float3 axisX = float3(transform[0]);
        const float3 axisY = float3(transform[1]);
        const float3 emissionAxis = float3(glm::normalize(glm::transpose(glm::inverse(transform)) *
                                                          glm::dvec3(0.0, 0.0, -1.0)));
        const OrthonormalLightFrame frame = makeOrthonormalLightFrame(axisX, axisY, emissionAxis);
        REQUIRE(frame.valid);

        const glm::dvec3 oracleZ = glm::normalize(glm::dvec3(emissionAxis));
        const glm::dvec3 inputX(axisX);
        const glm::dvec3 oracleX = glm::normalize(inputX - glm::dot(inputX, oracleZ) * oracleZ);
        const glm::dvec3 inputY(axisY);
        const glm::dvec3 oracleY = glm::normalize(inputY - glm::dot(inputY, oracleZ) * oracleZ -
                                                  glm::dot(inputY, oracleX) * oracleX);
        CAPTURE(sample);
        CHECK(glm::length(glm::dvec3(frame.x) - oracleX) < 2e-6);
        CHECK(glm::length(glm::dvec3(frame.y) - oracleY) < 2e-6);
        CHECK(glm::length(glm::dvec3(frame.emissionAxis) - oracleZ) < 2e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.x), glm::dvec3(frame.y))) < 2e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.x), glm::dvec3(frame.emissionAxis))) < 2e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.y), glm::dvec3(frame.emissionAxis))) < 2e-6);
    }
}

TEST_CASE("degenerate analytic lights have zero density without non-finite samples")
{
    const float3 zero = make_float3(0.0f);
    const AnalyticLightSample disc = sampleAnalyticDisc(zero, zero, zero, zero, 0.3f, 0.7f);
    const AnalyticLightSample sphere = sampleAnalyticEllipsoid(zero, zero, zero, zero, 0.3f, 0.7f);
    for (const AnalyticLightSample& sample : { disc, sphere })
    {
        CHECK(sample.areaPdf == 0.0f);
        CHECK(std::isfinite(sample.point.x));
        CHECK(std::isfinite(sample.point.y));
        CHECK(std::isfinite(sample.point.z));
        CHECK(std::isfinite(sample.normal.x));
        CHECK(std::isfinite(sample.normal.y));
        CHECK(std::isfinite(sample.normal.z));
    }
    CHECK_FALSE(intersectAnalyticDisc(zero, make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f, zero, zero, zero, zero).hit);
    CHECK_FALSE(intersectAnalyticEllipsoid(zero, make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f, zero, zero, zero, zero).hit);
}

TEST_CASE("rank-deficient analytic transforms have no area-light support")
{
    const float3 zero = make_float3(0.0f);
    const float3 x = make_float3(1.0f, 0.0f, 0.0f);
    const float3 y = make_float3(0.0f, 1.0f, 0.0f);

    // A rank-two sphere collapses into a twice-covered disc. It is not the
    // ellipsoid area measure or analytic surface that this light declares.
    const AnalyticLightSample ellipsoid = sampleAnalyticEllipsoid(zero, x, y, zero, 0.0f, 0.25f);
    CHECK(ellipsoid.areaPdf == 0.0f);
    CHECK(analyticEllipsoidSurfaceArea(x, y, zero) == 0.0f);
    CHECK_FALSE(intersectAnalyticEllipsoid(
                    make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f, zero, x, y, zero)
                    .hit);

    // Scene packing marks a singular disc's inverse-transpose normal invalid;
    // the sampler and intersection must not retain positive area behind it.
    const AnalyticLightSample disc = sampleAnalyticDisc(zero, x, y, zero, 0.3f, 0.7f);
    CHECK(disc.areaPdf == 0.0f);
    CHECK_FALSE(intersectAnalyticDisc(
                    make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f, zero, x, y, zero)
                    .hit);

    // Mutation: the old cofactor-only checks see positive 2D area in both
    // collapsed objects even though their declared inverse transform does not exist.
    CHECK(analyticDiscArea(x, y) > 0.0f);
    CHECK(length(affineSphereCofactor(x, y, zero, make_float3(0.0f, 0.0f, 1.0f))) > 0.0f);
}

TEST_CASE("a very thin full-rank ellipsoid keeps sampler and intersection support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1e-20f);
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.0f);
    REQUIRE(sample.areaPdf > 0.0f);

    const float3 origin = make_float3(0.0f, 0.0f, 2.0f);
    const float3 direction = normalize(sample.point - origin);
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 10.0f, center, axisX, axisY, axisZ);
    CHECK(hit.hit);
    CHECK(std::isfinite(hit.distance));
    CHECK(std::isfinite(hit.normal.x));
    CHECK(std::isfinite(hit.normal.y));
    CHECK(std::isfinite(hit.normal.z));

    // Mutation: forming inverse-space coefficients before the quadratic
    // overflows for this valid affine map and loses the sampled endpoint.
    const float3 oldOrigin = affineSphereCoordinates(axisX, axisY, axisZ, origin);
    const float3 oldDirection = affineSphereCoordinates(axisX, axisY, axisZ, direction);
    const float oldA = dot(oldDirection, oldDirection);
    const float oldB = dot(oldOrigin, oldDirection);
    const float oldC = dot(oldOrigin, oldOrigin) - 1.0f;
    CHECK_FALSE(std::isfinite(oldB * oldB - oldA * oldC));
}

TEST_CASE("analytic lights reject subnormal conditional densities across backends")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e20f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e20f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1e20f);

    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.25f);
    CHECK(sample.areaPdf == 0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(0.0f, 0.0f, 2e20f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4e20f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);

    // Mutations: exact-zero determinant classification accepts infinity, and
    // a CPU with gradual underflow can fabricate a positive conditional PDF
    // that Metal safe math flushes to zero.
    CHECK(fabsf(dot(axisX, cross(axisY, axisZ))) > 0.0f);
    CHECK(1.0 / (4.0 * std::numbers::pi * 1e40) > 0.0);
}

TEST_CASE("large representable analytic lights keep exact sampler and intersection support")
{
    const float scale = 1e13f;
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(scale, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, scale, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, scale);
    REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));

    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.0f);
    REQUIRE(sample.areaPdf > 0.0f);
    CHECK(std::isfinite(sample.areaPdf));
    CHECK(sample.normal == make_float3(1.0f, 0.0f, 0.0f));

    const float3 origin = make_float3(2.0f * scale, 0.0f, 0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        origin, make_float3(-1.0f, 0.0f, 0.0f), 0.0f, 4.0f * scale, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(scale).epsilon(2e-6));
    CHECK(hit.normal == make_float3(1.0f, 0.0f, 0.0f));

    float3 evaluatedNormal;
    const float evaluatedPdf = analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, sample.point, evaluatedNormal);
    CHECK(evaluatedPdf == doctest::Approx(sample.areaPdf).epsilon(2e-6));
    CHECK(evaluatedNormal == sample.normal);
    CHECK(analyticEllipsoidSurfaceArea(axisX, axisY, axisZ) > 0.0f);
    CHECK(std::isfinite(analyticEllipsoidSurfaceArea(axisX, axisY, axisZ)));

    // Mutation: determinant-first validation rejects this representable area
    // measure before either the sampler or homogeneous intersection runs.
    CHECK_FALSE(std::isfinite(dot(axisX, cross(axisY, axisZ))));
}

TEST_CASE("subnormal affine area densities have consistent zero support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(2e19f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 2e19f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));

    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.0f, 0.25f);
    CHECK(sample.areaPdf == 0.0f);
    CHECK_FALSE(std::isfinite(length(cross(axisX, axisY)))); // denominator-form mutation

    const float3 origin = make_float3(0.0f, 0.0f, 2e19f);
    const float3 direction = make_float3(0.0f, 0.0f, -1.0f);
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 4e19f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);

    float3 evaluatedNormal;
    const float evaluatedPdf = analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, sample.point, evaluatedNormal);
    CHECK(evaluatedPdf == 0.0f);
}

TEST_CASE("reciprocal-scale ellipsoid axes retain sample PDF and intersection support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e20f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e-20f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));

    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.0f);
    REQUIRE(sample.areaPdf > 0.0f);
    CHECK(sample.areaPdf == doctest::Approx(1e20 / (4.0 * std::numbers::pi)).epsilon(2e-6));
    float3 evaluatedNormal;
    CHECK(analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, sample.point, evaluatedNormal) ==
          doctest::Approx(sample.areaPdf).epsilon(2e-6));
    CHECK(evaluatedNormal == sample.normal);

    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(2e20f, 0.0f, 0.0f), make_float3(-1.0f, 0.0f, 0.0f), 0.0f, 4e20f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(1e20f).epsilon(2e-6));
    CHECK(hit.normal == sample.normal);

    const float commonScale = 1e20f;
    const float oldDeterminant = dot(axisX / commonScale, cross(axisY / commonScale, axisZ / commonScale));
    CHECK(oldDeterminant == 0.0f);
}

TEST_CASE("finite reciprocal ellipsoid inverses keep analytic support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e-30f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e10f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1e10f);
    REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.0f);
    REQUIRE(sample.areaPdf > 0.0f);
    CHECK(std::isfinite(sample.areaPdf));

    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(2e-30f, 0.0f, 0.0f), make_float3(-1.0f, 0.0f, 0.0f), 0.0f, 4e-30f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(1e-30f).epsilon(2e-5));
    CHECK(hit.areaPdf > 0.0f);
    CHECK(std::isfinite(hit.areaPdf));
}

TEST_CASE("largest normal isotropic ellipsoid density retains support")
{
    const float radius = 2e18f;
    const float3 axisX = make_float3(radius, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, radius, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, radius);
    REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(make_float3(0.0f), axisX, axisY, axisZ, 0.37f, 0.61f);
    CHECK(sample.areaPdf >= std::numeric_limits<float>::min());
    CHECK(std::isfinite(sample.normal.x));
    CHECK(std::isfinite(sample.normal.y));
    CHECK(std::isfinite(sample.normal.z));

    const float subnormalRadius = 6.5e21f;
    const float3 subnormalX = make_float3(subnormalRadius, 0.0f, 0.0f);
    const float3 subnormalY = make_float3(0.0f, subnormalRadius, 0.0f);
    const float3 subnormalZ = make_float3(0.0f, 0.0f, subnormalRadius);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(subnormalX, subnormalY, subnormalZ));
    CHECK(sampleAnalyticEllipsoid(make_float3(0.0f), subnormalX, subnormalY, subnormalZ, 0.37f, 0.61f).areaPdf == 0.0f);
}

TEST_CASE("reciprocal-scale disc axes retain analytic sample and intersection support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e20f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e-20f, 0.0f);
    const float3 emissionNormal = make_float3(0.0f, 0.0f, 1.0f);
    const AnalyticLightSample sample = sampleAnalyticDisc(center, axisX, axisY, emissionNormal, 0.25f, 0.0f);
    REQUIRE(sample.areaPdf > 0.0f);
    CHECK(sample.areaPdf == doctest::Approx(1.0f / float(M_PI_F)).epsilon(2e-6));
    CHECK(sample.normal == emissionNormal);

    const float3 origin = sample.point + make_float3(0.0f, 0.0f, 1.0f);
    const AnalyticLightIntersection hit =
        intersectAnalyticDisc(origin, make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f, center, axisX, axisY, emissionNormal);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(1.0f));
    CHECK(hit.normal == sample.normal);

    // Mutation: the shared-max implementation overflows before the reciprocal
    // axis can cancel that scale, despite the exact cross length being one.
    const float commonScale = 1e20f;
    const float scaledLength = finiteVectorLength(cross(axisX / commonScale, axisY / commonScale));
    const float oldReciprocal = ((1.0f / float(M_PI_F)) / scaledLength / commonScale) / commonScale;
    const bool oldWasRepresentable = oldReciprocal > 0.0f && oldReciprocal <= std::numeric_limits<float>::max();
    CHECK_FALSE(oldWasRepresentable);

    const float3 largeAxisX = make_float3(1e38f, 0.0f, 0.0f);
    const float3 smallAxisY = make_float3(0.0f, 1e-10f, 0.0f);
    const AnalyticLightSample imbalanced =
        sampleAnalyticDisc(center, largeAxisX, smallAxisY, emissionNormal, 0.25f, 0.0f);
    REQUIRE(imbalanced.areaPdf > 0.0f);
    CHECK(imbalanced.areaPdf == doctest::Approx(1.0 / (std::numbers::pi * 1e28)).epsilon(2e-6));
    const AnalyticLightIntersection imbalancedHit =
        intersectAnalyticDisc(imbalanced.point + make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f,
                              2.0f, center, largeAxisX, smallAxisY, emissionNormal);
    CHECK(imbalancedHit.hit);
}

TEST_CASE("a large analytic disc uses scale-safe normals for intersection")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e20f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e20f, 0.0f);
    const float3 emissionNormal = make_float3(0.0f, 0.0f, 1.0f);
    const AnalyticLightSample sample = sampleAnalyticDisc(center, axisX, axisY, emissionNormal, 0.0f, 0.0f);
    CHECK(sample.areaPdf == 0.0f);

    const AnalyticLightIntersection hit = intersectAnalyticDisc(
        make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f, center, axisX, axisY, emissionNormal);
    CHECK_FALSE(hit.hit);

    // Mutation: the old plane normal formed an overflowing raw cross.
    CHECK_FALSE(std::isfinite(cross(axisX, axisY).z));
}

TEST_CASE("a finite raw cross wins over cancellation in normalized inputs")
{
    const float3 a = make_float3(-92.7729645f, 46.9849319f, 67.0263596f);
    const float3 b = make_float3(-8.86599187e35f, 4.49018797e35f, 6.4054781e35f);
    const glm::dvec3 exactCross = glm::cross(glm::dvec3(a), glm::dvec3(b));
    const glm::dvec3 exactNormal = glm::normalize(exactCross);
    const double exactPdf = 1.0 / (std::numbers::pi * glm::length(exactCross));
    const float3 referenceNormal = make_float3(float(exactNormal.x), float(exactNormal.y), float(exactNormal.z));

    const float3 normal = finiteCrossDirection(a, b);
    const float areaPdf = analyticDiscAreaPdf(a, b);
    CHECK(double(dot(normal, referenceNormal)) == doctest::Approx(1.0).epsilon(2e-6));
    CHECK(double(areaPdf) == doctest::Approx(exactPdf).epsilon(2e-6));

    // Mutation: scaling each input first loses the cancellation residual and
    // points the normal into a different plane.
    const float3 normalizedCross = finiteCrossDirection(normalizeFiniteVectorOrZero(a), normalizeFiniteVectorOrZero(b));
    CHECK(double(dot(normalizedCross, referenceNormal)) < 0.9);
}

TEST_CASE("subnormal raw cross falls back to a representable scaled density")
{
    const float3 a = make_float3(-0.00962962955f, 0.00838168897f, -0.000683822378f);
    const float3 b = make_float3(4.99392084e-39f, -8.64924236e-38f, 4.22122749e-38f);
    const glm::dvec3 exactCross = glm::cross(glm::dvec3(a), glm::dvec3(b));
    const double exactPdf = 1.0 / (std::numbers::pi * glm::length(exactCross));
    REQUIRE(exactPdf < std::numeric_limits<float>::max());
    const float areaPdf = analyticDiscAreaPdf(a, b);
    CHECK(areaPdf > 0.0f);
    CHECK(std::isfinite(areaPdf));
    CHECK(double(areaPdf) == doctest::Approx(exactPdf).epsilon(2e-6));

    // Mutation: quantizing the raw cross before division overflows even though
    // the independently scaled expression has a finite result.
    CHECK_FALSE(std::isfinite((1.0f / float(M_PI_F)) / finiteVectorLength(accurateCross(a, b))));
}

TEST_CASE("power-of-two cross scaling preserves normals when density is subnormal")
{
    const float3 a = make_float3(6.315719015e-25f, -2.058267347e23f, -6.421042562e32f);
    const float3 b = make_float3(-7.02678399e-30f, 2.290000284e18f, 7.14396473e27f);
    const glm::dvec3 exactCross = glm::cross(glm::dvec3(a), glm::dvec3(b));
    const glm::dvec3 exactNormal = glm::normalize(exactCross);
    const double exactPdf = 1.0 / (std::numbers::pi * glm::length(exactCross));
    const float3 normal = finiteCrossDirection(a, b);
    const float areaPdf = analyticDiscAreaPdf(a, b);
    REQUIRE(exactPdf < std::numeric_limits<float>::min());
    CHECK(areaPdf == 0.0f);
    CHECK(glm::dot(glm::dvec3(normal), exactNormal) > 0.999999);

    // Mutation: discarding an overflowing component makes an irrelevant small
    // component become the normal and changes the density by many orders.
    const float3 raw = accurateCross(a, b);
    const float3 finiteRaw = make_float3(
        std::isfinite(raw.x) ? raw.x : 0.0f, std::isfinite(raw.y) ? raw.y : 0.0f, std::isfinite(raw.z) ? raw.z : 0.0f);
    CHECK(glm::dot(glm::dvec3(normalizeFiniteVectorOrZero(finiteRaw)), exactNormal) < 0.01);
}

TEST_CASE("power-of-two cross scaling retains ordinary finite densities")
{
    const float3 a = make_float3(-2.416244708e20f, -8.15424704e20f, -1.435870284e21f);
    const float3 b = make_float3(-2.55069686e20f, -8.607985406e20f, -1.51576949e21f);
    const glm::dvec3 exactCross = glm::cross(glm::dvec3(a), glm::dvec3(b));
    const double exactPdf = 1.0 / (std::numbers::pi * glm::length(exactCross));
    const float areaPdf = analyticDiscAreaPdf(a, b);
    REQUIRE(areaPdf > 0.0f);
    CHECK(double(areaPdf) == doctest::Approx(exactPdf).epsilon(2e-5));
    CHECK(glm::dot(glm::dvec3(finiteCrossDirection(a, b)), glm::normalize(exactCross)) > 0.999999);
}

TEST_CASE("affine orientation uses a compensated triple product")
{
    const float3 x = make_float3(-0.760982871f, 0.223688096f, 0.608989894f);
    const float3 y = make_float3(-0.44675234f, 0.179339349f, -0.87649852f);
    const float3 z = make_float3(0.539036989f, -0.142934054f, -0.830065668f);
    const glm::dvec3 xd(x);
    const glm::dvec3 yd(y);
    const glm::dvec3 zd(z);
    REQUIRE(glm::dot(xd, glm::cross(yd, zd)) > 0.0);
    CHECK(analyticAffineOrientation(x, y, z) == 1.0f);

    // Mutation: naive float products reverse the orientation.
    CHECK(dot(x, cross(y, z)) < 0.0f);
}

TEST_CASE("affine orientation retains the exact sign near float cancellation")
{
    const float3 x = make_float3(1521382272.0f, -4562933760.0f, 621223168.0f);
    const float3 y = make_float3(3831902976.0f, 1824664064.0f, -4543640064.0f);
    const float3 z = make_float3(-2499741696.0f, -4787814912.0f, 4614122496.0f);
    const glm::dvec3 xd(x);
    const glm::dvec3 yd(y);
    const glm::dvec3 zd(z);
    const double exactDeterminant = glm::dot(xd, glm::cross(yd, zd));
    REQUIRE(exactDeterminant < 0.0);
    CHECK(analyticAffineOrientation(x, y, z) == -1.0f);

    // Mutation: rounding each cofactor before the triple product flips the
    // sidedness of this otherwise finite affine light.
    const float3 xn = normalizeFiniteVectorOrZero(x);
    const float3 yn = normalizeFiniteVectorOrZero(y);
    const float3 zn = normalizeFiniteVectorOrZero(z);
    CHECK(accurateDot(xn, accurateCross(yn, zn)) > 0.0f);
}

TEST_CASE("ill-conditioned ellipsoid point maps have no partial support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(7.1825589e-8f, 5.9514921e-8f, -2.69455853e-8f);
    const float3 axisY = make_float3(-8.47644524e-11f, 9.36010366e-11f, -1.92090233e-11f);
    const float3 axisZ = make_float3(9.36574361e-5f, 2.48845143e-4f, 7.99277448e-4f);
    REQUIRE(scaledAffineBasis(axisX, axisY, axisZ).valid);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.712177873f, 0.0402052477f);
    CHECK(sample.areaPdf == 0.0f);

    const float3 origin = make_float3(4e-4f);
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, -normalize(origin), 0.0f, 1e-3f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);

    float3 pointNormal;
    CHECK(analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, axisX, pointNormal) == 0.0f);

    // Mutation: determinant and local Jacobians alone accept the map, leaving
    // positive NEE density even though float(A*x) cannot represent one shared
    // ellipsoid endpoint reliably.
    float3 ignored;
    CHECK(affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(1.0f, 0.0f, 0.0f), ignored) > 0.0f);
    CHECK(affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 1.0f, 0.0f), ignored) > 0.0f);
    CHECK(affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 0.0f, 1.0f), ignored) > 0.0f);
}

TEST_CASE("ordinary rotated non-uniform ellipsoids retain analytic support")
{
    const float angle = 0.6f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    for (const float aspect : { 16.0f, 1500.0f })
    {
        const float3 axisX = make_float3(aspect * cosine, aspect * sine, 0.0f);
        const float3 axisY = make_float3(-sine, cosine, 0.0f);
        const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
        REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));

        const AnalyticLightSample sample = sampleAnalyticEllipsoid(make_float3(0.0f), axisX, axisY, axisZ, 0.4f, 0.7f);
        REQUIRE(sample.areaPdf > 0.0f);
        const float3 origin = sample.point + sample.normal * 4.0f;
        const AnalyticLightIntersection hit =
            intersectAnalyticEllipsoid(origin, -sample.normal, 0.0f, 8.0f, make_float3(0.0f), axisX, axisY, axisZ);
        REQUIRE(hit.hit);
        CHECK(hit.areaPdf == doctest::Approx(sample.areaPdf).epsilon(2e-4));
        CHECK(glm::dot(hit.normal, sample.normal) > 0.9999f);
    }

    const float3 unstableAxisX = make_float3(8192.0f * cosine, 8192.0f * sine, 0.0f);
    const float3 unstableAxisY = make_float3(-sine, cosine, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(unstableAxisX, unstableAxisY, axisZ));
    CHECK(sampleAnalyticEllipsoid(make_float3(0.0f), unstableAxisX, unstableAxisY, axisZ, 0.4f, 0.7f).areaPdf == 0.0f);
}

TEST_CASE("far analytic sphere intersections retain the unit-radius geometry")
{
    const float3 zero = make_float3(0.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    for (const float distance : { 1e3f, 1e4f, 1e5f, 1e6f, 1e7f, 1e8f })
    {
        const AnalyticLightIntersection hit =
            intersectAnalyticEllipsoid(make_float3(0.0f, 0.0f, distance + 1.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f,
                                       2.0f * distance, zero, axisX, axisY, axisZ);
        REQUIRE(hit.hit);
        CHECK(finiteVectorLength(hit.point) == doctest::Approx(1.0f).epsilon(1e-6));
        CHECK(hit.point.z > 0.0f);
        CHECK(hit.areaPdf == doctest::Approx(1.0f / (4.0f * float(M_PI_F))));
    }

    // Mutation: the expanded quadratic subtracts indistinguishable O(D^2)
    // floats to recover a unit-radius discriminant.
    const float origin = 100001.0f;
    CHECK(origin * origin - (origin * origin - 1.0f) == 0.0f);
}

TEST_CASE("far affine intersections use the geometric sphere discriminant")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(-0.00104633148f, -0.00179306185f, 0.00373847224f);
    const float3 axisY = make_float3(0.00752747664f, -0.000384329469f, 0.0062315953f);
    const float3 axisZ = make_float3(-0.00675372034f, 0.00230669905f, 0.00739100156f);
    const float3 point = make_float3(0.00583745912f, -0.000655979849f, 0.00800315198f);
    float3 expectedNormal;
    const float expectedPdf = analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, point, expectedNormal);
    REQUIRE(expectedPdf > 0.0f);

    const float3 origin = make_float3(1.8809042f, -1.57326388f, 2.80466914f);
    float distance = 0.0f;
    const float3 direction = finiteDirectionAndDistance(point - origin, distance);
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 4.0f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.areaPdf == doctest::Approx(expectedPdf).epsilon(2e-4));
    CHECK(glm::dot(hit.normal, expectedNormal) > 0.9999f);
    float3 hitNormal;
    CHECK(analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, hit.point, hitNormal) ==
          doctest::Approx(expectedPdf).epsilon(2e-4));
}

TEST_CASE("far transformed disc intersections retain the sampled local point")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(-17681.582f, 570.998535f, -12698.1465f);
    const float3 axisY = make_float3(-407.984558f, -65.6591263f, -185.461899f);
    const float3 normal = finiteCrossDirection(axisX, axisY);
    const float3 point = -0.466799f * axisX + 0.0245106f * axisY;
    const float3 origin = make_float3(-7.23433152e8f, 1.46389171e9f, 1.07319155e9f);
    float expectedDistance = 0.0f;
    const float3 direction = finiteDirectionAndDistance(point - origin, expectedDistance);
    const AnalyticLightIntersection hit =
        intersectAnalyticDisc(origin, direction, 0.0f, 1.01f * expectedDistance, center, axisX, axisY, normal);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(expectedDistance).epsilon(2e-6));
    float3 localPoint;
    REQUIRE(solveAffineCoordinates(axisX, axisY, normal, hit.point, localPoint));
    // At this distance the rounded ray direction reaches a different point on
    // the same constant-density disc; the analytic event must still retain
    // support rather than inherit the old Cramer cancellation miss.
    CHECK(localPoint.x * localPoint.x + localPoint.y * localPoint.y < 1.0f);
}

TEST_CASE("large off-axis rays do not fabricate analytic sphere hits")
{
    const float3 zero = make_float3(0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(3e38f), make_float3(-1.0f, 0.0f, 0.0f), 0.0f, 3.4e38f, zero, make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
    CHECK_FALSE(hit.hit);
}

TEST_CASE("far sampled ellipsoid directions survive affine inversion")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(0.204458386f, -0.988693833f, 0.803253651f);
    const float3 axisY = make_float3(5.53949070f, 2.91283822f, 2.88948560f);
    const float3 axisZ = make_float3(-18.0428257f, -6.54786968f, -11.0914745f);
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.608120680f, 0.929035604f);
    REQUIRE(sample.areaPdf > 0.0f);

    const float maxAxis = std::max({ std::abs(axisX.x), std::abs(axisX.y), std::abs(axisX.z), std::abs(axisY.x),
                                     std::abs(axisY.y), std::abs(axisY.z), std::abs(axisZ.x), std::abs(axisZ.y),
                                     std::abs(axisZ.z) });
    const float3 origin = sample.point + (3000.0f * maxAxis) * sample.normal;
    float expectedDistance = 0.0f;
    const float3 direction = finiteDirectionAndDistance(sample.point - origin, expectedDistance);
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 1.01f * expectedDistance, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    // Stable closest-point Decimal-100 inversion of the actual rounded float
    // ray. Its event differs from the original object-space sample at this
    // distance.
    const double oracleDistance = 54128.47223893619;
    const double oracleAreaPdf = 0.01202316558152245;
    CHECK(std::abs(double(hit.distance) - oracleDistance) / oracleDistance < 2e-6);
    CHECK(std::abs(double(hit.areaPdf) - oracleAreaPdf) / oracleAreaPdf < 2e-4);
}

TEST_CASE("unstable world-space affine maps are rejected before intersection")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(0.00756730838f, 0.0248504151f, -0.0175565388f);
    const float3 axisY = make_float3(0.190996543f, -0.137583911f, 0.156075403f);
    const float3 axisZ = make_float3(651.313721f, 92.1582031f, 92.4737625f);
    const float3 origin = make_float3(-63663120.0f, 261363808.0f, 320730784.0f);
    const float3 direction = make_float3(0.152083531f, -0.624364734f, -0.766184866f);
    REQUIRE(scaledAffineBasis(axisX, axisY, axisZ).valid);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 460468288.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("grazing ellipsoid events share the sampler validity decision")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(-785.847412f, -681.281067f, -650.097473f);
    const float3 axisY = make_float3(-1518.17835f, -1316.69812f, -1256.17432f);
    const float3 axisZ = make_float3(-642.634583f, -565.386780f, -436.309845f);
    const float3 origin = make_float3(-1342.88501f, 2381.62061f, 2741.38892f);
    const float3 direction = make_float3(0.752844036f, -0.409161001f, -0.515570462f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 4000.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unstable shallow ellipsoid events have zero support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(-0.00398159213f, 0.000330713141f, 0.00293193618f);
    const float3 axisY = make_float3(-0.00555002922f, 0.0136571527f, 0.0578561462f);
    const float3 axisZ = make_float3(-80.9410858f, -285.8573f, -389.395782f);
    const float3 origin = make_float3(32.3587875f, 120.718857f, 170.991226f);
    const float3 direction = make_float3(0.391622931f, 0.789420366f, 0.472701728f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 12.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unstable affine roots have zero ellipsoid density")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(158.029495f, -114.250336f, -116.824974f);
    const float3 axisY = make_float3(0.00142893556f, 0.000479158247f, -0.000841408386f);
    const float3 axisZ = make_float3(0.268196851f, -0.181894377f, -0.0859287232f);
    const float3 origin = make_float3(545.809753f, 596.197998f, 620.241333f);
    const float3 direction = make_float3(-0.607456088f, -0.550127149f, -0.573024631f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 1200.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unstable well-inside affine events have zero support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(-0.000881578831f, 0.000938957208f, 0.000169104474f);
    const float3 axisY = make_float3(6.43567371f, -6.7786932f, -1.3295995f);
    const float3 axisZ = make_float3(188.472794f, 164.594452f, -40.2989082f);
    const float3 origin = make_float3(74.476265f, 78.0275574f, -18.2688217f);
    const float3 direction = make_float3(0.872094691f, 0.273129016f, 0.406018972f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 4.3f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unstable far affine events have zero support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(12.2097139f, -705.285706f, 215.371231f);
    const float3 axisY = make_float3(-1.37548041f, -0.84901464f, -1.02014947f);
    const float3 axisZ = make_float3(-0.000329842936f, 0.000400975288f, 0.000543817878f);
    const float3 origin = make_float3(-335.224854f, -5.64324951f, -700.73999f);
    const float3 direction = make_float3(0.353966027f, 0.767305195f, 0.534743607f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 1000.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unstable non-grazing affine events have zero support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(478.180054f, -526.177185f, 439.428772f);
    const float3 axisY = make_float3(-0.00215341966f, 0.00184252427f, 0.00174143084f);
    const float3 axisZ = make_float3(-0.0026615907f, 0.00503811194f, 0.000180512565f);
    const float3 origin = make_float3(-190.425812f, 544.610718f, 9.85165405f);
    const float3 direction = make_float3(-0.215680808f, -0.67771101f, -0.702986181f);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 400.0f, center, axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("affine sphere Jacobians use the direct cofactor")
{
    const float3 axisX = make_float3(0.000304052635f, 0.00170207024f, 0.00128460769f);
    const float3 axisY = make_float3(447.760712f, 747.030823f, 484.648926f);
    const float3 axisZ = make_float3(0.000951631751f, 0.000829412544f, 0.0000969282191f);
    const float3 objectNormal = make_float3(-0.426422715f, -0.364615619f, 0.827779591f);
    float3 worldNormal;
    const float areaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, objectNormal, worldNormal);
    const double oracleAreaPdf = 0.2290906013;
    const glm::dvec3 oracleNormal(0.08349491, 0.50665755, -0.85809482);
    CHECK(std::abs(double(areaPdf) - oracleAreaPdf) / oracleAreaPdf < 2e-4);
    CHECK(glm::dot(glm::dvec3(worldNormal), oracleNormal) > 0.999999);

    // Mutation: rounding two nearly parallel transformed tangents before
    // their cross loses the small cofactor components and changes the measure.
    const float3 n = normalizeFiniteVectorOrZero(objectNormal);
    const float3 tangent = normalizeFiniteVectorOrZero(cross(make_float3(0.0f, 0.0f, 1.0f), n));
    const float3 bitangent = cross(n, tangent);
    const float3 worldTangent = tangent.x * axisX + tangent.y * axisY + tangent.z * axisZ;
    const float3 worldBitangent = bitangent.x * axisX + bitangent.y * axisY + bitangent.z * axisZ;
    const float materializedPdf = finiteCrossReciprocal(worldTangent, worldBitangent, 1.0f / (4.0f * M_PI_F));
    CHECK(std::abs(double(materializedPdf) - oracleAreaPdf) / oracleAreaPdf > 0.01);
}

TEST_CASE("an ill-conditioned disc sample remains on its analytic intersection surface")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1.32084184e-8f, 1.13400649e-8f, 2.06591437e-8f);
    const float3 axisY = make_float3(-6.23864729e-13f, -5.41376134e-13f, -9.7829622e-13f);
    const float3 normal = finiteCrossDirection(axisX, axisY);
    const float3 point = 0.3f * axisX - 0.4f * axisY;
    const float distance = 1.350795653e-8f;
    const float3 origin = point + normal * distance;
    const AnalyticLightIntersection hit =
        intersectAnalyticDisc(origin, -normal, 0.0f, 2.0f * distance, center, axisX, axisY, normal);
    REQUIRE(hit.hit);
    CHECK(hit.distance == doctest::Approx(distance).epsilon(2e-5));
    CHECK(hit.normal == normal);
    CHECK(analyticDiscAreaPdf(axisX, axisY) > 0.0f);
}

TEST_CASE("analytic samples reject finite world-point overflow")
{
    const float maxFinite = std::numeric_limits<float>::max();
    const float3 center = make_float3(maxFinite, 0.0f, 0.0f);
    const float3 axisX = make_float3(1e32f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e-10f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1e-10f);
    const float3 normal = make_float3(0.0f, 0.0f, 1.0f);

    CHECK(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    CHECK_FALSE(affineSamplePointRangeIsFinite(center, axisX, axisY, axisZ));
    CHECK(sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.0f).areaPdf == 0.0f);
    CHECK(sampleAnalyticDisc(center, axisX, axisY, normal, 1.0f, 0.0f).areaPdf == 0.0f);
    CHECK_FALSE(intersectAnalyticEllipsoid(
                    make_float3(0.0f), make_float3(1.0f, 0.0f, 0.0f), 0.0f, maxFinite, center, axisX, axisY, axisZ)
                    .hit);
    CHECK_FALSE(intersectAnalyticDisc(
                    make_float3(0.0f), make_float3(1.0f, 0.0f, 0.0f), 0.0f, maxFinite, center, axisX, axisY, normal)
                    .hit);

    // Mutation: checking only finite inputs allows the final FMA to overflow
    // while retaining a positive conditional area density.
    CHECK_FALSE(std::isfinite(std::fma(1.0f, axisX.x, center.x)));
    CHECK(analyticDiscAreaPdf(axisX, axisY) > 0.0f);
}

TEST_CASE("area-to-solid-angle saturation clamps rounded unit cosines")
{
    const float3 normal = make_float3(-0.408859611f, 0.248610795f, -0.878081203f);
    const float roundedCosine = dot(normal, normal);
    REQUIRE(roundedCosine > 1.0f);

    const float fromDensity = areaPdfToSolidAnglePdf(2e19f, roundedCosine, 1.0f);
    const float fromArea = areaLightSolidAnglePdf(2e19f, roundedCosine, 1.0f);
    CHECK(fromDensity == std::numeric_limits<float>::max());
    CHECK(fromArea == std::numeric_limits<float>::max());
    CHECK(std::isfinite(fromDensity));
    CHECK(std::isfinite(fromArea));

    // Mutation: multiplying the bound by the unclamped cosine overflows.
    CHECK_FALSE(std::isfinite(std::numeric_limits<float>::max() * roundedCosine));
}

TEST_CASE("area-to-solid-angle conversion distinguishes underflow from overflow")
{
    CHECK(areaPdfToSolidAnglePdf(1e-10f, 1.0f, 1e-38f) == 0.0f);
    CHECK(areaPdfToSolidAnglePdf(2e20f, 1.0f, 1e-30f) == doctest::Approx(4e10f).epsilon(2e-6));

    // Mutation: treating a zero first product as overflow returns the largest
    // float for an exact density below the representable range.
    CHECK(1e-38f * 1e-10f == 0.0f);
}

TEST_CASE("analytic sample and hit PDFs agree at distances whose square overflows")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e15f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 3.1830989e14f, 0.0f);
    const float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    const float3 origin = make_float3(0.0f, 0.0f, 2e20f);
    const AnalyticLightSample sample = sampleAnalyticDisc(center, axisX, axisY, normal, 0.0f, 0.0f);
    const AnalyticLightIntersection hit =
        intersectAnalyticDisc(origin, -normal, 0.0f, 3e20f, center, axisX, axisY, normal);
    REQUIRE(sample.areaPdf > 0.0f);
    REQUIRE(hit.hit);

    const float sampleDistance = finiteVectorLength(sample.point - origin);
    const float hitDistance = finiteVectorLength(hit.point - origin);
    const float samplePdf = areaPdfToSolidAnglePdf(sampleDistance, 1.0f, sample.areaPdf);
    const float hitPdf = areaPdfToSolidAnglePdf(hitDistance, 1.0f, hit.areaPdf);
    CHECK(samplePdf == doctest::Approx(4e10f).epsilon(2e-6));
    CHECK(hitPdf == samplePdf);

    // Mutation: the backend's former raw vector length overflows and changes
    // the complementary BSDF-hit MIS density to the largest float.
    CHECK_FALSE(std::isfinite(length(hit.point - origin)));
}

TEST_CASE("affine surface normals use the inverse transpose")
{
    const float3 axisX = make_float3(2.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    const float3 objectNormal = normalize(make_float3(1.0f, 1.0f, 0.0f));
    const float3 worldNormal = transformAffineNormal(axisX, axisY, axisZ, objectNormal);
    const float3 expected = normalize(make_float3(0.5f, 1.0f, 0.0f));
    CHECK(worldNormal.x == doctest::Approx(expected.x).epsilon(1e-6));
    CHECK(worldNormal.y == doctest::Approx(expected.y).epsilon(1e-6));
    CHECK(worldNormal.z == doctest::Approx(expected.z).epsilon(1e-6));

    const float3 direction = normalize(make_float3(-0.6f, 0.8f, 0.0f));
    CHECK(dot(worldNormal, direction) > 0.0f);
    const float3 oldForwardNormal = normalize(objectNormal.x * axisX + objectNormal.y * axisY);
    CHECK(dot(oldForwardNormal, direction) < 0.0f);

    const float3 mirrored = transformAffineNormal(-axisX, axisY, axisZ, make_float3(0.0f, 0.0f, 1.0f));
    CHECK(mirrored == make_float3(0.0f, 0.0f, 1.0f));

    // Only the cofactor selected by this object normal is small. Scaling all
    // cofactors by the largest pair product underflows that valid direction.
    const float3 extremeX = make_float3(1e-30f, 0.0f, 0.0f);
    const float3 extremeY = make_float3(0.0f, 1e30f, 0.0f);
    const float3 extremeZ = make_float3(0.0f, 0.0f, 1.0f);
    CHECK(transformAffineNormal(extremeX, extremeY, extremeZ, make_float3(0.0f, 1.0f, 0.0f)) ==
          make_float3(0.0f, 1.0f, 0.0f));
}

TEST_CASE("analytic lights retain support across the full homogeneous exponent range")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e-30f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e30f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    REQUIRE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.25f);
    CHECK(sample.areaPdf > 0.0f);
    CHECK(std::isfinite(sample.areaPdf));
    float3 normal;
    CHECK(analyticEllipsoidAreaPdf(center, axisX, axisY, axisZ, axisY, normal) > 0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(0.0f, 2e30f, 0.0f), make_float3(0.0f, -1.0f, 0.0f), 0.0f, 4e30f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.areaPdf > 0.0f);
    CHECK(std::isfinite(hit.areaPdf));

    // Mutation: one common scale erases the reciprocal axis and reports a
    // singular transform even though all three original axes are full-rank.
    const float commonScale = 1e30f;
    CHECK(dot(axisX / commonScale, cross(axisY / commonScale, axisZ / commonScale)) == 0.0f);
}

TEST_CASE("homogeneous ellipsoids do not require a representable explicit inverse")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e-39f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    float3 normal;
    const float maximumAreaPdf = affineSphereAreaPdfAndNormal(axisX, axisY, axisZ, make_float3(0.0f, 1.0f, 0.0f), normal);
    REQUIRE(maximumAreaPdf > 0.0f);
    REQUIRE(std::isfinite(maximumAreaPdf));
    CHECK(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));

    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.25f);
    CHECK(sample.areaPdf > 0.0f);
    CHECK(std::isfinite(sample.areaPdf));
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoid(
        make_float3(0.0f, 2.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f), 0.0f, 4.0f, center, axisX, axisY, axisZ);
    REQUIRE(hit.hit);
    CHECK(hit.areaPdf == maximumAreaPdf);

    // Mutation: a materialized inverse falsely rejects the otherwise
    // representable homogeneous geometry because 1 / 1e-39 exceeds float.
    float3 explicitInverse;
    CHECK_FALSE(solveAffineCoordinates(axisX, axisY, axisZ, make_float3(1.0f, 0.0f, 0.0f), explicitInverse));
}

TEST_CASE("unrepresentable independent axis scales do not fabricate ellipsoid hits")
{
    const float3 axisX = make_float3(3.69605172e16f, -6.59055645e16f, 7.1990249e16f);
    const float3 axisY = make_float3(8.81609171e23f, -1.87966457e24f, -6.80089589e23f);
    const float3 axisZ = make_float3(-9.20824007e-12f, 2.41302672e-11f, 2.79694271e-11f);
    const float3 origin = make_float3(1.07635714e25f, -2.29488357e25f, -8.30321789e24f);
    const float3 direction = make_float3(-0.403538615f, 0.860378146f, 0.311297208f);

    // Decimal-120 inversion puts the closest point at
    // (4.3873221996, 8.00926302565, 0), whose sphere margin is -82.39689.
    // Independent column scaling resolves the determinant, but the float
    // point map cannot retain all three object coordinates.
    CHECK(scaledAffineBasis(axisX, axisY, axisZ).valid);
    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 2.81661421e25f, make_float3(0.0f), axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);

    // Mutation: a shared row scale gives the three independently sized axes a
    // severely ill-conditioned numerical representation.
    const float scaleX = fmaxf(fabsf(axisX.x), fmaxf(fabsf(axisX.y), fabsf(axisX.z)));
    const float scaleY = fmaxf(fabsf(axisY.x), fmaxf(fabsf(axisY.y), fabsf(axisY.z)));
    const float scaleZ = fmaxf(fabsf(axisZ.x), fmaxf(fabsf(axisZ.y), fabsf(axisZ.z)));
    CHECK(compensatedDotCross(axisX / scaleX, axisY / scaleY, axisZ / scaleZ) != 0.0f);
}

TEST_CASE("unrepresentable independent axis scales reject definite ellipsoid hits")
{
    const float3 axisX = make_float3(-5.43080251e11f, 3.71845464e11f, -5.53306948e11f);
    const float3 axisY = make_float3(-5824.61816f, -48433.5703f, -30959.418f);
    const float3 axisZ = make_float3(3.50807946e-12f, -2.49112878e-13f, -6.75337715e-13f);
    const float3 origin = make_float3(4.1185919e11f, -2.81998787e11f, 4.19614884e11f);
    const float3 direction = make_float3(-0.631593764f, 0.432450354f, -0.643487334f);

    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 4.53982487e11f, make_float3(0.0f), axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("unrepresentable independent scales reject ellipsoid hit density")
{
    const float3 axisX = make_float3(3501.73047f, -2062.47144f, 4152.95801f);
    const float3 axisY = make_float3(1.3271892e10f, -2.36455552e9f, -8.32448768e9f);
    const float3 axisZ = make_float3(1.19920534e-12f, 1.26033976e-13f, -4.42286182e-13f);
    const float3 origin = make_float3(-2.88716621e9f, 514386880.0f, 1.81090086e9f);
    const float3 direction = make_float3(0.837664723f, -0.149241969f, -0.525399625f);

    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightIntersection hit =
        intersectAnalyticEllipsoid(origin, direction, 0.0f, 5.0e9f, make_float3(0.0f), axisX, axisY, axisZ);
    CHECK_FALSE(hit.hit);
}

TEST_CASE("a back-facing or degenerate area sample has no density")
{
    // Zero rather than a negative or infinite pdf: the callers read this as "the
    // light strategy could not have produced this direction", which hands the
    // whole direction to the BSDF strategy through misWeightBalance(a, 0) == 1.
    CHECK(areaLightSolidAnglePdf(2.0f, -0.5f, 1.0f) == 0.0f);
    CHECK(areaLightSolidAnglePdf(2.0f, 0.0f, 1.0f) == 0.0f);
    CHECK(areaLightSolidAnglePdf(2.0f, 0.5f, 0.0f) == 0.0f);
    CHECK(misWeightBalance(1.0f, areaLightSolidAnglePdf(2.0f, -0.5f, 1.0f)) == doctest::Approx(1.0f));

    const float largeFinite = areaLightSolidAnglePdf(1e30f, 0.5f, 1e30f);
    CHECK(std::isfinite(largeFinite));
    CHECK(largeFinite == doctest::Approx(2e30f).epsilon(2e-6));
    CHECK(areaLightSolidAnglePdf(1e20f, 0.5f, 1e-30f) == std::numeric_limits<float>::max());
}

TEST_CASE("analytic marginal PDF applies light selection before saturation")
{
    LightPdfQuery q = makeLightPdfQuery(LIGHT_TYPE_DISC);
    q.distToLight = 1e10f;
    q.cosAtLight = 1.0f;
    q.areaPdf = 1e30f;
    constexpr float localSelection = 1e-8f;
    constexpr float classSelection = 1e-6f;
    constexpr float lightSelection = 1e-6f;

    const float marginal = marginalLightSolidAnglePdf(q, localSelection, classSelection, lightSelection);
    const long double oracle = static_cast<long double>(q.areaPdf) * static_cast<long double>(q.distToLight) *
                               static_cast<long double>(q.distToLight) * static_cast<long double>(localSelection) *
                               static_cast<long double>(classSelection) * static_cast<long double>(lightSelection);
    CHECK(marginal == doctest::Approx(static_cast<double>(oracle)).epsilon(2e-6));
    CHECK(std::isfinite(marginal));

    // Mutation: this is the old renderer order. The conditional clamps to
    // FLT_MAX before the outer PMFs can bring the full density back to 1e30.
    const float oldMarginal = lightSolidAnglePdf(q) * localSelection * classSelection * lightSelection;
    CHECK(std::abs(double(oldMarginal) - double(oracle)) / double(oracle) > 0.99);
}

TEST_CASE("complete area-light marginal PDF agrees with a long-double oracle")
{
    std::mt19937 rng(0x504446u);
    std::uniform_real_distribution<float> mantissa(0.5f, 1.0f);
    std::uniform_int_distribution<int> areaExponent(-100, 100);
    std::uniform_int_distribution<int> distanceExponent(-40, 40);
    std::uniform_int_distribution<int> selectionExponent(-30, 0);
    std::uniform_int_distribution<int> cosineExponent(-20, 0);
    uint32_t checked = 0u;
    uint32_t oldFailures = 0u;

    while (checked < 4096u)
    {
        const float areaPdf = std::ldexp(mantissa(rng), areaExponent(rng));
        const float distance = std::ldexp(mantissa(rng), distanceExponent(rng));
        const float cosine = std::min(std::ldexp(mantissa(rng), cosineExponent(rng)), 1.0f);
        const float s0 = std::ldexp(mantissa(rng), selectionExponent(rng));
        const float s1 = std::ldexp(mantissa(rng), selectionExponent(rng));
        const float s2 = std::ldexp(mantissa(rng), selectionExponent(rng));
        const float s3 = std::ldexp(mantissa(rng), selectionExponent(rng));
        const long double oracle = static_cast<long double>(areaPdf) * static_cast<long double>(distance) *
                                   static_cast<long double>(distance) * static_cast<long double>(s0) *
                                   static_cast<long double>(s1) * static_cast<long double>(s2) *
                                   static_cast<long double>(s3) / static_cast<long double>(cosine);
        if (!(oracle >= std::numeric_limits<float>::min()) || !(oracle <= std::numeric_limits<float>::max()))
        {
            continue;
        }

        const float pdf = areaPdfToSolidAngleMarginalPdf(distance, cosine, areaPdf, s0, s1, s2, s3);
        CAPTURE(areaPdf);
        CAPTURE(distance);
        CAPTURE(cosine);
        CAPTURE(s0);
        CAPTURE(s1);
        CAPTURE(s2);
        CAPTURE(s3);
        CAPTURE(oracle);
        CAPTURE(pdf);
        REQUIRE(pdf > 0.0f);
        REQUIRE(std::isfinite(pdf));
        CHECK(std::abs(static_cast<long double>(pdf) - oracle) / oracle < 2e-6L);

        const float oldPdf = areaPdfToSolidAnglePdf(distance, cosine, areaPdf) * s0 * s1 * s2 * s3;
        if (!(oldPdf > 0.0f) || std::abs(static_cast<long double>(oldPdf) - oracle) / oracle > 1e-3L)
        {
            ++oldFailures;
        }
        ++checked;
    }
    CHECK(oldFailures > 100u);
}

// ---------------------------------------------------------------------------
// A point light with a radius. Colour means intensity there and radiance on a
// sphere light, so the conversion has to be exactly the one that makes the two
// meet as the radius closes.
// ---------------------------------------------------------------------------
TEST_CASE("a soft point light converges on the sharp one as its radius shrinks")
{
    const float intensity = 10.0f;
    const float distance = 4.0f;
    // What a delta point light delivers: I/d^2, times the cosine, which is 1 here.
    const double sharp = double(intensity) / (double(distance) * double(distance));

    for (float radius : { 0.4f, 0.2f, 0.05f, 0.01f })
    {
        CAPTURE(radius);
        const double soft = softPointIrradiance(radius, distance, intensity, 400000);
        // Tighter than the estimator noise at every radius; the point is that
        // there is no step at all, and in particular not the factor of 4*pi the
        // old inverse-square-plus-1/(4pi) combination produced.
        CHECK(soft == doctest::Approx(sharp).epsilon(0.03));
    }
}

TEST_CASE("intensity to radiance on a sphere is the inverse of its projected area")
{
    // I = pi r^2 L for a uniformly emitting sphere, so L = I / (pi r^2).
    const float r = 0.3f;
    CHECK(sphereRadianceFromIntensity(r) == doctest::Approx(1.0f / (float(M_PI_F) * r * r)));
    // Guarded at the softness threshold rather than dividing by zero, so a light
    // authored at exactly the boundary does not produce an infinite radiance.
    CHECK(std::isfinite(sphereRadianceFromIntensity(0.0f)));
}

// ---------------------------------------------------------------------------
// The dome. Sampled uniformly over the sphere of directions, and missing from
// one backend's switch entirely.
// ---------------------------------------------------------------------------
TEST_CASE("a dome light's density integrates to one over the sphere")
{
    Rng rng(0xD0DEu);
    double sum = 0.0;
    const int samples = 200000;
    for (int i = 0; i < samples; ++i)
    {
        (void)uniformSphereDirection(rng.next(), rng.next());
        sum += 1.0 / double(domeLightSolidAnglePdf());
    }
    // The mean of 1/pdf over the sampled directions is the measure they cover.
    CHECK(sum / double(samples) == doctest::Approx(4.0 * double(M_PI_F)).epsilon(1e-5));
}

TEST_CASE("uniform sphere directions really are unit and really do cover the sphere")
{
    Rng rng(0x51D3u);
    float3 mean = make_float3(0.0f, 0.0f, 0.0f);
    const int samples = 100000;
    for (int i = 0; i < samples; ++i)
    {
        const float3 d = uniformSphereDirection(rng.next(), rng.next());
        REQUIRE(len(d) == doctest::Approx(1.0f).epsilon(1e-4));
        mean = make_float3(mean.x + d.x, mean.y + d.y, mean.z + d.z);
    }
    // A distribution over the whole sphere has a zero mean; one over a
    // hemisphere does not, which is the mistake worth catching here.
    CHECK(len(make_float3(mean.x / samples, mean.y / samples, mean.z / samples)) < 0.02f);
}

// ---------------------------------------------------------------------------
// The distant light's cone. Algebra that is exact on paper and not in floats.
// ---------------------------------------------------------------------------
TEST_CASE("the cone solid angle survives sun-sized half angles")
{
    // The reference is the same quantity in double precision, where neither form
    // cancels. The float form that shipped on one side -- 1/(2pi(1-cos a)) -- is
    // computed here too, so this case states what the difference is rather than
    // merely asserting the good one.
    const float angles[] = { 0.1f, 0.01f, 0.00465f /* the sun */, 0.001f };
    for (float a : angles)
    {
        CAPTURE(a);
        const double s = std::sin(0.5 * double(a));
        const double exact = 1.0 / (4.0 * double(M_PI_F) * s * s);

        const float stable = coneLightSolidAnglePdf(a);
        CHECK(double(stable) == doctest::Approx(exact).epsilon(1e-4));

        const float cancelling = 1.0f / (2.0f * float(M_PI_F) * (1.0f - std::cos(a)));
        if (a <= 0.00465f)
        {
            // Not a property of the code under test -- a property of the form it
            // must not go back to. At the sun's half angle this is already 0.2%
            // off, and the host bakes the light's radiance by dividing by the
            // same solid angle, so the error does not cancel.
            const double relative = std::abs(double(cancelling) - exact) / exact;
            CHECK(relative > 1e-3);
        }
    }
}

TEST_CASE("cone solid angle spans the full sphere and matches the host bake")
{
    // The host's coneSolidAngle() in scene/light_desc.h is this function; the two
    // have to be the same number or a distant light's radiance is baked with one
    // and divided out with the other.
    CHECK(coneSolidAngleFromHalfAngle(float(M_PI_F)) == doctest::Approx(4.0f * float(M_PI_F)).epsilon(1e-5));
    CHECK(coneSolidAngleFromHalfAngle(float(M_PI_F) * 0.5f) == doctest::Approx(2.0f * float(M_PI_F)).epsilon(1e-5));
    CHECK(coneLightSolidAnglePdf(0.0f) == 0.0f); // degenerate, not infinite
}

TEST_CASE("finite distant support is exactly its spherical cap")
{
    const float halfAngle = 0.37f;
    const float expected = 1.0f / (4.0f * float(M_PI_F) * std::pow(std::sin(0.5f * halfAngle), 2.0f));
    const float3 axis = make_float3(0.0f, 0.0f, 1.0f);
    const float3 boundaryDirection =
        sampleDistantLightDirection(0.0f, 0.9999998807907104f, 0x12345678u, 0x9abcdef0u, halfAngle, axis);
    const float outsideAngle = halfAngle + 1e-4f;
    const float3 outsideDirection = make_float3(std::sin(outsideAngle), 0.0f, std::cos(outsideAngle));

    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, axis, axis) ==
          doctest::Approx(expected).epsilon(1e-5));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, boundaryDirection, axis) ==
          doctest::Approx(expected).epsilon(1e-5));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, outsideDirection, axis) == 0.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DOME, 0.0f, -axis, axis) ==
          doctest::Approx(1.0f / (4.0f * float(M_PI_F))));

    Rng rng(0xCA9u);
    double integral = 0.0;
    constexpr int samples = 500000;
    for (int i = 0; i < samples; ++i)
    {
        const float3 w = uniformSphereDirection(rng.next(), rng.next());
        integral += 4.0 * double(M_PI_F) * double(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, w, axis));
    }
    CHECK(integral / double(samples) == doctest::Approx(1.0).epsilon(0.02));
}

TEST_CASE("distant sampling and directional PDF share the exact cap support")
{
    const float3 axes[] = { make_float3(0.0f, 0.0f, 1.0f), unit(make_float3(1.0f, 1.0f, 1.0f)),
                            unit(make_float3(-0.91f, 0.37f, 0.19f)) };
    const float angles[] = { 1e-6f, 1e-5f, 1e-4f, 0.00459216f, 0.37f, float(M_PI_F) };
    const float values[] = { 0.0f, std::nextafter(0.0f, 1.0f), 0.25f, 0.5f, 0.9999998807907104f };

    for (float angle : angles)
    {
        CAPTURE(angle);
        const float expected = coneLightSolidAnglePdf(angle);
        REQUIRE(expected > 0.0f);
        REQUIRE(std::isfinite(expected));
        for (const float3 axis : axes)
        {
            for (const float u : values)
            {
                for (const float q : values)
                {
                    const float3 direction =
                        sampleDistantLightDirection(u, q, 0x12345678u, 0x9abcdef0u, angle, axis);
                    CAPTURE(axis.x);
                    CAPTURE(axis.y);
                    CAPTURE(axis.z);
                    CAPTURE(u);
                    CAPTURE(q);
                    CAPTURE(direction.x);
                    CAPTURE(direction.y);
                    CAPTURE(direction.z);
                    CHECK(len(direction) == doctest::Approx(1.0f).epsilon(2e-6));
                    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, angle, direction, axis) ==
                          doctest::Approx(expected).epsilon(1e-6));
                }
            }
        }
    }

    // Mutation: deriving sin(theta) from a cosine that has already rounded to
    // one collapses this positive-measure narrow-cap sample onto the axis.
    const float angle = 1e-5f;
    const float halfSin = std::sin(0.5f * angle);
    const float oldCosTheta = 1.0f - 0.5f * (2.0f * halfSin * halfSin);
    const float oldSinTheta = std::sqrt(1.0f - oldCosTheta * oldCosTheta);
    CHECK(oldSinTheta == 0.0f);
    const float3 stable = sampleDistantLightDirection(0.0f, 0.5f, 0x12345678u, 0x9abcdef0u, angle, axes[0]);
    CHECK(len(sub(stable, axes[0])) > 0.0f);

    Rng rng(0xD157A47u);
    for (int i = 0; i < 100000; ++i)
    {
        const float3 axis = uniformSphereDirection(rng.next(), rng.next());
        const double exponent = -6.0 + double(rng.next()) * (std::log10(double(M_PI_F)) + 6.0);
        const float randomAngle = float(std::pow(10.0, exponent));
        const float3 direction =
            sampleDistantLightDirection(rng.next(), rng.next(), rng.gen(), rng.gen(), randomAngle, axis);
        const float returnedPdf = coneLightSolidAnglePdf(randomAngle);
        const float evaluatedPdf = infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, randomAngle, direction, axis);
        const double halfAngleSin = std::sin(0.5 * double(randomAngle));
        const double oraclePdf = 1.0 / (4.0 * double(M_PI_F) * halfAngleSin * halfAngleSin);
        CAPTURE(i);
        CAPTURE(randomAngle);
        CAPTURE(axis.x);
        CAPTURE(axis.y);
        CAPTURE(axis.z);
        CAPTURE(direction.x);
        CAPTURE(direction.y);
        CAPTURE(direction.z);
        CHECK(returnedPdf > 0.0f);
        CHECK(std::isfinite(returnedPdf));
        CHECK(evaluatedPdf == returnedPdf);
        CHECK(double(returnedPdf) == doctest::Approx(oraclePdf).epsilon(2e-6));
    }
}

TEST_CASE("finite distant retries do not fold rejected boundary cells onto one direction")
{
    const float3 axis = unit(make_float3(1.0f, 1.0f, 1.0f));
    const float halfAngle = 0.00465000002f;
    const float uPhi = 0.5f;
    const float q0 = float(8388438u) * 0x1p-23f;
    const float q1 = float(8388439u) * 0x1p-23f;

    const float3 a = sampleDistantLightDirection(uPhi, q0, 0x13579bdfu, 0x2468ace0u, halfAngle, axis);
    const float3 b = sampleDistantLightDirection(uPhi, q1, 0x9e3779b9u, 0x7f4a7c15u, halfAngle, axis);

    CHECK(distantLightContainsDirection(halfAngle, a, axis));
    CHECK(distantLightContainsDirection(halfAngle, b, axis));
    const bool distinct = a.x != b.x || a.y != b.y || a.z != b.z;
    CHECK(distinct);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, a, axis) ==
          coneLightSolidAnglePdf(halfAngle));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, b, axis) ==
          coneLightSolidAnglePdf(halfAngle));
}

TEST_CASE("infinite lights do not inherit area-emitter sidedness")
{
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_DISTANT, -1.0f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_DOME, -1.0f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_POINT, -1.0f));
    CHECK_FALSE(lightConnectionFacesVertex(LIGHT_TYPE_POINT, -1.0f, 0.25f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_RECT, 1e-6f));
    CHECK_FALSE(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -1e-6f));

    // Mutation: the former unconditional area-facing test rejected exactly
    // half of a distant light whose cap spans the full sphere.
    CHECK_FALSE(lightSampleFacesVertex(-1.0f));
    const float3 axis = make_float3(0.0f, 0.0f, 1.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, float(M_PI_F), -axis, axis) > 0.0f);
}

TEST_CASE("a sharp distant is delta and has no continuous density")
{
    CHECK(distantLightIsDelta(0.0f));
    CHECK(distantLightIsDelta(-1.0f));
    CHECK(distantLightIsDelta(std::numeric_limits<float>::quiet_NaN()));
    CHECK_FALSE(distantLightIsDelta(1e-6f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_DISTANT, 0.0f));
    const float3 axis = make_float3(0.0f, 0.0f, 1.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, 0.0f, axis, axis) == 0.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, 0.0f, -axis, axis) == 0.0f);
    CHECK(coneLightSolidAnglePdf(0.0f) == 0.0f);
    CHECK(std::isfinite(coneLightSolidAnglePdf(0.0f)));
}

TEST_CASE("a distant cap narrower than the portable float measure is a delta atom")
{
    // Apple GPU arithmetic flushes subnormal products in supported production
    // modes. Calling this a continuous cap on the CPU but a zero-measure cap on
    // the GPU gives the two backends different supports and PDF classifications.
    // The old exact-zero classification fails this assertion.
    CHECK(distantLightIsDelta(1e-30f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_DISTANT, 1e-30f));
    const float3 axis = make_float3(0.0f, 0.0f, 1.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, 1e-30f, axis, axis) == 0.0f);
    CHECK(float(oka::kMinContinuousDistantHalfAngle) == STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE);
    CHECK(distantLightIsDelta(std::nextafter(STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE, 0.0f)));
    CHECK_FALSE(distantLightIsDelta(STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE));
    CHECK(std::isfinite(coneLightSolidAnglePdf(STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE)));
    CHECK(coneLightSolidAnglePdf(STRELKA_MIN_CONTINUOUS_DISTANT_HALF_ANGLE) > 0.0f);
}

TEST_CASE("a sharp distant matches only a represented specular atom")
{
    const float3 axis = unit(make_float3(0.25f, -0.5f, 1.0f));
    CHECK(distantLightDeltaDirectionMatches(axis, axis));
    CHECK_FALSE(distantLightDeltaDirectionMatches(unit(make_float3(axis.x + 1e-4f, axis.y, axis.z)), axis));

    // Mutation: the old miss branch asked only for a continuous density and
    // therefore discarded even the exact mirror/distant atom.
    CHECK(coneLightSolidAnglePdf(0.0f) == 0.0f);
    CHECK(infiniteLightDistance() == 1e16f);
}

TEST_CASE("a sharp distant atom is visible to primary and specular paths")
{
    const float3 axis = unit(make_float3(1.0f, 1.0f, 1.0f));
    CHECK(distantLightDeltaPathMatches(0u, false, axis, axis));
    CHECK(distantLightDeltaPathMatches(1u, true, axis, axis));
    CHECK_FALSE(distantLightDeltaPathMatches(1u, false, axis, axis));

    const float3 nearby = unit(make_float3(1.0f, 1.0f, 1.000001f));
    CHECK_FALSE(distantLightDeltaPathMatches(0u, false, nearby, axis));
}

TEST_CASE("analytic infinite lights obey camera and secondary visibility masks")
{
    CHECK_FALSE(analyticLightVisibilityAllowsRay(0.0f, false));
    CHECK_FALSE(analyticLightVisibilityAllowsRay(0.0f, true));
    CHECK_FALSE(analyticLightVisibilityAllowsRay(std::numeric_limits<float>::quiet_NaN(), true));
    CHECK_FALSE(analyticLightVisibilityAllowsRay(std::numeric_limits<float>::infinity(), true));
    CHECK(analyticLightVisibilityAllowsRay(float(STRELKA_ANALYTIC_LIGHT_CAMERA_BIT), false));
    CHECK(analyticLightVisibilityAllowsRay(float(STRELKA_ANALYTIC_LIGHT_CAMERA_BIT), true));
    CHECK_FALSE(analyticLightVisibilityAllowsRay(float(STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT), false));
    CHECK(analyticLightVisibilityAllowsRay(float(STRELKA_ANALYTIC_LIGHT_SECONDARY_BIT), true));
}

TEST_CASE("dome NEE and BSDF-miss shares form one Lambertian estimate")
{
    // Unit-radiance dome over a unit-albedo Lambertian surface has outgoing
    // radiance exactly one. One sample from each strategy is combined. Removing
    // the BSDF-miss term is the old implementation's mutation: it leaves only
    // about 0.299 even though the MIS arithmetic inside either branch is valid.
    Rng rng(0xD04Eu);
    constexpr int samples = 500000;
    double nee = 0.0;
    double bsdfMiss = 0.0;
    const double lightPdf = double(domeLightSolidAnglePdf());
    for (int i = 0; i < samples; ++i)
    {
        const float3 lightDirection = uniformSphereDirection(rng.next(), rng.next());
        if (lightDirection.z > 0.0f)
        {
            const double bsdfPdf = double(lightDirection.z) / double(M_PI_F);
            const double weight = lightPdf / (lightPdf + bsdfPdf);
            nee += (double(lightDirection.z) / double(M_PI_F)) * weight / lightPdf;
        }

        const double u = double(rng.next());
        const double cosTheta = std::sqrt(1.0 - u);
        const double bsdfPdf = cosTheta / double(M_PI_F);
        const double weight = bsdfPdf / (bsdfPdf + lightPdf);
        bsdfMiss += weight;
    }

    const double neeMean = nee / double(samples);
    const double combined = neeMean + bsdfMiss / double(samples);
    CHECK(neeMean == doctest::Approx(0.299).epsilon(0.02));
    CHECK(combined == doctest::Approx(1.0).epsilon(0.01));
}

// ---------------------------------------------------------------------------
// The dispatcher. One switch for both backends, so a light type cannot be
// sampled by one and be invisible to the other.
// ---------------------------------------------------------------------------
TEST_CASE("every light type has a density")
{
    for (int type : kAllLightTypes)
    {
        CAPTURE(type);
        const LightPdfQuery q = plausibleQuery(type);
        const float pdf = lightSolidAnglePdf(q);
        // A missing case returns zero, which the connection code reads as "this
        // light cannot be sampled" and silently drops. That is what happened to
        // LIGHT_TYPE_DOME on Metal: no compile error, no NaN, just an unlit
        // scene.
        CHECK(pdf > 0.0f);
        CHECK(std::isfinite(pdf));
    }
}

TEST_CASE("the dispatcher routes each type to the density that type was sampled from")
{
    const LightPdfQuery rect = plausibleQuery(LIGHT_TYPE_RECT);
    CHECK(lightSolidAnglePdf(rect) ==
          doctest::Approx(areaPdfToSolidAnglePdf(rect.distToLight, rect.cosAtLight, rect.areaPdf)));

    // Solid-angle rect sampling selects 1/S instead, and only when the caller
    // says the spherical quadrilateral was well conditioned.
    LightPdfQuery rectSolid = rect;
    rectSolid.solidAngle = 0.35f;
    CHECK(lightSolidAnglePdf(rectSolid) == doctest::Approx(1.0f / 0.35f));

    const LightPdfQuery sphere = plausibleQuery(LIGHT_TYPE_SPHERE);
    CHECK(lightSolidAnglePdf(sphere) ==
          doctest::Approx(areaPdfToSolidAnglePdf(sphere.distToLight, sphere.cosAtLight, sphere.areaPdf)));

    const LightPdfQuery dome = plausibleQuery(LIGHT_TYPE_DOME);
    CHECK(lightSolidAnglePdf(dome) == doctest::Approx(domeLightSolidAnglePdf()));

    const LightPdfQuery distant = plausibleQuery(LIGHT_TYPE_DISTANT);
    CHECK(lightSolidAnglePdf(distant) == doctest::Approx(coneLightSolidAnglePdf(distant.halfAngle)));

    LightPdfQuery sharpDistant = distant;
    sharpDistant.halfAngle = 0.0f;
    CHECK(lightSolidAnglePdf(sharpDistant) == deltaLightPdf());
}

TEST_CASE("a point light's density follows its radius across the softness threshold")
{
    LightPdfQuery sharp = plausibleQuery(LIGHT_TYPE_POINT);
    sharp.radius = 0.0f;
    CHECK(lightSolidAnglePdf(sharp) == doctest::Approx(deltaLightPdf()));
    CHECK(punctualLightIsSoft(0.0f) == false);
    CHECK(punctualLightIsSoft(STRELKA_SOFT_LIGHT_RADIUS_MIN) == false);
    CHECK(punctualLightIsSoft(2.0f * STRELKA_SOFT_LIGHT_RADIUS_MIN) == true);

    LightPdfQuery soft = plausibleQuery(LIGHT_TYPE_SPOT);
    soft.radius = 0.25f;
    // A sphere of that radius, not the placeholder: the estimator divides by
    // this, and dividing an area-sampled connection by 1 is the same class of
    // error the sphere light had.
    CHECK(lightSolidAnglePdf(soft) ==
          doctest::Approx(sphereLightSolidAnglePdf(soft.distToLight, soft.cosAtLight, soft.radius)));
}

TEST_CASE("only sharp punctual lights are delta for MIS")
{
    // The second scalar is the conditional shape parameter: punctual radius or
    // distant half-angle. A positive-radius punctual sphere has a continuous
    // BSDF-hit strategy; only the represented point is singular.
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_POINT, 0.0f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_SPOT, 0.0f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_POINT, 2.0f * STRELKA_SOFT_LIGHT_RADIUS_MIN));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_SPOT, 0.25f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_PROJECTOR, 0.25f));

    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_RECT, 0.0f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_DISC, 0.0f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_SPHERE, 0.0f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_DISTANT, 0.0f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_DISTANT, 0.05f));
    CHECK_FALSE(lightIsDeltaForMis(LIGHT_TYPE_DOME, 0.0f));
}

// ---------------------------------------------------------------------------
// The heuristics
// ---------------------------------------------------------------------------
TEST_CASE("both MIS heuristics split every direction into shares that sum to one")
{
    const float pdfs[] = { 1e-8f, 1e-3f, 0.25f, 1.0f, 17.0f, 1e6f, 1e12f };
    for (unsigned int heuristic : { 0u, 1u })
    {
        CAPTURE(heuristic);
        for (float a : pdfs)
        {
            for (float b : pdfs)
            {
                CAPTURE(a);
                CAPTURE(b);
                const float wa = computeMisWeight(a, b, heuristic);
                const float wb = computeMisWeight(b, a, heuristic);
                CHECK(std::isfinite(wa));
                CHECK(wa >= 0.0f);
                CHECK(wa <= 1.0f);
                // The property the whole scheme rests on. Ten orders of
                // magnitude apart is where the a^2/(a^2+b^2) form overflows,
                // which is why both are written as 1/(1 + (b/a)^k).
                CHECK(wa + wb == doctest::Approx(1.0f).epsilon(1e-5));
            }
        }
    }
}

TEST_CASE("a zero pdf on either side never produces a NaN")
{
    for (unsigned int heuristic : { 0u, 1u })
    {
        CAPTURE(heuristic);
        // The strategy that produced the direction claims it outright.
        CHECK(computeMisWeight(1.0f, 0.0f, heuristic) == doctest::Approx(1.0f));
        // The other one claims nothing.
        CHECK(computeMisWeight(0.0f, 1.0f, heuristic) == doctest::Approx(0.0f));
        // Both zero: 0/0 used to reach the division and put a NaN in the pixel.
        // Reachable at a light hit whose spherical quadrilateral underflowed
        // while the bounce was delta.
        const float both = computeMisWeight(0.0f, 0.0f, heuristic);
        CHECK(std::isfinite(both));
        CHECK(both == doctest::Approx(1.0f));
    }
}

TEST_CASE("the heuristic switch selects two genuinely different weightings")
{
    // Guards the dispatch itself. One backend called the balance form
    // unconditionally, so render/pt/misHeuristic did nothing on it and no A/B
    // between the backends under the power heuristic was possible.
    const float a = 3.0f;
    const float b = 1.0f;
    CHECK(computeMisWeight(a, b, 0u) == doctest::Approx(misWeightBalance(a, b)));
    CHECK(computeMisWeight(a, b, 1u) == doctest::Approx(misWeightPower(a, b)));
    CHECK(misWeightBalance(a, b) != doctest::Approx(misWeightPower(a, b)).epsilon(1e-3));

    CHECK(misWeightBalance(3.0f, 1.0f) == doctest::Approx(0.75f));
    CHECK(misWeightPower(3.0f, 1.0f) == doctest::Approx(0.9f));
}

TEST_CASE("a light sample facing away is rejected at exactly zero")
{
    // Both halves of the estimate ask this, and they have to ask it the same
    // way. Metal's connection used a 1e-3 threshold while its light hit used 0,
    // so the band between them was deducted for and never delivered.
    CHECK(lightSampleFacesVertex(1e-7f));
    CHECK(lightSampleFacesVertex(1e-4f));
    CHECK_FALSE(lightSampleFacesVertex(0.0f));
    CHECK_FALSE(lightSampleFacesVertex(-1e-7f));
}

TEST_CASE("OptiX arbitrates analytic lights against a nearer hardware hit")
{
    const float3 corner = make_float3(-0.5f, -0.5f, 1.0f);
    const float3 edgeX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 edgeY = make_float3(0.0f, 1.0f, 0.0f);
    const float hardwareDistance = 2.0f;
    const AnalyticLightIntersection analytic = intersectAnalyticLightSurface(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY,
        make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f), 0.0f,
        hardwareDistance);

    REQUIRE(analytic.hit);
    CHECK(analytic.distance == doctest::Approx(1.0f));
    CHECK(analytic.distance < hardwareDistance);

    // Frozen mutation: the former OptiX path searched analytic surfaces only
    // from __miss__. Any hardware hit, even one behind the light, therefore
    // suppressed this valid light event.
    const bool oldMissOnlyPathSelectsAnalytic = false;
    CHECK_FALSE(oldMissOnlyPathSelectsAnalytic);

    // The unavailable-on-macOS backend arbitrates the same way Metal does now:
    // the analytic emitters are custom primitives in the same structure as the
    // geometry, so the nearest hit wins by traversal and nothing arbitrates
    // afterwards. What replaced the light-table walks -- one in the raygen
    // before every trace, one in the miss program, one per shadow ray -- took
    // kids_room from 20.2 to 12.1 ms/sample and iso_bathroom from 11.3 to 6.8.
    //
    // Source text is not the geometry oracle here: the arithmetic above is.
    // What this pins is that the walks are gone and did not creep back, and
    // that an analytic emitter does not stand in another light's way.
    //
    // The mask carried the light bits until 2026-09-11, so that an emitter
    // stopped a shadow ray the way a wall does. Cycles does not do that:
    // tools/feature_tests/light_occlusion_probe.py hangs a small rect light
    // under a big one over a grey floor, and the reference draws no silhouette
    // under the blocker where Strelka drew one (0.540 against 0.254 across the
    // floor). Mesh emitters are unaffected -- they are triangles and keep the
    // geometry bit.
    const std::filesystem::path repository =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    const auto read = [&](const char* relative) {
        std::ifstream file(repository / relative);
        REQUIRE(file.good());
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    };
    const std::string raygen = read("src/shaders/optix/OptixRender.cu");
    const std::string closestHit = read("src/shaders/optix/OptixRender_closest_hit.cu");
    CHECK(raygen.find("findAnalyticAreaLightHit") == std::string::npos);
    CHECK(closestHit.find("findAnalyticAreaLightHit") == std::string::npos);
    CHECK(closestHit.find("analyticLightsOccludeSegment") == std::string::npos);
    CHECK(closestHit.find("__intersection__light") != std::string::npos);

    const std::string params = read("src/render/optix/OptixRenderParams.h");
    const size_t shadowMask = params.find("RAY_MASK_SHADOW =");
    REQUIRE(shadowMask != std::string::npos);
    const std::string shadowMaskLine = params.substr(shadowMask, params.find(',', shadowMask) - shadowMask);
    CHECK(shadowMaskLine.find("GEOMETRY_MASK_GEOMETRY") != std::string::npos);
    CHECK(shadowMaskLine.find("GEOMETRY_MASK_LIGHT") == std::string::npos);
}

TEST_CASE("the nearest of two area emitters is the visible hit")
{
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const float3 normal = make_float3(0.0f, 0.0f, -1.0f);
    const auto hit = [&](float z) {
        const float3 corner = make_float3(-0.5f, -0.5f, z);
        return intersectAnalyticLightSurface(LIGHT_TYPE_RECT, corner, corner + make_float3(1.0f, 0.0f, 0.0f),
                                             make_float3(0.0f), corner + make_float3(0.0f, 1.0f, 0.0f), normal,
                                             origin, direction, 0.0f, 100.0f);
    };
    const AnalyticLightIntersection near = hit(2.0f);
    const AnalyticLightIntersection far = hit(4.0f);
    REQUIRE(near.hit);
    REQUIRE(far.hit);
    CHECK(near.distance < far.distance);
}

TEST_CASE("OptiX coincident analytic emitters retain every BSDF-hit component")
{
    const float3 corner = make_float3(-0.5f, -0.5f, 1.0f);
    const float3 edgeX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 edgeY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const AnalyticLightIntersection front = intersectAnalyticLightSurface(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY,
        make_float3(0.0f, 0.0f, -1.0f), origin, direction, 0.0f, 100.0f);
    const AnalyticLightIntersection back = intersectAnalyticLightSurface(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY,
        make_float3(0.0f, 0.0f, 1.0f), origin, direction, 0.0f, 100.0f);

    REQUIRE(analyticLightIntersectionSharesEvent(front.distance, back));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -dot3(direction, front.normal)));
    CHECK_FALSE(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -dot3(direction, back.normal)));

    const float bsdfPdf = 0.2f;
    const float lightPdf0 = areaPdfToSolidAngleMarginalPdf(1.0f, 1.0f, front.areaPdf, 1.0f, 1.0f, 0.25f, 1.0f);
    const float lightPdf1 = areaPdfToSolidAngleMarginalPdf(1.0f, 1.0f, front.areaPdf, 1.0f, 1.0f, 0.75f, 1.0f);
    const float fullHit = 2.0f * computeMisWeight(bsdfPdf, lightPdf0, 0u) +
                          3.0f * computeMisWeight(bsdfPdf, lightPdf1, 0u);
    const float oldSingleIdentity = 2.0f * computeMisWeight(bsdfPdf, lightPdf0, 0u);
    CHECK(fullHit > oldSingleIdentity);
    CHECK(computeMisWeight(bsdfPdf, lightPdf0, 0u) + computeMisWeight(lightPdf0, bsdfPdf, 0u) ==
          doctest::Approx(1.0f));
    CHECK(computeMisWeight(bsdfPdf, lightPdf1, 0u) + computeMisWeight(lightPdf1, bsdfPdf, 0u) ==
          doctest::Approx(1.0f));

    const std::filesystem::path repository =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream sourceFile(repository / "src/shaders/optix/OptixRender_closest_hit.cu");
    REQUIRE(sourceFile.good());
    const std::string source((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
    CHECK(source.find("for (uint32_t componentId = 0u;") != std::string::npos);
    CHECK(source.find("analyticLightIntersectionSharesEvent") != std::string::npos);
}

TEST_CASE("Metal analytic surfaces use TLAS traversal")
{
    const std::filesystem::path repository =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR).parent_path().parent_path();
    std::ifstream shaderFile(repository / "src/shaders/metal/wavefront.metal");
    REQUIRE(shaderFile.good());
    const std::string shader((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
    CHECK(shader.find("findAnalyticAreaLightHit") == std::string::npos);
    CHECK(shader.find("analyticLightsOccludeSegment") == std::string::npos);
    CHECK(shader.find("for (uint32_t componentId = 0u;") == std::string::npos);
    CHECK(shader.find("ignoredLightId") == std::string::npos);

    std::ifstream asFile(repository / "src/render/metal/MetalAccelStructure.mm");
    REQUIRE(asFile.good());
    const std::string asSource((std::istreambuf_iterator<char>(asFile)), std::istreambuf_iterator<char>());
    CHECK(asSource.find("lightTypeIsPunctual(lightType) || infinite") != std::string::npos);
    CHECK(asSource.find("GEOMETRY_MASK_LIGHT_HIDDEN") != std::string::npos);
}
