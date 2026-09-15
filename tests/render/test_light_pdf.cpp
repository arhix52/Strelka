#include <doctest/doctest.h>

#include <analytic_light.h>
#include <light_pdf.h>
#include <strelka/scene/light_desc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

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

        const AnalyticLightSample sphere =
            sampleAnalyticEllipsoidUnchecked(center, axisX, axisY, axisZ, rng.next(), rng.next());
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
        const float evaluatedPdf =
            analyticEllipsoidAreaPdfUnchecked(center, axisX, axisY, axisZ, sphere.point, evaluatedNormal);
        CHECK(evaluatedPdf == doctest::Approx(sphere.areaPdf).epsilon(3e-5));
        CHECK(len(sub(evaluatedNormal, sphere.normal)) < 3e-5f);

        // Near tangency the intersection location is condition-numbered by
        // 1/cos(theta); all such rays are still checked for finite output, while
        // the pointwise identity test is restricted to well-conditioned hits.
        if (cosine > 0.05)
        {
            const AnalyticLightIntersection sphereHit =
                intersectAnalyticEllipsoidUnchecked(shadingPoint, wi, 0.0f, 1e9f, center, axisX, axisY, axisZ);
            REQUIRE(sphereHit.hit);
            CHECK(sphereHit.distance == doctest::Approx(float(distance)).epsilon(5e-5));
            CHECK(len(sub(sphereHit.normal, sphere.normal)) < 5e-5f);
        }
        else
        {
            const AnalyticLightIntersection grazingHit =
                intersectAnalyticEllipsoidUnchecked(shadingPoint, wi, 0.0f, 1e9f, center, axisX, axisY, axisZ);
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
    CHECK(intersectAnalyticLightSurfaceUnchecked(LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f),
                                                 corner + edgeY, normal, boundaryOrigin, boundaryDirection, 0.0f, 20.0f)
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
            intersectAnalyticLightSurfaceUnchecked(LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f),
                                                   corner + edgeY, normal, rayOrigin, rayDirection, 0.0f, 20.0f);
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
            intersectAnalyticLightSurfaceUnchecked(LIGHT_TYPE_POINT, make_float3(radius, 0.0f, 0.0f), center, zero,
                                                   zero, zero, origin, sampleDirection, 0.0f, 10.0f);
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
}

TEST_CASE("selected sphere light keeps near-side self-occlusion")
{
    const float3 center = make_float3(0.0f, 0.0f, 3.0f);
    const float3 axisX = make_float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1.0f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1.0f);
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);

    const AnalyticLightIntersection nearSample = intersectAnalyticEllipsoidUnchecked(
        origin, direction, 0.0f, std::nextafter(2.0f, 0.0f), center, axisX, axisY, axisZ);
    const AnalyticLightIntersection farSample = intersectAnalyticEllipsoidUnchecked(
        origin, direction, 0.0f, std::nextafter(4.0f, 0.0f), center, axisX, axisY, axisZ);
    CHECK_FALSE(nearSample.hit);
    REQUIRE(farSample.hit);
    CHECK(farSample.distance == doctest::Approx(2.0f));
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
        const float3 emissionAxis =
            float3(glm::normalize(glm::transpose(glm::inverse(transform)) * glm::dvec3(0.0, 0.0, -1.0)));
        const OrthonormalLightFrame frame = makeOrthonormalLightFrame(axisX, axisY, emissionAxis);
        REQUIRE(frame.valid);

        const glm::dvec3 oracleZ = glm::normalize(glm::dvec3(emissionAxis));
        const glm::dvec3 inputX(axisX);
        const glm::dvec3 oracleX = glm::normalize(inputX - glm::dot(inputX, oracleZ) * oracleZ);
        const glm::dvec3 inputY(axisY);
        const glm::dvec3 oracleY =
            glm::normalize(inputY - glm::dot(inputY, oracleZ) * oracleZ - glm::dot(inputY, oracleX) * oracleX);
        CAPTURE(sample);
        CHECK(glm::length(glm::dvec3(frame.x) - oracleX) < 4e-6);
        CHECK(glm::length(glm::dvec3(frame.y) - oracleY) < 4e-6);
        CHECK(glm::length(glm::dvec3(frame.emissionAxis) - oracleZ) < 4e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.x), glm::dvec3(frame.y))) < 4e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.x), glm::dvec3(frame.emissionAxis))) < 4e-6);
        CHECK(std::abs(glm::dot(glm::dvec3(frame.y), glm::dvec3(frame.emissionAxis))) < 4e-6);
    }
}

TEST_CASE("degenerate analytic lights have zero density without non-finite samples")
{
    const float3 zero = make_float3(0.0f);
    const AnalyticLightSample disc = sampleAnalyticDisc(zero, zero, zero, zero, 0.3f, 0.7f);
    const AnalyticLightSample sphere = sampleAnalyticEllipsoidUnchecked(zero, zero, zero, zero, 0.3f, 0.7f);
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
    CHECK_FALSE(
        intersectAnalyticEllipsoidUnchecked(zero, make_float3(0.0f, 0.0f, 1.0f), 0.0f, 10.0f, zero, zero, zero, zero).hit);
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

TEST_CASE("large off-axis rays do not fabricate analytic sphere hits")
{
    const float3 zero = make_float3(0.0f);
    const AnalyticLightIntersection hit = intersectAnalyticEllipsoidUnchecked(
        make_float3(3e38f), make_float3(-1.0f, 0.0f, 0.0f), 0.0f, 3.4e38f, zero, make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f));
    CHECK_FALSE(hit.hit);
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
}

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
                    const float3 direction = sampleDistantLightDirection(u, q, 0x12345678u, 0x9abcdef0u, angle, axis);
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
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, a, axis) == coneLightSolidAnglePdf(halfAngle));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, b, axis) == coneLightSolidAnglePdf(halfAngle));
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

TEST_CASE("every light type has a density")
{
    for (int type : kAllLightTypes)
    {
        CAPTURE(type);
        const LightPdfQuery q = plausibleQuery(type);
        const float pdf = lightSolidAnglePdf(q);
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
    const AnalyticLightIntersection analytic = intersectAnalyticLightSurfaceUnchecked(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY, make_float3(0.0f, 0.0f, -1.0f),
        make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f), 0.0f, hardwareDistance);

    REQUIRE(analytic.hit);
    CHECK(analytic.distance == doctest::Approx(1.0f));
    CHECK(analytic.distance < hardwareDistance);
}

TEST_CASE("the nearest of two area emitters is the visible hit")
{
    const float3 origin = make_float3(0.0f);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    const float3 normal = make_float3(0.0f, 0.0f, -1.0f);
    const auto hit = [&](float z) {
        const float3 corner = make_float3(-0.5f, -0.5f, z);
        return intersectAnalyticLightSurfaceUnchecked(LIGHT_TYPE_RECT, corner, corner + make_float3(1.0f, 0.0f, 0.0f),
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
    const AnalyticLightIntersection front = intersectAnalyticLightSurfaceUnchecked(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY, make_float3(0.0f, 0.0f, -1.0f),
        origin, direction, 0.0f, 100.0f);
    const AnalyticLightIntersection back = intersectAnalyticLightSurfaceUnchecked(
        LIGHT_TYPE_RECT, corner, corner + edgeX, make_float3(0.0f), corner + edgeY, make_float3(0.0f, 0.0f, 1.0f),
        origin, direction, 0.0f, 100.0f);

    REQUIRE(analyticLightIntersectionSharesEvent(front.distance, back));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -dot3(direction, front.normal)));
    CHECK_FALSE(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -dot3(direction, back.normal)));

    const float bsdfPdf = 0.2f;
    const float lightPdf0 = areaPdfToSolidAngleMarginalPdf(1.0f, 1.0f, front.areaPdf, 1.0f, 1.0f, 0.25f, 1.0f);
    const float lightPdf1 = areaPdfToSolidAngleMarginalPdf(1.0f, 1.0f, front.areaPdf, 1.0f, 1.0f, 0.75f, 1.0f);
    const float fullHit =
        2.0f * computeMisWeight(bsdfPdf, lightPdf0, 0u) + 3.0f * computeMisWeight(bsdfPdf, lightPdf1, 0u);
    const float oldSingleIdentity = 2.0f * computeMisWeight(bsdfPdf, lightPdf0, 0u);
    CHECK(fullHit > oldSingleIdentity);
    CHECK(computeMisWeight(bsdfPdf, lightPdf0, 0u) + computeMisWeight(lightPdf0, bsdfPdf, 0u) == doctest::Approx(1.0f));
    CHECK(computeMisWeight(bsdfPdf, lightPdf1, 0u) + computeMisWeight(lightPdf1, bsdfPdf, 0u) == doctest::Approx(1.0f));
}
