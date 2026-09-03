#include <doctest/doctest.h>

#include <analytic_light.h>
#include <light_pdf.h>

#include <algorithm>
#include <cmath>
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
    q.area = 2.0f; // rect / disc
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
        const double areaPdfDenominator =
            analyticEllipsoidAreaPdfDenominator(make_float3(0.0f), axisX, axisY, axisZ, point, evaluatedNormal);
        integratedMass += (1.0 / areaPdfDenominator) * oracle * (4.0 * std::numbers::pi / 4096.0);
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
        CHECK(disc.areaPdfDenominator == doctest::Approx(analyticDiscArea(axisX, axisY)).epsilon(1e-6));

        const AnalyticLightSample sphere = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, rng.next(), rng.next());
        const glm::dvec3 q = objectCoordinates(transform, center, sphere.point);
        CHECK(glm::length(q) == doctest::Approx(1.0).epsilon(3e-5));
        const double jacobian = affineAreaJacobian(transform, glm::normalize(q));
        CHECK(double(sphere.areaPdfDenominator) == doctest::Approx(4.0 * std::numbers::pi * jacobian).epsilon(3e-5));
        CHECK(sphere.areaPdfDenominator > 0.0f);
        CHECK(std::isfinite(sphere.areaPdfDenominator));
        CHECK(std::isfinite(sphere.normal.x));
        CHECK(std::isfinite(sphere.normal.y));
        CHECK(std::isfinite(sphere.normal.z));
        const glm::dvec3 normalOracle = glm::normalize(glm::transpose(glm::inverse(transform)) * glm::normalize(q));
        CHECK(len(sub(sphere.normal, float3(normalOracle))) < 3e-5f);

        const float3 toLight = sub(sphere.point, shadingPoint);
        const double distance = double(len(toLight));
        const float3 wi = unit(toLight);
        const double cosine = -double(dot3(wi, sphere.normal));
        const float reported = areaLightSolidAnglePdf(float(distance), float(cosine), sphere.areaPdfDenominator);
        const double oracle = cosine > 0.0 ? distance * distance / (cosine * double(sphere.areaPdfDenominator)) : 0.0;
        CHECK(double(reported) == doctest::Approx(oracle).epsilon(2e-5));
        float3 evaluatedNormal;
        const float evaluatedDenominator =
            analyticEllipsoidAreaPdfDenominator(center, axisX, axisY, axisZ, sphere.point, evaluatedNormal);
        CHECK(evaluatedDenominator == doctest::Approx(sphere.areaPdfDenominator).epsilon(3e-5));
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
    CHECK_FALSE(lightUsesAnalyticAreaIntersection(LIGHT_TYPE_RECT));
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

TEST_CASE("degenerate analytic lights have zero density without non-finite samples")
{
    const float3 zero = make_float3(0.0f);
    const AnalyticLightSample disc = sampleAnalyticDisc(zero, zero, zero, zero, 0.3f, 0.7f);
    const AnalyticLightSample sphere = sampleAnalyticEllipsoid(zero, zero, zero, zero, 0.3f, 0.7f);
    for (const AnalyticLightSample& sample : { disc, sphere })
    {
        CHECK(sample.areaPdfDenominator == 0.0f);
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
    CHECK(ellipsoid.areaPdfDenominator == 0.0f);
    CHECK(analyticEllipsoidSurfaceArea(x, y, zero) == 0.0f);
    CHECK_FALSE(intersectAnalyticEllipsoid(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f,
                                           10.0f, zero, x, y, zero)
                    .hit);

    // Scene packing marks a singular disc's inverse-transpose normal invalid;
    // the sampler and intersection must not retain positive area behind it.
    const AnalyticLightSample disc = sampleAnalyticDisc(zero, x, y, zero, 0.3f, 0.7f);
    CHECK(disc.areaPdfDenominator == 0.0f);
    CHECK_FALSE(intersectAnalyticDisc(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f,
                                      zero, x, y, zero)
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
    REQUIRE(sample.areaPdfDenominator > 0.0f);

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

TEST_CASE("an ellipsoid whose float area measure overflows has no support")
{
    const float3 center = make_float3(0.0f);
    const float3 axisX = make_float3(1e20f, 0.0f, 0.0f);
    const float3 axisY = make_float3(0.0f, 1e20f, 0.0f);
    const float3 axisZ = make_float3(0.0f, 0.0f, 1e20f);

    CHECK_FALSE(analyticAffineTransformIsNonsingular(axisX, axisY, axisZ));
    const AnalyticLightSample sample = sampleAnalyticEllipsoid(center, axisX, axisY, axisZ, 0.5f, 0.25f);
    CHECK(sample.areaPdfDenominator == 0.0f);
    CHECK(std::isfinite(sample.normal.x));
    CHECK(std::isfinite(sample.normal.y));
    CHECK(std::isfinite(sample.normal.z));
    CHECK_FALSE(intersectAnalyticEllipsoid(make_float3(0.0f, 0.0f, 2e20f), make_float3(0.0f, 0.0f, -1.0f),
                                           0.0f, 4e20f, center, axisX, axisY, axisZ)
                    .hit);

    // Mutation: exact-zero determinant classification accepts infinity.
    CHECK(fabsf(dot(axisX, cross(axisY, axisZ))) > 0.0f);
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
    const float boundary = std::cos(halfAngle);
    const float expected = 1.0f / (4.0f * float(M_PI_F) * std::pow(std::sin(0.5f * halfAngle), 2.0f));

    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, 1.0f) ==
          doctest::Approx(expected).epsilon(1e-5));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, boundary) ==
          doctest::Approx(expected).epsilon(1e-5));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, boundary - 1e-4f) == 0.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DOME, 0.0f, -1.0f) ==
          doctest::Approx(1.0f / (4.0f * float(M_PI_F))));

    Rng rng(0xCA9u);
    double integral = 0.0;
    constexpr int samples = 500000;
    for (int i = 0; i < samples; ++i)
    {
        const float3 w = uniformSphereDirection(rng.next(), rng.next());
        integral += 4.0 * double(M_PI_F) *
                    double(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, halfAngle, w.z));
    }
    CHECK(integral / double(samples) == doctest::Approx(1.0).epsilon(0.02));
}

TEST_CASE("infinite lights do not inherit area-emitter sidedness")
{
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_DISTANT, -1.0f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_DOME, -1.0f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_POINT, -1.0f));
    CHECK(lightConnectionFacesVertex(LIGHT_TYPE_RECT, 1e-6f));
    CHECK_FALSE(lightConnectionFacesVertex(LIGHT_TYPE_RECT, -1e-6f));

    // Mutation: the former unconditional area-facing test rejected exactly
    // half of a distant light whose cap spans the full sphere.
    CHECK_FALSE(lightSampleFacesVertex(-1.0f));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, float(M_PI_F), -1.0f) > 0.0f);
}

TEST_CASE("a sharp distant is delta and has no continuous density")
{
    CHECK(distantLightIsDelta(0.0f));
    CHECK(distantLightIsDelta(-1.0f));
    CHECK(distantLightIsDelta(std::numeric_limits<float>::quiet_NaN()));
    CHECK_FALSE(distantLightIsDelta(1e-6f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_DISTANT, 0.0f));
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, 0.0f, 1.0f) == 0.0f);
    CHECK(infiniteLightConditionalPdf(LIGHT_TYPE_DISTANT, 0.0f, -1.0f) == 0.0f);
    CHECK(coneLightSolidAnglePdf(0.0f) == 0.0f);
    CHECK(std::isfinite(coneLightSolidAnglePdf(0.0f)));
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
          doctest::Approx(areaLightSolidAnglePdf(rect.distToLight, rect.cosAtLight, rect.area)));

    // Solid-angle rect sampling selects 1/S instead, and only when the caller
    // says the spherical quadrilateral was well conditioned.
    LightPdfQuery rectSolid = rect;
    rectSolid.solidAngle = 0.35f;
    CHECK(lightSolidAnglePdf(rectSolid) == doctest::Approx(1.0f / 0.35f));

    const LightPdfQuery sphere = plausibleQuery(LIGHT_TYPE_SPHERE);
    CHECK(lightSolidAnglePdf(sphere) ==
          doctest::Approx(areaLightSolidAnglePdf(sphere.distToLight, sphere.cosAtLight, sphere.area)));

    const LightPdfQuery dome = plausibleQuery(LIGHT_TYPE_DOME);
    CHECK(lightSolidAnglePdf(dome) == doctest::Approx(domeLightSolidAnglePdf()));

    const LightPdfQuery distant = plausibleQuery(LIGHT_TYPE_DISTANT);
    CHECK(lightSolidAnglePdf(distant) == doctest::Approx(coneLightSolidAnglePdf(distant.halfAngle)));
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

TEST_CASE("point and spot lights are delta for MIS whatever their radius")
{
    // Not a statement about geometry. Both backends give a point or spot proxy a
    // zero visibility mask, so no ray can hit one, and weighing the connection
    // against a BSDF pdf deducts a share the other strategy is switched off from
    // ever delivering. A radius used to exempt a light from this.
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_POINT, 0.0f));
    CHECK(lightIsDeltaForMis(LIGHT_TYPE_SPOT, 0.0f));

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
