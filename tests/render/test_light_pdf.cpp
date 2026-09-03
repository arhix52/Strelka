#include <doctest/doctest.h>

#include <light_pdf.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
          doctest::Approx(sphereLightSolidAnglePdf(sphere.distToLight, sphere.cosAtLight, sphere.radius)));

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
