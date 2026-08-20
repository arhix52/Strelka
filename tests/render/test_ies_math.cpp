#include <doctest/doctest.h>

#include <ies_math.h>

#include <strelka/scene/scene.h>
#include <strelka/sceneloader/iesloader.h>

#include <cmath>
#include <vector>

// ============================================================================
// test_ies_math.cpp -- how a photometric table is interpolated, and what it
// means outside itself.
//
// The interpolation is Catmull-Rom over four samples per axis, matching Cycles
// (intern/cycles/kernel/util/ies.h). That is not a stylistic choice: on the
// ladder's Philips CDM-R111 reflector, whose table steps 5 degrees while the
// beam falls 2.4x between 20 and 25 degrees, drawing straight lines between
// samples reads 12.7% high at the midpoint of that interval, and the row sat at
// ratio 1.050 against the reference with the whole of the gap coming from it.
//
// So the cases here pin two separate things: that the curve through the samples
// is the cubic one and not the straight one, and that the table's edges behave
// -- zero outside it rather than an extrapolation, and a seam that closes when
// the azimuth covers the full turn.
// ============================================================================

namespace
{

/// Straight lines between the same samples, for contrast.
float bilinearReference(const std::vector<float>& vAngles,
                        const std::vector<float>& hAngles,
                        const std::vector<float>& candela,
                        float vertDeg,
                        float azimuthDeg)
{
    const int nV = (int)vAngles.size();
    const int nH = (int)hAngles.size();
    const auto interval = [](const std::vector<float>& a, float x) {
        int i = 0;
        while (i + 2 < (int)a.size() && a[(size_t)i + 1] < x)
        {
            ++i;
        }
        return i;
    };
    const int vi = interval(vAngles, vertDeg);
    const int hi = interval(hAngles, azimuthDeg);
    const float tv = (vAngles[(size_t)vi + 1] > vAngles[(size_t)vi]) ?
                         (vertDeg - vAngles[(size_t)vi]) / (vAngles[(size_t)vi + 1] - vAngles[(size_t)vi]) :
                         0.0f;
    const float th = (nH > 1 && hAngles[(size_t)hi + 1] > hAngles[(size_t)hi]) ?
                         (azimuthDeg - hAngles[(size_t)hi]) / (hAngles[(size_t)hi + 1] - hAngles[(size_t)hi]) :
                         0.0f;
    const auto at = [&](int v, int h) { return candela[(size_t)v + (size_t)h * (size_t)nV]; };
    const float c0 = at(vi, hi) * (1.0f - tv) + at(vi + 1, hi) * tv;
    const float c1 = at(vi, hi + 1) * (1.0f - tv) + at(vi + 1, hi + 1) * tv;
    return c0 * (1.0f - th) + c1 * th;
}

/// A steep beam, one plane, already unfolded to the full turn the way the
/// loader leaves it.
struct Beam
{
    std::vector<float> v{ 0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f };
    std::vector<float> h{ 0.0f, 360.0f };
    std::vector<float> candela;

    Beam()
    {
        // The Philips reflector's own shape: a 2.4x drop across the 20-25 step.
        const std::vector<float> column = { 3418.9f, 2804.0f, 1966.7f, 1260.3f, 521.3f, 190.6f, 90.4f, 40.0f, 18.1f };
        candela = column;
        candela.insert(candela.end(), column.begin(), column.end());
    }

    float eval(float vertDeg, float azimuthDeg = 0.0f) const
    {
        return iesEvaluate(v.data(), (int)v.size(), h.data(), (int)h.size(), candela.data(), vertDeg, azimuthDeg);
    }
};

glm::float3 photometricDir(float verticalDeg, float azimuthDeg)
{
    const float th = verticalDeg * float(M_PI) / 180.0f;
    const float ph = azimuthDeg * float(M_PI) / 180.0f;
    return { std::sin(th) * std::sin(ph), -std::sin(th) * std::cos(ph), -std::cos(th) };
}

} // namespace

// ---------------------------------------------------------------------------
// The curve through the samples
// ---------------------------------------------------------------------------
TEST_CASE("Catmull-Rom reproduces the samples it passes through")
{
    // The defining property: at the ends of the interval it is the samples
    // themselves, whatever the neighbours are.
    CHECK(iesCubicInterp(10.0f, 20.0f, 30.0f, 40.0f, 0.0f) == doctest::Approx(20.0f));
    CHECK(iesCubicInterp(10.0f, 20.0f, 30.0f, 40.0f, 1.0f) == doctest::Approx(30.0f));
    CHECK(iesCubicInterp(1000.0f, 500.0f, 200.0f, 90.0f, 0.0f) == doctest::Approx(500.0f));

    // On a straight run of samples it *is* the straight line.
    CHECK(iesCubicInterp(0.0f, 10.0f, 20.0f, 30.0f, 0.5f) == doctest::Approx(15.0f));
    CHECK(iesCubicInterp(0.0f, 10.0f, 20.0f, 30.0f, 0.25f) == doctest::Approx(12.5f));

    // And the closed form, so the expression cannot drift into some other cubic.
    // 0.5 * (-a + 9b + 9c - d) / 8 at the midpoint.
    const float a = 1260.3f, b = 521.3f, c = 190.6f, d = 90.4f;
    CHECK(iesCubicInterp(a, b, c, d, 0.5f) == doctest::Approx((-a + 9.0f * b + 9.0f * c - d) / 16.0f));
}

TEST_CASE("the cubic differs from straight lines exactly where the table is steep")
{
    const Beam beam;

    // Halfway across the steepest step, 20 to 25 degrees. Straight lines cut the
    // corner of a convex curve and read high; this is the 12.7% that put the
    // ladder row at ratio 1.050.
    const float cubic = beam.eval(22.5f);
    const float linear = bilinearReference(beam.v, beam.h, beam.candela, 22.5f, 0.0f);
    CHECK(linear / cubic == doctest::Approx(1.127f).epsilon(0.02));

    // On the samples themselves the two agree -- the disagreement is about what
    // happens between them, not about the data.
    for (float deg : { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f })
    {
        CAPTURE(deg);
        CHECK(beam.eval(deg) == doctest::Approx(bilinearReference(beam.v, beam.h, beam.candela, deg, 0.0f)).epsilon(1e-4));
    }

    // And on a table that really is straight, the cubic is the straight line --
    // Catmull-Rom reproduces linear data exactly -- so the difference above is
    // about the curvature of this beam and not about the method.
    //
    // Interior intervals only. The first and last intervals have no fourth
    // sample to reach for: the evaluator substitutes the value across the pole
    // (or repeats the endpoint when the table does not reach one), which is
    // Cycles' choice and is why a linear ramp is reproduced to 4% rather than
    // exactly at its ends. Pinning those would pin the extrapolation policy, not
    // the interpolation.
    const std::vector<float> rampV{ 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f };
    const std::vector<float> rampH{ 0.0f, 360.0f };
    const std::vector<float> rampColumn{ 700.0f, 600.0f, 500.0f, 400.0f, 300.0f, 200.0f, 100.0f };
    std::vector<float> rampC = rampColumn;
    rampC.insert(rampC.end(), rampColumn.begin(), rampColumn.end());
    for (float deg : { 12.0f, 18.3f, 25.0f, 33.0f, 44.5f })
    {
        CAPTURE(deg);
        const float cubicRamp = iesEvaluate(rampV.data(), (int)rampV.size(), rampH.data(), 2, rampC.data(), deg, 0.0f);
        CHECK(cubicRamp == doctest::Approx(bilinearReference(rampV, rampH, rampC, deg, 0.0f)).epsilon(1e-4));
    }
}

TEST_CASE("the interpolation never returns a negative candela")
{
    // A cubic through non-negative samples can dip below zero on a steep flank,
    // and a negative candela is a light that subtracts from the image.
    const Beam beam;
    for (int i = 0; i <= 400; ++i)
    {
        const float deg = 40.0f * float(i) / 400.0f;
        CAPTURE(deg);
        CHECK(beam.eval(deg) >= 0.0f);
    }
}

// ---------------------------------------------------------------------------
// The edges of the table
// ---------------------------------------------------------------------------
TEST_CASE("outside the tabulated range the luminaire emits nothing")
{
    const Beam beam; // tabulated 0..40 degrees

    CHECK(beam.eval(40.0f) == doctest::Approx(18.1f));
    CHECK(beam.eval(40.1f) == 0.0f);
    CHECK(beam.eval(90.0f) == 0.0f);
    CHECK(beam.eval(180.0f) == 0.0f);

    // Extrapolating instead is what produced -1800 cd at 180 degrees on a
    // downlight whose table falls to zero, and +2800 on one that rises to its
    // last entry -- a lamp shining almost three times its own peak in a
    // direction the file says nothing about.
    CHECK(beam.eval(-1.0f) == 0.0f);
}

TEST_CASE("a table covering the full turn closes at the seam")
{
    // Two planes 180 degrees apart, plus the 360 duplicate the loader appends:
    // bright along 0, dark along 180.
    const std::vector<float> v{ 0.0f, 90.0f };
    const std::vector<float> h{ 0.0f, 90.0f, 180.0f, 270.0f, 360.0f };
    const std::vector<float> candela{ 1000.0f, 1000.0f, 500.0f, 500.0f, 100.0f, 100.0f, 500.0f, 500.0f, 1000.0f, 1000.0f };

    const auto at = [&](float azimuth) {
        return iesEvaluate(v.data(), (int)v.size(), h.data(), (int)h.size(), candela.data(), 45.0f, azimuth);
    };

    CHECK(at(0.0f) == doctest::Approx(1000.0f));
    CHECK(at(180.0f) == doctest::Approx(100.0f));

    // Across the seam the value has to be continuous: 359 and 1 degree sit on
    // either side of it and are the same direction to within a degree of the
    // peak. Without the wrap the cubic clamps at the last column and the seam
    // shows as a hard line in the render.
    CHECK(at(359.0f) == doctest::Approx(at(1.0f)).epsilon(0.02));
    CHECK(at(370.0f) == doctest::Approx(at(10.0f)).epsilon(1e-4));
    CHECK(at(-10.0f) == doctest::Approx(at(350.0f)).epsilon(1e-4));
}

TEST_CASE("a degenerate table evaluates to nothing rather than reading past itself")
{
    const std::vector<float> v{ 0.0f, 90.0f };
    const std::vector<float> h{ 0.0f };
    const std::vector<float> candela{ 1000.0f, 500.0f };
    // One azimuth column is not a table the evaluator can interpolate across;
    // the loader unfolds it to two before this is ever reached.
    CHECK(iesEvaluate(v.data(), 2, h.data(), 1, candela.data(), 45.0f, 0.0f) == 0.0f);
    CHECK(iesEvaluate(v.data(), 1, h.data(), 1, candela.data(), 45.0f, 0.0f) == 0.0f);
}

// ---------------------------------------------------------------------------
// End to end, through the loader
// ---------------------------------------------------------------------------
TEST_CASE("a quadrant-symmetric luminaire keeps its dark axis dark")
{
    // The loader unfolds one tabulated quadrant into the full turn by mirroring
    // it, which is what LM-63 means by the symmetry. Repeating it instead --
    // fmod(azimuth, 90) -- rotates the distribution by a quadrant, and against
    // the Cycles reference that showed up as 6.9x the error.
    oka::Scene::IesProfile p;
    p.verticalAngles = { 0.0f, 90.0f };
    p.horizontalAngles = { 0.0f, 45.0f, 90.0f };
    p.candela = { 1000.0f, 1000.0f, 500.0f, 500.0f, 0.0f, 0.0f };
    p.maxCandela = 1000.0f;
    oka::unfoldIesAzimuth(p);

    CHECK(oka::sampleIesCandela(p, photometricDir(45.0f, 0.0f)) == doctest::Approx(1000.0f));
    CHECK(oka::sampleIesCandela(p, photometricDir(45.0f, 90.0f)) == doctest::Approx(0.0f).epsilon(0.02));
    CHECK(oka::sampleIesCandela(p, photometricDir(45.0f, 180.0f)) == doctest::Approx(1000.0f));
    CHECK(oka::sampleIesCandela(p, photometricDir(45.0f, 270.0f)) == doctest::Approx(0.0f).epsilon(0.02));

    // Symmetric about both planes, everywhere -- not just at the tabulated
    // angles.
    for (int deg = 0; deg < 360; ++deg)
    {
        CAPTURE(deg);
        const float here = oka::sampleIesCandela(p, photometricDir(45.0f, float(deg)));
        CHECK(here == doctest::Approx(oka::sampleIesCandela(p, photometricDir(45.0f, float(-deg)))).epsilon(1e-3));
        CHECK(here == doctest::Approx(oka::sampleIesCandela(p, photometricDir(45.0f, float(180 - deg)))).epsilon(1e-3));
    }
}
