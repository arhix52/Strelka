// The arithmetic of the spectral free-flight walk that serves both subsurface
// scattering and a bounded fog volume on the OptiX backend.
//
// Three of these bugs cost 25_subsurface a factor and none of them was visible
// as a crash:
//
//   * choosing the driving channel uniformly rather than in proportion to what
//     the path carries, which lets a balance-heuristic weight exceed one and
//     compounds into fireflies over a walk tens of steps long;
//   * weighting by a distribution other than the one the channel was drawn
//     from, which is how an unbiased estimator stops being one;
//   * dropping the boundary weight, which is exactly 1 for a grey medium -- so
//     a fog gizmo never notices -- and is not 1 at all for mean free paths that
//     differ threefold, which is what that scene's spheres are.
//
// The check that catches all three is the estimator identity: summed over the
// two outcomes a free flight can have, with each weighted by the density it was
// actually drawn from, the walk has to transport exactly the medium's albedo
// per event and exactly the transmittance to the boundary. That is a numerical
// integral over the sampled distance, and it needs no GPU.

#include <doctest/doctest.h>

#include <medium_walk.h>

#include "../support/sampling.h"

#include <cmath>
#include <initializer_list>
#include <numbers>

using namespace oka::medium;
using oka::test::stratum;

namespace
{

Spectrum spec(float x, float y, float z)
{
    return makeSpectrum(x, y, z);
}

/// The distance a canonical sample maps to, for the channel it selects.
float distanceFor(const Spectrum& sigmaT, const Spectrum& pdf, float uChannel, float uDist)
{
    float t = 0.0f;
    sampleDistance(sigmaT, pdf, 1e30f, uChannel, uDist, t);
    return t;
}

} // namespace

TEST_CASE("medium: extinction is the reciprocal of the mean free path")
{
    const Spectrum s = sigmaTFromRadius(spec(0.05f, 0.025f, 0.015f));
    CHECK(s.x == doctest::Approx(20.0f));
    CHECK(s.y == doctest::Approx(40.0f));
    CHECK(s.z == doctest::Approx(1.0f / 0.015f));

    // A zero radius is a medium of infinite density, which has to come back as a
    // large finite number rather than as an infinity that turns the throughput
    // into a NaN two multiplies later.
    const Spectrum degenerate = sigmaTFromRadius(spec(0.0f, -1.0f, 1e-9f));
    CHECK(std::isfinite(degenerate.x));
    CHECK(std::isfinite(degenerate.y));
    CHECK(std::isfinite(degenerate.z));
    CHECK(degenerate.x == doctest::Approx(1e5f));
}

TEST_CASE("medium: the driving channel is chosen by throughput times albedo")
{
    // Red carries three times what green does and green twice what blue does.
    const Spectrum pdf = channelPdf(spec(1.0f, 0.5f, 0.25f), spec(0.6f, 0.6f, 0.6f));
    CHECK(pdf.x + pdf.y + pdf.z == doctest::Approx(1.0f));
    CHECK(pdf.x == doctest::Approx(4.0f / 7.0f));
    CHECK(pdf.y == doctest::Approx(2.0f / 7.0f));
    CHECK(pdf.z == doctest::Approx(1.0f / 7.0f));

    // A path that carries nothing still has to select *something*: a zero
    // distribution would divide by zero in the weights below.
    const Spectrum dead = channelPdf(spec(0.0f, 0.0f, 0.0f), spec(1.0f, 1.0f, 1.0f));
    CHECK(dead.x == doctest::Approx(1.0f / 3.0f));
    CHECK(dead.y == doctest::Approx(1.0f / 3.0f));
    CHECK(dead.z == doctest::Approx(1.0f / 3.0f));

    // Sign is not information here -- a throughput can go slightly negative
    // through a filtered texture -- so the magnitude drives the choice.
    const Spectrum signed_ = channelPdf(spec(-1.0f, 0.0f, 0.0f), spec(1.0f, 1.0f, 1.0f));
    CHECK(signed_.x == doctest::Approx(1.0f));
}

TEST_CASE("medium: selectChannel partitions the unit interval by the pdf")
{
    const Spectrum pdf = spec(0.5f, 0.3f, 0.2f);
    CHECK(selectChannel(pdf, 0.0f) == 0);
    CHECK(selectChannel(pdf, 0.4999f) == 0);
    CHECK(selectChannel(pdf, 0.5001f) == 1);
    CHECK(selectChannel(pdf, 0.7999f) == 1);
    CHECK(selectChannel(pdf, 0.8001f) == 2);
    CHECK(selectChannel(pdf, 0.9999f) == 2);

    // A channel of zero probability must never be selected, or the distance
    // below is drawn from an extinction the weights give no density to.
    const Spectrum edge = spec(0.0f, 1.0f, 0.0f);
    CHECK(selectChannel(edge, 0.0f) == 1);
    CHECK(selectChannel(edge, 0.9999f) == 1);
}

TEST_CASE("medium: the sampled distance inverts the selected channel's transmittance")
{
    const Spectrum sigmaT = spec(20.0f, 40.0f, 1.0f / 0.015f);
    const Spectrum pdf = spec(1.0f, 0.0f, 0.0f); // force the red channel

    // exp(-sigma_t * t) == 1 - u by construction.
    for (const float u : { 0.05f, 0.25f, 0.5f, 0.75f, 0.95f })
    {
        const float t = distanceFor(sigmaT, pdf, 0.0f, u);
        CHECK(std::exp(-sigmaT.x * t) == doctest::Approx(1.0f - u).epsilon(1e-4));
    }

    // u == 0 is a legitimate draw from a stratified sampler and log(0) is not a
    // number, so the largest draw has to stay finite.
    const float far = distanceFor(sigmaT, pdf, 0.0f, 1.0f);
    CHECK(std::isfinite(far));
    CHECK(far > 0.0f);

    // Nothing scatters in a vacuum.
    float t = -1.0f;
    CHECK_FALSE(sampleDistance(spec(0.0f, 0.0f, 0.0f), pdf, 1e30f, 0.0f, 0.5f, t));
}

TEST_CASE("medium: sampleDistance reports whether the event precedes the surface")
{
    const Spectrum sigmaT = spec(10.0f, 10.0f, 10.0f);
    const Spectrum pdf = spec(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    float t = 0.0f;
    // The median free flight of a sigma_t of 10 is ln(2)/10 = 0.069.
    CHECK(sampleDistance(sigmaT, pdf, 1.0f, 0.5f, 0.5f, t));
    CHECK(t == doctest::Approx(std::log(2.0f) / 10.0f).epsilon(1e-4));
    CHECK_FALSE(sampleDistance(sigmaT, pdf, 0.01f, 0.5f, 0.5f, t));
}

TEST_CASE("medium: a grey extinction leaves both weights at their scalar values")
{
    // This is the case a fog gizmo of uniform density is, and it is why omitting
    // the boundary weight is invisible on 18_bounded_volume: it is exactly one.
    const Spectrum sigmaT = spec(2.5f, 2.5f, 2.5f);
    const Spectrum albedo = spec(0.75f, 0.82f, 0.95f);
    const Spectrum pdf = channelPdf(spec(1.0f, 1.0f, 1.0f), albedo);

    const Spectrum sw = scatterWeight(sigmaT, albedo, pdf, 0.37f);
    CHECK(sw.x == doctest::Approx(albedo.x));
    CHECK(sw.y == doctest::Approx(albedo.y));
    CHECK(sw.z == doctest::Approx(albedo.z));

    for (const float t : { 0.0f, 0.1f, 1.0f, 5.0f })
    {
        const Spectrum bw = boundaryWeight(sigmaT, pdf, t);
        CHECK(bw.x == doctest::Approx(1.0f));
        CHECK(bw.y == doctest::Approx(1.0f));
        CHECK(bw.z == doctest::Approx(1.0f));
    }
}

TEST_CASE("medium: a coloured extinction does not leave the boundary weight at one")
{
    // 25_subsurface's mean free paths. The channel that travels furthest is the
    // one that reaches a distant boundary, and it has to be credited for it.
    const Spectrum sigmaT = sigmaTFromRadius(spec(0.05f, 0.025f, 0.015f));
    const Spectrum pdf = spec(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    const Spectrum bw = boundaryWeight(sigmaT, pdf, 0.1f);
    CHECK(bw.x > 1.0f); // red survives the crossing more often than the mean
    CHECK(bw.z < 1.0f); // blue much less
    CHECK(bw.x > bw.y);
    CHECK(bw.y > bw.z);
}

TEST_CASE("medium: the estimator transports exactly the albedo and the transmittance")
{
    // The identity that makes the walk unbiased, integrated numerically over the
    // distance the free flight draws:
    //
    //   E[scatterWeight] over the scattering outcomes  ==  albedo
    //   E[boundaryWeight] over the reaching outcome    ==  exp(-sigma_t * L)
    //
    // both per channel, where the expectation is over the *mixture* density the
    // channel selection produces. Weighting by any other distribution than the
    // one the channel was drawn from breaks the first of these, which is the bug
    // this test exists for.
    const Spectrum sigmaT = sigmaTFromRadius(spec(0.05f, 0.025f, 0.015f));
    const Spectrum albedo = spec(0.93f, 0.65f, 0.58f);
    const Spectrum throughput = spec(1.0f, 0.7f, 0.4f);
    const Spectrum pdf = channelPdf(throughput, albedo);

    const float L = 0.08f; // distance to the boundary
    const int kSteps = 200000;

    double scattered[3] = { 0.0, 0.0, 0.0 };
    double reached[3] = { 0.0, 0.0, 0.0 };

    for (int c = 0; c < 3; ++c)
    {
        const float sc = channel(sigmaT, c);
        const float pc = channel(pdf, c);
        if (!(pc > 0.0f))
        {
            continue;
        }
        // Midpoint rule over u in [0, 1) for this channel's inversion, which is
        // an exact change of variables: t = -ln(1-u)/sigma_c.
        for (int i = 0; i < kSteps; ++i)
        {
            const float u = stratum(i, kSteps);
            const float t = -std::log(1.0f - u) / sc;
            if (t < L)
            {
                const Spectrum w = scatterWeight(sigmaT, albedo, pdf, t);
                scattered[0] += pc * w.x / kSteps;
                scattered[1] += pc * w.y / kSteps;
                scattered[2] += pc * w.z / kSteps;
            }
            else
            {
                const Spectrum w = boundaryWeight(sigmaT, pdf, L);
                reached[0] += pc * w.x / kSteps;
                reached[1] += pc * w.y / kSteps;
                reached[2] += pc * w.z / kSteps;
            }
        }
    }

    for (int c = 0; c < 3; ++c)
    {
        const float sc = channel(sigmaT, c);
        const float transmittance = std::exp(-sc * L);
        // Everything that did not scatter reached the boundary, so the two
        // outcomes have to add up to the medium's albedo plus its transmittance
        // -- no more, which would be energy the walk invented, and no less,
        // which is the colour a translucent object loses.
        CHECK(scattered[c] == doctest::Approx(channel(albedo, c) * (1.0 - transmittance)).epsilon(2e-3));
        CHECK(reached[c] == doctest::Approx(transmittance).epsilon(2e-3));
    }
}

TEST_CASE("medium: choosing channels uniformly is what makes the weights blow up")
{
    // The uniform choice is unbiased -- the identity above still holds for it --
    // and unusable, because the individual weights are unbounded where the
    // proportional choice keeps them near one. That is the difference between a
    // converging walk and a field of fireflies, and it is not something a mean
    // can show.
    const Spectrum sigmaT = sigmaTFromRadius(spec(0.05f, 0.025f, 0.015f));
    const Spectrum albedo = spec(0.93f, 0.65f, 0.58f);
    const Spectrum uniform = spec(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    const Spectrum proportional = channelPdf(spec(1.0f, 0.15f, 0.02f), albedo);

    // A distance only the slowest-decaying channel would plausibly produce.
    const float t = 0.25f;
    const Spectrum wUniform = scatterWeight(sigmaT, albedo, uniform, t);
    const Spectrum wProportional = scatterWeight(sigmaT, albedo, proportional, t);
    CHECK(wUniform.x > 2.0f);
    CHECK(wProportional.x < wUniform.x);
}

TEST_CASE("medium: Henyey-Greenstein integrates to one over the sphere")
{
    for (const float g : { -0.8f, -0.3f, 0.0f, 0.3f, 0.8f })
    {
        // 2*pi * integral over cos(theta) in [-1, 1].
        const int kSteps = 200000;
        double total = 0.0;
        for (int i = 0; i < kSteps; ++i)
        {
            const float mu = -1.0f + 2.0f * stratum(i, kSteps);
            total += hgPhase(mu, g) * (2.0 / kSteps);
        }
        total *= 2.0 * std::numbers::pi;
        CHECK(total == doctest::Approx(1.0).epsilon(1e-3));
    }
}

TEST_CASE("medium: the HG inversion returns the cosine against the direction of travel")
{
    // Taken at face value the standard inversion gives the cosine against the
    // direction the ray came *from*, so a forward-scattering medium reads as a
    // backward-scattering one -- which does not show up in single scattering,
    // where the outgoing direction is fixed by the camera, and is worth sixty
    // times the first event by the second.
    //
    // Forward scattering must put the median draw near +1.
    CHECK(hgSampleCosine(0.8f, 0.5f) > 0.9f);
    CHECK(hgSampleCosine(-0.8f, 0.5f) < -0.9f);
    // Isotropic is the linear map, and its median is zero.
    CHECK(hgSampleCosine(0.0f, 0.5f) == doctest::Approx(0.0f));
    CHECK(hgSampleCosine(0.0f, 0.0f) == doctest::Approx(1.0f));
    CHECK(hgSampleCosine(0.0f, 1.0f) == doctest::Approx(-1.0f));
    // And nothing may leave the valid range, whatever the draw.
    for (const float g : { -0.99f, 0.0f, 0.99f })
    {
        for (const float u : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f })
        {
            const float mu = hgSampleCosine(g, u);
            CHECK(mu >= -1.0f);
            CHECK(mu <= 1.0f);
        }
    }
}

TEST_CASE("medium: the spectrum is layout-compatible with a device float3")
{
    // The device wrapper in src/shaders/optix/shading/medium.h reinterprets
    // rather than converts, and the backend asserts the same thing against the
    // real float3. Three floats, in order, no padding.
    static_assert(sizeof(Spectrum) == 3 * sizeof(float), "Spectrum must be three tight floats");
    const Spectrum s = makeSpectrum(1.0f, 2.0f, 3.0f);
    const float* raw = &s.x;
    CHECK(raw[0] == 1.0f);
    CHECK(raw[1] == 2.0f);
    CHECK(raw[2] == 3.0f);
    CHECK(channel(s, 0) == 1.0f);
    CHECK(channel(s, 1) == 2.0f);
    CHECK(channel(s, 2) == 3.0f);
    CHECK(dot3(s, makeSpectrum(1.0f, 1.0f, 1.0f)) == doctest::Approx(6.0f));
}
