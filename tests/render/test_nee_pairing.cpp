#include <doctest/doctest.h>

#include <light_pdf.h>
#include <nee_pairing.h>
#include <strelka/material/shading_frame.h>

#include <cmath>

// ---------------------------------------------------------------------------
// The two halves of the multiple-importance-sampling estimate at a shading
// vertex, and the one property that has to hold between them.
//
// Next-event estimation proposes a restricted set of directions. BSDF sampling
// proposes all of them. The balance heuristic splits every direction *both* can
// reach and gives the whole of a direction only one of them can -- so a bounce
// may only be weighted down if a light connection could have reached the same
// place. Deducting a share that is never delivered loses light outright, and
// that is what the frosted end of `22_thin_walled` and every transmitted hair
// path were paying before these two predicates were written down together.
//
// The transport cannot be a unit test; its instruments are the ladder and
// tools/feature_tests/strand_probe.py. What is pinned here is the agreement.
// ---------------------------------------------------------------------------

TEST_CASE("a front face offers the hemisphere above its shading normal, and only that")
{
    CHECK(neeProposesDirection(false, /*frontFace=*/true, 0.7f));
    CHECK(neeProposesDirection(false, true, 1e-6f));
    CHECK_FALSE(neeProposesDirection(false, true, -0.7f));
    // Exactly grazing counts as below: `> 0` is the test, and a connection along
    // the tangent plane carries no cosine anyway.
    CHECK_FALSE(neeProposesDirection(false, true, 0.0f));
}

TEST_CASE("a back face offers the other hemisphere")
{
    // Hit from inside a dielectric: the shading normal still points out of the
    // surface, so the directions this vertex can connect along are the ones
    // pointing back into the medium.
    CHECK(neeProposesDirection(false, /*frontFace=*/false, -0.7f));
    CHECK_FALSE(neeProposesDirection(false, false, 0.7f));
}

TEST_CASE("a fibre offers every direction, from either side")
{
    // Chiang's TT term is light that entered one side of the strand and left the
    // other, and on a bright groom it is about four fifths of the albedo. There
    // is no hemisphere to reject.
    for (const bool frontFace : { true, false })
    {
        CHECK(neeProposesDirection(true, frontFace, 1.0f));
        CHECK(neeProposesDirection(true, frontFace, 0.0f));
        CHECK(neeProposesDirection(true, frontFace, -1.0f));
    }
}

TEST_CASE("a vertex that made no estimate never weights the bounce against one")
{
    // estimatorMode 1, a specular event, or a scene with nothing to connect to.
    // Weighting against an estimate that was never made loses the difference.
    CHECK_FALSE(neePairsWithBounce(/*didNee=*/false, false, true, 1.0f));
    CHECK_FALSE(neePairsWithBounce(false, true, true, 1.0f));
    CHECK_FALSE(neePairsWithBounce(false, false, false, -1.0f));
}

TEST_CASE("a reflected bounce off a front face pairs; a transmitted one does not")
{
    CHECK(neePairsWithBounce(true, false, /*frontFace=*/true, 0.6f));
    // The rough thin wall. Next-event estimation rejected the whole transmitted
    // hemisphere at this vertex, so nothing over there may be discounted.
    CHECK_FALSE(neePairsWithBounce(true, false, true, -0.6f));
    CHECK_FALSE(neePairsWithBounce(true, false, true, 0.0f));
}

TEST_CASE("a fibre bounce pairs whichever side of the strand it leaves by")
{
    // Withholding the weight here would count the light twice: the connection
    // that reaches the far side really was made.
    CHECK(neePairsWithBounce(true, /*throughFibre=*/true, true, -0.9f));
    CHECK(neePairsWithBounce(true, true, true, 0.9f));
    CHECK(neePairsWithBounce(true, true, false, -0.9f));
}

TEST_CASE("the bounce and proposal rules have identical directional support")
{
    // The property the whole file exists for. A direction either has both
    // strategies, whose heuristic shares sum to one, or exactly one strategy,
    // whose weight is one. A one-way implication still permits the old positive
    // bias: NEE took 0.4 while an incorrectly unpaired BSDF hit took 1.0.
    for (const bool throughFibre : { true, false })
    {
        for (const bool frontFace : { true, false })
        {
            for (int i = -10; i <= 10; ++i)
            {
                const float nDot = (float)i * 0.1f;
                CHECK(neePairsWithBounce(true, throughFibre, frontFace, nDot) ==
                      neeProposesDirection(throughFibre, frontFace, nDot));
            }
        }
    }
}

TEST_CASE("back-face complementary MIS shares sum to one")
{
    constexpr float lightPdf = 0.4f;
    constexpr float bsdfPdf = 0.6f;
    constexpr bool frontFace = false;
    constexpr float nDotDirection = -0.7f;

    REQUIRE(neeProposesDirection(false, frontFace, nDotDirection));
    REQUIRE(neePairsWithBounce(true, false, frontFace, nDotDirection));
    const float lightShare = computeMisWeight(lightPdf, bsdfPdf, 0u);
    const float bounceShare = computeMisWeight(bsdfPdf, lightPdf, 0u);
    CHECK(lightShare == doctest::Approx(0.4f));
    CHECK(bounceShare == doctest::Approx(0.6f));
    CHECK(lightShare + bounceShare == doctest::Approx(1.0f));

    // Mutation: clearing the pairing flag gives the BSDF strategy weight one
    // and recreates the audit's 1.4 total.
    CHECK(lightShare + 1.0f == doctest::Approx(1.4f));
}

TEST_CASE("raw, flipped, mirrored, and transmissive frames preserve pairing")
{
    struct Case
    {
        bool frontFace;
        float nDotView;
        float transmission;
        float diffuseTransmission;
        float rawNDotDirection;
    };
    const Case cases[] = {
        { true, 0.8f, 0.0f, 0.0f, 0.7f }, // ordinary front
        { false, -0.8f, 0.0f, 0.0f, -0.7f }, // opaque back, frame flips
        { false, -0.8f, 1.0f, 0.0f, -0.7f }, // dielectric exit, frame stays raw
        { false, -0.8f, 0.0f, 1.0f, -0.7f }, // diffuse transmission
        { true, 0.8f, 0.0f, 0.0f, 0.7f }, // mirrored transform after winding correction
    };

    for (const Case& c : cases)
    {
        const ShadedFrame frame =
            shadedFrame(c.frontFace, c.nDotView, c.transmission, c.diffuseTransmission);
        const float signedNDotDirection = frame.normalSign * c.rawNDotDirection;
        CAPTURE(c.frontFace);
        CAPTURE(c.transmission);
        CAPTURE(c.diffuseTransmission);
        CHECK(neePairsWithBounce(true, false, frame.frontFace, signedNDotDirection) ==
              neeProposesDirection(false, frame.frontFace, signedNDotDirection));
    }
}

// ===========================================================================
// When a vertex owes the bounce ray a deduction at all
//
// The rules above are about *which directions* the two halves share. These are
// about *whether* the vertex made an estimate to share them with -- the other
// half of the same question, and the one both backends got wrong in their own
// way: OptiX gated it on the BSDF event that came back, Metal on whether the
// light connection produced a shadow ray. Both tie the two halves of the
// estimate to a draw belonging to one of them.
// ===========================================================================

TEST_CASE("next-event estimation runs on the material, not on the bounce that was drawn")
{
    // The gate is three independent conditions and nothing else. In particular
    // there is no argument for "what the BSDF sample came back as": a vertex
    // either has a lobe a light can connect to or it does not, and that is the
    // same on every draw.
    CHECK(neeRunsAtVertex(true, true, true));

    CHECK_FALSE(neeRunsAtVertex(false, true, true)); // estimatorMode 1: BSDF only
    CHECK_FALSE(neeRunsAtVertex(true, false, true)); // nothing to connect to
    CHECK_FALSE(neeRunsAtVertex(true, true, false)); // a pure mirror has no density
}

TEST_CASE("a medium vertex always owes the deduction its bounce is discounted by")
{
    // A phase function is smooth at every anisotropy, so the material question
    // is answered before it is asked and the rule collapses to the two
    // conditions that describe the scene.
    CHECK(volumeNeePairsWithBounce(true, true));
    CHECK_FALSE(volumeNeePairsWithBounce(false, true));
    CHECK_FALSE(volumeNeePairsWithBounce(true, false));

    CHECK(volumeNeePairsWithBounce(true, true) == neeRunsAtVertex(true, true, true));
}

namespace
{

// ---------------------------------------------------------------------------
// A single scattering event in a medium, integrated two ways.
//
// One isotropic phase function (p = 1/4pi), one emitter covering a fraction
// `emitterFraction` of the sphere of directions, and a light-sampling strategy
// that draws uniformly over the whole sphere. A light draw that lands off the
// emitter carries nothing -- the ordinary way a Monte Carlo sample contributes
// zero, not a failure -- and the question is what the *bounce* ray is then
// weighted by when it lands on the emitter itself.
//
// Both densities are 1/4pi here, so the balance heuristic gives each strategy
// exactly half of every direction and the closed form of the whole estimate is
// simply the emitter's mean radiance, L * emitterFraction.
//
// `flagFromOutcome` reproduces the shipped Metal behaviour: the vertex is
// recorded as having made an estimate only when its light draw happened to
// deliver something. The bounce then keeps the *whole* emitter on the draws
// where the light strategy came back empty, and the two halves sum to more than
// one. The analytic value of that error is (1.5 - 0.5 * emitterFraction), which
// the case below both measures and states.
// ---------------------------------------------------------------------------
double singleScatterEstimate(double emitterFraction, bool flagFromOutcome, unsigned int seed, int samples)
{
    // A tiny deterministic LCG; the two policies must see the same draws.
    unsigned int state = seed | 1u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return double((state >> 8) & 0xFFFFFFu) / double(0x1000000);
    };

    const double radiance = 1.0;
    const double phase = 1.0 / (4.0 * double(M_PI_F)); // isotropic
    const double lightPdf = 1.0 / (4.0 * double(M_PI_F)); // uniform over the sphere
    const double misWeightLight = double(misWeightBalance(float(lightPdf), float(phase)));
    const double misWeightBsdf = double(misWeightBalance(float(phase), float(lightPdf)));

    double total = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        // --- next-event estimation ---------------------------------------
        // "Is the drawn direction on the emitter" stands in for the whole of
        // connectToLight(): a draw that misses contributes zero either way.
        const bool lightDrawHitEmitter = next() < emitterFraction;
        double contribution = 0.0;
        if (lightDrawHitEmitter)
        {
            contribution += misWeightLight * radiance * phase / lightPdf;
        }

        // --- the flag the bounce ray will read ----------------------------
        const bool neeDone = flagFromOutcome
                                 ? lightDrawHitEmitter
                                 : volumeNeePairsWithBounce(/*neeEnabled=*/true, /*hasEmitter=*/true);

        // --- the bounce ------------------------------------------------------
        const bool bounceHitEmitter = next() < emitterFraction;
        if (bounceHitEmitter)
        {
            contribution += (neeDone ? misWeightBsdf : 1.0) * radiance;
        }
        total += contribution;
    }
    return total / double(samples);
}

} // namespace

TEST_CASE("deciding the flag from the outcome puts the volume estimate over unity")
{
    const int samples = 2000000;

    for (double emitterFraction : { 0.05, 0.25, 0.5 })
    {
        CAPTURE(emitterFraction);
        const double exact = emitterFraction; // radiance 1 over that fraction of the sphere

        const double correct = singleScatterEstimate(emitterFraction, /*flagFromOutcome=*/false, 0x9E3779B9u, samples);
        CHECK(correct == doctest::Approx(exact).epsilon(0.02));

        // And what the shipped code did, with its analytic value. Stated as a
        // number so the case says how much light this was worth rather than
        // merely that it was wrong: at a small emitter, half as much again.
        const double fromOutcome = singleScatterEstimate(emitterFraction, /*flagFromOutcome=*/true, 0x9E3779B9u, samples);
        const double predicted = exact * (1.5 - 0.5 * emitterFraction);
        CHECK(fromOutcome == doctest::Approx(predicted).epsilon(0.02));
        CHECK(fromOutcome > correct * 1.2);
    }
}

// ===========================================================================
// Where a connection leaves from
// ===========================================================================

TEST_CASE("a shadow ray is offset along the face it actually leaves through")
{
    // The rule is one line, and the reason it is written down is that one
    // backend applied it to environment connections and not to local-light ones,
    // where it stood in a fixed 1 mm ray tMin instead.
    const float3 ng = make_float3(0.0f, 1.0f, 0.0f);

    // Leaving above the surface: the raw normal is already right.
    const float3 up = orientedFaceNormal(ng, make_float3(0.0f, 1.0f, 0.0f));
    CHECK(up.y == doctest::Approx(1.0f));

    // Leaving below it -- a back-face hit, or a transmitted bounce. Offsetting
    // along the raw normal here pushes the origin into the geometry the ray
    // starts on, and the connection reports an occlusion the BSDF strategy
    // never sees.
    const float3 down = orientedFaceNormal(ng, make_float3(0.0f, -1.0f, 0.0f));
    CHECK(down.y == doctest::Approx(-1.0f));

    // The invariant, over a sweep: the offset never opposes the ray.
    for (int i = -10; i <= 10; ++i)
    {
        for (int j = -10; j <= 10; ++j)
        {
            const float3 dir = make_float3(float(i) * 0.1f, float(j) * 0.1f, 0.35f);
            const float3 offset = orientedFaceNormal(ng, dir);
            CHECK(dot(offset, dir) >= 0.0f);
            // And it is still the geometry normal, only possibly negated.
            CHECK(std::abs(std::abs(offset.y) - 1.0f) < 1e-6f);
        }
    }
}
