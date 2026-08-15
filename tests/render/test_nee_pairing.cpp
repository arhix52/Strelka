#include <doctest/doctest.h>

#include <shading/nee_pairing.h>

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

TEST_CASE("the bounce rule never claims a pairing the proposal rule would refuse")
{
    // The property the whole file exists for. Whenever a bounce is weighted down
    // against next-event estimation, next-event estimation had to be willing to
    // propose that same direction -- otherwise the deduction has no counterpart.
    // The converse is allowed to fail, and does, on a back face: see the note on
    // neePairsWithBounce().
    for (const bool throughFibre : { true, false })
    {
        for (const bool frontFace : { true, false })
        {
            for (int i = -10; i <= 10; ++i)
            {
                const float nDot = (float)i * 0.1f;
                if (neePairsWithBounce(true, throughFibre, frontFace, nDot))
                {
                    CHECK(neeProposesDirection(throughFibre, frontFace, nDot));
                }
            }
        }
    }
}
