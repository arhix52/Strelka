#ifndef STRELKA_OPTIX_NEE_PAIRING_H
#define STRELKA_OPTIX_NEE_PAIRING_H

// ============================================================================
// nee_pairing.h -- which directions the two halves of the MIS estimate share.
//
// Multiple importance sampling only works while both strategies agree on the
// set of directions they are competing over. Next-event estimation and BSDF
// sampling each get a share of every direction *both* can produce, and the whole
// of a direction only one of them can; the balance heuristic is the split, and
// the shares have to sum to one.
//
// Next-event estimation here does not propose every direction. It offers the
// ones on the side of the shading normal the surface was hit from and nothing on
// the other -- a light behind a wall is not connected to through it. So a bounce
// leaving into the other hemisphere is a direction next-event estimation could
// not have produced, and deducting a share for it is a deduction that is never
// delivered. The light lands once, weighted below one, and the missing part is
// simply lost.
//
// That is not a hypothetical: `28_hair` and the rough end of `22_thin_walled`
// were both paying it. The two predicates were written out by hand at opposite
// ends of the closest-hit program and quietly disagreed, which is the whole
// failure mode, so they live here together and are asserted against each other
// in tests/render/test_nee_pairing.cpp.
//
// Deliberately free of CUDA and OptiX: plain booleans and one float, so the
// closest-hit program and the unit tests compile the same code.
// ============================================================================

#include <strelka/material/material_math.h>

/// True when next-event estimation at this vertex is willing to propose a
/// direction `L`, given `nDotL = dot(shading_normal, L)`.
///
/// A fibre has no side that light does not reach. Chiang's TT and TRT terms are
/// light that entered one side of the strand and left the other, and TT alone is
/// about four fifths of a bright strand's albedo, so testing the shading
/// hemisphere the way a surface does discards the dominant lobe.
DEVICE_FUNC bool neeProposesDirection(bool throughFibre, bool frontFace, float nDotL)
{
    return throughFibre || ((nDotL > 0.0f) == frontFace);
}

/// True when a bounce leaving along `dir`, with `nDotDir = dot(shading_normal,
/// dir)`, may be weighted against next-event estimation when it lands on a
/// light. `didNee` is whether this vertex made an estimate at all.
///
/// Note the asymmetry against neeProposesDirection() on a *back* face: this is
/// the expression Metal's `wavefrontShade` applies, and it withholds the weight
/// from every back-face bounce rather than only from the ones that cross the
/// normal. A path leaving a back face is a path on its way out of a dielectric,
/// where the connection the proposal above would allow -- back into the medium
/// it is inside -- is blocked by the far wall of that same medium in every
/// measured scene, so the share it would claim is one it does not deliver
/// either. Keeping the two backends on one rule is worth more here than the
/// difference, which no ladder row can see.
DEVICE_FUNC bool neePairsWithBounce(bool didNee, bool throughFibre, bool frontFace, float nDotDir)
{
    return didNee && (throughFibre || (frontFace && nDotDir > 0.0f));
}

#endif // STRELKA_OPTIX_NEE_PAIRING_H
