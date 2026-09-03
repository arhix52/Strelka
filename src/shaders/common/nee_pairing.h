#ifndef STRELKA_NEE_PAIRING_H
#define STRELKA_NEE_PAIRING_H

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
// Deliberately free of CUDA, OptiX and Metal: plain booleans and one float, so
// the closest-hit program, the wavefront kernels and the unit tests compile the
// same code. It lives in shaders/common and not under one backend for that
// reason -- Metal used to restate these rules by hand, which is how the two
// ended up disagreeing about volumes (see volumeNeePairsWithBounce below).
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
/// This support must be exactly neeProposesDirection(), gated only by whether
/// NEE ran. A one-way implication is insufficient: Metal used to accept a
/// back-face NEE direction but withhold the complementary weight when the BSDF
/// generated the same direction. The resulting balance shares were 0.4 + 1.0.
DEVICE_FUNC bool neePairsWithBounce(bool didNee, bool throughFibre, bool frontFace, float nDotDir)
{
    return didNee && neeProposesDirection(throughFibre, frontFace, nDotDir);
}

/// Whether next-event estimation runs at this vertex at all.
///
/// `materialHasSmoothLobe` is a property of the material and is the same at
/// every draw. That is the whole point of the argument: the test used to be
/// whether the BSDF *sample* had come back non-delta, which ties the two halves
/// of the estimate to one shared draw. On a material carrying both a delta lobe
/// and a smooth one -- a clearcoat over a diffuse base, and glTF's default coat
/// roughness is 0 -- the smooth lobe's direct light was then delivered only on
/// the draws that missed the delta lobe and the rest was lost. Measured as the
/// fraction of samples flagged specular: 52% on a 0.18 grey base under a smooth
/// coat, 73% on a dark lacquered paint, 18% on plastic at roughness 0.02.
///
/// Deciding it this way costs nothing when the material really is a pure
/// mirror: no smooth lobe, no connection, and the bounce keeps the whole
/// contribution through the specular exemption at the light hit.
DEVICE_FUNC bool neeRunsAtVertex(bool neeEnabled, bool hasEmitter, bool materialHasSmoothLobe)
{
    return neeEnabled && hasEmitter && materialHasSmoothLobe;
}

/// Whether a scattering event *inside a medium* owes the bounce ray a MIS
/// deduction when it lands on a light or the environment.
///
/// Note what this does not take: whether the light connection at that vertex
/// produced anything. Deciding it from the outcome hands the bounce ray the
/// whole contribution on exactly the draws where the connection failed -- a
/// light sampled behind the vertex, a candidate whose weight underflowed -- and
/// the two strategies stop summing to one. It put the white-furnace sphere 15%
/// over unity, uniformly at every mean free path. What the vertex owes is
/// decided by what was *available* there, which is the same on every draw.
///
/// A phase function is smooth at every anisotropy, so a medium always has the
/// lobe that surfaces have to be asked about.
DEVICE_FUNC bool volumeNeePairsWithBounce(bool neeEnabled, bool hasEmitter)
{
    return neeRunsAtVertex(neeEnabled, hasEmitter, true);
}

/// The geometric normal a ray leaving along `dir` should be offset along.
///
/// The raw geometry normal points to a fixed side of the triangle, so on a
/// back-face hit offsetting along it pushes the origin *into* the surface the
/// ray starts on. The shadow ray then reports occlusion the BSDF strategy does
/// not see, and the two halves of the estimate stop summing to the integral.
/// Both the bounce ray and the light connection have to orient it the same way,
/// which is why it is written once: Metal's local-light connections used to skip
/// the offset entirely and lean on a fixed 1 mm ray tMin instead -- a world-space
/// constant, so too small at architectural scale and too large at prop scale.
DEVICE_FUNC float3 orientedFaceNormal(float3 geometryNormal, float3 dir)
{
    return (dot(geometryNormal, dir) > 0.0f) ? geometryNormal : -geometryNormal;
}

#endif // STRELKA_NEE_PAIRING_H
