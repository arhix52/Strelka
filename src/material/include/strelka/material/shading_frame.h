#ifndef STRELKA_MATERIAL_SHADING_FRAME_H
#define STRELKA_MATERIAL_SHADING_FRAME_H

// ============================================================================
// shading_frame.h -- which side of an opaque surface is the shaded one.
//
// Nothing in this renderer culls a back face. Every triangle is traced from
// both sides, so a hit with the shading normal pointing away from the viewer is
// routine rather than exceptional, and it arrives three ways:
//
//   * open geometry -- a leaf card, a cloth panel, a rock decimated to a shell.
//     There is no interior to be inside of; the far side is the same surface.
//   * inconsistent winding, which scanned and LOD-collapsed assets carry as a
//     matter of course.
//   * a normal map steep enough to tip the shading normal past the horizon at a
//     grazing angle, while the triangle itself still faces the camera.
//
// The lobes cannot answer any of those. dot(N, wo) < 0 is also how a ray leaving
// a dielectric announces itself, and standard_pbr reads it that way: it routes
// the hit to the transmission lobe, which flips the normal itself and picks eta
// by direction. An opaque material has no such lobe, so the hit used to return
// BSDF_EVENT_ABSORB from sample and zero from eval -- and, because the
// closest-hit program returned on absorb before reaching next-event estimation,
// the pixel lost its direct lighting too and came out exactly black.
//
// Measured on the pine forest at one sample: 40331 primary hits, 13.2% of the
// frame, absorbed with the shading normal below the geometric horizon, 99.7% of
// them on back faces of double-sided foliage. The rock the report came in about
// absorbed on front-facing triangles instead, where the normal map alone had
// tipped the normal past the viewer.
//
// So: an opaque surface seen from behind is shaded as the same surface seen
// from the front. That is what glTF's `doubleSided` means, and with no culling
// anywhere it is the only reading that leaves the far side of a leaf lit.
//
// Deliberately a predicate rather than a flip in the caller: standard_pbr_sample
// and standard_pbr_eval have to make this decision identically or MIS blends two
// different BRDFs, and the way that goes wrong is one of them being edited. The
// pair is asserted against each other in tests/material/test_opaque_back_face.cpp.
// ============================================================================

#include <strelka/material/material_math.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)
//
// Device-shared header: NVCC and the Metal compiler read this too, and
// clang-tidy only ever sees the host build, so these two suggestions cannot be
// taken here. Initialising the locals means a dead store in a BSDF inner loop --
// they are out-parameters written on the next line -- and the fixer spells the
// initialiser NAN, which needs <math.h>, which Metal rejects outright. Default
// member initialisers do the same to structs that are memcpy'd to the GPU.
// Suppressed rather than left to warn because these repeat in every translation
// unit that includes the header, and 700 lines of unactionable output per build
// is how the handful that matter get skipped.

/// True when a hit with `nDotV = dot(shading_normal, wo)` should be shaded with
/// the shading frame flipped to face the viewer.
///
/// Gated on the *geometry* being back-facing, not only on the shading normal
/// pointing away, and the difference is the whole scope of this file. Both
/// arrive as nDotV <= 0 and they are not the same defect:
///
///   * `frontFace == false` is a surface whose far side is being looked at.
///     There is no interior to be inside of, so the far side is the near side
///     and flipping is what it means to shade it at all.
///   * `frontFace == true` with nDotV <= 0 is a triangle facing the camera
///     whose *normal map* tipped past the horizon at a grazing angle. Flipping
///     that shades a surface pointing away from the light it is lit by, and the
///     ladder says so: it takes 06_normalmap from 1.061 to 1.079 against Cycles
///     while every other row stays put. That case belongs to whatever this
///     renderer eventually does about below-horizon shading normals -- Cycles
///     corrects the normal until the reflection clears the surface -- and it is
///     deliberately left alone here.
///
/// Measured before it was scoped this way: of the primary hits absorbing with
/// the shading normal below the horizon on the pine forest, 99.7% were geometric
/// back faces. This predicate takes those and leaves the rest.
///
/// `transmission` keeps the dielectric exit exactly as it was -- that hit is not
/// a back face, it is how light gets out of glass -- and `diffuseTransmission`
/// keeps the leaf that is lit through its own thickness, which is a different
/// question with a different answer.
DEVICE_FUNC bool opaqueBackHitFlipsFrame(bool frontFace, float nDotV, float transmission,
                                         float diffuseTransmission)
{
    return !frontFace && nDotV <= 0.0f && transmission <= 0.0f && diffuseTransmission <= 0.0f;
}

/// The frame the BSDF actually shaded in, for everything downstream that has to
/// agree with it about which hemisphere the surface scatters into.
///
/// Next-event estimation and BSDF sampling only share directions while they
/// share a normal. The estimate's two halves read the hemisphere off
/// `si.front_face` and `si.shading_normal`, which are what the *geometry* says;
/// once the flip above is in play they are no longer what was shaded, and the
/// half that notices is the one that goes wrong. Withholding the MIS weight from
/// a flipped bounce -- which is what happens if the raw front_face is passed --
/// lets the bounce take a whole contribution that next-event estimation has
/// already taken a weighted share of, and the light lands about twice.
struct ShadedFrame
{
    /// True when the shaded frame faces the viewer, whatever the winding did.
    bool frontFace;
    /// Multiplies a dot product taken against the raw shading normal.
    float normalSign;
};

DEVICE_FUNC ShadedFrame shadedFrame(bool frontFace, float nDotV, float transmission, float diffuseTransmission)
{
    ShadedFrame f;
    const bool flipped = opaqueBackHitFlipsFrame(frontFace, nDotV, transmission, diffuseTransmission);
    f.frontFace = frontFace || flipped;
    f.normalSign = flipped ? -1.0f : 1.0f;
    return f;
}

#endif // STRELKA_MATERIAL_SHADING_FRAME_H

// NOLINTEND(cppcoreguidelines-pro-type-member-init)
