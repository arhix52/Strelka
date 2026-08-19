#ifndef STRELKA_PROJECTOR_H
#define STRELKA_PROJECTOR_H

// ============================================================================
// projector.h -- the angular emission profile of a projector light.
//
// A projector is a spot whose cone is a rectangular pyramid and whose intensity
// across that pyramid is an image rather than a constant: a home cinema beamer
// throwing a frame onto a wall, a slide projector, a gobo in a theatre lantern.
// The shape is a camera's, run backwards -- an image plane at unit depth with
// half extents tan(fov/2), and a perspective divide to find where a direction
// lands on it -- which is what makes the projected rectangle keep its aspect and
// grow linearly with throw distance the way a real one does.
//
// Scalars in, place in the image out. Deliberately free of vector types, of
// UniformLight and of textures, for two reasons: the fetch itself cannot be
// shared (the three compilers that build this tree disagree about what a texture
// is -- see bsdf.h for the same split), and the host's radiometric bake in
// scene/light_desc.h has to divide by exactly the solid angle the shader
// integrates over, without dragging CUDA or Metal spellings into a scene header.
// Each backend unpacks the light's own basis, which it already does for IES.
//
// tests/render/test_projector.cpp compiles this file on the host.
// ============================================================================

#include <strelka/material/material_math.h>

/// Where a direction lands on the projected image.
struct ProjectorSample
{
    float u = 0.0f; ///< [0,1] across the image, 0 at the left as seen from behind the light
    float v = 0.0f; ///< [0,1] down the image, 0 at the top -- the row order a decoder gives
    float falloff = 0.0f; ///< edge feather: 1 well inside the frame, 0 at its border
    bool inside = false; ///< false when the direction misses the frame entirely
};

/// A direction that lands nowhere on the image. Named rather than spelled out at
/// each `return`, so that "missed the frame" is one thing and not three.
DEVICE_FUNC ProjectorSample makeProjectorMiss()
{
    ProjectorSample s;
    return s;
}

/// Half extent of the projected image at unit distance, along the horizontal.
///
/// Clamped below 90 degrees: at exactly 90 the tangent is infinite and the
/// pyramid degenerates into a half space, which is not a projector and would
/// hand the solid angle below a division by zero to work with.
DEVICE_FUNC float projectorTanHalfX(float halfFovX)
{
    const float a = fminf(fmaxf(halfFovX, 1e-4f), 1.55334f); // ~89 degrees
    return tanf(a);
}

/// The vertical companion, from the image's aspect ratio (width / height).
///
/// The aspect is authored rather than read from the image, because a projector's
/// frame is a property of its optics: a 16:9 beamer fed a 4:3 slide still throws
/// a 16:9 rectangle, and pillarboxing is the image's business, not the lamp's.
DEVICE_FUNC float projectorTanHalfY(float tanHalfX, float aspect)
{
    return tanHalfX / fmaxf(aspect, 1e-4f);
}

/// Smooth fade to zero over the outermost `softness` of the half extent.
///
/// `t` is |x| or |y| in frame coordinates, so 1 is the border. Softness 0 is a
/// crisp edge, which is what a focused projector actually has -- the feather is
/// there for the soft-edged look a shaped beam gets from a diffuser or from
/// being thrown out of focus, and it costs nothing when it is off.
DEVICE_FUNC float projectorEdgeFade(float t, float softness)
{
    if (!(softness > 0.0f))
    {
        return 1.0f;
    }
    const float e = fminf(fmaxf((1.0f - t) / softness, 0.0f), 1.0f);
    return e * e * (3.0f - 2.0f * e);
}

/// Project a direction, given in the light's own frame, onto the image.
///
/// `localZ` is the component along the emission axis (the light's -Z, which is
/// what UniformLight::normal holds), `localX` and `localY` the components along
/// the frame's right and up axes. A direction with localZ <= 0 is behind the
/// projector: no image, and no light either -- unlike a cone falloff, this is not
/// a smooth term that happens to reach zero, so the caller must not skip the
/// test and rely on the fade.
DEVICE_FUNC ProjectorSample
projectorProject(float localX, float localY, float localZ, float tanHalfX, float tanHalfY, float edgeSoftness)
{
    if (!(localZ > 1e-6f) || !(tanHalfX > 0.0f) || !(tanHalfY > 0.0f))
    {
        return makeProjectorMiss();
    }

    // The perspective divide. x and y are now in [-1, 1] inside the frame.
    const float x = localX / (localZ * tanHalfX);
    const float y = localY / (localZ * tanHalfY);
    if (x < -1.0f || x > 1.0f || y < -1.0f || y > 1.0f)
    {
        return makeProjectorMiss();
    }

    ProjectorSample s;
    // v is flipped because a decoded image starts at its top row while the
    // light's +Y is up. Getting this backwards does not look like a bug -- it
    // looks like a projector mounted upside down, which is a thing people do on
    // purpose, so it has to be pinned by a test rather than by eye.
    s.u = 0.5f + 0.5f * x;
    s.v = 0.5f - 0.5f * y;
    s.falloff = projectorEdgeFade(fabsf(x), edgeSoftness) * projectorEdgeFade(fabsf(y), edgeSoftness);
    s.inside = true;
    return s;
}

/// Solid angle of the rectangular pyramid the projector emits into.
///
/// Omega = 4 asin(sin a sin b) for half angles a and b -- the closed form for a
/// rectangular cone, and the analogue of coneSolidAngleFromHalfAngle() in
/// light_pdf.h. Written from the tangents because that is what the caller has,
/// and via sin = tan / sqrt(1 + tan^2) rather than atan then sin, so a narrow
/// beam keeps its digits.
///
/// This is the number a projector's Watts are divided by to become the radiant
/// intensity the shader reads out of UniformLight::color, so the bake in
/// scene/light_desc.h and anything that reasons about the light's total power
/// have to use this one function. The cone of the same half angle circumscribes
/// the pyramid rather than matching it -- on a 16:9 frame it is 36% larger at a
/// 60 degree horizontal field and 40% larger as the beam narrows -- so borrowing
/// it here would spread the same Watts over too much sky and leave the image
/// that much too dim.
DEVICE_FUNC float projectorSolidAngle(float tanHalfX, float tanHalfY)
{
    const float sinX = tanHalfX / sqrtf(1.0f + tanHalfX * tanHalfX);
    const float sinY = tanHalfY / sqrtf(1.0f + tanHalfY * tanHalfY);
    return 4.0f * asinf(fminf(sinX * sinY, 1.0f));
}

#endif // STRELKA_PROJECTOR_H
