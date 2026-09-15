#ifndef STRELKA_PROJECTOR_H
#define STRELKA_PROJECTOR_H

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

DEVICE_FUNC float projectorTanHalfX(float halfFovX)
{
    const float a = fminf(fmaxf(halfFovX, 1e-4f), 1.55334f); // ~89 degrees
    return tanf(a);
}

DEVICE_FUNC float projectorTanHalfY(float tanHalfX, float aspect)
{
    return tanHalfX / fmaxf(aspect, 1e-4f);
}

DEVICE_FUNC float projectorEdgeFade(float t, float softness)
{
    if (!(softness > 0.0f))
    {
        return 1.0f;
    }
    const float e = fminf(fmaxf((1.0f - t) / softness, 0.0f), 1.0f);
    return e * e * (3.0f - 2.0f * e);
}

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
    s.u = 0.5f + 0.5f * x;
    s.v = 0.5f - 0.5f * y;
    s.falloff = projectorEdgeFade(fabsf(x), edgeSoftness) * projectorEdgeFade(fabsf(y), edgeSoftness);
    s.inside = true;
    return s;
}

DEVICE_FUNC float projectorSolidAngle(float tanHalfX, float tanHalfY)
{
    const float sinX = tanHalfX / sqrtf(1.0f + tanHalfX * tanHalfX);
    const float sinY = tanHalfY / sqrtf(1.0f + tanHalfY * tanHalfY);
    return 4.0f * asinf(fminf(sinX * sinY, 1.0f));
}

#endif // STRELKA_PROJECTOR_H
