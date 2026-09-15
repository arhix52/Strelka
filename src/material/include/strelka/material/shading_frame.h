#ifndef STRELKA_MATERIAL_SHADING_FRAME_H
#define STRELKA_MATERIAL_SHADING_FRAME_H

#include <strelka/material/material_math.h>

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init)

DEVICE_FUNC bool opaqueBackHitFlipsFrame(bool frontFace, float nDotV, float transmission,
                                         float diffuseTransmission)
{
    return !frontFace && nDotV <= 0.0f && transmission <= 0.0f && diffuseTransmission <= 0.0f;
}

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
