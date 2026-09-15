#ifndef STRELKA_NEE_PAIRING_H
#define STRELKA_NEE_PAIRING_H

#include <strelka/material/material_math.h>
#include <strelka/material/shading_frame.h>

DEVICE_FUNC bool neeCrossesSurface(bool throughFibre, float transmission, float diffuseTransmission)
{
    return throughFibre || transmission > 0.0f || diffuseTransmission > 0.0f;
}

DEVICE_FUNC bool neeProposesDirection(bool crossesSurface, bool frontFace, float nDotL)
{
    return crossesSurface || ((nDotL > 0.0f) == frontFace);
}

/// Receiver-side support in the frame the BSDF actually uses. Inputs involving
/// the normal are measured against the raw shading normal stored in the hit.
DEVICE_FUNC bool neeSurfaceSupportsDirection(bool throughFibre, bool frontFace, float nDotV, float transmission,
                                             float diffuseTransmission, float nDotL)
{
    const ShadedFrame frame = shadedFrame(frontFace, nDotV, transmission, diffuseTransmission);
    return neeProposesDirection(neeCrossesSurface(throughFibre, transmission, diffuseTransmission), frame.frontFace,
                                frame.normalSign * nDotL);
}

/// The projected solid-angle factor in the same frame as the support test.
DEVICE_FUNC float neeSurfaceCosine(bool throughFibre, bool frontFace, float nDotV, float transmission,
                                   float diffuseTransmission, float nDotL)
{
    if (throughFibre)
    {
        return fabsf(nDotL);
    }
    return neeSurfaceSupportsDirection(false, frontFace, nDotV, transmission, diffuseTransmission, nDotL) ?
               fabsf(nDotL) :
               0.0f;
}

DEVICE_FUNC bool neePairsWithBounce(bool didNee, bool crossesSurface, bool frontFace, float nDotDir)
{
    return didNee && neeProposesDirection(crossesSurface, frontFace, nDotDir);
}

DEVICE_FUNC bool neeRunsAtVertex(bool neeEnabled, bool hasEmitter, bool materialHasSmoothLobe)
{
    return neeEnabled && hasEmitter && materialHasSmoothLobe;
}

DEVICE_FUNC bool volumeNeePairsWithBounce(bool neeEnabled, bool hasEmitter)
{
    return neeRunsAtVertex(neeEnabled, hasEmitter, true);
}

DEVICE_FUNC float3 orientedFaceNormal(float3 geometryNormal, float3 dir)
{
    return (dot(geometryNormal, dir) > 0.0f) ? geometryNormal : -geometryNormal;
}

#endif // STRELKA_NEE_PAIRING_H
