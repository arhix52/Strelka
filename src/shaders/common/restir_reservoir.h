#pragma once

#include <strelka/material/material_math.h>

struct RestirReservoirState
{
    float weightSum;
    float target;
    unsigned int M;
    unsigned int ageAndFlags;
};

#define RESTIR_RESERVOIR_VALID (1u << 31)
#define RESTIR_RESERVOIR_AGE_MASK 0xffu

DEVICE_FUNC bool restirReservoirUpdate(THREAD_REF RestirReservoirState& reservoir,
                                       float candidateWeight,
                                       float candidateTarget,
                                       unsigned int candidateM,
                                       float randomValue)
{
    reservoir.M += candidateM;
    if (!(candidateWeight > 0.0f) || !(candidateTarget > 0.0f))
    {
        return false;
    }
    reservoir.weightSum += candidateWeight;
    if (randomValue * reservoir.weightSum > candidateWeight)
    {
        return false;
    }
    reservoir.target = candidateTarget;
    reservoir.ageAndFlags = RESTIR_RESERVOIR_VALID;
    return true;
}

DEVICE_FUNC float restirReservoirNormalization(const THREAD_REF RestirReservoirState& reservoir)
{
    return (reservoir.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u && reservoir.M > 0u && reservoir.target > 0.0f ?
               reservoir.weightSum / (float(reservoir.M) * reservoir.target) :
               0.0f;
}

DEVICE_FUNC float restirReservoirMergeWeight(const THREAD_REF RestirReservoirState& source, float currentTarget)
{
    return (source.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u && source.M > 0u && source.target > 0.0f &&
                   currentTarget > 0.0f ?
               currentTarget * source.weightSum / source.target :
               0.0f;
}

DEVICE_FUNC bool restirSurfaceCompatible(float currentDepth,
                                         float previousDepth,
                                         float normalDot,
                                         unsigned int currentMaterial,
                                         unsigned int previousMaterial,
                                         bool previousValid)
{
    const float largerDepth = currentDepth > previousDepth ? currentDepth : previousDepth;
    const float scaledTolerance = 0.1f * largerDepth;
    const float depthTolerance = scaledTolerance > 0.01f ? scaledTolerance : 0.01f;
    const float depthDifference =
        currentDepth > previousDepth ? currentDepth - previousDepth : previousDepth - currentDepth;
    return previousValid && currentMaterial == previousMaterial && depthDifference <= depthTolerance && normalDot >= 0.9f;
}
