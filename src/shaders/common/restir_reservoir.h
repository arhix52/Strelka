#pragma once

#include <strelka/material/material_math.h>

#define RESTIR_SAMPLE_INVALID 0u
#define RESTIR_SAMPLE_ANALYTIC 1u
#define RESTIR_SAMPLE_ENVIRONMENT 2u
#define RESTIR_SAMPLE_EMISSIVE_TRIANGLE 3u
#define RESTIR_SAMPLE_TYPE_SHIFT 30u
#define RESTIR_SAMPLE_ID_MASK 0x3fffffffu
#define RESTIR_LIGHT_UNMAPPED 0xffffffffu
#define RESTIR_LIGHT_TYPE_CHANGED 0xfffffffeu

struct RestirLightSample
{
    unsigned int typeAndLightId;
    unsigned int data0;
    unsigned int data1;
    unsigned int data2;
};

DEVICE_FUNC unsigned int restirSampleKey(unsigned int type, unsigned int lightId)
{
    return (type << RESTIR_SAMPLE_TYPE_SHIFT) | (lightId & RESTIR_SAMPLE_ID_MASK);
}

DEVICE_FUNC unsigned int restirSampleType(const THREAD_REF RestirLightSample& sample)
{
    return sample.typeAndLightId >> RESTIR_SAMPLE_TYPE_SHIFT;
}

DEVICE_FUNC unsigned int restirSampleLightId(const THREAD_REF RestirLightSample& sample)
{
    return sample.typeAndLightId & RESTIR_SAMPLE_ID_MASK;
}

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

// Encode RTXDI BASIC's MIS-like normalization back into this reservoir's raw-weight representation.
DEVICE_FUNC void restirReservoirApplyBasicNormalization(THREAD_REF RestirReservoirState& reservoir,
                                                        float selectedSourceTarget,
                                                        float sourceTargetSum)
{
    if (!(selectedSourceTarget > 0.0f) || !(sourceTargetSum > 0.0f) || reservoir.M == 0u)
    {
        reservoir.weightSum = 0.0f;
        return;
    }
    reservoir.weightSum *= selectedSourceTarget * float(reservoir.M) / sourceTargetSum;
}

DEVICE_FUNC float restirReservoirMergeWeight(const THREAD_REF RestirReservoirState& source, float currentTarget)
{
    return (source.ageAndFlags & RESTIR_RESERVOIR_VALID) != 0u && source.M > 0u && source.target > 0.0f &&
                   currentTarget > 0.0f ?
               currentTarget * source.weightSum / source.target :
               0.0f;
}

DEVICE_FUNC void restirReservoirLimitM(THREAD_REF RestirReservoirState& reservoir, unsigned int maxM)
{
    if (maxM > 0u && reservoir.M > maxM)
    {
        reservoir.weightSum *= float(maxM) / float(reservoir.M);
        reservoir.M = maxM;
    }
}

DEVICE_FUNC void restirReservoirLimitHistoryM(THREAD_REF RestirReservoirState& reservoir,
                                              unsigned int currentM,
                                              unsigned int maxHistoryLength)
{
    restirReservoirLimitM(reservoir, currentM * maxHistoryLength);
}

DEVICE_FUNC unsigned int restirSampleSequenceBase(bool frameJitter,
                                                  bool accumulationEnabled,
                                                  bool temporalReuseEnabled,
                                                  unsigned int frameIndex,
                                                  unsigned int samplesPerLaunch,
                                                  unsigned int subframeIndex)
{
    return frameJitter || !accumulationEnabled || temporalReuseEnabled ?
               frameIndex * (samplesPerLaunch > 0u ? samplesPerLaunch : 1u) :
               subframeIndex;
}

DEVICE_FUNC void restirReservoirDiscardSample(THREAD_REF RestirReservoirState& reservoir)
{
    reservoir.weightSum = 0.0f;
    reservoir.target = 0.0f;
    reservoir.ageAndFlags &= ~RESTIR_RESERVOIR_VALID;
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
