#ifndef STRELKA_SHARC_QUERY_ELIGIBILITY_H
#define STRELKA_SHARC_QUERY_ELIGIBILITY_H

// A compact SHARC entry cannot reconstruct sharp, transmissive, or fibre lobes.
// Keep the receiver policy shared by Metal and its host regression test.

#include <strelka/material/material_math.h>

DEVICE_FUNC float sharcReceiverLobeRoughness(float effectiveRoughness, float lobeRoughness, float lobeWeight)
{
    return lobeWeight > 0.0f ? fminf(effectiveRoughness, saturate(lobeRoughness)) : effectiveRoughness;
}

DEVICE_FUNC bool sharcReceiverCacheEligible(float effectiveRoughness,
                                            float transmissionWeight,
                                            bool unsupportedAngularLobe,
                                            float minimumRoughness)
{
    return transmissionWeight <= 0.0f && !unsupportedAngularLobe &&
           saturate(effectiveRoughness) >= saturate(minimumRoughness);
}

#endif // STRELKA_SHARC_QUERY_ELIGIBILITY_H
