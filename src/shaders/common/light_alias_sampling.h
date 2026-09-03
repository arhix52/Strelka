#ifndef STRELKA_LIGHT_ALIAS_SAMPLING_H
#define STRELKA_LIGHT_ALIAS_SAMPLING_H

#include <strelka/material/material_math.h>

// Shared scalar part of the analytic-light Walker/Vose draw. The table fields
// live inside UniformLight on Metal, while host tests use the compact builder
// entries directly; keeping bucket selection and endpoint handling here makes
// both consumers realize the same discrete distribution.

DEVICE_FUNC unsigned int lightAliasBucket(unsigned int count, float bucketUniform)
{
    if (count == 0u)
    {
        return 0u;
    }
    float scaled = bucketUniform * float(count);
    if (!(scaled >= 0.0f))
    {
        scaled = 0.0f;
    }
    if (!(scaled < float(count)))
    {
        return count - 1u;
    }
    const unsigned int bucket = (unsigned int)scaled;
    return bucket < count ? bucket : count - 1u;
}

DEVICE_FUNC unsigned int lightAliasSelect(
    unsigned int count, unsigned int bucket, float aliasUniform, float aliasProbability, unsigned int alias)
{
    if (count == 0u || bucket >= count)
    {
        return count;
    }
    const float probability = fminf(fmaxf(aliasProbability, 0.0f), 1.0f);
    if (aliasUniform < probability)
    {
        return bucket;
    }
    return alias < count ? alias : count;
}

#endif // STRELKA_LIGHT_ALIAS_SAMPLING_H
