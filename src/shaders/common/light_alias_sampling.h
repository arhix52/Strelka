#ifndef STRELKA_LIGHT_ALIAS_SAMPLING_H
#define STRELKA_LIGHT_ALIAS_SAMPLING_H

#include <strelka/material/material_math.h>
#include <discrete_sampling.h>

// Shared scalar part of the analytic-light Walker/Vose draw. The table fields
// live inside UniformLight on Metal, while host tests use the compact builder
// entries directly; keeping bucket selection and endpoint handling here makes
// both consumers realize the same discrete distribution.

DEVICE_FUNC unsigned int lightAliasBucket(unsigned int count, unsigned int bucketWord)
{
    return discreteUniformIndex(count, bucketWord);
}

DEVICE_FUNC unsigned int lightAliasSelect(
    unsigned int count, unsigned int bucket, unsigned int coinWord, unsigned int threshold, unsigned int alias)
{
    return discreteAliasSelect(count, bucket, coinWord, threshold, alias);
}

#endif // STRELKA_LIGHT_ALIAS_SAMPLING_H
