#ifndef STRELKA_NORMAL_FILTER_H
#define STRELKA_NORMAL_FILTER_H

#include "material_math.h"

/// Fold the variance of a prefiltered tangent-space normal map into isotropic
/// GGX roughness. `concentration` is |E[n]| stored in mip alpha. For small
/// angles, E[theta^2] ~= 2(1-|E[n]|); normal scale multiplies slope variance.
DEVICE_FUNC float normal_filter_roughness(float roughness, float concentration, float normalScale)
{
    // BC5 supplies no alpha on OptiX. Zero therefore means "no statistic",
    // while generated moment mips are clamped to at least 1/255.
    if (concentration <= 0.0f || concentration >= 1.0f)
        return roughness;
    const float r2 = roughness * roughness;
    const float variance = 2.0f * (1.0f - concentration) * normalScale * normalScale;
    return fminf(sqrtf(sqrtf(r2 * r2 + variance)), 1.0f);
}

#endif // STRELKA_NORMAL_FILTER_H
