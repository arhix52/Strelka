#ifndef STRELKA_VOLUME_H
#define STRELKA_VOLUME_H

#include "material_math.h"

#define VOLUME_MODEL_GLTF   0u
#define VOLUME_MODEL_CYCLES 1u

// Extinction per unit length. A non-positive distance, or a fully white
// attenuation colour, means the medium absorbs nothing.
DEVICE_FUNC float3 volume_extinction(float3 attenuation_color, float attenuation_distance,
                                     unsigned int model)
{
    if (attenuation_distance <= 0.0f)
    {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    const float inv_d = 1.0f / attenuation_distance;
    float c[3] = { attenuation_color.x, attenuation_color.y, attenuation_color.z };
    float s[3];
    for (int i = 0; i < 3; ++i)
    {
        const float ci = c[i] < 0.0f ? 0.0f : (c[i] > 1.0f ? 1.0f : c[i]);
        if (model == VOLUME_MODEL_CYCLES)
        {
            s[i] = (1.0f - ci) * inv_d;
        }
        else
        {
            // -ln(0) is infinite; clamp so a black medium becomes very dense
            // rather than producing a NaN throughput.
            s[i] = ci <= 1e-4f ? 1e4f : -logf(ci) * inv_d;
        }
    }
    return make_float3(s[0], s[1], s[2]);
}

DEVICE_FUNC float3 beer_lambert_transmittance(float3 sigma_t, float distance)
{
    return make_float3(expf(-sigma_t.x * distance), expf(-sigma_t.y * distance),
                       expf(-sigma_t.z * distance));
}

#endif // STRELKA_VOLUME_H
