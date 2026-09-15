#ifndef STRELKA_OPTIX_TEXTURE_TRANSFORM_H
#define STRELKA_OPTIX_TEXTURE_TRANSFORM_H

#include <strelka/material/material_math.h>
#include <strelka/material/material_params.h>

DEVICE_FUNC float2 apply_texture_transform(float2 uv, const THREAD_REF MaterialParams& m)
{
    const float c = cosf(m.uv_rotation);
    const float s = sinf(m.uv_rotation);
    const float kx = (m.uv_scale_x != 0.0f) ? m.uv_scale_x : 1.0f;
    const float ky = (m.uv_scale_y != 0.0f) ? m.uv_scale_y : 1.0f;
    return make_float2(uv.x * kx * c - uv.y * ky * s + m.uv_offset_x, uv.x * kx * s + uv.y * ky * c + m.uv_offset_y);
}

// glTF COLOR_0, packed RGBA8 and LINEAR -- it carries no transfer function,
// unlike a base-colour texture, so nothing is decoded here.
DEVICE_FUNC float3 unpack_vertex_color(unsigned int val)
{
    const float s = 1.0f / 255.0f;
    return make_float3(static_cast<float>(val & 0xffu) * s, static_cast<float>((val >> 8) & 0xffu) * s,
                       static_cast<float>((val >> 16) & 0xffu) * s);
}

// Coverage of a surface, given the alpha the base-colour slot resolved to.
// MASK is a binary predicate, BLEND passes the alpha through, OPAQUE is always
// 1 -- so callers only ever see a float in [0,1] and never branch on the mode.
DEVICE_FUNC float resolve_opacity(const THREAD_REF MaterialParams& m, float alpha)
{
    if (m.alpha_mode == ALPHA_MODE_OPAQUE)
    {
        return 1.0f;
    }
    if (m.alpha_mode == ALPHA_MODE_MASK)
    {
        return alpha >= m.alpha_cutoff ? 1.0f : 0.0f;
    }
    return alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
}

#endif // STRELKA_OPTIX_TEXTURE_TRANSFORM_H
