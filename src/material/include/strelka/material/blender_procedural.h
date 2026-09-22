#ifndef STRELKA_MATERIAL_BLENDER_PROCEDURAL_H
#define STRELKA_MATERIAL_BLENDER_PROCEDURAL_H

#include <strelka/material/material_math.h>

// The two procedural height nodes present in the imported interior.  These
// scalar ports follow Cycles' hash/noise code so CUDA and Metal execute the
// source graph instead of a camera-dependent texture bake.

DEVICE_FUNC unsigned int blender_rotl(unsigned int x, unsigned int k)
{
    return (x << k) | (x >> (32u - k));
}

DEVICE_FUNC unsigned int blender_hash_uint3(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int a = 0xdeadbeefu + (3u << 2u) + 13u;
    unsigned int b = a;
    unsigned int c = a;
    c += z;
    b += y;
    a += x;
#define STRELKA_HASH_FINAL_STEP(lhs, rhs, rotation) lhs ^= rhs; lhs -= blender_rotl(rhs, rotation)
    STRELKA_HASH_FINAL_STEP(c, b, 14u);
    STRELKA_HASH_FINAL_STEP(a, c, 11u);
    STRELKA_HASH_FINAL_STEP(b, a, 25u);
    STRELKA_HASH_FINAL_STEP(c, b, 16u);
    STRELKA_HASH_FINAL_STEP(a, c, 4u);
    STRELKA_HASH_FINAL_STEP(b, a, 14u);
    STRELKA_HASH_FINAL_STEP(c, b, 24u);
#undef STRELKA_HASH_FINAL_STEP
    return c;
}

DEVICE_FUNC void blender_hash_int3(int x, int y, int z, THREAD_REF float& ox, THREAD_REF float& oy, THREAD_REF float& oz)
{
    unsigned int hx = (unsigned int)x * 1664525u + 1013904223u;
    unsigned int hy = (unsigned int)y * 1664525u + 1013904223u;
    unsigned int hz = (unsigned int)z * 1664525u + 1013904223u;
    hx += hy * hz;
    hy += hz * hx;
    hz += hx * hy;
    // Cycles deliberately uses signed PCG integers for OSL parity; its right
    // shift is arithmetic.  Recreate that on unsigned words without invoking
    // host signed-overflow UB.
    const unsigned int sx = (hx >> 16u) | ((hx & 0x80000000u) ? 0xffff0000u : 0u);
    const unsigned int sy = (hy >> 16u) | ((hy & 0x80000000u) ? 0xffff0000u : 0u);
    const unsigned int sz = (hz >> 16u) | ((hz & 0x80000000u) ? 0xffff0000u : 0u);
    hx ^= sx;
    hy ^= sy;
    hz ^= sz;
    hx += hy * hz;
    hy += hz * hx;
    hz += hx * hy;
    constexpr float inverse = 1.0f / 2147483647.0f;
    ox = float(hx & 0x7fffffffu) * inverse;
    oy = float(hy & 0x7fffffffu) * inverse;
    oz = float(hz & 0x7fffffffu) * inverse;
}

DEVICE_FUNC float blender_voronoi_ridge(float2 uv, float scale)
{
    const float px = uv.x * scale;
    const float py = uv.y * scale;
    const int cellX = (int)floorf(px);
    const int cellY = (int)floorf(py);
    const float localX = px - floorf(px);
    const float localY = py - floorf(py);
    float minimum = 3.402823466e38f;
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                float hx, hy, hz;
                blender_hash_int3(cellX + x, cellY + y, z, hx, hy, hz);
                const float dx = float(x) + hx - localX;
                const float dy = float(y) + hy - localY;
                const float dz = float(z) + hz;
                minimum = fminf(minimum, sqrtf(dx * dx + dy * dy + dz * dz));
            }
        }
    }
    return 2.0f * fminf(minimum, 1.0f - minimum);
}

DEVICE_FUNC float blender_fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

DEVICE_FUNC float blender_grad3(unsigned int hash, float x, float y, float z)
{
    const int h = int(hash & 15u);
    const float u = h < 8 ? x : y;
    const float vt = (h == 12 || h == 14) ? x : z;
    const float v = h < 4 ? y : vt;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

DEVICE_FUNC float blender_perlin3(float x, float y, float z)
{
    const int ix = (int)floorf(x);
    const int iy = (int)floorf(y);
    const int iz = (int)floorf(z);
    const float fx = x - floorf(x);
    const float fy = y - floorf(y);
    const float fz = z - floorf(z);
    const float u = blender_fade(fx);
    const float v = blender_fade(fy);
    const float w = blender_fade(fz);
#define STRELKA_GRAD(dx, dy, dz) blender_grad3(blender_hash_uint3((unsigned int)(ix + dx), (unsigned int)(iy + dy), (unsigned int)(iz + dz)), fx - float(dx), fy - float(dy), fz - float(dz))
    const float x00 = mix(STRELKA_GRAD(0, 0, 0), STRELKA_GRAD(1, 0, 0), u);
    const float x10 = mix(STRELKA_GRAD(0, 1, 0), STRELKA_GRAD(1, 1, 0), u);
    const float x01 = mix(STRELKA_GRAD(0, 0, 1), STRELKA_GRAD(1, 0, 1), u);
    const float x11 = mix(STRELKA_GRAD(0, 1, 1), STRELKA_GRAD(1, 1, 1), u);
#undef STRELKA_GRAD
    return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

DEVICE_FUNC float blender_signed_noise3(float2 uv, float scale)
{
    float x = fmodf(uv.x * scale, 100000.0f);
    float y = fmodf(uv.y * scale, 100000.0f);
    return 0.9820f * blender_perlin3(x, y, 0.0f);
}

DEVICE_FUNC float blender_noise_fbm(float2 uv, float scale, float detail, float roughness, float lacunarity)
{
    float frequency = 1.0f;
    float amplitude = 1.0f;
    float maximum = 0.0f;
    float sum = 0.0f;
    const int whole = (int)floorf(detail);
    for (int octave = 0; octave <= whole; ++octave)
    {
        sum += blender_signed_noise3(uv, scale * frequency) * amplitude;
        maximum += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    }
    const float remainder = detail - floorf(detail);
    if (remainder != 0.0f)
    {
        const float sum2 = sum + blender_signed_noise3(uv, scale * frequency) * amplitude;
        return mix(0.5f * sum / maximum + 0.5f,
                   0.5f * sum2 / (maximum + amplitude) + 0.5f, remainder);
    }
    return 0.5f * sum / maximum + 0.5f;
}

#endif
