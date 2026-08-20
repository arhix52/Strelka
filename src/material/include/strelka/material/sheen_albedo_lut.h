#ifndef STRELKA_SHEEN_ALBEDO_LUT_H
#define STRELKA_SHEEN_ALBEDO_LUT_H

// ============================================================================
// sheen_albedo_lut.h -- directional albedo of the Charlie sheen lobe
//
// E(NdotV, sheenRoughness): the fraction of light arriving from NdotV that the
// sheen layer sends back. KHR_materials_sheen needs it twice -- once to scale
// the base layer down by what the fabric already reflected, and once to
// normalise the lobe itself.
//
// The second use is the one the spec does not call for. Ashikhmin's visibility
// term does not conserve energy, and this table peaks at 2.77559 at low roughness
// and grazing incidence -- so a layer applied at face value reflects nearly
// three times the light that fell on it. Measured, not assumed: the additive
// version put a plain white cloth at 1.40 directional albedo, which
// tests/material/test_sheen.cpp now pins.
//
// Regenerate with tools/material/gen_sheen_albedo_lut.py. 16x16 over
// NdotV in (0,1] and sheenRoughness in (0,1], both bin-centred.
// ============================================================================

#include "material_math.h"

enum : int
{
    SHEEN_ALBEDO_LUT_SIZE = 16
};

// Row-major: [roughness][NdotV].
DEVICE_CONST float kSheenAlbedoLut[SHEEN_ALBEDO_LUT_SIZE * SHEEN_ALBEDO_LUT_SIZE] = {
    2.77559f, 0.18233f, 0.00690f, 0.00012f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
    2.34265f, 0.95516f, 0.39749f, 0.15820f, 0.05856f, 0.01974f, 0.00593f, 0.00155f, 0.00034f, 0.00006f, 0.00001f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
    1.75241f, 1.00024f, 0.59521f, 0.35360f, 0.20630f, 0.11695f, 0.06381f, 0.03318f, 0.01625f, 0.00738f, 0.00304f, 0.00110f, 0.00034f, 0.00008f, 0.00001f, 0.00000f,
    1.40059f, 0.92527f, 0.63718f, 0.44209f, 0.30523f, 0.20815f, 0.13934f, 0.09102f, 0.05762f, 0.03505f, 0.02024f, 0.01092f, 0.00536f, 0.00228f, 0.00076f, 0.00014f,
    1.17598f, 0.84365f, 0.62939f, 0.47446f, 0.35771f, 0.26810f, 0.19885f, 0.14531f, 0.10413f, 0.07276f, 0.04919f, 0.03183f, 0.01939f, 0.01080f, 0.00517f, 0.00179f,
    1.02297f, 0.77368f, 0.60709f, 0.48189f, 0.38348f, 0.30437f, 0.24005f, 0.18748f, 0.14447f, 0.10939f, 0.08093f, 0.05807f, 0.03993f, 0.02582f, 0.01510f, 0.00726f,
    0.91360f, 0.71664f, 0.58213f, 0.47857f, 0.39495f, 0.32573f, 0.26759f, 0.21834f, 0.17645f, 0.14076f, 0.11038f, 0.08461f, 0.06286f, 0.04467f, 0.02962f, 0.01735f,
    0.83257f, 0.67061f, 0.55854f, 0.47089f, 0.39887f, 0.33806f, 0.28587f, 0.24058f, 0.20102f, 0.16632f, 0.13582f, 0.10899f, 0.08542f, 0.06477f, 0.04676f, 0.03113f,
    0.77085f, 0.63341f, 0.53755f, 0.46183f, 0.39887f, 0.34499f, 0.29804f, 0.25662f, 0.21977f, 0.18679f, 0.15713f, 0.13040f, 0.10627f, 0.08447f, 0.06478f, 0.04703f,
    0.72280f, 0.60316f, 0.51935f, 0.45273f, 0.39690f, 0.34868f, 0.30622f, 0.26832f, 0.23415f, 0.20313f, 0.17480f, 0.14881f, 0.12490f, 0.10285f, 0.08247f, 0.06362f,
    0.68469f, 0.57837f, 0.50374f, 0.44419f, 0.39402f, 0.35043f, 0.31175f, 0.27695f, 0.24529f, 0.21624f, 0.18941f, 0.16450f, 0.14128f, 0.11954f, 0.09913f, 0.07992f,
    0.65401f, 0.55789f, 0.49040f, 0.43643f, 0.39082f, 0.35102f, 0.31554f, 0.28342f, 0.25401f, 0.22684f, 0.20154f, 0.17784f, 0.15554f, 0.13444f, 0.11442f, 0.09534f,
    0.62898f, 0.54084f, 0.47900f, 0.42951f, 0.38761f, 0.35095f, 0.31815f, 0.28836f, 0.26095f, 0.23549f, 0.21165f, 0.18919f, 0.16790f, 0.14762f, 0.12822f, 0.10958f,
    0.60832f, 0.52654f, 0.46924f, 0.42338f, 0.38453f, 0.35049f, 0.31997f, 0.29217f, 0.26652f, 0.24261f, 0.22014f, 0.19887f, 0.17862f, 0.15922f, 0.14056f, 0.12253f,
    0.59108f, 0.51445f, 0.46085f, 0.41799f, 0.38167f, 0.34982f, 0.32124f, 0.29516f, 0.27106f, 0.24853f, 0.22731f, 0.20716f, 0.18791f, 0.16941f, 0.15154f, 0.13419f,
    0.57657f, 0.50416f, 0.45362f, 0.41325f, 0.37905f, 0.34906f, 0.32213f, 0.29755f, 0.27479f, 0.25350f, 0.23341f, 0.21429f, 0.19599f, 0.17835f, 0.16127f, 0.14463f
};

// Bilinear lookup. Clamped rather than wrapped at both edges: NdotV -> 0 is a
// grazing view, which is exactly where the table is steepest and where
// extrapolating would be worst.
DEVICE_FUNC float sheen_albedo(float n_dot_v, float sheen_roughness)
{
    const float n = (float)SHEEN_ALBEDO_LUT_SIZE;
    float x = saturate(n_dot_v) * n - 0.5f;
    float y = saturate(sheen_roughness) * n - 0.5f;
    x = fmaxf(0.0f, fminf(x, n - 1.0f));
    y = fmaxf(0.0f, fminf(y, n - 1.0f));

    const int x0 = (int)x;
    const int y0 = (int)y;
    const int x1 = (x0 + 1 < SHEEN_ALBEDO_LUT_SIZE) ? x0 + 1 : x0;
    const int y1 = (y0 + 1 < SHEEN_ALBEDO_LUT_SIZE) ? y0 + 1 : y0;
    const float fx = x - (float)x0;
    const float fy = y - (float)y0;

    const float a = kSheenAlbedoLut[y0 * SHEEN_ALBEDO_LUT_SIZE + x0];
    const float b = kSheenAlbedoLut[y0 * SHEEN_ALBEDO_LUT_SIZE + x1];
    const float c = kSheenAlbedoLut[y1 * SHEEN_ALBEDO_LUT_SIZE + x0];
    const float d = kSheenAlbedoLut[y1 * SHEEN_ALBEDO_LUT_SIZE + x1];

    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

#endif // STRELKA_SHEEN_ALBEDO_LUT_H
