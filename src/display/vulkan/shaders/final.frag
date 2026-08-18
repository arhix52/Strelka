#version 450 core

layout(constant_id = 0) const bool hdrOutput = false;

layout(set = 0, binding = 0) uniform sampler2D compositionTexture;
layout(push_constant) uniform OutputParameters
{
    float paperWhiteNits;
    float peakNits;
}
parameters;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outputColor;

vec3 rec709ToRec2020(vec3 color)
{
    return mat3(
        0.6274040, 0.0690970, 0.0163916,
        0.3292820, 0.9195400, 0.0880132,
        0.0433136, 0.0113612, 0.8955950) * color;
}

vec3 st2084EncodeNits(vec3 luminanceNits)
{
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    vec3 luminance = clamp(luminanceNits / 10000.0, vec3(0.0), vec3(1.0));
    vec3 powered = pow(luminance, vec3(m1));
    return pow((c1 + c2 * powered) / (1.0 + c3 * powered), vec3(m2));
}

void main()
{
    vec4 composition = texture(compositionTexture, uv);
    if (hdrOutput)
    {
        vec3 rec2020 = rec709ToRec2020(composition.rgb);
        vec3 nits = clamp(rec2020 * parameters.paperWhiteNits,
                          vec3(0.0), vec3(parameters.peakNits));
        outputColor = vec4(st2084EncodeNits(nits), composition.a);
    }
    else
    {
        outputColor = composition;
    }
}
