#version 450 core

layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform texture2D imageTexture;
layout(set = 1, binding = 0) uniform sampler imageSampler;
layout(location = 0) in InputData
{
    vec4 color;
    vec2 uv;
}
inputData;

vec3 srgbToLinear(vec3 color)
{
    bvec3 linearRange = lessThanEqual(color, vec3(0.04045));
    vec3 low = color / 12.92;
    vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, linearRange);
}

void main()
{
    vec4 tint = vec4(srgbToLinear(inputData.color.rgb), inputData.color.a);
    fColor = tint * texture(sampler2D(imageTexture, imageSampler), inputData.uv);
}
