#pragma once
// ============================================================================
// shading_common.h -- everything the path tracers share.
//
// Extracted verbatim from pathtrace.metal so the megakernel and the wavefront
// kernels evaluate identical shading, sampling and light code. Nothing here
// depends on how paths are scheduled; only the tracer kernels do.
// ============================================================================

#include <metal_stdlib>
#include <simd/simd.h>

#include "random_metal.h"
#include "lights_metal.h"
#include "env_light_metal.h"

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h>
#include <strelka/material/volume.h>
#include <strelka/material/bsdf.h>

using namespace metal;
using namespace raytracing;



// ---------------------------------------------------------------------------
// Feature specialisation.
//
// Every branch below is on a scene- or settings-level fact that does not change
// between rays, so leaving it in the instruction stream costs every ray in every
// scene. Function constants let the compiler delete the untaken side outright,
// which matters less for the branch itself than for the registers and texture
// state the dead code was keeping alive.
//
// Each constant falls back to `true` when the pipeline does not supply it, so
// the megakernel — which is built without constant values — keeps the full,
// unspecialised behaviour and needs no changes.
// ---------------------------------------------------------------------------
constant bool kFcEnvMap [[function_constant(0)]];
constant bool kFcLights [[function_constant(1)]];
constant bool kFcMotionBlur [[function_constant(2)]];
constant bool kFcDof [[function_constant(3)]];
constant bool kFcDebug [[function_constant(4)]];
constant bool kFcAlpha [[function_constant(5)]];
constant bool kFcFog [[function_constant(6)]];
constant bool kFcSharc [[function_constant(7)]];
constant bool kFcSubsurface [[function_constant(8)]];
constant bool kFcCurves [[function_constant(9)]];

constant bool SPEC_FOG = is_function_constant_defined(kFcFog) ? kFcFog : false;
constant bool SPEC_SHARC = is_function_constant_defined(kFcSharc) ? kFcSharc : false;
constant bool SPEC_SSS = is_function_constant_defined(kFcSubsurface) ? kFcSubsurface : false;
constant bool SPEC_ENV_MAP = is_function_constant_defined(kFcEnvMap) ? kFcEnvMap : true;
constant bool SPEC_LIGHTS = is_function_constant_defined(kFcLights) ? kFcLights : true;
constant bool SPEC_MOTION_BLUR = is_function_constant_defined(kFcMotionBlur) ? kFcMotionBlur : true;
constant bool SPEC_DOF = is_function_constant_defined(kFcDof) ? kFcDof : true;
constant bool SPEC_DEBUG = is_function_constant_defined(kFcDebug) ? kFcDebug : true;
// Only set when the scene actually contains a MASK or BLEND material. Scenes
// without cutouts then compile the same kernels they compiled before and pay
// nothing for the feature -- which matters most in the shadow stage, where the
// alternative to any-hit traversal is a loop over closest hits.
constant bool SPEC_ALPHA = is_function_constant_defined(kFcAlpha) ? kFcAlpha : true;
// Whether the scene has curve geometry. The traversal side of this cannot be a
// constant -- the intersector's tags decide what its result type carries, so a
// curve-capable traversal is a different kernel entirely -- but `shade` has no
// intersector, only the branch that rebuilds a hit strand, and that one is worth
// compiling out of every scene that has no hair in it.
constant bool SPEC_CURVES = is_function_constant_defined(kFcCurves) ? kFcCurves : false;

struct PerRayData
{
    SamplerState sampler;
    uint32_t depth;
    float3 radiance;
    float3 throughput;
    float3 origin;
    float3 direction;
    float lastBsdfPdf;
    IorStack iorStack;
    bool specularBounce;
    bool neeDone; // did NEE run at the vertex that spawned this ray?
    bool shouldTerninate;
};




__attribute__((always_inline))
float3 transformDirection(float3 p, float4x4 transform) {
    return (transform * float4(p.x, p.y, p.z, 0.0f)).xyz;
}

//  valid range of coordinates [-1; 1]
//
// z is 10 bits wide, not 12: bit 30 carries the tangent handedness sign that
// packTangent() writes, and folding it into z would warp the shading frame.
static float3 unpackNormal(uint32_t val)
{
    constexpr float scale = 1.0f / 256.0f;
    float3 normal;
    normal.z = ((val & 0x3ff00000) >> 20) * scale - 1.0f;
    normal.y = ((val & 0x000ffc00) >> 10) * scale - 1.0f;
    normal.x = (val & 0x000003ff) * scale - 1.0f;
    return normal;
}

// KHR_texture_transform. The spec composes it as a row-vector multiply,
//   [u v 1] * [ sx*cos(r)  sx*sin(r)  0 ]
//             [-sy*sin(r)  sy*cos(r)  0 ]
//             [ tx         ty         1 ]
// so scale applies before rotation and the translation last. Getting the order
// wrong is invisible at rotation 0 -- which is what every exporter writes by
// default -- and wrong everywhere else.
static float2 applyTextureTransform(float2 uv, device const Material& m)
{
    const float c = cos(m.uv_rotation);
    const float s = sin(m.uv_rotation);
    const float2 k = float2(m.uv_scale);
    return float2(uv.x * k.x * c - uv.y * k.y * s, uv.x * k.x * s + uv.y * k.y * c) +
           float2(m.uv_offset);
}

// Coverage of a surface at a given uv. MASK is a binary predicate, BLEND passes
// the alpha through, OPAQUE is always 1 -- so callers only ever see a float in
// [0,1] and never need to branch on the mode themselves.
static float resolveOpacity(device const Material& material, float2 uv)
{
    if (material.alpha_mode == ALPHA_MODE_OPAQUE)
        return 1.0f;
    // address::repeat, not Metal's clamp_to_edge default: glTF's default wrap is
    // REPEAT (10497), and a UV transform is the normal way to tile -- the marble
    // worktop in the bathroom scene repeats 2x2 and the plant's ramp 50x50. Under
    // clamping neither tiles; the edge texel is smeared across the whole surface,
    // which reads as a texture that simply did not load rather than as a wrap-mode
    // bug.
    constexpr sampler alphaSampler(mag_filter::linear, min_filter::linear, address::repeat);
    float alpha = material.base_color_alpha;
    if (!is_null_texture(material.baseColorTexture))
    {
        uv = applyTextureTransform(uv, material);
        // RGBA8Unorm_sRGB puts only RGB through the transfer function, so the
        // alpha channel read here is already linear.
        alpha *= material.baseColorTexture.sample(alphaSampler, uv).a;
    }
    if (material.alpha_mode == ALPHA_MODE_MASK)
        return alpha >= material.alpha_cutoff ? 1.0f : 0.0f;
    return saturate(alpha);
}

// glTF COLOR_0, packed RGBA8 and LINEAR -- it carries no transfer function,
// unlike a base-colour texture, so nothing is decoded here.
static float3 unpackVertexColor(uint32_t val)
{
    constexpr float s = 1.0f / 255.0f;
    return float3((val & 0xffu) * s, ((val >> 8) & 0xffu) * s, ((val >> 16) & 0xffu) * s);
}

// glTF TANGENT.w: +1 or -1, deciding which way the bitangent points.
static float unpackTangentSign(uint32_t val)
{
    return (val & (1u << 30)) ? -1.0f : 1.0f;
}

//  valid range of coordinates [-10; 10]
static float2 unpackUV(uint32_t val)
{
    float2 uv;
    uv.y = ((val & 0xffff0000) >> 16) / 16383.99999f * 20.0f - 10.0f;
    uv.x = (val & 0x0000ffff) / 16383.99999f * 20.0f - 10.0f;
    return uv;
}

static __attribute__((always_inline)) float3 interpolateAttrib(const float3 attr1, const float3 attr2, const float3 attr3, const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

static __attribute__((always_inline)) float2 interpolateAttrib(const float2 attr1, const float2 attr2, const float2 attr3, const float2 bary)
{
    return attr1 * (1.0f - bary.x - bary.y) + attr2 * bary.x + attr3 * bary.y;
}

// Whether a radiance carries any energy at all. Testing every channel instead
// would drop a saturated light: a pure red one has two zero channels and still
// lights the scene.
static __attribute__((always_inline)) bool emitsLight(const float3 radiance)
{
    return radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f;
}

__attribute__((always_inline))
float4x4 lerpMatrix(float4x4 a, float4x4 b, float t)
{
    float4x4 r;
    r[0] = mix(a[0], b[0], t);
    r[1] = mix(a[1], b[1], t);
    r[2] = mix(a[2], b[2], t);
    r[3] = mix(a[3], b[3], t);
    return r;
}

// Concentric disk mapping (Shirley & Chiu 1997)
float2 concentricDiskSample(float u1, float u2)
{
    float2 offset = float2(2.0f * u1 - 1.0f, 2.0f * u2 - 1.0f);
    if (offset.x == 0.0f && offset.y == 0.0f)
        return float2(0.0f, 0.0f);

    float theta, r;
    if (abs(offset.x) > abs(offset.y))
    {
        r = offset.x;
        theta = (M_PI_F / 4.0f) * (offset.y / offset.x);
    }
    else
    {
        r = offset.y;
        theta = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (offset.x / offset.y);
    }
    return float2(r * cos(theta), r * sin(theta));
}

// Sample regular polygon aperture (blades >= 3)
float2 samplePolygonAperture(float u1, float u2, int blades)
{
    float sectorAngle = 2.0f * M_PI_F / (float)blades;
    int sector = (int)(u1 * blades);
    if (sector >= blades) sector = blades - 1;
    float u = u1 * blades - (float)sector;

    float su = sqrt(u);
    float bary0 = 1.0f - su;
    float bary1 = u2 * su;

    float angle0 = sectorAngle * sector;
    float angle1 = sectorAngle * (sector + 1);

    float x = bary1 * cos(angle0) + (1.0f - bary0 - bary1) * cos(angle1);
    float y = bary1 * sin(angle0) + (1.0f - bary0 - bary1) * sin(angle1);
    return float2(x, y);
}

float2 sampleAperture(thread SamplerState& sampler, const constant Uniforms& params)
{
    float u1 = random<SampleDimension::eLensU>(sampler, params.samplerType);
    float u2 = random<SampleDimension::eLensV>(sampler, params.samplerType);

    float2 p;
    if (params.apertureBlades < 3)
        p = concentricDiskSample(u1, u2);
    else
        p = samplePolygonAperture(u1, u2, params.apertureBlades);

    if (params.bladeRotation != 0.0f)
    {
        float cosR = cos(params.bladeRotation);
        float sinR = sin(params.bladeRotation);
        p = float2(p.x * cosR - p.y * sinR, p.x * sinR + p.y * cosR);
    }

    p.y *= params.anamorphicRatio;
    return p;
}

// Bound what a single indirect path may contribute.
//
// A firefly is a sample with an enormous weight and a tiny probability -- a
// caustic that found the light through a specular chain, which is most of what a
// bathroom full of glass and chrome produces. Averaging it in is unbiased and
// does converge; the estimator is correct and the sample budget is not. Clamping
// trades that for bias, so it is off by default and applied only past the first
// bounce, where those paths live: clamping depth 0 as well would dim every
// directly visible emitter and the environment behind it.
inline float3 clampIndirectContribution(float3 radiance, uint depth, float limit)
{
    if (limit <= 0.0f || depth == 0u)
    {
        return radiance;
    }
    const float m = max(radiance.x, max(radiance.y, radiance.z));
    return (m > limit) ? radiance * (limit / m) : radiance;
}

void generateCameraRay(uint2 pixelIndex,
                        thread SamplerState& samplerRnd,
                        thread float3& origin,
                        thread float3& direction,
                        const constant Uniforms& params,
                        float motionTime)
{
    // A temporal upscaler reconstructs detail from a known per-frame shift, so
    // when one is running the whole image moves together and the per-pixel random
    // jitter -- which is antialiasing for a still frame -- would only add noise it
    // has to filter out.
    const float2 subpixel_jitter =
        params.useFrameJitter ?
            float2(params.jitterX + 0.5f, params.jitterY + 0.5f) :
            float2(random<SampleDimension::ePixelX>(samplerRnd, params.samplerType),
                   random<SampleDimension::ePixelY>(samplerRnd, params.samplerType));
    float2 pixelPos {pixelIndex.x + subpixel_jitter.x, params.height - (pixelIndex.y + subpixel_jitter.y)};

    float2 dimension {(float)params.width, (float)params.height};
    float2 pixelNDC = (pixelPos / dimension) * 2.0f - 1.0f;

    // Lens shift
    pixelNDC.x += params.shiftX * 2.0f;
    pixelNDC.y += params.shiftY * 2.0f;

    // Interpolate camera matrices for camera motion blur
    float4x4 clipToView = params.clipToView;
    float4x4 viewToWorld = params.viewToWorld;
    if (SPEC_MOTION_BLUR && motionTime < 1.0f && params.enableCameraMotionBlur)
    {
        clipToView = lerpMatrix(params.prevClipToView, params.clipToView, motionTime);
        viewToWorld = lerpMatrix(params.prevViewToWorld, params.viewToWorld, motionTime);
    }

    if (params.projectionType == PROJECTION_ORTHOGRAPHIC)
    {
        // No centre of projection: every ray runs down the view axis and the
        // pixel picks where on the film it starts. clipToView is deliberately
        // unused -- for an orthographic frame it is a scale, and going through it
        // would only re-derive the half-extents that are already here.
        const float3 filmPos = float3(pixelNDC.x * params.orthoHalfWidth,
                                      pixelNDC.y * params.orthoHalfHeight,
                                      0.0f);
        origin = (viewToWorld * float4(filmPos, 1.0f)).xyz;
        direction = normalize((viewToWorld * float4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);
    }
    else
    {
        float4 clip{ pixelNDC.x, pixelNDC.y, 1.0f, 1.0f };
        float4 viewSpace = clipToView * clip;

        float4 wdir = viewToWorld * float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f);

        origin = (viewToWorld * float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
        direction = normalize(wdir.xyz);
    }

    // Thin lens depth of field
    if (SPEC_DOF && params.useDof && params.lensRadius > 0.0f)
    {
        // The camera basis in world space is the columns of viewToWorld. This read
        // its rows, which transposes the rotation: both the lens offset and the axis
        // the focal distance is measured along then point somewhere else. The visible
        // effect is not a wrong blur but no blur at all -- past 45 degrees of yaw
        // dot(direction, camFwd) crosses zero, the clamp below throws the focal point
        // out to a million units, and depth of field silently stops happening for
        // every camera that is not axis aligned.
        float3 camRight = viewToWorld[0].xyz;
        float3 camUp = viewToWorld[1].xyz;
        float3 camFwd = -viewToWorld[2].xyz;

        float t = params.focalDistance / max(dot(direction, camFwd), 1e-6f);
        float3 focalPoint = origin + direction * t;

        float2 lensSample = sampleAperture(samplerRnd, params) * params.lensRadius;
        origin += camRight * lensSample.x + camUp * lensSample.y;
        direction = normalize(focalPoint - origin);
    }
}

// Ray-cone level of detail, following Akenine-Moller et al: the triangle term
// carries texels per world unit, the cone term carries how wide the footprint
// has grown, and the texture contributes its own resolution here because one
// material's slots are rarely all the same size.
template <typename Tex2D>
inline float texLod(Tex2D tex, float lodBase, bool hasLod)
{
    if (!hasLod)
    {
        return 0.0f;
    }
    const float dim = float(tex.get_width() * tex.get_height());
    return max(0.0f, lodBase + 0.5f * log2(max(dim, 1.0f)));
}

// Fill SurfaceInteraction from hit geometry and sample Material textures
void initSurfaceInteraction(
    thread SurfaceInteraction& si,
    const device Material& material,
    float3 worldPosition,
    float3 worldNormal,
    float3 geomNormal,
    float3 worldTangent,
    float3 worldBinormal,
    float2 uv,
    float3 rayDir,
    float3 vertexColor = float3(1.0f),
    // Ray-cone footprint for this hit, in log2 texels-per-unit *before* the
    // texture's own resolution is folded in -- each texture adds its own, since
    // the slots of one material are rarely the same size. FLT_MAX_10_EXP as the
    // sentinel would be cute; -1e30 says "no cone, use level 0" and is checked once.
    float lodBase = -1e30f)
{
    // Two samplers, not one with mip_filter::linear always on. Turning mip
    // filtering on changes the image even when every fetch asks for level 0 --
    // measured, 4.4/255 mean over 55% of pixels against the same render without
    // it -- so leaving it on would make "level of detail off" mean something
    // other than what the renderer did before this existed. With the switch off
    // the sampler, and the call, are exactly the originals.
    // See the note on alphaSampler: glTF's default wrap is REPEAT.
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear, address::repeat);
    constexpr sampler texSamplerMip(mag_filter::linear, min_filter::linear, mip_filter::linear,
                                    address::repeat);

    si.position       = worldPosition;
    si.shading_normal = worldNormal;
    si.geometry_normal = geomNormal;
    si.tangent        = worldTangent;
    si.bitangent      = worldBinormal;
    si.uv             = uv;
    // The cone gives texels per world unit; a texture turns that into a level
    // once its own resolution is known. Clamped at zero because a cone narrower
    // than a texel still wants the sharpest mip, not a negative one.
    const bool hasLod = lodBase > -1e29f;
    // One transform for every slot of the material -- see readTextureTransform()
    // in the loader for why that is not a compromise in practice.
    const float2 tuv = applyTextureTransform(uv, material);
    si.wo             = -rayDir;
    si.front_face     = dot(geomNormal, -rayDir) > 0.0f;

    // Sample base color texture. glTF composes base colour as
    // baseColorFactor * baseColorTexture * COLOR_0, all three multiplicative.
    float3 baseColor = float3(material.base_color) * vertexColor;
    if (!is_null_texture(material.baseColorTexture))
    {
        baseColor *= (hasLod ? material.baseColorTexture.sample(texSamplerMip, tuv, level(texLod(material.baseColorTexture, lodBase, hasLod))) : material.baseColorTexture.sample(texSampler, tuv)).rgb;
    }
    si.albedo = baseColor;
    si.opacity = resolveOpacity(material, uv);   // applies the transform itself

    // Sample metallic-roughness texture (glTF: G = roughness, B = metallic)
    float resolvedRoughness = material.roughness;
    float resolvedMetallic = material.metallic;
    if (!is_null_texture(material.metallicRoughnessTexture))
    {
        float4 mrTex = (hasLod ? material.metallicRoughnessTexture.sample(texSamplerMip, tuv, level(texLod(material.metallicRoughnessTexture, lodBase, hasLod))) : material.metallicRoughnessTexture.sample(texSampler, tuv));
        resolvedRoughness *= mrTex.g;
        resolvedMetallic *= mrTex.b;
    }

    // Sample normal map
    if (!is_null_texture(material.normalTexture))
    {
        float3 bumpNormal = (hasLod ? material.normalTexture.sample(texSamplerMip, tuv, level(texLod(material.normalTexture, lodBase, hasLod))) : material.normalTexture.sample(texSampler, tuv)).xyz * 2.0f - 1.0f;
        bumpNormal.xy *= material.normal_scale;
        float3x3 TBN = float3x3(worldTangent, worldBinormal, worldNormal);
        si.shading_normal = normalize(TBN * bumpNormal);
    }

    // Sample emission texture
    float3 emissionColor = float3(material.emission);
    if (!is_null_texture(material.emissionTexture))
    {
        float4 emTex = (hasLod ? material.emissionTexture.sample(texSamplerMip, tuv, level(texLod(material.emissionTexture, lodBase, hasLod))) : material.emissionTexture.sample(texSampler, tuv));
        emissionColor *= emTex.rgb;
    }
    si.emission = emissionColor * material.emission_strength;

    // Fill remaining material parameters for bsdf_init. Zero-initialised because
    // that is the default bsdf_init is written against, and a field added to the
    // struct but forgotten here would otherwise be read as stack garbage.
    MaterialParams matParams = {};
    matParams.roughness = resolvedRoughness;
    matParams.metallic = resolvedMetallic;
    matParams.ior = material.ior;
    matParams.transmission = material.transmission;
    matParams.clearcoat = material.clearcoat;
    matParams.clearcoat_roughness = material.clearcoat_roughness;
    matParams.anisotropy = material.anisotropy;
    matParams.specular = material.specular;
    matParams.specular_color = float3(material.specular_color);
    // si.subsurface is what gates the random walk in shade(); leaving it out of
    // this copy left every subsurface material behaving as plain diffuse
    // transmission, with the mean free path having no effect on the image at all.
    matParams.subsurface = material.subsurface;
    matParams.subsurface_radius = float3(material.subsurface_radius);
    matParams.subsurface_anisotropy = material.subsurface_anisotropy;
    matParams.subsurface_reference = float3(material.subsurface_reference);
    matParams.iridescence = material.iridescence;
    matParams.iridescence_ior = material.iridescence_ior;
    matParams.iridescence_thickness = material.iridescence_thickness;
    matParams.diffuse_transmission = material.diffuse_transmission;
    matParams.diffuse_transmission_color = float3(material.diffuse_transmission_color);
    matParams.clearcoat_ior = material.clearcoat_ior;
    matParams.sheen = material.sheen;
    matParams.sheen_roughness = material.sheen_roughness;
    matParams.sheen_color = float3(material.sheen_color);
    matParams.material_type = material.material_type;
    matParams.thin_walled = material.thin_walled;
    matParams.dielectric_priority = material.dielectric_priority;

    // bsdf_init (Metal overload) clamps and finalizes derived values
    bsdf_init(si, matParams);
    // Restore texture-resolved values that bsdf_init may have overwritten
    si.roughness = max(resolvedRoughness, 0.0001f);
    si.metallic = saturate(resolvedMetallic);
}

bool traceOcclusion(
    acceleration_structure<instancing, primitive_motion> accelerationStructure,
    thread intersector<triangle_data, instancing, primitive_motion>& isect,
    const float3 origin, 
    const float3 direction,
    const float tMin,
    const float tMax,
    const float motionTime)
{
    struct ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = direction;
    shadowRay.min_distance = tMin;
    shadowRay.max_distance = tMax;
    isect.accept_any_intersection(true);

    bool res = true;
    typename intersector<triangle_data, instancing, primitive_motion>::result_type intersection;
    intersection = isect.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, motionTime);
    if (intersection.type == intersection_type::none)
    {
        res = false;
    }
    isect.accept_any_intersection(false);
    return res;
}

// A next-event connection, before the visibility test.
//
// Splitting the light sample from the occlusion trace is what lets the wavefront
// tracer defer the shadow ray into its own stage while the megakernel keeps
// tracing it inline: both build the same connection, they just resolve it at
// different times. Nothing here depends on the trace's result, so the split
// changes no arithmetic and draws no extra random numbers.
struct LightConnection
{
    float3 radiance;  // unoccluded Li times the cosine at the surface
    float3 toLight;   // shadow ray direction
    float3 origin;    // shadow ray origin
    float pdf;
    float tMin;
    float tMax;
    bool needsRay;    // false when the connection is degenerate and contributes nothing
    // A delta light has no area, so BSDF sampling can never generate a direction
    // that hits it and there is no second strategy to combine with. Its pdf is a
    // placeholder of 1, not a solid-angle density, so feeding it to the balance
    // heuristic would silently scale the contribution by 1/(1 + pdf_bsdf).
    bool isDelta;
};

static LightConnection makeEmptyConnection()
{
    LightConnection c;
    c.radiance = float3(0.0f);
    c.toLight = float3(0.0f);
    c.origin = float3(0.0f);
    c.pdf = 0.0f;
    c.tMin = 0.0f;
    c.tMax = 0.0f;
    c.needsRay = false;
    c.isDelta = false;
    return c;
}

LightConnection connectLight(constant Uniforms& uniforms,
                             thread SamplerState& samplerRnd,
                             device const UniformLight& light,
                             thread SurfaceInteraction& si,
                             // A scattering event in a medium has a position and no normal. The facing
                             // test and the cosine below are surface terms; applied to a volume they
                             // reject half of every connection and darken the other half.
                             bool volumeEvent,
                             device const IesGpuBufferHeader* iesBuffer)
{
    LightSampleData lightSampleData = {};
    const float2 uv = float2(random<SampleDimension::eLightPointX>(samplerRnd, uniforms.samplerType), random<SampleDimension::eLightPointY>(samplerRnd, uniforms.samplerType));
    switch (light.type)
    {
    case 0:
        if (uniforms.rectLightSamplingMethod == 0)
        {
            lightSampleData = SampleRectLightUniform(light, uv, si.position);
        }
        else
        {
            lightSampleData = SampleRectLight(light, uv, si.position);
        }
        break;
    case 1:
        lightSampleData = SampleDiscLight(light, uv, si.position);
        break;
    case 2:
        lightSampleData = SampleSphereLight(light, uv, si.position);
        break;
    case 3:
        lightSampleData = SampleDistantLight(light, uv, si.position);
        break;
    case 5: // point
    case 6: // spot
        lightSampleData = SamplePointLight(light, uv, si.position);
        break;
    }

    LightConnection c = makeEmptyConnection();
    c.toLight = lightSampleData.L;
    // Sharp point/spot only: give one a radius and it is sampled as a sphere,
    // which BSDF rays can hit and which therefore does need MIS.
    c.isDelta = (light.type == 5 || light.type == 6) && !(light.points[0].x > 1e-4f);

    float3 Li = float3(light.color);
    // Point/spot colour is radiant intensity: convert to irradiance on the
    // surface by the inverse-square law. Soft points sampled as spheres still
    // carry intensity, so divide by the distance to the sampled point.
    if (light.type == 5 || light.type == 6)
    {
        const float dist = max(lightSampleData.distToLight, 1e-4f);
        Li *= rangeWindow(light, dist) / (dist * dist);
        // IES replaces the isotropic (and, for spots, the cone) angular shape:
        // the file is already in candela, and the light's intensity is a
        // multiplier on top of it. No profile means the cone alone, as before.
        const bool hasIes = light.points[0].y >= 0.0f;
        if (hasIes)
        {
            Li *= sampleIesCandela(iesBuffer, light, -lightSampleData.L);
        }
        else if (light.type == 6)
        {
            Li *= spotAttenuation(light, -lightSampleData.L);
        }
    }

    // For area lights the facing test uses the light's surface normal; for a
    // sharp point the "normal" is -L, so -dot(L, normal) = 1 always.
    const bool facing =
        volumeEvent
            ? (emitsLight(Li) &&
               (light.type == 5 || light.type == 6 ||
                -dot(lightSampleData.L, lightSampleData.normal) > 0.001f))
        : (light.type == 5 || light.type == 6)
            ? (dot(si.shading_normal, lightSampleData.L) > 0.0f && emitsLight(Li))
            : (dot(si.shading_normal, lightSampleData.L) > 0.0f &&
               -dot(lightSampleData.L, lightSampleData.normal) > 0.001f && emitsLight(Li));
    if (facing)
    {
        // The cosine belongs here because bsdf_eval() returns f alone, unlike
        // bsdf_sample()'s bsdf_over_pdf which already carries it. See the note on
        // both result structs in bsdf_types.h.
        c.radiance = volumeEvent ? Li : Li * saturate(dot(si.shading_normal, lightSampleData.L));
        c.origin = si.position;
        c.pdf = lightSampleData.pdf;
        c.tMin = 0.001f;
        c.tMax = lightSampleData.distToLight - 1e-5f;
        c.needsRay = true;
    }
    return c;
}

__attribute__((always_inline))
int __float_as_int(float x)
{
    return as_type<int>(x);
}
__attribute__((always_inline))
float __int_as_float(int x)
{
    return as_type<float>(x);
}

static float3 offset_ray(const float3 p, const float3 n)
{
    const float origin = 1.0f / 32.0f;
    const float float_scale = 1.0f / 65536.0f;
    const float int_scale = 256.0f;

    int3 of_i = int3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    float3 p_i = float3(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                             __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                             __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                       abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                       abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

LightConnection connectEnvLight(
    constant Uniforms& uniforms,
    thread SamplerState& samplerRnd,
    thread SurfaceInteraction& si,
    device const EnvAliasEntry* envAliasTable,
    texture2d<float> envMapTexture,
    bool volumeEvent)
{
    const float2 xi = float2(
        random<SampleDimension::eLightPointX>(samplerRnd, uniforms.samplerType),
        random<SampleDimension::eLightPointY>(samplerRnd, uniforms.samplerType));

    float envPdf = 0.0f;
    float3 dir = sampleEnvMap(xi,
                              envAliasTable, envMapTexture,
                              uniforms.envMapWidth, uniforms.envMapHeight,
                              uniforms.envMapRotation,
                              uniforms.envPdfScale,
                              envPdf);

    LightConnection c = makeEmptyConnection();
    c.toLight = dir;
    c.pdf = envPdf;

    if (envPdf <= 0.0f)
        return c;

    if (!volumeEvent && dot(si.shading_normal, dir) <= 0.0f)
        return c;

    constexpr sampler envSampler(mag_filter::linear, min_filter::linear, address::repeat, coord::normalized);
    const float2 uv = dirToEnvUV(dir, uniforms.envMapRotation);
    const float4 envSample = envMapTexture.sample(envSampler, uv);
    float3 Li = envSample.xyz;
    Li *= uniforms.envMapIntensity * float3(uniforms.envMapColorTint);

    // Cosine folded in here for the same reason as in connectLight().
    c.radiance = volumeEvent ? Li : Li * max(dot(si.shading_normal, dir), 0.0f);
    // Offset along the face the shadow ray actually leaves from. The raw
    // geometry normal points to a fixed side of the triangle, so on a back-face
    // hit it pushes the origin *into* the surface and the ray immediately hits
    // the geometry it started on — NEE then reports occlusion that the BSDF
    // strategy does not see, and the two estimators disagree. The bounce ray in
    // the main loop already orients its offset this way.
    const float3 offsetNg = (dot(si.geometry_normal, dir) > 0.0f) ? si.geometry_normal : -si.geometry_normal;
    c.origin = offset_ray(si.position, offsetNg);
    c.tMin = 0.001f;
    c.tMax = 1e16f;
    c.needsRay = true;
    return c;
}

// Choose a strategy and build the connection. The caller decides when to test
// visibility.
LightConnection connectToLight(constant Uniforms& uniforms,
                               const uint32_t numLights,
                               device UniformLight* lights,
                               thread SamplerState& samplerRnd,
                               thread SurfaceInteraction& si,
                               device const EnvAliasEntry* envAliasTable,
                               texture2d<float> envMapTexture,
                               device const IesGpuBufferHeader* iesBuffer,
                               bool volumeEvent = false)
{
    if (SPEC_ENV_MAP && uniforms.hasEnvMap)
    {
        const float u = random<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);

        if (!SPEC_LIGHTS || numLights == 0 || u >= 0.5f)
        {
            const float selectionPdf = (numLights > 0) ? 0.5f : 1.0f;
            LightConnection c =
                connectEnvLight(uniforms, samplerRnd, si, envAliasTable, envMapTexture, volumeEvent);
            c.pdf *= selectionPdf;
            return c;
        }
        // Sample a local light (remap u from [0, 0.5) to [0, 1)).
        const float remappedU = u * 2.0f;
        const uint32_t lightId = min((uint32_t)(numLights * remappedU), numLights - 1);
        LightConnection c = connectLight(uniforms, samplerRnd, lights[lightId], si, volumeEvent, iesBuffer);
        c.pdf *= 0.5f / numLights;
        return c;
    }

    // No env map and no analytic lights: nothing to connect to. Falling through
    // would divide by numLights == 0, produce a NaN light PDF, and trip the
    // isnan() guard in the caller that paints the pixel bright red.
    if (!SPEC_LIGHTS || numLights == 0)
    {
        return makeEmptyConnection();
    }

    const float u = random<SampleDimension::eLightId>(samplerRnd, uniforms.samplerType);
    const uint32_t lightId = min((uint32_t)(numLights * u), numLights - 1);
    LightConnection c = connectLight(uniforms, samplerRnd, lights[lightId], si, volumeEvent, iesBuffer);
    c.pdf *= 1.0f / numLights;
    return c;
}

