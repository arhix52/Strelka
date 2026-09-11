#pragma once

#include <cstdint>


namespace oka::metal
{

// Strong bitmask for wavefront function-constant specialisation. Pack through
// packWavefrontFeatures() — do not assemble bits ad-hoc in the frame loop.
class WavefrontFeatures
{
public:
    static constexpr uint32_t kEnvMap = 1u << 0;
    static constexpr uint32_t kLights = 1u << 1;
    static constexpr uint32_t kMotionBlur = 1u << 2;
    static constexpr uint32_t kDof = 1u << 3;
    static constexpr uint32_t kDebug = 1u << 4;
    // Not a shader feature: Metal 3/4 pipelines are not interchangeable.
    static constexpr uint32_t kMetal4 = 1u << 5;
    static constexpr uint32_t kAlpha = 1u << 6;
    static constexpr uint32_t kFog = 1u << 7;
    static constexpr uint32_t kSharc = 1u << 8;
    static constexpr uint32_t kSubsurface = 1u << 9;
    static constexpr uint32_t kCurves = 1u << 10;
    // Dedicated sparse cache update. kSharc without this bit is the full-frame
    // query pass; update never queries its own writes.
    static constexpr uint32_t kSharcUpdate = 1u << 11;
    // OpenPBR Surface. Carries ~264 KB of lookup tables and a lobe stack that
    // no glTF scene needs, so a scene without an OpenPBR material must compile
    // a kernel in which none of it exists -- the room scenes are instruction
    // cache bound (docs/open-perf.md) and a second uber-BSDF compiled in
    // unconditionally would undo the specialisation work outright.
    static constexpr uint32_t kOpenPBR = 1u << 12;
    // Debug-only counters. This bit selects a separately specialised pipeline;
    // ordinary Release kernels contain no atomics or counter loads.
    static constexpr uint32_t kRenderWorkAudit = 1u << 13;
    static constexpr uint32_t kRestirRayTracedDiagnostic = 1u << 14;
    static constexpr uint32_t kRestir = 1u << 15;
    // Plain NEE with one candidate has no reservoir to maintain. This bit lets
    // the shader compile the generic RIS loop and its state out entirely.
    static constexpr uint32_t kRisOne = 1u << 16;
    // Primary denoiser/MetalFX guides are optional. A no-AOV render should not
    // carry their large material and motion path through wavefrontShade.
    static constexpr uint32_t kAov = 1u << 17;
    // Stronger than kOpenPBR: every shadeable material uses OpenPBR, so the
    // standard surface model can be deleted from the shader.
    static constexpr uint32_t kAllOpenPBR = 1u << 18;
    static constexpr uint32_t kSamplerShift = 19u;
    static constexpr uint32_t kSamplerMask = 7u << kSamplerShift;
    // Every surface was authored as OpenPBR rather than translated from glTF.
    // This lets shade delete the generic material initializer entirely.
    static constexpr uint32_t kAllNativeOpenPBR = 1u << 22;
    // Emissive triangle NEE pulls geometry reconstruction and mesh-light alias
    // sampling into shade. Keep it out of scenes that only have analytic lights.
    static constexpr uint32_t kEmissiveMeshLights = 1u << 23;
    // A common production-lighting case. It deletes the other seven analytic
    // samplers, IES/projector code and their PDF branches from shade.
    static constexpr uint32_t kAllAnalyticLightsRect = 1u << 24;
    // Rectangle sampling is a binary render setting. Specialising it avoids
    // charging the default uniform-area PSO for the much larger spherical-quad
    // sampler and its peak register footprint.
    static constexpr uint32_t kUniformRectLightSampling = 1u << 25;
    // Plain one-candidate Base NEE can generate its light proposal in a small
    // stage before OpenPBR prepare/eval, avoiding their combined register peak.
    static constexpr uint32_t kSplitBaseNee = 1u << 26;

    WavefrontFeatures() = default;
    explicit WavefrontFeatures(uint32_t bits) : mBits(bits)
    {
    }

    uint32_t bits() const
    {
        return mBits;
    }

    bool has(uint32_t flag) const
    {
        return (mBits & flag) != 0;
    }

    WavefrontFeatures with(uint32_t flag) const
    {
        return WavefrontFeatures(mBits | flag);
    }

    bool operator==(WavefrontFeatures o) const
    {
        return mBits == o.mBits;
    }
    bool operator!=(WavefrontFeatures o) const
    {
        return mBits != o.mBits;
    }

private:
    uint32_t mBits = 0;
};

struct IntegratorFeatureInputs
{
    bool hasEnvMap = false;
    bool hasLights = false;
    bool hasAlphaMaterials = false;
    bool enableMotionBlur = false;
    bool motionBlasBuilt = false;
    bool enableCameraMotionBlur = false;
    bool useDof = false;
    bool debug = false;
    bool hasFog = false;
    bool hasSharc = false;
    bool hasSubsurface = false;
    bool hasCurves = false;
    bool hasOpenPBR = false;
    bool useMetal4 = false;
    bool auditRenderWork = false;
    bool restirRayTracedDiagnostic = false;
    bool restir = false;
    bool risOne = false;
    bool writeAov = false;
    bool allOpenPBR = false;
    bool allNativeOpenPBR = false;
    bool hasEmissiveMeshLights = false;
    bool allAnalyticLightsRect = false;
    bool uniformRectLightSampling = false;
    bool splitBaseNee = false;
    uint32_t samplerType = 0u;
};

inline WavefrontFeatures packWavefrontFeatures(const IntegratorFeatureInputs& in)
{
    uint32_t features = 0;
    if (in.hasEnvMap)
        features |= WavefrontFeatures::kEnvMap;
    if (in.hasLights)
        features |= WavefrontFeatures::kLights;
    if (in.hasAlphaMaterials)
        features |= WavefrontFeatures::kAlpha;
    if (in.enableMotionBlur && (in.motionBlasBuilt || in.enableCameraMotionBlur))
        features |= WavefrontFeatures::kMotionBlur;
    if (in.useDof)
        features |= WavefrontFeatures::kDof;
    if (in.debug)
        features |= WavefrontFeatures::kDebug;
    if (in.hasFog)
        features |= WavefrontFeatures::kFog;
    if (in.hasSharc)
        features |= WavefrontFeatures::kSharc;
    if (in.hasSubsurface)
        features |= WavefrontFeatures::kSubsurface;
    if (in.hasCurves)
        features |= WavefrontFeatures::kCurves;
    if (in.hasOpenPBR)
        features |= WavefrontFeatures::kOpenPBR;
    if (in.useMetal4)
        features |= WavefrontFeatures::kMetal4;
    if (in.auditRenderWork)
        features |= WavefrontFeatures::kRenderWorkAudit;
    if (in.restirRayTracedDiagnostic)
        features |= WavefrontFeatures::kRestirRayTracedDiagnostic;
    if (in.restir)
        features |= WavefrontFeatures::kRestir;
    if (in.risOne && !in.restir)
        features |= WavefrontFeatures::kRisOne;
    if (in.writeAov)
        features |= WavefrontFeatures::kAov;
    if (in.allOpenPBR)
        features |= WavefrontFeatures::kAllOpenPBR | WavefrontFeatures::kOpenPBR;
    if (in.allNativeOpenPBR)
        features |= WavefrontFeatures::kAllNativeOpenPBR | WavefrontFeatures::kAllOpenPBR | WavefrontFeatures::kOpenPBR;
    if (in.hasEmissiveMeshLights)
        features |= WavefrontFeatures::kEmissiveMeshLights | WavefrontFeatures::kLights;
    if (in.allAnalyticLightsRect)
        features |= WavefrontFeatures::kAllAnalyticLightsRect;
    if (in.uniformRectLightSampling)
        features |= WavefrontFeatures::kUniformRectLightSampling;
    if (in.splitBaseNee && in.risOne && !in.restir)
        features |= WavefrontFeatures::kSplitBaseNee;
    features |= (in.samplerType & 7u) << WavefrontFeatures::kSamplerShift;
    return WavefrontFeatures(features);
}

} // namespace oka::metal
