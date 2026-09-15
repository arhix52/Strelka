#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define STRELKA_GUIDE_FN __host__ __device__ inline
#else
#    define STRELKA_GUIDE_FN inline
#endif

namespace oka
{
namespace guides
{

/// Depth encodings. Mirrors kDenoiseDepth* in the Metal ShaderTypes.h.
enum : uint32_t
{
    kDepthDevice = 0u, ///< clip z / w, the value a depth buffer holds
    kDepthViewZ = 1u, ///< distance along the camera's forward axis
    kDepthRadial = 2u ///< distance to the eye
};

/// Below this a surface reflects rather than scatters, and its own albedo is not
/// what the pixel's colour comes from. Same number as the Metal integrator uses.
inline constexpr float kGuideRoughnessFloor = 0.05f;

/// Never walk forever looking for a describable surface: past a couple of
/// bounces the reflected surface has little to do with this pixel, and no guides
/// at all is worse than imperfect ones.
inline constexpr uint32_t kGuideLastChanceDepth = 2u;

/// What the background writes into the depth guide. Device depth has a finite
/// far plane, so the sentinel has to match the convention or the denoiser reads
/// the sky as being nearer than the geometry.
STRELKA_GUIDE_FN float backgroundDepth(uint32_t depthMode)
{
    return depthMode == kDepthDevice ? 0.0f : 1e7f;
}

STRELKA_GUIDE_FN bool guideWorthy(bool guidePrimaryHit, uint32_t depth, float roughness)
{
    return guidePrimaryHit ? (depth == 0u) : (roughness > kGuideRoughnessFloor);
}

/// The whole decision: write the guide record for this hit, or walk on.
STRELKA_GUIDE_FN bool shouldWriteGuide(
    bool writeAov, bool aovDone, bool guidePrimaryHit, uint32_t depth, float roughness)
{
    if (!writeAov || aovDone)
    {
        return false;
    }
    return guideWorthy(guidePrimaryHit, depth, roughness) || depth >= kGuideLastChanceDepth;
}

STRELKA_GUIDE_FN float reactiveFor(uint32_t guideDepth)
{
    return guideDepth > 0u ? 1.0f : 0.0f;
}

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

STRELKA_GUIDE_FN Vec2 screenMotion(
    float prevClipX, float prevClipY, float prevClipW, float currX, float currY, uint32_t width, uint32_t height)
{
    const float kMinW = 1e-4f;
    if (!(prevClipW > kMinW))
    {
        return Vec2{ 0.0f, 0.0f };
    }
    const float ndcX = prevClipX / prevClipW;
    const float ndcY = prevClipY / prevClipW;
    const float prevPixelX = (ndcX * 0.5f + 0.5f) * static_cast<float>(width);
    const float prevPixelY = (1.0f - (ndcY * 0.5f + 0.5f)) * static_cast<float>(height);

    // Nothing that moved further than the frame is across in one frame can be
    // reprojected onto anything. Clamped rather than zeroed so a genuinely fast
    // object still drags its history in the right direction.
    const float limit = static_cast<float>(width + height);
    float mx = prevPixelX - currX;
    float my = prevPixelY - currY;
    mx = mx < -limit ? -limit : (mx > limit ? limit : mx);
    my = my < -limit ? -limit : (my > limit ? limit : my);
    return Vec2{ mx, my };
}

STRELKA_GUIDE_FN float fireflyScale(float luminance, float threshold)
{
    if (!(threshold > 0.0f) || !(luminance > threshold))
    {
        return 1.0f;
    }
    return threshold / luminance;
}

} // namespace guides

/// Which built-in OptiX denoiser model a set of switches asks for.
enum class DenoiseModelKind : uint32_t
{
    eNone = 0,
    eAov, ///< single image
    eTemporalAov, ///< image sequence, temporally stable -- the MetalFX temporal analogue
    eUpscale2x, ///< single image, 2x
    eTemporalUpscale2x ///< image sequence, 2x -- the MetalFX upscaling analogue
};

/// Everything the frame needs to know about denoising, resolved in one place.
struct DenoisePlan
{
    DenoiseModelKind kind = DenoiseModelKind::eNone;
    /// Resolution the path tracer runs at. Half the output when upscaling.
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    /// Resolution the denoiser writes, which is the resolution the caller asked
    /// for.
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    bool temporal = false;
    bool upscale = false;
    /// Whether the frame has to produce guide records at all.
    bool writeAov = false;

    bool enabled() const
    {
        return kind != DenoiseModelKind::eNone;
    }
};

inline DenoisePlan denoisePlan(
    bool denoise, bool upscale, uint32_t upscaleMode, uint32_t debugMode, uint32_t width, uint32_t height)
{
    DenoisePlan plan;
    plan.outputWidth = width;
    plan.outputHeight = height;
    plan.renderWidth = width;
    plan.renderHeight = height;
    plan.writeAov = debugMode >= 3u; // any AOV debug view needs the records

    const bool canUpscale = upscale && width >= 2u && height >= 2u && (width % 2u) == 0u && (height % 2u) == 0u;
    if (!denoise && !canUpscale)
    {
        return plan;
    }

    plan.temporal = (upscaleMode == 1u);
    plan.upscale = canUpscale;
    plan.writeAov = true;
    if (canUpscale)
    {
        plan.renderWidth = width / 2u;
        plan.renderHeight = height / 2u;
        plan.kind = plan.temporal ? DenoiseModelKind::eTemporalUpscale2x : DenoiseModelKind::eUpscale2x;
    }
    else
    {
        plan.kind = plan.temporal ? DenoiseModelKind::eTemporalAov : DenoiseModelKind::eAov;
    }
    return plan;
}

/// Bytes of device memory the guide records and denoiser layers need, at a given
/// render resolution. `aovSampleBytes` is sizeof(AovSample) from the params
/// header, passed in so this stays free of CUDA types.
struct DenoiseBufferLayout
{
    size_t aovBytes = 0;
    size_t colorBytes = 0;
    size_t albedoBytes = 0;
    size_t normalBytes = 0;
    size_t flowBytes = 0;
    size_t flowTrustBytes = 0;
    size_t denoisedBytes = 0;
    uint32_t renderPixels = 0;
    uint32_t outputPixels = 0;
};

inline DenoiseBufferLayout denoiseBufferLayout(const DenoisePlan& plan, size_t aovSampleBytes)
{
    DenoiseBufferLayout out;
    out.renderPixels = plan.renderWidth * plan.renderHeight;
    out.outputPixels = plan.outputWidth * plan.outputHeight;
    out.aovBytes = static_cast<size_t>(out.renderPixels) * aovSampleBytes;
    // float4 throughout: the denoiser wants a 16-byte pixel stride for its
    // colour and guide layers, and a float3 layer would need its own repack.
    out.colorBytes = static_cast<size_t>(out.renderPixels) * 4 * sizeof(float);
    out.albedoBytes = out.colorBytes;
    out.normalBytes = out.colorBytes;
    // Flow is two components, and OptiX reads it at that stride.
    out.flowBytes = static_cast<size_t>(out.renderPixels) * 2 * sizeof(float);
    // How far the flow vector at each pixel is to be believed: one component.
    out.flowTrustBytes = static_cast<size_t>(out.renderPixels) * sizeof(float);
    out.denoisedBytes = static_cast<size_t>(out.outputPixels) * 4 * sizeof(float);
    return out;
}

} // namespace oka
