#pragma once

// Guide-production and denoiser-configuration rules for the OptiX backend.
//
// Pure arithmetic and pure decisions, no CUDA and no OptiX headers, so the parts
// of the denoiser that can be wrong without a GPU can be tested without one.
// OptixRender.cpp static_asserts that the production struct sizes match what
// this header is told, in the same way src/render/host/integrator_buffer_sizes.h
// does for the Metal integrator.
//
// The device code includes this too, which is why the functions carry a
// qualifier macro instead of plain `inline`: the alternative is a second copy of
// the same rules living in a .cu, and a guide rule that exists twice is a guide
// rule that will disagree with itself.

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

/// Whether this surface is one the denoiser can be told about.
///
/// A mirror or a pane of glass has no albedo to demodulate against and a
/// roughness of nothing, so guides taken there say a featureless black surface
/// sits where a whole reflected world is. With `guidePrimaryHit` the
/// camera-visible surface is the answer by definition, so the roughness floor --
/// and the flicker it causes where a thin material sits on top of it -- does not
/// enter into it.
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

/// "The history for this pixel is not valid." The one thing that makes it so is
/// guides describing a surface other than the one the camera sees -- the
/// deferred-guide case, a primary hit too smooth to describe. Water, glass, a
/// mirror.
///
/// It deliberately does not scale with the motion vector: that marks the whole
/// frame the moment the camera moves at all, and a pixel that moved is
/// reprojectable rather than untrustworthy.
STRELKA_GUIDE_FN float reactiveFor(uint32_t guideDepth)
{
    return guideDepth > 0u ? 1.0f : 0.0f;
}

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

/// Where a point was on screen last frame, minus where it is now, in pixels,
/// y down.
///
/// `prevClip*` is the point run through the previous frame's world-to-clip
/// matrix; `curr*` is the *jittered* sample position the camera ray actually
/// went through, not the pixel centre.
STRELKA_GUIDE_FN Vec2 screenMotion(
    float prevClipX, float prevClipY, float prevClipW, float currX, float currY, uint32_t width, uint32_t height)
{
    // A w at or near zero is a point on the previous camera's plane, and dividing
    // by it does not produce a large motion vector, it produces a meaningless
    // one. Zero is the honest answer for a reprojection that has none: it says
    // "this pixel did not move", and every caller that can reach this case
    // already marks the pixel reactive.
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

/// Factor to multiply a colour by so its luminance does not exceed `threshold`.
///
/// A single unbounded sample is a bright dot that a temporal filter then smears
/// across many frames, so it costs far more than the energy it carries. Scaled
/// rather than dropped, so the pixel keeps its hue and most of its brightness.
/// Off when the threshold is zero.
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

/// Resolve the render.denoise / render.upscale / render.upscale_mode switches
/// and the debug view into one plan.
///
/// `upscale` on its own selects the non-temporal 2x model, which is the closest
/// OptiX has to MetalFX's spatial scaler; `upscaleMode == 1` ("temporal") or
/// denoising with a valid history selects the temporal one. Upscaling implies
/// denoising here, because unlike MetalFX there is no OptiX path that scales
/// without also running the network.
///
/// Odd render sizes are the reason the 2x models take the *floor* of half the
/// output and then produce an image that may be a pixel short: rather than
/// silently returning a differently sized frame, upscaling is declined when the
/// output dimensions are not even.
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

    // `render.upscale_mode = "temporal"` selects the temporally stable model,
    // for denoising as well as for upscaling. Off by default: a temporal model
    // asked to reuse a history it has no motion vectors for produces a smear,
    // and a headless render of a still camera has nothing to gain from it.
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
