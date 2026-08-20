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
    bool useMetal4 = false;
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
    if (in.useMetal4)
        features |= WavefrontFeatures::kMetal4;
    return WavefrontFeatures(features);
}

} // namespace oka::metal

