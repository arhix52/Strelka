#pragma once

// MetalFX upscaling.
//
// MetalFX has no metal-cpp binding, so this is the one place in the renderer
// that talks Objective-C directly. Everything crossing the boundary is either a
// metal-cpp pointer bridged in place or an opaque handle, so the rest of the
// renderer never sees an ObjC type.
//
// The spatial scaler is the simplest of the three effects: colour in, colour out,
// no history, no guides, no jitter. It is here first because it exercises the
// whole chain -- framework linkage, texture formats, encoding into our own
// command buffer -- with nothing else able to go wrong.

#include <Metal/Metal.hpp>

#include <cstdint>

namespace oka
{

class MetalFxContext
{
public:
    /// Colour processing mode, mirroring MTLFXSpatialScalerColorProcessingMode.
    enum class ColorMode : uint32_t
    {
        Perceptual = 0, ///< input is already tonemapped and gamma-encoded
        Linear = 1,
        HDR = 2,
    };

    /// Create or recreate the spatial scaler. A scaler is bound to its formats
    /// and both resolutions, so any change to those means a new one; the call is
    /// a no-op when nothing changed.
    bool ensureSpatialScaler(MTL::Device* device,
                             MTL::PixelFormat colorFormat,
                             MTL::PixelFormat outputFormat,
                             uint32_t inputWidth,
                             uint32_t inputHeight,
                             uint32_t outputWidth,
                             uint32_t outputHeight,
                             ColorMode colorMode,
                             /// MTL4::Compiler*, when the Metal 4 path is in use: the two
                             /// protocols are separate objects and each needs its own build.
                             void* metal4Compiler);

    /// Texture usage flags the scaler requires of the textures handed to it.
    /// Allocating without them is a validation failure at encode time.
    MTL::TextureUsage requiredColorUsage() const;
    MTL::TextureUsage requiredOutputUsage() const;

    bool hasSpatialScaler() const
    {
        return mSpatialScaler != nullptr;
    }

    /// Encode the upscale. `commandBuffer` is an MTL::CommandBuffer* for the
    /// Metal 3 path or an MTL4::CommandBuffer* for the Metal 4 one; which is
    /// meant is decided by `metal4`, because the two take different protocols.
    void encodeSpatial(void* commandBuffer,
                       bool metal4,
                       MTL::Texture* colorTexture,
                       MTL::Texture* outputTexture,
                       uint32_t inputContentWidth,
                       uint32_t inputContentHeight);

    /// The guide textures a temporal denoise needs, all at render resolution.
    struct DenoiseInputs
    {
        MTL::Texture* color = nullptr;    ///< linear radiance, before tonemapping
        MTL::Texture* depth = nullptr;
        MTL::Texture* motion = nullptr;   ///< previous-frame offset in pixels
        MTL::Texture* diffuseAlbedo = nullptr;
        MTL::Texture* specularAlbedo = nullptr;
        MTL::Texture* normal = nullptr;
        MTL::Texture* roughness = nullptr;
        MTL::Texture* output = nullptr;   ///< display resolution, linear
        float jitterX = 0.0f;             ///< the offset this frame was rendered with
        float jitterY = 0.0f;
        bool resetHistory = false;        ///< camera cut, scene change, resize
    };

    /// Create or recreate the temporal denoiser. Bound to its formats and both
    /// resolutions, like the spatial one.
    bool ensureDenoiser(MTL::Device* device,
                        uint32_t inputWidth,
                        uint32_t inputHeight,
                        uint32_t outputWidth,
                        uint32_t outputHeight,
                        void* metal4Compiler);

    bool hasDenoiser() const
    {
        return mDenoiser != nullptr;
    }

    /// Usage flags the denoiser demands of each input, in the order of
    /// DenoiseInputs. Guessing them fails at encode time, not at creation.
    MTL::TextureUsage denoiseColorUsage() const;
    MTL::TextureUsage denoiseGuideUsage() const;
    MTL::TextureUsage denoiseOutputUsage() const;

    void encodeDenoise(void* commandBuffer, bool metal4, const DenoiseInputs& inputs);

    void release();

private:
    void* mSpatialScaler = nullptr; ///< id<MTLFXSpatialScaler>, retained
    void* mSpatialScaler4 = nullptr; ///< id<MTL4FXSpatialScaler>, retained
    void* mDenoiser = nullptr; ///< id<MTLFXTemporalDenoisedScaler>, retained
    void* mDenoiser4 = nullptr; ///< id<MTL4FXTemporalDenoisedScaler>, retained
    uint32_t mDenoiseInputWidth = 0;
    uint32_t mDenoiseInputHeight = 0;
    uint32_t mDenoiseOutputWidth = 0;
    uint32_t mDenoiseOutputHeight = 0;
    MTL::PixelFormat mColorFormat = MTL::PixelFormatInvalid;
    MTL::PixelFormat mOutputFormat = MTL::PixelFormatInvalid;
    uint32_t mInputWidth = 0;
    uint32_t mInputHeight = 0;
    uint32_t mOutputWidth = 0;
    uint32_t mOutputHeight = 0;
    ColorMode mColorMode = ColorMode::Perceptual;
};

} // namespace oka
