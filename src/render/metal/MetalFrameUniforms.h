#pragma once

#include "MetalEnvironment.h"
#include "MetalMaterials.h"
#include "sampling_math.h"

#include "ShaderTypes.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/render/Camera.h>
#include <strelka/scene/scene.h>

#include <cstdint>

#include <glm/glm.hpp>

namespace oka
{
namespace metal
{

static constexpr size_t kFrameUniformSlots = 3;

// Camera / jitter / exposure / SHARC → Uniforms fill. Owns the per-frame uniform
// ring buffers and the radiance-cache buffer.
class MetalFrameUniforms
{
public:
    struct CameraView
    {
        oka::Camera::Matrices mCamMatrices;
    };

    // Inputs assembled by the orchestrator for one frame's Uniforms write.
    struct FillInput
    {
        SettingsManager* settings = nullptr;
        Scene* scene = nullptr;
        MetalMaterials* materials = nullptr;
        MetalEnvironment* environment = nullptr;
        MTL::Buffer* accumulationBuffer = nullptr;

        uint32_t frameSlot = 0;
        uint32_t subframeIndex = 0;
        uint64_t frameNumber = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t outWidth = 0;
        uint32_t outHeight = 0;
        uint32_t spp = 0;
        uint32_t sspTotal = 0;
        uint32_t maxDepth = 0;
        uint32_t debug = 0;
        uint32_t rectLightSamplingMethod = 0;
        uint32_t samplerType = 0;
        uint32_t blueNoiseSwitchSpp = 0;

        bool enableAccumulation = false;
        bool anyAnimationPlaying = false;
        bool denoising = false;
        bool enableMotionBlur = false;
        bool isMotionBlurVisible = false;
        bool enableCameraMotionBlur = false;
        bool pausedBlurRefine = false;
        bool resetDenoiseHistory = false;
        bool hasPrevFramePose = false;
        bool noPrevPose = false;
        bool noAccumColor = false;

        const oka::Camera* camera = nullptr;
        const CameraView* currView = nullptr;
        const CameraView* prevView = nullptr;
        const CameraView* prevMotionBlurView = nullptr;
    };

    struct FillResult
    {
        Uniforms* uniforms = nullptr;
        UniformsTonemap* tonemap = nullptr;
        MTL::Buffer* uniformBuffer = nullptr;
        MTL::Buffer* tonemapBuffer = nullptr;
        bool accumulationActive = false;
        bool effectiveAccumulation = false;
        uint32_t remainingSamples = 0;
        uint32_t samplesThisLaunch = 0;
        float jitterX = 0.0f;
        float jitterY = 0.0f;
        bool settingsChanged = false;
    };

    MetalFrameUniforms() = default;
    ~MetalFrameUniforms();

    void init(MTL::Device* device);
    void release();

    void allocateRings();
    void ensureSharc(SettingsManager* settings, const oka::Camera& camera, uint32_t width, uint32_t height);

    MTL::Buffer* uniformBuffer(uint32_t slot) const
    {
        return mUniformBuffers[slot % kFrameUniformSlots];
    }
    MTL::Buffer* tonemapBuffer(uint32_t slot) const
    {
        return mUniformTMBuffers[slot % kFrameUniformSlots];
    }
    MTL::Buffer* const* uniformBuffers() const
    {
        return mUniformBuffers;
    }
    MTL::Buffer* const* tonemapBuffers() const
    {
        return mUniformTMBuffers;
    }
    MTL::Buffer* sharcBuffer() const
    {
        return mSharcBuffer;
    }
    uint32_t sharcCapacity() const
    {
        return mSharcCapacity;
    }

    FillResult fill(const FillInput& in);

private:
    struct PrevSettings
    {
        uint32_t rectLightSamplingMethod = 0;
        uint32_t samplerType = 0;
        uint32_t blueNoiseSwitchSpp = 0;
        bool enableAccumulation = false;
        uint32_t sspTotal = 0;
        uint32_t spp = 0;
        bool playbackBlur = false;
        uint32_t shutterMode = 0;
        float shutterTime = 0.0f;
        bool enableMotionBlur = false;
        bool isMotionBlurVisible = true;
        bool enableCameraMotionBlur = false;
        int32_t useDof = 0;
        float focalDistance = 0.0f;
        float lensRadius = 0.0f;
        int32_t apertureBlades = 0;
        float shiftX = 0.0f;
        float shiftY = 0.0f;
        uint32_t maxDepth = 0;
        uint32_t debug = 0;
        float clampIndirect = 0.0f;
    };

    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mUniformBuffers[kFrameUniformSlots] = {};
    MTL::Buffer* mUniformTMBuffers[kFrameUniformSlots] = {};
    MTL::Buffer* mSharcBuffer = nullptr;
    uint32_t mSharcCapacity = 0;
    PrevSettings mPrevSettings;
};

} // namespace metal
} // namespace oka
