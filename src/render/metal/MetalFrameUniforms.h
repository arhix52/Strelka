#pragma once

#include "MetalEnvironment.h"
#include "MetalLights.h"
#include "MetalMaterials.h"
#include "sampling_math.h"

#include "ShaderTypes.h"

#include <Metal/Metal.hpp>
#include <settings.h>
#include <strelka/scene/scene.h>

#include <cstdint>

#include <glm/glm.hpp>


namespace oka::metal
{

class MetalAccelStructure;

static constexpr size_t kFrameUniformSlots = 3;

// Camera / jitter / exposure / SHARC → Uniforms fill. Owns the per-frame uniform
// ring buffers and the three SHARC resources.
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
        MetalLights* lights = nullptr;
        MetalAccelStructure* accel = nullptr;
        MetalEnvironment* environment = nullptr;
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
        bool temporalUpscaling = false;
        bool enableMotionBlur = false;
        bool isMotionBlurVisible = false;
        bool enableCameraMotionBlur = false;
        bool pausedBlurRefine = false;
        bool resetDenoiseHistory = false;
        bool resetRestirHistory = false;
        bool restirEnvironmentHistoryValid = true;
        bool restirMeshHistoryValid = true;
        bool hasPrevFramePose = false;
        uint32_t previousNumEmissiveMeshes = 0;
        float previousMeshLightSelectionPdf = 0.0f;
        float previousEnvSelectionPdf = 0.0f;
        bool noPrevPose = false;
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
    MTL::Buffer* sharcHashBuffer() const
    {
        return mSharcHashBuffer;
    }
    MTL::Buffer* sharcAccumulationBuffer() const
    {
        return mSharcAccumulationBuffer;
    }
    MTL::Buffer* sharcResolvedBuffer() const
    {
        return mSharcResolvedBuffer;
    }
    uint32_t sharcCapacity() const
    {
        return mSharcCapacity;
    }
    uint32_t sharcResourceGeneration() const
    {
        return mSharcResourceGeneration;
    }
    bool sharcResetPending() const
    {
        return mSharcResetPending;
    }
    void markSharcResetComplete()
    {
        mSharcResetPending = false;
    }
    void requestSharcReset()
    {
        if (mSharcCapacity != 0u)
        {
            mSharcResetPending = true;
        }
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
        float bladeRotation = 0.0f;
        float anamorphicRatio = 1.0f;
        float shiftX = 0.0f;
        float shiftY = 0.0f;
        uint32_t maxDepth = 0;
        uint32_t debug = 0;
        float clampIndirect = 0.0f;
        uint32_t restirDIEnabled = 0;
        uint32_t initialCandidateCount = 0;
        uint32_t temporalReuseEnabled = 0;
        uint32_t spatialReuseEnabled = 0;
        uint32_t spatialNeighborCount = 0;
        uint32_t reservoirMaxAge = 0;
        uint32_t restirDebugMode = 0;
        uint32_t restirBiasCorrection = 0;
        uint32_t restirInitialVisibility = 0;
    };

    MTL::Device* mDevice = nullptr;
    MTL::Buffer* mUniformBuffers[kFrameUniformSlots] = {};
    MTL::Buffer* mUniformTMBuffers[kFrameUniformSlots] = {};
    MTL::Buffer* mSharcHashBuffer = nullptr;
    MTL::Buffer* mSharcAccumulationBuffer = nullptr;
    MTL::Buffer* mSharcResolvedBuffer = nullptr;
    uint32_t mSharcCapacity = 0;
    // Changes whenever any SHARC allocation is replaced. Metal 4 residency is
    // queue-wide, so a runtime UI toggle must publish the new buffers even when
    // the wavefront resolution (and therefore its own generation) is unchanged.
    uint32_t mSharcResourceGeneration = 0;
    uint32_t mSharcFlags = 0;
    float mSharcBaseSize = -1.0f;
    int32_t mSharcLevelBias = 0;
    bool mSharcResetPending = false;
    PrevSettings mPrevSettings;
};

} // namespace oka::metal
