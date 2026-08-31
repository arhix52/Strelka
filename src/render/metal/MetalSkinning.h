#pragma once

#include "Metal4Context.h"
#include "MetalGeometry.h"

#include <Metal/Metal.hpp>
#include <strelka/scene/scene.h>

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>


namespace oka::metal
{

// Joint matrices, skin PSO, skinned VB writes. Does not own the vertex buffer
// (Geometry) and does not rebuild acceleration structures (Accel refit).
class MetalSkinning
{
public:
    MetalSkinning() = default;
    ~MetalSkinning();

    void init(MTL::Device* device, Metal4Context* metal4, MetalGeometry* geometry, uint32_t frameCount);
    void release();

    void setScene(Scene* scene)
    {
        mScene = scene;
    }

    void buildPipeline();
    void createSkinDataBuffer();
    void allocJointMatrices();
    /// Evaluate and upload the current pose into a frame-local slot. Two slots
    /// per frame preserve the shutter-open and shutter-close matrices until the
    /// command buffer consuming both reaches the GPU.
    bool uploadJointMatrices(uint32_t frameIndex, uint32_t poseIndex);
    void encode(MTL4::ComputeCommandEncoder* encoder, ConstantRing& constants, uint32_t frameIndex, uint32_t poseIndex);
    void encodeCopyVertexBufferToPrev(MTL4::ComputeCommandEncoder* encoder);

    MTL::Buffer* skinDataBuffer() const
    {
        return mSkinDataBuffer;
    }
    MTL::Buffer* jointMatricesBuffer() const
    {
        return mJointMatricesBuffer;
    }

private:
    MTL::Device* mDevice = nullptr;
    Metal4Context* mMetal4 = nullptr;
    MetalGeometry* mGeometry = nullptr;
    Scene* mScene = nullptr;

    MTL::ComputePipelineState* mSkinningPSO4 = nullptr;

    MTL::Buffer* mSkinDataBuffer = nullptr;
    MTL::Buffer* mJointMatricesBuffer = nullptr;
    std::vector<uint32_t> mJointMatOffsets;
    std::vector<glm::mat4> mJointMatScratch;
    size_t mJointMatricesPerPose = 0;
    uint32_t mFrameCount = 0;

    bool mLoggedSkinningPipelineGap = false;
};

} // namespace oka::metal

