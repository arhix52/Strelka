#pragma once

#include "Metal4Context.h"
#include "MetalGeometry.h"

#include <Metal/Metal.hpp>
#include <env.h>
#include <strelka/scene/scene.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <glm/glm.hpp>

namespace oka
{
namespace metal
{

// Joint matrices, skin PSO, skinned VB writes. Does not own the vertex buffer
// (Geometry) and does not rebuild acceleration structures (Accel refit).
class MetalSkinning
{
public:
    MetalSkinning() = default;
    ~MetalSkinning();

    void init(MTL::Device* device, MTL::CommandQueue* queue, Metal4Context* metal4, MetalGeometry* geometry);
    void release();

    void setScene(Scene* scene)
    {
        mScene = scene;
    }

    void buildPipeline();
    void createSkinDataBuffer();
    void allocJointMatrices();
    void apply();
    void copyVertexBufferToPrev();

    MTL::Buffer* skinDataBuffer() const
    {
        return mSkinDataBuffer;
    }
    MTL::Buffer* jointMatricesBuffer() const
    {
        return mJointMatricesBuffer;
    }

private:
    void applyMetal3();

    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    Metal4Context* mMetal4 = nullptr;
    MetalGeometry* mGeometry = nullptr;
    Scene* mScene = nullptr;

    MTL::ComputePipelineState* mSkinningPSO = nullptr;
    MTL::ComputePipelineState* mTriangleUpdatePSO = nullptr;
    MTL::ComputePipelineState* mSkinningPSO4 = nullptr;
    MTL::ComputePipelineState* mTriangleUpdatePSO4 = nullptr;

    MTL::Buffer* mSkinDataBuffer = nullptr;
    MTL::Buffer* mJointMatricesBuffer = nullptr;
    std::vector<uint32_t> mJointMatOffsets;
    std::vector<glm::mat4> mJointMatScratch;

    bool mLoggedSkinningPipelineGap = false;
    /// Opt back into the Metal 4 skinning submission, which is still wrong.
    const bool mSkinMetal4 = envFlag("STRELKA_SKIN_METAL4");
};

} // namespace metal
} // namespace oka
