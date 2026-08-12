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
    /// Name every buffer this class touches from the Metal 4 queue in that
    /// context's residency set. Metal 4 command buffers do not retain the
    /// resources they use, and a dispatch that reads or writes one which is not
    /// resident returns zeros rather than failing. MetalRender builds a set of
    /// its own, but only on frames it encodes through Metal 4 -- denoising pins
    /// those to Metal 3 -- so nothing else covers the buffers written here.
    void ensureMetal4Residency();

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

    // What ensureMetal4Residency() last declared. Committing a residency set is
    // not free, so it is redone when the allocations change and not per frame.
    MTL::Buffer* mResidentVertexBuffer = nullptr;
    MTL::Buffer* mResidentPrevVertexBuffer = nullptr;
    MTL::Buffer* mResidentIndexBuffer = nullptr;
    MTL::Buffer* mResidentSkinDataBuffer = nullptr;
    MTL::Buffer* mResidentJointMatricesBuffer = nullptr;
    size_t mResidentMeshCount = 0;

    bool mLoggedSkinningPipelineGap = false;
    /// Submit skinning through Metal 4 rather than Metal 3. It now measures the
    /// same as Metal 3 on BrainStem -- identical skinned extents in every audit
    /// scenario -- but the two queues still cost a blocking drain per frame, so
    /// which one wins is a question for the queue consolidation, not this flag.
    const bool mSkinMetal4 = envFlag("STRELKA_SKIN_METAL4");
};

} // namespace metal
} // namespace oka
