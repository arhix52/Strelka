#include "MetalSkinning.h"

#include "ShaderTypes.h"

#include <log.h>
#include <paths.h>

#include <algorithm>
#include <cstring>
#include <string>

#include <simd/simd.h>


namespace oka::metal
{

MetalSkinning::~MetalSkinning()
{
    release();
}

void MetalSkinning::init(MTL::Device* device, Metal4Context* metal4, MetalGeometry* geometry, uint32_t frameCount)
{
    mDevice = device;
    mMetal4 = metal4;
    mGeometry = geometry;
    mFrameCount = frameCount;
}

void MetalSkinning::release()
{
    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    safeRelease(mSkinDataBuffer);
    safeRelease(mJointMatricesBuffer);
    safeRelease(mSkinningPSO4);
    mJointMatOffsets.clear();
    mSkinNodes.clear();
    mSkinPrototypeInstances.clear();
    mSkinNodeToSlot.clear();
    mDirtySkinSlots = {};
    mDirtyMeshIds.clear();
    mJointMatScratch.clear();
    mJointMatricesPerPose = 0;
    mFrameCount = 0;
    mLoggedSkinningPipelineGap = false;
}

void MetalSkinning::buildPipeline()
{
    const std::string path = oka::resolveResourcePath("metal/shaders/skinning.metallib");
    NS::Error* loadErr = nullptr;
    MTL::Library* pLibrary = mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &loadErr);
    if (!pLibrary)
    {
        STRELKA_FATAL(
            "Failed to load {}: {}", path, loadErr ? loadErr->localizedDescription()->utf8String() : "unknown error");
        return;
    }
    if (!mMetal4 || !mMetal4->isValid())
    {
        STRELKA_FATAL("Metal 4 is required for skinning");
        pLibrary->release();
        return;
    }
    mSkinningPSO4 = mMetal4->newComputePipelineState(pLibrary, "skinningKernel", nullptr);
    if (!mSkinningPSO4)
    {
        STRELKA_FATAL("Failed to create Metal 4 skinning pipelines");
        assert(false);
    }

    pLibrary->release();
}

void MetalSkinning::createSkinDataBuffer()
{
    const std::vector<Scene::vertexSkinData>& skinData = mScene->getVerticesSkinData();
    if (skinData.empty())
        return;

    const size_t dataSize = skinData.size() * sizeof(Scene::vertexSkinData);
    mSkinDataBuffer = mDevice->newBuffer(dataSize, MTL::ResourceStorageModeShared);
    memcpy(mSkinDataBuffer->contents(), skinData.data(), dataSize);
}

void MetalSkinning::allocJointMatrices()
{
    size_t jointMatSize = 0;
    mJointMatOffsets.clear();
    mSkinNodes.clear();
    mSkinPrototypeInstances.clear();
    mSkinNodeToSlot.assign(mScene->mNodes.size(), -1);
    for (uint32_t nodeId = 0; nodeId < mScene->mNodes.size(); ++nodeId)
    {
        const Scene::Node& node = mScene->mNodes[nodeId];
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            mSkinNodeToSlot[nodeId] = static_cast<int32_t>(mSkinNodes.size());
            mSkinNodes.push_back(nodeId);
            mJointMatOffsets.push_back(static_cast<uint32_t>(jointMatSize));
            std::vector<uint32_t>& prototypes = mSkinPrototypeInstances.emplace_back();
            for (const uint32_t instanceId : node.instanceIds)
            {
                if (instanceId >= mScene->mInstances.size())
                {
                    continue;
                }
                const uint32_t meshId = mScene->mInstances[instanceId].mMeshId;
                const bool alreadyPresent = std::ranges::any_of(
                    prototypes, [&](uint32_t prototypeId) { return mScene->mInstances[prototypeId].mMeshId == meshId; });
                if (!alreadyPresent)
                {
                    prototypes.push_back(instanceId);
                }
            }
            const size_t jointCount = mScene->mSkines[node.skin].joints.size();
            jointMatSize += jointCount;
        }
    }
    for (std::vector<uint8_t>& slots : mDirtySkinSlots)
    {
        slots.assign(mSkinNodes.size(), 0);
    }

    mJointMatricesPerPose = jointMatSize;
    if (jointMatSize > 0 && mFrameCount > 0)
    {
        // Two pose slots preserve both shutter endpoints. Frame-local slices
        // keep an in-flight frame's matrices immutable until its allocator is
        // recycled by Metal4Context::beginFrame().
        const size_t poseCount = static_cast<size_t>(mFrameCount) * 2;
        mJointMatricesBuffer =
            mDevice->newBuffer(jointMatSize * poseCount * sizeof(simd::float4x4), MTL::ResourceStorageModeShared);
    }
}

bool MetalSkinning::uploadJointMatrices(uint32_t frameIndex, uint32_t poseIndex)
{
    if (!mSkinDataBuffer || !mJointMatricesBuffer || mJointMatricesPerPose == 0 || mFrameCount == 0 || poseIndex >= 2)
    {
        return false;
    }

    if (!mSkinningPSO4)
    {
        if (!mLoggedSkinningPipelineGap)
        {
            mLoggedSkinningPipelineGap = true;
            STRELKA_ERROR("Skinning disabled: Metal 4 pipelines are unavailable");
        }
        return false;
    }

    if (poseIndex == 0)
    {
        mDirtyMeshIds.clear();
    }
    std::vector<uint8_t>& dirtySlots = mDirtySkinSlots[poseIndex];
    std::ranges::fill(dirtySlots, 0);
    for (const uint32_t nodeId : mScene->dirtySkinNodes())
    {
        if (nodeId >= mSkinNodeToSlot.size() || mSkinNodeToSlot[nodeId] < 0)
        {
            continue;
        }
        const size_t slot = static_cast<size_t>(mSkinNodeToSlot[nodeId]);
        dirtySlots[slot] = 1;
        for (const uint32_t prototypeId : mSkinPrototypeInstances[slot])
        {
            mDirtyMeshIds.push_back(mScene->mInstances[prototypeId].mMeshId);
        }
    }
    if (std::ranges::none_of(dirtySlots, [](uint8_t dirty) { return dirty != 0; }))
    {
        return false;
    }

    static_assert(sizeof(glm::mat4) == sizeof(simd::float4x4), "matrix layout mismatch");
    const size_t frameSlot = static_cast<size_t>(frameIndex % mFrameCount) * 2 + poseIndex;
    auto* destination = static_cast<uint8_t*>(mJointMatricesBuffer->contents()) +
                        frameSlot * mJointMatricesPerPose * sizeof(simd::float4x4);
    for (size_t slot = 0; slot < mSkinNodes.size(); ++slot)
    {
        if (!dirtySlots[slot])
        {
            continue;
        }
        const Scene::Node& node = mScene->mNodes[mSkinNodes[slot]];
        const size_t jointCount = mScene->mSkines[node.skin].joints.size();
        mJointMatScratch.clear();
        mScene->computeJointMatrices(&mJointMatScratch, jointCount, static_cast<uint32_t>(node.skin));
        std::memcpy(destination + static_cast<size_t>(mJointMatOffsets[slot]) * sizeof(glm::mat4),
                    mJointMatScratch.data(), jointCount * sizeof(glm::mat4));
    }
    return true;
}

void MetalSkinning::encode(MTL4::ComputeCommandEncoder* pEncoder,
                           ConstantRing& constants,
                           uint32_t frameIndex,
                           uint32_t poseIndex)
{
    if (!pEncoder || !mSkinningPSO4 || !mSkinDataBuffer || !mJointMatricesBuffer || mFrameCount == 0 || poseIndex >= 2)
    {
        return;
    }
    MTL4::ArgumentTable* skinTable = mMetal4->argumentTable();
    pEncoder->setArgumentTable(skinTable);
    const size_t slot = static_cast<size_t>(frameIndex % mFrameCount) * 2 + poseIndex;
    const MTL::GPUAddress jointAddress =
        mJointMatricesBuffer->gpuAddress() + slot * mJointMatricesPerPose * sizeof(simd::float4x4);

    for (size_t skinSlot = 0; skinSlot < mSkinNodes.size(); ++skinSlot)
    {
        if (!mDirtySkinSlots[poseIndex][skinSlot])
        {
            continue;
        }
        for (const uint32_t instId : mSkinPrototypeInstances[skinSlot])
        {
            auto& mesh = mScene->mMeshes[mScene->mInstances[instId].mMeshId];

            SkinningParams skinParams = {};
            skinParams.vbOffset = mesh.mVbOffset;
            skinParams.sbOffset = mesh.mSbOffset;
            skinParams.jointMatOffset = mJointMatOffsets[skinSlot];
            skinParams.vertexCount = mesh.mVertexCount;

            pEncoder->setComputePipelineState(mSkinningPSO4);
            skinTable->setAddress(mGeometry->vertexBuffer()->gpuAddress(), 0);
            skinTable->setAddress(mSkinDataBuffer->gpuAddress(), 1);
            skinTable->setAddress(jointAddress, 2);
            skinTable->setAddress(constants.push(skinParams), 3);

            const uint32_t threadsPerGroup = 256;
            const MTL::Size groupSize = MTL::Size(threadsPerGroup, 1, 1);
            pEncoder->dispatchThreadgroups(
                MTL::Size((mesh.mVertexCount + threadsPerGroup - 1) / threadsPerGroup, 1, 1), groupSize);
        }
    }
}

void MetalSkinning::encodeCopyVertexBufferToPrev(MTL4::ComputeCommandEncoder* encoder)
{
    if (!encoder || !mGeometry || !mGeometry->vertexBuffer() || !mGeometry->prevVertexBuffer())
    {
        return;
    }
    encoder->copyFromBuffer(
        mGeometry->vertexBuffer(), 0, mGeometry->prevVertexBuffer(), 0, mGeometry->vertexBuffer()->length());
}


} // namespace oka::metal
