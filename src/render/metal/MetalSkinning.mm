#include "MetalSkinning.h"

#include "ShaderTypes.h"

#include <log.h>
#include <paths.h>

#include <algorithm>
#include <cstring>
#include <string>

#include <simd/simd.h>

namespace oka
{
namespace metal
{

MetalSkinning::~MetalSkinning()
{
    release();
}

void MetalSkinning::init(MTL::Device* device,
                         Metal4Context* metal4,
                         MetalGeometry* geometry,
                         uint32_t frameCount)
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
    safeRelease(mTriangleUpdatePSO4);
    mJointMatOffsets.clear();
    mJointMatScratch.clear();
    mJointMatricesPerPose = 0;
    mFrameCount = 0;
    mLoggedSkinningPipelineGap = false;
}

void MetalSkinning::buildPipeline()
{
    const std::string path = oka::resolveResourcePath("metal/shaders/skinning.metallib");
    NS::Error* loadErr = nullptr;
    MTL::Library* pLibrary =
        mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &loadErr);
    if (!pLibrary)
    {
        STRELKA_FATAL("Failed to load {}: {}", path,
                      loadErr ? loadErr->localizedDescription()->utf8String() : "unknown error");
        return;
    }
    if (!mMetal4 || !mMetal4->isValid())
    {
        STRELKA_FATAL("Metal 4 is required for skinning");
        pLibrary->release();
        return;
    }
    mSkinningPSO4 = mMetal4->newComputePipelineState(pLibrary, "skinningKernel", nullptr);
    mTriangleUpdatePSO4 = mMetal4->newComputePipelineState(pLibrary, "updateTriangleBufferKernel", nullptr);
    if (!mSkinningPSO4 || !mTriangleUpdatePSO4)
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
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            // Only the joint *count* matters here; evaluating the matrices was
            // wasted work at init time.
            const size_t jointCount = mScene->mSkines[node.skin].joints.size();
            jointMatSize += jointCount;
            mJointMatOffsets.push_back((uint32_t)jointCount);
        }
    }

    mJointMatricesPerPose = jointMatSize;
    if (jointMatSize > 0 && mFrameCount > 0)
    {
        // Two pose slots preserve both shutter endpoints. Frame-local slices
        // keep an in-flight frame's matrices immutable until its allocator is
        // recycled by Metal4Context::beginFrame().
        const size_t poseCount = static_cast<size_t>(mFrameCount) * 2;
        mJointMatricesBuffer = mDevice->newBuffer(
            jointMatSize * poseCount * sizeof(simd::float4x4), MTL::ResourceStorageModeShared);
    }
}

bool MetalSkinning::uploadJointMatrices(uint32_t frameIndex, uint32_t poseIndex)
{
    if (!mSkinDataBuffer || !mJointMatricesBuffer || mJointMatricesPerPose == 0 ||
        mFrameCount == 0 || poseIndex >= 2)
    {
        return false;
    }

    if (!mSkinningPSO4 || !mTriangleUpdatePSO4)
    {
        if (!mLoggedSkinningPipelineGap)
        {
            mLoggedSkinningPipelineGap = true;
            STRELKA_ERROR("Skinning disabled: Metal 4 pipelines are unavailable");
        }
        return false;
    }

    // Compute joint matrices on CPU. mJointMatScratch is a member so the two
    // skinning passes per frame (t_open / t_close for motion blur) reuse the same
    // allocation instead of churning two vectors each.
    mJointMatScratch.clear();
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            mScene->computeJointMatrices(&mJointMatScratch, static_cast<int>(jointCount), node.skin);
        }
    }

    // glm::mat4 and simd::float4x4 are both 4 column-major float4s with identical
    // layout, so the element-by-element conversion loop (and its temporary
    // vector) was pure overhead — copy straight into the GPU buffer.
    static_assert(sizeof(glm::mat4) == sizeof(simd::float4x4), "matrix layout mismatch");
    const size_t uploadBytes =
        std::min(mJointMatScratch.size(), mJointMatricesPerPose) * sizeof(glm::mat4);
    if (uploadBytes == 0)
    {
        return false;
    }
    const size_t slot = static_cast<size_t>(frameIndex % mFrameCount) * 2 + poseIndex;
    const size_t byteOffset = slot * mJointMatricesPerPose * sizeof(simd::float4x4);
    memcpy(static_cast<uint8_t*>(mJointMatricesBuffer->contents()) + byteOffset,
           mJointMatScratch.data(), uploadBytes);
    return true;
}

void MetalSkinning::encode(MTL4::ComputeCommandEncoder* pEncoder,
                           ConstantRing& constants,
                           uint32_t frameIndex,
                           uint32_t poseIndex)
{
    if (!pEncoder || !mSkinningPSO4 || !mTriangleUpdatePSO4 ||
        !mSkinDataBuffer || !mJointMatricesBuffer || mFrameCount == 0 || poseIndex >= 2)
    {
        return;
    }
    MTL4::ArgumentTable* skinTable = mMetal4->argumentTable();
    pEncoder->setArgumentTable(skinTable);
    const size_t slot = static_cast<size_t>(frameIndex % mFrameCount) * 2 + poseIndex;
    const MTL::GPUAddress jointAddress =
        mJointMatricesBuffer->gpuAddress() + slot * mJointMatricesPerPose * sizeof(simd::float4x4);

    int skinIndex = 0;
    uint32_t jointMatOffset = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            if (skinIndex > 0)
            {
                jointMatOffset += mJointMatOffsets[static_cast<size_t>(skinIndex - 1)];
            }
            skinIndex++;

            for (const auto instId : node.instanceIds)
            {
                auto& mesh = mScene->mMeshes[mScene->mInstances[instId].mMeshId];
                const uint32_t meshId = mScene->mInstances[instId].mMeshId;

                // Dispatch skinning kernel
                SkinningParams skinParams = {};
                skinParams.vbOffset = mesh.mVbOffset;
                skinParams.sbOffset = mesh.mSbOffset;
                skinParams.jointMatOffset = jointMatOffset;
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

                // Dispatch triangle update kernel. The per-mesh records are
                // created in the structures stage, and frames are published from
                // the environment stage onwards -- so a frame that poses a
                // streaming scene can arrive before they exist, and indexing an
                // empty vector here is a read through null.
                if (meshId >= mGeometry->meshes().size())
                {
                    continue;
                }
                const MetalGeometry::Mesh* metalMesh = mGeometry->meshes()[meshId];
                if (metalMesh->mPerPrimitiveBuffer)
                {
                    TriangleUpdateParams triParams = {};
                    triParams.triangleCount = metalMesh->mTriangleCount;
                    triParams.indexOffset = mesh.mIndex;
                    triParams.vbOffset = mesh.mVbOffset;

                    // Skinning writes the vertices this reads.
                    pEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                                        MTL4::VisibilityOptionDevice);
                    pEncoder->setComputePipelineState(mTriangleUpdatePSO4);
                    skinTable->setAddress(metalMesh->mPerPrimitiveBuffer->gpuAddress(), 0);
                    skinTable->setAddress(mGeometry->vertexBuffer()->gpuAddress(), 1);
                    skinTable->setAddress(mGeometry->indexBuffer()->gpuAddress(), 2);
                    skinTable->setAddress(constants.push(triParams), 3);

                    pEncoder->dispatchThreadgroups(
                        MTL::Size((metalMesh->mTriangleCount + 255) / 256, 1, 1), groupSize);
                }
            }
        }
    }

}

void MetalSkinning::encodeCopyVertexBufferToPrev(MTL4::ComputeCommandEncoder* encoder)
{
    if (!encoder || !mGeometry || !mGeometry->vertexBuffer() || !mGeometry->prevVertexBuffer())
    {
        return;
    }
    encoder->copyFromBuffer(mGeometry->vertexBuffer(), 0, mGeometry->prevVertexBuffer(), 0,
                            mGeometry->vertexBuffer()->length());
}


} // namespace metal
} // namespace oka
