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

void MetalSkinning::init(MTL::Device* device, MTL::CommandQueue* queue, Metal4Context* metal4, MetalGeometry* geometry)
{
    mDevice = device;
    mCommandQueue = queue;
    mMetal4 = metal4;
    mGeometry = geometry;
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
    safeRelease(mSkinningPSO);
    safeRelease(mTriangleUpdatePSO);
    safeRelease(mSkinningPSO4);
    safeRelease(mTriangleUpdatePSO4);
    mJointMatOffsets.clear();
    mJointMatScratch.clear();
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
    NS::Error* pError = nullptr;

    MTL::Function* pSkinningFn =
        pLibrary->newFunction(NS::String::string("skinningKernel", NS::UTF8StringEncoding));
    mSkinningPSO = mDevice->newComputePipelineState(pSkinningFn, &pError);
    if (!mSkinningPSO)
    {
        STRELKA_FATAL("Failed to create skinning PSO: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    pSkinningFn->release();

    MTL::Function* pTriUpdateFn =
        pLibrary->newFunction(NS::String::string("updateTriangleBufferKernel", NS::UTF8StringEncoding));
    mTriangleUpdatePSO = mDevice->newComputePipelineState(pTriUpdateFn, &pError);
    if (mMetal4 && mMetal4->isValid())
    {
        mSkinningPSO4 = mMetal4->newComputePipelineState(pLibrary, "skinningKernel", nullptr);
        mTriangleUpdatePSO4 = mMetal4->newComputePipelineState(pLibrary, "updateTriangleBufferKernel", nullptr);
    }
    if (!mTriangleUpdatePSO)
    {
        STRELKA_FATAL("Failed to create triangle update PSO: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    pTriUpdateFn->release();

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

    if (jointMatSize > 0)
    {
        mJointMatricesBuffer =
            mDevice->newBuffer(jointMatSize * sizeof(simd::float4x4), MTL::ResourceStorageModeShared);
    }
}

// The same two dispatches on the Metal 3 queue. Kept as a comparison path: the
// skinned vertices come out as exact zeros through the Metal 4 route, and the
// only way to tell a bad kernel from a bad submission is to run the same kernel
// through the other one.
void MetalSkinning::applyMetal3()
{
    if (!mSkinningPSO || !mTriangleUpdatePSO)
    {
        return;
    }
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    int skinIndex = 0;
    uint32_t jointMatOffset = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin == -1 || node.type != oka::Scene::Node::NodeType::mesh)
        {
            continue;
        }
        if (skinIndex > 0)
        {
            jointMatOffset += mJointMatOffsets[static_cast<size_t>(skinIndex - 1)];
        }
        skinIndex++;

        for (const auto instId : node.instanceIds)
        {
            auto& mesh = mScene->mMeshes[mScene->mInstances[instId].mMeshId];
            const uint32_t meshId = mScene->mInstances[instId].mMeshId;

            SkinningParams skinParams = {};
            skinParams.vbOffset = mesh.mVbOffset;
            skinParams.sbOffset = mesh.mSbOffset;
            skinParams.jointMatOffset = jointMatOffset;
            skinParams.vertexCount = mesh.mVertexCount;

            enc->setComputePipelineState(mSkinningPSO);
            enc->setBuffer(mGeometry->vertexBuffer(), 0, 0);
            enc->setBuffer(mSkinDataBuffer, 0, 1);
            enc->setBuffer(mJointMatricesBuffer, 0, 2);
            enc->setBytes(&skinParams, sizeof(skinParams), 3);
            enc->dispatchThreads(MTL::Size(mesh.mVertexCount, 1, 1), MTL::Size(256, 1, 1));

            MetalGeometry::Mesh* metalMesh = mGeometry->meshes()[meshId];
            if (metalMesh->mPerPrimitiveBuffer)
            {
                TriangleUpdateParams triParams = {};
                triParams.triangleCount = metalMesh->mTriangleCount;
                triParams.indexOffset = mesh.mIndex;
                triParams.vbOffset = mesh.mVbOffset;

                enc->setComputePipelineState(mTriangleUpdatePSO);
                enc->setBuffer(metalMesh->mPerPrimitiveBuffer, 0, 0);
                enc->setBuffer(mGeometry->vertexBuffer(), 0, 1);
                enc->setBuffer(mGeometry->indexBuffer(), 0, 2);
                enc->setBytes(&triParams, sizeof(triParams), 3);
                enc->dispatchThreads(MTL::Size(metalMesh->mTriangleCount, 1, 1), MTL::Size(256, 1, 1));
            }
        }
    }
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
}

void MetalSkinning::apply()
{
    // Guard the pipelines of the path actually taken. The original checked the
    // Metal 3 pipeline while every dispatch used the Metal 4 one, so a failed
    // Metal 4 build passed the check and then bound a null pipeline.
    const bool haveMetal3 = mSkinningPSO != nullptr && mTriangleUpdatePSO != nullptr;
    const bool haveMetal4 = mSkinningPSO4 != nullptr && mTriangleUpdatePSO4 != nullptr;
    const bool haveNeeded = mSkinMetal4 ? haveMetal4 : haveMetal3;
    if (!haveNeeded || !mSkinDataBuffer || !mJointMatricesBuffer)
    {
        if (!mLoggedSkinningPipelineGap)
        {
            mLoggedSkinningPipelineGap = true;
            STRELKA_ERROR("Skinning disabled: metal3Pipelines={} metal4Pipelines={} skinData={} jointMats={}",
                          haveMetal3, haveMetal4, mSkinDataBuffer != nullptr,
                          mJointMatricesBuffer != nullptr);
        }
        return;
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
    const size_t uploadBytes = std::min(mJointMatScratch.size() * sizeof(glm::mat4),
                                        (size_t)mJointMatricesBuffer->length());
    if (uploadBytes == 0)
        return;
    memcpy(mJointMatricesBuffer->contents(), mJointMatScratch.data(), uploadBytes);

    if (!mSkinMetal4)
    {
        applyMetal3();
        return;
    }

    // The zeros this path used to produce were a missing residency declaration,
    // not the shared argument table: the buffers below were only ever made
    // resident as a side effect of MetalRender encoding a frame through Metal 4,
    // so with denoising on -- which pins every frame to Metal 3 -- the skinning
    // dispatch read and wrote memory the queue did not hold, and the character
    // collapsed to a point. Measured on BrainStem: extent 1.469 with the frame
    // on Metal 4, 0.000 with it on Metal 3, from the same kernel and data.
    ensureMetal4Residency();

    // Dispatch skinning + triangle update kernels
    MTL4::CommandBuffer* pCmd = mMetal4->beginImmediate();
    MTL4::ComputeCommandEncoder* pEncoder = pCmd->computeCommandEncoder();
    MTL4::ArgumentTable* skinTable = mMetal4->argumentTable();
    pEncoder->setArgumentTable(skinTable);

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
                uint32_t meshId = mScene->mInstances[instId].mMeshId;

                // Dispatch skinning kernel
                SkinningParams skinParams = {};
                skinParams.vbOffset = mesh.mVbOffset;
                skinParams.sbOffset = mesh.mSbOffset;
                skinParams.jointMatOffset = jointMatOffset;
                skinParams.vertexCount = mesh.mVertexCount;

                pEncoder->setComputePipelineState(mSkinningPSO4);
                skinTable->setAddress(mGeometry->vertexBuffer()->gpuAddress(), 0);
                skinTable->setAddress(mSkinDataBuffer->gpuAddress(), 1);
                skinTable->setAddress(mJointMatricesBuffer->gpuAddress(), 2);
                skinTable->setAddress(mMetal4->immediateConstants().push(skinParams), 3);

                const uint32_t threadsPerGroup = 256;
                const MTL::Size groupSize = MTL::Size(threadsPerGroup, 1, 1);
                pEncoder->dispatchThreadgroups(
                    MTL::Size((mesh.mVertexCount + threadsPerGroup - 1) / threadsPerGroup, 1, 1), groupSize);

                // Dispatch triangle update kernel
                MetalGeometry::Mesh* metalMesh = mGeometry->meshes()[meshId];
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
                    skinTable->setAddress(mMetal4->immediateConstants().push(triParams), 3);

                    pEncoder->dispatchThreadgroups(
                        MTL::Size((metalMesh->mTriangleCount + 255) / 256, 1, 1), groupSize);
                }
            }
        }
    }

    pEncoder->endEncoding();
    mMetal4->submitAndWait(pCmd);
}

void MetalSkinning::ensureMetal4Residency()
{
    if (!mMetal4 || !mMetal4->isValid() || !mGeometry)
    {
        return;
    }
    const std::vector<MetalGeometry::Mesh*>& meshes = mGeometry->meshes();
    if (mGeometry->vertexBuffer() == mResidentVertexBuffer &&
        mGeometry->prevVertexBuffer() == mResidentPrevVertexBuffer &&
        mGeometry->indexBuffer() == mResidentIndexBuffer && mSkinDataBuffer == mResidentSkinDataBuffer &&
        mJointMatricesBuffer == mResidentJointMatricesBuffer && meshes.size() == mResidentMeshCount)
    {
        return;
    }
    mResidentVertexBuffer = mGeometry->vertexBuffer();
    mResidentPrevVertexBuffer = mGeometry->prevVertexBuffer();
    mResidentIndexBuffer = mGeometry->indexBuffer();
    mResidentSkinDataBuffer = mSkinDataBuffer;
    mResidentJointMatricesBuffer = mJointMatricesBuffer;
    mResidentMeshCount = meshes.size();

    mMetal4->addResident(mResidentVertexBuffer);
    mMetal4->addResident(mResidentPrevVertexBuffer);
    mMetal4->addResident(mResidentIndexBuffer);
    mMetal4->addResident(mResidentSkinDataBuffer);
    mMetal4->addResident(mResidentJointMatricesBuffer);
    // Written by the triangle update kernel, which runs in the same encoder.
    for (MetalGeometry::Mesh* mesh : meshes)
    {
        if (mesh)
        {
            mMetal4->addResident(mesh->mPerPrimitiveBuffer);
        }
    }
    mMetal4->commitResidency();
}

void MetalSkinning::copyVertexBufferToPrev()
{
    const size_t vertexDataSize = mGeometry->vertexBuffer()->length();
    if (mMetal4 && mMetal4->isValid())
    {
        ensureMetal4Residency();
        MTL4::CommandBuffer* cmd = mMetal4->beginImmediate();
        MTL4::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->copyFromBuffer(mGeometry->vertexBuffer(), 0, mGeometry->prevVertexBuffer(), 0, vertexDataSize);
        enc->endEncoding();
        // TODO: drop submitAndWait here (and in apply()). It drains the whole
        // Metal 4 queue every animated frame and kills pipelining. Encode the
        // copy / skinning ahead of the frame on the same queue instead, then
        // remove the SharedEvent cross-queue wait that only exists because
        // structure builds still live on Metal 3.
        mMetal4->submitAndWait(cmd);
        return;
    }
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* enc = cmd->blitCommandEncoder();
    enc->copyFromBuffer(mGeometry->vertexBuffer(), 0, mGeometry->prevVertexBuffer(), 0, vertexDataSize);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}


} // namespace metal
} // namespace oka
