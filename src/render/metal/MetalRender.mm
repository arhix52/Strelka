#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalRender.h"
#include "MetalBuffer.h"

#include <algorithm>
#include <map>
#include <cassert>
#include <filesystem>
#include <unistd.h>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <log.h>
#include <paths.h>

#include <simd/simd.h>

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h> // IorStack: sized per path in the wavefront side table

using namespace oka;
namespace fs = std::filesystem;

MetalRender::MetalRender(/* args */) = default;

MetalRender::~MetalRender()
{
    @autoreleasepool
    {
        // Drain the GPU: submit a fence and wait for all prior work to finish.
        // Then spin until the async completion handler has set mRenderBusy=false,
        // ensuring no handler is still accessing members when we release resources.
        if (mCommandQueue)
        {
            MTL::CommandBuffer* fence = mCommandQueue->commandBuffer();
            if (fence)
            {
                fence->commit();
                fence->waitUntilCompleted();
            }
        }
        while (mRenderBusy.load(std::memory_order_acquire))
        {
            usleep(100);
        }

        // Delete C++ wrapper objects (MetalBuffer) for async output.
        // MetalBuffer::~MetalBuffer calls release() on its inner MTL::Buffer.
        delete mAsyncOutputBuffers[0];
        mAsyncOutputBuffers[0] = nullptr;
        delete mAsyncOutputBuffers[1];
        mAsyncOutputBuffers[1] = nullptr;

        // Free heap-allocated Mesh structs (but NOT their acceleration structures,
        // which are released below via mPrimitiveAccelerationStructures).
        for (auto* mesh : mMetalMeshes)
        {
            if (mesh->mPerPrimitiveBuffer)
                mesh->mPerPrimitiveBuffer->release();
            delete mesh;
        }
        mMetalMeshes.clear();

        for (Blas& blas : mBlasList)
        {
            if (blas.mScratch)
                blas.mScratch->release();
            if (blas.mDescriptor)
                blas.mDescriptor->release();
        }
        mBlasList.clear();

        // Metal objects: release only those we explicitly created with newXxx().
        // Some objects (e.g. acceleration structures returned by newAccelerationStructure)
        // may share internal references. Use a flat release-and-null pattern to
        // avoid double-release from pointer authentication failures.
        auto safeRelease = [](auto*& p) {
            if (p) { p->release(); p = nullptr; }
        };

        // Acceleration structures
        for (auto*& as : mPrimitiveAccelerationStructures)
            safeRelease(as);
        safeRelease(mInstanceAccelerationStructure);

        // Material textures
        for (auto*& tex : mMaterialTextures)
            safeRelease(tex);

        // Buffers
        safeRelease(mAccumulationBuffer);
        safeRelease(mLightBuffer);
        safeRelease(mVertexBuffer);
        safeRelease(mIndexBuffer);
        safeRelease(mInstanceBuffer);
        safeRelease(mMaterialBuffer);
        safeRelease(mSkinDataBuffer);
        safeRelease(mJointMatricesBuffer);
        safeRelease(mPrevVertexBuffer);
        safeRelease(mGeometryEntryBuffer);
        safeRelease(mTlasScratchBuffer);
        for (auto*& buf : mUniformBuffers) safeRelease(buf);
        for (auto*& buf : mUniformTMBuffers) safeRelease(buf);

        // Environment map
        safeRelease(mEnvMapTexture);
        safeRelease(mEnvAliasBuffer);

        // Pipeline states
        safeRelease(mWavefrontGeneratePSO);
        safeRelease(mWavefrontExtendPSO);
        safeRelease(mWavefrontShadePSO);
        safeRelease(mWavefrontResolvePSO);
        safeRelease(mWavefrontPreparePSO);
        safeRelease(mWavefrontPrepareShadowPSO);
        safeRelease(mWavefrontShadowPSO);
        safeRelease(mPathStateBuffer);
        safeRelease(mHitBuffer);
        safeRelease(mIorStackBuffer);
        safeRelease(mRadianceBuffer);
        safeRelease(mPathQueueBuffer[0]);
        safeRelease(mPathQueueBuffer[1]);
        safeRelease(mWavefrontControlBuffer);
        safeRelease(mShadowRayBuffer);
        safeRelease(mStageTimestampBuffer);
        safeRelease(mStageStatsBuffer);
        safeRelease(mSortBinBuffer);
        safeRelease(mSortTgBaseBuffer);
        safeRelease(mWavefrontSortCountPSO);
        safeRelease(mWavefrontSortScanPSO);
        safeRelease(mWavefrontSortScatterPSO);
        safeRelease(mPathTracingPSO);
        safeRelease(mTonemapperPSO);
        safeRelease(mSkinningPSO);
        safeRelease(mTriangleUpdatePSO);

        // Queue & device (release last)
        safeRelease(mCommandQueue);
        safeRelease(mDevice);
    }
}

void MetalRender::triggerRenderIfIdle()
{
    if (mRenderBusy.load())
        return;

    const uint32_t w = getSettings()->getAs<uint32_t>("render/width");
    const uint32_t h = getSettings()->getAs<uint32_t>("render/height");

    // Pick the buffer that is NOT currently being displayed
    int ri = mReadyIndex.load();
    mWriteIndex = (ri >= 0) ? (1 - ri) : 0;

    // Create or resize the write buffer
    if (!mAsyncOutputBuffers[mWriteIndex])
    {
        BufferDesc desc{};
        desc.format = BufferFormat::FLOAT4;
        desc.width = w;
        desc.height = h;
        mAsyncOutputBuffers[mWriteIndex] = createBuffer(desc);
    }
    else if (mAsyncOutputBuffers[mWriteIndex]->width() != w ||
             mAsyncOutputBuffers[mWriteIndex]->height() != h)
    {
        mAsyncOutputBuffers[mWriteIndex]->resize(w, h);
    }

    // Also ensure the other buffer exists (display may need it)
    int otherIdx = 1 - mWriteIndex;
    if (!mAsyncOutputBuffers[otherIdx])
    {
        BufferDesc desc{};
        desc.format = BufferFormat::FLOAT4;
        desc.width = w;
        desc.height = h;
        mAsyncOutputBuffers[otherIdx] = createBuffer(desc);
    }
    else if (mAsyncOutputBuffers[otherIdx]->width() != w ||
             mAsyncOutputBuffers[otherIdx]->height() != h)
    {
        // Resize ready buffer too — display will pick up new size next frame
        mAsyncOutputBuffers[otherIdx]->resize(w, h);
        mReadyIndex.store(-1); // invalidate since we resized
    }

    mRenderBusy.store(true);
    render(mAsyncOutputBuffers[mWriteIndex]);
}

Buffer* MetalRender::getReadyBuffer()
{
    int ri = mReadyIndex.load();
    if (ri < 0)
        return nullptr;
    return mAsyncOutputBuffers[ri];
}

void MetalRender::init()
{
    static_assert(sizeof(PathState) == 48, "PathState must stay at 48 bytes: it is read and written for every live path on every bounce");
    static_assert(sizeof(HitRecord) == 24, "HitRecord size changed");
    static_assert(sizeof(GeometryEntry) == 16, "GeometryEntry size changed");

    mDevice = MTL::CreateSystemDefaultDevice();
    if (!mDevice)
    {
        STRELKA_FATAL("Failed to create Metal device");
        return;
    }
    mCommandQueue = mDevice->newCommandQueue();
    buildComputePipeline();
    buildTonemapperPipeline();
    buildWavefrontPipelines();
}

MTL::Texture* MetalRender::loadTextureFromFile(const std::string& fileName)
{
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (data == nullptr)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return nullptr;
    }
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(texWidth);
    pTextureDesc->setHeight(texHeight);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);

    MTL::Texture* pTexture = mDevice->newTexture(pTextureDesc);

    const MTL::Region region = MTL::Region::Make3D(0, 0, 0, texWidth, texHeight, 1);
    pTexture->replaceRegion(region, 0, data, 4ull * texWidth);

    pTextureDesc->release();
    return pTexture;
}

void MetalRender::createMetalMaterials()
{
    using simd::float3;
    const std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    std::vector<Material> gpuMaterials;
    const fs::path resourcePath = getSettings()->getAs<std::string>("resource/searchPath");

    auto loadTex = [&](const std::string& path) -> MTL::ResourceID {
        if (path.empty()) return MTL::ResourceID{};
        const fs::path fullPath = resourcePath / path;
        MTL::Texture* tex = loadTextureFromFile(fullPath.string());
        if (tex) mMaterialTextures.push_back(tex);
        return tex ? tex->gpuResourceID() : MTL::ResourceID{};
    };

    for (const Scene::MaterialDescription& currMatDesc : matDescs)
    {
        Material material = {};
        const auto& p = currMatDesc.params;
        material.base_color = packed_float3(simd_make_float3(p.base_color.x, p.base_color.y, p.base_color.z));
        material.metallic = p.metallic;
        material.roughness = p.roughness;
        material.ior = p.ior;
        material.specular = p.specular;
        material.specular_tint = p.specular_tint;
        material.transmission = p.transmission;
        material.clearcoat = p.clearcoat;
        material.clearcoat_roughness = p.clearcoat_roughness;
        material.anisotropy = p.anisotropy;
        material.emission = packed_float3(simd_make_float3(p.emission.x, p.emission.y, p.emission.z));
        material.emission_strength = p.emission_strength;
        material.normal_scale = p.normal_scale;
        material.occlusion_strength = p.occlusion_strength;
        material.alpha_cutoff = p.alpha_cutoff;
        material.material_type = p.material_type;
        material.thin_walled = p.thin_walled;
        material.dielectric_priority = p.dielectric_priority;

        material.baseColorTexture = loadTex(currMatDesc.baseColorTexPath);
        material.metallicRoughnessTexture = loadTex(currMatDesc.metallicRoughnessTexPath);
        material.normalTexture = loadTex(currMatDesc.normalTexPath);
        material.emissionTexture = loadTex(currMatDesc.emissionTexPath);
        material.occlusionTexture = loadTex(currMatDesc.occlusionTexPath);

        gpuMaterials.push_back(material);
    }

    const size_t materialsDataSize = sizeof(Material) * gpuMaterials.size();
    if (materialsDataSize > 0)
    {
        mMaterialBuffer = mDevice->newBuffer(materialsDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mMaterialBuffer->contents(), gpuMaterials.data(), materialsDataSize);
        mMaterialBuffer->didModifyRange(NS::Range::Make(0, mMaterialBuffer->length()));
    }
    else
    {
        mMaterialBuffer = nullptr;
    }
}

uint32_t MetalRender::computeBandHeight(uint32_t height) const
{
    if (height == 0)
    {
        return 1;
    }

    // First frame: no timing yet. Start with a conservative band so a pathological
    // scene cannot lock the UI on the very first submission.
    const double lastMs = mLastRenderTimeMs.load(std::memory_order_relaxed);
    if (lastMs <= 0.0)
    {
        return std::min<uint32_t>(height, 128);
    }

    const double msPerRow = lastMs / static_cast<double>(mLastBandTotalRows > 0 ? mLastBandTotalRows : height);
    if (msPerRow <= 0.0)
    {
        return height;
    }

    const double rows = kTargetSubmissionMs / msPerRow;
    // Round to a multiple of the 8-row threadgroup so bands stay aligned, and cap
    // the band count hard. Every extra band is another command buffer whose
    // bindings and resource residency have to be re-established, so past a
    // handful of bands the split costs more than the responsiveness it buys.
    const uint32_t minRows = 8;
    const uint32_t lowerBound = std::max(minRows, (height + kMaxBands - 1) / kMaxBands);
    const uint32_t clamped =
        static_cast<uint32_t>(std::clamp(rows, static_cast<double>(lowerBound), static_cast<double>(height)));
    return std::max(minRows, (clamped + 7u) & ~7u);
}

// Encode one full wavefront frame: for each sample, generate camera rays then
// alternate extend/shade for maxDepth bounces. Every stage is dispatched over
// all pixels; dead paths return immediately. Compaction replaces that next.
//
// Stages within a single compute encoder run in order with an implicit barrier
// (Metal's default serial dispatch type), which is exactly the dependency
// extend -> shade -> extend needs.

// Stage kinds, in the order the encode loop issues them.
// Two timestamps per stage (encoder start and end), so the counter buffer holds
// 2 * kMaxStageSamples entries.
namespace
{
enum StageKind : uint8_t
{
    kStageGenerate = 0,
    kStagePrepare,
    kStageExtend,
    kStageShade,
    kStagePrepareShadow,
    kStageShadow,
    kStageSort,
    kStageResolve,
    kStageCount
};
const char* const kStageNames[kStageCount] = { "generate", "prepare", "extend",  "shade",
                                               "prepShadow", "shadow", "sort", "resolve" };
} // namespace

// A timestamp counter buffer, if the device can sample at dispatch boundaries.
// Apple Silicon can; the check exists because the API does not promise it.
void MetalRender::createStageTimestampBuffer()
{
    if (mStageTimestampBuffer)
    {
        return;
    }
    // M1/M2 sample only at encoder boundaries, not at dispatch boundaries, which
    // is why profiling mode gives every stage its own encoder rather than
    // stamping around each dispatch.
    if (!mDevice->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary))
    {
        STRELKA_WARNING("stage profiling unavailable: no counter sampling at encoder boundaries");
        return;
    }
    MTL::CounterSet* timestampSet = nullptr;
    NS::Array* sets = mDevice->counterSets();
    for (NS::UInteger i = 0; sets && i < sets->count(); ++i)
    {
        MTL::CounterSet* set = static_cast<MTL::CounterSet*>(sets->object(i));
        if (set->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
        {
            timestampSet = set;
            break;
        }
    }
    if (!timestampSet)
    {
        return;
    }

    MTL::CounterSampleBufferDescriptor* desc = MTL::CounterSampleBufferDescriptor::alloc()->init();
    desc->setCounterSet(timestampSet);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setSampleCount(2 * kMaxStageSamples); // start and end per stage
    NS::Error* err = nullptr;
    mStageTimestampBuffer = mDevice->newCounterSampleBuffer(desc, &err);
    desc->release();
    if (!mStageTimestampBuffer)
    {
        STRELKA_WARNING("stage profiling unavailable: {}",
                        err ? err->localizedDescription()->utf8String() : "unknown error");
    }
}

// Resolve the timestamps and print the per-stage breakdown. GPU timestamps are
// in nanoseconds on Apple Silicon; a sample can come back as MTLCounterErrorValue
// when the GPU dropped it, and those gaps are skipped rather than counted as
// enormous durations.
void MetalRender::reportStageTimings()
{
    if (!mStageTimestampBuffer || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = 2 * mStageKinds.size();
    NS::Data* data = mStageTimestampBuffer->resolveCounterRange(NS::Range::Make(0, n));
    if (!data)
    {
        return;
    }
    const MTL::CounterResultTimestamp* ts = static_cast<const MTL::CounterResultTimestamp*>(data->bytes());

    double totals[kStageCount] = {};
    uint32_t counts[kStageCount] = {};
    // Per-bounce durations of the three traversal-heavy stages. The cost of a
    // bounce says more than the total does: bounce 0 is a coherent primary pass
    // and the later ones are not, which is what decides whether sorting rays is
    // worth anything.
    std::string perBounce[kStageCount];
    for (NS::UInteger i = 0; i < mStageKinds.size(); ++i)
    {
        const MTL::CounterResultTimestamp& a = ts[2 * i];
        const MTL::CounterResultTimestamp& b = ts[2 * i + 1];
        if (a.timestamp == MTL::CounterErrorValue || b.timestamp == MTL::CounterErrorValue ||
            b.timestamp <= a.timestamp)
        {
            continue;
        }
        const uint8_t kind = mStageKinds[i];
        const double ms = (b.timestamp - a.timestamp) / 1e6; // ns -> ms
        totals[kind] += ms;
        ++counts[kind];
        if (kind == kStageExtend || kind == kStageShade || kind == kStageShadow)
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f ", ms);
            perBounce[kind] += buf;
        }
    }

    double sum = 0.0;
    for (uint32_t k = 0; k < kStageCount; ++k)
    {
        sum += totals[k];
    }
    std::string line;
    for (uint32_t k = 0; k < kStageCount; ++k)
    {
        if (counts[k] == 0)
        {
            continue;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "%s %.2fms(%.0f%%, n=%u)  ", kStageNames[k], totals[k],
                 sum > 0.0 ? 100.0 * totals[k] / sum : 0.0, counts[k]);
        line += buf;
    }
    STRELKA_INFO("STAGES total {:.2f}ms  {}", sum, line);
    STRELKA_INFO("STAGES per bounce: extend [{}] shade [{}] shadow [{}]", perBounce[kStageExtend],
                 perBounce[kStageShade], perBounce[kStageShadow]);

    if (mStageStatsBuffer)
    {
        const uint32_t* stats = static_cast<const uint32_t*>(mStageStatsBuffer->contents());
        std::string paths, shadows;
        for (uint32_t i = 0; i < counts[kStageExtend]; ++i)
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.0fk ", stats[16 + i] / 1000.0);
            paths += buf;
            snprintf(buf, sizeof(buf), "%.0fk ", stats[48 + i] / 1000.0);
            shadows += buf;
        }
        STRELKA_INFO("STAGES rays per bounce: paths [{}] shadow [{}]", paths, shadows);
    }
}

MTL::ComputeCommandEncoder* MetalRender::encodeWavefront(MTL::CommandBuffer* pCmd,
                                  MTL::ComputeCommandEncoder* enc, MTL::Buffer* uniformBuffer,
                                  Buffer* output, uint32_t width, uint32_t height,
                                  uint32_t sampleCount)
{
    const uint32_t pixels = width * height;
    const uint32_t maxDepth = std::max(1u, getSettings()->getAs<uint32_t>("render/pt/depth"));
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();

    // Textures are reached through resource IDs inside the Material struct, so
    // Metal cannot infer their use from the bindings and every encoder has to be
    // told about them again.
    auto declareResidency = [&](MTL::ComputeCommandEncoder* e) {
        if (!mMaterialTextures.empty())
        {
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(mMaterialTextures.data()),
                            mMaterialTextures.size(), MTL::ResourceUsageRead);
        }
        if (!mPrimitiveAccelerationStructures.empty())
        {
            e->useResources(reinterpret_cast<const MTL::Resource* const*>(mPrimitiveAccelerationStructures.data()),
                            mPrimitiveAccelerationStructures.size(), MTL::ResourceUsageRead);
        }
        if (mInstanceAccelerationStructure)
        {
            e->useResource(mInstanceAccelerationStructure, MTL::ResourceUsageRead);
        }
        if (mEnvMapTexture)
        {
            e->useResource(mEnvMapTexture, MTL::ResourceUsageRead);
        }
        e->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageWrite);
    };
    declareResidency(enc);

    const MTL::Size grid = MTL::Size(pixels, 1, 1);
    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    // Byte offset of the indirect dispatch arguments inside the control buffer.
    const NS::UInteger kDispatchArgsOffset = 2 * sizeof(uint32_t);
    const NS::UInteger kShadowArgsOffset = 8 * sizeof(uint32_t);
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kSortArgsOffset = 11 * sizeof(uint32_t);
    const bool sortRays = getSettings()->getAs<uint32_t>("render/pt/sortRays") != 0 &&
                          mWavefrontSortScatterPSO != nullptr;
    const MTL::Size sortTg = MTL::Size(kSortThreadgroup, 1, 1);

    // Profiling gives each stage its own encoder, because this hardware samples
    // counters only at encoder boundaries. That costs encoder overhead, so it is
    // a measurement mode and not something to leave on.
    mStageKinds.clear();
    const bool profile = mProfileStages && mStageTimestampBuffer != nullptr;
    auto stamp = [&](uint8_t kind) {
        if (!profile || mStageKinds.size() >= kMaxStageSamples)
        {
            return;
        }
        enc->endEncoding();
        MTL::ComputePassDescriptor* desc = MTL::ComputePassDescriptor::computePassDescriptor();
        MTL::ComputePassSampleBufferAttachmentDescriptor* att =
            desc->sampleBufferAttachments()->object(0);
        att->setSampleBuffer(mStageTimestampBuffer);
        att->setStartOfEncoderSampleIndex(2 * mStageKinds.size());
        att->setEndOfEncoderSampleIndex(2 * mStageKinds.size() + 1);
        enc = pCmd->computeCommandEncoder(desc);
        declareResidency(enc);
        mStageKinds.push_back(kind);
    };

    for (uint32_t s = 0; s < sampleCount; ++s)
    {
        stamp(kStageGenerate);
        enc->setComputePipelineState(mWavefrontGeneratePSO);
        enc->setBuffer(uniformBuffer, 0, 0);
        enc->setBuffer(mPathStateBuffer, 0, 1);
        enc->setBuffer(mRadianceBuffer, 0, 2);
        enc->setBuffer(mIorStackBuffer, 0, 3);
        enc->setBytes(&s, sizeof(uint32_t), 4);
        enc->setBuffer(mPathQueueBuffer[0], 0, 5);
        enc->setBuffer(mWavefrontControlBuffer, 0, 6);
        enc->setBuffer(mSortBinBuffer, 0, 7);
        enc->dispatchThreads(grid, tg);

        for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)
        {
            // With sorting on, the queues take fixed roles -- [0] is what the
            // bounce traverses, [1] is what `shade` appends to and the sort
            // reads -- because the sort already rewrites [0] every bounce.
            const uint32_t src = sortRays ? 0u : (bounce & 1u);
            const uint32_t dst = src ^ 1u;

            // Publish this bounce's live count and clear the destination's, then
            // dispatch both stages indirectly from it. Nothing crosses to the CPU.
            stamp(kStagePrepare);
            enc->setComputePipelineState(mWavefrontPreparePSO);
            enc->setBuffer(mWavefrontControlBuffer, 0, 0);
            enc->setBytes(&src, sizeof(uint32_t), 1);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 2);
            enc->setBytes(&bounce, sizeof(uint32_t), 3);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageExtend);
            enc->setComputePipelineState(mWavefrontExtendPSO);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mInstanceBuffer, 0, 1);
            enc->setAccelerationStructure(mInstanceAccelerationStructure, 2);
            enc->setBuffer(mPathStateBuffer, 0, 3);
            enc->setBuffer(mHitBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            enc->setBuffer(mPathQueueBuffer[src], 0, 6);
            enc->setBuffer(mWavefrontControlBuffer, 0, 7);
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kDispatchArgsOffset, tg);

            stamp(kStageShade);
            enc->setComputePipelineState(mWavefrontShadePSO);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mInstanceBuffer, 0, 1);
            enc->setAccelerationStructure(mInstanceAccelerationStructure, 2);
            enc->setBuffer(mLightBuffer, 0, 3);
            enc->setBuffer(mMaterialBuffer, 0, 4);
            enc->setBuffer(mPathStateBuffer, 0, 5);
            enc->setBuffer(mHitBuffer, 0, 6);
            enc->setBuffer(mRadianceBuffer, 0, 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(mGeometryEntryBuffer, 0, 9);
            enc->setBuffer(mEnvAliasBuffer, 0, 10);
            enc->setBuffer(mVertexBuffer, 0, 11);
            enc->setBuffer(mPrevVertexBuffer, 0, 12);
            enc->setBuffer(mIndexBuffer, 0, 13);
            enc->setBytes(&s, sizeof(uint32_t), 14);
            enc->setBuffer(mPathQueueBuffer[src], 0, 15);
            enc->setBuffer(mPathQueueBuffer[dst], 0, 16);
            enc->setBuffer(mWavefrontControlBuffer, dst * sizeof(uint32_t), 17);
            enc->setBuffer(mWavefrontControlBuffer, 0, 18);
            enc->setBuffer(mShadowRayBuffer, 0, 19);
            enc->setBuffer(mWavefrontControlBuffer, kShadowCounterOffset, 20);
            if (mEnvMapTexture)
            {
                enc->setTexture(mEnvMapTexture, 0);
            }
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kDispatchArgsOffset, tg);

            // Deferred occlusion. It has to run before the next bounce's shade,
            // so that this bounce's direct lighting lands in the accumulator
            // ahead of the next bounce's emission -- the same order the
            // megakernel adds them in.
            stamp(kStagePrepareShadow);
            enc->setComputePipelineState(mWavefrontPrepareShadowPSO);
            enc->setBuffer(mWavefrontControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->setBytes(&bounce, sizeof(uint32_t), 2);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageShadow);
            enc->setComputePipelineState(mWavefrontShadowPSO);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setAccelerationStructure(mInstanceAccelerationStructure, 1);
            enc->setBuffer(mShadowRayBuffer, 0, 2);
            enc->setBuffer(mRadianceBuffer, 0, 3);
            enc->setBuffer(mWavefrontControlBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kShadowArgsOffset, tg);

            if (sortRays)
            {
                stamp(kStageSort);
                enc->setComputePipelineState(mWavefrontSortCountPSO);
                enc->setBuffer(mPathQueueBuffer[dst], 0, 0);
                enc->setBuffer(mPathStateBuffer, 0, 1);
                enc->setBuffer(mWavefrontControlBuffer, 0, 2);
                enc->setBuffer(mSortBinBuffer, 0, 3);
                enc->setBuffer(mSortTgBaseBuffer, 0, 4);
                enc->setThreadgroupMemoryLength(kSortBins * sizeof(uint32_t), 0);
                enc->dispatchThreadgroups(mWavefrontControlBuffer, kSortArgsOffset, sortTg);

                enc->setComputePipelineState(mWavefrontSortScanPSO);
                enc->setBuffer(mSortBinBuffer, 0, 0);
                enc->setBuffer(mSortBinBuffer, kSortBins * sizeof(uint32_t), 1);
                enc->setBuffer(mWavefrontControlBuffer, 0, 2);
                enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

                enc->setComputePipelineState(mWavefrontSortScatterPSO);
                enc->setBuffer(mPathQueueBuffer[dst], 0, 0);
                enc->setBuffer(mPathStateBuffer, 0, 1);
                enc->setBuffer(mWavefrontControlBuffer, 0, 2);
                enc->setBuffer(mSortBinBuffer, kSortBins * sizeof(uint32_t), 3);
                enc->setBuffer(mSortTgBaseBuffer, 0, 4);
                enc->setBuffer(mPathQueueBuffer[src], 0, 5);
                enc->setThreadgroupMemoryLength(kSortBins * sizeof(uint32_t), 0);
                enc->dispatchThreadgroups(mWavefrontControlBuffer, kSortArgsOffset, sortTg);
            }
        }
    }

    // Fold the accumulated radiance into the output exactly as the megakernel's
    // tail does, reusing the resolve kernel in wavefront.metal.
    stamp(kStageResolve);
    enc->setComputePipelineState(mWavefrontResolvePSO);
    enc->setBuffer(uniformBuffer, 0, 0);
    enc->setBuffer(mRadianceBuffer, 0, 1);
    enc->setBuffer(outputBuffer, 0, 2);
    enc->setBuffer(mAccumulationBuffer, 0, 3);
    enc->setBytes(&sampleCount, sizeof(uint32_t), 4);
    enc->dispatchThreads(grid, tg);

    if (profile && mStageStatsBuffer)
    {
        enc->endEncoding();
        MTL::BlitCommandEncoder* blit = pCmd->blitCommandEncoder();
        blit->copyFromBuffer(mWavefrontControlBuffer, 0, mStageStatsBuffer, 0, mStageStatsBuffer->length());
        blit->endEncoding();
        enc = pCmd->computeCommandEncoder();
        declareResidency(enc);
    }
    return enc;
}

void MetalRender::encodePathTraceBindings(MTL::ComputeCommandEncoder* enc, MTL::Buffer* uniformBuffer, Buffer* output)
{
    // Residency declarations. Textures are reached through resource IDs stored in
    // the Material struct, so Metal cannot infer their use from the bindings —
    // they must be declared explicitly. useResources() batches the whole array
    // into one call instead of one call per texture per band.
    if (!mMaterialTextures.empty())
    {
        enc->useResources(reinterpret_cast<const MTL::Resource* const*>(mMaterialTextures.data()),
                          mMaterialTextures.size(), MTL::ResourceUsageRead);
    }
    if (!mPrimitiveAccelerationStructures.empty())
    {
        enc->useResources(reinterpret_cast<const MTL::Resource* const*>(mPrimitiveAccelerationStructures.data()),
                          mPrimitiveAccelerationStructures.size(), MTL::ResourceUsageRead);
    }
    if (mInstanceAccelerationStructure)
    {
        enc->useResource(mInstanceAccelerationStructure, MTL::ResourceUsageRead);
    }

    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();

    enc->setComputePipelineState(mPathTracingPSO);
    enc->setBuffer(uniformBuffer, 0, 0);
    enc->setBuffer(mInstanceBuffer, 0, 1);
    enc->setAccelerationStructure(mInstanceAccelerationStructure, 2);
    enc->setBuffer(mLightBuffer, 0, 3);
    enc->setBuffer(mMaterialBuffer, 0, 4);
    // Output
    enc->setBuffer(outputBuffer, 0, 5);
    enc->setBuffer(mAccumulationBuffer, 0, 6);
    // Motion blur buffers
    enc->setBuffer(mPrevVertexBuffer, 0, 7);
    enc->setBuffer(mIndexBuffer, 0, 8);
    enc->setBuffer(mGeometryEntryBuffer, 0, 9);
    // Environment map
    enc->setBuffer(mEnvAliasBuffer, 0, 10);
    if (mEnvMapTexture)
    {
        enc->useResource(mEnvMapTexture, MTL::ResourceUsageRead);
        enc->setTexture(mEnvMapTexture, 0);
    }
}

void MetalRender::render(Buffer* output)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    SharedContext& ctx = getSharedContext();

    if (ctx.mFrameNumber == 0)
    {
        buildBuffers();
        createMetalMaterials();
        createAccelerationStructures();
        // create accum buffer, we don't need cpu access, make it device only
        mAccumulationBuffer = mDevice->newBuffer(
            output->width() * output->height() * output->getElementSize(), MTL::ResourceStorageModePrivate);

        // Initialize skinning pipeline if scene has skeletal data
        if (!mScene->getVerticesSkinData().empty())
        {
            buildSkinningPipeline();
            createSkinDataBuffer();
            allocJointMatrices();
        }

        // Load environment map if specified
        const auto& envLight = mScene->getEnvLight();
        if (envLight.has_value() && !envLight->texturePath.empty())
        {
            const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
            const fs::path envTexPath = fs::path(resourcePathStr) / envLight->texturePath;
            loadEnvMap(envTexPath.string());
        }
    }

    mFrameIndex = (mFrameIndex + 1) % kMaxFramesInFlight;

    // Recreate accumulation buffer if output size changed
    const uint32_t width = output->width();
    const uint32_t height = output->height();
    const size_t requiredSize = width * height * output->getElementSize();
    if (mAccumulationBuffer && requiredSize != mAccumulationBuffer->length())
    {
        mAccumulationBuffer->release();
        mAccumulationBuffer = mDevice->newBuffer(requiredSize, MTL::ResourceStorageModePrivate);
        ctx.mSubframeIndex = 0;
    }

    // Update motion blur enable state from settings each frame
    mEnableMotionBlur = getSettings()->getAs<bool>("render/enableMotionBlur");

    bool motionBlurCameraSet = false; // track if animation block sets prev camera

    // Animation detection: two-pass at t_open / t_close for motion blur (CHANGED-only)
    {
        SettingsManager& animSettings = *getSettings();
        std::vector<oka::Scene::Animation>& animations = mScene->getAnimations();

        // Collect target times and detect which animations actually changed
        constexpr float EPSILON = 1e-6f;
        bool animStateChanged = false;
        float maxTimeDelta = 0.0f; // track largest time jump for scrub detection
        const size_t animCount = animations.size();
        mAnimTargetTimes.resize(animCount);
        mAnimChanged.resize(animCount);
        std::fill(mAnimChanged.begin(), mAnimChanged.end(), false);
        for (int i = 0; i < (int)animCount; ++i)
        {
            char key[64];
            snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
            mAnimTargetTimes[i] = animSettings.getAs<float>(key);
            const float delta = std::abs(animations[i].current - mAnimTargetTimes[i]);
            if (delta > EPSILON)
            {
                mAnimChanged[i] = true;
                animStateChanged = true;
                maxTimeDelta = std::max(maxTimeDelta, delta);
            }
        }

        if (animStateChanged)
        {
            const float shutterDuration = mEnableMotionBlur
                ? animSettings.getAs<float>("render/motionBlur/shutterTime") : 0.0f;
            const uint32_t shutterMode = animSettings.getAs<uint32_t>("render/motionBlur/shutterMode");

            if (mEnableMotionBlur && mPrevVertexBuffer && shutterDuration > 0.0f)
            {
                // Shutter offset: Centered straddles t_anim, Leading closes at t_anim,
                // Trailing opens at t_anim
                float shutterOffset = 0.0f;
                switch (shutterMode)
                {
                case 0: shutterOffset = -shutterDuration * 0.5f; break; // Centered
                case 1: shutterOffset = -shutterDuration;        break; // Leading
                case 2: shutterOffset = 0.0f;                    break; // Trailing
                }

                // --- Pass 1: evaluate CHANGED animations at t_open ---
                bool pass1Skeletal = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    float tOpen = mAnimTargetTimes[i] + shutterOffset;
                    tOpen = std::clamp(tOpen, animations[i].start, animations[i].end);
                    animations[i].current = tOpen;
                    pass1Skeletal |= mScene->applyAnimation(i);
                }

                // Capture camera state at t_open for camera motion blur
                const uint32_t selCam = animSettings.getAs<uint32_t>("render/selectedCamera");
                oka::Camera& prevCam = mScene->getCamera(selCam);
                prevCam.updateAspectRatio(width / (float)height);
                prevCam.updateViewMatrix();
                mPrevMotionBlurView.mCamMatrices = prevCam.matrices;
                motionBlurCameraSet = true;

                if (pass1Skeletal)
                {
                    applySkinning();
                }
                // Always copy current VB to prevVB — ensures prev state is consistent
                // even when only camera (not skeleton) animation changed
                copyVertexBufferToPrev();

                // --- Pass 2: evaluate CHANGED animations at t_close ---
                bool pass2Skeletal = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    float tClose = mAnimTargetTimes[i] + shutterOffset + shutterDuration;
                    tClose = std::clamp(tClose, animations[i].start, animations[i].end);
                    animations[i].current = tClose;
                    pass2Skeletal |= mScene->applyAnimation(i);
                }

                if (pass2Skeletal)
                {
                    applySkinning();
                    updateSkeletalBLAS(maxTimeDelta > shutterDuration * 2.0f);
                }
                rebuildTLAS(); // already re-uploads the instance transforms

                // Restore target times so next-frame EPSILON check is stable
                for (int i = 0; i < (int)animations.size(); ++i)
                    animations[i].current = mAnimTargetTimes[i];
            }
            else
            {
                // Motion blur disabled or no shutter: single-pass at target time
                bool accelStructureDirty = false;
                for (int i = 0; i < (int)animations.size(); ++i)
                {
                    if (!mAnimChanged[i]) continue;
                    animations[i].current = mAnimTargetTimes[i];
                    accelStructureDirty |= mScene->applyAnimation(i);
                }

                if (accelStructureDirty)
                {
                    applySkinning();
                    // Sync prevVB with current VB — motion BVH needs both keyframes
                    // consistent when motion blur is off (otherwise keyframe 0 is stale)
                    copyVertexBufferToPrev();
                    updateSkeletalBLAS(maxTimeDelta > 0.1f);
                    rebuildTLAS();
                }
                else
                {
                    rebuildTLAS(); // already re-uploads the instance transforms
                }
            }
            ctx.mSubframeIndex = 0;
        }
    }

    SettingsManager& settings = *getSettings();

    const uint32_t selectedCamera = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCamera);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    const View currView{camera.matrices};

    if (!motionBlurCameraSet && ctx.mFrameNumber == 0)
    {
        mPrevMotionBlurView.mCamMatrices = camera.matrices;
    }

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        ctx.mSubframeIndex = 0;
    }

    // --- Cache all settings once per frame ---
    const uint32_t spp = settings.getAs<uint32_t>("render/pt/spp");
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    const uint32_t maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    const uint32_t debug = settings.getAs<uint32_t>("render/pt/debug");
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    const uint32_t samplerType = settings.getAs<uint32_t>("render/pt/samplerType");
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    const bool isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");
    const bool enableCameraMotionBlur = settings.getAs<bool>("render/enableCameraMotionBlur");

    MTL::Buffer* pUniformBuffer = mUniformBuffers[mFrameIndex];
    MTL::Buffer* pUniformTMBuffer = mUniformTMBuffers[mFrameIndex];
    auto pUniformData = reinterpret_cast<Uniforms*>(pUniformBuffer->contents());
    auto pUniformTonemap = reinterpret_cast<UniformsTonemap*>(pUniformTMBuffer->contents());
    pUniformData->frameIndex = mFrameIndex;
    pUniformData->subframeIndex = ctx.mSubframeIndex;
    pUniformData->height = height;
    pUniformData->width = width;
    const bool analyticLightsEnabled = settings.getAs<bool>("render/validate/analyticLights");
    pUniformData->numLights = analyticLightsEnabled ? (uint32_t)mScene->getLightsDesc().size() : 0u;
    pUniformData->primaryRayMask = analyticLightsEnabled ? RAY_MASK_PRIMARY : GEOMETRY_MASK_GEOMETRY;
    pUniformData->estimatorMode = settings.getAs<uint32_t>("render/validate/estimatorMode");
    pUniformData->samples_per_launch = spp;
    pUniformData->enableAccumulation = (uint32_t)enableAccumulation;
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = maxDepth;
    pUniformData->debug = debug;
    pUniformData->enableMotionBlur = mEnableMotionBlur ? 1 : 0;
    pUniformData->isMotionBlurVisible = (uint32_t)isMotionBlurVisible;
    pUniformData->enableCameraMotionBlur = (uint32_t)enableCameraMotionBlur;
    pUniformData->rectLightSamplingMethod = rectLightSamplingMethod;
    pUniformData->samplerType = samplerType;

    // Depth of field
    pUniformData->useDof = camera.useDof ? 1 : 0;
    pUniformData->focalDistance = camera.focalDistance;
    pUniformData->lensRadius = camera.useDof ? camera.focalLengthMm / (2.0f * camera.fStopDof * 1000.0f) : 0.0f;
    pUniformData->apertureBlades = camera.apertureBlades;
    pUniformData->bladeRotation = camera.bladeRotation;
    pUniformData->anamorphicRatio = camera.anamorphicRatio;

    // Lens shift
    pUniformData->shiftX = camera.shiftX;
    pUniformData->shiftY = camera.shiftY;

    // Environment map
    if (mEnvMapLoaded)
    {
        const auto& envLight = mScene->getEnvLight();
        pUniformData->hasEnvMap = 1;
        pUniformData->envMapWidth = (uint32_t)mEnvMapTexture->width();
        pUniformData->envMapHeight = (uint32_t)mEnvMapTexture->height();
        const float userIntensity = envLight.has_value() ? envLight->intensity : 1.0f;
        pUniformData->envMapIntensity = mEnvMapAutoScale * userIntensity;
        pUniformData->envMapRotation = envLight.has_value() ? envLight->rotationY * (M_PI / 180.0f) : 0.0f;
        pUniformData->envPdfScale = mEnvPdfScale;
        if (envLight.has_value())
        {
            pUniformData->envMapColorTint = { envLight->color.x, envLight->color.y, envLight->color.z };
        }
        else
        {
            pUniformData->envMapColorTint = { 1.0f, 1.0f, 1.0f };
        }
    }
    else
    {
        pUniformData->hasEnvMap = 0;
        pUniformData->envPdfScale = 0.0f;
    }

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->tonemapperType = settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = settings.getAs<float>("render/post/gamma");
    pUniformTonemap->maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");

    // --- Detect settings changes (member-based, not static) ---
    bool settingsChanged = false;
    settingsChanged |= (mPrevSettings.rectLightSamplingMethod != rectLightSamplingMethod);
    settingsChanged |= (mPrevSettings.samplerType != samplerType);
    settingsChanged |= (mPrevSettings.enableAccumulation != enableAccumulation);
    settingsChanged |= (mPrevSettings.sspTotal > sspTotal);
    settingsChanged |= (mPrevSettings.spp != spp);
    settingsChanged |= (mPrevSettings.enableMotionBlur != mEnableMotionBlur);
    settingsChanged |= (mPrevSettings.isMotionBlurVisible != isMotionBlurVisible);
    settingsChanged |= (mPrevSettings.enableCameraMotionBlur != enableCameraMotionBlur);
    settingsChanged |= (mPrevSettings.useDof != pUniformData->useDof);
    settingsChanged |= (mPrevSettings.focalDistance != pUniformData->focalDistance);
    settingsChanged |= (mPrevSettings.lensRadius != pUniformData->lensRadius);
    settingsChanged |= (mPrevSettings.apertureBlades != pUniformData->apertureBlades);
    settingsChanged |= (mPrevSettings.shiftX != pUniformData->shiftX) || (mPrevSettings.shiftY != pUniformData->shiftY);
    settingsChanged |= (mPrevSettings.maxDepth != maxDepth);
    settingsChanged |= (mPrevSettings.debug != debug);

    mPrevSettings.rectLightSamplingMethod = rectLightSamplingMethod;
    mPrevSettings.samplerType = samplerType;
    mPrevSettings.enableAccumulation = enableAccumulation;
    mPrevSettings.sspTotal = sspTotal;
    mPrevSettings.spp = spp;
    mPrevSettings.enableMotionBlur = mEnableMotionBlur;
    mPrevSettings.isMotionBlurVisible = isMotionBlurVisible;
    mPrevSettings.enableCameraMotionBlur = enableCameraMotionBlur;
    mPrevSettings.useDof = pUniformData->useDof;
    mPrevSettings.focalDistance = pUniformData->focalDistance;
    mPrevSettings.lensRadius = pUniformData->lensRadius;
    mPrevSettings.apertureBlades = pUniformData->apertureBlades;
    mPrevSettings.shiftX = pUniformData->shiftX;
    mPrevSettings.shiftY = pUniformData->shiftY;
    mPrevSettings.maxDepth = maxDepth;
    mPrevSettings.debug = debug;

    if (settingsChanged)
    {
        ctx.mSubframeIndex = 0;
    }

    // Matrix copies: glm and simd both use column-major layout
    const glm::float4x4 invView = glm::inverse(camera.matrices.view);
    std::memcpy(&pUniformData->viewToWorld, glm::value_ptr(invView), sizeof(float4x4));
    std::memcpy(&pUniformData->clipToView, glm::value_ptr(camera.matrices.invPerspective), sizeof(float4x4));

    {
        const glm::float4x4 prevInvView = glm::inverse(mPrevMotionBlurView.mCamMatrices.view);
        std::memcpy(&pUniformData->prevViewToWorld, glm::value_ptr(prevInvView), sizeof(float4x4));
        std::memcpy(&pUniformData->prevClipToView, glm::value_ptr(mPrevMotionBlurView.mCamMatrices.invPerspective), sizeof(float4x4));
    }

    pUniformData->subframeIndex = ctx.mSubframeIndex;

    // Photometric exposure
    const float filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    const float cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    const float fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    const float shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
    // Specifies the main color temperature of the light sources; the color that will be mapped to “white” on output,
    // e.g., an incoming color of this hue/saturation will be mapped to grayscale, but its intensity will remain
    // unchanged. This is similar to white balance controls on digital cameras.
    float3 whitePoint{ 1.0f, 1.0f, 1.0f };
    auto all = [](float3 v) { return v.x > 0.0f && v.y > 0.0f && v.z > 0.0f; };
    float3 exposureValue = all(whitePoint) ? 1.0f / whitePoint : float3(1.0f);
    const float lum = simd::dot(exposureValue, float3{ 0.299f, 0.587f, 0.114f });
    if (filmIso > 0.0f)
    {
        // See https://www.nayuki.io/page/the-photographic-exposure-equation
        exposureValue *= cm2_factor * filmIso / (shutterSpeed * fStop * fStop) / 100.0f;
    }
    else
    {
        exposureValue *= cm2_factor;
    }
    exposureValue /= lum;
    pUniformTonemap->exposureValue = exposureValue;
    pUniformData->exposureValue = exposureValue; // need for proper accumulation

    const auto samplesPerLaunch = pUniformData->samples_per_launch;
    const int32_t leftSpp = sspTotal - ctx.mSubframeIndex;
    // if accumulation is off then launch selected samples per pixel
    const uint32_t samplesThisLaunch =
        enableAccumulation ? std::min((int32_t)samplesPerLaunch, leftSpp) : samplesPerLaunch;
    if (samplesThisLaunch != 0 && mInstanceBuffer != nullptr)
    {
        pUniformData->samples_per_launch = samplesThisLaunch;

        pUniformBuffer->didModifyRange(NS::Range::Make(0, sizeof(Uniforms)));
        pUniformTMBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsTonemap)));

        // Environment map buffers must always be bound — the kernel declares
        // indices 10/11 unconditionally.
        // The kernel declares buffer(10) unconditionally, so it must always be bound.
        if (!mEnvAliasBuffer)
            mEnvAliasBuffer = mDevice->newBuffer(sizeof(EnvAliasEntry), MTL::ResourceStorageModeManaged);

        // --- Split the path trace into horizontal bands ---------------------
        //
        // A full-frame path-trace dispatch is a single indivisible unit of GPU
        // work. While it runs, the display queue's command buffer cannot start,
        // so nextDrawable() blocks and the whole UI thread stalls for the entire
        // render time — seconds per frame on a heavy scene.
        //
        // Splitting the frame into several command buffers gives the scheduler
        // preemption points between them, so the compositor keeps getting
        // drawables and the UI keeps its vsync cadence regardless of how long
        // the full frame takes. The band height is derived from the measured
        // per-row cost so each submission stays near kTargetSubmissionMs.
        // Wavefront mode replaces the banded megakernel dispatch entirely: it
        // already issues many short dispatches, so it needs no banding of its own.
        const bool useWavefront = settings.getAs<uint32_t>("render/pt/tracerMode") == 1 &&
                                  mWavefrontShadePSO != nullptr;
        if (useWavefront)
        {
            ensureWavefrontBuffers(width, height);
            mProfileStages = settings.getAs<uint32_t>("render/pt/profileStages") != 0;
            if (mProfileStages)
            {
                createStageTimestampBuffer();
            }

            MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = pCmd->computeCommandEncoder();
            enc = encodeWavefront(pCmd, enc, pUniformBuffer, output, width, height, samplesThisLaunch);

            if (pUniformData->debug == 0)
            {
                enc->setComputePipelineState(mTonemapperPSO);
                enc->useResource(((MetalBuffer*)output)->getNativePtr(),
                                 MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
                enc->setBuffer(pUniformTMBuffer, 0, 0);
                enc->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
                enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }
            enc->endEncoding();

            const int writeIdxWf = mWriteIndex;
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdxWf](MTL::CommandBuffer* cb) {
                mLastRenderTimeMs.store((cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0,
                                        std::memory_order_relaxed);
                if (mProfileStages)
                {
                    reportStageTimings();
                }
                mReadyIndex.store(writeIdxWf);
                mRenderBusy.store(false, std::memory_order_release);
            }));
            pCmd->commit();

            if (enableAccumulation)
            {
                ctx.mSubframeIndex += samplesThisLaunch;
            }
            else
            {
                ctx.mSubframeIndex = 0;
            }
            pPool->release();
            mPrevView = currView;
            ctx.mFrameNumber++;
            return;
        }

        // Banding exists to keep the UI responsive, and it costs a little: each
        // band is another command buffer, and the frame's measured span includes
        // the gaps between them. A benchmark comparing tracers has to be able to
        // turn it off, or it measures the submission strategy as much as the
        // tracer.
        const uint32_t rowsPerBand = settings.getAs<uint32_t>("render/pt/splitSubmissions")
                                         ? computeBandHeight(height)
                                         : height;
        const uint32_t bandCount = (height + rowsPerBand - 1) / rowsPerBand;
        mLastBandTotalRows = height;

        for (uint32_t band = 0; band < bandCount; ++band)
        {
            const uint32_t bandStart = band * rowsPerBand;
            const uint32_t bandRows = std::min(rowsPerBand, height - bandStart);
            const bool isLastBand = (band + 1 == bandCount);

            MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();

            encodePathTraceBindings(pComputeEncoder, pUniformBuffer, output);
            pComputeEncoder->setBytes(&bandStart, sizeof(uint32_t), 12);

            pComputeEncoder->dispatchThreads(MTL::Size(width, bandRows, 1), MTL::Size(8, 8, 1));

            // Tonemapping reads back the whole image, so it can only run once
            // every band has been written. Command buffers on a single queue
            // execute in submission order, so the last one is the right place.
            if (isLastBand && pUniformData->debug == 0)
            {
                pComputeEncoder->setComputePipelineState(mTonemapperPSO);
                pComputeEncoder->useResource(
                    ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
                pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
                pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
                pComputeEncoder->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }

            pComputeEncoder->endEncoding();

            const int writeIdx = mWriteIndex;
            const bool isFirstBand = (band == 0);
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx, isFirstBand, isLastBand](MTL::CommandBuffer* cb) {
                // Measure the span from the first band starting to the last one
                // finishing. Summing each band's own GPU interval instead would
                // overcount: the bands are separate submissions, so the sum also
                // picks up per-command-buffer setup and any time the GPU spent on
                // the display queue in between. That inflated number fed straight
                // back into computeBandHeight() and drove the split ever finer —
                // a feedback loop that made the renderer slower every frame.
                if (isFirstBand)
                {
                    mFrameGpuStartSeconds.store(cb->GPUStartTime(), std::memory_order_relaxed);
                }
                if (isLastBand)
                {
                    const double start = mFrameGpuStartSeconds.load(std::memory_order_relaxed);
                    const double spanMs = (cb->GPUEndTime() - start) * 1000.0;
                    if (spanMs > 0.0)
                    {
                        mLastRenderTimeMs.store(spanMs, std::memory_order_relaxed);
                    }
                    mReadyIndex.store(writeIdx);
                    // Must be released last: it is what lets the next frame start.
                    mRenderBusy.store(false, std::memory_order_release);
                }
            }));
            pCmd->commit();
        }

        if (enableAccumulation)
        {
            ctx.mSubframeIndex += samplesThisLaunch;
        }
        else
        {
            ctx.mSubframeIndex = 0;
        }
    }
    else
    {
        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();

        MTL::BlitCommandEncoder* pBlitEncoder = pCmd->blitCommandEncoder();
        pBlitEncoder->copyFromBuffer(
            mAccumulationBuffer, 0, ((MetalBuffer*)output)->getNativePtr(), 0, width * height * sizeof(float4));
        pBlitEncoder->endEncoding();

        // Disable tonemapping for debug output
        if (pUniformData->debug == 0)
        {
            MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();

            pComputeEncoder->setComputePipelineState(mTonemapperPSO);
            pComputeEncoder->useResource(
                ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
            pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
            pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
            {
                const MTL::Size gridSize = MTL::Size(width, height, 1);
                const MTL::Size threadgroupSize(8, 8, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
            pComputeEncoder->endEncoding();
        }

        // Completion handler for async double-buffered output. It must be
        // installed unconditionally: it is the only thing that clears
        // mRenderBusy, and skipping it would wedge the renderer permanently.
        const int writeIdx = mWriteIndex;
        pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx](MTL::CommandBuffer* cb) {
            const double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
            mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
            mReadyIndex.store(writeIdx);
            mRenderBusy.store(false, std::memory_order_release);
        }));
        pCmd->commit();
    }
    pPool->release();

    mPrevView = currView;
    ctx.mFrameNumber++;
}

Buffer* MetalRender::createBuffer(const BufferDesc& desc)
{
    assert(mDevice);
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    MTL::Buffer* buff = mDevice->newBuffer(size, MTL::ResourceStorageModeManaged);
    assert(buff);
    auto res = new MetalBuffer(buff, desc.format, desc.width, desc.height);
    assert(res);
    return res;
}

MTL::Library* MetalRender::loadShaderLibrary(const char* relativePath)
{
    const std::string path = oka::resolveResourcePath(relativePath);
    NS::Error* pError = nullptr;
    MTL::Library* pLibrary =
        mDevice->newLibrary(NS::String::string(path.c_str(), NS::UTF8StringEncoding), &pError);
    if (!pLibrary)
    {
        STRELKA_FATAL("Failed to load {}: {}", path,
                      pError ? pError->localizedDescription()->utf8String() : "unknown error");
    }
    return pLibrary;
}

void MetalRender::buildComputePipeline()
{
    MTL::Library* pComputeLibrary = loadShaderLibrary("metal/shaders/pathtrace.metallib");
    if (!pComputeLibrary)
    {
        return;
    }
    NS::Error* pError = nullptr;
    MTL::Function* pPathTraceFn =
        pComputeLibrary->newFunction(NS::String::string("raytracingKernel", NS::UTF8StringEncoding));
    mPathTracingPSO = mDevice->newComputePipelineState(pPathTraceFn, &pError);
    if (!mPathTracingPSO)
    {
        STRELKA_FATAL("{}", pError ? pError->localizedDescription()->utf8String() : "unknown error");
        assert(false);
    }

    pPathTraceFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildWavefrontPipelines()
{
    MTL::Library* lib = loadShaderLibrary("metal/shaders/wavefront.metallib");
    if (!lib)
    {
        return;
    }
    NS::Error* err = nullptr;
    auto make = [&](const char* name) -> MTL::ComputePipelineState* {
        MTL::Function* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        if (!fn)
        {
            STRELKA_FATAL("wavefront: missing function {}", name);
            return nullptr;
        }
        MTL::ComputePipelineState* pso = mDevice->newComputePipelineState(fn, &err);
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        fn->release();
        return pso;
    };
    mWavefrontGeneratePSO = make("wavefrontGenerate");
    mWavefrontExtendPSO = make("wavefrontExtend");
    mWavefrontShadePSO = make("wavefrontShade");
    mWavefrontResolvePSO = make("wavefrontResolve");
    mWavefrontPreparePSO = make("wavefrontPrepare");
    mWavefrontPrepareShadowPSO = make("wavefrontPrepareShadow");
    mWavefrontShadowPSO = make("wavefrontShadow");
    mWavefrontSortCountPSO = make("wavefrontSortCount");
    mWavefrontSortScanPSO = make("wavefrontSortScan");
    mWavefrontSortScatterPSO = make("wavefrontSortScatter");
    lib->release();

    if (mWavefrontShadePSO)
    {
        STRELKA_INFO("wavefront PSO: shade maxThreadsPerTG={} extend={} generate={}",
                     mWavefrontShadePSO->maxTotalThreadsPerThreadgroup(),
                     mWavefrontExtendPSO ? mWavefrontExtendPSO->maxTotalThreadsPerThreadgroup() : 0,
                     mWavefrontGeneratePSO ? mWavefrontGeneratePSO->maxTotalThreadsPerThreadgroup() : 0);
    }
}

void MetalRender::ensureWavefrontBuffers(uint32_t width, uint32_t height)
{
    const uint32_t pixels = width * height;
    if (pixels == mWavefrontCapacity && mPathStateBuffer)
    {
        return;
    }
    auto release = [](MTL::Buffer*& b) { if (b) { b->release(); b = nullptr; } };
    release(mPathStateBuffer);
    release(mHitBuffer);
    release(mIorStackBuffer);
    release(mRadianceBuffer);
    release(mPathQueueBuffer[0]);
    release(mPathQueueBuffer[1]);
    release(mWavefrontControlBuffer);
    release(mShadowRayBuffer);
    release(mStageStatsBuffer);
    release(mSortBinBuffer);
    release(mSortTgBaseBuffer);

    // Private storage: these never leave the GPU.
    mPathStateBuffer = mDevice->newBuffer(pixels * sizeof(PathState), MTL::ResourceStorageModePrivate);
    mHitBuffer = mDevice->newBuffer(pixels * sizeof(HitRecord), MTL::ResourceStorageModePrivate);
    mIorStackBuffer = mDevice->newBuffer(pixels * sizeof(IorStack), MTL::ResourceStorageModePrivate);
    mRadianceBuffer = mDevice->newBuffer(pixels * sizeof(simd::float4), MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[0] = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[1] = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // Queue counters, active counts, and two sets of indirect dispatch arguments.
    mWavefrontControlBuffer = mDevice->newBuffer(80 * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // At most one deferred connection per path per bounce.
    mShadowRayBuffer = mDevice->newBuffer(pixels * sizeof(ShadowRay), MTL::ResourceStorageModePrivate);
    mStageStatsBuffer = mDevice->newBuffer(80 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    const uint32_t sortGroups = (pixels + kSortThreadgroup - 1) / kSortThreadgroup;
    mSortBinBuffer = mDevice->newBuffer(2 * kSortBins * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mSortTgBaseBuffer =
        mDevice->newBuffer((size_t)sortGroups * kSortBins * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mWavefrontCapacity = pixels;

    STRELKA_INFO("wavefront buffers for {}x{}: {:.1f} MB total", width, height,
                 (pixels * (sizeof(PathState) + sizeof(HitRecord) + sizeof(IorStack) + sizeof(simd::float4)))
                     / (1024.0 * 1024.0));
}

void MetalRender::buildTonemapperPipeline()
{
    MTL::Library* pComputeLibrary = loadShaderLibrary("metal/shaders/tonemapper.metallib");
    if (!pComputeLibrary)
    {
        return;
    }
    NS::Error* pError = nullptr;
    MTL::Function* pTonemapperFn =
        pComputeLibrary->newFunction(NS::String::string("toneMappingComputeShader", NS::UTF8StringEncoding));
    mTonemapperPSO = mDevice->newComputePipelineState(pTonemapperFn, &pError);
    if (!mTonemapperPSO)
    {
        STRELKA_FATAL("{}", pError ? pError->localizedDescription()->utf8String() : "unknown error");
        assert(false);
    }

    pTonemapperFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildBuffers()
{
    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();

    static_assert(sizeof(Scene::Light) == sizeof(UniformLight));
    const size_t lightBufferSize = sizeof(Scene::Light) * lightDescs.size();
    const size_t vertexDataSize = sizeof(Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    MTL::Buffer* pLightBuffer = nullptr;
    if (lightBufferSize > 0)
    {
        pLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeManaged);
        memcpy(pLightBuffer->contents(), lightDescs.data(), lightBufferSize);
        pLightBuffer->didModifyRange(NS::Range::Make(0, pLightBuffer->length()));
    }
    MTL::Buffer* pVertexBuffer = nullptr;
    if (vertexDataSize > 0)
    {
        pVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(pVertexBuffer->contents(), vertices.data(), vertexDataSize);
        pVertexBuffer->didModifyRange(NS::Range::Make(0, pVertexBuffer->length()));
    };
    MTL::Buffer* pIndexBuffer = nullptr;
    if (indexDataSize > 0)
    {
        pIndexBuffer = mDevice->newBuffer(indexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(pIndexBuffer->contents(), indices.data(), indexDataSize);
        pIndexBuffer->didModifyRange(NS::Range::Make(0, pIndexBuffer->length()));
    }

    mLightBuffer = pLightBuffer;
    mVertexBuffer = pVertexBuffer;
    mIndexBuffer = pIndexBuffer;

    // Allocate prevVertexBuffer as copy of VB (needed for motion BVH keyframes at init time)
    if (vertexDataSize > 0)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mPrevVertexBuffer->contents(), vertices.data(), vertexDataSize);
        mPrevVertexBuffer->didModifyRange(NS::Range::Make(0, mPrevVertexBuffer->length()));
    }

    for (MTL::Buffer*& uniformBuffer : mUniformBuffers)
    {
        uniformBuffer = mDevice->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeManaged);
    }
    for (MTL::Buffer*& uniformBuffer : mUniformTMBuffers)
    {
        uniformBuffer = mDevice->newBuffer(sizeof(UniformsTonemap), MTL::ResourceStorageModeManaged);
    }
}

MTL::AccelerationStructure* MetalRender::createAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Query for the sizes needed to store and build the acceleration structure.
    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    // Allocate an acceleration structure large enough for this descriptor. This doesn't actually
    // build the acceleration structure, it just allocates memory.
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    // Allocate scratch space Metal uses to build the acceleration structure.
    // Use MTLResourceStorageModePrivate for best performance because the sample
    // doesn't need access to the buffer's contents.
    MTL::Buffer* scratchBuffer = mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    // Create a command buffer to perform the acceleration structure build.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    // Create an acceleration structure command encoder.
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
    // Allocate a buffer for Metal to write the compacted accelerated structure's size into.
    MTL::Buffer* compactedSizeBuffer = mDevice->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    // Schedule the actual acceleration structure build.
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    // Compute and write the compacted acceleration structure size into the buffer. You
    // need to already have a built accelerated structure because Metal determines the compacted
    // size based on the final size of the acceleration structure. Compacting an acceleration
    // structure can potentially reclaim significant amounts of memory because Metal must
    // create the initial structure using a conservative approach.
    commandEncoder->writeCompactedAccelerationStructureSize(accelerationStructure, compactedSizeBuffer, 0UL);
    // End encoding and commit the command buffer so the GPU can start building the
    // acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // The sample waits for Metal to finish executing the command buffer so that it can
    // read back the compacted size.

    // Note: Don't wait for Metal to finish executing the command buffer if you aren't compacting
    // the acceleration structure because doing so requires CPU/GPU synchronization. You don't have
    // to compact acceleration structures, but it's helpful when creating large static acceleration
    // structures, such as static scene geometry. Avoid compacting acceleration structures that
    // you rebuild every frame because the synchronization cost may be significant.

    commandBuffer->waitUntilCompleted();

    const uint32_t compactedSize = *(uint32_t*)compactedSizeBuffer->contents();

    // commandBuffer->release();
    // commandEncoder->release();

    // Allocate a smaller acceleration structure based on the returned size.
    MTL::AccelerationStructure* compactedAccelerationStructure = mDevice->newAccelerationStructure(compactedSize);

    // Create another command buffer and encoder.
    commandBuffer = mCommandQueue->commandBuffer();
    commandEncoder = commandBuffer->accelerationStructureCommandEncoder();

    // Encode the command to copy and compact the acceleration structure into the
    // smaller acceleration structure.
    commandEncoder->copyAndCompactAccelerationStructure(accelerationStructure, compactedAccelerationStructure);

    // End encoding and commit the command buffer. You don't need to wait for Metal to finish
    // executing this command buffer as long as you synchronize any ray-intersection work
    // to run after this command buffer completes. The sample relies on Metal's default
    // dependency tracking on resources to automatically synchronize access to the new
    // compacted acceleration structure.
    commandEncoder->endEncoding();
    commandBuffer->commit();

    // commandEncoder->release();
    // commandBuffer->release();
    accelerationStructure->release();
    scratchBuffer->release();
    compactedSizeBuffer->release();

    pPool->release();

    return compactedAccelerationStructure->retain();
}

MTL::AccelerationStructure* MetalRender::createAccelerationStructureNoCompact(
    MTL::AccelerationStructureDescriptor* descriptor)
{
    // Allow refitting on this descriptor
    descriptor->setUsage(MTL::AccelerationStructureUsageRefit);

    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);

    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
    commandEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    commandEncoder->endEncoding();
    commandBuffer->commit();
    // No waitUntilCompleted — Metal queue ordering guarantees subsequent
    // command buffers on the same queue see the built AS.

    scratchBuffer->release();
    return accelerationStructure;
}

void MetalRender::createMeshData(size_t meshIndex)
{
    const oka::Mesh& mesh = mScene->getMeshes()[meshIndex];
    auto result = new MetalRender::Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;
    result->mTriangleCount = triangleCount;
    result->mVbOffset = mesh.mVbOffset;
    result->mIndexOffset = mesh.mIndex;
    result->mIsSkeletal = mesh.isSkeletal;

    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    std::vector<Triangle> triangleData(triangleCount);
    for (uint32_t i = 0; i < triangleCount; ++i)
    {
        Triangle& curr = triangleData[i];
        const uint32_t i0 = indices[mesh.mIndex + i * 3 + 0];
        const uint32_t i1 = indices[mesh.mIndex + i * 3 + 1];
        const uint32_t i2 = indices[mesh.mIndex + i * 3 + 2];

        curr.positions[0] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i0].pos.x,
                                                          vertices[mesh.mVbOffset + i0].pos.y,
                                                          vertices[mesh.mVbOffset + i0].pos.z));
        curr.positions[1] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i1].pos.x,
                                                          vertices[mesh.mVbOffset + i1].pos.y,
                                                          vertices[mesh.mVbOffset + i1].pos.z));
        curr.positions[2] = packed_float3(simd_make_float3(vertices[mesh.mVbOffset + i2].pos.x,
                                                          vertices[mesh.mVbOffset + i2].pos.y,
                                                          vertices[mesh.mVbOffset + i2].pos.z));
        curr.normals[0] = vertices[mesh.mVbOffset + i0].normal;
        curr.normals[1] = vertices[mesh.mVbOffset + i1].normal;
        curr.normals[2] = vertices[mesh.mVbOffset + i2].normal;
        curr.tangent[0] = vertices[mesh.mVbOffset + i0].tangent;
        curr.tangent[1] = vertices[mesh.mVbOffset + i1].tangent;
        curr.tangent[2] = vertices[mesh.mVbOffset + i2].tangent;
        curr.uv[0] = vertices[mesh.mVbOffset + i0].uv;
        curr.uv[1] = vertices[mesh.mVbOffset + i1].uv;
        curr.uv[2] = vertices[mesh.mVbOffset + i2].uv;
    }

    result->mPerPrimitiveBuffer =
        mDevice->newBuffer(triangleData.size() * sizeof(Triangle), MTL::ResourceStorageModeManaged);
    memcpy(result->mPerPrimitiveBuffer->contents(), triangleData.data(), sizeof(Triangle) * triangleData.size());
    result->mPerPrimitiveBuffer->didModifyRange(NS::Range(0, result->mPerPrimitiveBuffer->length()));

    mMetalMeshes.push_back(result);
}

MTL::AccelerationStructureTriangleGeometryDescriptor* MetalRender::createStaticGeometryDescriptor(
    const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount)
{
    auto* geomDescriptor = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();

    geomDescriptor->setVertexBuffer(mVertexBuffer);
    geomDescriptor->setVertexBufferOffset(sceneMesh.mVbOffset * sizeof(Scene::Vertex));
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));
    geomDescriptor->setIndexBuffer(mIndexBuffer);
    geomDescriptor->setIndexBufferOffset(sceneMesh.mIndex * sizeof(uint32_t));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
    geomDescriptor->setPrimitiveDataBufferOffset(0);
    geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
    geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

    return geomDescriptor;
}

// Build one acceleration structure covering every listed instance's mesh as a
// separate geometry, and record the per-geometry lookup entries.
size_t MetalRender::buildBlas(const std::vector<uint32_t>& sceneInstanceIds, bool skeletal)
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();

    Blas blas;
    blas.mIsSkeletal = skeletal;
    blas.mGeometryBase = (uint32_t)mGeometryEntries.size();

    std::vector<const NS::Object*> geomDescriptors;
    geomDescriptors.reserve(sceneInstanceIds.size());

    for (const uint32_t instId : sceneInstanceIds)
    {
        const oka::Instance& inst = instances[instId];
        const uint32_t meshId = inst.mMeshId;
        const oka::Mesh& mesh = meshes[meshId];
        MetalRender::Mesh* meshData = mMetalMeshes[meshId];

        if (skeletal)
        {
            geomDescriptors.push_back(
                createMotionGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount));
        }
        else
        {
            geomDescriptors.push_back(
                createStaticGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount));
        }

        // The geometry index within this BLAS is the position in this list, so
        // the material of the instance that contributed it lands in the right slot.
        GeometryEntry entry{};
        entry.vbOffset = mesh.mVbOffset;
        entry.indexOffset = mesh.mIndex;
        entry.materialId = inst.mMaterialId;
        mGeometryEntries.push_back(entry);
    }

    NS::Array* geomArray = NS::Array::array(geomDescriptors.data(), geomDescriptors.size());
    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(geomArray);

    if (skeletal)
    {
        primDescriptor->setMotionKeyframeCount(2);
        primDescriptor->setMotionStartTime(0.0f);
        primDescriptor->setMotionEndTime(1.0f);
        primDescriptor->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
        primDescriptor->setMotionEndBorderMode(MTL::MotionBorderModeClamp);
        primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit);

        const MTL::AccelerationStructureSizes sizes = mDevice->accelerationStructureSizes(primDescriptor);
        blas.mRefitScratchSize = sizes.refitScratchBufferSize;
        blas.mBuildScratchSize = sizes.buildScratchBufferSize;

        blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
        blas.mDescriptor = primDescriptor; // kept for refit, released in the destructor
    }
    else
    {
        // Static geometry is built once, so it is worth compacting.
        blas.mAs = createAccelerationStructure(primDescriptor);
        primDescriptor->release();
    }

    for (const NS::Object* g : geomDescriptors)
    {
        ((NS::Object*)g)->release();
    }

    mBlasList.push_back(blas);
    mPrimitiveAccelerationStructures.push_back(blas.mAs);
    return mBlasList.size() - 1;
}

void MetalRender::createAccelerationStructures()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        pPool->release();
        return;
    }

    for (size_t mi = 0; mi < meshes.size(); ++mi)
    {
        createMeshData(mi);
    }

    // --- Group mesh instances that always move together -----------------------
    //
    // glTF splits a mesh into primitives by material, and the loader turns each
    // primitive into its own instance. Left alone that produces one BLAS per
    // primitive, all with the same transform and heavily overlapping bounds, and
    // every ray that has to traverse deeply pays for the overlap. Instances that
    // hang off the same node share a transform by construction, so they can be
    // merged into a single BLAS with one geometry per primitive.
    std::vector<int> instanceNode(instances.size(), -1);
    const std::vector<Scene::Node>& nodes = mScene->getNodes();
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        for (const uint32_t id : nodes[n].instanceIds)
        {
            if (id < instanceNode.size())
                instanceNode[id] = (int)n;
        }
    }

    // Key: (node, skeletal). Instances without a node, and any whose transform
    // does not actually match the group's, get a group of their own.
    std::map<std::pair<int, int>, size_t> groupOfKey;
    std::vector<std::vector<uint32_t>> groups;
    std::vector<bool> groupSkeletal;

    // Light instances keep one BLAS per mesh, shared between lights, because
    // their userID must stay the light index.
    std::map<uint32_t, size_t> lightBlasOfMesh;

    mGeometryEntries.clear();
    mEmittedInstances.clear();

    std::vector<size_t> groupBlas;

    for (size_t i = 0; i < instances.size(); ++i)
    {
        const oka::Instance& curr = instances[i];
        if (curr.type == oka::Instance::Type::eLight)
        {
            continue; // handled below
        }
        const bool skeletal = meshes[curr.mMeshId].isSkeletal;
        const int nodeId = instanceNode[i];
        const std::pair<int, int> key{ nodeId >= 0 ? nodeId : -(int)i - 2, skeletal ? 1 : 0 };

        auto it = groupOfKey.find(key);
        if (it == groupOfKey.end())
        {
            groupOfKey[key] = groups.size();
            groups.push_back({ (uint32_t)i });
            groupSkeletal.push_back(skeletal);
            continue;
        }
        // Merging is only valid while the members share a transform.
        const oka::Instance& rep = instances[groups[it->second].front()];
        if (memcmp(&rep.transform, &curr.transform, sizeof(glm::mat4)) == 0)
        {
            groups[it->second].push_back((uint32_t)i);
        }
        else
        {
            groups.push_back({ (uint32_t)i });
            groupSkeletal.push_back(skeletal);
        }
    }

    size_t mergedGeometries = 0;
    for (size_t g = 0; g < groups.size(); ++g)
    {
        const size_t blasIdx = buildBlas(groups[g], groupSkeletal[g]);
        groupBlas.push_back(blasIdx);
        mergedGeometries += groups[g].size();

        EmittedInstance emitted{};
        emitted.sceneInstanceId = groups[g].front();
        emitted.asIndex = (uint32_t)blasIdx;
        emitted.userID = mBlasList[blasIdx].mGeometryBase;
        emitted.mask = GEOMETRY_MASK_TRIANGLE;

        // Point every geometry of this BLAS back at the instance that carries
        // it. The megakernel reads the object-to-world transform off the
        // intersection, but the wavefront tracer shades in a separate kernel
        // where the intersection is gone, so this is its only route back to the
        // instance descriptor. Each group emits exactly one instance, so the
        // mapping is one run of entries per instance.
        const uint32_t instanceIndex = (uint32_t)mEmittedInstances.size();
        for (size_t e = mBlasList[blasIdx].mGeometryBase; e < mGeometryEntries.size(); ++e)
        {
            mGeometryEntries[e].instanceIndex = instanceIndex;
        }
        mEmittedInstances.push_back(emitted);
    }

    for (size_t i = 0; i < instances.size(); ++i)
    {
        const oka::Instance& curr = instances[i];
        if (curr.type != oka::Instance::Type::eLight)
        {
            continue;
        }
        auto it = lightBlasOfMesh.find(curr.mMeshId);
        if (it == lightBlasOfMesh.end())
        {
            const size_t blasIdx = buildBlas({ (uint32_t)i }, meshes[curr.mMeshId].isSkeletal);
            it = lightBlasOfMesh.emplace(curr.mMeshId, blasIdx).first;
        }
        EmittedInstance emitted{};
        emitted.sceneInstanceId = (uint32_t)i;
        emitted.asIndex = (uint32_t)it->second;
        emitted.userID = curr.mLightId; // lights address the light table, not geometry
        emitted.mask = GEOMETRY_MASK_LIGHT;
        mEmittedInstances.push_back(emitted);
    }

    STRELKA_INFO("Acceleration structures: {} BLAS ({} geometries), {} TLAS instances (from {} scene instances)",
                 mBlasList.size(), mergedGeometries, mEmittedInstances.size(), instances.size());

    // Per-geometry lookup table consumed by the kernel.
    if (!mGeometryEntries.empty())
    {
        mGeometryEntryBuffer = mDevice->newBuffer(
            mGeometryEntries.size() * sizeof(GeometryEntry), MTL::ResourceStorageModeManaged);
        memcpy(mGeometryEntryBuffer->contents(), mGeometryEntries.data(),
               mGeometryEntries.size() * sizeof(GeometryEntry));
        mGeometryEntryBuffer->didModifyRange(NS::Range::Make(0, mGeometryEntryBuffer->length()));
    }

    mInstanceBuffer = mDevice->newBuffer(
        sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor) * std::max<size_t>(mEmittedInstances.size(), 1),
        MTL::ResourceStorageModeManaged);
    auto instanceDescriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)mInstanceBuffer->contents();
    for (size_t d = 0; d < mEmittedInstances.size(); ++d)
    {
        const EmittedInstance& e = mEmittedInstances[d];
        instanceDescriptors[d].accelerationStructureIndex = e.asIndex;
        instanceDescriptors[d].options = MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[d].intersectionFunctionTableOffset = 0;
        instanceDescriptors[d].userID = e.userID;
        instanceDescriptors[d].mask = e.mask;
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));
    updateInstanceTransforms();

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(mEmittedInstances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);

    mInstanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    mTlasInstanceCount = mEmittedInstances.size();
    pPool->release();
}

void MetalRender::buildSkinningPipeline()
{
    MTL::Library* pLibrary = loadShaderLibrary("metal/shaders/skinning.metallib");
    if (!pLibrary)
    {
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
    if (!mTriangleUpdatePSO)
    {
        STRELKA_FATAL("Failed to create triangle update PSO: {}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    pTriUpdateFn->release();

    pLibrary->release();
}

void MetalRender::createSkinDataBuffer()
{
    const std::vector<Scene::vertexSkinData>& skinData = mScene->getVerticesSkinData();
    if (skinData.empty())
        return;

    const size_t dataSize = skinData.size() * sizeof(Scene::vertexSkinData);
    mSkinDataBuffer = mDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    memcpy(mSkinDataBuffer->contents(), skinData.data(), dataSize);
    mSkinDataBuffer->didModifyRange(NS::Range::Make(0, mSkinDataBuffer->length()));
}

void MetalRender::allocJointMatrices()
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
            mDevice->newBuffer(jointMatSize * sizeof(simd::float4x4), MTL::ResourceStorageModeManaged);
    }
}

void MetalRender::applySkinning()
{
    if (!mSkinningPSO || !mSkinDataBuffer || !mJointMatricesBuffer)
        return;

    // Compute joint matrices on CPU. mJointMatScratch is a member so the two
    // skinning passes per frame (t_open / t_close for motion blur) reuse the same
    // allocation instead of churning two vectors each.
    mJointMatScratch.clear();
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            mScene->computeJointMatrices(&mJointMatScratch, jointCount, node.skin);
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
    mJointMatricesBuffer->didModifyRange(NS::Range::Make(0, uploadBytes));

    // Dispatch skinning + triangle update kernels
    MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* pEncoder = pCmd->computeCommandEncoder();

    int skinIndex = 0;
    int jointMatOffset = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            if (skinIndex > 0)
            {
                jointMatOffset += mJointMatOffsets[skinIndex - 1];
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

                pEncoder->setComputePipelineState(mSkinningPSO);
                pEncoder->setBuffer(mVertexBuffer, 0, 0);
                pEncoder->setBuffer(mSkinDataBuffer, 0, 1);
                pEncoder->setBuffer(mJointMatricesBuffer, 0, 2);
                pEncoder->setBytes(&skinParams, sizeof(SkinningParams), 3);

                const uint32_t threadsPerGroup = 256;
                const MTL::Size gridSize = MTL::Size(mesh.mVertexCount, 1, 1);
                const MTL::Size groupSize = MTL::Size(threadsPerGroup, 1, 1);
                pEncoder->dispatchThreads(gridSize, groupSize);

                // Dispatch triangle update kernel
                MetalRender::Mesh* metalMesh = mMetalMeshes[meshId];
                if (metalMesh->mPerPrimitiveBuffer)
                {
                    TriangleUpdateParams triParams = {};
                    triParams.triangleCount = metalMesh->mTriangleCount;
                    triParams.indexOffset = mesh.mIndex;
                    triParams.vbOffset = mesh.mVbOffset;

                    pEncoder->setComputePipelineState(mTriangleUpdatePSO);
                    pEncoder->setBuffer(metalMesh->mPerPrimitiveBuffer, 0, 0);
                    pEncoder->setBuffer(mVertexBuffer, 0, 1);
                    pEncoder->setBuffer(mIndexBuffer, 0, 2);
                    pEncoder->setBytes(&triParams, sizeof(TriangleUpdateParams), 3);

                    const MTL::Size triGridSize = MTL::Size(metalMesh->mTriangleCount, 1, 1);
                    pEncoder->dispatchThreads(triGridSize, groupSize);
                }
            }
        }
    }

    pEncoder->endEncoding();
    pCmd->commit();
    // No waitUntilCompleted — queue ordering guarantees subsequent AS operations
    // on the same queue see skinning results.
}

void MetalRender::copyVertexBufferToPrev()
{
    const size_t vertexDataSize = mVertexBuffer->length();
    MTL::CommandBuffer* blitCmd = mCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* blit = blitCmd->blitCommandEncoder();
    blit->copyFromBuffer(mVertexBuffer, 0, mPrevVertexBuffer, 0, vertexDataSize);
    blit->endEncoding();
    blitCmd->commit();
}

MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MetalRender::createMotionGeometryDescriptor(
    const oka::Mesh& sceneMesh, MTL::Buffer* perPrimitiveBuffer, uint32_t triangleCount)
{
    auto* geomDescriptor =
        MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()->init();

    MTL::MotionKeyframeData* kf0 = MTL::MotionKeyframeData::alloc()->init();
    kf0->setBuffer(mPrevVertexBuffer);
    kf0->setOffset(sceneMesh.mVbOffset * sizeof(Scene::Vertex));

    MTL::MotionKeyframeData* kf1 = MTL::MotionKeyframeData::alloc()->init();
    kf1->setBuffer(mVertexBuffer);
    kf1->setOffset(sceneMesh.mVbOffset * sizeof(Scene::Vertex));

    const NS::Object* keyframes[] = { kf0, kf1 };
    NS::Array* vertexBuffers = NS::Array::array(keyframes, 2UL);
    geomDescriptor->setVertexBuffers(vertexBuffers);
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));

    geomDescriptor->setIndexBuffer(mIndexBuffer);
    geomDescriptor->setIndexBufferOffset(sceneMesh.mIndex * sizeof(uint32_t));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
    geomDescriptor->setPrimitiveDataBufferOffset(0);
    geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
    geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

    kf0->release();
    kf1->release();

    return geomDescriptor;
}

void MetalRender::ensureScratchBuffer(MTL::Buffer*& buffer, size_t requiredSize)
{
    if (requiredSize == 0)
        requiredSize = 1;
    if (buffer && buffer->length() >= requiredSize)
        return;
    if (buffer)
        buffer->release();
    buffer = mDevice->newBuffer(requiredSize, MTL::ResourceStorageModePrivate);
}

void MetalRender::updateSkeletalBLAS(bool largeTimeJump)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Refit degrades BVH quality as the pose drifts from the one it was built
    // for, so a periodic full rebuild is still needed — but rebuilding *every*
    // skeletal mesh on the same frame produced a visible hitch every 10 frames.
    // Rebuild a bounded slice per frame instead and rotate through the meshes.
    const bool wantsFullRebuild = (mBlasUpdateCount >= 10) || largeTimeJump;
    const bool doRebuildSlice = wantsFullRebuild && (mFramesSinceFullRebuild >= 5);

    // All refits go into a single command buffer and a single encoder. Each mesh
    // used to get its own command buffer, so BrainStem — 59 skeletal meshes —
    // submitted 59 command buffers per animated frame. Submission overhead alone
    // dominated the frame; the actual refit work is tiny.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();

    const size_t blasCount = mBlasList.size();
    size_t rebuiltThisFrame = 0;

    for (size_t mi = 0; mi < blasCount; ++mi)
    {
        Blas& blas = mBlasList[mi];
        if (!blas.mIsSkeletal || !blas.mDescriptor)
            continue;

        // The descriptor and its scratch requirements were computed once when the
        // structure was created: they only name buffers, offsets and triangle
        // counts, none of which change while the pose does. Rebuilding them per
        // frame meant a pile of Objective-C allocations plus an
        // accelerationStructureSizes() driver query describing geometry that
        // never changes shape, only contents.
        const bool inRebuildSlice = doRebuildSlice && rebuiltThisFrame < kMaxBlasRebuildsPerFrame &&
                                    mi >= mNextBlasRebuildIndex;
        if (inRebuildSlice)
        {
            // A rebuild needs build scratch, which is the larger of the two.
            ensureScratchBuffer(blas.mScratch, std::max(blas.mBuildScratchSize, blas.mRefitScratchSize));
            commandEncoder->buildAccelerationStructure(blas.mAs, blas.mDescriptor, blas.mScratch, 0UL);
            ++rebuiltThisFrame;
            mNextBlasRebuildIndex = mi + 1;
        }
        else
        {
            ensureScratchBuffer(blas.mScratch, blas.mRefitScratchSize);
            commandEncoder->refitAccelerationStructure(blas.mAs, blas.mDescriptor, blas.mAs, blas.mScratch, 0UL);
        }
    }

    commandEncoder->endEncoding();
    commandBuffer->commit();

    if (doRebuildSlice && mNextBlasRebuildIndex >= blasCount)
    {
        // Finished a full sweep over every skeletal mesh.
        mNextBlasRebuildIndex = 0;
        mBlasUpdateCount = 0;
        mFramesSinceFullRebuild = 0;
    }
    else
    {
        mBlasUpdateCount++;
        mFramesSinceFullRebuild++;
    }

    pPool->release();
}

void MetalRender::updateInstanceTransforms()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    auto instanceDescriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)mInstanceBuffer->contents();

    for (size_t d = 0; d < mEmittedInstances.size(); ++d)
    {
        const Instance& curr = instances[mEmittedInstances[d].sceneInstanceId];
        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[d].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));
}

void MetalRender::rebuildTLAS()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Update instance transforms
    updateInstanceTransforms();

    const std::vector<oka::Instance>& instances = mScene->getInstances();

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(mEmittedInstances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
    accelDescriptor->setUsage(MTL::AccelerationStructureUsageRefit);

    // Only the instance transforms change while an animation plays — the set of
    // instances and the BLAS list are fixed. Refitting in place avoids allocating
    // (and freeing) a whole acceleration structure plus a scratch buffer on every
    // single frame, which is what the previous full rebuild did.
    const MTL::AccelerationStructureSizes sizes = mDevice->accelerationStructureSizes(accelDescriptor);
    const bool canRefit = mInstanceAccelerationStructure != nullptr &&
                          mTlasInstanceCount == mEmittedInstances.size() &&
                          mInstanceAccelerationStructure->size() >= sizes.accelerationStructureSize;

    if (canRefit)
    {
        ensureScratchBuffer(mTlasScratchBuffer, sizes.refitScratchBufferSize);

        MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
        MTL::AccelerationStructureCommandEncoder* commandEncoder = commandBuffer->accelerationStructureCommandEncoder();
        commandEncoder->refitAccelerationStructure(
            mInstanceAccelerationStructure, accelDescriptor, mInstanceAccelerationStructure, mTlasScratchBuffer, 0UL);
        commandEncoder->endEncoding();
        commandBuffer->commit();
    }
    else
    {
        if (mInstanceAccelerationStructure)
        {
            mInstanceAccelerationStructure->release();
            mInstanceAccelerationStructure = nullptr;
        }
        mInstanceAccelerationStructure = createAccelerationStructureNoCompact(accelDescriptor);
        mTlasInstanceCount = mEmittedInstances.size();
    }

    pPool->release();
}

void MetalRender::loadEnvMap(const std::string& texturePath)
{
    // Release previous env map resources to avoid leaks on reload
    if (mEnvMapTexture) { mEnvMapTexture->release(); mEnvMapTexture = nullptr; }
    if (mEnvAliasBuffer) { mEnvAliasBuffer->release(); mEnvAliasBuffer = nullptr; }
    mEnvMapLoaded = false;

    int width = 0, height = 0;
    float* pixelData = nullptr;
    bool isExr = false;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        int ret = LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env map: {} ({})", texturePath, err ? err : "unknown");
            if (err) FreeEXRErrorMessage(err);
            return;
        }
        isExr = true;
    }
    else
    {
        int channels = 0;
        pixelData = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixelData)
        {
            STRELKA_ERROR("Failed to load env map: {}", texturePath);
            return;
        }
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, width, height);

    // Create Metal texture (RGBA32Float)
    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(width);
    pTextureDesc->setHeight(height);
    pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    // TextureUsage flags, not ResourceUsage: the old value happened to set
    // ShaderRead's bit but also asked for RenderTarget. The PDF path now point-
    // reads this texture, so ShaderRead must be declared correctly.
    pTextureDesc->setUsage(MTL::TextureUsageShaderRead);

    mEnvMapTexture = mDevice->newTexture(pTextureDesc);
    pTextureDesc->release();

    const MTL::Region region = MTL::Region::Make3D(0, 0, 0, width, height, 1);
    mEnvMapTexture->replaceRegion(region, 0, pixelData, width * sizeof(float) * 4);

    // --- Build a flat alias table over every texel (Walker/Vose) --------------
    //
    // Replaces the previous 2D CDF (marginal over rows + conditional per row).
    // Sampling that needed two binary searches, ~21 dependent and scattered
    // loads into an 8 MB buffer for a 2K map, on every NEE sample at every
    // bounce. An alias table answers the same query with one 8-byte load.
    //
    // Texel weight is luminance times sin(theta) of the row (the equirectangular
    // solid-angle Jacobian) — unchanged from the CDF version, so the sampling
    // distribution itself is identical.
    const size_t texelCount = (size_t)width * (size_t)height;
    std::vector<double> weights(texelCount);
    double totalPower = 0.0;

    for (int y = 0; y < height; ++y)
    {
        const double v = ((double)y + 0.5) / (double)height;
        const double sinTheta = std::sin(v * M_PI);
        for (int x = 0; x < width; ++x)
        {
            const size_t i = (size_t)y * width + x;
            const float* px = pixelData + i * 4;
            const double lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            const double w = std::max(lum, 0.0) * sinTheta;
            weights[i] = w;
            totalPower += w;
        }
    }

    std::vector<EnvAliasEntry> alias(texelCount);
    if (totalPower > 0.0)
    {
        // Normalise so the mean probability is exactly 1; then every bucket is
        // either "under" (<1) or "over" (>=1) and they pair up.
        const double scale = (double)texelCount / totalPower;
        std::vector<double> p(texelCount);
        std::vector<uint32_t> small, large;
        small.reserve(texelCount / 2);
        large.reserve(texelCount / 2);
        for (size_t i = 0; i < texelCount; ++i)
        {
            p[i] = weights[i] * scale;
            (p[i] < 1.0 ? small : large).push_back((uint32_t)i);
        }

        while (!small.empty() && !large.empty())
        {
            const uint32_t l = small.back(); small.pop_back();
            const uint32_t g = large.back(); large.pop_back();

            alias[l].prob = (float)p[l];
            alias[l].alias = g;

            p[g] = (p[g] + p[l]) - 1.0;
            (p[g] < 1.0 ? small : large).push_back(g);
        }
        // Whatever is left is 1.0 up to rounding.
        for (const uint32_t i : large)  { alias[i].prob = 1.0f; alias[i].alias = i; }
        for (const uint32_t i : small)  { alias[i].prob = 1.0f; alias[i].alias = i; }
    }
    else
    {
        // Black environment: nothing to importance sample.
        for (size_t i = 0; i < texelCount; ++i)
        {
            alias[i].prob = 1.0f;
            alias[i].alias = (uint32_t)i;
        }
    }

    // pdf(texel) / dOmega(texel) reduces to lum * envPdfScale — see envTexelPdf().
    mEnvPdfScale = (totalPower > 0.0)
        ? (float)((double)texelCount / (2.0 * M_PI * M_PI * totalPower))
        : 0.0f;

    mEnvAliasBuffer = mDevice->newBuffer(
        alias.data(), alias.size() * sizeof(EnvAliasEntry), MTL::ResourceStorageModeManaged);

    // Free host pixel data
    if (isExr)
        free(pixelData);
    else
        stbi_image_free(pixelData);

    // Auto-calibrate env map intensity
    const float avgWeightedLum = (float)(totalPower / (double)(width * height));
    const float kCalibrationTarget = 1000.0f;
    mEnvMapAutoScale = (avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;
    mEnvMapLoaded = true;

    STRELKA_INFO("Env map alias table built: {} texels ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, autoScale: {:.1f}",
                 texelCount, alias.size() * sizeof(EnvAliasEntry) / (1024.0 * 1024.0),
                 totalPower, avgWeightedLum, mEnvMapAutoScale);
}
