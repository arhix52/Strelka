#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalRender.h"

#include <chrono>
#include <thread>
#include "MetalBuffer.h"
#include "texture_compress.h"

#include <fstream>

#include <algorithm>
#include <map>
#include <cassert>
#include <filesystem>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/task_info.h>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <log.h>
#include <paths.h>

#include <simd/simd.h>

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h> // IorStack: sized per path in the wavefront side table

using namespace oka;
namespace fs = std::filesystem;

// What the OS charges this process, which on unified memory includes everything
// the device allocated. `phys_footprint` is the number Activity Monitor shows;
// resident size is not, and undercounts GPU allocations badly.
static size_t processFootprintBytes()
{
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
    {
        return 0;
    }
    return (size_t)info.phys_footprint;
}


// Working state of a resumable acceleration structure build.
//
// Everything here used to be a local of createAccelerationStructures(). It has
// to survive between calls now that the build runs a slice at a time, and it is
// heap-allocated for the duration of one build rather than kept as members: the
// two maps hold an entry per scene instance, which is 1.1 million on the pine
// forest, and there is no reason to carry that between loads.
// Working state of a resumable material build.
//
// Same reason as AsBuildState below: these were locals, and they have to survive
// between calls now that the build runs a slice at a time. The texture cache in
// particular must, or a material in the second slice would re-decode a map the
// first slice already uploaded.
struct oka::MaterialBuildState
{
    std::vector<Material> gpuMaterials;
    // One texture per file, not per slot. A scene routinely uses the same map in
    // several materials -- the pine forest fills 83 slots from 56 files -- and
    // without this each slot decoded and uploaded its own copy, which cost 2.8 GB
    // there. Keyed on path and colour space together, because the same file can
    // legitimately be needed both sRGB-decoded and linear.
    std::unordered_map<std::string, MTL::Texture*> textureCache;
    size_t cursor = 0;
};

struct oka::AsBuildState
{
    // Instances that hang off the same node share a transform by construction.
    // The transform is part of the key because one node can carry a million
    // placements -- that is what EXT_mesh_gpu_instancing is -- and keying on the
    // node alone scatters every placement after the first into groups of its own.
    struct GroupKey
    {
        int node;
        int skeletal;
        uint64_t transform;
        bool operator<(const GroupKey& o) const
        {
            if (node != o.node)
                return node < o.node;
            if (skeletal != o.skeletal)
                return skeletal < o.skeletal;
            return transform < o.transform;
        }
    };

    enum class Phase : uint32_t
    {
        Meshes = 0,
        Grouping,
        Blas,
        Lights,
        Finish,
        Done,
    };
    Phase phase = Phase::Meshes;

    // Meshes
    size_t meshCursor = 0;
    size_t primitiveBytes = 0;

    // Grouping
    std::vector<int> instanceNode;
    std::map<GroupKey, size_t> groupOfKey;
    std::vector<std::vector<uint32_t>> groups;
    std::vector<bool> groupSkeletal;
    size_t groupCursor = 0;

    // One BLAS per distinct geometry, shared by every group that holds the same.
    std::map<std::vector<uint64_t>, size_t> blasOfSignature;
    std::vector<size_t> groupBlas;
    size_t blasCursor = 0;
    size_t sharedBlas = 0;
    size_t mergedGeometries = 0;

    // Light instances keep one BLAS per mesh, shared between lights, because
    // their userID must stay the light index.
    std::map<uint32_t, size_t> lightBlasOfMesh;
    size_t lightCursor = 0;

    // Per-phase wall clock, so the split between grouping and building is a
    // measurement rather than an assumption.
    double phaseMs[(size_t)Phase::Done] = {};
};


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

        if (mLastCommandBuffer)
        {
            mLastCommandBuffer->release();
            mLastCommandBuffer = nullptr;
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
        for (auto*& tex : mDisplayTextures)
            safeRelease(tex);
        for (auto*& tex : mUpscaleTextures)
            safeRelease(tex);
        mMetalFx.release();

        // Buffers
        safeRelease(mAccumulationBuffer);
        safeRelease(mLightBuffer);
        safeRelease(mVertexBuffer);
        safeRelease(mIndexBuffer);
        safeRelease(mInstanceBuffer);
        safeRelease(mMaterialBuffer);
        safeRelease(mSkinDataBuffer);
        safeRelease(mJointMatricesBuffer);
        // Shared with mVertexBuffer when nothing in the scene deforms; releasing
        // it through both paths would be a double free.
        if (mOwnsPrevVertexBuffer)
            safeRelease(mPrevVertexBuffer);
        else
            mPrevVertexBuffer = nullptr;
        safeRelease(mPrevFrameVertexBuffer);
        safeRelease(mPrevFrameInstanceBuffer);
        safeRelease(mGeometryEntryBuffer);
        safeRelease(mTlasScratchBuffer);
        for (auto*& buf : mUniformBuffers) safeRelease(buf);
        for (auto*& buf : mUniformTMBuffers) safeRelease(buf);

        // Environment map
        safeRelease(mEnvMapTexture);
        safeRelease(mEnvBackgroundTexture);
        safeRelease(mSharcBuffer);
        safeRelease(mEnvAliasBuffer);

        // Pipeline states
        safeRelease(mWavefrontResolvePSO);
        safeRelease(mWavefrontPreparePSO);
        safeRelease(mWavefrontPrepareShadowPSO);
        safeRelease(mPathStateBuffer);
        safeRelease(mPathRayBuffer);
        safeRelease(mHitBuffer);
        safeRelease(mIorStackBuffer);
        safeRelease(mRadianceBuffer);
        safeRelease(mGuideRadianceBuffer);
        safeRelease(mPathQueueBuffer[0]);
        safeRelease(mPathQueueBuffer[1]);
        safeRelease(mWavefrontControlBuffer);
        safeRelease(mShadowRayBuffer);
        safeRelease(mHitQueueBuffer);
        safeRelease(mAovBuffer);
        safeRelease(mMissQueueBuffer);
        for (auto& kv : mWavefrontVariants)
        {
            safeRelease(kv.second.generate);
            safeRelease(kv.second.extendMotion);
            safeRelease(kv.second.extendStatic);
            safeRelease(kv.second.shade);
            safeRelease(kv.second.miss);
            safeRelease(kv.second.shadowMotion);
            safeRelease(kv.second.shadowStatic);
            safeRelease(kv.second.shadowTableMotion);
            safeRelease(kv.second.shadowTableStatic);
            safeRelease(kv.second.sharcDeposit);
        }
        mWavefrontVariants.clear();
        safeRelease(mWavefrontLibrary);
        safeRelease(mWavefrontPrepareHitMissPSO);
        safeRelease(mWavefrontResolvePSO4);
        safeRelease(mWavefrontPreparePSO4);
        safeRelease(mWavefrontPrepareShadowPSO4);
        safeRelease(mWavefrontPrepareHitMissPSO4);
        safeRelease(mAovResolvePSO);
        safeRelease(mAovResolvePSO4);
        releaseGuideTextures();
        safeRelease(mStageTimestampBuffer);
        safeRelease(mStageStatsBuffer);
        safeRelease(mTonemapperPSO);
        safeRelease(mTonemapperPSO4);
        safeRelease(mTonemapperTexPSO);
        safeRelease(mSkinningPSO);
        safeRelease(mTriangleUpdatePSO);
        safeRelease(mSkinningPSO4);
        safeRelease(mTriangleUpdatePSO4);

        // A build abandoned mid-slice -- the window was closed while the scene
        // was still coming up -- still owns its working state.
        delete mAsBuild;
        mAsBuild = nullptr;
        delete mMaterialBuild;
        mMaterialBuild = nullptr;

        // Queue & device (release last)
        mMetal4.release();
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


// The texture the tonemapper writes and the display reads. One per async slot,
// matching the output buffers, so a frame being shown is never the one being
// written.

// The reduced-resolution target the tracer renders into when upscaling.
//
// Usage flags come from the scaler rather than being guessed: MetalFX validates
// them at encode time, and a texture allocated without what it wants fails there
// rather than at creation.

// Halton, base 2 and 3, centred on the pixel. A temporal upscaler needs the
// sample positions to cover the pixel evenly over a handful of frames and to know
// where each one was; a random offset would do the first but not the second.
static float haltonAt(uint64_t index, uint32_t base)
{
    float result = 0.0f;
    float f = 1.0f / (float)base;
    while (index > 0)
    {
        result += f * (float)(index % base);
        index /= base;
        f /= (float)base;
    }
    return result;
}

void MetalRender::frameJitter(uint64_t frameIndex, uint32_t phaseCount, float& x, float& y) const
{
    // Wrapped to a phase count tied to the upscale ratio, the way every temporal
    // upscaler does it: the sequence has to close over the output pixel grid, and
    // an unbounded Halton index keeps drifting to positions the reconstruction
    // has no use for. 8 * (out/in)^2 is the usual figure.
    //
    // Index from 1: Halton's first entry is 0, which would leave one frame in each
    // cycle unjittered and bias the sequence.
    const uint64_t phase = frameIndex % std::max<uint32_t>(phaseCount, 1u);
    x = haltonAt(phase + 1, 2) - 0.5f;
    y = haltonAt(phase + 1, 3) - 0.5f;
}

// Guides live at render resolution, the denoised result at display resolution.
// Usage flags come from the denoiser for the same reason they do for the spatial
// scaler: it validates them and a guess fails at encode time.
void MetalRender::releaseGuideTextures()
{
    auto release = [](MTL::Texture*& texture) {
        if (texture)
        {
            texture->release();
            texture = nullptr;
        }
    };
    release(mGuides.color);
    release(mGuides.depth);
    release(mGuides.motion);
    release(mGuides.diffuse);
    release(mGuides.specular);
    release(mGuides.normal);
    release(mGuides.roughness);
    release(mGuides.specularHitDistance);
    release(mGuides.reactive);
    release(mDenoisedTexture);
    mGuideWidth = 0;
    mGuideHeight = 0;
    mGuideOutWidth = 0;
    mGuideOutHeight = 0;
}

void MetalRender::ensureGuideTextures(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight)
{
    // The output size is part of the key, not just the render size. Miss it and a
    // resize that leaves the render resolution rounding to the same value keeps a
    // stale output texture: the denoiser, which *is* rebuilt, then writes its
    // corner of a texture sized for a different frame and the rest of the picture
    // stays black.
    if (width == mGuideWidth && height == mGuideHeight && outWidth == mGuideOutWidth &&
        outHeight == mGuideOutHeight && mGuides.color && mDenoisedTexture)
    {
        return;
    }
    releaseGuideTextures();

    // Usage comes from whichever scaler will read these, not always from the
    // denoiser: the temporal scaler reads colour, depth and motion too, and the
    // two need not want the same flags. Asking the denoiser while only the
    // temporal scaler exists worked by accident -- the accessor falls back to a
    // plain ShaderRead when there is no denoiser to ask.
    const bool temporalOnly = !mMetalFx.hasDenoiser() && mMetalFx.hasTemporalScaler();
    const MTL::TextureUsage guideUsage =
        MTL::TextureUsageShaderWrite |
        (temporalOnly ? (mMetalFx.temporalDepthUsage() | mMetalFx.temporalMotionUsage())
                      : mMetalFx.denoiseGuideUsage());
    auto make = [&](MTL::PixelFormat fmt, uint32_t w, uint32_t h, MTL::TextureUsage usage) {
        MTL::TextureDescriptor* d = MTL::TextureDescriptor::alloc()->init();
        d->setWidth(w);
        d->setHeight(h);
        d->setPixelFormat(fmt);
        d->setTextureType(MTL::TextureType2D);
        d->setStorageMode(MTL::StorageModePrivate);
        d->setUsage(usage);
        MTL::Texture* t = mDevice->newTexture(d);
        d->release();
        return t;
    };
    // Formats must match the descriptor in MetalFxContext exactly.
    mGuides.color = make(MTL::PixelFormatRGBA16Float, width, height,
                         MTL::TextureUsageShaderWrite |
                             (temporalOnly ? mMetalFx.temporalColorUsage() : mMetalFx.denoiseColorUsage()));
    mGuides.depth = make(MTL::PixelFormatR32Float, width, height, guideUsage);
    mGuides.motion = make(MTL::PixelFormatRG16Float, width, height, guideUsage);
    mGuides.diffuse = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.specular = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.normal = make(MTL::PixelFormatRGBA16Float, width, height, guideUsage);
    mGuides.roughness = make(MTL::PixelFormatR16Float, width, height, guideUsage);
    mGuides.specularHitDistance = make(MTL::PixelFormatR16Float, width, height, guideUsage);
    mGuides.reactive = make(MTL::PixelFormatR8Unorm, width, height, guideUsage);
    mDenoisedTexture = make(MTL::PixelFormatRGBA16Float, outWidth, outHeight,
                            MTL::TextureUsageShaderRead |
                                (temporalOnly ? mMetalFx.temporalOutputUsage()
                                              : mMetalFx.denoiseOutputUsage()));

    mGuideWidth = width;
    mGuideHeight = height;
    mGuideOutWidth = outWidth;
    mGuideOutHeight = outHeight;
    mMetal4ResidencyGeneration = 0;
    // Fresh textures hold nothing, and a scaler built for new dimensions has no
    // history either.
    mResetDenoiseHistory = true;
}

// Hand MetalFX the exposure the renderer is actually working at.
//
// The alternative, autoExposureEnabled, estimates it from the pixels every frame,
// and on a frame where nothing moves that estimate drifts -- measured at 4.5x too
void MetalRender::ensureUpscaleTextures(uint32_t width, uint32_t height)
{
    if (width == mUpscaleTextureWidth && height == mUpscaleTextureHeight && mUpscaleTextures[0])
    {
        return;
    }
    for (MTL::Texture*& tex : mUpscaleTextures)
    {
        if (tex)
        {
            tex->release();
            tex = nullptr;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageShaderWrite | mMetalFx.requiredColorUsage());
    for (MTL::Texture*& tex : mUpscaleTextures)
    {
        tex = mDevice->newTexture(desc);
    }
    desc->release();

    mUpscaleTextureWidth = width;
    mUpscaleTextureHeight = height;
    mMetal4ResidencyGeneration = 0;
    // Whatever the display was about to show lived in a texture that no longer
    // exists; the one now at that index has never been written.
    mReadyIndex.store(-1);
}

// Snapshot the pose the previous frame was rendered with.
//
// Called at the top of a frame, before skinning rewrites the vertices and before
// the animation block re-uploads the instance transforms -- so what is captured
// is what the last frame actually used. Doing it at the end of a frame instead
// would have to be ordered against work already in flight for no benefit.
void MetalRender::capturePrevFramePose()
{
    if (!mInstanceBuffer)
    {
        return;
    }
    // Instance transforms are always worth keeping: a node animation moves rigid
    // geometry without touching a single vertex. Skipping the copy for a scene
    // with no animation and no skinning was tried on a two-million-instance
    // forest, where it is 143 MB a sample: it measured as nothing against a
    // 0.49 s sample, and it costs the editor a correct motion vector on the
    // frame after something is dragged.
    if (!mPrevFrameInstanceBuffer || mPrevFrameInstanceBuffer->length() != mInstanceBuffer->length())
    {
        if (mPrevFrameInstanceBuffer)
        {
            mPrevFrameInstanceBuffer->release();
        }
        mPrevFrameInstanceBuffer =
            mDevice->newBuffer(mInstanceBuffer->length(), MTL::ResourceStorageModeManaged);
        mHasPrevFramePose = false; // a new allocation holds nothing
    }
    std::memcpy(mPrevFrameInstanceBuffer->contents(), mInstanceBuffer->contents(),
                mInstanceBuffer->length());
    mPrevFrameInstanceBuffer->didModifyRange(NS::Range::Make(0, mPrevFrameInstanceBuffer->length()));

    // Vertices only when something can actually rewrite them. With no skinning in
    // the scene the current buffer *is* the previous one, and the shader is given
    // it directly rather than a copy that could never differ.
    const bool deforming = mSkinDataBuffer != nullptr && mVertexBuffer != nullptr;
    if (deforming)
    {
        if (!mPrevFrameVertexBuffer || mPrevFrameVertexBuffer->length() != mVertexBuffer->length())
        {
            if (mPrevFrameVertexBuffer)
            {
                mPrevFrameVertexBuffer->release();
            }
            mPrevFrameVertexBuffer =
                mDevice->newBuffer(mVertexBuffer->length(), MTL::ResourceStorageModePrivate);
            mHasPrevFramePose = false;
        }
        MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        blit->copyFromBuffer(mVertexBuffer, 0, mPrevFrameVertexBuffer, 0, mVertexBuffer->length());
        blit->endEncoding();
        cmd->commit();
    }
}

MTL::Texture* MetalRender::tonemapTarget(bool upscaling) const
{
    return upscaling ? mUpscaleTextures[mWriteIndex] : mDisplayTextures[mWriteIndex];
}

void MetalRender::ensureDisplayTextures(uint32_t width, uint32_t height)
{
    // The usage MetalFX demands is only known once a scaler exists, so turning
    // upscaling on at runtime changes what these textures need. Recreate them
    // when it does, or the first upscaled frame fails validation.
    const MTL::TextureUsage usage =
        MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | mMetalFx.requiredOutputUsage();
    if (width == mDisplayTextureWidth && height == mDisplayTextureHeight && mDisplayTextures[0] &&
        usage == mDisplayTextureUsage)
    {
        return;
    }
    mDisplayTextureUsage = usage;
    for (MTL::Texture*& tex : mDisplayTextures)
    {
        if (tex)
        {
            tex->release();
            tex = nullptr;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    // Half float: the display image is post-tonemap and never exceeds the range
    // a half can hold, and MetalFX takes this format directly.
    desc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(usage);
    for (MTL::Texture*& tex : mDisplayTextures)
    {
        tex = mDevice->newTexture(desc);
    }
    desc->release();

    mDisplayTextureWidth = width;
    mDisplayTextureHeight = height;
    mMetal4ResidencyGeneration = 0; // new allocations: residency has to be redone
    // Both textures were just replaced, and the index still pointed into the old
    // pair. Presenting from it hands the display a texture with no defined
    // contents -- which is what the flash of noise on switching modes was: the
    // usage MetalFX requires changes when upscaling turns on, so a switch
    // reallocates these even when the size does not change.
    mReadyIndex.store(-1);
}


// Half to float by hand: the alternative is pulling in a conversion library for
// a readback path that only debug and validation code takes.
static float halfToFloat(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int32_t exponent = (h >> 10) & 0x1F;
    const uint32_t mantissa = h & 0x3FF;
    uint32_t bits;
    if (exponent == 0)
    {
        bits = sign; // zero or subnormal, close enough for a preview
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7F800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((uint32_t)(exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Read the display texture back to the CPU.
//
// This is what the screen shows -- after tonemapping, after MetalFX -- which the
// output buffer no longer is. Every effect on the MetalFX plan is invisible
// without it, and "no validation error" has already been shown this session not
// to mean "correct image".
bool MetalRender::readDisplayTexture(std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    const int ri = mReadyIndex.load();
    if (ri < 0 || !mDisplayTextures[ri])
    {
        return false;
    }
    MTL::Texture* tex = mDisplayTextures[ri];
    width = (uint32_t)tex->width();
    height = (uint32_t)tex->height();

    // Half float on the GPU, so the staging buffer is 8 bytes a pixel.
    const size_t packedRowBytes = (size_t)width * 8;
    const size_t rowBytes = (packedRowBytes + 255u) & ~size_t(255u);
    MTL::Buffer* staging = mDevice->newBuffer(rowBytes * height, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(tex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), staging, 0, rowBytes,
                          rowBytes * height);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    rgba.resize((size_t)width * height * 4);
    for (size_t y = 0; y < height; ++y)
    {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(
            static_cast<const uint8_t*>(staging->contents()) + y * rowBytes);
        for (size_t x = 0; x < (size_t)width * 4; ++x)
        {
            rgba[(y * width * 4) + x] = halfToFloat(src[x]);
        }
    }
    staging->release();
    return true;
}

// Read any of the denoiser's textures back as RGBA floats.
//
// The point of reading these rather than the finished frame is that a guide can
// be wrong in a way the picture does not obviously show -- a motion vector that
// ignores a moving limb looks like slightly soft shading until you difference it
// against the truth. Unused channels come back as zero so one checker can walk
// every guide.

// Bounding-box diagonal of the skinned vertices, straight off the GPU.
float MetalRender::skinnedGeometryExtent()
{
    size_t first = SIZE_MAX, last = 0;
    for (const auto& m : mScene->getMeshes())
    {
        if (!m.isSkeletal)
            continue;
        first = std::min(first, (size_t)m.mVbOffset);
        last = std::max(last, (size_t)m.mVbOffset + (size_t)m.mVertexCount);
    }
    if (first == SIZE_MAX || !mVertexBuffer)
    {
        return -1.0f;
    }
    const size_t bytes = std::min(mVertexBuffer->length(), last * 32);
    if (bytes <= first * 32)
    {
        return -1.0f;
    }
    MTL::Buffer* staging = mDevice->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(mVertexBuffer, 0, staging, 0, bytes);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    glm::float3 lo(1e30f), hi(-1e30f);
    const char* base = static_cast<const char*>(staging->contents());
    for (size_t v = first * 32; v + 32 <= bytes; v += 32)
    {
        float p[3];
        std::memcpy(p, base + v, sizeof(p));
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
        {
            staging->release();
            return -2.0f; // non-finite: a different fault from a collapse
        }
        lo = glm::min(lo, glm::float3(p[0], p[1], p[2]));
        hi = glm::max(hi, glm::float3(p[0], p[1], p[2]));
    }
    staging->release();
    return glm::length(hi - lo);
}

bool MetalRender::readGuideTexture(Guide guide, std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    MTL::Texture* tex = nullptr;
    switch (guide)
    {
    case Guide::Color:          tex = mGuides.color; break;
    case Guide::Depth:          tex = mGuides.depth; break;
    case Guide::Motion:         tex = mGuides.motion; break;
    case Guide::DiffuseAlbedo:  tex = mGuides.diffuse; break;
    case Guide::SpecularAlbedo: tex = mGuides.specular; break;
    case Guide::Normal:         tex = mGuides.normal; break;
    case Guide::Roughness:      tex = mGuides.roughness; break;
    case Guide::SpecularHitDistance: tex = mGuides.specularHitDistance; break;
    case Guide::Reactive:       tex = mGuides.reactive; break;
    case Guide::Denoised:       tex = mDenoisedTexture; break;
    default: return false;
    }
    if (!tex)
    {
        return false;
    }
    width = (uint32_t)tex->width();
    height = (uint32_t)tex->height();

    uint32_t channels = 0;
    uint32_t bytesPerChannel = 0;
    bool isHalf = true;
    bool isUnorm8 = false;
    switch (tex->pixelFormat())
    {
    case MTL::PixelFormatRGBA16Float: channels = 4; bytesPerChannel = 2; break;
    case MTL::PixelFormatRG16Float:   channels = 2; bytesPerChannel = 2; break;
    case MTL::PixelFormatR16Float:    channels = 1; bytesPerChannel = 2; break;
    case MTL::PixelFormatR32Float:    channels = 1; bytesPerChannel = 4; isHalf = false; break;
    case MTL::PixelFormatR8Unorm:
        channels = 1;
        bytesPerChannel = 1;
        isHalf = false;
        isUnorm8 = true;
        break;
    default: return false;
    }

    const size_t packedRowBytes =
        (size_t)width * channels * bytesPerChannel;
    const size_t rowBytes = (packedRowBytes + 255u) & ~size_t(255u);
    MTL::Buffer* staging = mDevice->newBuffer(rowBytes * height, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(tex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), staging, 0, rowBytes,
                          rowBytes * height);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    rgba.assign((size_t)width * height * 4, 0.0f);
    for (size_t y = 0; y < height; ++y)
    {
        const uint8_t* row =
            static_cast<const uint8_t*>(staging->contents()) + y * rowBytes;
        for (size_t x = 0; x < width; ++x)
        {
            const size_t t = y * width + x;
            for (uint32_t c = 0; c < channels; ++c)
            {
                const size_t si = x * channels + c;
                if (isHalf)
                {
                    rgba[t * 4 + c] =
                        halfToFloat(reinterpret_cast<const uint16_t*>(row)[si]);
                }
                else if (isUnorm8)
                {
                    rgba[t * 4 + c] = row[si] / 255.0f;
                }
                else
                {
                    rgba[t * 4 + c] =
                        reinterpret_cast<const float*>(row)[si];
                }
            }
        }
    }
    staging->release();
    return true;
}

void* MetalRender::getReadyTexture()
{
    const int ri = mReadyIndex.load();
    if (ri < 0)
    {
        return nullptr;
    }
    return mDisplayTextures[ri];
}

Buffer* MetalRender::getReadyBuffer()
{
    int ri = mReadyIndex.load();
    if (ri < 0)
        return nullptr;
    return mAsyncOutputBuffers[ri];
}


// Where the memory went, measured from the objects themselves.
//
// Deliberately not a running tally kept at allocation sites: those drift the
// moment someone adds a buffer and forgets the counter, and the first symptom is
// a total that no longer matches the device's. Walking the members costs a few
// microseconds and cannot be wrong about anything it looks at -- and what it
// fails to look at shows up as the unaccounted remainder rather than vanishing.
bool MetalRender::memoryReport(MemoryReport& report) const
{
    report.gpu.clear();
    report.cpu.clear();

    auto add = [&report](const char* name, size_t bytes) {
        if (bytes > 0)
        {
            report.gpu.push_back({ name, bytes });
        }
    };
    auto bufBytes = [](MTL::Buffer* b) { return b ? b->length() : 0; };
    auto texBytes = [](MTL::Texture* t) { return t ? t->allocatedSize() : 0; };

    add("Vertices", bufBytes(mVertexBuffer));
    add("Indices", bufBytes(mIndexBuffer));
    // A second copy of every vertex, for motion blur and for the denoiser's
    // reprojection. Shared with the current one on a scene with nothing skinned,
    // in which case this reports zero rather than double-counting.
    add("Vertices (previous)",
        bufBytes(mPrevVertexBuffer != mVertexBuffer ? mPrevVertexBuffer : nullptr) +
            bufBytes(mPrevFrameVertexBuffer));

    {
        size_t bytes = 0;
        for (MTL::Texture* t : mMaterialTextures)
        {
            bytes += texBytes(t);
        }
        add("Textures", bytes);
    }
    add("Environment", texBytes(mEnvMapTexture) + texBytes(mEnvBackgroundTexture) + bufBytes(mEnvAliasBuffer));

    {
        size_t as = 0, scratch = 0;
        for (const Blas& b : mBlasList)
        {
            as += b.mAs ? b.mAs->size() : 0;
            scratch += bufBytes(b.mScratch);
        }
        add("BLAS", as);
        // Kept for the lifetime of the structure so a refit needs no allocation.
        add("BLAS scratch", scratch);
    }
    add("TLAS", mInstanceAccelerationStructure ? mInstanceAccelerationStructure->size() : 0);
    add("TLAS scratch", bufBytes(mTlasScratchBuffer));
    add("Instance descriptors", bufBytes(mInstanceBuffer) + bufBytes(mPrevFrameInstanceBuffer));
    add("Geometry table", bufBytes(mGeometryEntryBuffer));
    add("Materials", bufBytes(mMaterialBuffer));
    add("Lights", bufBytes(mLightBuffer));
    add("Skinning", bufBytes(mSkinDataBuffer) + bufBytes(mJointMatricesBuffer));

    // The per-path side tables: one entry per pixel per stage, so they scale with
    // resolution rather than with the scene, and are the reason a render at
    // native resolution costs more than the upscaled one before a ray is cast.
    add("Wavefront queues",
        bufBytes(mPathStateBuffer) + bufBytes(mPathRayBuffer) + bufBytes(mHitBuffer) +
            bufBytes(mIorStackBuffer) + bufBytes(mRadianceBuffer) + bufBytes(mGuideRadianceBuffer) +
            bufBytes(mPathQueueBuffer[0]) + bufBytes(mPathQueueBuffer[1]) + bufBytes(mWavefrontControlBuffer) +
            bufBytes(mShadowRayBuffer) + bufBytes(mHitQueueBuffer) + bufBytes(mMissQueueBuffer) +
            bufBytes(mAovBuffer) + bufBytes(mStageStatsBuffer));
    add("Radiance cache", bufBytes(mSharcBuffer));
    add("Accumulation", bufBytes(mAccumulationBuffer));

    {
        size_t bytes = texBytes(mDisplayTextures[0]) + texBytes(mDisplayTextures[1]) +
                       texBytes(mUpscaleTextures[0]) + texBytes(mUpscaleTextures[1]) +
                       texBytes(mDenoisedTexture);
        bytes += texBytes(mGuides.color) + texBytes(mGuides.depth) + texBytes(mGuides.motion) +
                 texBytes(mGuides.diffuse) + texBytes(mGuides.specular) + texBytes(mGuides.normal) +
                 texBytes(mGuides.roughness) + texBytes(mGuides.specularHitDistance) +
                 texBytes(mGuides.reactive);
        add("Display & guides", bytes);
    }

    {
        size_t bytes = 0;
        for (MTL::Buffer* b : mUniformBuffers)
        {
            bytes += bufBytes(b);
        }
        for (MTL::Buffer* b : mUniformTMBuffers)
        {
            bytes += bufBytes(b);
        }
        for (MTL::Buffer* b : mAsGroupScratch)
        {
            bytes += bufBytes(b);
        }
        for (const Mesh* m : mMetalMeshes)
        {
            bytes += m ? bufBytes(m->mPerPrimitiveBuffer) : 0;
        }
        add("Uniforms & misc", bytes);
    }

    // The host arrays the editor keeps so Scene::pick() can walk them. A full
    // duplicate of the two largest GPU buffers, which is why it is worth naming.
    if (mScene && !mScene->hostGeometryReleased())
    {
        report.cpu.push_back({ "Host geometry (picking)", mHostGeometryBytes.first + mHostGeometryBytes.second });
    }

    report.deviceAllocated = mDevice ? mDevice->currentAllocatedSize() : 0;
    report.processFootprint = processFootprintBytes();
    return true;
}

void MetalRender::init()
{
    static_assert(sizeof(PathRay) == 24, "PathRay is what `extend` streams per path; keep it minimal");
    // 48 rather than 20: the radiance cache adds a slot index, the pixel's
    // radiance at the moment the path passed through it, and the reciprocal
    // throughput there. Read and written for every live path on every bounce, so
    // worth watching.
    static_assert(sizeof(PathState) == 52, "PathState is read and written for every live path on every bounce");
    // 32 rather than 24: the hit now carries the TLAS instance, because a shared
    // BLAS belongs to no single one. One extra word per live path.
    static_assert(sizeof(HitRecord) == 32, "HitRecord size changed");
    static_assert(sizeof(GeometryEntry) == 12, "GeometryEntry size changed");
    static_assert(sizeof(AovSample) == 64,
                  "AovSample is written once per pixel per frame; keep an eye on the size");

    mDevice = MTL::CreateSystemDefaultDevice();
    if (!mDevice)
    {
        STRELKA_FATAL("Failed to create Metal device");
        return;
    }
    mCommandQueue = mDevice->newCommandQueue();
    // 64 KB of constants per frame is far more than the tracer's handful of
    // small values needs; the ring is cheap and running out is a hard error.
    // STRELKA_NO_MTL4 keeps the layer out of the process entirely, which is what
    // isolates it when a measurement looks wrong.
    if (!getenv("STRELKA_NO_MTL4"))
    {
        mMetal4.init(mDevice, (uint32_t)kMaxFramesInFlight, 64 * 1024);
    }
    buildTonemapperPipeline();
    buildWavefrontPipelines();
    // Arm the deferred scene build. The first render() call picks it up a stage
    // at a time; a synchronous caller drives it to the end through renderSync.
    mBuildStage = BuildStage::Buffers;
}

namespace
{
// Bumped whenever the cache's layout or the encoder changes, so an old cache is
// ignored rather than misread.
constexpr uint32_t kTextureCacheVersion = 1;

struct CachedTextureHeader
{
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t pixelFormat;
    uint32_t blockBytes; // 0 when the payload is uncompressed RGBA8
    uint32_t reserved;
};
} // namespace

// A stable name for a texture as it will be uploaded: the file it came from, when
// that file last changed, and every setting that alters the result.
std::string MetalRender::textureCacheKey(const std::string& fileName, bool srgb, TextureKind kind) const
{
    std::error_code ec;
    const auto size = fs::file_size(fileName, ec);
    const auto stamp = fs::last_write_time(fileName, ec).time_since_epoch().count();
    const uint32_t maxDim = const_cast<MetalRender*>(this)->getSettings()->getAs<uint32_t>("render/texture/maxDimension");
    const uint32_t divisor = const_cast<MetalRender*>(this)->getSettings()->getAs<uint32_t>("render/texture/downscale");

    std::string blob = fileName;
    blob += "|" + std::to_string((unsigned long long)size);
    blob += "|" + std::to_string((long long)stamp);
    blob += "|" + std::to_string(maxDim) + "|" + std::to_string(divisor);
    blob += srgb ? "|srgb" : "|linear";
    blob += "|" + std::to_string((int)kind);
    blob += "|v" + std::to_string(kTextureCacheVersion);

    // FNV-1a. The key never leaves this machine and never has to resist anything,
    // so a short hash that is cheap to compute is the right one.
    uint64_t hash = 1469598103934665603ull;
    for (const char c : blob)
    {
        hash ^= (uint64_t)(unsigned char)c;
        hash *= 1099511628211ull;
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.btex", (unsigned long long)hash);
    return name;
}

MTL::Texture* MetalRender::loadCachedTexture(const std::string& cachePath)
{
    std::ifstream in(cachePath, std::ios::binary);
    if (!in)
        return nullptr;

    CachedTextureHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "BTEX", 4) != 0 || header.version != kTextureCacheVersion)
        return nullptr;

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(header.width);
    desc->setHeight(header.height);
    desc->setMipmapLevelCount(header.levels);
    desc->setPixelFormat((MTL::PixelFormat)header.pixelFormat);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeManaged);
    desc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);
    MTL::Texture* texture = mDevice->newTexture(desc);
    desc->release();
    if (!texture)
        return nullptr;

    std::vector<uint8_t> level;
    for (uint32_t l = 0; l < header.levels; ++l)
    {
        uint32_t byteLength = 0;
        in.read(reinterpret_cast<char*>(&byteLength), sizeof(byteLength));
        if (!in || byteLength == 0)
        {
            texture->release();
            return nullptr;
        }
        level.resize(byteLength);
        in.read(reinterpret_cast<char*>(level.data()), byteLength);
        if (!in)
        {
            texture->release();
            return nullptr;
        }
        const uint32_t w = std::max(1u, header.width >> l);
        const uint32_t h = std::max(1u, header.height >> l);
        const size_t rowBytes = header.blockBytes
                                    ? (size_t)((w + 3) / 4) * header.blockBytes
                                    : (size_t)w * 4;
        texture->replaceRegion(MTL::Region::Make3D(0, 0, 0, w, h, 1), l, level.data(), rowBytes);
    }
    return texture;
}

MTL::Texture* MetalRender::loadTextureFromFile(const std::string& fileName, bool srgb, TextureKind kind)
{
    // The cache holds the finished article: downscaled, mipped and compressed.
    // A hit skips the PNG decode, the resample, the mip chain and the encode --
    // on the pine forest that is 37 seconds of blit and most of the decode time,
    // every launch, for textures that never change.
    const fs::path cacheDir = getSettings()->getAs<std::string>("render/texture/cachePath");
    std::string cacheFile;
    if (!cacheDir.empty())
    {
        cacheFile = (cacheDir / textureCacheKey(fileName, srgb, kind)).string();
        if (MTL::Texture* cached = loadCachedTexture(cacheFile))
        {
            ++mTextureCacheHits;
            return cached;
        }
    }

    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (data == nullptr)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return nullptr;
    }

    // Optional downscale on load. A forest of 4K maps is 3.4 GB decoded, and
    // block compression -- the right answer -- needs an encoder this project does
    // not vendor. Halving the longest side is a 4x cut per step, it is a
    // deliberate quality trade the user asks for, and 0 disables it.
    const uint32_t maxDim = getSettings()->getAs<uint32_t>("render/texture/maxDimension");
    // A plain divisor as well as a ceiling: halving everything is the quickest
    // way to see whether a scene fits at all, and it keeps relative detail
    // between a 4K map and a 512 one, which a ceiling does not.
    const uint32_t divisor = std::max(1u, getSettings()->getAs<uint32_t>("render/texture/downscale"));
    if ((maxDim > 0 && (uint32_t)std::max(texWidth, texHeight) > maxDim) || divisor > 1)
    {
        int dstW = std::max(1, texWidth / (int)divisor);
        int dstH = std::max(1, texHeight / (int)divisor);
        while (maxDim > 0 && (uint32_t)std::max(dstW, dstH) > maxDim && dstW > 1 && dstH > 1)
        {
            dstW = std::max(1, dstW / 2);
            dstH = std::max(1, dstH / 2);
        }
        auto* scaled = (stbi_uc*)malloc((size_t)dstW * dstH * 4);
        // Colour maps must be resampled through the transfer function, or the
        // average of two sRGB bytes is not the sRGB of their average and every
        // downscale darkens.
        const int ok = scaled ? (srgb ? stbir_resize_uint8_srgb(data, texWidth, texHeight, 0, scaled,
                                                                dstW, dstH, 0, 4, 3, 0)
                                      : stbir_resize_uint8(data, texWidth, texHeight, 0, scaled,
                                                           dstW, dstH, 0, 4))
                              : 0;
        if (ok)
        {
            stbi_image_free(data);
            data = scaled;
            texWidth = dstW;
            texHeight = dstH;
        }
        else if (scaled)
        {
            free(scaled);
        }
    }

    uint32_t levels = 1;
    while ((1u << levels) <= (uint32_t)std::max(texWidth, texHeight))
        ++levels;

    // The mip chain is built here rather than by a blit pass, because a
    // compressed texture cannot be mipped on the GPU -- generateMipmaps does not
    // accept a block format -- and because building it on the CPU is what lets
    // the result be cached at all.
    std::vector<std::vector<uint8_t>> chain;
    chain.reserve(levels);
    chain.emplace_back(data, data + (size_t)texWidth * texHeight * 4);
    stbi_image_free(data);
    for (uint32_t l = 1; l < levels; ++l)
    {
        const int prevW = std::max(1, texWidth >> (l - 1));
        const int prevH = std::max(1, texHeight >> (l - 1));
        const int w = std::max(1, texWidth >> l);
        const int h = std::max(1, texHeight >> l);
        std::vector<uint8_t> next((size_t)w * h * 4);
        // Colour goes through the transfer function, for the same reason the
        // downscale above does: the average of two sRGB bytes is not the sRGB of
        // their average, and every level would come out darker than the last.
        const int ok = srgb ? stbir_resize_uint8_srgb(chain[l - 1].data(), prevW, prevH, 0,
                                                      next.data(), w, h, 0, 4, 3, 0)
                            : stbir_resize_uint8(chain[l - 1].data(), prevW, prevH, 0,
                                                 next.data(), w, h, 0, 4);
        if (!ok)
        {
            levels = l;
            break;
        }
        chain.push_back(std::move(next));
    }

    const bool canCompress = kind != TextureKind::Normal && mDevice->supportsBCTextureCompression() &&
                             getSettings()->getAs<bool>("render/texture/compress");
    const bool withAlpha = canCompress && oka::bc::hasAlpha(chain[0].data(), texWidth, texHeight);
    MTL::PixelFormat format;
    if (!canCompress)
        format = srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm;
    else if (withAlpha)
        format = srgb ? MTL::PixelFormatBC3_RGBA_sRGB : MTL::PixelFormatBC3_RGBA;
    else
        format = srgb ? MTL::PixelFormatBC1_RGBA_sRGB : MTL::PixelFormatBC1_RGBA;

    std::vector<std::vector<uint8_t>> payload;
    payload.reserve(levels);
    for (uint32_t l = 0; l < levels; ++l)
    {
        const int w = std::max(1, texWidth >> l);
        const int h = std::max(1, texHeight >> l);
        payload.push_back(canCompress ? oka::bc::compressImage(chain[l].data(), w, h, withAlpha)
                                      : std::move(chain[l]));
    }
    chain.clear();
    chain.shrink_to_fit();

    MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
    pTextureDesc->setWidth(texWidth);
    pTextureDesc->setHeight(texHeight);
    // Mipmaps buy nothing on their own in a path tracer -- a compute kernel has
    // no implicit derivatives, so sample() reads level 0 until something computes
    // an explicit LOD. They are here so the LOD path has something to read; see
    // the ray-cone estimate in shading_common.h.
    pTextureDesc->setMipmapLevelCount(levels);
    // Colour maps carry sRGB-encoded bytes. Loading them as a linear format
    // hands the encoded values straight to the BSDF, which lifts every midtone
    // and desaturates the result; the _sRGB format makes the sampler decode.
    pTextureDesc->setPixelFormat(format);
    pTextureDesc->setTextureType(MTL::TextureType2D);
    pTextureDesc->setStorageMode(MTL::StorageModeManaged);
    pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead);

    MTL::Texture* pTexture = mDevice->newTexture(pTextureDesc);
    pTextureDesc->release();
    if (!pTexture)
        return nullptr;

    const uint32_t blockBytes = canCompress ? (withAlpha ? 16u : 8u) : 0u;
    for (uint32_t l = 0; l < levels; ++l)
    {
        const uint32_t w = std::max(1, texWidth >> l);
        const uint32_t h = std::max(1, texHeight >> l);
        const size_t rowBytes = blockBytes ? (size_t)((w + 3) / 4) * blockBytes : (size_t)w * 4;
        pTexture->replaceRegion(MTL::Region::Make3D(0, 0, 0, w, h, 1), l, payload[l].data(), rowBytes);
    }

    if (!cacheFile.empty())
    {
        std::error_code ec;
        fs::create_directories(cacheDir, ec);
        // Written to a temporary name and renamed, so a run interrupted halfway
        // leaves no half-file for the next one to read as valid.
        const std::string tmp = cacheFile + ".tmp";
        std::ofstream out(tmp, std::ios::binary);
        if (out)
        {
            CachedTextureHeader header{};
            std::memcpy(header.magic, "BTEX", 4);
            header.version = kTextureCacheVersion;
            header.width = (uint32_t)texWidth;
            header.height = (uint32_t)texHeight;
            header.levels = levels;
            header.pixelFormat = (uint32_t)format;
            header.blockBytes = blockBytes;
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            for (uint32_t l = 0; l < levels; ++l)
            {
                const uint32_t byteLength = (uint32_t)payload[l].size();
                out.write(reinterpret_cast<const char*>(&byteLength), sizeof(byteLength));
                out.write(reinterpret_cast<const char*>(payload[l].data()), byteLength);
            }
            out.close();
            fs::rename(tmp, cacheFile, ec);
        }
    }

    ++mTextureCacheMisses;
    return pTexture;
}

// One blit pass for every texture loaded this scene. Doing it per texture would
// mean a command buffer and a round trip each, which on a scene with a hundred
// 4K maps is the dominant part of load time.
void MetalRender::generateTextureMips()
{
    if (mTexturesNeedingMips.empty())
        return;
    MTL::CommandBuffer* cb = mCommandQueue->commandBuffer();
    cb->retain();
    MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
    for (MTL::Texture* t : mTexturesNeedingMips)
        blit->generateMipmaps(t);
    blit->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    cb->release();
    STRELKA_INFO("Generated mipmaps for {} textures", mTexturesNeedingMips.size());
    mTexturesNeedingMips.clear();
    // Nothing reaches here any more: mip chains are built on the CPU alongside
    // the block compression, because generateMipmaps does not accept a block
    // format and because a CPU chain is what can be cached. Kept for a scene
    // that somehow loads an uncompressed texture outside that path.
}

void MetalRender::createMetalMaterials()
{
    // No budget: one call does the lot, which is what the headless path wants.
    while (!stepMetalMaterials(0.0))
    {
    }
}

bool MetalRender::stepMetalMaterials(double budgetMs)
{
    using simd::float3;
    const std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    const fs::path resourcePath = getSettings()->getAs<std::string>("resource/searchPath");

    if (!mMaterialBuild)
    {
        mMaterialBuild = new MaterialBuildState();
        mMaterialBuild->gpuMaterials.reserve(matDescs.size());
        // Appended to below, so it has to start empty even if a previous build
        // for this renderer got as far as filling some of it.
        mMaterialIsCutout.clear();
    }
    MaterialBuildState& st = *mMaterialBuild;

    auto loadTex = [&](const std::string& path, bool srgb,
                       TextureKind kind = TextureKind::Color) -> MTL::ResourceID {
        if (path.empty()) return MTL::ResourceID{};
        const fs::path fullPath = resourcePath / path;
        const std::string key =
            fullPath.string() + (srgb ? "|srgb" : "|linear") + "|" + std::to_string((int)kind);
        const auto it = st.textureCache.find(key);
        if (it != st.textureCache.end())
        {
            return it->second ? it->second->gpuResourceID() : MTL::ResourceID{};
        }
        MTL::Texture* tex = loadTextureFromFile(fullPath.string(), srgb, kind);
        st.textureCache.emplace(key, tex);
        if (tex) mMaterialTextures.push_back(tex);
        return tex ? tex->gpuResourceID() : MTL::ResourceID{};
    };

    // Checked after every material rather than every so many: one material can
    // pull in five maps, and a map that misses the cache is a decode and an
    // upload -- far more than the clock read that guards it.
    const auto sliceStart = std::chrono::steady_clock::now();
    while (st.cursor < matDescs.size())
    {
        const Scene::MaterialDescription& currMatDesc = matDescs[st.cursor];
        ++st.cursor;
        Material material = {};
        const auto& p = currMatDesc.params;
        material.base_color = packed_float3(simd_make_float3(p.base_color.x, p.base_color.y, p.base_color.z));
        material.metallic = p.metallic;
        material.roughness = p.roughness;
        material.ior = p.ior;
        material.specular = p.specular;
        material.specular_tint = p.specular_tint;
        material.diffuse_transmission = p.diffuse_transmission;
        material.diffuse_transmission_color = packed_float3(simd_make_float3(
            p.diffuse_transmission_color.x, p.diffuse_transmission_color.y, p.diffuse_transmission_color.z));
        material.transmission = p.transmission;
        material.clearcoat = p.clearcoat;
        material.clearcoat_roughness = p.clearcoat_roughness;
        material.anisotropy = p.anisotropy;
        material.emission = packed_float3(simd_make_float3(p.emission.x, p.emission.y, p.emission.z));
        material.emission_strength = p.emission_strength;
        material.normal_scale = p.normal_scale;
        material.occlusion_strength = p.occlusion_strength;
        material.alpha_cutoff = p.alpha_cutoff;
        material.alpha_mode = p.alpha_mode;
        material.base_color_alpha = p.base_color_alpha;
        material.attenuation_color = packed_float3(
            simd_make_float3(p.attenuation_color.x, p.attenuation_color.y, p.attenuation_color.z));
        material.attenuation_distance = p.attenuation_distance;
        material.uv_offset = simd_make_float2(p.uv_offset_x, p.uv_offset_y);
        material.uv_scale = simd_make_float2(p.uv_scale_x, p.uv_scale_y);
        material.uv_rotation = p.uv_rotation;
        material.material_type = p.material_type;
        material.thin_walled = p.thin_walled;
        material.dielectric_priority = p.dielectric_priority;

        material.baseColorTexture = loadTex(currMatDesc.baseColorTexPath, true);
        material.metallicRoughnessTexture =
            loadTex(currMatDesc.metallicRoughnessTexPath, false, TextureKind::NonColor);
        material.normalTexture = loadTex(currMatDesc.normalTexPath, false, TextureKind::Normal);
        material.emissionTexture = loadTex(currMatDesc.emissionTexPath, true);
        material.occlusionTexture = loadTex(currMatDesc.occlusionTexPath, false, TextureKind::NonColor);

        if (p.alpha_mode != ALPHA_MODE_OPAQUE)
            mSceneHasAlphaMaterials = true;
        // Per material, so a BLAS can say which of its geometries actually need
        // the alpha test. The scene-wide flag below only answers "is there any
        // cutout anywhere", which in a forest is always yes and drags trunks,
        // rocks and ground into the callback with the needles.
        mMaterialIsCutout.push_back(p.alpha_mode != ALPHA_MODE_OPAQUE ? 1u : 0u);
        st.gpuMaterials.push_back(material);

        if (budgetMs > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() >=
                budgetMs)
        {
            if (mLoadProgress)
            {
                mLoadProgress->total.store((uint32_t)matDescs.size(), std::memory_order_relaxed);
                mLoadProgress->done.store((uint32_t)st.cursor, std::memory_order_relaxed);
            }
            return false;
        }
    }

    generateTextureMips();
    STRELKA_INFO("Textures: {} from cache, {} built and cached", mTextureCacheHits, mTextureCacheMisses);

    const size_t materialsDataSize = sizeof(Material) * st.gpuMaterials.size();
    if (materialsDataSize > 0)
    {
        mMaterialBuffer = mDevice->newBuffer(materialsDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mMaterialBuffer->contents(), st.gpuMaterials.data(), materialsDataSize);
        mMaterialBuffer->didModifyRange(NS::Range::Make(0, mMaterialBuffer->length()));
    }
    else
    {
        mMaterialBuffer = nullptr;
    }

    delete mMaterialBuild;
    mMaterialBuild = nullptr;
    return true;
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
    kStageMiss,
    kStageResolve,
    kStageSort,
    kStageCount
};
const char* const kStageNames[kStageCount] = { "generate",   "prepare", "extend", "shade",
                                               "prepShadow", "shadow",  "miss",   "resolve",
                                               "sort" };
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
void MetalRender::createStageCounterHeap()
{
    if (mStageCounterHeap || !mMetal4.isValid())
    {
        return;
    }
    MTL4::CounterHeapDescriptor* d = MTL4::CounterHeapDescriptor::alloc()->init();
    d->setType(MTL4::CounterHeapTypeTimestamp);
    d->setCount(kMaxStageSamples * 2);
    NS::Error* err = nullptr;
    mStageCounterHeap = mDevice->newCounterHeap(d, &err);
    d->release();
    if (!mStageCounterHeap)
    {
        STRELKA_WARNING("Metal 4 counter heap unavailable: {}",
                        err && err->localizedDescription() ? err->localizedDescription()->utf8String()
                                                           : "unknown error");
    }
}

// Same report as the Metal 3 path, from timestamps written inline rather than at
// encoder boundaries. Stage kinds are recorded in encode order, one pair each.
void MetalRender::reportStageTimingsMetal4()
{
    if (!mStageCounterHeap || mStageKinds.empty())
    {
        return;
    }
    const NS::UInteger n = mStageKinds.size() + 1;
    NS::Data* data = mStageCounterHeap->resolveCounterRange(NS::Range::Make(0, n));
    if (!data)
    {
        return;
    }
    const MTL4::TimestampHeapEntry* ts = static_cast<const MTL4::TimestampHeapEntry*>(data->bytes());

    // Anchor the tick scale once, against the frame this heap just timed.
    //
    // The obvious calibration -- MTLDevice::sampleTimestamps, which pairs a CPU
    // clock with a GPU one -- does not apply: it reports the GPU side in
    // nanoseconds, while the counter heap ticks at its own rate (about 24 MHz on
    // this part, so ~42 ns each). Calibrating against it returned a factor of
    // exactly 1 ns/tick and a report forty times too small. The command buffer's
    // measured GPU time is the one number known to be in milliseconds and to
    // cover the same work, so the marks are scaled to it.
    //
    // Everything outside the marks -- the resolve and the tonemapper, together
    // well under a millisecond -- is folded into the factor, which biases it by
    // under a percent and only once.
    // A mark that was never written reads back as zero, and a run that wrote
    // fewer marks than the last one leaves the tail stale. Anchoring on either
    // gives a span orders of magnitude too large and a scale that rounds every
    // stage to nothing -- which is how this first presented.
    bool marksValid = ts[0].timestamp != 0;
    for (NS::UInteger i = 0; marksValid && i < mStageKinds.size(); ++i)
    {
        marksValid = ts[i + 1].timestamp >= ts[i].timestamp && ts[i + 1].timestamp != 0;
    }
    if (!marksValid)
    {
        return;
    }

    const uint64_t spanTicks = ts[mStageKinds.size()].timestamp - ts[0].timestamp;
    if (mGpuTicksToMs == 0.0 && spanTicks > 0)
    {
        const double frameMs = getLastRenderTimeMs();
        if (frameMs > 0.0)
        {
            mGpuTicksToMs = frameMs / double(spanTicks);
            STRELKA_INFO("GPU timestamp scale anchored: 1 tick = {:.2f} ns ({} ticks over {:.2f} ms)",
                         mGpuTicksToMs * 1e6, spanTicks, frameMs);
        }
    }
    if (mGpuTicksToMs == 0.0)
    {
        return;
    }

    double totals[kStageCount] = {};
    uint32_t counts[kStageCount] = {};
    std::string perBounce[kStageCount];
    for (NS::UInteger i = 0; i < mStageKinds.size(); ++i)
    {
        const uint64_t a = ts[i].timestamp;
        const uint64_t b = ts[i + 1].timestamp;
        if (a == 0 || b <= a)
        {
            continue;
        }
        const uint8_t kind = mStageKinds[i];
        const double ms = double(b - a) * mGpuTicksToMs;
        totals[kind] += ms;
        ++counts[kind];
        if (kind == kStageExtend || kind == kStageShade || kind == kStageShadow)
        {
            perBounce[kind] += fmt::format("{:.2f} ", ms);
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
        line += fmt::format("{} {:.2f}ms({:.0f}%, n={})  ", kStageNames[k], totals[k],
                            sum > 0.0 ? 100.0 * totals[k] / sum : 0.0, counts[k]);
    }
    // Diagnostic: the span between the first and last mark against the command
    // buffer's own GPU time. If the span matches and the gaps do not, the marks
    // are misplaced; if neither matches, the clock is not the one being assumed.
    STRELKA_INFO("STAGES total {:.2f}ms  {}", sum, line);
    STRELKA_INFO("STAGES per bounce: extend [{}] shade [{}] shadow [{}]", perBounce[kStageExtend],
                 perBounce[kStageShade], perBounce[kStageShadow]);
}

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
            perBounce[kind] += fmt::format("{:.2f} ", ms);
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
        line += fmt::format("{} {:.2f}ms({:.0f}%, n={})  ", kStageNames[k], totals[k],
                            sum > 0.0 ? 100.0 * totals[k] / sum : 0.0, counts[k]);
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
            paths += fmt::format("{:.0f}k ", stats[32 + i] / 1000.0);
            shadows += fmt::format("{:.0f}k ", stats[64 + i] / 1000.0);
        }
        STRELKA_INFO("STAGES rays per bounce: paths [{}] shadow [{}]", paths, shadows);
    }
}


// Metal 4 encode of the wavefront tracer.
//
// Same stage order and the same dispatch counts as the Metal 3 path; what
// changes is how the GPU is told about them. Bindings become addresses in an
// argument table, the values that used to ride in setBytes come out of the
// per-frame constant ring, and every dependency between dispatches is stated
// with a barrier because nothing tracks them any more.

// Declare every persistent allocation resident for the Metal 4 queue.
//
// Metal 3 infers residency from the bindings an encoder makes; Metal 4 does not,
// and an address in an argument table pointing at a non-resident allocation is a
// GPU fault rather than a validation message. This is the price of the argument
// table: the caller owns lifetime and residency both.
void MetalRender::makeResourcesResidentForMetal4(Buffer* output)
{
    if (!mMetal4.isValid())
    {
        return;
    }
    auto add = [&](MTL::Allocation* a) { mMetal4.addResident(a); };

    for (MTL::Buffer* b : mUniformBuffers) add(b);
    for (MTL::Buffer* b : mUniformTMBuffers) add(b);
    add(mVertexBuffer);
    add(mPrevVertexBuffer);
    add(mPrevFrameVertexBuffer);
    add(mPrevFrameInstanceBuffer);
    add(mIndexBuffer);
    add(mInstanceBuffer);
    add(mMaterialBuffer);
    add(mLightBuffer);
    add(mGeometryEntryBuffer);
    add(mEnvAliasBuffer);
    add(mAccumulationBuffer);
    add(mPathStateBuffer);
    add(mPathRayBuffer);
    add(mHitBuffer);
    add(mIorStackBuffer);
    add(mRadianceBuffer);
    add(mGuideRadianceBuffer);
    add(mPathQueueBuffer[0]);
    add(mPathQueueBuffer[1]);
    add(mHitQueueBuffer);
    add(mMissQueueBuffer);
    add(mShadowRayBuffer);
    add(mAovBuffer);
    add(mWavefrontControlBuffer);
    add(mSkinDataBuffer);
    add(mJointMatricesBuffer);
    add(mEnvMapTexture);
    // Bound by the miss stage alongside mEnvMapTexture. Metal 4 has no
    // useResource to fall back on, so a texture the argument table names and the
    // residency set does not is a page fault, and the command buffer just fails.
    add(mEnvBackgroundTexture);
    add(mSharcBuffer);
    add(mGuides.color);
    add(mGuides.depth);
    add(mGuides.motion);
    add(mGuides.diffuse);
    add(mGuides.specular);
    add(mGuides.normal);
    add(mGuides.roughness);
    add(mGuides.specularHitDistance);
    add(mGuides.reactive);
    add(mDenoisedTexture);
    // Intersection function tables are resources like any other, and they are
    // built on first use of a variant -- which is why residency is settled after
    // the variant exists rather than before.
    for (const auto& entry : mWavefrontVariants)
    {
        add(entry.second.shadowTableMotion);
        add(entry.second.shadowTableStatic);
    }
    for (MTL::Texture* t : mMaterialTextures) add(t);
    for (MTL::Texture* t : mDisplayTextures) add(t);
    for (MTL::Texture* t : mUpscaleTextures) add(t);
    for (MTL::AccelerationStructure* as : mPrimitiveAccelerationStructures) add(as);
    add(mInstanceAccelerationStructure);
    for (Mesh* mesh : mMetalMeshes)
    {
        if (mesh && mesh->mPerPrimitiveBuffer) add(mesh->mPerPrimitiveBuffer);
    }
    // The renderer alternates between output buffers, so declaring only the one
    // this frame happens to use leaves every other frame writing into an
    // allocation the queue does not know about -- which reads back as black
    // rather than as an error.
    for (Buffer* b : mAsyncOutputBuffers)
    {
        if (b)
        {
            add(((MetalBuffer*)b)->getNativePtr());
        }
    }
    if (output)
    {
        add(((MetalBuffer*)output)->getNativePtr());
    }
    mMetal4.commitResidency();
}

void MetalRender::encodeWavefrontMetal4(MTL4::CommandBuffer* cmd, MTL4::ComputeCommandEncoder*& enc,
                                        MTL::Buffer* uniformBuffer,
                                        Buffer* output, uint32_t width, uint32_t height,
                                        uint32_t sampleCount, uint32_t features)
{
    const WavefrontVariant* variant = wavefrontVariantFor(features | kFeatureMetal4);
    if (!variant)
    {
        return;
    }
    const uint32_t pixels = width * height;
    const uint32_t maxDepth = std::max(1u, getSettings()->getAs<uint32_t>("render/pt/depth"));
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = reinterpret_cast<const Uniforms*>(uniformBuffer->contents());
    const uint32_t dispatchSampleCount = sampleCount + (uniforms->canonicalGuideSample ? 1u : 0u);

    MTL4::ArgumentTable* table = mMetal4.argumentTable();
    ConstantRing& ring = mMetal4.constants();
    enc->setArgumentTable(table);

    const uint32_t kThreadsPerGroup = 64;
    const MTL::Size tg = MTL::Size(kThreadsPerGroup, 1, 1);
    const MTL::Size fullGrid = MTL::Size((pixels + kThreadsPerGroup - 1) / kThreadsPerGroup, 1, 1);
    const MTL::GPUAddress control = mWavefrontControlBuffer->gpuAddress();
    const NS::UInteger kDispatchArgsOffset = 2 * sizeof(uint32_t);
    const NS::UInteger kShadowArgsOffset = 8 * sizeof(uint32_t);
    const NS::UInteger kShadowCounterOffset = 6 * sizeof(uint32_t);
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);

    const bool useMotion = mMotionBlasBuilt || variant->extendStatic == nullptr ||
                           getSettings()->getAs<uint32_t>("render/pt/staticTraversal") == 0;

    // One timestamp per stage boundary. Each mark is written in the *same*
    // encoder as the work it closes, immediately after a queue-scoped barrier
    // that waits for that work -- not at the start of the next encoder, which is
    // where the first working version put it and which attributed each stage's
    // wait to its successor (extend and shadow swapped places against Metal 3).
    //
    // The barrier has to be queue-scoped: barrierAfterEncoderStages, which every
    // stage below uses, orders only within one encoder, so a split without it
    // left stages reading indirect arguments the previous one had not finished
    // writing and the frame collapsed to a tenth of its work.
    //
    // A stage's duration is the gap to the next mark, so there is one more mark
    // than stage. This reshapes the frame exactly as the Metal 3 path does -- a
    // measurement mode, not something to leave on.
    const bool profile = mProfileStages && mStageCounterHeap != nullptr;
    uint32_t markIndex = 0;
    auto writeMark = [&]() {
        enc->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        enc->writeTimestamp(MTL4::TimestampGranularityPrecise, mStageCounterHeap, markIndex++);
        enc->endEncoding();
        enc = cmd->computeCommandEncoder();
        enc->setArgumentTable(table);
    };
    auto mark = [&](uint8_t kind) {
        if (!profile || mStageKinds.size() + 1 >= kMaxStageSamples)
        {
            return;
        }
        // The opening mark rides in the encoder that carries the first stage's
        // work. On its own, in an encoder holding nothing else, it never lands:
        // the timestamp reads back zero and the whole run is discarded. Nothing
        // precedes it, so it needs no barrier either.
        if (markIndex == 0)
        {
            enc->writeTimestamp(MTL4::TimestampGranularityPrecise, mStageCounterHeap, markIndex++);
        }
        mStageKinds.push_back(kind);
    };
    auto closeStage = [&]() {
        if (!profile || mStageKinds.empty())
        {
            return;
        }
        writeMark();
    };

    // Every dispatch here reads what the one before it wrote. Metal 4 does not
    // work that out, so say it: dispatch-to-dispatch, visible device-wide.
    auto barrier = [&]() {
        enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    };
    auto bind = [&](MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index) {
        table->setAddress(buffer ? buffer->gpuAddress() + offset : 0, index);
    };

    for (uint32_t s = 0; s < dispatchSampleCount; ++s)
    {
        const MTL::GPUAddress sampleIdx = ring.push(s);
        MTL::Buffer* sampleRadiance =
            (uniforms->canonicalGuideSample && s == 0u) ? mGuideRadianceBuffer : mRadianceBuffer;

        mark(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        bind(uniformBuffer, 0, 0);
        bind(mPathStateBuffer, 0, 1);
        bind(sampleRadiance, 0, 2);
        bind(mIorStackBuffer, 0, 3);
        table->setAddress(sampleIdx, 4);
        bind(mPathQueueBuffer[0], 0, 5);
        bind(mWavefrontControlBuffer, 0, 6);
        bind(mAovBuffer, 0, 7);
        bind(mPathRayBuffer, 0, 8);
        enc->dispatchThreadgroups(fullGrid, tg);
        closeStage();
        barrier();

        for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)
        {
            const uint32_t src = bounce & 1u;
            const uint32_t dst = src ^ 1u;
            const MTL::GPUAddress srcIdx = ring.push(src);
            const MTL::GPUAddress groupSize = ring.push(kThreadsPerGroup);
            const MTL::GPUAddress bounceIdx = ring.push(bounce);

            enc->setComputePipelineState(mWavefrontPreparePSO4);
            bind(mWavefrontControlBuffer, 0, 0);
            table->setAddress(srcIdx, 1);
            table->setAddress(groupSize, 2);
            table->setAddress(bounceIdx, 3);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageExtend);
            enc->setComputePipelineState(useMotion ? variant->extendMotion : variant->extendStatic);
            bind(uniformBuffer, 0, 0);
            bind(mInstanceBuffer, 0, 1);
            table->setResource(mInstanceAccelerationStructure->gpuResourceID(), 2);
            bind(mPathRayBuffer, 0, 3);
            bind(mHitBuffer, 0, 4);
            table->setAddress(sampleIdx, 5);
            bind(mPathQueueBuffer[src], 0, 6);
            bind(mWavefrontControlBuffer, 0, 7);
            bind(mHitQueueBuffer, 0, 8);
            bind(mWavefrontControlBuffer, kHitCounterOffset, 9);
            bind(mMissQueueBuffer, 0, 10);
            bind(mWavefrontControlBuffer, kMissCounterOffset, 11);
            bind(mPathStateBuffer, 0, 12);
            enc->dispatchThreadgroups(control + kDispatchArgsOffset, tg);
            closeStage();
            barrier();

            enc->setComputePipelineState(mWavefrontPrepareHitMissPSO4);
            bind(mWavefrontControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageMiss);
            enc->setComputePipelineState(variant->miss);
            bind(uniformBuffer, 0, 0);
            bind(mPathStateBuffer, 0, 1);
            bind(mPathRayBuffer, 0, 2);
            bind(sampleRadiance, 0, 3);
            bind(mMissQueueBuffer, 0, 4);
            bind(mWavefrontControlBuffer, 0, 5);
            bind(mAovBuffer, 0, 6);
            table->setAddress(sampleIdx, 7);
            if (mEnvMapTexture)
            {
                table->setTexture(mEnvMapTexture->gpuResourceID(), 0);
                table->setTexture((mEnvBackgroundTexture ? mEnvBackgroundTexture : mEnvMapTexture)
                                      ->gpuResourceID(),
                                  1);
            }
            enc->dispatchThreadgroups(control + kMissArgsOffset, tg);
            closeStage();

            mark(kStageShade);
            enc->setComputePipelineState(variant->shade);
            bind(uniformBuffer, 0, 0);
            bind(mInstanceBuffer, 0, 1);
            bind(mLightBuffer, 0, 3);
            bind(mMaterialBuffer, 0, 4);
            bind(mPathStateBuffer, 0, 5);
            bind(mHitBuffer, 0, 6);
            bind(sampleRadiance, 0, 7);
            bind(mIorStackBuffer, 0, 8);
            bind(mGeometryEntryBuffer, 0, 9);
            bind(mEnvAliasBuffer, 0, 10);
            bind(mVertexBuffer, 0, 11);
            bind(mPrevVertexBuffer, 0, 12);
            bind(mIndexBuffer, 0, 13);
            table->setAddress(sampleIdx, 14);
            bind(mHitQueueBuffer, 0, 15);
            bind(mPathQueueBuffer[dst], 0, 16);
            bind(mWavefrontControlBuffer, dst * sizeof(uint32_t), 17);
            bind(mWavefrontControlBuffer, 0, 18);
            bind(mShadowRayBuffer, 0, 19);
            bind(mWavefrontControlBuffer, kShadowCounterOffset, 20);
            bind(mPathRayBuffer, 0, 21);
            bind(mAovBuffer, 0, 22);
            bind(mPrevFrameVertexBuffer ? mPrevFrameVertexBuffer : mVertexBuffer, 0, 23);
            bind(mPrevFrameInstanceBuffer ? mPrevFrameInstanceBuffer : mInstanceBuffer, 0, 24);
            enc->dispatchThreadgroups(control + kHitArgsOffset, tg);
            closeStage();
            barrier();

            enc->setComputePipelineState(mWavefrontPrepareShadowPSO4);
            bind(mWavefrontControlBuffer, 0, 0);
            table->setAddress(groupSize, 1);
            table->setAddress(bounceIdx, 2);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            barrier();

            mark(kStageShadow);
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            bind(uniformBuffer, 0, 0);
            table->setResource(mInstanceAccelerationStructure->gpuResourceID(), 1);
            bind(mShadowRayBuffer, 0, 2);
            bind(sampleRadiance, 0, 3);
            bind(mWavefrontControlBuffer, 0, 4);
            table->setAddress(sampleIdx, 5);
            bind(mInstanceBuffer, 0, 6);
            bind(mMaterialBuffer, 0, 7);
            bind(mGeometryEntryBuffer, 0, 8);
            bind(mVertexBuffer, 0, 9);
            bind(mIndexBuffer, 0, 10);
            // Cutout shadows: without this the shadow kernel calls into a table
            // that was never bound, and a canopy blocks light outright instead
            // of letting the alpha test decide.
            MTL::IntersectionFunctionTable* shadowTable =
                useMotion ? variant->shadowTableMotion : variant->shadowTableStatic;
            if (shadowTable)
            {
                shadowTable->setBuffer(mInstanceBuffer, 0, 0);
                shadowTable->setBuffer(mMaterialBuffer, 0, 1);
                shadowTable->setBuffer(mGeometryEntryBuffer, 0, 2);
                shadowTable->setBuffer(mVertexBuffer, 0, 3);
                shadowTable->setBuffer(mIndexBuffer, 0, 4);
                table->setResource(shadowTable->gpuResourceID(), 11);
            }
            enc->dispatchThreadgroups(control + kShadowArgsOffset, tg);
            closeStage();
            barrier();
        }
    }

    enc->setComputePipelineState(mWavefrontResolvePSO4);
    bind(uniformBuffer, 0, 0);
    bind(mRadianceBuffer, 0, 1);
    bind(outputBuffer, 0, 2);
    bind(mAccumulationBuffer, 0, 3);
    table->setAddress(ring.push(sampleCount), 4);
    bind(mAovBuffer, 0, 5);
    enc->dispatchThreadgroups(fullGrid, tg);

    // Guide resolve. The guides exist only when something downstream reads them,
    // which is either MetalFX mode, so their presence is the condition -- the
    // encoder has no other way to know which one the frame selected.
    if (mGuides.color && mAovResolvePSO4)
    {
        barrier();
        enc->setComputePipelineState(mAovResolvePSO4);
        bind(uniformBuffer, 0, 0);
        bind(mAovBuffer, 0, 1);
        bind(mRadianceBuffer, 0, 2);
        table->setAddress(ring.push(sampleCount), 3);
        bind(mAccumulationBuffer, 0, 4);
        table->setTexture(mGuides.color->gpuResourceID(), 0);
        table->setTexture(mGuides.depth->gpuResourceID(), 1);
        table->setTexture(mGuides.motion->gpuResourceID(), 2);
        table->setTexture(mGuides.diffuse->gpuResourceID(), 3);
        table->setTexture(mGuides.specular->gpuResourceID(), 4);
        table->setTexture(mGuides.normal->gpuResourceID(), 5);
        table->setTexture(mGuides.roughness->gpuResourceID(), 6);
        table->setTexture(mGuides.specularHitDistance->gpuResourceID(), 7);
        table->setTexture(mGuides.reactive->gpuResourceID(), 8);
        enc->dispatchThreadgroups(MTL::Size((width + 7) / 8, (height + 7) / 8, 1), MTL::Size(8, 8, 1));
    }
}

MTL::ComputeCommandEncoder* MetalRender::encodeWavefront(MTL::CommandBuffer* pCmd,
                                  MTL::ComputeCommandEncoder* enc, MTL::Buffer* uniformBuffer,
                                  Buffer* output, uint32_t width, uint32_t height,
                                  uint32_t sampleCount, uint32_t features)
{
    const uint32_t pixels = width * height;
    const WavefrontVariant* variant = wavefrontVariantFor(features);
    const uint32_t maxDepth = std::max(1u, getSettings()->getAs<uint32_t>("render/pt/depth"));
    MTL::Buffer* outputBuffer = ((MetalBuffer*)output)->getNativePtr();
    const auto* uniforms = reinterpret_cast<const Uniforms*>(uniformBuffer->contents());
    const uint32_t dispatchSampleCount = sampleCount + (uniforms->canonicalGuideSample ? 1u : 0u);

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
    const NS::UInteger kHitArgsOffset = 13 * sizeof(uint32_t);
    const NS::UInteger kHitCounterOffset = 11 * sizeof(uint32_t);
    const NS::UInteger kMissArgsOffset = 18 * sizeof(uint32_t);
    const NS::UInteger kMissCounterOffset = 16 * sizeof(uint32_t);
    // Nothing in the scene deforms -> traverse it as a static structure. Every ray
    // was otherwise paying for motion-BVH traversal it could not use.
    const bool useMotion = mMotionBlasBuilt || !variant || variant->extendStatic == nullptr ||
                          getSettings()->getAs<uint32_t>("render/pt/staticTraversal") == 0;
    if (!variant)
    {
        return enc;
    }


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
        // Named so that a capture can attribute to a stage. Instruments reports
        // register spills against an encoder id and nothing else, and an
        // unlabelled trace says only that *something* spilled.
        enc->setLabel(NS::String::string(kStageNames[kind], NS::UTF8StringEncoding));
        declareResidency(enc);
        mStageKinds.push_back(kind);
    };

    for (uint32_t s = 0; s < dispatchSampleCount; ++s)
    {
        MTL::Buffer* sampleRadiance =
            (uniforms->canonicalGuideSample && s == 0u) ? mGuideRadianceBuffer : mRadianceBuffer;
        stamp(kStageGenerate);
        enc->setComputePipelineState(variant->generate);
        enc->setBuffer(uniformBuffer, 0, 0);
        enc->setBuffer(mPathStateBuffer, 0, 1);
        enc->setBuffer(sampleRadiance, 0, 2);
        enc->setBuffer(mIorStackBuffer, 0, 3);
        enc->setBytes(&s, sizeof(uint32_t), 4);
        enc->setBuffer(mPathQueueBuffer[0], 0, 5);
        enc->setBuffer(mWavefrontControlBuffer, 0, 6);
        enc->setBuffer(mAovBuffer, 0, 7);
        enc->setBuffer(mPathRayBuffer, 0, 8);
        enc->dispatchThreads(grid, tg);

        for (uint32_t bounce = 0; bounce < maxDepth; ++bounce)
        {
            const uint32_t src = bounce & 1u;
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

            // Sort the queue this bounce is about to traverse. Bounce 0 is the
            // camera and already perfectly coherent, so it is skipped -- sorting
            // it is pure cost.


            stamp(kStageExtend);
            enc->pushDebugGroup(NS::String::string("extend", NS::UTF8StringEncoding));
            enc->setComputePipelineState(useMotion ? variant->extendMotion : variant->extendStatic);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mInstanceBuffer, 0, 1);
            enc->setAccelerationStructure(mInstanceAccelerationStructure, 2);
            enc->setBuffer(mPathRayBuffer, 0, 3);
            enc->setBuffer(mHitBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            enc->setBuffer(mPathQueueBuffer[src], 0, 6);
            enc->setBuffer(mWavefrontControlBuffer, 0, 7);
            enc->setBuffer(mHitQueueBuffer, 0, 8);
            enc->setBuffer(mWavefrontControlBuffer, kHitCounterOffset, 9);
            enc->setBuffer(mMissQueueBuffer, 0, 10);
            enc->setBuffer(mWavefrontControlBuffer, kMissCounterOffset, 11);
            enc->setBuffer(mPathStateBuffer, 0, 12);
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kDispatchArgsOffset, tg);
            enc->popDebugGroup();

            enc->setComputePipelineState(mWavefrontPrepareHitMissPSO);
            enc->setBuffer(mWavefrontControlBuffer, 0, 0);
            enc->setBytes(&kThreadsPerGroup, sizeof(uint32_t), 1);
            enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

            stamp(kStageMiss);
            enc->setComputePipelineState(variant->miss);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mPathStateBuffer, 0, 1);
            enc->setBuffer(mPathRayBuffer, 0, 2);
            enc->setBuffer(sampleRadiance, 0, 3);
            enc->setBuffer(mMissQueueBuffer, 0, 4);
            enc->setBuffer(mWavefrontControlBuffer, 0, 5);
            enc->setBuffer(mAovBuffer, 0, 6);
            enc->setBytes(&s, sizeof(uint32_t), 7);
            if (mEnvMapTexture)
            {
                enc->setTexture(mEnvMapTexture, 0);
                enc->setTexture(mEnvBackgroundTexture ? mEnvBackgroundTexture : mEnvMapTexture, 1);
            }
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kMissArgsOffset, tg);

            stamp(kStageShade);
            enc->pushDebugGroup(NS::String::string("shade", NS::UTF8StringEncoding));
            enc->setComputePipelineState(variant->shade);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mInstanceBuffer, 0, 1);
            enc->setBuffer(mLightBuffer, 0, 3);
            enc->setBuffer(mMaterialBuffer, 0, 4);
            enc->setBuffer(mPathStateBuffer, 0, 5);
            enc->setBuffer(mHitBuffer, 0, 6);
            enc->setBuffer(sampleRadiance, 0, 7);
            enc->setBuffer(mIorStackBuffer, 0, 8);
            enc->setBuffer(mGeometryEntryBuffer, 0, 9);
            enc->setBuffer(mEnvAliasBuffer, 0, 10);
            enc->setBuffer(mVertexBuffer, 0, 11);
            enc->setBuffer(mPrevVertexBuffer, 0, 12);
            enc->setBuffer(mIndexBuffer, 0, 13);
            enc->setBytes(&s, sizeof(uint32_t), 14);
            enc->setBuffer(mHitQueueBuffer, 0, 15);
            enc->setBuffer(mPathQueueBuffer[dst], 0, 16);
            enc->setBuffer(mWavefrontControlBuffer, dst * sizeof(uint32_t), 17);
            enc->setBuffer(mWavefrontControlBuffer, 0, 18);
            enc->setBuffer(mShadowRayBuffer, 0, 19);
            enc->setBuffer(mWavefrontControlBuffer, kShadowCounterOffset, 20);
            enc->setBuffer(mPathRayBuffer, 0, 21);
            enc->setBuffer(mAovBuffer, 0, 22);
            // With nothing deforming, the current vertex buffer already is the
            // previous pose, so it is bound directly rather than copied.
            enc->setBuffer(mPrevFrameVertexBuffer ? mPrevFrameVertexBuffer : mVertexBuffer, 0, 23);
            enc->setBuffer(mPrevFrameInstanceBuffer ? mPrevFrameInstanceBuffer : mInstanceBuffer, 0, 24);
            if (mEnvMapTexture)
            {
                enc->setTexture(mEnvMapTexture, 0);
            }
            if (mSharcBuffer)
            {
                enc->setBuffer(mSharcBuffer, 0, 25);
            }
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kHitArgsOffset, tg);
            enc->popDebugGroup();

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
            enc->setComputePipelineState(useMotion ? variant->shadowMotion : variant->shadowStatic);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setAccelerationStructure(mInstanceAccelerationStructure, 1);
            enc->setBuffer(mShadowRayBuffer, 0, 2);
            enc->setBuffer(sampleRadiance, 0, 3);
            enc->setBuffer(mWavefrontControlBuffer, 0, 4);
            enc->setBytes(&s, sizeof(uint32_t), 5);
            // Cutout shadows need to resolve the material and its uv at each hit.
            enc->setBuffer(mInstanceBuffer, 0, 6);
            enc->setBuffer(mMaterialBuffer, 0, 7);
            enc->setBuffer(mGeometryEntryBuffer, 0, 8);
            enc->setBuffer(mVertexBuffer, 0, 9);
            enc->setBuffer(mIndexBuffer, 0, 10);
            // The intersection function reads the same tables the kernel does,
            // through the function table's own binding points.
            MTL::IntersectionFunctionTable* shadowTable =
                useMotion ? variant->shadowTableMotion : variant->shadowTableStatic;
            if (shadowTable)
            {
                shadowTable->setBuffer(mInstanceBuffer, 0, 0);
                shadowTable->setBuffer(mMaterialBuffer, 0, 1);
                shadowTable->setBuffer(mGeometryEntryBuffer, 0, 2);
                shadowTable->setBuffer(mVertexBuffer, 0, 3);
                shadowTable->setBuffer(mIndexBuffer, 0, 4);
                enc->setIntersectionFunctionTable(shadowTable, 11);
                enc->useResource(shadowTable, MTL::ResourceUsageRead);
            }
            enc->dispatchThreadgroups(mWavefrontControlBuffer, kShadowArgsOffset, tg);

        }

        // Every path that passed through a cache voxel deposits what it gathered
        // after it -- here, at the end of the sample, because by now the sample's
        // deferred shadow rays have landed in the accumulator too.
        if (mSharcBuffer && variant->sharcDeposit)
        {
            enc->setComputePipelineState(variant->sharcDeposit);
            enc->setBuffer(uniformBuffer, 0, 0);
            enc->setBuffer(mPathStateBuffer, 0, 1);
            enc->setBuffer(sampleRadiance, 0, 2);
            enc->setBuffer(mSharcBuffer, 0, 3);
            const MTL::Size depositTg = MTL::Size::Make(
                std::min<NS::UInteger>(variant->sharcDeposit->maxTotalThreadsPerThreadgroup(), 256u),
                1, 1);
            enc->dispatchThreads(MTL::Size::Make(width * height, 1, 1), depositTg);
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
    enc->setBuffer(mAovBuffer, 0, 5);
    enc->dispatchThreads(grid, tg);

    if (profile && mStageStatsBuffer)
    {
        enc->endEncoding();
        MTL::BlitCommandEncoder* blit = pCmd->blitCommandEncoder();
        blit->copyFromBuffer(mWavefrontControlBuffer, 0, mStageStatsBuffer, 0, mStageStatsBuffer->length());
        blit->endEncoding();
        enc = pCmd->computeCommandEncoder();
        enc->setLabel(NS::String::string("stats readback", NS::UTF8StringEncoding));
        declareResidency(enc);
    }
    return enc;
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
        if (!stepSceneBuild(output))
        {
            // Nothing to trace yet. The busy flag has to come off here: it is set
            // by triggerRenderIfIdle before every call, and a build stage that
            // returns without submitting anything leaves no completion handler to
            // clear it, so the build would stall one stage in.
            mRenderBusy.store(false, std::memory_order_release);
            pPool->release();
            return;
        }
    }
    else
    {
        handleSceneChanges();
    }

    mFrameIndex = (mFrameIndex + 1) % kMaxFramesInFlight;

    // Recreate accumulation buffer if output size changed
    // Render resolution. Upscaling renders fewer pixels and lets MetalFX bring
    // them up to the display size; the factor is clamped because below a quarter
    // the scaler has too little to work with and the result is mush.
    const uint32_t outWidth = output->width();
    const uint32_t outHeight = output->height();
    const bool wantUpscale = getSettings()->getAs<bool>("render/pt/enableUpscale");
    float upscaleFactor =
        wantUpscale ? std::clamp(getSettings()->getAs<float>("render/pt/upscaleFactor"), 0.25f, 1.0f) : 1.0f;
    // MetalFX publishes the range of output/input ratios it can actually do, and
    // going outside it is not refused -- the scaler is created and then produces
    // NaN. Measured: at a factor of 0.25 (a 4x ratio) the denoised frame comes
    // back entirely non-finite. So the request is clamped to what the device says
    // it supports rather than to a number chosen by hand.
    if (wantUpscale && getSettings()->getAs<bool>("render/pt/denoise"))
    {
        float minScale = 1.0f, maxScale = 2.0f;
        MetalFxContext::denoiserScaleRange(mDevice, minScale, maxScale);
        const float lowest = maxScale > 0.0f ? 1.0f / maxScale : 0.5f;
        if (upscaleFactor < lowest)
        {
            if (!mLoggedUpscaleClamp)
            {
                mLoggedUpscaleClamp = true;
                STRELKA_WARNING("MetalFX denoiser supports {:.2f}x-{:.2f}x; clamping render scale {:.2f} to {:.2f}",
                                minScale, maxScale, upscaleFactor, lowest);
            }
            upscaleFactor = lowest;
        }
    }
    const uint32_t width = std::max(1u, (uint32_t)(outWidth * upscaleFactor));
    const uint32_t height = std::max(1u, (uint32_t)(outHeight * upscaleFactor));
    const bool upscaling = (width != outWidth || height != outHeight);

    // Temporal denoising subsumes upscaling: the denoised scaler takes the
    // reduced-resolution frame and produces the display-resolution one, so the
    // spatial scaler is only for when denoising is off.
    // Denoising needs the guides, and the wavefront tracer writes them. It is
    // also the only tracer left: the megakernel went with the banded Metal 3
    // submission it was encoded through.
    const bool useWavefrontTracer = mWavefrontLibrary != nullptr;
    const uint32_t debug = getSettings()->getAs<uint32_t>("render/pt/debug");
    // Debug views are final outputs. Sending them through MetalFX would alter
    // their values, while the debug path deliberately skips the final tonemap.
    bool denoising = getSettings()->getAs<bool>("render/pt/denoise") && useWavefrontTracer && debug == 0;
    const char* shaderValidationEnv = getenv("MTL_SHADER_VALIDATION");
    const bool shaderValidation =
        shaderValidationEnv && atoi(shaderValidationEnv) != 0;
    if (denoising && shaderValidation)
    {
        if (!mLoggedShaderValidationDenoiserGap)
        {
            mLoggedShaderValidationDenoiserGap = true;
            STRELKA_WARNING(
                "MetalFX denoiser disabled: MTL_SHADER_VALIDATION lowers the "
                "threadgroup limit below MetalFX's internal kernel requirement");
        }
        denoising = false;
    }
    if (denoising)
    {
        // No Metal 4 variant on purpose: newTemporalDenoisedScalerWithDevice:compiler:
        // aborts, and Apple has confirmed it as a framework bug (FB22575333) with
        // the Metal 3 constructor as the recommended workaround -- which is this
        // call. supportsMetal4FX answers YES and is not to be trusted; the same
        // trap is reported on A17 Pro, where it aborts differently again.
        // tools/metalfx_mtl4_denoiser_repro.mm reproduces it and lists what was
        // ruled out.
        if (!mLoggedMetal4DenoiserGap && mMetal4.isValid())
        {
            mLoggedMetal4DenoiserGap = true;
            STRELKA_INFO("MetalFX denoiser stays on Metal 3 (supportsMetal4FX={}, FB22575333)",
                         MetalFxContext::denoiserSupportsMetal4(mDevice));
        }
        denoising =
            mMetalFx.ensureDenoiser(mDevice, width, height, outWidth, outHeight);
        if (denoising)
        {
            ensureGuideTextures(width, height, outWidth, outHeight);
        }
    }
    if (!denoising && mGuides.color)
    {
        releaseGuideTextures();
    }
    // Three ways to get from render resolution to output resolution, and the
    // choice matters most at one sample: the spatial scaler has no history and
    // resamples noise as-is, the temporal scaler accumulates across frames, and
    // the denoiser does that plus a guided denoise -- at the price of a second
    // traced sample for clean guides.
    const uint32_t upscaleMode = getSettings()->getAs<uint32_t>("render/pt/upscaleMode");
    const bool wantTemporal = upscaling && !denoising && upscaleMode == 1u;
    if (wantTemporal)
    {
        void* compiler = mMetal4.isValid() && getSettings()->getAs<uint32_t>("render/pt/metal4") != 0
                             ? (void*)mMetal4.compiler()
                             : nullptr;
        if (mMetalFx.ensureTemporalScaler(mDevice, MTL::PixelFormatRGBA16Float, MTL::PixelFormatR32Float,
                                          MTL::PixelFormatRG16Float, MTL::PixelFormatRGBA16Float, width,
                                          height, outWidth, outHeight, compiler))
        {
            // Depth and motion are guides the temporal scaler needs as much as
            // the denoiser does, so the same textures and the same producing
            // pass serve both.
            ensureGuideTextures(width, height, outWidth, outHeight);
        }
    }
    if (upscaling && !denoising && !mMetalFx.hasTemporalScaler())
    {
        void* spatialCompiler =
            mMetal4.isValid() &&
                    getSettings()->getAs<uint32_t>("render/pt/metal4") != 0
                ? (void*)mMetal4.compiler()
                : nullptr;
        mMetalFx.ensureSpatialScaler(mDevice, MTL::PixelFormatRGBA16Float, MTL::PixelFormatRGBA16Float, width,
                                     height, outWidth, outHeight,
                                     // The tonemapper has already applied the tone curve and gamma,
                                     // so what the scaler sees is display-referred.
                                     MetalFxContext::ColorMode::Perceptual,
                                     spatialCompiler);
        ensureUpscaleTextures(width, height);
    }
    const size_t requiredSize = width * height * output->getElementSize();
    if (mAccumulationBuffer && requiredSize != mAccumulationBuffer->length())
    {
        mAccumulationBuffer->release();
        mAccumulationBuffer = mDevice->newBuffer(requiredSize, MTL::ResourceStorageModePrivate);
        ctx.mSubframeIndex = 0;
    }

    // Update motion blur enable state from settings each frame
    mEnableMotionBlur = getSettings()->getAs<bool>("render/enableMotionBlur");

    // Deforming geometry only needs a two-keyframe structure while the shutter is
    // actually open across them. When it is not, the sample time is pinned to
    // keyframe 1, the second keyframe is dead weight, and every ray pays for
    // motion traversal it cannot use.
    // ...and while nothing is animating, the two keyframes hold the same pose, so
    // there is nothing to interpolate between either. That is the case that
    // matters: accumulation runs with playback paused.
    bool anyAnimationPlaying = false;
    for (size_t a = 0; a < mScene->getAnimations().size(); ++a)
    {
        const std::string key = fmt::format("render/animation/anim{}/state", a);
        anyAnimationPlaying =
            anyAnimationPlaying || getSettings()->getAs<bool>(key);
    }
    // Playing is not the condition -- a shutter spanning two different poses is.
    //
    // Pausing mid-animation should freeze a frame *of the film*, and a frame of
    // the film has motion blur in it; the estimator then keeps refining that
    // frame. Tying the motion structures to playback instead threw the blur away
    // a few frames after the pause and left a crisp still, because the pose
    // keyframes are only identical once something has made them so. Right after a
    // pause they still hold t_open and t_close of the last rendered frame, which
    // is exactly the interval that should stay.
    const bool wantMotionBlas = mEnableMotionBlur &&
                                getSettings()->getAs<bool>("render/isMotionBlurVisible") &&
                                (anyAnimationPlaying || mShutterIntervalActive);
    if (wantMotionBlas != mMotionBlasBuilt && !mBlasList.empty())
    {
        mBuildMotionBlas = wantMotionBlas;
        rebuildAccelerationStructures();
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
        mHasPrevFramePose = false;
    }

    // Before anything this frame can move: skinning rewrites the vertices and the
    // animation block re-uploads the instance transforms, so this is the last
    // moment at which both still describe the frame that was just displayed.
    capturePrevFramePose();

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
            const std::string key = fmt::format("render/animation/anim{}/time", i);
            mAnimTargetTimes[i] = animSettings.getAs<float>(key);
            const float delta = std::abs(animations[i].current - mAnimTargetTimes[i]);
            if (delta > EPSILON)
            {
                mAnimChanged[i] = true;
                animStateChanged = true;
                maxTimeDelta = std::max(maxTimeDelta, delta);
            }
        }

        // A jump in animation time is a cut: the frame after it has no valid
        // predecessor to reproject from. Playback advances a sixtieth of a second
        // at a time, so a fraction of the clip length separates the two cases by a
        // wide margin. This was already being computed and then not used.
        for (int i = 0; i < (int)animCount; ++i)
        {
            const float clip = animations[i].end - animations[i].start;
            if (mAnimChanged[i] && clip > 0.0f && maxTimeDelta > 0.05f * clip)
            {
                mResetDenoiseHistory = true;
                mHasPrevFramePose = false;
                break;
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
                    float tOpen = mAnimTargetTimes[i];
                    if (mAnimChanged[i])
                    {
                        tOpen += shutterOffset;
                    }
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
                    float tClose = mAnimTargetTimes[i];
                    if (mAnimChanged[i])
                    {
                        tClose += shutterOffset + shutterDuration;
                    }
                    tClose = std::clamp(tClose, animations[i].start, animations[i].end);
                    animations[i].current = tClose;
                    pass2Skeletal |= mScene->applyAnimation(i);
                }

                if (pass2Skeletal)
                {
                    applySkinning();
                    updateSkeletalBLAS();
                }
                mShutterIntervalActive = true;
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
                    animations[i].current = mAnimTargetTimes[i];
                    accelStructureDirty |= mScene->applyAnimation(i);
                }

                if (accelStructureDirty)
                {
                    applySkinning();
                    // Sync prevVB with current VB — motion BVH needs both keyframes
                    // consistent when motion blur is off (otherwise keyframe 0 is stale)
                    copyVertexBufferToPrev();
                    updateSkeletalBLAS();
                    rebuildTLAS();
                    // Both keyframes are the same pose again: nothing to blur, and
                    // the scene can go back to static structures.
                    mShutterIntervalActive = false;
                }
                else
                {
                    rebuildTLAS(); // already re-uploads the instance transforms
                }
            }
            ctx.mSubframeIndex = 0;
        }
    }

    const bool enteredPause = mWasAnimationPlaying && !anyAnimationPlaying;
    mPausedBlurRefine =
        denoising && mEnableMotionBlur &&
        getSettings()->getAs<bool>("render/isMotionBlurVisible") &&
        !anyAnimationPlaying && mShutterIntervalActive;
    if (enteredPause && mPausedBlurRefine)
    {
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
    }
    mWasAnimationPlaying = anyAnimationPlaying;

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

    // Temporal history: reset on a cut, not on a move.
    //
    // Reprojection maps this frame's pixels onto the previous frame's. Panning,
    // orbiting and flying keep that mapping meaningful and the motion vectors
    // describe it. What breaks it is a discontinuity -- a teleport, a switch to
    // another camera, a projection change -- after which the history describes a
    // different place and blending it in is ghosting.
    {
        const glm::float3 camPos = glm::float3(glm::inverse(currView.mCamMatrices.view)[3]);
        // Third column of the view matrix is the camera's backward axis.
        const glm::float3 camForward =
            -glm::float3(currView.mCamMatrices.view[0][2], currView.mCamMatrices.view[1][2],
                         currView.mCamMatrices.view[2][2]);
        if (mHasPrevCamera)
        {
            const float step = glm::length(camPos - mPrevCameraPos);
            const float turn = glm::dot(camForward, mPrevCameraForward);
            // A jump is a step far larger than the one before it -- scale-free, so
            // it works on a scene of any size -- or a turn no hand makes in a frame.
            const bool teleported = mPrevCameraStep > 0.0f && step > 8.0f * mPrevCameraStep;
            const bool spun = turn < 0.5f; // more than 60 degrees in one frame
            const bool projectionChanged = glm::any(
                glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective));
            if (teleported || spun || projectionChanged)
            {
                mResetDenoiseHistory = true;
            }
            mPrevCameraStep = step;
        }
        mPrevCameraPos = camPos;
        mPrevCameraForward = camForward;
        mHasPrevCamera = true;
    }

    // Turning the denoiser on hands it a history from whenever it last ran.
    // Tracked on the effective state, not the setting: switching to a tracer that
    // writes no guides turns the denoiser off just as surely as the checkbox does.
    const bool denoiseEnabledNow = denoising;
    if (denoiseEnabledNow != mPrevDenoiseEnabled)
    {
        mResetDenoiseHistory = true;
        mPrevDenoiseEnabled = denoiseEnabledNow;
    }

    // --- Cache all settings once per frame ---
    const uint32_t spp = settings.getAs<uint32_t>("render/pt/spp");
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    const uint32_t maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    const uint32_t samplerType = settings.getAs<uint32_t>("render/pt/samplerType");
    const uint32_t blueNoiseSwitchSpp = settings.getAs<uint32_t>("render/pt/blueNoiseSwitchSpp");
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    const bool isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");
    const bool enableCameraMotionBlur = settings.getAs<bool>("render/enableCameraMotionBlur");
    const bool playbackBlur =
        settings.getAs<bool>("render/pt/denoisePlaybackMotionBlur");
    const float shutterTime = settings.getAs<float>("render/motionBlur/shutterTime");
    const uint32_t shutterMode =
        settings.getAs<uint32_t>("render/motionBlur/shutterMode");
    const bool qualityPlaybackBlur =
        denoising && anyAnimationPlaying && mEnableMotionBlur &&
        isMotionBlurVisible && playbackBlur;
    const bool effectiveAccumulation = enableAccumulation && !anyAnimationPlaying;
    const uint32_t accumulatedSamples = (uint32_t)ctx.mSubframeIndex;
    const uint32_t remainingSamples =
        accumulatedSamples < sspTotal ? sspTotal - accumulatedSamples : 0u;
    const bool accumulationActive = effectiveAccumulation && remainingSamples > 0u;

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
    // 0 = glTF (-ln(C)/d), 1 = Cycles ((1-C)/d). See volume.h.
    pUniformData->volumeModel = settings.getAs<uint32_t>("render/material/volumeModel");
    pUniformData->samples_per_launch = spp;
    pUniformData->enableAccumulation = (uint32_t)accumulationActive;
    pUniformData->risCandidates = std::max(settings.getAs<uint32_t>("render/pt/risCandidates"), 1u);
    pUniformData->textureLodMode = settings.getAs<uint32_t>("render/pt/textureLod");
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = maxDepth;
    pUniformData->debug = debug;
    // Denoiser guides. Off unless something downstream consumes them: writing
    // them costs a 64-byte store per pixel at the primary hit.
    // Looking at a guide implies producing it, and so does denoising.
    const bool denoiseOn = denoising;
    pUniformData->writeAov =
        settings.getAs<uint32_t>("render/pt/writeAov") || debug >= DEBUG_MODE_FIRST_AOV || denoiseOn;
    pUniformData->useAccumulatedColor =
        (mPausedBlurRefine && effectiveAccumulation &&
         ctx.mSubframeIndex > 0 && mAccumulationBuffer && !mNoAccumColor)
            ? 1u
            : 0u;
    const bool accumulating = pUniformData->useAccumulatedColor != 0u;
    float jx = 0.0f, jy = 0.0f;
    if (denoiseOn && !mPausedBlurRefine && !accumulating)
    {
        const double ratio = (double)outWidth / (double)std::max(width, 1u);
        const uint32_t phaseCount =
            (uint32_t)std::clamp(std::lround(8.0 * ratio * ratio), 8L, 128L);
        frameJitter(ctx.mFrameNumber, phaseCount, jx, jy);
    }
    pUniformData->jitterX = jx;
    pUniformData->jitterY = jy;
    // Temporal reconstruction of any kind needs the frame shifted by a known
    // amount it can undo; without jitter the scaler has nothing new to
    // accumulate between frames and degenerates to a blur.
    const bool temporalOn =
        denoiseOn || (settings.getAs<bool>("render/pt/enableUpscale") &&
                      settings.getAs<uint32_t>("render/pt/upscaleMode") == 1u);
    pUniformData->useFrameJitter = temporalOn ? 1u : 0u;
    pUniformData->canonicalGuideSample = denoiseOn ? 1u : 0u;
    pUniformData->denoiseFireflyClamp = settings.getAs<float>("render/pt/denoiseFireflyClamp");
    {
        // Previous frame's world-to-clip for screen-space reprojection. The
        // motion-blur uniforms hold the inverses and cannot serve here.
        const glm::float4x4 prevWorldToClip =
            mPrevView.mCamMatrices.perspective * mPrevView.mCamMatrices.view;
        std::memcpy(&pUniformData->prevWorldToClip, glm::value_ptr(prevWorldToClip), sizeof(float4x4));
        const glm::float4x4 worldToClip = currView.mCamMatrices.perspective * currView.mCamMatrices.view;
        std::memcpy(&pUniformData->worldToClip, glm::value_ptr(worldToClip), sizeof(float4x4));
    }
    pUniformData->denoiseDepthMode = settings.getAs<uint32_t>("render/pt/denoiseDepthMode");
    // A pose is only usable once one has been captured *and* the frame it belongs
    // to still corresponds to this one. Anything that resets the history has
    // already declared that it does not.
    pUniformData->hasPrevFramePose = (mHasPrevFramePose && !mResetDenoiseHistory && !mNoPrevPose) ? 1u : 0u;
    pUniformData->enableMotionBlur = mEnableMotionBlur ? 1 : 0;
    const bool stochasticShutter =
        isMotionBlurVisible && (!denoising || qualityPlaybackBlur || mPausedBlurRefine);
    pUniformData->isMotionBlurVisible = (uint32_t)stochasticShutter;
    pUniformData->enableCameraMotionBlur = (uint32_t)enableCameraMotionBlur;
    pUniformData->rectLightSamplingMethod = rectLightSamplingMethod;
    pUniformData->samplerType = samplerType;
    pUniformData->blueNoiseSwitchSpp = blueNoiseSwitchSpp;

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

    // Radiance cache
    {
        const bool wantSharc = getSettings()->getAs<bool>("render/pt/sharc");
        const uint32_t capacity =
            wantSharc ? std::max(1u << 16, getSettings()->getAs<uint32_t>("render/pt/sharcCapacity"))
                      : 0u;
        if (capacity != mSharcCapacity)
        {
            if (mSharcBuffer)
            {
                mSharcBuffer->release();
                mSharcBuffer = nullptr;
            }
            mSharcCapacity = 0;
            if (capacity)
            {
                // 20 bytes an entry: a key, three sums and a count.
                mSharcBuffer = mDevice->newBuffer((size_t)capacity * 20, MTL::ResourceStorageModePrivate);
                if (mSharcBuffer)
                {
                    mSharcCapacity = capacity;
                    STRELKA_INFO("Radiance cache: {} entries ({:.1f} MB)", capacity,
                                 capacity * 20 / 1e6);
                }
            }
        }
        pUniformData->sharcCapacity = mSharcCapacity;
        pUniformData->sharcMinSamples = getSettings()->getAs<uint32_t>("render/pt/sharcMinSamples");
        pUniformData->sharcDepth = getSettings()->getAs<uint32_t>("render/pt/sharcDepth");
        // The world size of one pixel at unit distance, times the number of
        // pixels a voxel should span. Everything scene-dependent -- field of
        // view, resolution -- is folded in here so the setting itself is not.
        const float tanHalfFov = std::tan(glm::radians(camera.fovForAspect((float)width / height)) * 0.5f);
        const float pixelAngle = 2.0f * tanHalfFov / (float)height;
        pUniformData->sharcBaseSize =
            pixelAngle * std::max(1.0f, getSettings()->getAs<float>("render/pt/sharcVoxelPixels"));
    }

    // Atmosphere
    {
        const auto& atmosphere = mScene->getAtmosphere();
        const bool on = atmosphere.has_value() && atmosphere->density > 0.0f;
        pUniformData->hasFog = on ? 1u : 0u;
        pUniformData->fogSigmaT = on ? atmosphere->density : 0.0f;
        pUniformData->fogAnisotropy = on ? atmosphere->anisotropy : 0.0f;
        pUniformData->fogHeight = on ? atmosphere->height : 0.0f;
        pUniformData->fogAlbedo = on ? (vector_float3){ atmosphere->color.x, atmosphere->color.y,
                                                        atmosphere->color.z }
                                     : (vector_float3){ 0.0f, 0.0f, 0.0f };
    }

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
        const bool hasBackdrop = mEnvBackgroundTexture != nullptr;
        pUniformData->hasEnvBackground = hasBackdrop ? 1u : 0u;
        pUniformData->envBackgroundIntensity =
            hasBackdrop && envLight.has_value() ? envLight->backgroundIntensity : 1.0f;
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
        pUniformData->hasEnvBackground = 0;
        pUniformData->envPdfScale = 0.0f;
    }

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->outWidth = outWidth;
    pUniformTonemap->outHeight = outHeight;
    // A debug view is data, not a picture: a normal, a roughness or a motion
    // vector means what it means, and a tone curve would misreport it. So the
    // pass runs as a straight copy instead of being skipped -- it is the only
    // thing that writes the texture the display samples, and skipping it left
    // every debug view showing the last tonemapped frame, which reads as the
    // control doing nothing at all.
    const bool debugView = debug != 0;
    // 0 is ToneMapperType::eNone, which lives in the shader-side tonemappers.h.
    pUniformTonemap->tonemapperType = debugView ? 0u : settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = debugView ? 0.0f : settings.getAs<float>("render/post/gamma");
    pUniformTonemap->maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");

    // --- Detect settings changes (member-based, not static) ---
    bool settingsChanged = false;
    settingsChanged |= (mPrevSettings.rectLightSamplingMethod != rectLightSamplingMethod);
    settingsChanged |= (mPrevSettings.samplerType != samplerType);
    settingsChanged |= (mPrevSettings.blueNoiseSwitchSpp != blueNoiseSwitchSpp);
    settingsChanged |= (mPrevSettings.enableAccumulation != enableAccumulation);
    settingsChanged |= (mPrevSettings.sspTotal > sspTotal);
    settingsChanged |= (mPrevSettings.spp != spp);
    settingsChanged |= (mPrevSettings.playbackBlur != playbackBlur);
    settingsChanged |= (mPrevSettings.shutterTime != shutterTime);
    settingsChanged |= (mPrevSettings.shutterMode != shutterMode);
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
    mPrevSettings.blueNoiseSwitchSpp = blueNoiseSwitchSpp;
    mPrevSettings.enableAccumulation = enableAccumulation;
    mPrevSettings.sspTotal = sspTotal;
    mPrevSettings.spp = spp;
    mPrevSettings.playbackBlur = playbackBlur;
    mPrevSettings.shutterTime = shutterTime;
    mPrevSettings.shutterMode = shutterMode;
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
        // The pixels still correspond but their content jumps -- a new depth, a
        // different sampler, a debug view. Blending across that is ghosting, and a
        // reset costs one noisy frame after a change the user asked for.
        mResetDenoiseHistory = true;
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
    // Exposure is part of the picture, not part of the data, so it goes too.
    pUniformTonemap->exposureValue = debugView ? float3(1.0f) : exposureValue;
    pUniformData->exposureValue = exposureValue; // need for proper accumulation

    const auto samplesPerLaunch = pUniformData->samples_per_launch;
    // if accumulation is off then launch selected samples per pixel
    // The sample limit stops the *estimator*, not the denoiser. A temporal
    // denoiser is a continuous filter: starve it and the frame it was maintaining
    // is simply gone -- and with a reduced-resolution render there is not even a
    // display-sized image left to fall back on, so the screen empties.
    const uint32_t samplesThisLaunch =
        accumulationActive
            ? std::min(samplesPerLaunch, remainingSamples)
            : (effectiveAccumulation && !denoising ? 0u : samplesPerLaunch);
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
        {
            ensureWavefrontBuffers(width, height);
            // Output resolution, not render resolution: this is what the display
            // shows and what MetalFX upscales into.
            ensureDisplayTextures(outWidth, outHeight);

            // Metal 4 path. Residency has to be refreshed whenever the set of
            // allocations can have changed; ensureWavefrontBuffers only does work
            // when the resolution does, so its capacity doubles as the generation.
            // Denoising pins the frame to Metal 3: the denoiser has no working
            // Metal 4 variant to encode into an MTL4 command buffer.
            // Only the denoiser pins a frame to Metal 3, and only because its
            // Metal 4 constructor asserts inside MPSGraph -- see the note above
            // ensureDenoiser and tools/metalfx_mtl4_denoiser_repro.mm. The
            // temporal and spatial scalers both build through the Metal 4
            // compiler, so upscaling no longer forces the older path.
            const bool useMetal4 = mMetal4.isValid() && settings.getAs<uint32_t>("render/pt/metal4") != 0 &&
                                   mWavefrontResolvePSO4 != nullptr && !denoising;

            mProfileStages = settings.getAs<uint32_t>("render/pt/profileStages") != 0;
            if (mProfileStages)
            {
                createStageTimestampBuffer();
            }

            const auto encodeStart = std::chrono::high_resolution_clock::now();
            uint32_t features = 0;
            if (pUniformData->hasEnvMap)
                features |= kFeatureEnvMap;
            if (pUniformData->numLights > 0)
                features |= kFeatureLights;
            // Cutouts change how the shadow stage traverses, so scenes without
            // any pay nothing: they compile the any-hit variant.
            if (mSceneHasAlphaMaterials)
                features |= kFeatureAlpha;
            // Motion blur is only a feature of the shader if something can
            // actually move: with neither deforming geometry nor a moving
            // camera, the shutter time is a number nothing reads.
            if (pUniformData->enableMotionBlur &&
                (mMotionBlasBuilt || pUniformData->enableCameraMotionBlur))
                features |= kFeatureMotionBlur;
            if (pUniformData->useDof)
                features |= kFeatureDof;
            if (pUniformData->debug != 0)
                features |= kFeatureDebug;
            // Every ray, including every shadow ray, pays for the medium test.
            // A scene without an atmosphere compiles the variant that has none.
            if (pUniformData->hasFog)
                features |= kFeatureFog;
            if (pUniformData->sharcCapacity != 0)
                features |= kFeatureSharc;

            if (useMetal4)
            {
                // Build the variant first: its intersection function tables are
                // allocations, and residency has to name every allocation the
                // frame will touch before the frame is committed.
                if (mProfileStages)
                {
                    createStageCounterHeap();
                    if (mStageCounterHeap)
                    {
                        mStageCounterHeap->invalidateCounterRange(
                            NS::Range::Make(0, mStageCounterHeap->count()));
                    }
                }
                mStageKinds.clear();
                wavefrontVariantFor(features | kFeatureMetal4);
                if (mMetal4ResidencyGeneration != mWavefrontCapacity || mMetal4ResidencyDirty)
                {
                    makeResourcesResidentForMetal4(output);
                    mMetal4ResidencyGeneration = mWavefrontCapacity;
                    mMetal4ResidencyDirty = false;
                }

                MTL4::CommandBuffer* cmd4 = mMetal4.beginFrame((uint32_t)ctx.mFrameNumber);
                MTL4::ComputeCommandEncoder* enc4 = cmd4->computeCommandEncoder();
                encodeWavefrontMetal4(cmd4, enc4, pUniformBuffer, output, width, height,
                                      samplesThisLaunch, features);
                if (mTonemapperPSO4)
                {
                    enc4->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                                    MTL4::VisibilityOptionDevice);
                    enc4->setComputePipelineState(mTonemapperPSO4);
                    mMetal4.argumentTable()->setAddress(pUniformTMBuffer->gpuAddress(), 0);
                    mMetal4.argumentTable()->setAddress(((MetalBuffer*)output)->getNativePtr()->gpuAddress(), 1);
                    mMetal4.argumentTable()->setTexture(tonemapTarget(upscaling)->gpuResourceID(), 0);
                    enc4->dispatchThreadgroups(MTL::Size((width + 7) / 8, (height + 7) / 8, 1),
                                               MTL::Size(8, 8, 1));
                }
                enc4->endEncoding();
                if (upscaling)
                {
                    // MetalFX encodes into the command buffer, not an encoder of
                    // ours, so this has to follow endEncoding().
                    mMetalFx.encodeSpatial(cmd4, true, mUpscaleTextures[mWriteIndex],
                                           mDisplayTextures[mWriteIndex], width, height);
                }
                cmd4->endCommandBuffer();

                // Completion arrives through commit options rather than a
                // handler on the command buffer, and carries the GPU interval
                // with it, so the Metal 3 timing path needs no counterpart.
                const int writeIdx4 = mWriteIndex;
                MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();
                options->addFeedbackHandler(
                    MTL4::CommitFeedbackHandlerFunction([this, writeIdx4](MTL4::CommitFeedback* fb) {
                        mLastRenderTimeMs.store((fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0,
                                                std::memory_order_relaxed);
                        mReadyIndex.store(writeIdx4);
                        mRenderBusy.store(false, std::memory_order_release);
                    }));
                const MTL4::CommandBuffer* buffers[] = { cmd4 };
                mMetal4.queue()->commit(buffers, 1, options);
                options->release();
                // Signalled on the queue after the commit, so it orders behind
                // this frame's work. renderSync waits on it; the async loop
                // ignores it and keeps using the feedback handler above.
                mMetal4FrameValue = mMetal4.signalFrame();

                ctx.mSubframeIndex =
                    accumulationActive
                        ? ctx.mSubframeIndex + samplesThisLaunch
                        : (effectiveAccumulation ? ctx.mSubframeIndex : 0);
                pPool->release();
                mPrevView = currView;
                mHasPrevFramePose = true;
                ctx.mFrameNumber++;
                return;
            }

            MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
            {
                // Numbered, so a capture names the sample a frame came from
                // rather than "Command Buffer 0" four hundred times over.
                char label[32];
                std::snprintf(label, sizeof(label), "sample %u", (uint32_t)ctx.mSubframeIndex);
                pCmd->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
            }
            MTL::ComputeCommandEncoder* enc = pCmd->computeCommandEncoder();
            enc = encodeWavefront(pCmd, enc, pUniformBuffer, output, width, height, samplesThisLaunch,
                                  features);
            if (mProfileStages)
            {
                const double encodeMs =
                    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - encodeStart)
                        .count();
                STRELKA_INFO("STAGES cpu encode {:.3f} ms", encodeMs);
            }

            if (!denoising)
            {
                enc->setComputePipelineState(mTonemapperPSO);
                enc->useResource(((MetalBuffer*)output)->getNativePtr(),
                                 MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
                enc->setBuffer(pUniformTMBuffer, 0, 0);
                enc->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
                enc->setTexture(tonemapTarget(upscaling), 0);
                enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }
            const bool temporalUpscale = upscaling && !denoising && mMetalFx.hasTemporalScaler();
            if (denoising || temporalUpscale)
            {
                // Spread the packed guides into the textures MetalFX reads, and
                // hand it linear radiance: exposure and the tone curve come after
                // the denoise, not before it. The temporal scaler reads two of
                // the same textures -- depth and motion -- so it takes the same
                // pass rather than a second one that writes a subset.
                enc->setComputePipelineState(mAovResolvePSO);
                enc->setBuffer(pUniformBuffer, 0, 0);
                enc->setBuffer(mAovBuffer, 0, 1);
                enc->setBuffer(mRadianceBuffer, 0, 2);
                enc->setBytes(&samplesThisLaunch, sizeof(uint32_t), 3);
                enc->setBuffer(mAccumulationBuffer, 0, 4);
                enc->setTexture(mGuides.color, 0);
                enc->setTexture(mGuides.depth, 1);
                enc->setTexture(mGuides.motion, 2);
                enc->setTexture(mGuides.diffuse, 3);
                enc->setTexture(mGuides.specular, 4);
                enc->setTexture(mGuides.normal, 5);
                enc->setTexture(mGuides.roughness, 6);
                enc->setTexture(mGuides.specularHitDistance, 7);
                enc->setTexture(mGuides.reactive, 8);
                enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }
            enc->endEncoding();

            if (denoising)
            {
                MetalFxContext::DenoiseInputs in;
                in.color = mGuides.color;
                in.depth = mGuides.depth;
                in.motion = mGuides.motion;
                in.diffuseAlbedo = mGuides.diffuse;
                in.specularAlbedo = mGuides.specular;
                in.normal = mGuides.normal;
                in.roughness = mGuides.roughness;
                in.specularHitDistance = mGuides.specularHitDistance;
                in.reactive = mGuides.reactive;
                in.output = mDenoisedTexture;
                // The header documents this property twice and the two readings
                // have opposite signs: "the subpixel sampling coordinate you use to
                // generate the color texture input" is what we applied, while "the
                // pixel offset this scaler samples to return to the frame's
                // reference frame" is its negation. Measured rather than reasoned
                // about -- see runJitterTest; the sign that wins is recorded in the
                // default of render/pt/jitterSign. The Y term carries our own flip
                // as well (generateCameraRay builds pixelPos.y as height - y), so
                // the two axes are switched independently.
                const uint32_t jitterSign = settings.getAs<uint32_t>("render/pt/jitterSign");
                in.jitterX = (jitterSign & 1u) ? -pUniformData->jitterX : pUniformData->jitterX;
                in.jitterY = (jitterSign & 2u) ? -pUniformData->jitterY : pUniformData->jitterY;
                in.depthReversed =
                    pUniformData->denoiseDepthMode == kDenoiseDepthDevice;
                // Reset when this frame has no valid predecessor to reproject
                // from -- *not* when the estimator restarts. Accumulation restarts
                // on every camera move, and resetting the denoiser with it throws
                // the history away exactly when it is worth most: while the camera
                // moves every frame is one sample and the history is all there is.
                // Motion vectors exist to carry smooth motion, so let them.
                in.resetHistory = mResetDenoiseHistory;
                std::memcpy(in.worldToView, glm::value_ptr(currView.mCamMatrices.view), sizeof(in.worldToView));
                std::memcpy(in.viewToClip, glm::value_ptr(currView.mCamMatrices.perspective),
                            sizeof(in.viewToClip));
                mResetDenoiseHistory = false;
                mMetalFx.encodeDenoise(pCmd, in);

                if (mTonemapperTexPSO)
                {
                    MTL::ComputeCommandEncoder* tm = pCmd->computeCommandEncoder();
                    tm->setComputePipelineState(mTonemapperTexPSO);
                    tm->setBuffer(pUniformTMBuffer, 0, 0);
                    tm->setTexture(mDisplayTextures[mWriteIndex], 0);
                    tm->setTexture(mDenoisedTexture, 1);
                    tm->dispatchThreads(MTL::Size(outWidth, outHeight, 1), MTL::Size(8, 8, 1));
                    tm->endEncoding();
                }
            }
            else if (temporalUpscale)
            {
                // Colour comes from the guide texture, which holds linear
                // radiance -- the same input the denoiser takes -- so the tone
                // curve is applied after the scaler, not before it.
                MetalFxContext::TemporalInputs tin;
                tin.color = mGuides.color;
                tin.depth = mGuides.depth;
                tin.motion = mGuides.motion;
                tin.output = mDenoisedTexture;
                const uint32_t jitterSign = settings.getAs<uint32_t>("render/pt/jitterSign");
                tin.jitterX = (jitterSign & 1u) ? -pUniformData->jitterX : pUniformData->jitterX;
                tin.jitterY = (jitterSign & 2u) ? -pUniformData->jitterY : pUniformData->jitterY;
                tin.depthReversed = pUniformData->denoiseDepthMode == kDenoiseDepthDevice;
                tin.resetHistory = mResetDenoiseHistory;
                mResetDenoiseHistory = false;
                mMetalFx.encodeTemporal(pCmd, false, tin);

                if (mTonemapperTexPSO)
                {
                    MTL::ComputeCommandEncoder* tm = pCmd->computeCommandEncoder();
                    tm->setComputePipelineState(mTonemapperTexPSO);
                    tm->setBuffer(pUniformTMBuffer, 0, 0);
                    tm->setTexture(mDisplayTextures[mWriteIndex], 0);
                    tm->setTexture(mDenoisedTexture, 1);
                    tm->dispatchThreads(MTL::Size(outWidth, outHeight, 1), MTL::Size(8, 8, 1));
                    tm->endEncoding();
                }
            }
            else if (upscaling)
            {
                mMetalFx.encodeSpatial(pCmd, false, mUpscaleTextures[mWriteIndex],
                                       mDisplayTextures[mWriteIndex], width, height);
            }

            const int writeIdxWf = mWriteIndex;
            if (!mSyncMode)
            {
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
            }
            else
            {
                retainCommandBufferForSync(pCmd);
            }
            pCmd->commit();

            if (accumulationActive)
            {
                ctx.mSubframeIndex += samplesThisLaunch;
            }
            else if (!effectiveAccumulation)
            {
                ctx.mSubframeIndex = 0;
            }
            pPool->release();
            mPrevView = currView;
            mHasPrevFramePose = true;
            ctx.mFrameNumber++;
            return;
        }
    }
    else
    {
        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
        {
            // Numbered, so a capture names the frame it came from rather than
            // "Command Buffer 0" four hundred times.
            char label[32];
            std::snprintf(label, sizeof(label), "sample %u", (uint32_t)ctx.mSubframeIndex);
            pCmd->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
        }

        MTL::BlitCommandEncoder* pBlitEncoder = pCmd->blitCommandEncoder();
        pBlitEncoder->copyFromBuffer(
            mAccumulationBuffer, 0, ((MetalBuffer*)output)->getNativePtr(), 0, width * height * sizeof(float4));
        pBlitEncoder->endEncoding();

        {
            MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();

            pComputeEncoder->setComputePipelineState(mTonemapperPSO);
            pComputeEncoder->useResource(
                ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
            pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
            pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
            pComputeEncoder->setTexture(tonemapTarget(upscaling), 0);
            {
                const MTL::Size gridSize = MTL::Size(width, height, 1);
                const MTL::Size threadgroupSize(8, 8, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
            pComputeEncoder->endEncoding();
        }

        // Completion handler for async double-buffered output. It must be
        // installed unconditionally in interactive mode: it is the only thing
        // that clears mRenderBusy, and skipping it would wedge the renderer.
        const int writeIdx = mWriteIndex;
        if (!mSyncMode)
        {
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx](MTL::CommandBuffer* cb) {
                const double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
                mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
                mReadyIndex.store(writeIdx);
                mRenderBusy.store(false, std::memory_order_release);
            }));
        }
        else
        {
            retainCommandBufferForSync(pCmd);
        }
        pCmd->commit();
    }
    pPool->release();

    mPrevView = currView;

    mHasPrevFramePose = true;
    ctx.mFrameNumber++;
}

void MetalRender::retainCommandBufferForSync(MTL::CommandBuffer* pCmd)
{
    if (mLastCommandBuffer)
    {
        mLastCommandBuffer->release();
    }
    mLastCommandBuffer = pCmd->retain();
}

// One frame into a .gputrace, for Xcode's shader profiler -- the only tool that
// reports a shader's register allocation. `MTL_CAPTURE_ENABLED` has to be set
// before the device exists, which main() does when --capture is given: the
// capture layer is inserted when Metal initialises, and asking for it later
// fails with "Capture layer is not inserted".
void MetalRender::beginGpuCapture(const std::string& path)
{
    MTL::CaptureManager* mgr = MTL::CaptureManager::sharedCaptureManager();
    if (!mgr->supportsDestination(MTL::CaptureDestinationGPUTraceDocument))
    {
        STRELKA_ERROR("GPU capture unavailable. Set MTL_CAPTURE_ENABLED=1 before launching.");
        return;
    }
    MTL::CaptureDescriptor* desc = MTL::CaptureDescriptor::alloc()->init();
    desc->setCaptureObject(mDevice);
    desc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    desc->setOutputURL(
        NS::URL::fileURLWithPath(NS::String::string(path.c_str(), NS::UTF8StringEncoding)));
    NS::Error* err = nullptr;
    if (!mgr->startCapture(desc, &err))
    {
        STRELKA_ERROR("GPU capture failed to start: {}",
                      err && err->localizedDescription() ? err->localizedDescription()->utf8String()
                                                         : "unknown");
    }
    else
    {
        STRELKA_INFO("GPU capture -> {}", path);
    }
    desc->release();
}

void MetalRender::endGpuCapture()
{
    MTL::CaptureManager::sharedCaptureManager()->stopCapture();
}

void MetalRender::renderSync(Buffer* output)
{
    // Frame boundaries, for anything watching from outside.
    //
    // Metal infers a frame from presentDrawable, and a headless renderer never
    // presents one -- so Instruments sees a single unbroken stretch of work and
    // every number it reports is an average over the whole run. A capture scope
    // is the API that says "this is a frame"; with one per sample the timeline
    // has countable, numbered frames and the counters can be read per frame
    // instead of per session.
    if (mFrameScope == nullptr && mCommandQueue)
    {
        mFrameScope = MTL::CaptureManager::sharedCaptureManager()->newCaptureScope(mCommandQueue);
        if (mFrameScope)
        {
            mFrameScope->setLabel(NS::String::string("Strelka sample", NS::UTF8StringEncoding));
        }
    }
    if (mFrameScope)
    {
        mFrameScope->beginScope();
    }
    mSyncMode = true;
    mLastCommandBuffer = nullptr;
    mMetal4FrameValue = 0;
    // A synchronous caller wants the frame, not a responsive window, so the
    // build runs to completion here rather than one stage per call.
    finishSceneBuild(output);
    const auto tEncode = std::chrono::steady_clock::now();
    render(output);
    const auto tSubmitted = std::chrono::steady_clock::now();
    if (!mLastCommandBuffer && mMetal4FrameValue != 0)
    {
        // Metal 4 frame. The event is signalled on the queue behind this
        // frame's work, so returning from here means the output buffer holds it.
        if (!mMetal4.waitForFrame(mMetal4FrameValue))
        {
            mDeviceError = true;
            if (!mDeviceErrorReported)
            {
                mDeviceErrorReported = true;
                STRELKA_ERROR("Metal 4 frame did not complete within the timeout");
            }
        }
        if (mProfileStages)
        {
            const auto tDone = std::chrono::steady_clock::now();
            STRELKA_INFO("LAUNCH encode {:.1f} ms, wait {:.1f} ms",
                         std::chrono::duration<double, std::milli>(tSubmitted - tEncode).count(),
                         std::chrono::duration<double, std::milli>(tDone - tSubmitted).count());
            // The commit feedback carries the GPU interval, so this is the same
            // number the Metal 3 path reads off its command buffer.
            STRELKA_INFO("CMDBUF gpu {:.1f} ms (metal4)", getLastRenderTimeMs());
            reportStageTimingsMetal4();
        }
    }
    else if (mLastCommandBuffer)
    {
        mLastCommandBuffer->waitUntilCompleted();
        if (mProfileStages)
        {
            const auto tDone = std::chrono::steady_clock::now();
            STRELKA_INFO("LAUNCH encode {:.1f} ms, wait {:.1f} ms",
                         std::chrono::duration<double, std::milli>(tSubmitted - tEncode).count(),
                         std::chrono::duration<double, std::milli>(tDone - tSubmitted).count());
            STRELKA_INFO("CMDBUF gpu {:.1f} ms (start->end {:.1f} ms, kernel {:.1f} ms)",
                         (mLastCommandBuffer->GPUEndTime() - mLastCommandBuffer->GPUStartTime()) * 1000.0,
                         (mLastCommandBuffer->kernelEndTime() - mLastCommandBuffer->kernelStartTime()) * 1000.0,
                         (mLastCommandBuffer->kernelEndTime() - mLastCommandBuffer->GPUStartTime()) * 1000.0);
        }
        // The async path reports these from a completion handler; the synchronous
        // one had nowhere to report from, so a headless profiling run printed the
        // CPU encode time and nothing about the GPU.
        if (mProfileStages)
        {
            reportStageTimings();
        }
        if (mLastCommandBuffer->status() == MTL::CommandBufferStatusError)
        {
            // Once, not once per sample: a scene that does not fit fails every
            // launch, and sixteen identical lines bury the one that matters.
            mDeviceError = true;
            if (!mDeviceErrorReported)
            {
                mDeviceErrorReported = true;
            // A render command buffer that failed produces a black frame and
            // nothing else. The usual cause is that the scene's resources do not
            // all fit on the device at once -- acceleration structures, vertex
            // and index buffers and textures are all needed resident -- and the
            // failure is otherwise completely silent.
            NS::Error* err = mLastCommandBuffer->error();
            STRELKA_ERROR("Render command buffer failed: {}. The scene may not fit on the device; "
                          "try render/texture/maxDimension.",
                          err && err->localizedDescription()
                              ? err->localizedDescription()->utf8String()
                              : "unknown error");
            }
        }
        const double gpuMs =
            (mLastCommandBuffer->GPUEndTime() - mLastCommandBuffer->GPUStartTime()) * 1000.0;
        mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
        mLastCommandBuffer->release();
        mLastCommandBuffer = nullptr;
    }

    // Managed storage needs an explicit GPU→CPU sync before the host can read.
    // On Apple silicon Managed behaves like Shared, but this keeps Intel Macs correct.
    if (output)
    {
        MTL::Buffer* native = ((MetalBuffer*)output)->getNativePtr();
        if (native && native->storageMode() == MTL::StorageModeManaged)
        {
            MTL::CommandBuffer* syncCmd = mCommandQueue->commandBuffer();
            MTL::BlitCommandEncoder* blit = syncCmd->blitCommandEncoder();
            blit->synchronizeResource(native);
            blit->endEncoding();
            syncCmd->commit();
            syncCmd->waitUntilCompleted();
        }
    }

    if (mFrameScope)
    {
        mFrameScope->endScope();
    }
    mSyncMode = false;
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



// Build (or return) the pipeline set specialised for one combination of scene
// features. Compiling seven kernels takes a few milliseconds, which is fine
// because the key only changes when a setting or the scene does — never per
// frame.
const MetalRender::WavefrontVariant* MetalRender::wavefrontVariantFor(uint32_t features)
{
    const auto it = mWavefrontVariants.find(features);
    if (it != mWavefrontVariants.end())
    {
        return &it->second;
    }
    if (!mWavefrontLibrary)
    {
        return nullptr;
    }

    MTL::FunctionConstantValues* values = MTL::FunctionConstantValues::alloc()->init();
    const bool envMap = (features & kFeatureEnvMap) != 0;
    const bool lights = (features & kFeatureLights) != 0;
    const bool motionBlur = (features & kFeatureMotionBlur) != 0;
    const bool dof = (features & kFeatureDof) != 0;
    const bool debug = (features & kFeatureDebug) != 0;
    const bool alpha = (features & kFeatureAlpha) != 0;
    const bool fog = (features & kFeatureFog) != 0;
    values->setConstantValue(&envMap, MTL::DataTypeBool, (NS::UInteger)0);
    values->setConstantValue(&lights, MTL::DataTypeBool, (NS::UInteger)1);
    values->setConstantValue(&motionBlur, MTL::DataTypeBool, (NS::UInteger)2);
    values->setConstantValue(&dof, MTL::DataTypeBool, (NS::UInteger)3);
    values->setConstantValue(&debug, MTL::DataTypeBool, (NS::UInteger)4);
    values->setConstantValue(&alpha, MTL::DataTypeBool, (NS::UInteger)5);
    values->setConstantValue(&fog, MTL::DataTypeBool, (NS::UInteger)6);
    const bool sharc = (features & kFeatureSharc) != 0;
    values->setConstantValue(&sharc, MTL::DataTypeBool, (NS::UInteger)7);

    // A pipeline built the Metal 3 way cannot be used with an argument table, so
    // the two paths need separate pipelines and the mode is part of the cache key.
    const bool useMetal4 = (features & kFeatureMetal4) != 0;
    NS::Error* err = nullptr;
    auto make = [&](const char* name) -> MTL::ComputePipelineState* {
        if (useMetal4)
        {
            return mMetal4.newComputePipelineState(mWavefrontLibrary, name, values);
        }
        MTL::Function* fn = mWavefrontLibrary->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding), values, &err);
        if (!fn)
        {
            STRELKA_FATAL("wavefront: specialising {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
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

    // The alpha-shadow intersection function has to be linked into the pipelines
    // that call it, and the pipeline then hands out a table to bind it through.
    MTL::Function* anyHitFn = nullptr;
    MTL::LinkedFunctions* linked = nullptr;
    if (alpha && !useMetal4)
    {
        anyHitFn = mWavefrontLibrary->newFunction(
            NS::String::string("shadowAlphaAnyHit", NS::UTF8StringEncoding), values, &err);
        if (anyHitFn)
        {
            const NS::Object* fns[] = { anyHitFn };
            linked = MTL::LinkedFunctions::alloc()->init();
            linked->setFunctions(NS::Array::array(fns, 1));
        }
        else
        {
            STRELKA_ERROR("wavefront: specialising shadowAlphaAnyHit -> {}",
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
    }
    auto makeLinked = [&](const char* name) -> MTL::ComputePipelineState* {
        if (useMetal4)
        {
            // Metal 4 states linking through descriptors, so it never builds the
            // MTL::Function above and cannot go through the branch below.
            return alpha ? mMetal4.newComputePipelineStateLinked(mWavefrontLibrary, name,
                                                                 "shadowAlphaAnyHit", values)
                         : make(name);
        }
        if (!linked)
        {
            return make(name);
        }
        MTL::Function* fn = mWavefrontLibrary->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding), values, &err);
        if (!fn)
        {
            STRELKA_FATAL("wavefront: specialising {} -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
            return nullptr;
        }
        MTL::ComputePipelineDescriptor* desc = MTL::ComputePipelineDescriptor::alloc()->init();
        desc->setComputeFunction(fn);
        desc->setLinkedFunctions(linked);
        MTL::ComputePipelineState* pso =
            mDevice->newComputePipelineState(desc, MTL::PipelineOptionNone, nullptr, &err);
        if (!pso)
        {
            STRELKA_FATAL("wavefront: {} with linked functions -> {}", name,
                          err ? err->localizedDescription()->utf8String() : "unknown error");
        }
        desc->release();
        fn->release();
        return pso;
    };

    WavefrontVariant v;
    v.generate = make("wavefrontGenerate");
    v.extendMotion = make("wavefrontExtend");
    v.extendStatic = make("wavefrontExtendStatic");
    v.shade = make("wavefrontShade");
    v.miss = make("wavefrontMiss");
    v.sharcDeposit = make("wavefrontSharcDeposit");
    v.shadowMotion = makeLinked("wavefrontShadow");
    v.shadowStatic = makeLinked("wavefrontShadowStatic");

    // One table per pipeline: it is created from the pipeline that will bind it,
    // and the two shadow pipelines are different pipelines.
    if ((linked && anyHitFn) || (useMetal4 && alpha))
    {
        auto makeTable = [&](MTL::ComputePipelineState* pso) -> MTL::IntersectionFunctionTable* {
            if (!pso)
                return nullptr;
            MTL::IntersectionFunctionTableDescriptor* d =
                MTL::IntersectionFunctionTableDescriptor::alloc()->init();
            d->setFunctionCount(1);
            MTL::IntersectionFunctionTable* table = pso->newIntersectionFunctionTable(d);
            d->release();
            if (!table)
                return nullptr;
            // By name on the Metal 4 path: linking there is stated with
            // descriptors, so there is no MTL::Function to ask for a handle.
            MTL::FunctionHandle* handle =
                anyHitFn ? pso->functionHandle(anyHitFn)
                         : pso->functionHandle(
                               NS::String::string("shadowAlphaAnyHit", NS::UTF8StringEncoding));
            if (!handle)
            {
                STRELKA_ERROR("wavefront: no function handle for shadowAlphaAnyHit");
                table->release();
                return nullptr;
            }
            table->setFunction(handle, 0);
            return table;
        };
        v.shadowTableMotion = makeTable(v.shadowMotion);
        v.shadowTableStatic = makeTable(v.shadowStatic);
        mMetal4ResidencyDirty = true;
    }
    if (anyHitFn)
        anyHitFn->release();
    if (linked)
        linked->release();
    values->release();

    if (!v.shade)
    {
        return nullptr;
    }
    STRELKA_INFO("wavefront variant env={} lights={} motion={} dof={} debug={} alpha={} fog={} metal4={}",
                 envMap, lights, motionBlur, dof, debug, alpha, fog, useMetal4);
    // Every pipeline's threadgroup limit, not just two of them.
    //
    // This is the only figure the public API gives on register pressure -- the
    // driver's own answer to how many threads fit -- and against the 1024 a
    // register-light kernel reaches it reads directly: 640 is roughly 1.6x the
    // registers per thread, 384 is 2.7x. There is no breakdown of what they
    // hold anywhere in Metal; the way to find that is to remove something and
    // watch this number, and having all of them at once makes each rebuild
    // answer for the whole renderer rather than for one kernel.
    auto tgLimit = [](MTL::ComputePipelineState* p) -> uint32_t {
        return p ? (uint32_t)p->maxTotalThreadsPerThreadgroup() : 0u;
    };
    STRELKA_INFO("  maxThreadsPerTG: generate {} extend {} (motion {}) shade {} shadow {} (motion {}) "
                 "miss {} sharcDeposit {}",
                 tgLimit(v.generate), tgLimit(v.extendStatic), tgLimit(v.extendMotion),
                 tgLimit(v.shade), tgLimit(v.shadowStatic), tgLimit(v.shadowMotion),
                 tgLimit(v.miss), tgLimit(v.sharcDeposit));
    return &mWavefrontVariants.emplace(features, v).first->second;
}

void MetalRender::buildWavefrontPipelines()
{
    MTL::Library* lib = loadShaderLibrary("metal/shaders/wavefront.metallib");
    if (!lib)
    {
        return;
    }
    mWavefrontLibrary = lib->retain();
    NS::Error* err = nullptr;
    // Only the kernels that reference no function constants are built here. The
    // rest are specialised per scene by wavefrontVariantFor(), and Metal refuses
    // to build a pipeline from an unspecialised function that declares any.
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
    mWavefrontResolvePSO = make("wavefrontResolve");
    mWavefrontPreparePSO = make("wavefrontPrepare");
    mWavefrontPrepareShadowPSO = make("wavefrontPrepareShadow");
    mWavefrontPrepareHitMissPSO = make("wavefrontPrepareHitMiss");
    mAovResolvePSO = make("wavefrontAovResolve");
    if (mMetal4.isValid())
    {
        // The same four stages again, built by the other compiler: a pipeline is
        // tied to the binding model it was compiled for.
        mWavefrontResolvePSO4 = mMetal4.newComputePipelineState(lib, "wavefrontResolve", nullptr);
        mWavefrontPreparePSO4 = mMetal4.newComputePipelineState(lib, "wavefrontPrepare", nullptr);
        mWavefrontPrepareShadowPSO4 = mMetal4.newComputePipelineState(lib, "wavefrontPrepareShadow", nullptr);
        mWavefrontPrepareHitMissPSO4 = mMetal4.newComputePipelineState(lib, "wavefrontPrepareHitMiss", nullptr);
        // The guide resolve too: without it the Metal 4 path cannot feed either
        // MetalFX mode, both of which read depth and motion from these textures.
        mAovResolvePSO4 = mMetal4.newComputePipelineState(lib, "wavefrontAovResolve", nullptr);
    }
    lib->release();
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
    release(mPathRayBuffer);
    release(mHitBuffer);
    release(mIorStackBuffer);
    release(mRadianceBuffer);
    release(mGuideRadianceBuffer);
    release(mPathQueueBuffer[0]);
    release(mPathQueueBuffer[1]);
    release(mWavefrontControlBuffer);
    release(mShadowRayBuffer);
    release(mStageStatsBuffer);
    release(mHitQueueBuffer);
    release(mAovBuffer);
    release(mMissQueueBuffer);


    // Private storage: these never leave the GPU.
    mPathStateBuffer = mDevice->newBuffer(pixels * sizeof(PathState), MTL::ResourceStorageModePrivate);
    mPathRayBuffer = mDevice->newBuffer(pixels * sizeof(PathRay), MTL::ResourceStorageModePrivate);
    mHitBuffer = mDevice->newBuffer(pixels * sizeof(HitRecord), MTL::ResourceStorageModePrivate);
    mIorStackBuffer = mDevice->newBuffer(pixels * sizeof(IorStack), MTL::ResourceStorageModePrivate);
    mRadianceBuffer = mDevice->newBuffer(pixels * sizeof(simd::float4), MTL::ResourceStorageModePrivate);
    mGuideRadianceBuffer = mDevice->newBuffer(pixels * sizeof(simd::float4), MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[0] = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mPathQueueBuffer[1] = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // Queue counters, active counts, and two sets of indirect dispatch arguments.
    mWavefrontControlBuffer = mDevice->newBuffer(96 * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    // At most one deferred connection per path per bounce.
    mShadowRayBuffer = mDevice->newBuffer(pixels * sizeof(ShadowRay), MTL::ResourceStorageModePrivate);
    mStageStatsBuffer = mDevice->newBuffer(96 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    mAovBuffer = mDevice->newBuffer(pixels * sizeof(AovSample), MTL::ResourceStorageModePrivate);
    mHitQueueBuffer = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);
    mMissQueueBuffer = mDevice->newBuffer(pixels * sizeof(uint32_t), MTL::ResourceStorageModePrivate);

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
    {
        NS::Error* e2 = nullptr;
        MTL::Function* fn = pComputeLibrary->newFunction(
            NS::String::string("toneMappingTextureShader", NS::UTF8StringEncoding));
        mTonemapperTexPSO = fn ? mDevice->newComputePipelineState(fn, &e2) : nullptr;
        if (fn) fn->release();
    }
    if (mMetal4.isValid())
    {
        mTonemapperPSO4 = mMetal4.newComputePipelineState(pComputeLibrary, "toneMappingComputeShader", nullptr);
    }
    if (!mTonemapperPSO)
    {
        STRELKA_FATAL("{}", pError ? pError->localizedDescription()->utf8String() : "unknown error");
        assert(false);
    }

    pTonemapperFn->release();
    pComputeLibrary->release();
}

void MetalRender::uploadLightBuffer()
{
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();
    static_assert(sizeof(Scene::Light) == sizeof(UniformLight));
    const size_t lightBufferSize = sizeof(Scene::Light) * lightDescs.size();

    if (lightBufferSize == 0)
    {
        if (mLightBuffer)
        {
            mLightBuffer->release();
            mLightBuffer = nullptr;
        }
        return;
    }

    if (!mLightBuffer || mLightBuffer->length() < lightBufferSize)
    {
        if (mLightBuffer)
            mLightBuffer->release();
        mLightBuffer = mDevice->newBuffer(lightBufferSize, MTL::ResourceStorageModeManaged);
    }
    memcpy(mLightBuffer->contents(), lightDescs.data(), lightBufferSize);
    mLightBuffer->didModifyRange(NS::Range::Make(0, lightBufferSize));
}

void MetalRender::handleSceneChanges()
{
    SharedContext& ctx = getSharedContext();
    const ChangeBits changes = mScene->peekChanges();
    if (!any(changes))
        return;

    bool needReset = false;
    if (any(changes & ChangeBits::Lights))
    {
        uploadLightBuffer();
        needReset = true;
    }
    if (any(changes & ChangeBits::Transforms))
    {
        if (!mBlasList.empty())
            rebuildTLAS();
        needReset = true;
    }
    if (any(changes & ChangeBits::Materials))
    {
        createMetalMaterials();
        needReset = true;
    }
    if (any(changes & ChangeBits::Env))
    {
        const auto& envLight = mScene->getEnvLight();
        if (envLight.has_value() && !envLight->texturePath.empty())
        {
            const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
            const fs::path envTexPath = fs::path(resourcePathStr) / envLight->texturePath;
            loadEnvMap(envTexPath.string());
        }
        needReset = true;
    }

    mScene->consumeChanges();
    if (needReset)
    {
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
    }
}

void MetalRender::buildBuffers()
{
    if (mScene->hostGeometryReleased())
    {
        // Rebuilding from arrays that were handed back would quietly replace the
        // buffers with empty ones and render nothing.
        STRELKA_ERROR("buildBuffers() after releaseHostGeometry(); reload the scene instead");
        return;
    }
    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    const size_t vertexDataSize = sizeof(Scene::Vertex) * vertices.size();
    const size_t indexDataSize = sizeof(uint32_t) * indices.size();

    uploadLightBuffer();

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

    mVertexBuffer = pVertexBuffer;
    mIndexBuffer = pIndexBuffer;

    // The previous shutter keyframe, for motion blur and for the denoiser's
    // reprojection. It is a second copy of every vertex in the scene -- 1.66 GB
    // on the pine forest -- and it only ever differs from the current one where
    // something deforms. A scene with no skeletal geometry can share the buffer
    // instead of duplicating it, which is the difference between that scene
    // fitting in memory and not.
    bool anySkeletal = false;
    for (const oka::Mesh& mesh : mScene->getMeshes())
    {
        anySkeletal = anySkeletal || mesh.isSkeletal;
    }
    if (vertexDataSize > 0 && anySkeletal)
    {
        mPrevVertexBuffer = mDevice->newBuffer(vertexDataSize, MTL::ResourceStorageModeManaged);
        memcpy(mPrevVertexBuffer->contents(), vertices.data(), vertexDataSize);
        mPrevVertexBuffer->didModifyRange(NS::Range::Make(0, mPrevVertexBuffer->length()));
        mOwnsPrevVertexBuffer = true;
    }
    else
    {
        mPrevVertexBuffer = mVertexBuffer;
        mOwnsPrevVertexBuffer = false;
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
    if (!accelerationStructure)
    {
        // newAccelerationStructure returns nil when the device cannot find the
        // memory, and nothing downstream notices: the TLAS ends up null, every
        // ray misses, and the result is a black frame that looks like a lighting
        // problem. Say it here instead.
        STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                      "{:.2f} GB max buffer. The scene does not fit -- lower "
                      "render/texture/maxDimension or reduce geometry.",
                      accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
        return nullptr;
    }
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
    if (commandBuffer->status() == MTL::CommandBufferStatusError)
    {
        // Out of device memory during a build leaves the structure allocated but
        // empty, so every ray misses and the frame comes out black with nothing
        // in the log. This is the only place that failure is visible.
        NS::Error* err = commandBuffer->error();
        STRELKA_ERROR("Acceleration structure build failed on the GPU: {}",
                      err && err->localizedDescription()
                          ? err->localizedDescription()->utf8String()
                          : "unknown error (most likely out of device memory)");
    }

    const uint32_t compactedSize = *(uint32_t*)compactedSizeBuffer->contents();

    // commandBuffer->release();
    // commandEncoder->release();

    // Allocate a smaller acceleration structure based on the returned size.
    MTL::AccelerationStructure* compactedAccelerationStructure = mDevice->newAccelerationStructure(compactedSize);
    if (!compactedAccelerationStructure)
    {
        STRELKA_ERROR("Compacted acceleration structure allocation failed: {:.2f} GB requested",
                      compactedSize / 1e9);
        accelerationStructure->release();
        scratchBuffer->release();
        compactedSizeBuffer->release();
        pPool->release();
        return nullptr;
    }

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


// Close the current group: end its encoder, commit, and drop the scratch the
// group was using. Metal retains what a committed command buffer references, so
// releasing here hands the buffers to the GPU's lifetime rather than ours.
void MetalRender::flushAccelerationStructureGroup()
{
    if (!mAsGroupCommandBuffer)
    {
        return;
    }
    mAsGroupEncoder->endEncoding();
    mAsGroupCommandBuffer->commit();
    mAsGroupEncoder->release();
    mAsGroupCommandBuffer->release();
    mAsGroupEncoder = nullptr;
    mAsGroupCommandBuffer = nullptr;
    for (MTL::Buffer* b : mAsGroupScratch)
    {
        b->release();
    }
    mAsGroupScratch.clear();
    mAsGroupPending = 0;
}

static double sBlasSizesMs = 0.0, sBlasAllocMs = 0.0, sBlasScratchMs = 0.0, sBlasEncodeMs = 0.0;
static uint32_t sBlasCount = 0;

MTL::AccelerationStructure* MetalRender::createAccelerationStructureNoCompact(
    MTL::AccelerationStructureDescriptor* descriptor)
{
    // The usage flags belong to the caller. This used to force Refit on
    // everything it was handed, which silently gave static geometry a tree built
    // to survive a vertex update -- a worse tree to traverse -- for nothing.
    const auto tSizes = std::chrono::steady_clock::now();
    const MTL::AccelerationStructureSizes accelSizes = mDevice->accelerationStructureSizes(descriptor);
    const auto tAlloc = std::chrono::steady_clock::now();
    MTL::AccelerationStructure* accelerationStructure =
        mDevice->newAccelerationStructure(accelSizes.accelerationStructureSize);
    const auto tScratch = std::chrono::steady_clock::now();
    if (!accelerationStructure)
    {
        // newAccelerationStructure returns nil when the device cannot find the
        // memory, and nothing downstream notices: the TLAS ends up null, every
        // ray misses, and the result is a black frame that looks like a lighting
        // problem. Say it here instead.
        STRELKA_ERROR("Acceleration structure allocation failed: {:.2f} GB requested, "
                      "{:.2f} GB max buffer. The scene does not fit -- lower "
                      "render/texture/maxDimension or reduce geometry.",
                      accelSizes.accelerationStructureSize / 1e9, mDevice->maxBufferLength() / 1e9);
        return nullptr;
    }
    MTL::Buffer* scratchBuffer =
        mDevice->newBuffer(accelSizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    const auto tEncode = std::chrono::steady_clock::now();

    // Builds are grouped: STRELKA_AS_GROUP structures share one encoder, each
    // with its own scratch, and the group is committed together. Apple's guidance
    // is that several builds in *one encoder* run in parallel on this hardware,
    // which needs a scratch buffer per build rather than one reused -- two builds
    // sharing scratch is a data race, not a slow path. Group size 1 is the
    // one-per-command-buffer arrangement this replaces.
    static const uint32_t kGroupSize = [] {
        const char* v = getenv("STRELKA_AS_GROUP");
        return v ? std::max(1u, (uint32_t)atoi(v)) : 1u;
    }();

    if (!mAsGroupCommandBuffer)
    {
        mAsGroupCommandBuffer = mCommandQueue->commandBuffer()->retain();
        mAsGroupEncoder = mAsGroupCommandBuffer->accelerationStructureCommandEncoder()->retain();
    }
    mAsGroupEncoder->buildAccelerationStructure(accelerationStructure, descriptor, scratchBuffer, 0UL);
    // Held until the group is committed: the GPU reads it for the whole build,
    // and dropping the last reference before commit is what a race would look
    // like if Metal did not retain committed resources.
    mAsGroupScratch.push_back(scratchBuffer);
    if (++mAsGroupPending >= kGroupSize)
    {
        flushAccelerationStructureGroup();
    }
    // No waitUntilCompleted — Metal queue ordering guarantees subsequent
    // command buffers on the same queue see the built AS.

    {
        using ms = std::chrono::duration<double, std::milli>;
        const auto tEnd = std::chrono::steady_clock::now();
        sBlasSizesMs += ms(tAlloc - tSizes).count();
        sBlasAllocMs += ms(tScratch - tAlloc).count();
        sBlasScratchMs += ms(tEncode - tScratch).count();
        sBlasEncodeMs += ms(tEnd - tEncode).count();
        ++sBlasCount;
    }
    return accelerationStructure;
}

void MetalRender::createMeshData(size_t meshIndex)
{
    const oka::Mesh& mesh = mScene->getMeshes()[meshIndex];
    auto result = new MetalRender::Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;
    result->mTriangleCount = triangleCount;
    result->mVbOffset = mesh.mVbOffset;
    result->mIsSkeletal = mesh.isSkeletal;

    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    // Per-primitive data is a second copy of every triangle's attributes, 72
    // bytes each, held both here and inside the acceleration structure Metal
    // builds from it. Only the megakernel reads it: the wavefront tracer cannot,
    // because intersection.primitive_data is addressable solely inside the
    // kernel that ran the intersect, so its shade stage refetches attributes
    // from the vertex buffer instead (see fetchTriangle in wavefront.metal).
    //
    // Skipping it under the wavefront tracer is worth far more than it sounds:
    // on a 50 M triangle forest it is 3.7 GB of host memory, the same again
    // inside the acceleration structures, and the CPU time to fill it.
    if (!mNeedsPrimitiveData)
    {
        mMetalMeshes.push_back(result);
        return;
    }

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
    if (perPrimitiveBuffer)
    {
        geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
        geomDescriptor->setPrimitiveDataBufferOffset(0);
        geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
        geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));
    }

    return geomDescriptor;
}

// Build one acceleration structure covering every listed instance's mesh as a
// separate geometry, and record the per-geometry lookup entries.
// Metal 4's flag asking the builder to trade build time and structure size for
// traversal speed. A separate bit from Refit, so it composes with the one this
// renderer already needs for skinning.
//
// On by default because it measured as a free 4.1% on the pine forest: 48.94 ms
// against 46.92 (n=8 each, two mirrored ABBA blocks to cancel thermal drift,
// t=3.14), with the image unchanged and the structures the same 5.46 GB -- the
// concern that a scene this close to the buffer limit could not afford it did
// not materialise. NVIDIA report the same order for PREFER_FAST_TRACE.
//
// The escape hatch is here because that is one scene on one driver, and the
// flag is exactly the kind of hint whose sign can change with either.
static MTL::AccelerationStructureUsage blasExtraUsage()
{
    static const bool disabled = getenv("STRELKA_NO_PREFER_FAST_INTERSECTION") != nullptr;
    return disabled ? MTL::AccelerationStructureUsageNone
                    : MTL::AccelerationStructureUsagePreferFastIntersection;
}


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

        // Opaque unless this geometry's own material has a cutout. Leaving the
        // instance flag alone lets this decide, and traversal then resolves a
        // trunk or a rock without ever leaving the ray tracing unit.
        // STRELKA_ALL_GEOM_OPAQUE=1 forces every geometry opaque -- a diagnostic
        // for whether the alpha test is running at all, not a rendering mode.
        static const bool forceAllOpaque = getenv("STRELKA_ALL_GEOM_OPAQUE") != nullptr;
        const bool isCutout = !forceAllOpaque && inst.mMaterialId < mMaterialIsCutout.size() &&
                              mMaterialIsCutout[inst.mMaterialId] != 0;

        MTL::AccelerationStructureGeometryDescriptor* geom = nullptr;
        if (skeletal && mBuildMotionBlas)
        {
            geom = createMotionGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount);
        }
        else
        {
            geom = createStaticGeometryDescriptor(mesh, meshData->mPerPrimitiveBuffer, meshData->mTriangleCount);
        }
        // STRELKA_NO_SET_OPAQUE=1 leaves the descriptor's own default in place,
        // which is what the code did before this flag existed -- the control for
        // telling "the alpha test changed" apart from "the alpha test started".
        static const bool leaveDefault = getenv("STRELKA_NO_SET_OPAQUE") != nullptr;
        if (!leaveDefault)
        {
            geom->setOpaque(!isCutout);
        }
        (isCutout ? mCutoutGeometryCount : mOpaqueGeometryCount)++;
        geomDescriptors.push_back(geom);

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
        if (mBuildMotionBlas)
        {
            primDescriptor->setMotionKeyframeCount(2);
            primDescriptor->setMotionStartTime(0.0f);
            primDescriptor->setMotionEndTime(1.0f);
            primDescriptor->setMotionStartBorderMode(MTL::MotionBorderModeClamp);
            primDescriptor->setMotionEndBorderMode(MTL::MotionBorderModeClamp);
        }
        primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit | blasExtraUsage());

        const MTL::AccelerationStructureSizes sizes = mDevice->accelerationStructureSizes(primDescriptor);
        blas.mRefitScratchSize = sizes.refitScratchBufferSize;
        blas.mBuildScratchSize = sizes.buildScratchBufferSize;

        blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
        blas.mDescriptor = primDescriptor; // kept for refit, released in the destructor
    }
    else
    {
        // Refit even though nothing here is ever refitted, and no compaction.
        //
        // Both of those are backwards from what the flags suggest, and both are
        // measured. Refit is meant to trade tree quality for the ability to
        // update in place; on this driver it also halves what the builder
        // allocates -- 4.80 GB of structures against 8.80 without, which on a
        // 9.53 GB buffer limit is the difference between fitting and not -- and
        // the result traverses no slower.
        //
        // Compaction is a loss on every axis here. It needs a
        // round trip per structure to read the compacted size back, and with a
        // hundred and thirty of them that was 4.9 seconds of a 13.8 second load.
        // It did not even save memory: 4.80 GB of acceleration structures
        // without it against 4.97 GB with. And the uncompacted trees traverse
        // faster -- 27.9 s against 29.8 for 128 spp of the pine forest.
        primDescriptor->setUsage(MTL::AccelerationStructureUsageRefit | blasExtraUsage());
        blas.mAs = createAccelerationStructureNoCompact(primDescriptor);
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


// Rebuild the acceleration structures for a different motion setting.
//
// Only reachable from the motion-blur toggle, which is a UI action, so a hitch is
// acceptable — but the structures about to be released may still be referenced by
// the last frame's command buffers, hence the drain.
void MetalRender::rebuildAccelerationStructures()
{
    MTL::CommandBuffer* drain = mCommandQueue->commandBuffer();
    drain->retain();
    drain->commit();
    drain->waitUntilCompleted();
    drain->release();

    auto safeRelease = [](auto*& p) {
        if (p)
        {
            p->release();
            p = nullptr;
        }
    };
    for (Blas& blas : mBlasList)
    {
        safeRelease(blas.mScratch);
        safeRelease(blas.mDescriptor);
    }
    mBlasList.clear();
    // blas.mAs and the entries here are the same objects; release through one path.
    for (auto*& as : mPrimitiveAccelerationStructures)
    {
        safeRelease(as);
    }
    mPrimitiveAccelerationStructures.clear();
    safeRelease(mInstanceAccelerationStructure);
    safeRelease(mInstanceBuffer);
    safeRelease(mGeometryEntryBuffer);
    safeRelease(mTlasScratchBuffer);

    const auto rebuildStart = std::chrono::high_resolution_clock::now();
    createAccelerationStructures();
    STRELKA_INFO("Acceleration structures rebuilt for motion={} in {:.1f} ms", mBuildMotionBlas,
                 std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - rebuildStart)
                     .count());
}

// How much work one call of the sliced stages may do. Long enough that the
// per-slice overhead is noise, short enough that the window still answers the
// mouse -- one slice per displayed frame, at a frame that is not a fast one.
static constexpr double kBuildSliceMs = 24.0;

// One stage of the scene build per call; see BuildStage in the header for why it
// is split at all. The stages are ordered by dependency, not by cost: materials
// index into the buffers and the acceleration structures reference both.
bool MetalRender::stepSceneBuild(Buffer* output)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    const auto started = std::chrono::steady_clock::now();
    const BuildStage ran = mBuildStage;

    switch (mBuildStage)
    {
    case BuildStage::Buffers:
        // New scene: nothing from before relates to it.
        mResetDenoiseHistory = true;
        mHasPrevCamera = false;
        mHasPrevFramePose = false;
        mShutterIntervalActive = false;
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Geometry);
        }
        buildBuffers();
        mBuildStage = BuildStage::Materials;
        break;

    case BuildStage::Materials:
        if (mLoadProgress && !mMaterialBuild)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Textures,
                                      (uint32_t)mScene->getMaterials().size());
        }
        if (stepMetalMaterials(kBuildSliceMs))
        {
            mBuildStage = BuildStage::Structures;
        }
        break;

    case BuildStage::Structures:
        if (mLoadProgress && !mAsBuild)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Structures);
        }
        // Nothing is playing yet, so start static; the per-frame check in render()
        // switches to motion structures if playback begins.
        mBuildMotionBlas = false;
        // The one stage too long to run whole: two seconds on the pine forest,
        // against roughly one frame's worth per slice here. The stage stays
        // current until the build says it is finished.
        if (stepAccelerationStructures(kBuildSliceMs))
        {
            mBuildStage = BuildStage::Tail;
        }
        break;

    case BuildStage::Tail:
    {
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Environment);
        }
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
            if (!envLight->backgroundTexturePath.empty())
            {
                loadEnvBackground((fs::path(resourcePathStr) / envLight->backgroundTexturePath).string());
            }
        }
        // Fresh scene: drop any pending edit bits from load-time createLight.
        mScene->consumeChanges();
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Done);
        }
        mBuildStage = BuildStage::Done;
        // What the scene actually cost, once. The per-category figures scattered
        // through the build are estimates made at allocation time; this is the
        // sizes the API reports, and it is the only place the two totals -- the
        // device's and the OS's -- can be seen next to each other.
        {
            MemoryReport report;
            if (memoryReport(report))
            {
                std::sort(report.gpu.begin(), report.gpu.end(),
                          [](const MemoryReport::Entry& a, const MemoryReport::Entry& b) {
                              return a.bytes > b.bytes;
                          });
                std::string top;
                for (size_t i = 0; i < std::min<size_t>(4, report.gpu.size()); ++i)
                {
                    top += fmt::format("{}{} {:.2f} GB", i ? ", " : "", report.gpu[i].name,
                                       report.gpu[i].bytes / 1073741824.0);
                }
                STRELKA_INFO("Memory: device {:.2f} GB, process {:.2f} GB; largest: {}",
                             report.deviceAllocated / 1073741824.0, report.processFootprint / 1073741824.0, top);
            }
        }
        break;
    }

    case BuildStage::Done:
        break;
    }

    if (ran != BuildStage::Done)
    {
        static const char* kStageNames[] = { "buffers", "materials", "structures", "tail" };
        STRELKA_DEBUG("Scene build stage '{}' took {:.0f} ms", kStageNames[(uint32_t)ran],
                      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    pPool->release();
    return mBuildStage == BuildStage::Done;
}

void MetalRender::finishSceneBuild(Buffer* output)
{
    while (mBuildStage != BuildStage::Done)
    {
        stepSceneBuild(output);
    }
}

void MetalRender::createAccelerationStructures()
{
    // No budget: one call does the lot, which is what a rebuild driven by an
    // animation and what the headless path both want.
    while (!stepAccelerationStructures(0.0))
    {
    }
}

bool MetalRender::stepAccelerationStructures(double budgetMs)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    auto finish = [&](bool complete) {
        pPool->release();
        return complete;
    };

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        delete mAsBuild;
        mAsBuild = nullptr;
        return finish(true);
    }

    using Phase = AsBuildState::Phase;
    if (!mAsBuild)
    {
        mAsBuild = new AsBuildState();
    }
    AsBuildState& st = *mAsBuild;

    // The clock is read every so many iterations rather than on each one: for the
    // cheap majority of items -- a group that shares an existing structure --
    // reading it costs more than the work it is measuring.
    constexpr size_t kCheckEvery = 64;
    const auto sliceStart = std::chrono::steady_clock::now();
    auto mark = sliceStart;
    auto chargePhase = [&]() {
        const auto t = std::chrono::steady_clock::now();
        st.phaseMs[(size_t)st.phase] += std::chrono::duration<double, std::milli>(t - mark).count();
        mark = t;
    };
    // Grouping and building are one item per scene instance each, so the two
    // cursors against twice the instance count is close enough for a bar -- the
    // group count is within a percent of the instance count on any scene where
    // this stage is long enough to matter.
    auto report = [&]() {
        if (mLoadProgress)
        {
            mLoadProgress->total.store((uint32_t)(instances.size() * 2), std::memory_order_relaxed);
            mLoadProgress->done.store((uint32_t)(st.groupCursor + st.blasCursor), std::memory_order_relaxed);
        }
    };
    // `force` is for the items that are not cheap: one BLAS build can outlast the
    // whole slice budget on its own, and sampling the clock every kCheckEvery
    // items would then let 511 more of them through behind it.
    auto outOfTime = [&](size_t iteration, bool force = false) {
        if (budgetMs <= 0.0 || (!force && (iteration % kCheckEvery) != 0))
        {
            return false;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sliceStart).count() >=
               budgetMs;
    };

    if (st.phase == Phase::Meshes)
    {
        // Per-mesh buffers survive a rebuild of the structures that reference them.
        if (mMetalMeshes.empty())
        {
            // Read once, here: the meshes are built now and the acceleration
            // structures embed whatever they are given, so a later change of tracer
            // cannot retroactively add the data.
            // Nothing reads per-primitive data now: it existed for the megakernel.
            mNeedsPrimitiveData = false;
            while (st.meshCursor < meshes.size())
            {
                createMeshData(st.meshCursor);
                st.primitiveBytes += (size_t)(meshes[st.meshCursor].mCount / 3) * sizeof(Triangle);
                ++st.meshCursor;
                if (outOfTime(st.meshCursor))
                {
                    chargePhase();
                    report();
                    return finish(false);
                }
            }
            if (!mNeedsPrimitiveData && st.primitiveBytes > 0)
            {
                STRELKA_INFO("Skipped {:.2f} GB of per-primitive attribute data: the wavefront tracer "
                             "refetches from the vertex buffer",
                             st.primitiveBytes / 1e9);
            }
        }
        mMotionBlasBuilt = mBuildMotionBlas;

        // Before the structures are built, not after. The host arrays are a full
        // duplicate of the GPU buffers -- 2.27 GB here -- and holding them across the
        // acceleration structure build stacks that on top of the largest allocation
        // the renderer makes. Nothing below this point reads them: buildBlas() works
        // from buffer offsets and triangle counts.
        //
        // Off by default because Scene::pick() walks these arrays -- the editor needs
        // them, a headless render does not.
        mHostGeometryBytes = { mScene->getVertices().size() * sizeof(Scene::Vertex),
                               mScene->getIndices().size() * sizeof(uint32_t) };
        if (getSettings()->getAs<bool>("scene/releaseHostGeometry") && !mScene->hostGeometryReleased())
        {
            mScene->releaseHostGeometry();
        }

        // --- Group mesh instances that always move together -----------------------
        //
        // glTF splits a mesh into primitives by material, and the loader turns each
        // primitive into its own instance. Left alone that produces one BLAS per
        // primitive, all with the same transform and heavily overlapping bounds, and
        // every ray that has to traverse deeply pays for the overlap. Instances that
        // hang off the same node share a transform by construction, so they can be
        // merged into a single BLAS with one geometry per primitive.
        st.instanceNode.assign(instances.size(), -1);
        const std::vector<Scene::Node>& nodes = mScene->getNodes();
        for (size_t n = 0; n < nodes.size(); ++n)
        {
            for (const uint32_t id : nodes[n].instanceIds)
            {
                if (id < st.instanceNode.size())
                    st.instanceNode[id] = (int)n;
            }
        }

        mGeometryEntries.clear();
        mEmittedInstances.clear();

        chargePhase();
        st.phase = Phase::Grouping;
    }

    auto hashTransform = [](const glm::mat4& m) {
        uint64_t h = 1469598103934665603ull;
        const auto* raw = reinterpret_cast<const unsigned char*>(&m);
        for (size_t i = 0; i < sizeof(glm::mat4); ++i)
        {
            h ^= raw[i];
            h *= 1099511628211ull;
        }
        return h;
    };

    if (st.phase == Phase::Grouping)
    {
        while (st.groupCursor < instances.size())
        {
            const size_t i = st.groupCursor++;
            const oka::Instance& curr = instances[i];
            if (curr.type != oka::Instance::Type::eLight) // lights are handled in their own phase
            {
                const bool skeletal = meshes[curr.mMeshId].isSkeletal;
                const int nodeId = st.instanceNode[i];
                const AsBuildState::GroupKey key{ nodeId >= 0 ? nodeId : -(int)i - 2, skeletal ? 1 : 0,
                                                  hashTransform(curr.transform) };

                auto it = st.groupOfKey.find(key);
                if (it == st.groupOfKey.end())
                {
                    st.groupOfKey[key] = st.groups.size();
                    st.groups.push_back({ (uint32_t)i });
                    st.groupSkeletal.push_back(skeletal);
                }
                // Merging is only valid while the members share a transform.
                else if (memcmp(&instances[st.groups[it->second].front()].transform, &curr.transform,
                                sizeof(glm::mat4)) == 0)
                {
                    st.groups[it->second].push_back((uint32_t)i);
                }
                else
                {
                    st.groups.push_back({ (uint32_t)i });
                    st.groupSkeletal.push_back(skeletal);
                }
            }
            if (outOfTime(st.groupCursor))
            {
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Blas;
        // The key map has done its job and is the largest thing here -- an entry
        // per scene instance. Nothing below reads it.
        st.groupOfKey.clear();
        st.instanceNode.clear();
        st.instanceNode.shrink_to_fit();
    }

    if (st.phase == Phase::Blas)
    {
        // Share one BLAS between every group that holds the same geometry.
        //
        // Scattered scenes are built almost entirely out of repeats: the pine forest
        // places 38 000 instances drawn from 50 distinct objects. A BLAS per instance
        // would be 38 000 structures over the same 50 meshes, which is both the build
        // time and the memory of a scene 700 times larger than the one authored.
        //
        // The signature is (mesh, material) per geometry rather than mesh alone,
        // because the material is baked into the shared geometry entries -- two
        // instances of the same mesh with different materials are not the same BLAS.
        // Skeletal groups are excluded: their vertices are rewritten per frame and
        // the structure refit alongside, so sharing one would mean two instances
        // deforming the same geometry.
        while (st.blasCursor < st.groups.size())
        {
            const size_t g = st.blasCursor++;
            std::vector<uint64_t> signature;
            signature.reserve(st.groups[g].size());
            for (const uint32_t id : st.groups[g])
            {
                signature.push_back(((uint64_t)instances[id].mMeshId << 32) | instances[id].mMaterialId);
            }

            size_t blasIdx;
            auto shared = st.groupSkeletal[g] ? st.blasOfSignature.end() : st.blasOfSignature.find(signature);
            const bool built = shared == st.blasOfSignature.end();
            if (!built)
            {
                blasIdx = shared->second;
                ++st.sharedBlas;
            }
            else
            {
                blasIdx = buildBlas(st.groups[g], st.groupSkeletal[g]);
                st.mergedGeometries += st.groups[g].size();
                if (!st.groupSkeletal[g])
                {
                    st.blasOfSignature.emplace(std::move(signature), blasIdx);
                }
            }

            EmittedInstance emitted{};
            emitted.sceneInstanceId = st.groups[g].front();
            emitted.asIndex = (uint32_t)blasIdx;
            emitted.userID = mBlasList[blasIdx].mGeometryBase;
            emitted.mask = GEOMETRY_MASK_TRIANGLE;

            mEmittedInstances.push_back(emitted);

            if (outOfTime(st.blasCursor, built))
            {
                // Close whatever group the last build landed in before handing
                // control back, so the scratch buffers it holds are not carried
                // across frames waiting for a group that may be many slices away
                // from filling up.
                flushAccelerationStructureGroup();
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Lights;
    }

    if (st.phase == Phase::Lights)
    {
        while (st.lightCursor < instances.size())
        {
            const size_t i = st.lightCursor++;
            const oka::Instance& curr = instances[i];
            if (curr.type == oka::Instance::Type::eLight)
            {
                auto it = st.lightBlasOfMesh.find(curr.mMeshId);
                if (it == st.lightBlasOfMesh.end())
                {
                    const size_t blasIdx = buildBlas({ (uint32_t)i }, meshes[curr.mMeshId].isSkeletal);
                    it = st.lightBlasOfMesh.emplace(curr.mMeshId, blasIdx).first;
                }
                EmittedInstance emitted{};
                emitted.sceneInstanceId = (uint32_t)i;
                emitted.asIndex = (uint32_t)it->second;
                emitted.userID = curr.mLightId; // lights address the light table, not geometry
                // Point/spot proxies exist for picking and the gizmo; they are not
                // emissive surfaces. Putting them on the light mask would treat their
                // radiant intensity as radiance and blow out the frame.
                //
                // They must not be on the geometry mask either. A point light is sampled
                // at its centre, which sits inside the proxy sphere, so a shadow ray to
                // it necessarily crosses the shell -- and RAY_MASK_SHADOW *is*
                // GEOMETRY_MASK_GEOMETRY, so every next-event connection to a point or
                // spot light was reported occluded and those lights lit nothing at all.
                // Picking runs on the CPU in Scene::pick() and never consults these
                // masks, so a proxy invisible to every ray costs nothing.
                const int lightType =
                    curr.mLightId < mScene->getLightsDesc().size() ? mScene->getLightsDesc()[curr.mLightId].type : -1;
                const bool enabled = curr.mLightId < mScene->getLightsDesc().size() ?
                                         mScene->getLightsDesc()[curr.mLightId].enabled :
                                         true;
                if (!enabled)
                    emitted.mask = 0;
                else if (lightType == LIGHT_TYPE_POINT || lightType == LIGHT_TYPE_SPOT)
                    emitted.mask = 0;
                else
                    emitted.mask = GEOMETRY_MASK_LIGHT;
                mEmittedInstances.push_back(emitted);
            }
            if (outOfTime(st.lightCursor))
            {
                flushAccelerationStructureGroup();
                chargePhase();
                report();
                return finish(false);
            }
        }
        chargePhase();
        st.phase = Phase::Finish;
        // Not needed past this point and one vector per group, so worth dropping
        // before the top level is built rather than after.
        st.groups.clear();
        st.groups.shrink_to_fit();
        st.groupSkeletal.clear();
        st.blasOfSignature.clear();
    }

    // --- Finish: buffers and the top level ------------------------------------
    //
    // Not sliced. What is left is one pass over the emitted instances and a single
    // top-level build, which measures in the tens of milliseconds even on the
    // largest scene here -- and it cannot be interrupted anyway, since the TLAS
    // descriptor has to see every structure at once.

    // Hand the host copies back once everything that reads them has run: the
    // vertex and index buffers are uploaded, the per-primitive data (if the
    // megakernel wanted any) is built, and the structures are up. On unified
    // memory the GPU-visible buffers and these vectors come out of the same pool,
    // so the duplicate is real, not bookkeeping.
    //
    // Off by default because Scene::pick() walks these arrays -- the editor needs
    // them, a headless render does not.
    const size_t vtxBytes = mHostGeometryBytes.first;
    const size_t idxBytes = mHostGeometryBytes.second;
    const bool hostFreed = mScene->hostGeometryReleased();

    // Where the memory goes. On a 50 M triangle scene the total runs past what a
    // 16 GB machine holds resident, and the first question is always which part
    // -- so state it rather than leave it to Activity Monitor.
    {
        size_t texBytes = 0;
        for (MTL::Texture* t : mMaterialTextures)
        {
            if (!t)
                continue;
            // Every level of a full mip chain adds a third again.
            texBytes += (size_t)t->width() * t->height() * 4 * 4 / 3;
        }
        STRELKA_INFO("Memory: vertices {:.2f} GB, indices {:.2f} GB, textures {:.2f} GB with mips; "
                     "host geometry {}",
                     vtxBytes / 1e9, idxBytes / 1e9, texBytes / 1e9,
                     hostFreed ? "released (picking disabled for this scene)" : "kept (doubles the first two)");
    }

    STRELKA_INFO("Acceleration structures: {} BLAS ({} geometries, {} groups shared one), "
                 "{} TLAS instances (from {} scene instances)",
                 mBlasList.size(), st.mergedGeometries, st.sharedBlas, mEmittedInstances.size(), instances.size());
    STRELKA_INFO("Geometry opacity: {} opaque, {} cutout ({:.1f}% of geometries need the alpha test)",
                 mOpaqueGeometryCount, mCutoutGeometryCount,
                 100.0 * mCutoutGeometryCount /
                     std::max<uint32_t>(1u, mOpaqueGeometryCount + mCutoutGeometryCount));

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
        // Not marked opaque when the scene has cutouts: the flag makes traversal
        // skip the intersection function, which is what performs the alpha test.
        // The kernels that do not want the test -- extend, and the shadow path of
        // a scene without cutouts -- force opacity on the intersector instead,
        // which overrides this and costs them nothing.
        instanceDescriptors[d].options = mSceneHasAlphaMaterials
                                             ? MTL::AccelerationStructureInstanceOptionNone
                                             : MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[d].intersectionFunctionTableOffset = 0;
        instanceDescriptors[d].userID = e.userID;
        instanceDescriptors[d].mask = e.mask;
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));
    updateInstanceTransforms();

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    // Every BLAS must be committed before the TLAS that references them is
    // encoded, so close whatever group the last one landed in.
    flushAccelerationStructureGroup();
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(mEmittedInstances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeUserID);
    // The top level was built with no usage flags at all, which cost two things.
    // Refit is one of them: rebuildAccelerationStructures refits this structure
    // when an animation moves instances, and refitting one not built to be
    // refittable is undefined -- canRefit checks the instance count and the size
    // and never asked what it was built as. The other is that every ray goes
    // through this structure and a million instances of it, so it wants the same
    // fast-intersection preference the bottom level already gets.
    //
    // STRELKA_TLAS_USAGE selects them for measurement: 1 = Refit, 2 = prefer fast
    // intersection, 3 = both.
    static const uint32_t kTlasUsage = [] {
        const char* v = getenv("STRELKA_TLAS_USAGE");
        return v ? (uint32_t)atoi(v) : 3u;
    }();
    accelDescriptor->setUsage(
        ((kTlasUsage & 1u) ? MTL::AccelerationStructureUsageRefit : MTL::AccelerationStructureUsageNone) |
        ((kTlasUsage & 2u) ? MTL::AccelerationStructureUsagePreferFastIntersection
                           : MTL::AccelerationStructureUsageNone));

    mInstanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    if (!mInstanceAccelerationStructure)
    {
        STRELKA_ERROR("Top-level acceleration structure could not be built; every ray will miss "
                      "and the image will be black.");
    }
    {
        size_t asBytes = 0;
        size_t nullAs = 0;
        for (const Blas& b : mBlasList)
        {
            if (b.mAs)
                asBytes += b.mAs->size();
            else
                ++nullAs;
        }
                // The largest few, because a structure that should have been shared and
        // was not is worth several gigabytes and is invisible in the total.
        {
            std::vector<std::pair<size_t, size_t>> bySize;
            bySize.reserve(mBlasList.size());
            for (size_t bi = 0; bi < mBlasList.size(); ++bi)
            {
                if (mBlasList[bi].mAs)
                    bySize.emplace_back(mBlasList[bi].mAs->size(), bi);
            }
            std::sort(bySize.rbegin(), bySize.rend());
            for (size_t k = 0; k < std::min<size_t>(5, bySize.size()); ++k)
            {
                STRELKA_INFO("  BLAS {} : {:.2f} GB, geometry base {}", bySize[k].second,
                             bySize[k].first / 1e9, mBlasList[bySize[k].second].mGeometryBase);
            }
        }
STRELKA_INFO("BLAS build CPU: sizes {:.0f} ms, alloc {:.0f} ms, scratch {:.0f} ms, encode {:.0f} ms ({} structures)",
                 sBlasSizesMs, sBlasAllocMs, sBlasScratchMs, sBlasEncodeMs, sBlasCount);
    STRELKA_INFO("Structures: BLAS {:.2f} GB ({} failed), TLAS {:.3f} GB, device max buffer {:.2f} GB",
                     asBytes / 1e9, nullAs,
                     mInstanceAccelerationStructure ? mInstanceAccelerationStructure->size() / 1e9 : 0.0,
                     mDevice->maxBufferLength() / 1e9);
    }
    mTlasInstanceCount = mEmittedInstances.size();

    chargePhase();
    STRELKA_DEBUG("Acceleration structures CPU: meshes {:.0f} ms, grouping {:.0f} ms, blas {:.0f} ms, "
                  "lights {:.0f} ms, finish {:.0f} ms",
                  st.phaseMs[(size_t)Phase::Meshes], st.phaseMs[(size_t)Phase::Grouping],
                  st.phaseMs[(size_t)Phase::Blas], st.phaseMs[(size_t)Phase::Lights],
                  st.phaseMs[(size_t)Phase::Finish]);
    delete mAsBuild;
    mAsBuild = nullptr;
    return finish(true);
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
    if (mMetal4.isValid())
    {
        mSkinningPSO4 = mMetal4.newComputePipelineState(pLibrary, "skinningKernel", nullptr);
        mTriangleUpdatePSO4 = mMetal4.newComputePipelineState(pLibrary, "updateTriangleBufferKernel", nullptr);
    }
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
    mSkinDataBuffer = mDevice->newBuffer(dataSize, MTL::ResourceStorageModeShared);
    memcpy(mSkinDataBuffer->contents(), skinData.data(), dataSize);
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
            mDevice->newBuffer(jointMatSize * sizeof(simd::float4x4), MTL::ResourceStorageModeShared);
    }
}

// The same two dispatches on the Metal 3 queue. Kept as a comparison path: the
// skinned vertices come out as exact zeros through the Metal 4 route, and the
// only way to tell a bad kernel from a bad submission is to run the same kernel
// through the other one.
void MetalRender::applySkinningMetal3()
{
    if (!mSkinningPSO || !mTriangleUpdatePSO)
    {
        return;
    }
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    int skinIndex = 0;
    int jointMatOffset = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin == -1 || node.type != oka::Scene::Node::NodeType::mesh)
        {
            continue;
        }
        if (skinIndex > 0)
        {
            jointMatOffset += mJointMatOffsets[skinIndex - 1];
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
            enc->setBuffer(mVertexBuffer, 0, 0);
            enc->setBuffer(mSkinDataBuffer, 0, 1);
            enc->setBuffer(mJointMatricesBuffer, 0, 2);
            enc->setBytes(&skinParams, sizeof(skinParams), 3);
            enc->dispatchThreads(MTL::Size(mesh.mVertexCount, 1, 1), MTL::Size(256, 1, 1));

            MetalRender::Mesh* metalMesh = mMetalMeshes[meshId];
            if (metalMesh->mPerPrimitiveBuffer)
            {
                TriangleUpdateParams triParams = {};
                triParams.triangleCount = metalMesh->mTriangleCount;
                triParams.indexOffset = mesh.mIndex;
                triParams.vbOffset = mesh.mVbOffset;

                enc->setComputePipelineState(mTriangleUpdatePSO);
                enc->setBuffer(metalMesh->mPerPrimitiveBuffer, 0, 0);
                enc->setBuffer(mVertexBuffer, 0, 1);
                enc->setBuffer(mIndexBuffer, 0, 2);
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

void MetalRender::applySkinning()
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

    // Skinning runs through the Metal 3 queue.
    //
    // The Metal 4 route below produces exact zeros for every skinned vertex --
    // the character collapses to a point and vanishes the instant playback
    // starts -- while the same kernel, over the same skin data and the same
    // joint matrices, gives a correct pose through Metal 3. Measured on
    // BrainStem: Metal 3 yields [-0.402 0.014 -0.219]..[0.459 1.137 0.215],
    // Metal 4 yields [0 0 0]..[0 0 0]. Everything the kernel reads was verified
    // healthy at the point of dispatch: 34274 skin records with no zero weights,
    // 18 joint matrices with none degenerate, joint indices within range, both
    // pipelines built, the constant ring far from exhausted, and the resources
    // resident.
    //
    // The remaining difference is the submission itself, and the leading suspect
    // is the shared argument table: it is mutated between dispatches inside one
    // encoder (BrainStem has 59 primitives), and Metal 4 has the GPU read that
    // table at execution time rather than capturing it at encode time. That is
    // not proven, so the Metal 4 path is kept and selectable rather than
    // deleted -- but it is not what runs by default until it is right.
    if (!mSkinMetal4)
    {
        applySkinningMetal3();
        return;
    }

    // Dispatch skinning + triangle update kernels
    MTL4::CommandBuffer* pCmd = mMetal4.beginImmediate();
    MTL4::ComputeCommandEncoder* pEncoder = pCmd->computeCommandEncoder();
    MTL4::ArgumentTable* skinTable = mMetal4.argumentTable();
    pEncoder->setArgumentTable(skinTable);

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

                pEncoder->setComputePipelineState(mSkinningPSO4);
                skinTable->setAddress(mVertexBuffer->gpuAddress(), 0);
                skinTable->setAddress(mSkinDataBuffer->gpuAddress(), 1);
                skinTable->setAddress(mJointMatricesBuffer->gpuAddress(), 2);
                skinTable->setAddress(mMetal4.immediateConstants().push(skinParams), 3);

                const uint32_t threadsPerGroup = 256;
                const MTL::Size groupSize = MTL::Size(threadsPerGroup, 1, 1);
                pEncoder->dispatchThreadgroups(
                    MTL::Size((mesh.mVertexCount + threadsPerGroup - 1) / threadsPerGroup, 1, 1), groupSize);

                // Dispatch triangle update kernel
                MetalRender::Mesh* metalMesh = mMetalMeshes[meshId];
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
                    skinTable->setAddress(mVertexBuffer->gpuAddress(), 1);
                    skinTable->setAddress(mIndexBuffer->gpuAddress(), 2);
                    skinTable->setAddress(mMetal4.immediateConstants().push(triParams), 3);

                    pEncoder->dispatchThreadgroups(
                        MTL::Size((metalMesh->mTriangleCount + 255) / 256, 1, 1), groupSize);
                }
            }
        }
    }

    pEncoder->endEncoding();
    mMetal4.submitAndWait(pCmd);
}

void MetalRender::copyVertexBufferToPrev()
{
    const size_t vertexDataSize = mVertexBuffer->length();
    if (mMetal4.isValid())
    {
        MTL4::CommandBuffer* cmd = mMetal4.beginImmediate();
        MTL4::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->copyFromBuffer(mVertexBuffer, 0, mPrevVertexBuffer, 0, vertexDataSize);
        enc->endEncoding();
        mMetal4.submitAndWait(cmd);
        return;
    }
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    MTL::BlitCommandEncoder* enc = cmd->blitCommandEncoder();
    enc->copyFromBuffer(mVertexBuffer, 0, mPrevVertexBuffer, 0, vertexDataSize);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
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
    if (perPrimitiveBuffer)
    {
        geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
        geomDescriptor->setPrimitiveDataBufferOffset(0);
        geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
        geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));
    }

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

void MetalRender::updateSkeletalBLAS()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    // Rebuild rather than refit, every frame, up to the per-frame cap.
    //
    // The old policy refitted and rebuilt one slice every ten frames, from when a
    // skinned scene meant one BLAS per mesh primitive — 59 on BrainStem — and
    // rebuilding them together hitched. Merging a node's primitives into one
    // structure left 6, and at that size the arithmetic reverses: a refitted BVH
    // is bad enough that traversal pays far more than the rebuild costs.
    // BrainStem, animating, depth 8, median of 32 frames:
    //
    //   refit + periodic slice   render 91-117 ms, wall 96-121 ms
    //   rebuild every frame      render 77-79 ms,  wall 86 ms
    //
    // The refit numbers also swing by 50% depending where in the ten-frame cycle
    // the samples land; rebuilding every frame is steady.

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
        // The cap only exists so a scene with very many skeletal structures
        // cannot hitch on one frame; anything it skips is refitted and rebuilt
        // on the next.
        const bool rebuild = rebuiltThisFrame < kMaxBlasRebuildsPerFrame && mi >= mNextBlasRebuildIndex;
        if (rebuild)
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

    // Under the cap means everything eligible was covered, so start again from
    // the top rather than from one past the last skeletal structure — which may
    // be well short of blasCount when the scene also has static ones.
    if (rebuiltThisFrame < kMaxBlasRebuildsPerFrame)
    {
        mNextBlasRebuildIndex = 0;
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

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(mEmittedInstances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeUserID);
    // Measured: building the top level without Refit -- which lets the builder
    // split more freely -- made no difference to traversal on a scene with a
    // million instances (184 ms against 188). Kept refittable, because that is
    // what the animation path needs and the alternative buys nothing.
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

// The backdrop camera rays see, when the scene wants a different one from what
// lights it. No alias table: it is never sampled, only looked up along the ray
// that missed, so it needs no sampling distribution.
void MetalRender::loadEnvBackground(const std::string& texturePath)
{
    if (mEnvBackgroundTexture)
    {
        mEnvBackgroundTexture->release();
        mEnvBackgroundTexture = nullptr;
    }

    int width = 0, height = 0;
    float* pixelData = nullptr;
    bool isExr = false;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        if (LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err) != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env background: {} ({})", texturePath, err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
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
            STRELKA_ERROR("Failed to load env background: {}", texturePath);
            return;
        }
    }

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    desc->setTextureType(MTL::TextureType2D);
    desc->setStorageMode(MTL::StorageModeManaged);
    desc->setUsage(MTL::TextureUsageShaderRead);
    mEnvBackgroundTexture = mDevice->newTexture(desc);
    desc->release();
    mEnvBackgroundTexture->replaceRegion(MTL::Region::Make3D(0, 0, 0, width, height, 1), 0, pixelData,
                                         width * sizeof(float) * 4);
    if (isExr)
        free(pixelData);
    else
        stbi_image_free(pixelData);

    STRELKA_INFO("Loaded env background: {} ({}x{})", texturePath, width, height);
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

    // An HDRI's values *are* radiance. Normalising them to a fixed average
    // discards that: the same environment lights a scene differently depending
    // on what its own mean happens to be, and nothing physically lit can be
    // matched against a reference renderer. On the pine forest the factor came
    // out at ~6200, which put every surface three orders of magnitude too bright
    // and read as a lighting bug rather than a normalisation.
    //
    // Off by default, therefore. It stays available because it is genuinely
    // useful for look-dev -- drop in an arbitrary HDRI and see something
    // sensible without touching exposure -- but that is a convenience, not the
    // default a renderer should have.
    const float avgWeightedLum = (float)(totalPower / (double)(width * height));
    const bool autoCalibrate = getSettings()->getAs<bool>("render/env/autoCalibrate");
    const float kCalibrationTarget = 1000.0f;
    mEnvMapAutoScale = (autoCalibrate && avgWeightedLum > 1e-6f)
                           ? kCalibrationTarget / avgWeightedLum
                           : 1.0f;
    mEnvMapLoaded = true;

    STRELKA_INFO("Env map alias table built: {} texels ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, autoScale: {:.1f}",
                 texelCount, alias.size() * sizeof(EnvAliasEntry) / (1024.0 * 1024.0),
                 totalPower, avgWeightedLum, mEnvMapAutoScale);
}
