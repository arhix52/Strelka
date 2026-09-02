#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalRender.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include "MetalBuffer.h"
#include "MetalTextures.h"
#include "MetalEnvironment.h"
#include "MetalGeometry.h"
#include "MetalMaterials.h"
#include "MetalLights.h"
#include "MetalAccelStructure.h"
#include "MetalSkinning.h"
#include "MetalFrameUniforms.h"
#include "MetalPostProcess.h"
#include "MetalScenePreparation.h"
#include "MetalWavefrontIntegrator.h"
#include "sampling_math.h"
#include "integrator_features.h"
#include "temporal_history_policy.h"
#include <host/render_resolution.h>
#include "residency_set_diff.h"
#include <host/integrator_buffer_sizes.h>

#include <fstream>

#include <algorithm>
// The display transform the readback replays; the same header the tonemap
// kernel is compiled from, so the two cannot drift.
#include "tonemappers.h"
#include <cstdlib>
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

#include <env.h>
#include <log.h>

#include <simd/simd.h>

#include "ShaderTypes.h"
#include <strelka/material/ior_stack.h> // IorStack: sized per path in the wavefront side table

using namespace oka;
namespace fs = std::filesystem;

// What the OS charges this process, which on unified memory includes everything
// the device allocated. `phys_footprint` is the number Activity Monitor shows;
// resident size is not, and undercounts GPU allocations badly.
namespace
{
// How much a failed commit's error is worth saying out loud.
//
// A GPU reset kills every command buffer in flight, and the driver reports the
// bystanders as "Discarded (victim of GPU error/recovery)" -- an error the chunk
// that actually hung never receives. Feedback arrives in queue order, not in
// causal order, so reporting the first failure and muting the rest reliably
// names a bystander and buries the cause: the log ends up pointing at whichever
// group the feedback queue happened to drain first, which is not evidence about
// anything. Rank the errors instead, and let a more informative one replace a
// victim's report.
enum class Metal4FailureRank : int
{
    Victim = 1, // discarded by someone else's fault; names no cause
    Cause = 2 // a timeout, page fault or the like -- this is the one worth having
};

int metal4FailureRank(const NS::Error* error)
{
    if (!error)
    {
        return 0;
    }
    const NS::String* description = error->localizedDescription();
    const char* text = description ? description->utf8String() : nullptr;
    if (text != nullptr && (std::strstr(text, "victim") != nullptr || std::strstr(text, "Discarded") != nullptr))
    {
        return static_cast<int>(Metal4FailureRank::Victim);
    }
    return static_cast<int>(Metal4FailureRank::Cause);
}

struct Metal4FrameFeedbackState
{
    // Commit feedback is delivered on Metal4Context's serial feedback queue.
    double gpuMs = 0.0;
    double slowestGroupGpuMs = 0.0;
    size_t slowestGroup = 0;
    std::vector<const MTL4::CommandBuffer*> buffers;
    std::vector<metal::WavefrontChunk> chunks;
    std::vector<metal::WavefrontChunkGroup> groups;
    std::function<void(size_t)> submit;
};

size_t processFootprintBytes()
{
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
    {
        return 0;
    }
    return (size_t)info.phys_footprint;
}
} // namespace

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

        auto safeRelease = [](auto*& p) {
            if (p)
            {
                p->release();
                p = nullptr;
            }
        };

        // Domain-owned GPU resources
        mTextures.releaseAll();
        mMaterials.release();
        mLights.release();
        mGeometry.release();
        mEnvironment.release();
        mAccel.release();
        mSkinning.release();
        mFrameUniforms.release();
        mPost.release();
        mIntegrator.release();

        safeRelease(mAccumulationBuffer);
        safeRelease(mSceneTablePlaceholder);
        safeRelease(mPrevFrameVertexBuffer);
        mHasPrevFramePose = false;

        // Queue & device (release last)
        mMetal4.release();
        safeRelease(mCommandQueue);
        safeRelease(mDevice);
    }
}

void MetalRender::triggerRenderIfIdle()
{
    if (mRenderBusy.load() || deviceError())
        return;

    const uint32_t w = getSettings()->getAs<uint32_t>("render/width");
    const uint32_t h = getSettings()->getAs<uint32_t>("render/height");

    // Pick the buffer that is NOT currently being displayed
    const int ri = mReadyIndex.load();
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
    else if (mAsyncOutputBuffers[mWriteIndex]->width() != w || mAsyncOutputBuffers[mWriteIndex]->height() != h)
    {
        mAsyncOutputBuffers[mWriteIndex]->resize(w, h);
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


// Guides live at render resolution, the denoised result at display resolution.
// Usage flags come from the denoiser for the same reason they do for the spatial
// scaler: it validates them and a guess fails at encode time.

// Prepare storage for the pose the previous frame was rendered with.
bool MetalRender::capturePrevFramePose()
{
    if (!mAccel.instanceBuffer())
    {
        return false;
    }
    // Vertices only when something can actually rewrite them. With no skinning in
    // the scene the current buffer *is* the previous one, and the shader is given
    // it directly rather than a copy that could never differ.
    const bool deforming = mSkinning.skinDataBuffer() != nullptr && mGeometry.vertexBuffer() != nullptr;
    if (deforming)
    {
        if (!mPrevFrameVertexBuffer || mPrevFrameVertexBuffer->length() != mGeometry.vertexBuffer()->length())
        {
            if (mPrevFrameVertexBuffer)
            {
                mPrevFrameVertexBuffer->release();
            }
            mPrevFrameVertexBuffer =
                mDevice->newBuffer(mGeometry.vertexBuffer()->length(), MTL::ResourceStorageModePrivate);
            mHasPrevFramePose = false;
        }
    }
    return deforming;
}


// Half to float by hand: the alternative is pulling in a conversion library for
// a readback path that only debug and validation code takes.
namespace
{
float halfToFloat(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const int32_t exponent = (h >> 10) & 0x1F;
    const uint32_t mantissa = h & 0x3FF;
    uint32_t bits = 0;
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
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
} // namespace

// Rebuild the display image on the host, at a headroom of the caller's choosing.
//
// Not a copy of the finished display texture, and deliberately so. That texture
// is the frame *as presented*: already at the display's EDR headroom and already
// through the transfer encoding. A PNG needs neither -- it has nowhere to put a
// value above white, and its writer applies the encoding itself -- so handing it
// that texture clipped every highlight the headroom had lifted and encoded the
// transfer twice, which is a washed-out screenshot of an image that looked right
// on screen.
//
// The OptiX backend answers the same way for the same reason, with a CUDA kernel
// instead of this loop; both replay the transform from the slot's linear frame,
// through the shared tone curve, at whatever headroom was asked for and with no
// gamma. The loop is fine here: this is an inspection path, a few frames taken
// by hand, not something the frame loop runs.
bool MetalRender::readDisplayReferred(std::vector<float>& rgba, uint32_t& width, uint32_t& height, float maxOutput)
{
    const int ri = mReadyIndex.load();
    if (ri < 0 || ri > 1 || mAsyncOutputBuffers[ri] == nullptr)
    {
        return false;
    }
    Buffer* const buffer = mAsyncOutputBuffers[ri];
    const PresentationMetadata& presentation = mPresentation[ri];
    if (presentation.resampling == PresentationResampling::Spatial)
    {
        return readSpatialDisplayReferred(ri, rgba, width, height, maxOutput);
    }

    const float* const linear = static_cast<const float*>(buffer->getHostPointer());
    if (linear == nullptr)
    {
        return false;
    }

    width = buffer->width();
    height = buffer->height();
    if (width == 0 || height == 0)
    {
        return false;
    }

    const oka::tonemap::float3 exposure =
        oka::tonemap::make_float3(presentation.exposure[0], presentation.exposure[1], presentation.exposure[2]);
    const float headroom = std::max(maxOutput, 1.0f);
    const auto curve = static_cast<oka::tonemap::ToneMapperType>(presentation.tonemapper);
    const size_t pixels = static_cast<size_t>(width) * height;

    rgba.resize(pixels * 4);
    for (size_t i = 0; i < pixels; ++i)
    {
        oka::tonemap::float3 c = oka::tonemap::make_float3(linear[i * 4 + 0], linear[i * 4 + 1], linear[i * 4 + 2]);
        if (shouldApplyPresentationTransform(presentation))
        {
            c = c * exposure;
            switch (curve)
            {
            case oka::tonemap::ToneMapperType::eReinhard:
                c = oka::tonemap::reinhard(c, headroom);
                break;
            case oka::tonemap::ToneMapperType::eACES:
                c = oka::tonemap::ACESFitted(c, headroom);
                break;
            case oka::tonemap::ToneMapperType::eFilmic:
                c = oka::tonemap::ACESFilm(c, headroom);
                break;
            case oka::tonemap::ToneMapperType::eNone:
                break;
            }
        }
        rgba[i * 4 + 0] = c.x;
        rgba[i * 4 + 1] = c.y;
        rgba[i * 4 + 2] = c.z;
        rgba[i * 4 + 3] = linear[i * 4 + 3];
    }
    return true;
}

// Re-run only the spatial frame's presentation chain for the requested output
// headroom. The scene-linear source is already in the shared output buffer, so
// this adds no frame-path copy: screenshot time performs one low-resolution
// tone-map, one MetalFX scale, and the unavoidable GPU-to-CPU readback.
bool MetalRender::readSpatialDisplayReferred(
    int readyIndex, std::vector<float>& rgba, uint32_t& width, uint32_t& height, float maxOutput)
{
    if (readyIndex < 0 || readyIndex > 1 || !mPost.metalFx().hasSpatialScaler() || !mPost.upscaleTexture(readyIndex) ||
        !mPost.displayTexture(readyIndex) || !mAsyncOutputBuffers[readyIndex])
    {
        return false;
    }

    const PresentationMetadata& presentation = mPresentation[readyIndex];
    const uint32_t sourceWidth = presentation.sourceWidth;
    const uint32_t sourceHeight = presentation.sourceHeight;
    const uint32_t outputWidth = static_cast<uint32_t>(mPost.displayTexture(readyIndex)->width());
    const uint32_t outputHeight = static_cast<uint32_t>(mPost.displayTexture(readyIndex)->height());
    if (sourceWidth == 0 || sourceHeight == 0 || outputWidth == 0 || outputHeight == 0)
    {
        return false;
    }

    MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setWidth(outputWidth);
    descriptor->setHeight(outputHeight);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA16Float);
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setUsage((MTL::TextureUsage)(mPost.metalFx().requiredOutputUsage() | MTL::TextureUsageShaderRead));
    MTL::Texture* screenshotTexture = mDevice->newTexture(descriptor);
    descriptor->release();
    if (!screenshotTexture)
    {
        return false;
    }

    UniformsTonemap uniforms{};
    uniforms.width = sourceWidth;
    uniforms.height = sourceHeight;
    uniforms.outWidth = outputWidth;
    uniforms.outHeight = outputHeight;
    uniforms.tonemapperType = presentation.tonemapper;
    uniforms.gamma = presentation.gamma;
    uniforms.maxEDR = std::max(maxOutput, 1.0f);
    uniforms.exposureValue = { presentation.exposure[0], presentation.exposure[1], presentation.exposure[2] };

    const MTL::Buffer* const sourceBuffer =
        static_cast<const MTL::Buffer*>(mAsyncOutputBuffers[readyIndex]->getDevicePointer());
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    commandBuffer->retain();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(mPost.tonemapperPSO());
    encoder->useResource(sourceBuffer, MTL::ResourceUsageRead);
    encoder->setBytes(&uniforms, sizeof(uniforms), 0);
    encoder->setBuffer(sourceBuffer, 0, 1);
    encoder->setTexture(mPost.upscaleTexture(readyIndex), 0);
    encoder->dispatchThreads(MTL::Size(sourceWidth, sourceHeight, 1), MTL::Size(8, 8, 1));
    encoder->endEncoding();
    mPost.metalFx().encodeSpatial(
        commandBuffer, false, mPost.upscaleTexture(readyIndex), screenshotTexture, sourceWidth, sourceHeight);
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    const bool completed = commandBuffer->status() == MTL::CommandBufferStatusCompleted;
    commandBuffer->release();

    const bool read = completed && readHalfTexture(screenshotTexture, rgba, width, height);
    screenshotTexture->release();
    if (!read)
    {
        return false;
    }

    // The spatial scaler is configured for perceptual (sRGB) input/output, as
    // required by the viewport pipeline. Screenshot writers consume
    // display-linear floats and apply their own transfer encoding, so undo that
    // one reversible part here. The tone curve and MetalFX scale stay intact.
    if (presentation.gamma > 0.0f)
    {
        for (size_t i = 0; i < rgba.size(); i += 4)
        {
            rgba[i + 0] = oka::tonemap::inverseGammaFloat(rgba[i + 0], presentation.gamma);
            rgba[i + 1] = oka::tonemap::inverseGammaFloat(rgba[i + 1], presentation.gamma);
            rgba[i + 2] = oka::tonemap::inverseGammaFloat(rgba[i + 2], presentation.gamma);
        }
    }
    return true;
}

bool MetalRender::readDisplayTextureSdr(std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    return readDisplayReferred(rgba, width, height, 1.0f);
}

bool MetalRender::readDisplayTextureHdr(std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    const int ri = mReadyIndex.load();
    const float headroom = (ri >= 0 && ri <= 1) ? mPresentation[ri].maxOutput : 1.0f;

    return readDisplayReferred(rgba, width, height, headroom);
}

// Read the display texture back to the CPU.
//
// This is what the screen shows -- after tonemapping, after MetalFX -- which the
// output buffer no longer is. Every effect on the MetalFX plan is invisible
// without it, and "no validation error" has already been shown this session not
// to mean "correct image".
bool MetalRender::readHalfTexture(const MTL::Texture* tex, std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    if (!tex || tex->pixelFormat() != MTL::PixelFormatRGBA16Float)
    {
        return false;
    }
    width = (uint32_t)tex->width();
    height = (uint32_t)tex->height();

    // Half float on the GPU, so the staging buffer is 8 bytes a pixel.
    const size_t packedRowBytes = (size_t)width * 8;
    const size_t rowBytes = (packedRowBytes + 255u) & ~size_t(255u);
    MTL::Buffer* staging = mDevice->newBuffer(rowBytes * height, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(
        tex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), staging, 0, rowBytes, rowBytes * height);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    rgba.resize((size_t)width * height * 4);
    for (size_t y = 0; y < height; ++y)
    {
        const uint8_t* src = static_cast<const uint8_t*>(staging->contents()) + y * rowBytes;
        for (size_t x = 0; x < (size_t)width * 4; ++x)
        {
            uint16_t half = 0;
            std::memcpy(&half, src + x * sizeof(half), sizeof(half));
            rgba[(y * width * 4) + x] = halfToFloat(half);
        }
    }
    staging->release();
    return true;
}

bool MetalRender::readDisplayTexture(std::vector<float>& rgba, uint32_t& width, uint32_t& height)
{
    const int ri = mReadyIndex.load();
    return ri >= 0 ? readHalfTexture(mPost.displayTexture(ri), rgba, width, height) : false;
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
    if (first == SIZE_MAX || !mGeometry.vertexBuffer())
    {
        return -1.0f;
    }
    const size_t bytes = std::min(mGeometry.vertexBuffer()->length(), last * 32);
    if (bytes <= first * 32)
    {
        return -1.0f;
    }
    MTL::Buffer* staging = mDevice->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(mGeometry.vertexBuffer(), 0, staging, 0, bytes);
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
    const MTL::Texture* tex = nullptr;
    switch (guide)
    {
    case Guide::Color:
        tex = mPost.guides().color;
        break;
    case Guide::Depth:
        tex = mPost.guides().depth;
        break;
    case Guide::Motion:
        tex = mPost.guides().motion;
        break;
    case Guide::DiffuseAlbedo:
        tex = mPost.guides().diffuse;
        break;
    case Guide::SpecularAlbedo:
        tex = mPost.guides().specular;
        break;
    case Guide::Normal:
        tex = mPost.guides().normal;
        break;
    case Guide::Roughness:
        tex = mPost.guides().roughness;
        break;
    case Guide::SpecularHitDistance:
        tex = mPost.guides().specularHitDistance;
        break;
    case Guide::Reactive:
        tex = mPost.guides().reactive;
        break;
    case Guide::Denoised:
        tex = mPost.denoisedTexture();
        break;
    default:
        return false;
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
    case MTL::PixelFormatRGBA16Float:
        channels = 4;
        bytesPerChannel = 2;
        break;
    case MTL::PixelFormatRG16Float:
        channels = 2;
        bytesPerChannel = 2;
        break;
    case MTL::PixelFormatR16Float:
        channels = 1;
        bytesPerChannel = 2;
        break;
    case MTL::PixelFormatR32Float:
        channels = 1;
        bytesPerChannel = 4;
        isHalf = false;
        break;
    case MTL::PixelFormatR8Unorm:
        channels = 1;
        bytesPerChannel = 1;
        isHalf = false;
        isUnorm8 = true;
        break;
    default:
        return false;
    }

    const size_t packedRowBytes = (size_t)width * channels * bytesPerChannel;
    const size_t rowBytes = (packedRowBytes + 255u) & ~size_t(255u);
    MTL::Buffer* staging = mDevice->newBuffer(rowBytes * height, MTL::ResourceStorageModeShared);
    MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
    cmd->retain();
    MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(
        tex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), staging, 0, rowBytes, rowBytes * height);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    rgba.assign((size_t)width * height * 4, 0.0f);
    for (size_t y = 0; y < height; ++y)
    {
        const uint8_t* row = static_cast<const uint8_t*>(staging->contents()) + y * rowBytes;
        for (size_t x = 0; x < width; ++x)
        {
            const size_t t = y * width + x;
            for (uint32_t c = 0; c < channels; ++c)
            {
                const size_t si = x * channels + c;
                if (isHalf)
                {
                    uint16_t half = 0;
                    std::memcpy(&half, row + si * sizeof(half), sizeof(half));
                    rgba[t * 4 + c] = halfToFloat(half);
                }
                else if (isUnorm8)
                {
                    rgba[t * 4 + c] = static_cast<float>(row[si]) / 255.0f;
                }
                else
                {
                    float value = 0.0f;
                    std::memcpy(&value, row + si * sizeof(value), sizeof(value));
                    rgba[t * 4 + c] = value;
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
    return mPost.displayTexture(ri);
}

Buffer* MetalRender::getReadyBuffer()
{
    const int ri = mReadyIndex.load();
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

    add("Vertices", bufBytes(mGeometry.vertexBuffer()));
    add("Indices", bufBytes(mGeometry.indexBuffer()));
    // A second copy of every vertex, for motion blur and for the denoiser's
    // reprojection. Shared with the current one on a scene with nothing skinned,
    // in which case this reports zero rather than double-counting.
    add("Vertices (previous)",
        bufBytes(mGeometry.prevVertexBuffer() != mGeometry.vertexBuffer() ? mGeometry.prevVertexBuffer() : nullptr) +
            bufBytes(mPrevFrameVertexBuffer));

    {
        size_t bytes = 0;
        for (MTL::Texture* t : mTextures.materialTextures())
        {
            bytes += texBytes(t);
        }
        add("Textures", bytes);
    }
    add("Environment", texBytes(mEnvironment.state().mapTexture) + texBytes(mEnvironment.state().backgroundTexture) +
                           bufBytes(mEnvironment.state().aliasBuffer));

    {
        size_t as = 0, scratch = 0;
        for (const metal::MetalAccelStructure::Blas& b : mAccel.blasList())
        {
            as += b.mAs ? b.mAs->size() : 0;
            scratch += bufBytes(b.mScratch);
        }
        add("BLAS", as);
        // Kept for the lifetime of the structure so a refit needs no allocation.
        add("BLAS scratch", scratch);
    }
    add("TLAS", mAccel.instanceAccelerationStructure() ? mAccel.instanceAccelerationStructure()->size() : 0);
    add("TLAS scratch", bufBytes(mAccel.tlasScratchBuffer()));
    add("Instance descriptors", bufBytes(mAccel.instanceBuffer()) + bufBytes(mAccel.previousInstanceBuffer()));
    add("Geometry table", bufBytes(mGeometry.geometryEntryBuffer()));
    add("Curves", bufBytes(mGeometry.curvePointBuffer()) + bufBytes(mGeometry.curveRadiusBuffer()) +
                      bufBytes(mGeometry.curveSegmentBuffer()));
    add("Materials", bufBytes(mMaterials.buffer()));
    {
        // The slides projector lights throw belong here rather than with the
        // material maps: they are owned by the light domain and survive a
        // material reload, so counting them there would show them vanishing and
        // reappearing for reasons that have nothing to do with them.
        size_t lightBytes = bufBytes(mLights.buffer()) + bufBytes(mLights.iesBuffer());
        for (MTL::Texture* t : mLights.projectorTextures())
        {
            lightBytes += texBytes(t);
        }
        add("Lights", lightBytes);
    }
    add("Skinning", bufBytes(mSkinning.skinDataBuffer()) + bufBytes(mSkinning.jointMatricesBuffer()));

    // The per-path side tables: one entry per pixel per stage, so they scale with
    // resolution rather than with the scene, and are the reason a render at
    // native resolution costs more than the upscaled one before a ray is cast.
    add("Wavefront queues", mIntegrator.queueBytes());
    add("SHARC hash", bufBytes(mFrameUniforms.sharcHashBuffer()));
    add("SHARC accumulation", bufBytes(mFrameUniforms.sharcAccumulationBuffer()));
    add("SHARC resolved", bufBytes(mFrameUniforms.sharcResolvedBuffer()));
    add("Accumulation", bufBytes(mAccumulationBuffer));

    {
        size_t bytes = texBytes(mPost.displayTexture(0)) + texBytes(mPost.displayTexture(1)) +
                       texBytes(mPost.upscaleTexture(0)) + texBytes(mPost.upscaleTexture(1)) +
                       texBytes(mPost.denoisedTexture());
        bytes += texBytes(mPost.guides().color) + texBytes(mPost.guides().depth) + texBytes(mPost.guides().motion) +
                 texBytes(mPost.guides().diffuse) + texBytes(mPost.guides().specular) + texBytes(mPost.guides().normal) +
                 texBytes(mPost.guides().roughness) + texBytes(mPost.guides().specularHitDistance) +
                 texBytes(mPost.guides().reactive) + texBytes(mPost.guides().denoiseStrength);
        add("Display & guides", bytes);
    }

    {
        size_t bytes = 0;
        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
        {
            bytes += bufBytes(mFrameUniforms.uniformBuffer(i));
            bytes += bufBytes(mFrameUniforms.tonemapBuffer(i));
        }
        add("Uniforms & misc", bytes);
    }

    // The host arrays the editor keeps so Scene::pick() can walk them. A full
    // duplicate of the two largest GPU buffers, which is why it is worth naming.
    if (mScene && !mScene->hostGeometryReleased() && !mGeometry.vertexBufferAliasesHost())
    {
        report.cpu.push_back(
            { "Host geometry (picking)", mGeometry.hostGeometryBytes().first + mGeometry.hostGeometryBytes().second });
    }

    report.deviceAllocated = mDevice ? mDevice->currentAllocatedSize() : 0;
    report.processFootprint = processFootprintBytes();
    return true;
}

namespace
{
// CreateSystemDefaultDevice is for apps with a display. A CLI, a daemon, and a
// GitHub Actions session have none, and Apple documents MTLCopyAllDevices as
// the replacement. Prefer a device that can actually trace; a stub GPU that
// enumerates but cannot is how a headless runner used to get past init and
// SIGSEGV on the first acceleration structure.
MTL::Device* acquireMetalDevice()
{
    NS::Array* devices = MTL::CopyAllDevices();
    MTL::Device* device = nullptr;
    if (devices)
    {
        const NS::UInteger count = devices->count();
        for (NS::UInteger i = 0; i < count; ++i)
        {
            // metal-cpp exposes this framework-owned collection as NS::Object.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
            MTL::Device* candidate = static_cast<MTL::Device*>(devices->object(i));
            if (candidate && candidate->supportsRaytracing())
            {
                device = candidate;
                device->retain();
                break;
            }
        }
        devices->release();
    }
    if (!device)
    {
        device = MTL::CreateSystemDefaultDevice();
    }
    return device;
}
} // namespace

void MetalRender::init()
{
    static_assert(offsetof(::Vertex, tangent) == 12);
    static_assert(offsetof(::Vertex, normal) == 16);
    static_assert(offsetof(::Vertex, uv) == 20);
    static_assert(offsetof(::Vertex, uv1) == 24);
    static_assert(offsetof(::Vertex, color) == 28);
    static_assert(offsetof(Uniforms, openpbrParams) == 784);
    static_assert(offsetof(Uniforms, openpbrTextures) == 792);
    static_assert(offsetof(Uniforms, guideRays) == 800);
    static_assert(offsetof(Material, baseColorTexture) == 256);
    static_assert(sizeof(PathRay) == 24, "PathRay is what `extend` streams per path; keep it minimal");
    static_assert(sizeof(GuideRay) == 32, "GuideRay is a cold one-per-pixel continuation record");
    // The hot record is what every live path streams on every bounce. Medium
    // bookkeeping lives in an exact, eight-byte side record so surface-only
    // specialisations do not pay for it.
    static_assert(sizeof(PathState) == 24, "PathState is read and written for every live path on every bounce");
    static_assert(sizeof(MediumPathState) == 8, "Medium state is a cold side table, not part of PathState");
    static_assert(
        sizeof(SharcUpdateState) == 168, "SHARC update state is sparse and must stay ABI-compatible with Metal");
    static_assert(sizeof(ShadowRay) == 68, "ShadowRay host/Metal ABI changed");
    // 32 rather than 24: the hit now carries the TLAS instance, because a shared
    // BLAS belongs to no single one. One extra word per live path.
    static_assert(sizeof(HitRecord) == 32, "HitRecord size changed");
    // 16 rather than 12: the fourth word says whether the geometry is a triangle
    // mesh or a curve set, and for a curve set how many segments a strand has.
    // One entry per geometry, not per primitive or per ray, so the word is free.
    static_assert(sizeof(GeometryEntry) == 16, "GeometryEntry size changed");
    static_assert(sizeof(AovSample) == 64, "AovSample is written once per pixel per frame; keep an eye on the size");

    mDevice = acquireMetalDevice();
    if (!mDevice)
    {
        STRELKA_FATAL("Failed to create Metal device (no GPU, or none visible to this process)");
        return;
    }
    STRELKA_INFO("Metal device: {} (ray tracing {})", mDevice->name()->utf8String(),
                 mDevice->supportsRaytracing() ? "yes" : "no");
    if (!mDevice->supportsRaytracing())
    {
        STRELKA_FATAL("Metal device '{}' does not support ray tracing", mDevice->name()->utf8String());
        return;
    }
    mCommandQueue = mDevice->newCommandQueue();
    mSceneTablePlaceholder =
        mDevice->newBuffer(std::max(sizeof(Material), sizeof(GeometryEntry)), MTL::ResourceStorageModeShared);
    if (!mSceneTablePlaceholder)
    {
        STRELKA_FATAL("Failed to allocate the Metal scene-table placeholder");
        return;
    }
    std::memset(mSceneTablePlaceholder->contents(), 0, mSceneTablePlaceholder->length());
    // 64 KB of constants per frame is far more than the tracer's handful of
    // small values needs; the ring is cheap and running out is a hard error.
    mMetal4.init(mDevice, (uint32_t)kMaxFramesInFlight, static_cast<size_t>(64) * 1024);
    if (!mMetal4.isValid())
    {
        STRELKA_FATAL("Metal 4 is required by the Metal renderer");
        return;
    }
    mTextures.init(mDevice, mCommandQueue, getSettings());
    mEnvironment.init(mDevice, getSettings());
    mGeometry.init(mDevice);
    mMaterials.init(mDevice, &mTextures, getSettings());
    mLights.init(mDevice);
    mAccel.init(mDevice, mCommandQueue, &mMetal4, &mGeometry, &mMaterials, &mTextures);
    mSkinning.init(mDevice, &mMetal4, &mGeometry, (uint32_t)kMaxFramesInFlight);
    mFrameUniforms.init(mDevice);
    {
        metal::MetalPostProcess::HostHooks hooks;
        hooks.readyIndex = &mReadyIndex;
        hooks.metal4ResidencyGeneration = &mMetal4ResidencyGeneration;
        hooks.resetDenoiseHistory = &mResetDenoiseHistory;
        hooks.writeIndex = &mWriteIndex;
        mPost.init(mDevice, &mMetal4, hooks);
    }
    // MetalFX builds a large MPS graph when a denoised scaler is created. Do it
    // during renderer initialization, while startup work is already expected,
    // instead of stalling the first frame after the default-off editor control
    // is enabled. When it is off, prewarm the scale that control will select;
    // an explicitly enabled configuration keeps its exact current dimensions.
    // A real resize still recreates it, as required by the descriptor.
    const bool denoiseConfigured = getSettings()->getAs<bool>("render/pt/denoise");
    const bool prewarmDenoiser = getSettings()->getAs<bool>("render/pt/prewarmDenoiser");
    if ((denoiseConfigured || prewarmDenoiser) && !envFlag("MTL_SHADER_VALIDATION"))
    {
        const uint32_t outputWidth = getSettings()->getAs<uint32_t>("render/width");
        const uint32_t outputHeight = getSettings()->getAs<uint32_t>("render/height");
        const float requestedScale = getSettings()->getAs<float>("render/pt/upscaleFactor");
        const bool wantUpscale =
            denoiseConfigured ? getSettings()->getAs<bool>("render/pt/enableUpscale") : requestedScale < 1.0f;
        const render_resolution::Resolution resolution =
            render_resolution::resolve(outputWidth, outputHeight, wantUpscale, requestedScale);
        if (resolution.pathTraceWidth != 0u && resolution.pathTraceHeight != 0u)
        {
            float minScale = 1.0f, maxScale = 1.0f;
            MetalFxContext::denoiserScaleRange(mDevice, minScale, maxScale);
            const float scaleX = (float)outputWidth / (float)resolution.pathTraceWidth;
            const float scaleY = (float)outputHeight / (float)resolution.pathTraceHeight;
            if (scaleX >= minScale && scaleX <= maxScale && scaleY >= minScale && scaleY <= maxScale)
            {
                mPost.metalFx().ensureDenoiser(
                    mDevice, resolution.pathTraceWidth, resolution.pathTraceHeight, outputWidth, outputHeight);
            }
        }
    }
    mIntegrator.init(mDevice, &mMetal4);
    mPost.buildTonemapperPipeline();
    mIntegrator.buildPipelines();
    // Arm the deferred scene build. The first render() call picks it up a stage
    // at a time; a synchronous caller drives it to the end through renderSync.
    mScenePrep.begin();
}

void MetalRender::createMetalMaterials()
{
    const std::string resourcePath = getSettings()->getAs<std::string>("resource/searchPath");
    mMaterials.create(mScene, mLoadProgress, resourcePath);
}

bool MetalRender::stepMetalMaterials(double budgetMs)
{
    const std::string resourcePath = getSettings()->getAs<std::string>("resource/searchPath");
    return mMaterials.step(mScene, mLoadProgress, resourcePath, budgetMs);
}


// Declare every persistent allocation resident for the Metal 4 queue.
//
// Metal 3 infers residency from the bindings an encoder makes; Metal 4 does not,
// and an address in an argument table pointing at a non-resident allocation is a
// GPU fault rather than a validation message. This is the price of the argument
// table: the caller owns lifetime and residency both.
// Persistent allocations are gathered here because residency spans every
// domain and is committed once for the Metal 4 queue.
void MetalRender::makeResourcesResidentForMetal4(Buffer* output)
{
    if (!mMetal4.isValid())
    {
        return;
    }
    std::unordered_set<MTL::Allocation*> currentResidents;
    auto add = [&](MTL::Allocation* allocation) {
        if (allocation)
        {
            currentResidents.insert(allocation);
        }
    };

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
        add(mFrameUniforms.uniformBuffer(i));
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
        add(mFrameUniforms.tonemapBuffer(i));
    add(mGeometry.vertexBuffer());
    add(mGeometry.prevVertexBuffer());
    add(mPrevFrameVertexBuffer);
    add(mGeometry.indexBuffer());
    add(mAccel.instanceBuffer());
    add(mAccel.previousInstanceBuffer());
    add(mMaterials.buffer());
    add(mLights.buffer());
    add(mLights.iesBuffer());
    // The light buffer names these by handle and nothing else does. Metal 4 has
    // no useResource to fall back on, so a projector's slide that the argument
    // table reaches and the residency set does not is a page fault.
    for (MTL::Texture* t : mLights.projectorTextures())
        add(t);
    add(mGeometry.geometryEntryBuffer());
    add(mSceneTablePlaceholder);
    add(mGeometry.curvePointBuffer());
    add(mGeometry.curveRadiusBuffer());
    add(mGeometry.curveSegmentBuffer());
    add(mEnvironment.state().aliasBuffer);
    add(mAccumulationBuffer);
    mIntegrator.addResidentAllocations(add);
    add(mSkinning.skinDataBuffer());
    add(mSkinning.jointMatricesBuffer());
    add(mEnvironment.state().mapTexture);
    // Bound by the miss stage alongside mEnvironment.state().mapTexture. Metal 4 has no
    // useResource to fall back on, so a texture the argument table names and the
    // residency set does not is a page fault, and the command buffer just fails.
    add(mEnvironment.state().backgroundTexture);
    add(mFrameUniforms.sharcHashBuffer());
    add(mFrameUniforms.sharcAccumulationBuffer());
    add(mFrameUniforms.sharcResolvedBuffer());
    add(mPost.guides().color);
    add(mPost.guides().depth);
    add(mPost.guides().motion);
    add(mPost.guides().diffuse);
    add(mPost.guides().specular);
    add(mPost.guides().normal);
    add(mPost.guides().roughness);
    add(mPost.guides().specularHitDistance);
    add(mPost.guides().reactive);
    add(mPost.guides().denoiseStrength);
    add(mPost.denoisedTexture());
    for (MTL::Texture* t : mTextures.materialTextures())
        add(t);
    // Reached by address out of the uniforms, so Metal cannot infer its use from
    // any binding -- exactly the same reason the material textures above have to
    // be named here one by one.
    add(mMaterials.openpbrBuffer());
    add(mMaterials.openpbrTextureBuffer());
    for (uint32_t i = 0; i < 2; ++i)
        add(mPost.displayTexture((int)i));
    for (uint32_t i = 0; i < 2; ++i)
        add(mPost.upscaleTexture((int)i));
    for (MTL::AccelerationStructure* as : mAccel.primitiveAccelerationStructures())
        add(as);
    add(mAccel.instanceAccelerationStructure());
    add(mAccel.volumeAccelerationStructure());
    for (MTL::Buffer* buffer : mAccel.accelerationStructureAuxiliaryBuffers())
        add(buffer);
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

    const std::vector<MTL::Allocation*> retired =
        metal::retiredResidencyAllocations(mMetal4FrameResidents, currentResidents);
    for (MTL::Allocation* allocation : retired)
    {
        mMetal4.removeResident(allocation);
    }
    // Add all current pointers, not only the apparent set difference. Some
    // domains (notably acceleration structures) also manage residency directly,
    // and an allocator may reuse an address after that owner removed it.
    // Residency is a mathematical set; insertion order cannot affect command
    // execution or resource lifetime.
    // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
    for (MTL::Allocation* allocation : currentResidents)
    {
        mMetal4.addResident(allocation);
    }
    mMetal4.commitResidency();
    STRELKA_DEBUG("Metal 4 residency refreshed: {} current, {} retired, {} total allocations ({:.2f} GB)",
                  currentResidents.size(), retired.size(), mMetal4.residencyAllocationCount(),
                  static_cast<double>(mMetal4.residencyAllocatedSize()) / 1073741824.0);
    mMetal4FrameResidents = std::move(currentResidents);
}

// How many extend/shade iterations one sample needs to reach `maxDepth` bounces.
//
// More than maxDepth, whenever the scene contains something that consumes an
// iteration without advancing the path's depth: a cutout pass-through, a medium
// boundary crossing, a subsurface walk step. Each of those deliberately leaves
// `depth` alone, and each is documented as doing so to avoid exhausting the
// bounce budget -- but the budget that ends a path is this loop, not `depth`, so
// without headroom the two disagree and the deepest transport is never encoded.
// A hedge of cutout leaves goes black at a maxDepth that looks generous.
//
// The headroom is per feature and a scene without them pays nothing, which
// matters because an iteration is five stage encodes even when the queue it
// dispatches over is empty. It is a budget rather than a guarantee: the paths'
// own counters (PATH_PASSTHROUGH_MAX, MEDIUM_MAX_STEPS) still bound how many
// pass-throughs any one path may take, and those are larger than this.

metal::IntegratorSceneBindings MetalRender::integratorSceneBindings()
{
    metal::IntegratorSceneBindings b;
    b.instanceBuffer = mAccel.instanceBuffer();
    b.instanceAccelerationStructure = mAccel.instanceAccelerationStructure();
    b.volumeAccelerationStructure = mAccel.volumeAccelerationStructure();
    b.primitiveAccelerationStructures = &mAccel.primitiveAccelerationStructures();
    b.materialBuffer = mMaterials.buffer() ? mMaterials.buffer() : mSceneTablePlaceholder;
    b.lightBuffer = mLights.buffer();
    b.iesBuffer = mLights.iesBuffer();
    b.geometryEntryBuffer = mGeometry.geometryEntryBuffer() ? mGeometry.geometryEntryBuffer() : mSceneTablePlaceholder;
    b.placeholderBuffer = mSceneTablePlaceholder;
    b.vertexBuffer = mGeometry.vertexBuffer();
    b.prevVertexBuffer = mGeometry.prevVertexBuffer();
    b.indexBuffer = mGeometry.indexBuffer();
    b.prevFrameVertexBuffer = mPrevFrameVertexBuffer;
    b.prevFrameInstanceBuffer =
        mAccel.instanceTransformsChanged() ? mAccel.previousInstanceBuffer() : mAccel.instanceBuffer();
    b.curvePointBuffer = mGeometry.curvePointBuffer();
    b.curveSegmentBuffer = mGeometry.curveSegmentBuffer();
    b.sharcHashBuffer = mFrameUniforms.sharcHashBuffer();
    b.sharcAccumulationBuffer = mFrameUniforms.sharcAccumulationBuffer();
    b.sharcResolvedBuffer = mFrameUniforms.sharcResolvedBuffer();
    b.sharcStatsBuffer = mIntegrator.sharcStatsBuffer();
    b.accumulationBuffer = mAccumulationBuffer;
    b.environment = &mEnvironment;
    b.textures = &mTextures;
    b.guideColor = mPost.guides().color;
    b.guideDepth = mPost.guides().depth;
    b.guideMotion = mPost.guides().motion;
    b.guideDiffuse = mPost.guides().diffuse;
    b.guideSpecular = mPost.guides().specular;
    b.guideNormal = mPost.guides().normal;
    b.guideRoughness = mPost.guides().roughness;
    b.guideSpecularHitDistance = mPost.guides().specularHitDistance;
    b.guideReactive = mPost.guides().reactive;
    b.guideDenoiseStrength = mPost.guides().denoiseStrength;
    return b;
}

uint32_t MetalRender::wavefrontIterations(uint32_t maxDepth, uint32_t subsurfaceIterations) const
{
    uint32_t iterations = maxDepth;
    // Cutouts and medium boundaries share PATH_PASSTHROUGH_MAX, so one budget
    // covers both, and a scene with both does not need it twice: a path spends
    // from the same counter whichever kind it crosses.
    if (mMaterials.hasAlphaMaterials() || mMaterials.hasBoundedMedium())
    {
        iterations += kPassthroughIterations;
    }
    // Subsurface walk steps have a separate ceiling; cap host iterations so rare long tails do not pad every launch.
    if (mMaterials.hasSubsurfaceMaterials())
    {
        iterations += std::min(subsurfaceIterations, 256u);
    }
    return iterations;
}

// Cache resampling bounds update paths by propagation depth; the allowance covers medium events without cache vertices.
// Limit subsurface steps because sparse update paths do not justify the render pass's full walk budget.
uint32_t MetalRender::sharcUpdateIterations(uint32_t maxDepth, uint32_t subsurfaceIterations) const
{
    uint32_t depth = maxDepth;
    if (getSettings()->getAs<bool>("render/pt/sharcCacheResampling"))
    {
        const uint32_t propagationDepth = std::clamp(getSettings()->getAs<uint32_t>("render/pt/sharcPropagationDepth"),
                                                     1u, static_cast<uint32_t>(SHARC_MAX_PROPAGATION_DEPTH));
        depth = std::min(maxDepth, propagationDepth + 1u + kSharcUpdateVolumeAllowance);
    }
    return wavefrontIterations(depth, std::min(subsurfaceIterations, kSharcUpdateSubsurfaceIterations));
}

void MetalRender::render(Buffer* output)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    SharedContext& ctx = getSharedContext();

    if (mScenePrep.isBuilding())
    {
        const bool complete = stepSceneBuild(output);
        // Each slice that moved the scene forward is one more thing worth
        // showing; the clock decides how many of them are worth a frame.
        mPublishClock.noteArrivals();

        metal::StreamReadiness readiness;
        readiness.hasOutputTargets = mAccumulationBuffer != nullptr;
        readiness.hasEnvironment = mEnvironment.state().loaded;
        readiness.hasTopLevel = mAccel.instanceAccelerationStructure() != nullptr;
        readiness.buildComplete = complete;

        const double nowMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const double intervalMs = getSettings()->getAs<float>("render/stream/publishIntervalMs");
        if (!canTracePartial(readiness) || !mPublishClock.shouldPublish(nowMs, intervalMs, complete))
        {
            // Nothing to trace yet, or nothing new since the last frame. The busy
            // flag has to come off here: it is set by triggerRenderIfIdle before
            // every call, and a build stage that returns without submitting
            // anything leaves no completion handler to clear it, so the build
            // would stall one stage in.
            mRenderBusy.store(false, std::memory_order_release);
            pPool->release();
            return;
        }
        // Geometry can be published only together with the lookup tables shade
        // uses for every hit. The structure builder currently uploads the
        // geometry table with the completed TLAS; publishing an earlier partial
        // TLAS would expose real hits while geometryEntries is still null and
        // Metal Shader Validation correctly reports out-of-bounds device loads.
        // Until then the already-published empty TLAS safely shows the scene's
        // environment.
        if (mScenePrep.stage() == metal::BuildStage::Structures && mMaterials.buffer() != nullptr &&
            mGeometry.geometryEntryBuffer() != nullptr)
        {
            mAccel.publishPartialTopLevel();
        }
        mPublishClock.notePublished(nowMs);
        // Every slice creates resources the last one did not have: the vertex and
        // index buffers, the material table, each acceleration structure, and the
        // top level that replaces the empty one. Metal 4 has no useResource to
        // fall back on, so anything the residency set does not name is simply not
        // there for the tracer -- and the set is otherwise only refreshed when the
        // integrator's capacity changes, which a loading scene never does. Left
        // alone, the set keeps naming the empty top level for the whole session
        // and every ray reaches the environment: a sky, and no scene in it.
        mMetal4ResidencyGeneration = 0;
        // Time to first pixel is the number this whole path exists to move, so
        // it is reported rather than inferred from watching a window.
        if (!mReportedFirstPartialFrame)
        {
            mReportedFirstPartialFrame = true;
            STRELKA_INFO("First frame shown {:.0f} ms into the scene build (stage {})", nowMs - mBuildStartMs,
                         (uint32_t)mScenePrep.stage());
        }
        // The scene under the accumulated image just changed, so what has been
        // accumulated is of a different scene.
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
        mFrameUniforms.requestSharcReset();
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
    const render_resolution::Resolution resolution = render_resolution::resolve(
        outWidth, outHeight, wantUpscale, getSettings()->getAs<float>("render/pt/upscaleFactor"));
    const uint32_t width = resolution.pathTraceWidth;
    const uint32_t height = resolution.pathTraceHeight;
    const bool upscaling = resolution.upscaling;
    bool denoiserScaleSupported = true;
    // MetalFX publishes the range of output/input ratios it can actually do, and
    // going outside it is not refused -- the scaler is created and then produces
    // NaN. Do not silently raise the PT resolution to satisfy it: curve-heavy
    // scenes need the quarter-resolution preview to stay below the Metal command
    // queue watchdog. Spatial scaling is the safe fallback at that resolution.
    if (wantUpscale && getSettings()->getAs<bool>("render/pt/denoise"))
    {
        float minScale = 1.0f, maxScale = 2.0f;
        MetalFxContext::denoiserScaleRange(mDevice, minScale, maxScale);
        const render_resolution::DenoiserPolicy policy =
            render_resolution::resolveDenoiserPolicy(true, resolution, maxScale);
        if (policy.useSpatialFallback)
        {
            if (!mPost.loggedUpscaleClamp())
            {
                mPost.loggedUpscaleClamp() = true;
                STRELKA_WARNING(
                    "MetalFX denoiser supports {:.2f}x-{:.2f}x; render scale {:.2f} needs at least "
                    "{:.2f}, using spatial upscale instead",
                    minScale, maxScale, resolution.appliedScale, policy.lowestSupportedScale);
            }
            denoiserScaleSupported = false;
        }
    }
    mDenoiserFallbackActive.store(!denoiserScaleSupported, std::memory_order_relaxed);

    // Temporal denoising subsumes upscaling: the denoised scaler takes the
    // reduced-resolution frame and produces the display-resolution one, so the
    // spatial scaler is only for when denoising is off.
    // Denoising requires the wavefront guides.
    const bool useWavefrontTracer = mIntegrator.library() != nullptr;
    const uint32_t debug = getSettings()->getAs<uint32_t>("render/pt/debug");
    const uint32_t sharcDebug = getSettings()->getAs<uint32_t>("render/pt/sharcDebug");
    // Every Metal-only cache view: a denoiser fed a per-voxel diagnostic would
    // smooth away the thing being diagnosed. The four cross-backend ones come in
    // through `debug` below, which already turns denoising off.
    const bool sharcVisualization =
        getSettings()->getAs<bool>("render/pt/sharc") && SHARC_DEBUG_IS_SURFACE_VIEW(sharcDebug);
    // Debug views are final outputs. Sending them through MetalFX would alter
    // their values, while the debug path deliberately skips the final tonemap.
    bool denoising = getSettings()->getAs<bool>("render/pt/denoise") && denoiserScaleSupported && useWavefrontTracer &&
                     debug == 0 && !sharcVisualization;
    const bool shaderValidation = envUint("MTL_SHADER_VALIDATION", 0) != 0;
    if (denoising && shaderValidation)
    {
        if (!mPost.loggedShaderValidationDenoiserGap())
        {
            mPost.loggedShaderValidationDenoiserGap() = true;
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
        if (!mPost.loggedMetal4DenoiserGap() && mMetal4.isValid())
        {
            mPost.loggedMetal4DenoiserGap() = true;
            STRELKA_INFO("MetalFX denoiser stays on Metal 3 (supportsMetal4FX={}, FB22575333)",
                         MetalFxContext::denoiserSupportsMetal4(mDevice));
        }
        denoising = mPost.metalFx().ensureDenoiser(mDevice, width, height, outWidth, outHeight);
        if (denoising)
        {
            mPost.ensureGuideTextures(width, height, outWidth, outHeight);
        }
    }
    // Three ways to get from render resolution to output resolution, and the
    // choice matters most at one sample: the spatial scaler has no history and
    // resamples noise as-is, the temporal scaler accumulates across frames, and
    // the denoiser does that plus a guided denoise -- at the price of a second
    // traced sample for clean guides.
    const uint32_t upscaleMode = getSettings()->getAs<uint32_t>("render/pt/upscaleMode");
    const bool wantTemporal = upscaling && !denoising && denoiserScaleSupported && upscaleMode == 1u;
    bool temporalUpscaling = false;
    if (wantTemporal)
    {
        void* compiler = mMetal4.isValid() ? (void*)mMetal4.compiler() : nullptr;
        temporalUpscaling = mPost.metalFx().ensureTemporalScaler(
            mDevice, MTL::PixelFormatRGBA16Float, MTL::PixelFormatR32Float, MTL::PixelFormatRG16Float,
            MTL::PixelFormatRGBA16Float, width, height, outWidth, outHeight, compiler);
        if (temporalUpscaling)
        {
            // Depth and motion are guides the temporal scaler needs as much as
            // the denoiser does, so the same textures and the same producing
            // pass serve both.
            mPost.ensureGuideTextures(width, height, outWidth, outHeight);
        }
    }
    if (!denoising && !temporalUpscaling && mPost.guides().color)
    {
        mPost.releaseGuideTextures();
        mHasDenoisedFrame = false;
    }
    if (upscaling && !denoising && !temporalUpscaling)
    {
        void* spatialCompiler = mMetal4.isValid() ? (void*)mMetal4.compiler() : nullptr;
        mPost.metalFx().ensureSpatialScaler(mDevice, MTL::PixelFormatRGBA16Float, MTL::PixelFormatRGBA16Float, width,
                                            height, outWidth, outHeight,
                                            // The tonemapper has already applied the tone curve and gamma,
                                            // so what the scaler sees is display-referred.
                                            MetalFxContext::ColorMode::Perceptual, spatialCompiler);
        mPost.ensureUpscaleTextures(width, height);
    }
    const size_t requiredSize = static_cast<size_t>(width) * height * output->getElementSize();
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
        anyAnimationPlaying = anyAnimationPlaying || getSettings()->getAs<bool>(animationStateKey(a));
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
    const bool wantMotionBlas = mEnableMotionBlur && getSettings()->getAs<bool>("render/isMotionBlurVisible") &&
                                (anyAnimationPlaying || mShutterIntervalActive);
    if (wantMotionBlas != mAccel.motionBlasBuilt() && !mAccel.blasList().empty())
    {
        mAccel.setBuildMotionBlas(wantMotionBlas);
        rebuildAccelerationStructures();
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
        mHasPrevFramePose = false;
        // Every structure the residency set names was just freed and replaced.
        mMetal4ResidencyGeneration = 0;
    }

    // Before anything this frame can move: skinning rewrites the vertices and the
    // animation block re-uploads the instance transforms, so this is the last
    // moment at which both still describe the frame that was just displayed.
    const bool capturePrevVertices = capturePrevFramePose();

    bool motionBlurCameraSet = false; // track if animation block sets prev camera
    bool encodeSkinOpen = false;
    bool encodeSkinClose = false;
    bool copyVerticesAfterOpen = false;
    bool copyVerticesAfterClose = false;
    bool encodeSkeletalBlas = false;
    bool encodeTlas = false;

    // Animation detection: two-pass at t_open / t_close for motion blur (CHANGED-only)
    {
        const SettingsManager& animSettings = *getSettings();
        std::vector<oka::Scene::Animation>& animations = mScene->getAnimations();

        // Collect target times and detect which animations actually changed
        constexpr float EPSILON = 1e-6f;
        bool animStateChanged = false;
        float maxTimeDelta = 0.0f; // track largest time jump for scrub detection
        const size_t animCount = animations.size();
        mAnimTargetTimes.resize(animCount);
        mAnimChanged.resize(animCount);
        // Not std::ranges::fill: mAnimChanged is a vector<bool>, whose proxy
        // reference does not model indirectly_writable, so the ranges overload
        // does not apply to it.
        // NOLINTNEXTLINE(modernize-use-ranges)
        std::fill(mAnimChanged.begin(), mAnimChanged.end(), false);
        for (size_t i = 0; i < animCount; ++i)
        {
            mAnimTargetTimes[i] = animSettings.getAs<float>(animationTimeKey(i));
            const float delta = std::abs(animations[i].current - mAnimTargetTimes[i]);
            if (delta > EPSILON)
            {
                mAnimChanged[i] = true;
                animStateChanged = true;
                maxTimeDelta = std::max(maxTimeDelta, delta);
            }
        }

        // A scene that has just been built has never been posed, and no time has
        // changed to say so: the loader sets each animation's current time to its
        // start and the editor asks for that same start, so every comparison
        // above is equal and the block below would do nothing. What is left on
        // screen is the bind pose the loader uploaded -- for a character, a shape
        // nobody framed a camera on, and often not in the shot at all.
        //
        // Deferred to the first frame past the build rather than taken during it:
        // skinning indexes the per-mesh records, and those are created in the
        // structures stage, several published frames after the first.
        if (mNeedsInitialPose && !mScenePrep.isBuilding())
        {
            mNeedsInitialPose = false;
            animStateChanged = true;
        }

        // Treat time jumps larger than a fraction of the clip as cuts with no valid reprojection history.
        for (size_t i = 0; i < animCount; ++i)
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
            const float shutterDuration =
                mEnableMotionBlur ? animSettings.getAs<float>("render/motionBlur/shutterTime") : 0.0f;
            const uint32_t shutterMode = animSettings.getAs<uint32_t>("render/motionBlur/shutterMode");

            if (mEnableMotionBlur && mGeometry.prevVertexBuffer() && shutterDuration > 0.0f)
            {
                // Shutter offset: Centered straddles t_anim, Leading closes at t_anim,
                // Trailing opens at t_anim
                float shutterOffset = 0.0f;
                switch (shutterMode)
                {
                case 0:
                    shutterOffset = -shutterDuration * 0.5f;
                    break; // Centered
                case 1:
                    shutterOffset = -shutterDuration;
                    break; // Leading
                case 2:
                    shutterOffset = 0.0f;
                    break; // Trailing
                default:
                    shutterOffset = -shutterDuration * 0.5f;
                    break;
                }

                // --- Pass 1: evaluate CHANGED animations at t_open ---
                bool pass1Skeletal = false;
                for (size_t i = 0; i < animations.size(); ++i)
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
                prevCam.updateAspectRatio(static_cast<float>(width) / static_cast<float>(height));
                prevCam.updateViewMatrix();
                mPrevMotionBlurView.mCamMatrices = prevCam.matrices;
                motionBlurCameraSet = true;

                if (pass1Skeletal)
                {
                    encodeSkinOpen = mSkinning.uploadJointMatrices((uint32_t)ctx.mFrameNumber, 0);
                    encodeSkeletalBlas = encodeSkinOpen;
                }
                // Always copy current VB to prevVB — ensures prev state is consistent
                // even when only camera (not skeleton) animation changed
                copyVerticesAfterOpen = true;

                // --- Pass 2: evaluate CHANGED animations at t_close ---
                bool pass2Skeletal = false;
                for (size_t i = 0; i < animations.size(); ++i)
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
                    encodeSkinClose = mSkinning.uploadJointMatrices((uint32_t)ctx.mFrameNumber, 1);
                    encodeSkeletalBlas = encodeSkeletalBlas || encodeSkinClose;
                }
                mShutterIntervalActive = true;
                mAccel.updateInstanceTransforms();
                encodeTlas = true;

                // Restore target times so next-frame EPSILON check is stable
                for (size_t i = 0; i < animations.size(); ++i)
                    animations[i].current = mAnimTargetTimes[i];
            }
            else
            {
                // Motion blur disabled or no shutter: single-pass at target time
                bool accelStructureDirty = false;
                for (size_t i = 0; i < animations.size(); ++i)
                {
                    animations[i].current = mAnimTargetTimes[i];
                    accelStructureDirty |= mScene->applyAnimation(i);
                }

                if (accelStructureDirty)
                {
                    encodeSkinOpen = mSkinning.uploadJointMatrices((uint32_t)ctx.mFrameNumber, 0);
                    // Sync prevVB with current VB — motion BVH needs both keyframes
                    // consistent when motion blur is off (otherwise keyframe 0 is stale)
                    copyVerticesAfterClose = true;
                    encodeSkeletalBlas = encodeSkinOpen;
                    mAccel.updateInstanceTransforms();
                    encodeTlas = true;
                    // Both keyframes are the same pose again: nothing to blur, and
                    // the scene can go back to static structures.
                    mShutterIntervalActive = false;
                }
                else
                {
                    mAccel.updateInstanceTransforms();
                    encodeTlas = true;
                }
            }
            ctx.mSubframeIndex = 0;
        }
    }

    const bool enteredPause = mWasAnimationPlaying && !anyAnimationPlaying;
    mPausedBlurRefine = denoising && mEnableMotionBlur && getSettings()->getAs<bool>("render/isMotionBlurVisible") &&
                        !anyAnimationPlaying && mShutterIntervalActive;
    if (enteredPause && mPausedBlurRefine)
    {
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
    }
    mWasAnimationPlaying = anyAnimationPlaying;

    const SettingsManager& settings = *getSettings();

    const uint32_t selectedCamera = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCamera);
    camera.updateAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    camera.updateViewMatrix();

    const View currView{ camera.matrices };

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
    // A pose cannot reveal whether it arrived through continuous navigation or
    // an edit/camera switch. Inferring a cut from adjacent movement magnitudes
    // made a tiny accelerating drag look like a teleport. Smooth translation and
    // rotation always keep history; operations that replace a pose call
    // resetTemporalHistory() explicitly.
    if (metal::temporal_history::projectionChanged(currView.mCamMatrices, mPrevView.mCamMatrices))
    {
        mResetDenoiseHistory = true;
    }

    // Turning the denoiser on hands it a history from whenever it last ran.
    // Tracked on the effective state, not the setting: switching to a tracer that
    // writes no guides turns the denoiser off just as surely as the checkbox does.
    const bool denoiseEnabledNow = denoising;
    if (denoiseEnabledNow != mPost.prevDenoiseEnabled())
    {
        mResetDenoiseHistory = true;
        mPost.prevDenoiseEnabled() = denoiseEnabledNow;
    }

    // --- Cache all settings once per frame ---
    const uint32_t spp = settings.getAs<uint32_t>("render/pt/spp");
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    const uint32_t maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    const uint32_t subsurfaceIterations = settings.getAs<uint32_t>("render/pt/subsurfaceIterations");
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    const uint32_t samplerType = settings.getAs<uint32_t>("render/pt/samplerType");
    const uint32_t blueNoiseSwitchSpp = settings.getAs<uint32_t>("render/pt/blueNoiseSwitchSpp");
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    const bool isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");
    const bool enableCameraMotionBlur = settings.getAs<bool>("render/enableCameraMotionBlur");
    const metal::MetalFrameUniforms::CameraView currCamView{ currView.mCamMatrices };
    const metal::MetalFrameUniforms::CameraView prevCamView{ mPrevView.mCamMatrices };
    const metal::MetalFrameUniforms::CameraView prevMbView{ mPrevMotionBlurView.mCamMatrices };

    metal::MetalFrameUniforms::FillInput fin{};
    fin.settings = getSettings();
    fin.scene = mScene;
    fin.materials = &mMaterials;
    fin.lights = &mLights;
    fin.environment = &mEnvironment;
    fin.frameSlot = mFrameIndex;
    fin.subframeIndex = (uint32_t)ctx.mSubframeIndex;
    fin.frameNumber = ctx.mFrameNumber;
    fin.width = width;
    fin.height = height;
    fin.outWidth = outWidth;
    fin.outHeight = outHeight;
    fin.spp = spp;
    fin.sspTotal = sspTotal;
    fin.maxDepth = maxDepth;
    fin.debug = debug;
    fin.rectLightSamplingMethod = rectLightSamplingMethod;
    fin.samplerType = samplerType;
    fin.blueNoiseSwitchSpp = blueNoiseSwitchSpp;
    fin.enableAccumulation = enableAccumulation;
    fin.anyAnimationPlaying = anyAnimationPlaying;
    fin.denoising = denoising;
    fin.temporalUpscaling = temporalUpscaling;
    fin.enableMotionBlur = mEnableMotionBlur;
    fin.isMotionBlurVisible = isMotionBlurVisible;
    fin.enableCameraMotionBlur = enableCameraMotionBlur;
    fin.pausedBlurRefine = mPausedBlurRefine;
    fin.resetDenoiseHistory = mResetDenoiseHistory;
    fin.hasPrevFramePose = mHasPrevFramePose;
    fin.noPrevPose = mNoPrevPose;
    fin.camera = &camera;
    fin.currView = &currCamView;
    fin.prevView = &prevCamView;
    fin.prevMotionBlurView = &prevMbView;

    metal::MetalFrameUniforms::FillResult filled = mFrameUniforms.fill(fin);
    if (filled.settingsChanged)
    {
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
        mFrameUniforms.requestSharcReset();
        // Re-fill so subframeIndex/exposure paths see the reset.
        fin.subframeIndex = 0;
        fin.resetDenoiseHistory = true;
        filled = mFrameUniforms.fill(fin);
    }

    // Record the display transform this slot's frame is being encoded with.
    // A screenshot is taken frames later, after the user has stopped moving
    // sliders; reading the settings at that point would describe a transform the
    // pixels never went through.
    if (filled.tonemap != nullptr)
    {
        PresentationMetadata& presentation = mPresentation[mWriteIndex];
        presentation.exposure[0] = filled.tonemap->exposureValue.x;
        presentation.exposure[1] = filled.tonemap->exposureValue.y;
        presentation.exposure[2] = filled.tonemap->exposureValue.z;
        presentation.maxOutput = filled.tonemap->maxEDR;
        presentation.gamma = filled.tonemap->gamma;
        presentation.tonemapper = filled.tonemap->tonemapperType;
        // Always SceneLinear: a debug view arrives here as tonemapper None with
        // gamma 0 and unit exposure, so replaying the transform is already the
        // identity and needs no second way to say so.
        presentation.content = PresentationContent::SceneLinear;
        const bool spatialPresentation = upscaling && !denoising && !temporalUpscaling;
        presentation.sourceWidth = spatialPresentation ? width : outWidth;
        presentation.sourceHeight = spatialPresentation ? height : outHeight;
        presentation.resampling = spatialPresentation ? PresentationResampling::Spatial : PresentationResampling::None;
    }

    MTL::Buffer* pUniformBuffer = filled.uniformBuffer;
    const MTL::Buffer* pUniformTMBuffer = filled.tonemapBuffer;
    auto* pUniformData = filled.uniforms;
    const bool accumulationActive = filled.accumulationActive;
    const bool effectiveAccumulation = filled.effectiveAccumulation;
    if (encodeSkeletalBlas || encodeTlas)
    {
        // Cache keys are world-space. A refit can move an occupied surface away
        // from its key, so retaining resolved data across geometry motion would
        // turn temporal reuse into stale light transport.
        mFrameUniforms.requestSharcReset();
    }

    // Whether the denoised frame already on hand describes this scene and this
    // camera. Everything that invalidates it -- the denoiser switched on, a
    // scaler recreated at another resolution, a camera cut -- asks for a history
    // reset, and the frame that denoises is the one that clears that ask.
    const bool denoisedFrameUsable =
        denoising && mHasDenoisedFrame && !mResetDenoiseHistory && mPost.denoisedTexture() != nullptr;

    // The sample limit stops the estimator, and with it the denoiser: its output
    // is a texture, so the frame it produced at the last sample is what the post
    // path re-tonemaps (see MetalFrameUniforms::fill).
    uint32_t samplesThisLaunch = filled.samplesThisLaunch;
    // Unless there is no such frame to freeze -- the denoiser was switched on
    // after the last sample, or its history was dropped since. One more traced
    // launch hands it a fresh color frame and the guides that go with it.
    // The accumulation buffer is not written at the cap, so that sample cannot
    // disturb what has already converged.
    const bool traceUsesMetal4 = mMetal4.isValid();
    if (samplesThisLaunch == 0 && denoising && !denoisedFrameUsable)
    {
        samplesThisLaunch = std::max(spp, 1u);
    }

    // Verbatim copy of a display-resolution texture into the buffer headless
    // callers (StrelkaCLI, screenshots) read back. Both the denoiser and the
    // plain MetalFX spatial upscale produce a display-sized texture and
    // neither is otherwise readable from the CPU side -- the buffer this
    // function was handed only ever holds render-resolution data on its own.
    auto copyTextureToOutputBuffer = [&](MTL::CommandBuffer* cmd, MTL::Texture* source) {
        if (!mPost.textureToBufferPSO() || !source)
        {
            return;
        }
        MTL::ComputeCommandEncoder* cp = cmd->computeCommandEncoder();
        cp->setComputePipelineState(mPost.textureToBufferPSO());
        cp->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        cp->setBuffer(pUniformTMBuffer, 0, 0);
        cp->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
        cp->setTexture(source, 0);
        cp->dispatchThreads(MTL::Size(outWidth, outHeight, 1), MTL::Size(8, 8, 1));
        cp->endEncoding();
    };

    if (samplesThisLaunch != 0 && mAccel.instanceBuffer() != nullptr)
    {
        pUniformData->samples_per_launch = samplesThisLaunch;

        // Environment buffers 10/11 are declared unconditionally and must always be bound.
        mEnvironment.ensurePlaceholderAliasBuffer();

        {
            mIntegrator.ensureBuffers(width, height, pUniformData->sharcUpdateDownscale);
            // Output resolution, not render resolution: this is what the display
            // shows and what MetalFX upscales into.
            mPost.ensureDisplayTextures(outWidth, outHeight);

            // Metal 4 path. Residency has to be refreshed whenever the set of
            // allocations can have changed; ensureBuffers only does work
            // when the resolution does, so its capacity doubles as the generation.
            // On Apple9+ skinning, acceleration-structure maintenance and tracing
            // share the Metal 4 queue; on earlier GPUs the structures build on the
            // Metal 3 queue and skinning is retired before them (see the frame
            // encode below). The denoiser remains a Metal 3 post-process,
            // synchronized from the Metal 4 frame event after guide resolve.
            const bool useMetal4 = mMetal4.isValid() && mIntegrator.resolvePSO4() != nullptr;
            if (!useMetal4)
            {
                STRELKA_FATAL("Metal 4 wavefront pipeline is required");
                mRenderBusy.store(false, std::memory_order_release);
                pPool->release();
                return;
            }

            const bool profileStages = settings.getAs<uint32_t>("render/pt/profileStages") != 0;
            if (profileStages && !useMetal4)
            {
                mIntegrator.createStageTimestampBuffer();
            }

            const auto encodeStart = std::chrono::high_resolution_clock::now();
            metal::IntegratorFeatureInputs featureIn;
            featureIn.hasEnvMap = pUniformData->hasEnvMap;
            featureIn.hasLights = pUniformData->numLights > 0;
            featureIn.hasAlphaMaterials = mMaterials.hasAlphaMaterials();
            featureIn.enableMotionBlur = pUniformData->enableMotionBlur;
            featureIn.motionBlasBuilt = mAccel.motionBlasBuilt();
            featureIn.enableCameraMotionBlur = pUniformData->enableCameraMotionBlur;
            featureIn.useDof = pUniformData->useDof;
            featureIn.debug = pUniformData->debug != 0;
            featureIn.hasFog = pUniformData->hasFog;
            featureIn.hasSharc = pUniformData->sharcCapacity != 0;
            featureIn.hasSubsurface = mMaterials.hasSubsurfaceMaterials();
            featureIn.hasCurves = mGeometry.hasCurves();
            featureIn.hasOpenPBR = mMaterials.hasOpenPBRMaterials();
            const uint32_t features = metal::packWavefrontFeatures(featureIn).bits();

            // The OpenPBR table is reached through the uniforms rather than a
            // binding -- the shade stage has no slot left. Written here and not
            // in MetalFrameUniforms because the address belongs to the material
            // table, which that file cannot see. Null when the scene has none,
            // and the kernel that reads it is compiled out in that case anyway.
            pUniformData->openpbrParams = mMaterials.openpbrBuffer() ? mMaterials.openpbrBuffer()->gpuAddress() : 0ull;
            pUniformData->openpbrTextures =
                mMaterials.openpbrTextureBuffer() ? mMaterials.openpbrTextureBuffer()->gpuAddress() : 0ull;
            pUniformData->guideRays = mIntegrator.guideRayAddress();

            metal::IntegratorSceneBindings sceneBind = integratorSceneBindings();
            metal::IntegratorFrameRequest frameReq;
            frameReq.uniformBuffer = pUniformBuffer;
            frameReq.output = output;
            frameReq.width = width;
            frameReq.height = height;
            frameReq.sampleCount = samplesThisLaunch;
            frameReq.features = features;
            frameReq.bounceIterations = wavefrontIterations(maxDepth, subsurfaceIterations);
            frameReq.pathCount = width * height;
            frameReq.motionBlasBuilt = mAccel.motionBlasBuilt();
            frameReq.profileStages = profileStages;
            frameReq.settings = getSettings();
            const uint32_t iterationsPerChunk = metal::wavefrontChunkIterations(width, height);
            const std::vector<metal::WavefrontChunk> logicalWavefrontChunks =
                metal::makeWavefrontChunkPlan(samplesThisLaunch, frameReq.bounceIterations, iterationsPerChunk);
            uint32_t traversalBatchThreads = useMetal4 && !featureIn.hasCurves ?
                                                 metal::kWavefrontTriangleTraversalBatchThreads :
                                                 metal::kWavefrontTraversalBatchThreads;
            uint32_t traversalBatchesPerGroup = metal::kWavefrontTraversalBatchesPerCommandBuffer;
            if (useMetal4 && featureIn.hasCurves)
            {
                // Keep extend independent of the shadow batches: curve closest-
                // hit traversal needs the lower tested hardware-dispatch ceiling,
                // while shadow any-hit remained stable at the throughput size.
                traversalBatchThreads =
                    envUint("STRELKA_CURVE_BATCH_THREADS", metal::kWavefrontCurveTraversalBatchThreads);
                traversalBatchThreads =
                    std::clamp(traversalBatchThreads, metal::kWavefrontMinDiagnosticTraversalBatchThreads,
                               metal::kWavefrontTraversalBatchThreads);
                traversalBatchThreads -= traversalBatchThreads % 64u;
                traversalBatchesPerGroup = std::max(
                    1u, envUint("STRELKA_CURVE_BATCHES_PER_GROUP", metal::kWavefrontCurveTraversalBatchesPerGroup));
                static bool loggedCurveBatchPolicy = false;
                if (!loggedCurveBatchPolicy)
                {
                    STRELKA_INFO("Curve traversal batches: {} threads/dispatch, {} dispatches/group",
                                 traversalBatchThreads, traversalBatchesPerGroup);
                    loggedCurveBatchPolicy = true;
                }
            }
            frameReq.traversalBatchThreads = traversalBatchThreads;
            const uint32_t traversalBatchCount =
                metal::wavefrontTraversalBatchCount(width * height, traversalBatchThreads);
            const std::vector<metal::WavefrontChunk> wavefrontChunks = metal::makeMetal4WavefrontChunkPlan(
                logicalWavefrontChunks, traversalBatchCount, traversalBatchesPerGroup, std::min(maxDepth, 16u),
                featureIn.hasCurves);

            if (useMetal4)
            {
                // Build the variant first: compiling a pipeline can allocate,
                // and residency has to name every allocation the frame will
                // touch before the frame is committed.
                mIntegrator.resetStageProfilingMetal4();
                mIntegrator.variantFor(features | metal::WavefrontFeatures::kMetal4);
                if (featureIn.hasSharc)
                {
                    mIntegrator.variantFor(features | metal::WavefrontFeatures::kSharcUpdate |
                                           metal::WavefrontFeatures::kMetal4);
                }
                if (mMetal4ResidencyGeneration != mIntegrator.capacity() ||
                    mMetal4SharcResidencyGeneration != mFrameUniforms.sharcResourceGeneration() ||
                    mIntegrator.residencyDirty())
                {
                    makeResourcesResidentForMetal4(output);
                    mMetal4ResidencyGeneration = mIntegrator.capacity();
                    mMetal4SharcResidencyGeneration = mFrameUniforms.sharcResourceGeneration();
                    mIntegrator.clearResidencyDirty();
                }

                metal::AsFrameUpdate asUpdate;
                asUpdate.skeletal = encodeSkeletalBlas;
                asUpdate.tlas = encodeTlas;
                const bool asSideQueue = (encodeSkeletalBlas || encodeTlas) && !mAccel.inlineWithTracer();

                // Skinning writes the vertices both the acceleration-structure
                // build and the trace read; the copies preserve the shutter-open
                // keyframe and the previous-frame pose. The same sequence is
                // encoded onto whichever encoder the active path uses.
                //
                // The previous-pose snapshot is taken unconditionally. Skipping it
                // while the pose is static looks free, but mHasPrevFramePose is
                // raised elsewhere and independently, so a frame that skips the
                // copy still tells the shader the buffer is meaningful -- and
                // before the first skin that buffer has never been written at all.
                const bool anySkinWork = capturePrevVertices || encodeSkinOpen || copyVerticesAfterOpen ||
                                         encodeSkinClose || copyVerticesAfterClose;
                auto encodeSkinningAndCopies = [&](MTL4::ComputeCommandEncoder* e, oka::ConstantRing& ring) {
                    if (capturePrevVertices)
                    {
                        e->copyFromBuffer(
                            mGeometry.vertexBuffer(), 0, mPrevFrameVertexBuffer, 0, mGeometry.vertexBuffer()->length());
                        e->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                    }
                    if (encodeSkinOpen)
                    {
                        mSkinning.encode(e, ring, (uint32_t)ctx.mFrameNumber, 0);
                    }
                    if (copyVerticesAfterOpen)
                    {
                        if (encodeSkinOpen)
                        {
                            e->barrierAfterEncoderStages(
                                MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
                        }
                        mSkinning.encodeCopyVertexBufferToPrev(e);
                    }
                    if (encodeSkinClose)
                    {
                        if (copyVerticesAfterOpen)
                        {
                            e->barrierAfterEncoderStages(
                                MTL::StageBlit, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                        }
                        mSkinning.encode(e, ring, (uint32_t)ctx.mFrameNumber, 1);
                    }
                    if (copyVerticesAfterClose)
                    {
                        if (encodeSkinOpen || encodeSkinClose)
                        {
                            e->barrierAfterEncoderStages(
                                MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
                        }
                        mSkinning.encodeCopyVertexBufferToPrev(e);
                    }
                };

                MTL4::CommandBuffer* cmd4 = nullptr;
                MTL4::ComputeCommandEncoder* enc4 = nullptr;
                if (asSideQueue)
                {
                    // Software RT spans Metal 4 skinning, Metal 3 AS build and Metal 4 tracing.
                    // GPU events chain the queues without a mid-frame CPU wait.
                    if (anySkinWork)
                    {
                        MTL4::CommandBuffer* skinBuf = mMetal4.beginSkin((uint32_t)ctx.mFrameNumber);
                        MTL4::ComputeCommandEncoder* skinEnc = skinBuf->computeCommandEncoder();
                        labelMetal4(skinEnc, "skinning");
                        encodeSkinningAndCopies(skinEnc, mMetal4.skinConstants());
                        skinEnc->endEncoding();
                        asUpdate.afterSkinningValue = mMetal4.submitSkin(skinBuf);
                        asUpdate.afterSkinning = mMetal4.skinEvent();
                    }
                    mAccel.submitSide(asUpdate);
                    if (encodeTlas)
                    {
                        sceneBind.instanceAccelerationStructure = mAccel.instanceAccelerationStructure();
                        sceneBind.volumeAccelerationStructure = mAccel.volumeAccelerationStructure();
                    }
                    mMetal4.wait(mAccel.readyEvent(), mAccel.readyValue());
                    cmd4 = mMetal4.beginFrame((uint32_t)ctx.mFrameNumber);
                    enc4 = cmd4->computeCommandEncoder();
                    labelMetal4(enc4, "trace chunk 0");
                }
                else
                {
                    // Apple9+: skinning, acceleration-structure updates and the
                    // trace all share one Metal 4 command buffer.
                    cmd4 = mMetal4.beginFrame((uint32_t)ctx.mFrameNumber);
                    enc4 = cmd4->computeCommandEncoder();
                    labelMetal4(enc4, "skinning + accel + trace chunk 0");
                    encodeSkinningAndCopies(enc4, mMetal4.constants());
                    if (encodeSkeletalBlas || encodeTlas)
                    {
                        mAccel.encodeInline(enc4, asUpdate);
                    }
                    if (encodeTlas)
                    {
                        sceneBind.instanceAccelerationStructure = mAccel.instanceAccelerationStructure();
                        sceneBind.volumeAccelerationStructure = mAccel.volumeAccelerationStructure();
                    }
                }
                std::vector<const MTL4::CommandBuffer*> integrateBuffers;
                integrateBuffers.reserve(wavefrontChunks.size());

                // SHARC has a strict producer/consumer contract. Sparse update
                // paths write atomics, resolve publishes fp16 persistent data,
                // and only then may the full image pass query it.
                if (featureIn.hasSharc)
                {
                    const bool responsive = (pUniformData->sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
                    if (mFrameUniforms.sharcResetPending() || responsive)
                    {
                        mIntegrator.encodeSharcClearMetal4(enc4, sceneBind, frameReq, mFrameUniforms.sharcResetPending());
                        mFrameUniforms.markSharcResetComplete();
                    }

                    metal::IntegratorFrameRequest updateReq = frameReq;
                    updateReq.features = features | metal::WavefrontFeatures::kSharcUpdate;
                    updateReq.sampleCount = 1u;
                    updateReq.pathCount = pUniformData->sharcUpdatePathCount;
                    updateReq.bounceIterations = sharcUpdateIterations(maxDepth, subsurfaceIterations);
                    const std::vector<metal::WavefrontChunk> updateChunks =
                        metal::makeWavefrontChunkPlan(1u, updateReq.bounceIterations, iterationsPerChunk);
                    for (metal::WavefrontChunk updateChunk : updateChunks)
                    {
                        // Image resolve is not part of the update pass. SHARC's
                        // own resolve follows after every update chunk has ended.
                        updateChunk.resolve = false;
                        mIntegrator.encodeMetal4(enc4, sceneBind, updateReq, updateChunk);
                    }
                    mIntegrator.encodeSharcResolveMetal4(enc4, sceneBind, updateReq);
                }

                for (size_t chunkIndex = 0; chunkIndex < wavefrontChunks.size(); ++chunkIndex)
                {
                    if (chunkIndex > 0)
                    {
                        // Encoder barriers do not cross a Metal 4 command-buffer
                        // boundary. Publish the queue and indirect arguments the
                        // next bounce chunk consumes before closing this one.
                        // Publish this chunk's queue writes to command encoders
                        // that follow it on the Metal 4 queue. A consumer
                        // barrier (`barrierAfterQueueStages`) belongs in the
                        // *next* encoder; placed here at the end of the producer
                        // it waited for prior encoders and ordered nothing.
                        // Tail command buffers are committed as one batch, so
                        // that mistake let adjacent bounce chunks race over the
                        // ping-pong queues and indirect arguments.
                        enc4->barrierAfterStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                        enc4->endEncoding();
                        cmd4->endCommandBuffer();
                        integrateBuffers.push_back(cmd4);
                        cmd4 = mMetal4.continueFrame((uint32_t)ctx.mFrameNumber, static_cast<uint32_t>(chunkIndex - 1));
                        if (!cmd4)
                        {
                            STRELKA_FATAL("Metal 4 continuation command buffer allocation failed");
                            mRenderBusy.store(false, std::memory_order_release);
                            pPool->release();
                            return;
                        }
                        enc4 = cmd4->computeCommandEncoder();
                        const metal::WavefrontChunk& labelled = wavefrontChunks[chunkIndex];
                        labelMetal4(enc4, fmt::format("trace chunk {} sample {} bounce {}..{} {}", chunkIndex,
                                                      labelled.sampleIndex, labelled.bounceBegin, labelled.bounceEnd,
                                                      metal::wavefrontChunkPhaseName(labelled.phase)));
                    }
                    mIntegrator.encodeMetal4(enc4, sceneBind, frameReq, wavefrontChunks[chunkIndex]);
                }
                MTL4::CommandBuffer* cmdIntegrate = cmd4;
                if (!denoising && mPost.tonemapperPSO4())
                {
                    enc4->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                    enc4->setComputePipelineState(mPost.tonemapperPSO4());
                    mMetal4.argumentTable()->setAddress(pUniformTMBuffer->gpuAddress(), 0);
                    mMetal4.argumentTable()->setAddress(((MetalBuffer*)output)->getNativePtr()->gpuAddress(), 1);
                    mMetal4.argumentTable()->setTexture(mPost.tonemapTarget(upscaling)->gpuResourceID(), 0);
                    enc4->dispatchThreadgroups(MTL::Size((width + 7) / 8, (height + 7) / 8, 1), MTL::Size(8, 8, 1));
                }
                enc4->endEncoding();
                if (!denoising && upscaling)
                {
                    // MetalFX encodes into the command buffer, not an encoder of
                    // ours, so this has to follow endEncoding().
                    mPost.metalFx().encodeSpatial(cmdIntegrate, true, mPost.upscaleTexture(mWriteIndex),
                                                  mPost.displayTexture(mWriteIndex), width, height);
                    // The scaler's output is a display-sized texture that the
                    // headless writer (StrelkaCLI, screenshots) cannot read: it
                    // wants the linear `output` buffer, which still holds only
                    // whatever the render-resolution tonemap above wrote, laid out
                    // at the wrong stride for a display-resolution buffer.
                    //
                    // A second Metal4 encoder here to copy it back crashed deep in
                    // AGXMetalG13X (EXC_BAD_ACCESS in emitUscStateLoad) -- the
                    // scaler's internal encoding appears to leave the command
                    // buffer unable to open another compute encoder, the same
                    // family of MetalFX/Metal4 driver fragility already worked
                    // around above (FB22575333). Flagged for renderSync() to copy
                    // out on the CPU instead, once the frame is known complete.
                    mPendingSpatialUpscaleReadback = true;
                }
                cmdIntegrate->endCommandBuffer();
                integrateBuffers.push_back(cmdIntegrate);
                // Per-frame TLAS growth and scratch resizing can add allocations
                // while encoding; publish those changes before submission.
                mMetal4.commitResidency();

                // Completion arrives through commit options rather than a
                // handler on the command buffer, and carries the GPU interval
                // with it, so the Metal 3 timing path needs no counterpart.
                const int writeIdx4 = mWriteIndex;
                const bool asyncPresent = !denoising;
                const bool reportSharcDiagnostics = pUniformData->sharcDebug != 0u;
                const auto commitStartedAt = std::chrono::steady_clock::now();
                auto feedbackState = std::make_shared<Metal4FrameFeedbackState>();
                feedbackState->buffers = std::move(integrateBuffers);
                feedbackState->chunks = wavefrontChunks;
                feedbackState->groups = metal::makeWavefrontChunkGroups(wavefrontChunks);
                mMetal4FrameValue = mMetal4.reserveFrameSignal();
                const uint64_t frameSignalValue = mMetal4FrameValue;
                const std::weak_ptr<Metal4FrameFeedbackState> weakFeedbackState = feedbackState;
                feedbackState->submit = [this, weakFeedbackState, writeIdx4, asyncPresent, commitStartedAt, profileStages,
                                         width, height, maxDepth, samplesThisLaunch, features, traversalBatchThreads,
                                         frameSignalValue, reportSharcDiagnostics](size_t groupIndex) {
                    const std::shared_ptr<Metal4FrameFeedbackState> state = weakFeedbackState.lock();
                    if (!state)
                    {
                        return;
                    }
                    MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();
                    options->addFeedbackHandler(MTL4::CommitFeedbackHandlerFunction(
                        [this, state, writeIdx4, asyncPresent, commitStartedAt, profileStages, width, height, maxDepth,
                         samplesThisLaunch, features, traversalBatchThreads, groupIndex, frameSignalValue,
                         reportSharcDiagnostics](MTL4::CommitFeedback* fb) {
                            // The feedback is the only place a Metal 4 frame reports
                            // failure: there is no status() to poll afterwards the way
                            // the Metal 3 path polls its command buffer. Without this
                            // the sole symptom is the frame event never reaching its
                            // value, which surfaces as a timeout and names no cause.
                            const NS::Error* error = fb ? fb->error() : nullptr;
                            double chunkGpuMs = 0.0;
                            if (fb && fb->GPUEndTime() >= fb->GPUStartTime())
                            {
                                chunkGpuMs = (fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0;
                                state->gpuMs += chunkGpuMs;
                                if (chunkGpuMs > state->slowestGroupGpuMs)
                                {
                                    state->slowestGroupGpuMs = chunkGpuMs;
                                    state->slowestGroup = groupIndex;
                                }
                            }
                            if (error)
                            {
                                mMetal4FrameFailed.store(true, std::memory_order_relaxed);
                                const int rank = metal4FailureRank(error);
                                int reported = mMetal4FrameFailReportRank.load(std::memory_order_relaxed);
                                bool report = false;
                                while (rank > reported)
                                {
                                    if (mMetal4FrameFailReportRank.compare_exchange_weak(
                                            reported, rank, std::memory_order_relaxed))
                                    {
                                        report = true;
                                        break;
                                    }
                                }
                                if (report)
                                {
                                    const double wallMs = std::chrono::duration<double, std::milli>(
                                                              std::chrono::steady_clock::now() - commitStartedAt)
                                                              .count();
                                    // A killed command buffer can report only its last
                                    // short execution interval. Wall time says how long
                                    // the queue actually spent executing or waiting.
                                    STRELKA_ERROR(
                                        "Metal 4 frame chunk group {}/{} failed after {:.1f} ms wall / "
                                        "{:.1f} ms group GPU: {} (domain {}, code {}){}",
                                        groupIndex + 1, state->groups.size(), wallMs, chunkGpuMs,
                                        error->localizedDescription() ? error->localizedDescription()->utf8String() :
                                                                        "unknown error",
                                        error->domain() ? error->domain()->utf8String() : "?", (long)error->code(),
                                        rank == static_cast<int>(Metal4FailureRank::Victim) ?
                                            " -- a bystander of a GPU reset, so the chunk and workload below are not "
                                            "the cause; the causing chunk reports its own error if it reaches us" :
                                            "");
                                    STRELKA_ERROR("Metal 4 failed workload: PT={}x{} spp={} depth={} features=0x{:x}",
                                                  width, height, samplesThisLaunch, maxDepth, features);
                                    if (groupIndex < state->groups.size())
                                    {
                                        const metal::WavefrontChunkGroup& failedGroup = state->groups[groupIndex];
                                        if (failedGroup.begin < failedGroup.end && failedGroup.end <= state->chunks.size())
                                        {
                                            const metal::WavefrontChunk& first = state->chunks[failedGroup.begin];
                                            const metal::WavefrontChunk& last = state->chunks[failedGroup.end - 1];
                                            STRELKA_ERROR(
                                                "Metal 4 failed chunk: sample={}..{} bounce={}..{} phase={} "
                                                "traversal_batches={}..{}",
                                                first.sampleIndex, last.sampleIndex, first.bounceBegin, last.bounceEnd,
                                                metal::wavefrontChunkPhaseName(first.phase), first.traversalBatchBegin,
                                                last.traversalBatchEnd);
                                            if (first.phase == metal::WavefrontChunkPhase::Extend)
                                            {
                                                STRELKA_ERROR("Metal 4 failed traversal queue gids={}..{}",
                                                              first.traversalBatchBegin * traversalBatchThreads,
                                                              std::min(last.traversalBatchEnd * traversalBatchThreads,
                                                                       width * height));
                                            }
                                        }
                                    }
                                    if (profileStages)
                                    {
                                        mIntegrator.reportStageFailureMetal4();
                                    }
                                    else
                                    {
                                        STRELKA_ERROR("Metal 4 stage diagnosis disabled; reproduce with STRELKA_STAGES=1");
                                    }
                                }
                                mMetal4.signalFrame(frameSignalValue);
                                mRenderBusy.store(false, std::memory_order_release);
                                return;
                            }

                            if (groupIndex + 1 < state->groups.size())
                            {
                                // Do not recursively commit from a Metal feedback
                                // handler. The completed workload is not fully
                                // retired until the handler returns, so recursive
                                // commits turn nominally separate bounce buffers
                                // into one watchdog-scale scheduler residency.
                                mMetal4.afterFeedback([state, groupIndex]() { state->submit(groupIndex + 1); });
                                return;
                            }

                            mMetal4.signalFrame(frameSignalValue);
                            const double frameGpuMs = state->gpuMs;
                            mLastRenderTimeMs.store(frameGpuMs, std::memory_order_relaxed);
                            if (profileStages && state->slowestGroup < state->groups.size())
                            {
                                const metal::WavefrontChunkGroup& slowGroup = state->groups[state->slowestGroup];
                                if (slowGroup.begin < slowGroup.end && slowGroup.end <= state->chunks.size())
                                {
                                    const metal::WavefrontChunk& first = state->chunks[slowGroup.begin];
                                    const metal::WavefrontChunk& last = state->chunks[slowGroup.end - 1];
                                    STRELKA_INFO(
                                        "STAGES Metal4 slowest group {}/{}: sample={}..{} bounce={}..{} "
                                        "phase={} traversal_batches={}..{} GPU={:.2f} ms, "
                                        "frame chunks={:.2f} ms",
                                        state->slowestGroup + 1, state->groups.size(), first.sampleIndex,
                                        last.sampleIndex, first.bounceBegin, last.bounceEnd,
                                        metal::wavefrontChunkPhaseName(first.phase), first.traversalBatchBegin,
                                        last.traversalBatchEnd, state->slowestGroupGpuMs, frameGpuMs);
                                }
                            }
                            if (asyncPresent)
                            {
                                if (reportSharcDiagnostics)
                                {
                                    mIntegrator.reportSharcStats();
                                }
                                mReadyIndex.store(writeIdx4);
                                mRenderBusy.store(false, std::memory_order_release);
                            }
                        }));
                    const metal::WavefrontChunkGroup& group = state->groups[groupIndex];
                    mMetal4.queue()->commit(state->buffers.data() + group.begin,
                                            static_cast<NS::UInteger>(group.end - group.begin), options);
                    options->release();
                };
                feedbackState->submit(0);
                mAccel.markInstanceTransformsRendered();

                if (!denoising)
                {
                    ctx.mSubframeIndex = accumulationActive ? ctx.mSubframeIndex + samplesThisLaunch :
                                                              (effectiveAccumulation ? ctx.mSubframeIndex : 0);
                    pPool->release();
                    mPrevView = currView;
                    mHasPrevFramePose = true;
                    ctx.mFrameNumber++;
                    return;
                }
            }

            MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
            if (useMetal4)
            {
                // MetalFX denoising still requires an MTLCommandBuffer. Its
                // guide textures were produced by the Metal 4 trace above.
                pCmd->encodeWait(mMetal4.frameEvent(), mMetal4FrameValue);
            }
            {
                // Numbered, so a capture names the sample a frame came from
                // rather than "Command Buffer 0" four hundred times over.
                const std::string label = fmt::format("sample {}", (uint32_t)ctx.mSubframeIndex);
                pCmd->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
            }
            MTL::ComputeCommandEncoder* enc = nullptr;
            if (!useMetal4)
            {
                enc = pCmd->computeCommandEncoder();
                if (featureIn.hasSharc)
                {
                    const bool responsive = (pUniformData->sharcFlags & SHARC_FLAG_RESPONSIVE) != 0u;
                    if (mFrameUniforms.sharcResetPending() || responsive)
                    {
                        mIntegrator.encodeSharcClear(enc, sceneBind, frameReq, mFrameUniforms.sharcResetPending());
                        mFrameUniforms.markSharcResetComplete();
                    }

                    metal::IntegratorFrameRequest updateReq = frameReq;
                    updateReq.features = features | metal::WavefrontFeatures::kSharcUpdate;
                    updateReq.sampleCount = 1u;
                    updateReq.pathCount = pUniformData->sharcUpdatePathCount;
                    updateReq.bounceIterations = sharcUpdateIterations(maxDepth, subsurfaceIterations);
                    enc = mIntegrator.encode(pCmd, enc, sceneBind, updateReq);
                    mIntegrator.encodeSharcResolve(enc, sceneBind, updateReq);
                }
                enc = mIntegrator.encode(pCmd, enc, sceneBind, frameReq);
            }
            if (profileStages)
            {
                const double encodeMs =
                    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - encodeStart)
                        .count();
                STRELKA_INFO("STAGES cpu encode {:.3f} ms", encodeMs);
            }

            if (!useMetal4 && !denoising)
            {
                enc->setComputePipelineState(mPost.tonemapperPSO());
                enc->useResource(
                    ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
                enc->setBuffer(pUniformTMBuffer, 0, 0);
                enc->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
                enc->setTexture(mPost.tonemapTarget(upscaling), 0);
                enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }
            const bool temporalUpscale = temporalUpscaling;
            if (!useMetal4 && (denoising || temporalUpscale))
            {
                // Spread the packed guides into the textures MetalFX reads, and
                // hand it linear radiance: exposure and the tone curve come after
                // the denoise, not before it. The temporal scaler reads two of
                // the same textures -- depth and motion -- so it takes the same
                // pass rather than a second one that writes a subset.
                enc->setComputePipelineState(mIntegrator.aovResolvePSO());
                enc->setBuffer(pUniformBuffer, 0, 0);
                enc->setBuffer(mIntegrator.aovBuffer(), 0, 1);
                enc->setBuffer(mIntegrator.radianceBuffer(), 0, 2);
                enc->setBytes(&samplesThisLaunch, sizeof(uint32_t), 3);
                enc->setTexture(mPost.guides().color, 0);
                enc->setTexture(mPost.guides().depth, 1);
                enc->setTexture(mPost.guides().motion, 2);
                enc->setTexture(mPost.guides().diffuse, 3);
                enc->setTexture(mPost.guides().specular, 4);
                enc->setTexture(mPost.guides().normal, 5);
                enc->setTexture(mPost.guides().roughness, 6);
                enc->setTexture(mPost.guides().specularHitDistance, 7);
                enc->setTexture(mPost.guides().reactive, 8);
                enc->setTexture(mPost.guides().denoiseStrength, 9);
                enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
            }
            if (enc)
            {
                enc->endEncoding();
            }

            if (denoising)
            {
                MetalFxContext::DenoiseInputs in;
                in.color = mPost.guides().color;
                in.depth = mPost.guides().depth;
                in.motion = mPost.guides().motion;
                in.diffuseAlbedo = mPost.guides().diffuse;
                in.specularAlbedo = mPost.guides().specular;
                in.normal = mPost.guides().normal;
                in.roughness = mPost.guides().roughness;
                in.specularHitDistance = mPost.guides().specularHitDistance;
                in.reactive = mPost.guides().reactive;
                in.denoiseStrength = mPost.guides().denoiseStrength;
                in.output = mPost.denoisedTexture();
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
                // MetalFX accepts one exposure scalar; the renderer's current
                // white point is neutral, but luminance remains correct if that
                // becomes chromatic later.
                in.exposure = simd::dot(pUniformData->exposureValue, float3{ 0.2126f, 0.7152f, 0.0722f });
                in.depthReversed = denoiseDepthReversed(pUniformData->denoiseDepthMode, pUniformData->projectionType);
                // Reset when this frame has no valid predecessor to reproject
                // from -- *not* when the estimator restarts. Accumulation restarts
                // on every camera move, and resetting the denoiser with it throws
                // the history away exactly when it is worth most: while the camera
                // moves every frame is one sample and the history is all there is.
                // Motion vectors exist to carry smooth motion, so let them.
                in.resetHistory = mResetDenoiseHistory;
                std::memcpy(in.worldToView, glm::value_ptr(currView.mCamMatrices.view), sizeof(in.worldToView));
                std::memcpy(in.viewToClip, glm::value_ptr(currView.mCamMatrices.perspective), sizeof(in.viewToClip));
                mResetDenoiseHistory = false;
                mPost.metalFx().encodeDenoise(pCmd, in);
                mHasDenoisedFrame = true;

                // Copy denoised output back to the linear buffer consumed by headless image writers.
                copyTextureToOutputBuffer(pCmd, mPost.denoisedTexture());

                if (mPost.tonemapperTexPSO())
                {
                    MTL::ComputeCommandEncoder* tm = pCmd->computeCommandEncoder();
                    tm->setComputePipelineState(mPost.tonemapperTexPSO());
                    tm->setBuffer(pUniformTMBuffer, 0, 0);
                    tm->setTexture(mPost.displayTexture(mWriteIndex), 0);
                    tm->setTexture(mPost.denoisedTexture(), 1);
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
                tin.color = mPost.guides().color;
                tin.depth = mPost.guides().depth;
                tin.motion = mPost.guides().motion;
                tin.output = mPost.denoisedTexture();
                const uint32_t jitterSign = settings.getAs<uint32_t>("render/pt/jitterSign");
                tin.jitterX = (jitterSign & 1u) ? -pUniformData->jitterX : pUniformData->jitterX;
                tin.jitterY = (jitterSign & 2u) ? -pUniformData->jitterY : pUniformData->jitterY;
                tin.depthReversed =
                    denoiseDepthReversed(pUniformData->denoiseDepthMode, pUniformData->projectionType);
                tin.resetHistory = mResetDenoiseHistory;
                mResetDenoiseHistory = false;
                mPost.metalFx().encodeTemporal(pCmd, false, tin);

                if (mPost.tonemapperTexPSO())
                {
                    MTL::ComputeCommandEncoder* tm = pCmd->computeCommandEncoder();
                    tm->setComputePipelineState(mPost.tonemapperTexPSO());
                    tm->setBuffer(pUniformTMBuffer, 0, 0);
                    tm->setTexture(mPost.displayTexture(mWriteIndex), 0);
                    tm->setTexture(mPost.denoisedTexture(), 1);
                    tm->dispatchThreads(MTL::Size(outWidth, outHeight, 1), MTL::Size(8, 8, 1));
                    tm->endEncoding();
                }
            }
            else if (upscaling)
            {
                mPost.metalFx().encodeSpatial(
                    pCmd, false, mPost.upscaleTexture(mWriteIndex), mPost.displayTexture(mWriteIndex), width, height);
                // Same reason as the denoiser above: the scaler's output is a
                // display-sized texture, and the headless writer reads the buffer.
                copyTextureToOutputBuffer(pCmd, mPost.displayTexture(mWriteIndex));
            }

            const int writeIdxWf = mWriteIndex;
            if (!mSyncMode)
            {
                pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdxWf](MTL::CommandBuffer* cb) {
                    if (cb->status() == MTL::CommandBufferStatusCompleted &&
                        !mMetal4FrameFailed.load(std::memory_order_relaxed))
                    {
                        mReadyIndex.store(writeIdxWf);
                    }
                    else if (!mMetal4FrameFailed.load(std::memory_order_relaxed))
                    {
                        mMetal4FrameFailed.store(true, std::memory_order_relaxed);
                        const NS::Error* error = cb->error();
                        STRELKA_ERROR("Metal denoise frame failed: {}", error && error->localizedDescription() ?
                                                                            error->localizedDescription()->utf8String() :
                                                                            "unknown error");
                    }
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
        // Post-only frame. The estimator has spent its sample budget, so the
        // picture behind it is final and the only thing still worth running is
        // what turns radiance into pixels: exposure, the tone curve, gamma. Those
        // read the frame rather than produce it, so they keep responding at the
        // price of a dispatch or two instead of a re-render -- which is the point,
        // and holds with MetalFX in the pipeline as much as without it. What
        // differs there is the frame they read: the denoised texture, kept from
        // the last traced sample, or the accumulation buffer re-upscaled.
        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
        {
            // Numbered, so a capture names the frame it came from rather than
            // "Command Buffer 0" four hundred times.
            const std::string label = fmt::format("sample {}", (uint32_t)ctx.mSubframeIndex);
            pCmd->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
        }

        // Normally the trace path allocates these before the tonemapper ever runs.
        // An empty scene never takes that path -- there is no instance buffer to
        // trace against -- so the tonemapper would be handed a nil texture and the
        // display a frame that was never written.
        mPost.ensureDisplayTextures(outWidth, outHeight);
        const bool frozenDenoised = denoisedFrameUsable && mPost.tonemapperTexPSO() != nullptr;
        if (upscaling && !frozenDenoised)
        {
            mPost.ensureUpscaleTextures(width, height);
        }

        if (frozenDenoised)
        {
            // Linear denoised radiance back into the output buffer, for the same
            // readers the traced path writes it for: a screenshot taken while the
            // frame is frozen has to hold what is on the screen.
            copyTextureToOutputBuffer(pCmd, mPost.denoisedTexture());

            MTL::ComputeCommandEncoder* tm = pCmd->computeCommandEncoder();
            tm->setComputePipelineState(mPost.tonemapperTexPSO());
            tm->setBuffer(pUniformTMBuffer, 0, 0);
            tm->setTexture(mPost.displayTexture(mWriteIndex), 0);
            tm->setTexture(mPost.denoisedTexture(), 1);
            tm->dispatchThreads(MTL::Size(outWidth, outHeight, 1), MTL::Size(8, 8, 1));
            tm->endEncoding();
        }
        else
        {
            MTL::BlitCommandEncoder* pBlitEncoder = pCmd->blitCommandEncoder();
            pBlitEncoder->copyFromBuffer(mAccumulationBuffer, 0, ((MetalBuffer*)output)->getNativePtr(), 0,
                                         static_cast<NS::UInteger>(width) * height * sizeof(float4));
            pBlitEncoder->endEncoding();

            {
                MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();

                pComputeEncoder->setComputePipelineState(mPost.tonemapperPSO());
                pComputeEncoder->useResource(
                    ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
                pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
                pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
                pComputeEncoder->setTexture(mPost.tonemapTarget(upscaling), 0);
                {
                    const MTL::Size gridSize = MTL::Size(width, height, 1);
                    const MTL::Size threadgroupSize(8, 8, 1);
                    pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
                }
                pComputeEncoder->endEncoding();
            }

            // The spatial scaler takes a display-referred frame, so the tone curve
            // runs before it and re-tonemapping means re-upscaling. Left out, a
            // tonemap edit at the sample cap wrote a texture nothing sampled and
            // the display kept the last traced frame: the control looked inert.
            if (upscaling && mPost.metalFx().hasSpatialScaler())
            {
                mPost.metalFx().encodeSpatial(
                    pCmd, false, mPost.upscaleTexture(mWriteIndex), mPost.displayTexture(mWriteIndex), width, height);
                // The blit and tonemap above only fed the scaler its low-resolution
                // source through `output`; the display-resolution result the
                // headless writer actually reads still has to overwrite it.
                copyTextureToOutputBuffer(pCmd, mPost.displayTexture(mWriteIndex));
            }
        }

        // Completion handler for async double-buffered output. It must be
        // installed unconditionally in interactive mode: it is the only thing
        // that clears mRenderBusy, and skipping it would wedge the renderer.
        const int writeIdx = mWriteIndex;
        if (!mSyncMode)
        {
            pCmd->addCompletedHandler(MTL::HandlerFunction([this, writeIdx, traceUsesMetal4](MTL::CommandBuffer* cb) {
                if (cb->status() == MTL::CommandBufferStatusCompleted &&
                    !mMetal4FrameFailed.load(std::memory_order_relaxed))
                {
                    if (!traceUsesMetal4)
                    {
                        const double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
                        mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
                    }
                    mReadyIndex.store(writeIdx);
                }
                else if (!mMetal4FrameFailed.load(std::memory_order_relaxed))
                {
                    mMetal4FrameFailed.store(true, std::memory_order_relaxed);
                    const NS::Error* error = cb->error();
                    STRELKA_ERROR("Metal post-process frame failed: {}",
                                  error && error->localizedDescription() ? error->localizedDescription()->utf8String() :
                                                                           "unknown error");
                }
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
    desc->setOutputURL(NS::URL::fileURLWithPath(NS::String::string(path.c_str(), NS::UTF8StringEncoding)));
    NS::Error* err = nullptr;
    if (!mgr->startCapture(desc, &err))
    {
        STRELKA_ERROR("GPU capture failed to start: {}",
                      err && err->localizedDescription() ? err->localizedDescription()->utf8String() : "unknown");
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
        //
        // Waited for in slices rather than with one deadline: a launch carrying
        // many samples runs for seconds by design, and the Metal 3 path it
        // replaces had no deadline at all (waitUntilCompleted). A single short
        // timeout therefore reported a healthy GPU as a device error and threw
        // the render away. What a real failure looks like is the commit feedback
        // firing with an error, which is what ends the wait early; the budget is
        // only a backstop for a frame that neither completes nor reports.
        constexpr uint32_t kFrameWaitSliceMs = 1000;
        constexpr uint32_t kFrameWaitBudgetMs = 60000;
        bool completed = false;
        uint32_t waitedMs = 0;
        while (waitedMs < kFrameWaitBudgetMs)
        {
            if (mMetal4.waitForFrame(mMetal4FrameValue, kFrameWaitSliceMs))
            {
                completed = true;
                break;
            }
            if (mMetal4FrameFailed.load(std::memory_order_relaxed))
            {
                break;
            }
            waitedMs += kFrameWaitSliceMs;
        }
        if (!completed)
        {
            mDeviceError = true;
            // A frame that failed has already been described by the feedback
            // handler, in more detail than this could manage.
            if (!mDeviceErrorReported && !mMetal4FrameFailed.load(std::memory_order_relaxed))
            {
                mDeviceErrorReported = true;
                STRELKA_ERROR("Metal 4 frame did not complete in {} s and reported no error", kFrameWaitBudgetMs / 1000);
            }
        }
        if (getSettings()->getAs<uint32_t>("render/pt/profileStages") != 0)
        {
            const auto tDone = std::chrono::steady_clock::now();
            STRELKA_INFO("LAUNCH encode {:.1f} ms, wait {:.1f} ms",
                         std::chrono::duration<double, std::milli>(tSubmitted - tEncode).count(),
                         std::chrono::duration<double, std::milli>(tDone - tSubmitted).count());
            // Metal 4 commit feedback supplies the frame's GPU interval.
            STRELKA_INFO("CMDBUF gpu {:.1f} ms (metal4)", getLastRenderTimeMs());
        }
    }
    else if (mLastCommandBuffer)
    {
        mLastCommandBuffer->waitUntilCompleted();
        if (getSettings()->getAs<uint32_t>("render/pt/profileStages") != 0)
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
        // Metal 3 exposes timestamp-buffer stage timings here; Metal 4 provides breadcrumb diagnostics instead.
        if (getSettings()->getAs<uint32_t>("render/pt/profileStages") != 0)
        {
            mIntegrator.reportStageTimings();
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
                const NS::Error* err = mLastCommandBuffer->error();
                STRELKA_ERROR(
                    "Render command buffer failed: {}. The scene may not fit on the device; "
                    "try render/texture/maxDimension.",
                    err && err->localizedDescription() ? err->localizedDescription()->utf8String() : "unknown error");
            }
        }
        const double gpuMs = (mLastCommandBuffer->GPUEndTime() - mLastCommandBuffer->GPUStartTime()) * 1000.0;
        mLastRenderTimeMs.store(gpuMs, std::memory_order_relaxed);
        mLastCommandBuffer->release();
        mLastCommandBuffer = nullptr;
    }

    // The GPU has finished, so the counters it wrote are readable. Only here, in
    // the synchronous path -- the async one would be reading a buffer the next
    // frame is already clearing, and the answer is the same either way.
    mIntegrator.reportIorStackStats();
    mIntegrator.reportSharcStats();

    // See the comment where this is set: the Metal4 spatial-upscale path could
    // not copy its own result back into `output`, so it is done here, on the
    // CPU, once the frame is known complete -- the same blit-to-staging-buffer
    // pattern readHalfTexture() uses for screenshots, on the classic Metal3
    // queue rather than a second Metal4 encoder.
    if (mPendingSpatialUpscaleReadback && output)
    {
        mPendingSpatialUpscaleReadback = false;
        const MTL::Texture* tex = mPost.displayTexture(mWriteIndex);
        if (tex && tex->pixelFormat() == MTL::PixelFormatRGBA16Float)
        {
            const uint32_t w = (uint32_t)tex->width();
            const uint32_t h = (uint32_t)tex->height();
            const size_t packedRowBytes = (size_t)w * 8;
            const size_t rowBytes = (packedRowBytes + 255u) & ~size_t(255u);
            MTL::Buffer* staging = mDevice->newBuffer(rowBytes * h, MTL::ResourceStorageModeShared);
            MTL::CommandBuffer* cmd = mCommandQueue->commandBuffer();
            cmd->retain();
            MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
            blit->copyFromTexture(
                tex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(w, h, 1), staging, 0, rowBytes, rowBytes * h);
            blit->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
            cmd->release();

            auto* dst = static_cast<float*>(((MetalBuffer*)output)->getNativePtr()->contents());
            for (size_t y = 0; y < h; ++y)
            {
                const uint8_t* src = static_cast<const uint8_t*>(staging->contents()) + y * rowBytes;
                for (size_t x = 0; x < (size_t)w * 4; ++x)
                {
                    uint16_t half = 0;
                    std::memcpy(&half, src + x * sizeof(half), sizeof(half));
                    dst[(y * w * 4) + x] = halfToFloat(half);
                }
            }
            staging->release();
        }
    }

    // Managed storage needs an explicit GPU→CPU sync before the host can read.
    // On Apple silicon Managed behaves like Shared, but this keeps Intel Macs correct.
    if (output)
    {
        const MTL::Buffer* native = ((MetalBuffer*)output)->getNativePtr();
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
    if (!mDevice)
    {
        STRELKA_ERROR("createBuffer called before a Metal device exists");
        return nullptr;
    }
    const size_t size = static_cast<size_t>(desc.height) * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    MTL::Buffer* buff = mDevice->newBuffer(size, MTL::ResourceStorageModeShared);
    assert(buff);
    auto res = new MetalBuffer(buff, desc.format, desc.width, desc.height);
    assert(res);
    return res;
}

void MetalRender::uploadLightBuffer()
{
    mLights.upload(mScene->getLights(), mScene->getIesProfiles(), mScene->getProjectorImages(), mTextures);
}

void MetalRender::handleSceneChanges()
{
    SharedContext& ctx = getSharedContext();
    const ChangeBits changes = mScene->peekChanges();
    if (!any(changes))
        return;

    bool needReset = false;
    bool needSharcReset = false;
    const bool responsiveSharc = getSettings()->getAs<bool>("render/pt/sharcMetalResponsive");
    if (any(changes & ChangeBits::Lights))
    {
        uploadLightBuffer();
        needReset = true;
        needSharcReset = !responsiveSharc;
    }
    if (any(changes & ChangeBits::Transforms))
    {
        if (!mAccel.blasList().empty())
            mAccel.rebuildTLAS();
        needReset = true;
        needSharcReset = true;
    }
    if (any(changes & ChangeBits::Materials))
    {
        createMetalMaterials();
        needReset = true;
        needSharcReset = true;
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
        needSharcReset |= !responsiveSharc;
    }

    // Any branch above can have replaced an allocation the residency set names:
    // a light edit grows the light buffer or decodes a projector's slide, a
    // material edit reloads its maps, a new environment is a new texture. Metal 4
    // has no useResource to fall back on, so an address in an argument table that
    // the set does not know about is a fault rather than a validation message --
    // and the set is otherwise refreshed only when the integrator's capacity
    // changes, which an edit never touches.
    mMetal4ResidencyGeneration = 0;

    mScene->consumeChanges();
    if (needReset)
    {
        ctx.mSubframeIndex = 0;
        mResetDenoiseHistory = true;
        if (needSharcReset)
        {
            mFrameUniforms.requestSharcReset();
        }
    }
}

void MetalRender::buildBuffers()
{
    mGeometry.buildBuffers(mScene);
    uploadLightBuffer();
    mFrameUniforms.allocateRings();
}

void MetalRender::rebuildAccelerationStructures()
{
    mAccel.setScene(mScene);
    mAccel.setSettings(getSettings());
    mAccel.setLoadProgress(mLoadProgress);
    mAccel.rebuild();
}


metal::SceneBuildHooks MetalRender::makeSceneBuildHooks()
{
    metal::SceneBuildHooks hooks;
    hooks.onBuffersEnter = [this]() {
        mBuildStartMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
        mReportedFirstPartialFrame = false;
        // New scene: nothing from before relates to it.
        mResetDenoiseHistory = true;
        mFrameUniforms.requestSharcReset();
        mHasPrevFramePose = false;
        mShutterIntervalActive = false;
        mNeedsInitialPose = !mScene->getAnimations().empty();
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Geometry);
        }
    };
    hooks.buildBuffers = [this]() { buildBuffers(); };
    hooks.onEnvironmentEnter = [this]() {
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Environment);
        }
    };
    hooks.buildEnvironment = [this](Buffer* output) { buildSceneEnvironment(output); };
    hooks.publishMaterialParams = [this]() { mMaterials.publishParameters(mScene); };
    hooks.onMaterialTexturesEnter = [this]() {
        if (mLoadProgress && !mMaterials.buildActive())
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Textures, (uint32_t)mScene->getMaterials().size());
        }
    };
    hooks.stepMaterialTextures = [this](double budgetMs) { return stepMetalMaterials(budgetMs); };
    hooks.onStructuresEnter = [this]() {
        if (mLoadProgress && !mAccel.buildActive())
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Structures);
        }
    };
    hooks.onStructuresBegin = [this]() {
        // Nothing is playing yet, so start static; the per-frame check in render()
        // switches to motion structures if playback begins.
        mAccel.setScene(mScene);
        mAccel.setSettings(getSettings());
        mAccel.setLoadProgress(mLoadProgress);
        mAccel.setBuildMotionBlas(false);
    };
    hooks.stepStructures = [this](double budgetMs) { return mAccel.step(budgetMs); };
    hooks.onTailEnter = [this]() {
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Done);
        }
    };
    hooks.buildTail = [this](Buffer* output) { buildSceneTail(output); };
    return hooks;
}

// The stage that makes a scene visible before it is loaded.
//
// Nothing here depends on geometry or materials, and together these three
// things are already a complete picture: somewhere to accumulate, the sky, and
// a top level to trace against. The top level is built empty on purpose -- every
// ray then misses and reaches the environment, so the first frame is the scene's
// own lighting with none of its objects in it yet, and the objects appear in
// that rather than replacing a black screen.
void MetalRender::buildSceneEnvironment(Buffer* output)
{
    // Device only: nothing reads it back.
    mAccumulationBuffer =
        mDevice->newBuffer(static_cast<size_t>(output->width()) * output->height() * output->getElementSize(),
                           MTL::ResourceStorageModePrivate);

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

    // Before the first published frame, not in the last stage. Frames are now
    // traced from this stage onwards, and each one runs the animation block; a
    // frame that found no skinning pipeline would consume the pose it was asked
    // for and dispatch nothing, leaving the character in whatever pose the
    // vertex buffer happened to hold, with nothing left to mark dirty.
    if (!mScene->getVerticesSkinData().empty())
    {
        mSkinning.setScene(mScene);
        mSkinning.buildPipeline();
        mSkinning.createSkinDataBuffer();
        mSkinning.allocJointMatrices();
    }

    mAccel.setScene(mScene);
    mAccel.setSettings(getSettings());
    mAccel.buildEmptyTopLevel();
}

void MetalRender::buildSceneTail(Buffer* output)
{
    (void)output;

    // The environment is loaded in its own stage, long before this one.
    // Fresh scene: drop any pending edit bits from load-time createLight.
    mScene->consumeChanges();
    if (mLoadProgress)
    {
        mLoadProgress->beginStage(LoadProgress::Stage::Done);
    }
    // What the scene actually cost, once. The per-category figures scattered
    // through the build are estimates made at allocation time; this is the
    // sizes the API reports, and it is the only place the two totals -- the
    // device's and the OS's -- can be seen next to each other.
    {
        MemoryReport report;
        if (memoryReport(report))
        {
            std::ranges::sort(report.gpu, [](const MemoryReport::Entry& a, const MemoryReport::Entry& b) {
                return a.bytes > b.bytes;
            });
            std::string top;
            for (size_t i = 0; i < std::min<size_t>(4, report.gpu.size()); ++i)
            {
                top += fmt::format("{}{} {:.2f} GB", i ? ", " : "", report.gpu[i].name,
                                   static_cast<double>(report.gpu[i].bytes) / 1073741824.0);
            }
            STRELKA_INFO("Memory: device {:.2f} GB, process {:.2f} GB; largest: {}",
                         static_cast<double>(report.deviceAllocated) / 1073741824.0,
                         static_cast<double>(report.processFootprint) / 1073741824.0, top);
        }
    }
}

// One stage of the scene build per call; see MetalScenePreparation for why it
// is split at all. The stages are ordered by dependency, not by cost: materials
// index into the buffers and the acceleration structures reference both.
bool MetalRender::stepSceneBuild(Buffer* output)
{
    metal::SceneBuildHooks hooks = makeSceneBuildHooks();
    return mScenePrep.step(hooks, output);
}

void MetalRender::finishSceneBuild(Buffer* output)
{
    metal::SceneBuildHooks hooks = makeSceneBuildHooks();
    mScenePrep.finish(hooks, output);
}


// Environment / IBL loading lives in MetalEnvironment.
void MetalRender::loadEnvBackground(const std::string& texturePath)
{
    mEnvironment.loadBackground(texturePath);
}

void MetalRender::loadEnvMap(const std::string& texturePath)
{
    mEnvironment.loadMap(texturePath);
}
