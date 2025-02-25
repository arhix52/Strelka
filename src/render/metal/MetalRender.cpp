#include <cstddef>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MetalRender.h"
#include "MetalBuffer.h"

#include <cassert>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <log.h>

#include <simd/simd.h>

#include "shaders/ShaderTypes.h"

using namespace oka;
namespace fs = std::filesystem;

MetalRender::MetalRender(/* args */) = default;

MetalRender::~MetalRender() = default;

void MetalRender::init()
{
    mDevice = MTL::CreateSystemDefaultDevice();
    mCommandQueue = mDevice->newCommandQueue();
    buildComputePipeline();
    buildTonemapperPipeline();
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
    for (const Scene::MaterialDescription& currMatDesc : matDescs)
    {
        Material material = {};
        material.diffuse = { 1.0f, 1.0f, 1.0f };
        for (const auto& param : currMatDesc.params)
        {
            if (param.name == "diffuse_color" || param.name == "diffuseColor" || param.name == "diffuse_color_constant")
            {
                memcpy(&material.diffuse, param.value.data(), sizeof(float) * 3);
            }
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.size(), 0);
                memcpy(texPath.data(), param.value.data(), param.value.size());
                const fs::path fullTextureFilePath = resourcePath / texPath;
                if (param.name == "diffuse_texture")
                {
                    MTL::Texture* diffuseTex = loadTextureFromFile(fullTextureFilePath.string());
                    mMaterialTextures.push_back(diffuseTex);
                    material.diffuseTexture = diffuseTex->gpuResourceID();
                }
                if (param.name == "normalmap_texture")
                {
                    MTL::Texture* normalTex = loadTextureFromFile(fullTextureFilePath.string());
                    mMaterialTextures.push_back(normalTex);
                    material.normalTexture = normalTex->gpuResourceID();
                }
            }
        }
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

void MetalRender::render(Buffer* output)
{
    using simd::float3;
    using simd::float4;
    using simd::float4x4;
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    if (getSharedContext().mFrameNumber == 0)
    {
        buildBuffers();
        createMetalMaterials();
        createAccelerationStructures();
        // create accum buffer, we don't need cpu access, make it device only
        mAccumulationBuffer = mDevice->newBuffer(
            output->width() * output->height() * output->getElementSize(), MTL::ResourceStorageModePrivate);
    }

    mFrameIndex = (mFrameIndex + 1) % kMaxFramesInFlight;

    // Update camera state:
    const uint32_t width = output->width();
    const uint32_t height = output->height();

    oka::Camera& camera = mScene->getCamera(0);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView = {};

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mSubframeIndex = 0;
    }

    SettingsManager& settings = *getSettings();

    MTL::Buffer* pUniformBuffer = mUniformBuffers[mFrameIndex];
    MTL::Buffer* pUniformTMBuffer = mUniformTMBuffers[mFrameIndex];
    auto* pUniformData = reinterpret_cast<Uniforms*>(pUniformBuffer->contents());
    auto* pUniformTonemap = reinterpret_cast<UniformsTonemap*>(pUniformTMBuffer->contents());
    pUniformData->frameIndex = mFrameIndex;
    pUniformData->subframeIndex = getSharedContext().mSubframeIndex;
    pUniformData->height = height;
    pUniformData->width = width;
    pUniformData->numLights = mScene->getLightsDesc().size();
    pUniformData->samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    pUniformData->enableAccumulation = (uint32_t)settings.getAs<bool>("render/pt/enableAcc");
    pUniformData->missColor = float3(0.0f);
    pUniformData->maxDepth = settings.getAs<uint32_t>("render/pt/depth");
    pUniformData->debug = settings.getAs<uint32_t>("render/pt/debug");

    pUniformTonemap->width = width;
    pUniformTonemap->height = height;
    pUniformTonemap->tonemapperType = settings.getAs<uint32_t>("render/pt/tonemapperType");
    pUniformTonemap->gamma = settings.getAs<float>("render/post/gamma");

    bool settingsChanged = false;

    static uint32_t rectLightSamplingMethodPrev = 0;
    pUniformData->rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    settingsChanged = (rectLightSamplingMethodPrev != pUniformData->rectLightSamplingMethod);
    rectLightSamplingMethodPrev = pUniformData->rectLightSamplingMethod;

    static bool enableAccumulationPrev = false;
    const bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    settingsChanged |= (enableAccumulationPrev != enableAccumulation);
    enableAccumulationPrev = enableAccumulation;

    static uint32_t sspTotalPrev = 0;
    const auto sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    settingsChanged |= (sspTotalPrev > sspTotal); // reset only if new spp less than already accumulated
    sspTotalPrev = sspTotal;

    static uint32_t sppPrev = 0;
    pUniformData->samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    settingsChanged |= (sppPrev != pUniformData->samples_per_launch);
    sppPrev = pUniformData->samples_per_launch;

    if (settingsChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    glm::float4x4 invView = glm::inverse(camera.matrices.view);
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->viewToWorld.columns[column][row] = invView[column][row];
        }
    }
    for (int column = 0; column < 4; column++)
    {
        for (int row = 0; row < 4; row++)
        {
            pUniformData->clipToView.columns[column][row] = camera.matrices.invPerspective[column][row];
        }
    }
    pUniformData->subframeIndex = getSharedContext().mSubframeIndex;

    // Photometric Units from iray documentation
    // Controls the sensitivity of the “camera film” and is expressed as an index; the ISO number of the film, also
    // known as “film speed.” The higher this value, the greater the exposure. If this is set to a non-zero value,
    // “Photographic” mode is enabled. If this is set to 0, “Arbitrary” mode is enabled, and all color scaling is then
    // strictly defined by the value of cm^2 Factor.
    auto filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    // The candela per meter square factor
    auto cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    // The fractional aperture number; e.g., 11 means aperture “f/11.” It adjusts the size of the opening of the “camera
    // iris” and is expressed as a ratio. The higher this value, the lower the exposure.
    auto fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    // Controls the duration, in fractions of a second, that the “shutter” is open; e.g., the value 100 means that the
    // “shutter” is open for 1/100th of a second. The higher this value, the greater the exposure
    auto shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
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
    const int32_t leftSpp = sspTotal - getSharedContext().mSubframeIndex;
    // if accumulation is off then launch selected samples per pixel
    const uint32_t samplesThisLaunch =
        enableAccumulation ? std::min((int32_t)samplesPerLaunch, leftSpp) : samplesPerLaunch;
    if (samplesThisLaunch != 0)
    {
        pUniformData->samples_per_launch = samplesThisLaunch; // TODO: implement in pt kernel

        pUniformBuffer->didModifyRange(NS::Range::Make(0, sizeof(Uniforms)));
        pUniformTMBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsTonemap)));

        MTL::CommandBuffer* pCmd = mCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pComputeEncoder = pCmd->computeCommandEncoder();
        if (mMaterialBuffer != nullptr)
        {
            pComputeEncoder->useResource(mMaterialBuffer, MTL::ResourceUsageRead);
        }
        if (mLightBuffer != nullptr)
        {
            pComputeEncoder->useResource(mLightBuffer, MTL::ResourceUsageRead);
        }
        if (mInstanceAccelerationStructure != nullptr)
        {
            pComputeEncoder->useResource(mInstanceAccelerationStructure, MTL::ResourceUsageRead);
        }
        for (const MTL::AccelerationStructure* primitiveAccel : mPrimitiveAccelerationStructures)
        {
            pComputeEncoder->useResource(primitiveAccel, MTL::ResourceUsageRead);
        }
        for (auto& materialTexture : mMaterialTextures)
        {
            pComputeEncoder->useResource(materialTexture, MTL::ResourceUsageRead);
        }
        pComputeEncoder->useResource(((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageWrite);

        pComputeEncoder->setComputePipelineState(mPathTracingPSO);
        pComputeEncoder->setBuffer(pUniformBuffer, 0, 0);
        pComputeEncoder->setBuffer(mInstanceBuffer, 0, 1);
        pComputeEncoder->setAccelerationStructure(mInstanceAccelerationStructure, 2);
        pComputeEncoder->setBuffer(mLightBuffer, 0, 3);
        pComputeEncoder->setBuffer(mMaterialBuffer, 0, 4);
        // Output
        pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 5);
        pComputeEncoder->setBuffer(mAccumulationBuffer, 0, 6);
        if (mInstanceBuffer != nullptr)
        {
            const MTL::Size gridSize = MTL::Size(width, height, 1);
            const NS::UInteger threadGroupSize = mPathTracingPSO->maxTotalThreadsPerThreadgroup();
            const MTL::Size threadgroupSize(threadGroupSize, 1, 1);
            pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
        }
        // Disable tonemapping for debug output
        if (pUniformData->debug == 0)
        {
            pComputeEncoder->setComputePipelineState(mTonemapperPSO);
            pComputeEncoder->useResource(
                ((MetalBuffer*)output)->getNativePtr(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
            pComputeEncoder->setBuffer(pUniformTMBuffer, 0, 0);
            pComputeEncoder->setBuffer(((MetalBuffer*)output)->getNativePtr(), 0, 1);
            {
                const MTL::Size gridSize = MTL::Size(width, height, 1);
                const NS::UInteger threadGroupSize = mTonemapperPSO->maxTotalThreadsPerThreadgroup();
                const MTL::Size threadgroupSize(threadGroupSize, 1, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
        }

        pComputeEncoder->endEncoding();

        pCmd->commit();
        pCmd->waitUntilCompleted();

        if (enableAccumulation)
        {
            getSharedContext().mSubframeIndex += samplesThisLaunch;
        }
        else
        {
            getSharedContext().mSubframeIndex = 0;
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
                const NS::UInteger threadGroupSize = mTonemapperPSO->maxTotalThreadsPerThreadgroup();
                const MTL::Size threadgroupSize(threadGroupSize, 1, 1);
                pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
            }
            pComputeEncoder->endEncoding();
        }

        pCmd->commit();
        pCmd->waitUntilCompleted();
    }
    pPool->release();

    mPrevView = currView;
    getSharedContext().mFrameNumber++;
}

Buffer* MetalRender::createBuffer(const BufferDesc& desc)
{
    assert(mDevice);
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    MTL::Buffer* buff = mDevice->newBuffer(size, MTL::ResourceStorageModeManaged);
    assert(buff);
    auto* res = new MetalBuffer(buff, desc.format, desc.width, desc.height);
    assert(res);
    return res;
}


void MetalRender::buildComputePipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/pathtrace.metallib", NS::UTF8StringEncoding), &pError);
    if (!pComputeLibrary)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    MTL::Function* pPathTraceFn =
        pComputeLibrary->newFunction(NS::String::string("raytracingKernel", NS::UTF8StringEncoding));
    mPathTracingPSO = mDevice->newComputePipelineState(pPathTraceFn, &pError);
    if (!mPathTracingPSO)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pPathTraceFn->release();
    pComputeLibrary->release();
}

void MetalRender::buildTonemapperPipeline()
{
    NS::Error* pError = nullptr;
    MTL::Library* pComputeLibrary =
        mDevice->newLibrary(NS::String::string("./metal/shaders/tonemapper.metallib", NS::UTF8StringEncoding), &pError);
    if (!pComputeLibrary)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }
    MTL::Function* pTonemapperFn =
        pComputeLibrary->newFunction(NS::String::string("toneMappingComputeShader", NS::UTF8StringEncoding));
    mTonemapperPSO = mDevice->newComputePipelineState(pTonemapperFn, &pError);
    if (!mTonemapperPSO)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
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

constexpr simd_float4x4 makeIdentity()
{
    using simd::float4;
    return (simd_float4x4){ (float4){ 1.f, 0.f, 0.f, 0.f }, (float4){ 0.f, 1.f, 0.f, 0.f },
                            (float4){ 0.f, 0.f, 1.f, 0.f }, (float4){ 0.f, 0.f, 0.f, 1.f } };
}

MetalRender::Mesh* MetalRender::createMesh(const oka::Mesh& mesh)
{
    auto* result = new MetalRender::Mesh();

    const uint32_t triangleCount = mesh.mCount / 3;

    const std::vector<Scene::Vertex>& vertices = mScene->getVertices();
    const std::vector<uint32_t>& indices = mScene->getIndices();

    std::vector<Triangle> triangleData(triangleCount);
    for (int i = 0; i < triangleCount; ++i)
    {
        Triangle& curr = triangleData[i];
        const uint32_t i0 = indices[mesh.mIndex + i * 3 + 0];
        const uint32_t i1 = indices[mesh.mIndex + i * 3 + 1];
        const uint32_t i2 = indices[mesh.mIndex + i * 3 + 2];
        // Positions
        using simd::float3;

        curr.positions[0] = { vertices[mesh.mVbOffset + i0].pos.x, vertices[mesh.mVbOffset + i0].pos.y,
                              vertices[mesh.mVbOffset + i0].pos.z };
        curr.positions[1] = { vertices[mesh.mVbOffset + i1].pos.x, vertices[mesh.mVbOffset + i1].pos.y,
                              vertices[mesh.mVbOffset + i1].pos.z };
        curr.positions[2] = { vertices[mesh.mVbOffset + i2].pos.x, vertices[mesh.mVbOffset + i2].pos.y,
                              vertices[mesh.mVbOffset + i2].pos.z };
        // Normals
        curr.normals[0] = vertices[mesh.mVbOffset + i0].normal;
        curr.normals[1] = vertices[mesh.mVbOffset + i1].normal;
        curr.normals[2] = vertices[mesh.mVbOffset + i2].normal;
        // Tangents
        curr.tangent[0] = vertices[mesh.mVbOffset + i0].tangent;
        curr.tangent[1] = vertices[mesh.mVbOffset + i1].tangent;
        curr.tangent[2] = vertices[mesh.mVbOffset + i2].tangent;
        // UVs
        curr.uv[0] = vertices[mesh.mVbOffset + i0].uv;
        curr.uv[1] = vertices[mesh.mVbOffset + i1].uv;
        curr.uv[2] = vertices[mesh.mVbOffset + i2].uv;
    }

    MTL::Buffer* perPrimitiveBuffer =
        mDevice->newBuffer(triangleData.size() * sizeof(Triangle), MTL::ResourceStorageModeManaged);

    memcpy(perPrimitiveBuffer->contents(), triangleData.data(), sizeof(Triangle) * triangleData.size());

    perPrimitiveBuffer->didModifyRange(NS::Range(0, perPrimitiveBuffer->length()));

    MTL::AccelerationStructureTriangleGeometryDescriptor* geomDescriptor =
        MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();

    geomDescriptor->setVertexBuffer(mVertexBuffer);
    geomDescriptor->setVertexBufferOffset(mesh.mVbOffset * sizeof(Scene::Vertex));
    geomDescriptor->setVertexStride(sizeof(Scene::Vertex));
    geomDescriptor->setIndexBuffer(mIndexBuffer);
    geomDescriptor->setIndexBufferOffset(mesh.mIndex * sizeof(uint32_t));
    geomDescriptor->setIndexType(MTL::IndexTypeUInt32);
    geomDescriptor->setTriangleCount(triangleCount);
    // Setup per primitive data
    geomDescriptor->setPrimitiveDataBuffer(perPrimitiveBuffer);
    geomDescriptor->setPrimitiveDataBufferOffset(0);
    geomDescriptor->setPrimitiveDataElementSize(sizeof(Triangle));
    geomDescriptor->setPrimitiveDataStride(sizeof(Triangle));

    const NS::Array* geomDescriptors = NS::Array::array((const NS::Object* const*)&geomDescriptor, 1UL);

    MTL::PrimitiveAccelerationStructureDescriptor* primDescriptor =
        MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    primDescriptor->setGeometryDescriptors(geomDescriptors);

    result->mGas = createAccelerationStructure(primDescriptor);

    primDescriptor->release();
    geomDescriptor->release();
    perPrimitiveBuffer->release();
    return result;
}

void MetalRender::createAccelerationStructures()
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        return;
    }

    for (const oka::Mesh& currMesh : meshes)
    {
        MetalRender::Mesh* metalMesh = createMesh(currMesh);
        mMetalMeshes.push_back(metalMesh);
        mPrimitiveAccelerationStructures.push_back(metalMesh->mGas);
    }

    mInstanceBuffer = mDevice->newBuffer(
        sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor) * instances.size(), MTL::ResourceStorageModeManaged);
    auto* instanceDescriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)mInstanceBuffer->contents();
    for (int i = 0; i < instances.size(); ++i)
    {
        const Instance& curr = instances[i];
        instanceDescriptors[i].accelerationStructureIndex = curr.mMeshId;
        instanceDescriptors[i].options = MTL::AccelerationStructureInstanceOptionOpaque;
        instanceDescriptors[i].intersectionFunctionTableOffset = 0;
        instanceDescriptors[i].userID = curr.type == Instance::Type::eLight ? curr.mLightId : curr.mMaterialId;
        instanceDescriptors[i].mask = curr.type == Instance::Type::eLight ? GEOMETRY_MASK_LIGHT : GEOMETRY_MASK_TRIANGLE;

        for (int column = 0; column < 4; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                instanceDescriptors[i].transformationMatrix.columns[column][row] = curr.transform[column][row];
            }
        }
    }
    mInstanceBuffer->didModifyRange(NS::Range::Make(0, mInstanceBuffer->length()));

    const NS::Array* instancedAccelerationStructures = NS::Array::array(
        (const NS::Object* const*)mPrimitiveAccelerationStructures.data(), mPrimitiveAccelerationStructures.size());
    MTL::InstanceAccelerationStructureDescriptor* accelDescriptor =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    accelDescriptor->setInstancedAccelerationStructures(instancedAccelerationStructures);
    accelDescriptor->setInstanceCount(instances.size());
    accelDescriptor->setInstanceDescriptorBuffer(mInstanceBuffer);
    accelDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);

    mInstanceAccelerationStructure = createAccelerationStructure(accelDescriptor);
    pPool->release();
}
