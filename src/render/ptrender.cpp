#include "ptrender.h"

//#include "debugUtils.h"
#include "instanceconstants.h"

#include <glm/ext/matrix_relational.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

// profiler
//#include "Tracy.hpp"

namespace fs = std::filesystem;
const uint32_t MAX_LIGHT_COUNT = 100;
const uint32_t HEIGHT = 600;
const uint32_t WIDTH = 800;

using namespace oka;

void PtRender::initDefaultSettings()
{
    mSettings.enableUpscale = true;
    mSettings.enableAccumulation = true;
    mSettings.stratifiedSamplingType = 0;
    mSettings.debugView = 0;
}

void PtRender::readSettings()
{
    mSettings.enableUpscale = getSettingsManager()->getAs<bool>("render/pt/enableUpscale");
    mSettings.enableAccumulation = getSettingsManager()->getAs<bool>("render/pt/enableAcc");
    mSettings.stratifiedSamplingType = getSettingsManager()->getAs<uint32_t>("render/pt/stratifiedSamplingType");
    mSettings.debugView = getSettingsManager()->getAs<uint32_t>("render/pt/debug");
}

void PtRender::init()
{
    initDefaultSettings();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        mView[i] = createView(WIDTH, HEIGHT, getSettingsManager()->getAs<uint32_t>("render/pt/spp"));
    }

    {
        ResourceManager* resManager = getResManager();
        TextureManager* texManager = getTexManager();
        const std::string imageName = "Accumulation Image";
        mAccumulatedPt = resManager->createImage(WIDTH, HEIGHT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, imageName.c_str());
        texManager->transitionImageLayout(resManager->getVkImage(mAccumulatedPt), VK_FORMAT_R32G32B32A32_SFLOAT,
                                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        resManager->setImageLayout(mAccumulatedPt, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    mDebugView = new DebugView(getSharedContext());
    mDebugView->initialize();

    mTonemap = new Tonemap(getSharedContext());
    mTonemap->initialize();

    mUpscalePass = new UpscalePass(getSharedContext());
    mUpscalePass->initialize();

    mReductionPass = new ReductionPass(getSharedContext());
    mReductionPass->initialize();

    mMaterialManager = new MaterialManager();

    const char* envUSDPath = std::getenv("USD_DIR");
    if (!envUSDPath)
    {
        printf("Please, set USD_DIR variable\n");
        assert(0);
    }
    const std::string usdMdlLibPath = std::string(envUSDPath) + "/libraries/mdl/materialx/"; // USD 22.08

    const char* paths[4] = { "./misc/test_data/mtlx", "./misc/test_data/mdl/", "./misc/test_data/mdl/resources/",
                             usdMdlLibPath.c_str() };
    bool res = mMaterialManager->addMdlSearchPath(paths, 4);
    if (!res)
    {
        // failed to load MDL
        return;
    }

    // default material
    {
        oka::Scene::MaterialDescription defaultMaterial{};
        defaultMaterial.file = "default.mdl";
        defaultMaterial.name = "default_material";
        defaultMaterial.type = oka::Scene::MaterialDescription::Type::eMdl;
        mScene->addMaterial(defaultMaterial);
    }

    std::unordered_map<std::string, MaterialManager::Module*> mNameToModule;
    std::vector<MaterialManager::CompiledMaterial*> materials;

    std::vector<Scene::MaterialDescription> matDescs = mScene->getMaterials();
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        oka::Scene::MaterialDescription& currMatDesc = matDescs[i];
        if (currMatDesc.type == oka::Scene::MaterialDescription::Type::eMdl)
        {
            MaterialManager::Module* mdlModule = nullptr;
            if (mNameToModule.find(currMatDesc.file) != mNameToModule.end())
            {
                mdlModule = mNameToModule[currMatDesc.file];
            }
            else
            {
                mdlModule = mMaterialManager->createModule(currMatDesc.file.c_str());
                mNameToModule[currMatDesc.file] = mdlModule;
            }
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst =
                mMaterialManager->createMaterialInstance(mdlModule, currMatDesc.name.c_str());
            assert(materialInst);
            if (currMatDesc.hasColor)
            {
                // bool res = mMaterialManager->changeParam(
                //     materialInst, oka::MaterialManager::ParamType::eColor, "tint", (void*)&currMatDesc.color);
                assert(res);
            }
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager->compileMaterial(materialInst);
            assert(materialComp);
            materials.push_back(materialComp);
        }
        else
        {
            MaterialManager::Module* mdlModule = mMaterialManager->createMtlxModule(currMatDesc.code.c_str());
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = mMaterialManager->createMaterialInstance(mdlModule, "");
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager->compileMaterial(materialInst);
            assert(materialComp);
            materials.push_back(materialComp);
        }
    }

    const fs::path cwd = fs::current_path();
    std::ifstream pt(cwd.string() + "/shaders/pathtracerMdl.hlsl");
    std::stringstream ptcode;
    ptcode << pt.rdbuf();

    assert(materials.size() != 0);
    const MaterialManager::TargetCode* mdlTargetCode =
        mMaterialManager->generateTargetCode(materials.data(), (uint32_t) materials.size());
    const char* hlsl = mMaterialManager->getShaderCode(mdlTargetCode);

    mCurrentSceneRenderData = new SceneRenderData(getResManager());

    mCurrentSceneRenderData->mMaterialTargetCode = mdlTargetCode;

    std::string newPTCode = std::string(hlsl) + "\n" + ptcode.str();

    std::ofstream fout("shader_output_init.hlsl");
    fout << newPTCode.c_str();
    fout.close();

    mPathTracer = new PathTracer(getSharedContext(), newPTCode);
    mPathTracer->initialize();

    mAccumulationPathTracer = new Accumulation(getSharedContext());
    mAccumulationPathTracer->initialize();

    TextureManager::TextureSamplerDesc defSamplerDesc{ VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                                       VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT };
    getTexManager()->createTextureSampler(defSamplerDesc);
}

void PtRender::cleanup()
{
    delete mView[0];
    delete mView[1];
    delete mView[2];

    delete mTonemap;
    delete mReductionPass;
    delete mUpscalePass;
    delete mDebugView;
    delete mPathTracer;
    delete mAccumulationPathTracer;

    delete mDefaultSceneRenderData;
    if (mCurrentSceneRenderData != mDefaultSceneRenderData)
    {
        delete mCurrentSceneRenderData;
    }
}

void PtRender::reloadPt()
{
    std::unordered_map<std::string, MaterialManager::Module*> mNameToModule;
    std::unordered_map<std::string, MaterialManager::MaterialInstance*> mNameToInstance;
    std::unordered_map<std::string, MaterialManager::CompiledMaterial*> mNameToCompiled;

    std::vector<MaterialManager::CompiledMaterial*> compiledMaterials;
    std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        oka::Scene::MaterialDescription& currMatDesc = matDescs[i];
        if (currMatDesc.type == oka::Scene::MaterialDescription::Type::eMdl)
        {
            MaterialManager::Module* mdlModule = nullptr;
            if (mNameToModule.find(currMatDesc.file) != mNameToModule.end())
            {
                mdlModule = mNameToModule[currMatDesc.file];
            }
            else
            {
                mdlModule = mMaterialManager->createModule(currMatDesc.file.c_str());
                mNameToModule[currMatDesc.file] = mdlModule;
            }
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = nullptr;
            if (mNameToInstance.find(currMatDesc.name) != mNameToInstance.end())
            {
                materialInst = mNameToInstance[currMatDesc.name];
            }
            else
            {
                materialInst = mMaterialManager->createMaterialInstance(mdlModule, currMatDesc.name.c_str());
                mNameToInstance[currMatDesc.name] = materialInst;
            }
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager->compileMaterial(materialInst);
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
        }
        else
        {
            MaterialManager::Module* mdlModule = mMaterialManager->createMtlxModule(currMatDesc.code.c_str());
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = mMaterialManager->createMaterialInstance(mdlModule, "");
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager->compileMaterial(materialInst);
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
        }
    }

    const fs::path cwd = fs::current_path();
    std::ifstream pt(cwd.string() + "/shaders/pathtracerMdl.hlsl");
    std::stringstream ptcode;
    ptcode << pt.rdbuf();

    assert(compiledMaterials.size() != 0);
    MaterialManager::TargetCode* mdlTargetCode =
        mMaterialManager->generateTargetCode(compiledMaterials.data(), (uint32_t) compiledMaterials.size());

    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        for (const auto& param : matDescs[i].params)
        {
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.size(), 0);
                memcpy(texPath.data(), param.value.data(), param.value.size());
                int texId = getTexManager()->loadTextureMdl(texPath);
                int resId = mMaterialManager->registerResource(mdlTargetCode, texId);
                assert(resId > 0);
                MaterialManager::Param newParam;
                newParam.name = param.name;
                newParam.type = MaterialManager::Param::Type::eInt;
                newParam.value.resize(sizeof(resId));
                memcpy(newParam.value.data(), &resId, sizeof(resId));
                mMaterialManager->setParam(mdlTargetCode, compiledMaterials[i], newParam);
            }
            else
            {
                mMaterialManager->setParam(mdlTargetCode, compiledMaterials[i], param);
            }
        }
        mMaterialManager->dumpParams(mdlTargetCode, compiledMaterials[i]);
    }

    const char* hlsl = mMaterialManager->getShaderCode(mdlTargetCode);

    mCurrentSceneRenderData->mMaterialTargetCode = mdlTargetCode;

    std::string newPTCode = std::string(hlsl) + "\n" + ptcode.str();

    std::ofstream fout("shader_output.hlsl");
    fout << newPTCode.c_str();
    fout.close();

    mPathTracer->updateShader(newPTCode.c_str());
}

PtRender::ViewData* PtRender::createView(uint32_t width, uint32_t height, uint32_t spp)
{
    assert(mSharedCtx->mResManager);
    assert(mSharedCtx->mTextureManager);
    ResourceManager* resManager = getResManager();
    TextureManager* texManager = getTexManager();
    SettingsManager* settingsManager = getSettingsManager();
    ViewData* view = new ViewData();
    view->spp = spp;
    view->finalWidth = width;
    view->finalHeight = height;
    const float upscaleFactor = settingsManager->getAs<float>("render/pt/upscaleFactor");
    view->renderWidth = (uint32_t)(width * upscaleFactor);
    view->renderHeight = (uint32_t)(height * upscaleFactor);
    view->mResManager = resManager;
    view->gbuffer = createGbuffer(view->renderWidth, view->renderHeight);
    view->prevDepth = resManager->createImage(
        view->renderWidth, view->renderHeight, view->gbuffer->depthFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Prev depth");
    texManager->transitionImageLayout(resManager->getVkImage(view->prevDepth), view->gbuffer->depthFormat,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resManager->setImageLayout(view->prevDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    view->textureDebugViewImage = resManager->createImage(
        width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DebugView result");
    texManager->transitionImageLayout(resManager->getVkImage(view->textureDebugViewImage),
                                      VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    resManager->setImageLayout(view->textureDebugViewImage, VK_IMAGE_LAYOUT_GENERAL);

    view->textureTonemapImage = resManager->createImage(
        view->renderWidth, view->renderHeight, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Tonemap result");
    view->textureUpscaleImage = resManager->createImage(
        width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Upscale Output");
    texManager->transitionImageLayout(resManager->getVkImage(view->textureUpscaleImage), VK_FORMAT_R32G32B32A32_SFLOAT,
                                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    resManager->setImageLayout(view->textureUpscaleImage, VK_IMAGE_LAYOUT_GENERAL);

    const size_t compositingBufferSize = view->renderWidth * view->renderHeight * 3 * sizeof(float); // 3 - rgb
    const size_t sampleBufferSize = compositingBufferSize * spp;
    view->mSampleBuffer = resManager->createBuffer(
        sampleBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Sample buffer");
    view->mCompositingBuffer = resManager->createBuffer(compositingBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Compositing buffer");

    view->mPathTracerImage = resManager->createImage(
        view->renderWidth, view->renderHeight, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Path Tracer Output");
    // for (int i = 0; i < 2; ++i)
    {
        const std::string imageName = "PT Accumulation Image";
        view->mAccumulationPathTracerImage = resManager->createImage(
            view->renderWidth, view->renderHeight, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, imageName.c_str());
        texManager->transitionImageLayout(resManager->getVkImage(view->mAccumulationPathTracerImage),
                                          VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        resManager->setImageLayout(view->mAccumulationPathTracerImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    return view;
}

GBuffer* PtRender::createGbuffer(uint32_t width, uint32_t height)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    assert(resManager);
    GBuffer* res = new GBuffer();
    res->mResManager = resManager;
    res->width = width;
    res->height = height;
    // Depth
    res->depthFormat = getSharedContext().depthFormat;
    res->depth = resManager->createImage(
        width, height, res->depthFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "depth");
    // Normals
    res->normal = resManager->createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "normal");
    // Tangent
    res->tangent = resManager->createImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "tangent");
    // wPos
    res->wPos = resManager->createImage(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "wPos");
    // UV
    res->uv = resManager->createImage(width, height, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "UV");
    // InstId
    res->instId = resManager->createImage(width, height, VK_FORMAT_R32_SINT, VK_IMAGE_TILING_OPTIMAL,
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "instId");
    // Motion
    res->motion = resManager->createImage(width, height, VK_FORMAT_R32G32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Motion");
    // Debug
    res->debug = resManager->createImage(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Debug");
    return res;
}

void PtRender::createVertexBuffer(oka::Scene& scene)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    std::vector<oka::Scene::Vertex>& sceneVertices = scene.getVertices();
    VkDeviceSize bufferSize = sizeof(oka::Scene::Vertex) * sceneVertices.size();
    if (bufferSize == 0)
    {
        return;
    }
    Buffer* stagingBuffer =
        resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);
    memcpy(stagingBufferMemory, sceneVertices.data(), (size_t)bufferSize);
    mCurrentSceneRenderData->mVertexBuffer = resManager->createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "VB");
    resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer),
                           resManager->getVkBuffer(mCurrentSceneRenderData->mVertexBuffer), bufferSize);
    resManager->destroyBuffer(stagingBuffer);
}

void PtRender::createLightsBuffer(oka::Scene& scene)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    std::vector<oka::Scene::Light>& sceneLights = scene.getLights();

    VkDeviceSize bufferSize = sizeof(oka::Scene::Light) * MAX_LIGHT_COUNT;
    mCurrentSceneRenderData->mLightsBuffer =
        resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Lights");

    if (!sceneLights.empty())
    {
        Buffer* stagingBuffer =
            resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);
        memcpy(stagingBufferMemory, sceneLights.data(), sceneLights.size() * sizeof(oka::Scene::Light));
        resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer),
                               resManager->getVkBuffer(mCurrentSceneRenderData->mLightsBuffer), bufferSize);
        resManager->destroyBuffer(stagingBuffer);
    }
}

void PtRender::createBvhBuffer(oka::Scene& scene)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    const std::vector<Scene::Vertex>& vertices = scene.getVertices();
    const std::vector<uint32_t>& indices = scene.getIndices();
    const std::vector<Instance>& instances = scene.getInstances();
    const std::vector<Mesh>& meshes = scene.getMeshes();

    std::vector<BVHInputPosition> positions;
    positions.reserve(indices.size());
    uint32_t currInstId = 0;
    for (const Instance& currInstance : instances)
    {
        const uint32_t currentMeshId = currInstance.mMeshId;
        const Mesh& mesh = meshes[currentMeshId];
        const uint32_t indexOffset = mesh.mIndex;
        const uint32_t indexCount = mesh.mCount;
        assert(indexCount % 3 == 0);
        const uint32_t triangleCount = indexCount / 3;
        const glm::float4x4 m = currInstance.transform;

        for (uint32_t i = 0; i < triangleCount; ++i)
        {
            uint32_t i0 = indices[indexOffset + i * 3 + 0];
            uint32_t i1 = indices[indexOffset + i * 3 + 1];
            uint32_t i2 = indices[indexOffset + i * 3 + 2];

            glm::float3 v0 = m * glm::float4(vertices[i0].pos, 1.0);
            glm::float3 v1 = m * glm::float4(vertices[i1].pos, 1.0);
            glm::float3 v2 = m * glm::float4(vertices[i2].pos, 1.0);

            BVHInputPosition p0;
            p0.pos = v0;
            p0.instId = currInstId;
            p0.primId = i;
            positions.push_back(p0);

            BVHInputPosition p1;
            p1.pos = v1;
            p1.instId = currInstId;
            p1.primId = i;
            positions.push_back(p1);

            BVHInputPosition p2;
            p2.pos = v2;
            p2.instId = currInstId;
            p2.primId = i;
            positions.push_back(p2);
        }
        ++currInstId;
    }

    BVH sceneBvh = mBvhBuilder.build(positions);
    {
        VkDeviceSize bufferSize = sizeof(BVHNode) * sceneBvh.nodes.size();
        if (bufferSize == 0)
        {
            return;
        }
        Buffer* stagingBuffer =
            resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);
        memcpy(stagingBufferMemory, sceneBvh.nodes.data(), (size_t)bufferSize);
        mCurrentSceneRenderData->mBvhNodeBuffer =
            resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "BVH");
        resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer),
                               resManager->getVkBuffer(mCurrentSceneRenderData->mBvhNodeBuffer), bufferSize);
        resManager->destroyBuffer(stagingBuffer);
    }
}

void PtRender::createIndexBuffer(oka::Scene& scene)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    std::vector<uint32_t>& sceneIndices = scene.getIndices();
    mCurrentSceneRenderData->mIndicesCount = (uint32_t)sceneIndices.size();
    VkDeviceSize bufferSize = sizeof(uint32_t) * sceneIndices.size();
    if (bufferSize == 0)
    {
        return;
    }

    Buffer* stagingBuffer =
        resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);
    memcpy(stagingBufferMemory, sceneIndices.data(), (size_t)bufferSize);
    mCurrentSceneRenderData->mIndexBuffer = resManager->createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "IB");
    resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer),
                           resManager->getVkBuffer(mCurrentSceneRenderData->mIndexBuffer), bufferSize);
    resManager->destroyBuffer(stagingBuffer);
}

void PtRender::createInstanceBuffer(oka::Scene& scene)
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    const std::vector<Mesh>& meshes = scene.getMeshes();
    const std::vector<oka::Instance>& sceneInstances = scene.getInstances();
    mCurrentSceneRenderData->mInstanceCount = (uint32_t)sceneInstances.size();
    VkDeviceSize bufferSize = sizeof(InstanceConstants) * (sceneInstances.size() + MAX_LIGHT_COUNT); // Reserve some
                                                                                                     // extra space for
                                                                                                     // lights
    if (bufferSize == 0)
    {
        return;
    }
    std::vector<InstanceConstants> instanceConsts;
    instanceConsts.resize(sceneInstances.size());
    for (int i = 0; i < sceneInstances.size(); ++i)
    {
        instanceConsts[i].materialId = sceneInstances[i].mMaterialId;
        instanceConsts[i].objectToWorld = sceneInstances[i].transform;
        instanceConsts[i].worldToObject = glm::inverse(sceneInstances[i].transform);
        instanceConsts[i].normalMatrix = glm::inverse(glm::transpose(sceneInstances[i].transform));

        const uint32_t currentMeshId = sceneInstances[i].mMeshId;
        instanceConsts[i].indexOffset = meshes[currentMeshId].mIndex;
        instanceConsts[i].indexCount = meshes[currentMeshId].mCount;
        instanceConsts[i].lightId = sceneInstances[i].lightId;
    }

    Buffer* stagingBuffer =
        resManager->createBuffer(sizeof(InstanceConstants) * sceneInstances.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);
    memcpy(stagingBufferMemory, instanceConsts.data(), sizeof(InstanceConstants) * sceneInstances.size());
    mCurrentSceneRenderData->mInstanceBuffer =
        resManager->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "Instance consts");
    resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer),
                           resManager->getVkBuffer(mCurrentSceneRenderData->mInstanceBuffer),
                           sizeof(InstanceConstants) * sceneInstances.size());
    resManager->destroyBuffer(stagingBuffer);
}

void oka::PtRender::createMdlBuffers()
{
    assert(mSharedCtx->mResManager);
    ResourceManager* resManager = getResManager();
    const MaterialManager::TargetCode* code = mCurrentSceneRenderData->mMaterialTargetCode;
    mMaterialManager->getArgBufferData(code);

    const uint32_t argSize = mMaterialManager->getArgBufferSize(code);
    const uint32_t roSize = mMaterialManager->getReadOnlyBlockSize(code);
    const uint32_t infoSize = std::max(mMaterialManager->getResourceInfoSize(code), (uint32_t)4);
    const uint32_t mdlMaterialSize = mMaterialManager->getMdlMaterialSize(code);

    VkDeviceSize stagingSize = std::max(std::max(roSize, mdlMaterialSize), std::max(argSize, infoSize));

    Buffer* stagingBuffer = resManager->createBuffer(
        stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "Staging MDL");
    void* stagingBufferMemory = resManager->getMappedMemory(stagingBuffer);

    auto createGpuBuffer = [&](Buffer*& dest, const uint8_t* src, uint32_t size, const char* name) {
        memcpy(stagingBufferMemory, src, size);
        dest = resManager->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, name);
        resManager->copyBuffer(resManager->getVkBuffer(stagingBuffer), resManager->getVkBuffer(dest), size);
    };

    createGpuBuffer(
        mCurrentSceneRenderData->mMdlArgBuffer, mMaterialManager->getArgBufferData(code), argSize, "MDL args");
    createGpuBuffer(
        mCurrentSceneRenderData->mMdlInfoBuffer, mMaterialManager->getResourceInfoData(code), infoSize, "MDL info");
    createGpuBuffer(
        mCurrentSceneRenderData->mMdlRoBuffer, mMaterialManager->getReadOnlyBlockData(code), roSize, "MDL read only");
    createGpuBuffer(mCurrentSceneRenderData->mMdlMaterialBuffer, mMaterialManager->getMdlMaterialData(code),
                    mdlMaterialSize, "MDL mdl material");

    resManager->destroyBuffer(stagingBuffer);
}

void PtRender::drawFrame(Image* result)
{
    assert(mSharedCtx->mResManager);
    assert(mSharedCtx->mTextureManager);
    ResourceManager* resManager = getResManager();
    TextureManager* texManager = getTexManager();

    assert(mScene);
    // ZoneScoped;

    // FrameData& currFrame = mSharedCtx.getCurrentFrameData();
    const uint32_t frameIndex = getSharedContext().mFrameNumber % MAX_FRAMES_IN_FLIGHT;
    const uint64_t frameNumber = getSharedContext().mFrameNumber;

    // TODO: handle changes: resize, update
    if (getSharedContext().mFrameNumber == 0)
    {
        createVertexBuffer(*mScene);
        createIndexBuffer(*mScene);
        createInstanceBuffer(*mScene);
        createLightsBuffer(*mScene);
        createBvhBuffer(*mScene); // need to update descriptors after it

        reloadPt();
        // createMaterialBuffer(*mScene);
        createMdlBuffers();
    }

    bool needRecreateView = false;
    bool needResetAccumulation = false;
    const bool enableAccumulation = getSettingsManager()->getAs<bool>("render/pt/enableAcc");
    const bool enableUpscale = getSettingsManager()->getAs<bool>("render/pt/enableUpscale");
    const bool enableTonemap = getSharedContext().mSettingsManager->getAs<bool>("render/pt/enableTonemap");
    const uint32_t stratifiedSamplingType = getSettingsManager()->getAs<uint32_t>("render/pt/stratifiedSamplingType");
    const uint32_t debugView = getSettingsManager()->getAs<uint32_t>("render/pt/debug");

    if (mSettings.enableUpscale != enableUpscale)
    {
        needRecreateView = true;
    }
    if (mSettings.enableAccumulation != enableAccumulation ||
        stratifiedSamplingType != mSettings.stratifiedSamplingType || mSettings.debugView != debugView)
    {
        needResetAccumulation = true;
    }
    readSettings();

    if (needRecreateView)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            mNeedRecreateView[i] = true;
        }
    }
    if (mNeedRecreateView[frameIndex])
    {
        mView[frameIndex] = createView(WIDTH, HEIGHT, getSettingsManager()->getAs<uint32_t>("render/pt/spp"));
        mNeedRecreateView[frameIndex] = false;
    }

    ViewData* currView = mView[frameIndex];
    uint32_t renderWidth = currView->renderWidth;
    uint32_t renderHeight = currView->renderHeight;
    uint32_t finalWidth = currView->finalWidth;
    uint32_t finalHeight = currView->finalHeight;

    mScene->updateCamerasParams(renderWidth, renderHeight);

    Camera& cam = mScene->getCamera(getActiveCameraIndex());
    cam.updateViewMatrix();

    currView->mCamMatrices = cam.matrices;

    // check if camera is dirty?
    if (needRecreateView || needResetAccumulation ||
        (mPrevView && (glm::any(glm::notEqual(currView->mCamMatrices.perspective, mPrevView->mCamMatrices.perspective)) ||
                       glm::any(glm::notEqual(currView->mCamMatrices.view, mPrevView->mCamMatrices.view)))))
    {
        // need to reset pt iteration and accumulation
        currView->mPtIteration = 0;
    }
    else if (mPrevView)
    {
        currView->mPtIteration = mPrevView->mPtIteration;
    }
    getSettingsManager()->setAs<uint32_t>("render/pt/iteration", currView->mPtIteration);

    // at this point we reseive opened cmd buffer
    VkCommandBuffer& cmd = getSharedContext().getFrameData(frameIndex).cmdBuffer;

    Image* finalPathTracerImage = currView->mPathTracerImage;
    Image* finalImage = nullptr;

    // Path Tracer
    for (int i = 0; i < 1; ++i)
    {
        PathTracerDesc ptDesc{};
        // desc.result = mView[imageIndex]->mPathTracerImage;
        ptDesc.bvhNodes = mCurrentSceneRenderData->mBvhNodeBuffer;
        ptDesc.vb = mCurrentSceneRenderData->mVertexBuffer;
        ptDesc.ib = mCurrentSceneRenderData->mIndexBuffer;
        ptDesc.instanceConst = mCurrentSceneRenderData->mInstanceBuffer;
        ptDesc.lights = mCurrentSceneRenderData->mLightsBuffer;
        // ptDesc.materials = mCurrentSceneRenderData->mMaterialBuffer;
        ptDesc.matSampler = texManager->mMdlSampler;
        ptDesc.matTextures = texManager->textureImages;
        ptDesc.sampleBuffer = currView->mSampleBuffer;
        // desc.compositingBuffer = currView->mCompositingBuffer;

        ptDesc.mdl_argument_block = mCurrentSceneRenderData->mMdlArgBuffer;
        ptDesc.mdl_ro_data_segment = mCurrentSceneRenderData->mMdlRoBuffer;
        ptDesc.mdl_resource_infos = mCurrentSceneRenderData->mMdlInfoBuffer;
        ptDesc.mdl_mdlMaterial = mCurrentSceneRenderData->mMdlMaterialBuffer;

        PathTracerParam& pathTracerParam = ptDesc.constants;
        pathTracerParam.dimension = glm::int2(renderWidth, renderHeight);
        pathTracerParam.frameNumber = (uint32_t)getSharedContext().mFrameNumber;
        const uint32_t depth = getSharedContext().mSettingsManager->getAs<uint32_t>("render/pt/depth");
        pathTracerParam.maxDepth = depth;
        pathTracerParam.debug = (uint32_t)(mScene->mDebugViewSettings == Scene::DebugView::ePTDebug);
        pathTracerParam.camPos = glm::float4(cam.getPosition(), 1.0f);
        pathTracerParam.viewToWorld = glm::inverse(cam.matrices.view);
        pathTracerParam.worldToView = cam.matrices.view; //
        // pathTracerParam.clipToView = glm::inverse(cam.matrices.perspective);
        pathTracerParam.clipToView = cam.matrices.invPerspective;
        pathTracerParam.viewToClip = cam.matrices.perspective; //
        pathTracerParam.len = (int)0;
        pathTracerParam.spp = currView->spp;
        pathTracerParam.iteration = currView->mPtIteration;
        pathTracerParam.numLights = (uint32_t)mScene->getLights().size();
        pathTracerParam.stratifiedSamplingType = stratifiedSamplingType;
        pathTracerParam.debug = debugView;
        pathTracerParam.invDimension.x = 1.0f / (float)renderWidth;
        pathTracerParam.invDimension.y = 1.0f / (float)renderHeight;
        recordBufferBarrier(cmd, currView->mSampleBuffer, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        mPathTracer->execute(cmd, ptDesc, renderWidth * renderHeight * pathTracerParam.spp, 1, frameNumber);

        ReductionDesc reductionDesc{};
        reductionDesc.constants = ptDesc.constants;
        reductionDesc.result = currView->mPathTracerImage;
        reductionDesc.sampleBuffer = currView->mSampleBuffer;

        // buffer barrier
        recordBufferBarrier(cmd, currView->mSampleBuffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // image barrier
        recordImageBarrier(cmd, currView->mPathTracerImage, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        mReductionPass->execute(cmd, reductionDesc, renderWidth * renderHeight, 1, frameNumber);

        recordImageBarrier(cmd, currView->mPathTracerImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        finalPathTracerImage = currView->mPathTracerImage;

        if (enableAccumulation && mPrevView)
        {
            // Accumulation pass
            AccumulationDesc accDesc{};
            AccumulationParam& accParam = accDesc.constants;
            accParam.alpha = 0.1f;
            accParam.dimension = glm::int2(renderWidth, renderHeight);
            // glm::double4x4 persp = cam.prevMatrices.perspective;
            // accParam.prevClipToView = glm::inverse(persp);
            accParam.prevClipToView = mPrevView->mCamMatrices.invPerspective;
            accParam.prevViewToClip = mPrevView->mCamMatrices.perspective;
            accParam.prevWorldToView = mPrevView->mCamMatrices.view;
            glm::double4x4 view = mPrevView->mCamMatrices.view;
            accParam.prevViewToWorld = glm::inverse(view);
            // debug
            // accParam.clipToView = glm::inverse(cam.matrices.perspective);
            accParam.clipToView = cam.matrices.invPerspective;
            accParam.viewToWorld = glm::inverse(cam.matrices.view);
            accParam.isPt = 1;
            accParam.iteration = currView->mPtIteration;

            accDesc.input = finalPathTracerImage;
            accDesc.history = mPrevView->mAccumulationPathTracerImage;
            Image* accOut = currView->mAccumulationPathTracerImage;
            accDesc.output = accOut;

            recordImageBarrier(cmd, accDesc.history, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            recordImageBarrier(cmd, accOut, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            mAccumulationPathTracer->execute(cmd, accDesc, renderWidth, renderHeight, frameNumber);
            finalPathTracerImage = accOut;
        }

        ++currView->mPtIteration;
    }
    finalImage = finalPathTracerImage;

    {
        // Tonemap
        if (enableTonemap)
        {
            recordImageBarrier(cmd, currView->textureTonemapImage, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            recordImageBarrier(cmd, finalImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            TonemapDesc toneDesc{};
            Tonemapparam& toneParams = toneDesc.constants;
            toneParams.dimension.x = renderWidth;
            toneParams.dimension.y = renderHeight;
            toneParams.tonemapperType = getSharedContext().mSettingsManager->getAs<uint32_t>("render/pt/tonemapperType");
            toneDesc.input = finalImage;
            toneDesc.output = currView->textureTonemapImage;
            mTonemap->execute(cmd, toneDesc, renderWidth, renderHeight, frameNumber);
            finalImage = currView->textureTonemapImage;
        }

        // Upscale
        if (enableUpscale)
        {
            recordImageBarrier(cmd, finalImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            UpscaleDesc upscaleDesc{};
            Upscalepassparam& upscalePassParam = upscaleDesc.constants;
            upscalePassParam.dimension.x = finalWidth;
            upscalePassParam.dimension.y = finalHeight;
            upscalePassParam.invDimension.x = 1.0f / (float)finalWidth;
            upscalePassParam.invDimension.y = 1.0f / (float)finalHeight;

            upscaleDesc.input = finalImage;
            upscaleDesc.output = currView->textureUpscaleImage;

            mUpscalePass->execute(cmd, upscaleDesc, finalWidth, finalHeight, frameNumber);

            // recordImageBarrier(cmd, finalImage, VK_IMAGE_LAYOUT_GENERAL,
            //                   VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            //                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            finalImage = currView->textureUpscaleImage;
        }
    }

    recordImageBarrier(cmd, finalImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkOffset3D blitSize{};
    blitSize.x = finalWidth;
    blitSize.y = finalHeight;
    blitSize.z = 1;
    VkImageBlit imageBlitRegion{};
    imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlitRegion.srcSubresource.layerCount = 1;
    imageBlitRegion.srcOffsets[1] = blitSize;
    imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlitRegion.dstSubresource.layerCount = 1;
    imageBlitRegion.dstOffsets[1] = blitSize;
    vkCmdBlitImage(cmd, resManager->getVkImage(finalImage), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   resManager->getVkImage(result), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlitRegion,
                   VK_FILTER_NEAREST);

    recordImageBarrier(cmd, finalImage, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // assign prev resources for next frame rendering
    mPrevView = currView;
}

void PtRender::recordImageBarrier(VkCommandBuffer& cmd,
                                  Image* image,
                                  VkImageLayout newLayout,
                                  VkAccessFlags srcAccess,
                                  VkAccessFlags dstAccess,
                                  VkPipelineStageFlags sourceStage,
                                  VkPipelineStageFlags destinationStage,
                                  VkImageAspectFlags aspectMask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = getSharedContext().mResManager->getVkImage(image);
    barrier.oldLayout = getSharedContext().mResManager->getImageLayout(image);
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    getSharedContext().mResManager->setImageLayout(image, newLayout);

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void PtRender::recordBufferBarrier(VkCommandBuffer& cmd,
                                   Buffer* buff,
                                   VkAccessFlags srcAccess,
                                   VkAccessFlags dstAccess,
                                   VkPipelineStageFlags sourceStage,
                                   VkPipelineStageFlags destinationStage)
{
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer = getSharedContext().mResManager->getVkBuffer(buff);
    barrier.offset = 0;
    barrier.size = getSharedContext().mResManager->getSize(buff);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}
