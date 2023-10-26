#include "RenderPass.h"

#include "Camera.h"
#include "Instancer.h"
#include "Material.h"
#include "Mesh.h"
#include "BasisCurves.h"
#include "Light.h"
#include "RenderBuffer.h"
#include "Tokens.h"

#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/basisCurves.h>

#include <log.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>

PXR_NAMESPACE_OPEN_SCOPE

HdStrelkaRenderPass::HdStrelkaRenderPass(HdRenderIndex* index,
                                         const HdRprimCollection& collection,
                                         const HdRenderSettingsMap& settings,
                                         oka::Render* renderer,
                                         oka::Scene* scene)
    : HdRenderPass(index, collection),
      m_settings(settings),
      m_isConverged(false),
      m_lastSceneStateVersion(UINT32_MAX),
      m_lastRenderSettingsVersion(UINT32_MAX),
      mRenderer(renderer),
      mScene(scene)
{
}

HdStrelkaRenderPass::~HdStrelkaRenderPass()
{
}

bool HdStrelkaRenderPass::IsConverged() const
{
    return m_isConverged;
}

//  valid range of coordinates [-1; 1]
uint32_t packNormal(const glm::float3& normal)
{
    uint32_t packed = (uint32_t)((normal.x + 1.0f) / 2.0f * 511.99999f);
    packed += (uint32_t)((normal.y + 1.0f) / 2.0f * 511.99999f) << 10;
    packed += (uint32_t)((normal.z + 1.0f) / 2.0f * 511.99999f) << 20;
    return packed;
}

//  valid range of coordinates [-10; 10]
uint32_t packUV(const glm::float2& uv)
{
    int32_t packed = (uint32_t)((uv.x + 10.0f) / 20.0f * 16383.99999f);
    packed += (uint32_t)((uv.y + 10.0f) / 20.0f * 16383.99999f) << 16;
    return packed;
}

void HdStrelkaRenderPass::_BakeMeshInstance(const HdStrelkaMesh* mesh, GfMatrix4d transform, uint32_t materialIndex)
{
    const GfMatrix4d normalMatrix = transform.GetInverse().GetTranspose();

    const std::vector<GfVec3f>& meshPoints = mesh->GetPoints();
    const std::vector<GfVec3f>& meshNormals = mesh->GetNormals();
    const std::vector<GfVec3f>& meshTangents = mesh->GetTangents();
    const std::vector<GfVec3i>& meshFaces = mesh->GetFaces();
    const std::vector<GfVec2f>& meshUVs = mesh->GetUVs();

    TF_VERIFY(meshPoints.size() == meshNormals.size());
    const size_t vertexCount = meshPoints.size();

    std::vector<oka::Scene::Vertex> vertices(vertexCount);
    std::vector<uint32_t> indices(meshFaces.size() * 3);

    for (size_t j = 0; j < meshFaces.size(); ++j)
    {
        const GfVec3i& vertexIndices = meshFaces[j];
        indices[j * 3 + 0] = vertexIndices[0];
        indices[j * 3 + 1] = vertexIndices[1];
        indices[j * 3 + 2] = vertexIndices[2];
    }
    for (size_t j = 0; j < vertexCount; ++j)
    {
        const GfVec3f& point = meshPoints[j];
        const GfVec3f& normal = meshNormals[j];
        const GfVec3f& tangent = meshTangents[j];

        oka::Scene::Vertex& vertex = vertices[j];
        vertex.pos[0] = point[0];
        vertex.pos[1] = point[1];
        vertex.pos[2] = point[2];

        const glm::float3 glmNormal = glm::float3(normal[0], normal[1], normal[2]);
        vertex.normal = packNormal(glmNormal);
        const glm::float3 glmTangent = glm::float3(tangent[0], tangent[1], tangent[2]);
        vertex.tangent = packNormal(glmTangent);

        // Texture coord
        if (!meshUVs.empty())
        {
            const GfVec2f& uv = meshUVs[j];
            const glm::float2 glmUV = glm::float2(uv[0], 1.0f - uv[1]); // Flip v coordinate
            vertex.uv = packUV(glmUV);
        }
    }

    glm::float4x4 glmTransform;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            glmTransform[i][j] = (float)transform[i][j];
        }
    }

    uint32_t meshId = mScene->createMesh(vertices, indices);
    assert(meshId != -1);
    uint32_t instId = mScene->createInstance(oka::Instance::Type::eMesh, meshId, materialIndex, glmTransform);
    assert(instId != -1);
}

void HdStrelkaRenderPass::_BakeMeshes(HdRenderIndex* renderIndex, GfMatrix4d rootTransform)
{
    TfHashMap<SdfPath, uint32_t, SdfPath::Hash> materialMapping;
    materialMapping[SdfPath::EmptyPath()] = 0;
    auto getOrCreateMaterial = [&](const SdfPath& materialId) {
        uint32_t materialIndex = 0;
        if (materialMapping.find(materialId) != materialMapping.end())
        {
            materialIndex = materialMapping[materialId];
        }
        else
        {
            HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->material, materialId);
            HdStrelkaMaterial* material = dynamic_cast<HdStrelkaMaterial*>(sprim);

            if (material->isMdl())
            {
                const std::string& fileUri = material->getFileUri();
                const std::string& name = material->getSubIdentifier();
                oka::Scene::MaterialDescription materialDesc;
                materialDesc.file = fileUri;
                materialDesc.name = name;
                materialDesc.type = oka::Scene::MaterialDescription::Type::eMdl;
                materialDesc.params = material->getParams();
                materialIndex = mScene->addMaterial(materialDesc);
            }
            else
            {
                const std::string& code = material->GetStrelkaMaterial();
                const std::string& name = material->getSubIdentifier();
                oka::Scene::MaterialDescription materialDesc;
                materialDesc.name = name;
                materialDesc.code = code;
                materialDesc.type = oka::Scene::MaterialDescription::Type::eMaterialX;
                materialDesc.params = material->getParams();
                materialIndex = mScene->addMaterial(materialDesc);
            }
            materialMapping[materialId] = materialIndex;
        }
        return materialIndex;
    };
    for (const auto& rprimId : renderIndex->GetRprimIds())
    {
        const HdRprim* rprim = renderIndex->GetRprim(rprimId);
        if (dynamic_cast<const HdMesh*>(rprim))
        {
            const HdStrelkaMesh* mesh = dynamic_cast<const HdStrelkaMesh*>(rprim);

            if (!mesh->IsVisible())
            {
                // TODO: add UI/setting control here
                continue;
            }

            const TfToken renderTag = mesh->GetRenderTag();
            if ((renderTag != "geometry") && (renderTag != "render"))
            {
                // skip all proxy meshes
                continue;
            }

            VtMatrix4dArray transforms;
            const SdfPath& instancerId = mesh->GetInstancerId();

            if (instancerId.IsEmpty())
            {
                transforms.resize(1);
                transforms[0] = GfMatrix4d(1.0);
            }
            else
            {
                HdInstancer* boxedInstancer = renderIndex->GetInstancer(instancerId);
                HdStrelkaInstancer* instancer = dynamic_cast<HdStrelkaInstancer*>(boxedInstancer);

                const SdfPath& meshId = mesh->GetId();
                transforms = instancer->ComputeInstanceTransforms(meshId);
            }

            const SdfPath& materialId = mesh->GetMaterialId();
            const std::string& materialName = materialId.GetString();

            STRELKA_INFO("Hydra: Mesh: {0} \t Material: {1}", mesh->getName(), materialName.c_str());

            uint32_t materialIndex = 0;

            if (materialId.IsEmpty())
            {
                GfVec3f color(1.0f);
                if (mesh->HasColor())
                {
                    color = mesh->GetColor();
                }
                // materialName += "_color";
                const std::string& fileUri = "default.mdl";
                const std::string& name = "default_material";
                oka::Scene::MaterialDescription material;
                material.file = fileUri;
                material.name = name;
                material.type = oka::Scene::MaterialDescription::Type::eMdl;
                material.color = glm::float3(color[0], color[1], color[2]);
                material.hasColor = true;
                oka::MaterialManager::Param colorParam = {};
                colorParam.name = "diffuse_color";
                colorParam.type = oka::MaterialManager::Param::Type::eFloat3;
                colorParam.value.resize(sizeof(float) * 3);
                memcpy(colorParam.value.data(), glm::value_ptr(material.color), sizeof(float) * 3);
                material.params.push_back(colorParam);
                materialIndex = mScene->addMaterial(material);
            }
            else
            {
                materialIndex = getOrCreateMaterial(materialId);
            }
            const GfMatrix4d& prototypeTransform = mesh->GetPrototypeTransform();

            for (size_t i = 0; i < transforms.size(); i++)
            {
                const GfMatrix4d transform = prototypeTransform * transforms[i]; // *rootTransform;
                // GfMatrix4d transform = GfMatrix4d(1.0);
                _BakeMeshInstance(mesh, transform, materialIndex);
            }
        }
        else if (dynamic_cast<const HdBasisCurves*>(rprim))
        {
            const HdStrelkaBasisCurves* curve = dynamic_cast<const HdStrelkaBasisCurves*>(rprim);
            const std::vector<glm::float3>& points = curve->GetPoints();
            const std::vector<float>& widths = curve->GetWidths();
            const std::vector<uint32_t>& vertexCounts = curve->GetVertexCounts();

            const SdfPath& materialId = curve->GetMaterialId();
            const std::string& materialName = materialId.GetString();

            STRELKA_INFO("Hydra: Curve: {0} \t Material: {1}", curve->getName(), materialName.c_str());
            const uint32_t materialIndex = getOrCreateMaterial(materialId);

            const GfMatrix4d& prototypeTransform = curve->GetPrototypeTransform();
            glm::float4x4 glmTransform;
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    glmTransform[i][j] = (float)prototypeTransform[i][j];
                }
            }
            uint32_t curveId = mScene->createCurve(oka::Curve::Type::eCubic, vertexCounts, points, widths);
            mScene->createInstance(oka::Instance::Type::eCurve, curveId, materialIndex, glmTransform, -1);
        }
    }
    STRELKA_INFO("Meshes: {}", mScene->getMeshes().size());
    STRELKA_INFO("Instances: {}", mScene->getInstances().size());
    STRELKA_INFO("Materials: {}", mScene->getMaterials().size());
    STRELKA_INFO("Curves: {}", mScene->getCurves().size());
}


void HdStrelkaRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState, const TfTokenVector& renderTags)
{
    TF_UNUSED(renderTags);

    HD_TRACE_FUNCTION();
    HF_MALLOC_TAG_FUNCTION();

    m_isConverged = false;

    const auto* camera = dynamic_cast<const HdStrelkaCamera*>(renderPassState->GetCamera());

    if (!camera)
    {
        return;
    }

    const HdRenderPassAovBindingVector& aovBindings = renderPassState->GetAovBindings();

    if (aovBindings.empty())
    {
        return;
    }

    const HdRenderPassAovBinding* colorAovBinding = nullptr;

    for (const HdRenderPassAovBinding& aovBinding : aovBindings)
    {
        if (aovBinding.aovName != HdAovTokens->color)
        {
            HdStrelkaRenderBuffer* renderBuffer = dynamic_cast<HdStrelkaRenderBuffer*>(aovBinding.renderBuffer);
            renderBuffer->SetConverged(true);
            continue;
        }

        colorAovBinding = &aovBinding;
    }

    if (!colorAovBinding)
    {
        return;
    }

    HdRenderIndex* renderIndex = GetRenderIndex();
    HdChangeTracker& changeTracker = renderIndex->GetChangeTracker();
    HdRenderDelegate* renderDelegate = renderIndex->GetRenderDelegate();
    HdStrelkaRenderBuffer* renderBuffer = dynamic_cast<HdStrelkaRenderBuffer*>(colorAovBinding->renderBuffer);

    uint32_t sceneStateVersion = changeTracker.GetSceneStateVersion();
    uint32_t renderSettingsStateVersion = renderDelegate->GetRenderSettingsVersion();
    bool sceneChanged = (sceneStateVersion != m_lastSceneStateVersion);
    bool renderSettingsChanged = (renderSettingsStateVersion != m_lastRenderSettingsVersion);

    // if (!sceneChanged && !renderSettingsChanged)
    //{
    //    renderBuffer->SetConverged(true);
    //    return;
    //}

    oka::Buffer* outputImage = renderBuffer->GetResource(false).UncheckedGet<oka::Buffer*>();

    renderBuffer->SetConverged(false);

    m_lastSceneStateVersion = sceneStateVersion;
    m_lastRenderSettingsVersion = renderSettingsStateVersion;

    // Transform scene into camera space to increase floating point precision.
    GfMatrix4d viewMatrix = camera->GetTransform().GetInverse();

    static int counter = 0;

    if (counter == 0)
    {
        ++counter;

        _BakeMeshes(renderIndex, viewMatrix);

        m_rootMatrix = viewMatrix;

        mRenderer->setScene(mScene);

        const uint32_t camIndex = camera->GetCameraIndex();
        // mRenderer->setActiveCameraIndex(camIndex);

        oka::Scene::UniformLightDesc desc{};
        desc.color = glm::float3(1.0f);
        desc.height = 0.4f;
        desc.width = 0.4f;
        desc.position = glm::float3(0, 1.1, 0.67);
        desc.orientation = glm::float3(179.68, 29.77, -89.97);
        desc.intensity = 160.0f;

        static const TfTokenVector lightTypes = { HdPrimTypeTokens->domeLight,   HdPrimTypeTokens->simpleLight,
                                                  HdPrimTypeTokens->sphereLight, HdPrimTypeTokens->rectLight,
                                                  HdPrimTypeTokens->diskLight,   HdPrimTypeTokens->cylinderLight,
                                                  HdPrimTypeTokens->distantLight };
        size_t count = 0;
        // TF_FOR_ALL(it, lightTypes)
        {
            // TODO: refactor this to more generic code, templates?
            if (renderIndex->IsSprimTypeSupported(HdPrimTypeTokens->rectLight))
            {
                SdfPathVector sprimPaths =
                    renderIndex->GetSprimSubtree(HdPrimTypeTokens->rectLight, SdfPath::AbsoluteRootPath());
                for (int lightIdx = 0; lightIdx < sprimPaths.size(); ++lightIdx)
                {
                    HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->rectLight, sprimPaths[lightIdx]);
                    HdStrelkaLight* light = dynamic_cast<HdStrelkaLight*>(sprim);
                    mScene->createLight(light->getLightDesc());
                }
            }
            if (renderIndex->IsSprimTypeSupported(HdPrimTypeTokens->diskLight))
            {
                SdfPathVector sprimPaths =
                    renderIndex->GetSprimSubtree(HdPrimTypeTokens->diskLight, SdfPath::AbsoluteRootPath());
                for (int lightIdx = 0; lightIdx < sprimPaths.size(); ++lightIdx)
                {
                    HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->diskLight, sprimPaths[lightIdx]);
                    HdStrelkaLight* light = dynamic_cast<HdStrelkaLight*>(sprim);
                    mScene->createLight(light->getLightDesc());
                }
            }
            if (renderIndex->IsSprimTypeSupported(HdPrimTypeTokens->sphereLight))
            {
                SdfPathVector sprimPaths =
                    renderIndex->GetSprimSubtree(HdPrimTypeTokens->sphereLight, SdfPath::AbsoluteRootPath());
                for (int lightIdx = 0; lightIdx < sprimPaths.size(); ++lightIdx)
                {
                    HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->sphereLight, sprimPaths[lightIdx]);
                    HdStrelkaLight* light = dynamic_cast<HdStrelkaLight*>(sprim);
                    mScene->createLight(light->getLightDesc());
                }
            }
            if (renderIndex->IsSprimTypeSupported(HdPrimTypeTokens->distantLight))
            {
                SdfPathVector sprimPaths =
                    renderIndex->GetSprimSubtree(HdPrimTypeTokens->distantLight, SdfPath::AbsoluteRootPath());
                for (int lightIdx = 0; lightIdx < sprimPaths.size(); ++lightIdx)
                {
                    HdSprim* sprim = renderIndex->GetSprim(HdPrimTypeTokens->distantLight, sprimPaths[lightIdx]);
                    HdStrelkaLight* light = dynamic_cast<HdStrelkaLight*>(sprim);
                    mScene->createLight(light->getLightDesc());
                }
            }
        }
    }
    // mScene.createLight(desc);

    float* img_data = (float*)renderBuffer->Map();

    mRenderer->render(outputImage);

    renderBuffer->Unmap();
    // renderBuffer->SetConverged(true);

    m_isConverged = true;
}

PXR_NAMESPACE_CLOSE_SCOPE
