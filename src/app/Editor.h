#include <display/Display.h>
#include <render/render.h>

#include "CameraController.h"

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <optional>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "ImGuizmo.h"

#include "ImGuiFileDialog.h"

namespace oka
{

class Editor
{
private:
    std::unique_ptr<Display> m_display;
    std::unique_ptr<SettingsManager> m_settingsManager;

    std::unique_ptr<Render> m_render;

    std::unique_ptr<GltfLoader> m_sceneLoader;

    std::unique_ptr<SharedContext> m_sharedCtx;

    Scene m_scene;
    std::unique_ptr<CameraController> m_cameraController;

public:
    Editor()
    {
        m_settingsManager = std::unique_ptr<SettingsManager>(new SettingsManager());

        m_display = std::unique_ptr<Display>(DisplayFactory::createDisplay());
        m_render = std::unique_ptr<Render>(RenderFactory::createRender());
        m_sharedCtx = std::unique_ptr<SharedContext>(new SharedContext());

        m_sceneLoader = std::unique_ptr<GltfLoader>(new GltfLoader());

        m_display->init(1024, 768, m_settingsManager.get());

        m_render->setScene(&m_scene);
        m_render->setSettingsManager(m_settingsManager.get());
        m_render->setSharedContext(m_sharedCtx.get());

        prepare();

        m_render->init();
    }
    ~Editor()
    {
    }

    void prepare()
    {
        m_sceneLoader->loadGltf("C:/work/vespa/vespa.gltf", m_scene);
        oka::Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;
        camera.position = glm::vec3(0, 0, -10);
        camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
        camera.updateViewMatrix();
        m_scene.addCamera(camera);

        m_cameraController = std::unique_ptr<CameraController>(new CameraController(m_scene.getCamera(0), true));
        m_display->setInputHandler(m_cameraController.get());
        loadSettings();
    }

    void loadSettings()
    {
        const std::string sceneFile = "C:/work/vespa/vespa.gltf";
        const std::filesystem::path sceneFilePath = { sceneFile.c_str() };
        const std::string resourceSearchPath = sceneFilePath.parent_path().string();
        STRELKA_DEBUG("Resource search path {}", resourceSearchPath);

        const uint32_t imageWidth = 1024;
        const uint32_t imageHeight = 768;

        m_settingsManager->setAs<uint32_t>("render/width", imageWidth);
        m_settingsManager->setAs<uint32_t>("render/height", imageHeight);
        m_settingsManager->setAs<uint32_t>("render/pt/depth", 4);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 256);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        m_settingsManager->setAs<uint32_t>("render/pt/iteration", 0);
        m_settingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0); // 0 - none, 1 - random, 2 -
                                                                                   // stratified sampling, 3 -
                                                                                   // optimized stratified sampling
        m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0); // 0 - reinhard, 1 - aces, 2 - filmic
        m_settingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
        m_settingsManager->setAs<float>("render/cameraSpeed", 1.0f);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_settingsManager->setAs<bool>("render/pt/enableTonemap", true);
        m_settingsManager->setAs<bool>("render/pt/isResized", false);
        m_settingsManager->setAs<bool>("render/pt/needScreenshot", false);
        m_settingsManager->setAs<bool>("render/pt/screenshotSPP", false);
        m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
        m_settingsManager->setAs<bool>("render/enableValidation", 0);
        m_settingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
        // Postprocessing settings:
        m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", 100.0f);
        m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", 1.0f);
        m_settingsManager->setAs<float>("render/post/tonemapper/fStop", 4.0f);
        m_settingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", 100.0f);

        m_settingsManager->setAs<float>("render/post/gamma", 2.4f); // 0.0f - off
        // Dev settings:
        m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", 0.0f); // offset to avoid self-collision in
                                                                              // light sampling
        m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", 0.0f); // offset to avoid self-collision in
    }

    void run()
    {
        // Main render loop
        oka::BufferDesc desc{};
        desc.format = oka::BufferFormat::FLOAT4;
        desc.width = 1024;
        desc.height = 768;

        oka::Buffer* outputBuffer = m_render->createBuffer(desc);
        while (!m_display->windowShouldClose())
        {
            auto start = std::chrono::high_resolution_clock::now();

            m_display->pollEvents();

            static auto prevTime = std::chrono::high_resolution_clock::now();
            auto currentTime = std::chrono::high_resolution_clock::now();
            const double deltaTime = std::chrono::duration<double, std::milli>(currentTime - prevTime).count() / 1000.0;

            const auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
            m_cameraController->update(deltaTime, cameraSpeed);
            prevTime = currentTime;

            m_scene.updateCamera(m_cameraController->getCamera(), 0);

            m_display->onBeginFrame();

            m_render->render(outputBuffer);
            outputBuffer->map();
            oka::ImageBuffer outputImage;
            outputImage.data = outputBuffer->getHostPointer();
            outputImage.dataSize = outputBuffer->getHostDataSize();
            outputImage.height = outputBuffer->height();
            outputImage.width = outputBuffer->width();
            outputImage.pixel_format = oka::BufferFormat::FLOAT4;
            m_display->drawFrame(outputImage); // blit rendered image to swapchain

            drawUI(); // render ui to swapchain image in window resolution
            m_display->drawUI();
            m_display->onEndFrame(); // submit command buffer and present

            outputBuffer->unmap();

            const uint32_t currentSpp = m_sharedCtx->mSubframeIndex;
            auto finish = std::chrono::high_resolution_clock::now();
            const double frameTime = std::chrono::duration<double, std::milli>(finish - start).count();

            m_display->setWindowTitle((std::string("Strelka") + " [" + std::to_string(frameTime) + " ms]" + " [" +
                                       std::to_string(currentSpp) + " spp]")
                                          .c_str());
        }
    }

    void drawUI()
    {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();

        displayLightSettings(1, m_scene, 0);

        ImGuiIO& io = ImGui::GetIO();

        ImGui::BeginMainMenuBar();
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open File"))
            {
                IGFD::FileDialogConfig config;
                config.path = ".";
                ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".gltf", config);
            }

            if (ImGui::MenuItem("Exit"))
            {
                exit(0);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();

        // display Open file dialog if needed
        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }

        const char* debugItems[] = { "None", "Normals", "Diffuse AOV", "Specular AOV" };
        static int currentDebugItemId = 0;

        ImGui::Begin("Menu:"); // begin window

        if (ImGui::BeginCombo("Debug view", debugItems[currentDebugItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(debugItems); n++)
            {
                bool is_selected = (currentDebugItemId == n);
                if (ImGui::Selectable(debugItems[n], is_selected))
                {
                    currentDebugItemId = n;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        m_settingsManager->setAs<uint32_t>("render/pt/debug", currentDebugItemId);

        if (ImGui::TreeNode("Path Tracer"))
        {
            const char* rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
            static int currentRectlightSamplingMethodItemId = 0;
            if (ImGui::BeginCombo(
                    "Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
            {
                for (int n = 0; n < IM_ARRAYSIZE(rectlightSamplingMethodItems); n++)
                {
                    bool is_selected = (currentRectlightSamplingMethodItemId == n);
                    if (ImGui::Selectable(rectlightSamplingMethodItems[n], is_selected))
                    {
                        currentRectlightSamplingMethodItemId = n;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", currentRectlightSamplingMethodItemId);

            uint32_t maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
            ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16);
            m_settingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);

            uint32_t sppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
            ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000);
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);

            uint32_t sppSubframe = m_settingsManager->getAs<uint32_t>("render/pt/spp");
            ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32);
            m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);

            bool enableAccumulation = m_settingsManager->getAs<bool>("render/pt/enableAcc");
            ImGui::Checkbox("Enable Path Tracer Acc", &enableAccumulation);
            m_settingsManager->setAs<bool>("render/pt/enableAcc", enableAccumulation);

            ImGui::TreePop();
        }

        if (ImGui::Button("Capture Screen"))
        {
            m_settingsManager->setAs<bool>("render/pt/needScreenshot", true);
        }

        float cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
        ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
        m_settingsManager->setAs<float>("render/cameraSpeed", cameraSpeed);

        const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
        static int currentTonemapItemId = 1;
        if (ImGui::BeginCombo("Tonemap", tonemapItems[currentTonemapItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
            {
                bool is_selected = (currentTonemapItemId == n);
                if (ImGui::Selectable(tonemapItems[n], is_selected))
                {
                    currentTonemapItemId = n;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", currentTonemapItemId);

        float gamma = m_settingsManager->getAs<float>("render/post/gamma");
        ImGui::InputFloat("Gamma", (float*)&gamma, 0.5);
        m_settingsManager->setAs<float>("render/post/gamma", gamma);

        float materialRayTmin = m_settingsManager->getAs<float>("render/pt/dev/materialRayTmin");
        ImGui::InputFloat("Material ray T min", (float*)&materialRayTmin, 0.1);
        m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);
        float shadowRayTmin = m_settingsManager->getAs<float>("render/pt/dev/shadowRayTmin");
        ImGui::InputFloat("Shadow ray T min", (float*)&shadowRayTmin, 0.1);
        m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);

        ImGui::End(); // end window

        // Rendering
        ImGui::Render();
    }

    void showGizmo(Camera& cam, float camDistance, float* matrix, ImGuizmo::OPERATION operation)
    {
        static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
        glm::float4x4 cameraView = cam.getView();
        glm::float4x4 cameraProjection = cam.getPerspective();
        ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(cameraProjection), operation, mCurrentGizmoMode,
                             matrix, NULL, nullptr, nullptr, nullptr);
    }

    void displayLightSettings(uint32_t lightId, Scene& scene, const uint32_t& selectedCamera)
    {
        static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);

        Camera& cam = scene.getCamera(selectedCamera);
        glm::float3 camPos = cam.getPosition();

        // get CPU light
        std::vector<Scene::UniformLightDesc>& lightDescs = scene.getLightsDesc();
        Scene::UniformLightDesc& currLightDesc = lightDescs[lightId];

        if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
            mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
            mCurrentGizmoOperation = ImGuizmo::ROTATE;

        glm::float2 scale = { currLightDesc.width, currLightDesc.height };
        ImGui::Text("Rectangle light");
        ImGui::Spacing();
        ImGui::AlignTextToFramePadding();
        ImGui::DragFloat3("Position", &currLightDesc.position.x);
        ImGui::Spacing();
        ImGui::DragFloat3("Orientation", &currLightDesc.orientation.x);
        ImGui::Spacing();
        ImGui::DragFloat2("Width/Height", &scale.x, 0.1f, 0.005f);
        ImGui::Spacing();
        ImGui::ColorEdit3("Color", &currLightDesc.color.x);
        ImGui::DragFloat("Intensity", &currLightDesc.intensity, 1.0f, 1.0f);
        currLightDesc.intensity = glm::clamp(currLightDesc.intensity, 1.0f, std::numeric_limits<float>::max());
        // upd current scale params.
        scale = glm::clamp(scale, 0.005f, std::numeric_limits<float>::max());
        currLightDesc.width = scale.x;
        currLightDesc.height = scale.y;

        ImGuizmo::SetID(lightId);

        // construct final xform for imguizmo
        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), currLightDesc.position);
        glm::quat rotation = glm::quat(glm::radians(currLightDesc.orientation)); // to quaternion
        const glm::float4x4 rotationMatrix{ rotation };
        // light have o-y o-z scaling
        const glm::float4x4 scaleMatrix =
            glm::scale(glm::float4x4(1.0f), glm::float3(1.0f, currLightDesc.width, currLightDesc.height));

        float camDist = glm::distance(camPos, currLightDesc.position);
        glm::float4x4 lightXform = translationMatrix * rotationMatrix * scaleMatrix;

        // show controls
        showGizmo(cam, camDist, &lightXform[0][0], mCurrentGizmoOperation);

        // need to deconstruct final xform to components
        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(&lightXform[0][0], matrixTranslation, matrixRotation, matrixScale);

        // write result to description
        currLightDesc.position = glm::float3(matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
        currLightDesc.orientation = glm::float3(matrixRotation[0], matrixRotation[1], matrixRotation[2]);
        currLightDesc.width = matrixScale[1];
        currLightDesc.height = matrixScale[2];

        // update in scene
        Scene::UniformLightDesc desc{};
        desc.position = currLightDesc.position;
        desc.orientation = currLightDesc.orientation;
        desc.width = currLightDesc.width;
        desc.height = currLightDesc.height;
        desc.color = currLightDesc.color;
        desc.intensity = currLightDesc.intensity;
        scene.updateLight(lightId, desc);
        scene.updateInstanceTransform(scene.mLightIdToInstanceId[lightId], lightXform);
    }
};
} // namespace oka
