#include <display/Display.h>
#include <render/render.h>

#include "CameraController.h"

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include "gltfloader.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "ImGuizmo.h"
#include "ImGuiFileDialog.h"
#include "log.h"

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

    // Scene m_scene;
    std::unique_ptr<Scene> m_scene;

    std::unique_ptr<CameraController> m_cameraController;

public:
    Editor()
    {
        m_settingsManager = std::make_unique<SettingsManager>();

        m_scene = std::make_unique<Scene>();
        m_display = std::unique_ptr<Display>(DisplayFactory::createDisplay());
        m_render = std::unique_ptr<Render>(RenderFactory::createRender());
        m_sharedCtx = std::make_unique<SharedContext>();

        m_sceneLoader = std::make_unique<GltfLoader>();

        m_render->setScene(m_scene.get());
        m_render->setSettingsManager(m_settingsManager.get());
        m_render->setSharedContext(m_sharedCtx.get());

        prepare();

        m_render->init();
#ifdef __APPLE__
        m_display->setNativeDevice(m_render->getNativeDevicePtr());
#endif
        m_display->init(1024, 768, m_settingsManager.get());
    }
    ~Editor() = default;

    void prepare()
    {
        oka::Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;
        camera.position = glm::vec3(0, 0, -10);
        camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
        camera.updateViewMatrix();
        m_scene->addCamera(camera);

        m_cameraController = std::make_unique<CameraController>(m_scene->getCamera(0), true);
        m_display->setInputHandler(m_cameraController.get());
        loadSettings();
    }

    void loadSettings()
    {
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
        m_settingsManager->setAs<bool>("render/enableValidation", false);
        m_settingsManager->setAs<std::string>("resource/searchPath", "");
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

            m_scene->updateCamera(m_cameraController->getCamera(), 0);

            m_display->onBeginFrame();

            auto maxEDR = m_display->getMaxEDR();
            m_settingsManager->setAs<float>("render/post/tonemapper/maxEDR", maxEDR);

            m_render->render(outputBuffer);
            oka::ImageBuffer outputImage;
            outputImage.deviceData = outputBuffer->getDevicePointer();
            outputImage.height = outputBuffer->height();
            outputImage.width = outputBuffer->width();
            outputImage.pixel_format = oka::BufferFormat::FLOAT4;
            outputImage.dataSize = outputBuffer->width() * outputBuffer->height() * outputBuffer->getElementSize();
            m_display->drawFrame(outputImage); // blit rendered image to swapchain

            drawUI(); // render ui to swapchain image in window resolution
            m_display->drawUI();
            m_display->onEndFrame(); // submit command buffer and present

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


        ImGuiIO& io = ImGui::GetIO();

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

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
                std::string sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                STRELKA_DEBUG("Resource search path {}", resourceSearchPath);
                m_settingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
                std::unique_ptr<Scene> new_scene(new Scene());
                if (m_sceneLoader->loadGltf(sceneFile, *new_scene))
                {
                    m_scene = std::move(new_scene);

                    oka::Camera camera;
                    camera.name = "Main";
                    camera.fov = 45.0f;
                    camera.position = glm::vec3(0, 0, -10);
                    camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
                    camera.updateViewMatrix();
                    m_scene->addCamera(camera);

                    m_sharedCtx = std::make_unique<SharedContext>();

                    m_render.reset(RenderFactory::createRender());
                    m_render->setSettingsManager(m_settingsManager.get());
                    m_render->setSharedContext(m_sharedCtx.get());
                    m_render->setScene(m_scene.get());
                    m_render->init();

                    m_cameraController->setCamera(m_scene->getCamera(0));
                }
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }

        static bool mIsHoveredViewport = false; // need to track previous state
        bool thisFrameHovered = false;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        if (ImGui::Begin("Viewport"))
        {
            const ImVec2 availableSize = ImGui::GetContentRegionAvail();
            const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;

            auto calculateAspectRatioSize = [](ImVec2 availableSize, int fixedWidth, int fixedHeight)
            {
                float aspectRatio = static_cast<float>(fixedWidth) / static_cast<float>(fixedHeight);
                float width = availableSize.x;
                float height = availableSize.x / aspectRatio;
                if (height > availableSize.y)
                {
                    height = availableSize.y;
                    width = height * aspectRatio;
                }
                return ImVec2(width, height);
            };
            
            // Even padding top and bottom
            auto calculateVerticalPadding = [](ImVec2 availableSize, float renderedHeight) {
                return (availableSize.y - renderedHeight) / 2.0f; 
            };
            
            ImVec2 viewportSize = calculateAspectRatioSize(availableSize, 1024, 768);
            float verticalPadding = calculateVerticalPadding(availableSize, viewportSize.y);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + verticalPadding);
            ImGui::ImageButton(m_display->getDisplayNativeTexure(), viewportSize);

            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y + verticalPadding, viewportSize.x, viewportSize.y);

            ImGui::PopStyleVar();

            if (ImGui::IsItemHovered())
            {
                if (ImGui::IsKeyDown(ImGuiKey_Space))
                {
                    // picking code, selecting
                }
                m_display->setViewPortHovered(true);
                thisFrameHovered = true;
            }
        }

        if (mIsHoveredViewport && !thisFrameHovered)
        {
            // if mouse leaves viewport -> reset camera movement affected by keyboard
            m_display->setViewPortHovered(false);
            m_cameraController->setViewportHovered(false);
        } 
        mIsHoveredViewport = thisFrameHovered;

        ImGui::End();
        ImGui::PopStyleVar();

        // TODO: move to separate imgui widget
        // displayLightSettings(1, *m_scene, 0);

        ImGui::Begin("Render Settings:"); // begin window

        const char* debugViewOptions[] = { "None", "Normals", "Diffuse AOV", "Specular AOV" };
        static int currentDebugViewOption = 0;
        if (ImGui::BeginCombo("Debug view", debugViewOptions[currentDebugViewOption]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(debugViewOptions); n++)
            {
                bool is_selected = (currentDebugViewOption == n);
                if (ImGui::Selectable(debugViewOptions[n], is_selected))
                {
                    if (currentDebugViewOption != n)
                    {
                        currentDebugViewOption = n;
                        m_settingsManager->setAs<uint32_t>("render/pt/debug", currentDebugViewOption);
                    }
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (ImGui::TreeNode("Path Tracer"))
        {
            const char* rectlightSamplingMethodItems[] = { "Uniform", "Advanced" };
            static int currentRectlightSamplingMethodItemId = 0;
            if (ImGui::BeginCombo("Rect Light Sampling", rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]))
            {
                for (const auto& item : rectlightSamplingMethodItems)
                {
                    bool is_selected = (item == rectlightSamplingMethodItems[currentRectlightSamplingMethodItemId]);
                    if (ImGui::Selectable(item, is_selected))
                    {
                        currentRectlightSamplingMethodItemId = &item - rectlightSamplingMethodItems;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", currentRectlightSamplingMethodItemId);
                ImGui::EndCombo();
            }

            auto maxDepth = m_settingsManager->getAs<uint32_t>("render/pt/depth");
            if (ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);
            }

            auto sppTotal = m_settingsManager->getAs<uint32_t>("render/pt/sppTotal");
            if (ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);
            }

            auto sppSubframe = m_settingsManager->getAs<uint32_t>("render/pt/spp");
            if (ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32))
            {
                m_settingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);
            }

            bool accumulationEnabled = m_settingsManager->getAs<bool>("render/pt/enableAcc");
            if (ImGui::Checkbox("Enable Path Tracer Acc", &accumulationEnabled))
            {
                m_settingsManager->setAs<bool>("render/pt/enableAcc", accumulationEnabled);
            }

            ImGui::TreePop();
        }

        if (ImGui::Button("Capture Screen"))
        {
            m_settingsManager->setAs<bool>("render/pt/needScreenshot", true);
        }

        auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
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

        auto gamma = m_settingsManager->getAs<float>("render/post/gamma");
        ImGui::InputFloat("Gamma", (float*)&gamma, 0.5);
        m_settingsManager->setAs<float>("render/post/gamma", gamma);

        auto materialRayTmin = m_settingsManager->getAs<float>("render/pt/dev/materialRayTmin");
        ImGui::InputFloat("Material ray T min", (float*)&materialRayTmin, 0.1);
        m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", materialRayTmin);
        auto shadowRayTmin = m_settingsManager->getAs<float>("render/pt/dev/shadowRayTmin");
        ImGui::InputFloat("Shadow ray T min", (float*)&shadowRayTmin, 0.1);
        m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", shadowRayTmin);

        ImGui::End(); // end window

        // Rendering
        ImGui::Render();
    }

    void showGizmo(Camera& cam, float* matrix, ImGuizmo::OPERATION operation)
    {
        static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
        glm::float4x4 cameraView = cam.matrices.view;
        glm::float4x4 cameraProjection = cam.matrices.perspective;
        ImGuizmo::Manipulate(
            glm::value_ptr(cameraView), glm::value_ptr(cameraProjection), operation, mCurrentGizmoMode, matrix);
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

        
        ImGui::Text("Rectangle light");
        ImGui::Spacing();
        ImGui::AlignTextToFramePadding();
        ImGui::DragFloat3("Position", &currLightDesc.position.x);
        ImGui::Spacing();
        ImGui::DragFloat3("Orientation", &currLightDesc.orientation.x);
        ImGui::Spacing();
        float width_height[2] = { currLightDesc.width, currLightDesc.height };
        ImGui::DragFloat2("Width/Height", width_height, 0.1f, 0.005f);
        ImGui::Spacing();
        ImGui::ColorEdit3("Color", &currLightDesc.color.x);
        ImGui::DragFloat("Intensity", &currLightDesc.intensity, 1.0f, 1.0f);
        currLightDesc.intensity = glm::clamp(currLightDesc.intensity, 1.0f, std::numeric_limits<float>::max());
        // upd current scale params.
        currLightDesc.width = glm::clamp(width_height[0], 0.005f, std::numeric_limits<float>::max());
        currLightDesc.height = glm::clamp(width_height[1], 0.005f, std::numeric_limits<float>::max());

        ImGuizmo::SetID(lightId);

        // construct final xform for imguizmo
        const glm::float4x4 translationMatrix = glm::translate(glm::float4x4(1.0f), currLightDesc.position);
        glm::quat rotation = glm::quat(glm::radians(currLightDesc.orientation)); // to quaternion
        const glm::float4x4 rotationMatrix{ rotation };
        glm::float3 scale = { currLightDesc.width, currLightDesc.height, 1.0f };
        const glm::float4x4 scaleMatrix = glm::scale(glm::float4x4(1.0f), scale);

        glm::float4x4 lightXform = translationMatrix * rotationMatrix * scaleMatrix;

        // show controls
        showGizmo(cam, &lightXform[0][0], mCurrentGizmoOperation);

        // need to deconstruct final xform to components
        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(&lightXform[0][0], matrixTranslation, matrixRotation, matrixScale);

        // write result to description
        currLightDesc.position = glm::float3(matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
        currLightDesc.orientation = glm::float3(matrixRotation[0], matrixRotation[1], matrixRotation[2]);
        // currLightDesc.width = matrixScale[1];
        // currLightDesc.height = matrixScale[2];

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
