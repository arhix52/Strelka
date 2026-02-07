#include <display/Display.h>
#include <render/render.h>

#include "CameraController.h"

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <optional>
#include <future>

#include "gltfloader.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "ImGuizmo.h"
#include "ImGuiFileDialog.h"

#include "Params.h"

namespace oka
{

class Editor : public ResizeHandler
{
private:
    bool m_resized = false;
    std::unique_ptr<Display> m_display;
    std::unique_ptr<SettingsManager> m_settingsManager;

    std::unique_ptr<Render> m_render;

    std::unique_ptr<GltfLoader> m_sceneLoader;

    std::unique_ptr<SharedContext> m_sharedCtx;

    // Scene m_scene;
    std::unique_ptr<Scene> m_scene;

    std::unique_ptr<CameraController> m_cameraController;

    int m_selectedCamera = 0;
    bool m_cameraDetached = false; // true when user takes manual control of a GLTF camera

    std::future<std::unique_ptr<Scene>> m_loadingFuture;
    std::string m_pendingResourcePath;
    bool m_isLoading = false;

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
        m_display->setCommandQueue(m_render->getNativeCommandQueue());
#endif
        m_display->init(1024, 768, m_settingsManager.get());
        m_display->setResizeHandler(this);
    }
    ~Editor() = default;

    void framebufferResize(int newWidth, int newHeight) override
    {
        m_settingsManager->setAs<uint32_t>("render/width", static_cast<uint32_t>(newWidth));
        m_settingsManager->setAs<uint32_t>("render/height", static_cast<uint32_t>(newHeight));
        m_resized = true;
    }

    // Compute camera position that fits the entire scene in the view frustum
    glm::vec3 computeSceneFitPosition(float fovDegrees) const
    {
        const auto& vertices = m_scene->getVertices();
        if (vertices.empty())
            return glm::vec3(0, 0, -10);

        // Compute AABB
        glm::vec3 aabbMin(std::numeric_limits<float>::max());
        glm::vec3 aabbMax(std::numeric_limits<float>::lowest());
        for (const auto& v : vertices)
        {
            aabbMin = glm::min(aabbMin, v.pos);
            aabbMax = glm::max(aabbMax, v.pos);
        }

        glm::vec3 center = (aabbMin + aabbMax) * 0.5f;
        float radius = glm::length(aabbMax - center);
        if (radius < 1e-6f)
            radius = 1.0f;

        // Distance so the bounding sphere fits in the vertical FOV
        float halfFovRad = glm::radians(fovDegrees * 0.5f);
        float distance = radius / std::tan(halfFovRad);

        // Position camera along -Z looking at center (worldForward = (0,0,-1))
        return center + glm::vec3(0.0f, 0.0f, distance);
    }

    void prepare()
    {
        m_sceneLoader->loadGltf(Params::sceneFile, *m_scene);

        // Add a free-fly "Main" camera as the last entry
        oka::Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;
        camera.position = computeSceneFitPosition(camera.fov);
        camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
        camera.updateViewMatrix();
        m_scene->addCamera(camera);

        // Select first GLTF camera (index 0) by default
        m_selectedCamera = 0;

        m_cameraController = std::make_unique<CameraController>(m_scene->getCamera(m_selectedCamera), true);
        m_display->setInputHandler(m_cameraController.get());
        loadSettings();
    }

    void loadSettings()
    {
        const std::string resourceSearchPath = Params::resourceSearchPath;
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
        m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 0); // 0 - Halton, 1 - PCG
        m_settingsManager->setAs<bool>("render/enableValidation", false);
        m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
        m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
        m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", true);
        m_settingsManager->setAs<float>("render/motionBlur/shutterTime", 1.0f / 24.0f);
        m_settingsManager->setAs<uint32_t>("render/motionBlur/shutterMode", 1); // 0=centered, 1=leading, 2=trailing
        m_settingsManager->setAs<float>("render/animation/speed", 1.0f);
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

        loadAnimSettings();
    }

    void loadAnimSettings(){
        // Animation settings
        for (int i = 0; i < m_scene->getAnimations().size(); ++i) 
        {
            //TODO: need to erase all previous settings like render/animation/anim
            std::string checkboxName = "render/animation/anim" + std::to_string(i) + "/state";
            std::string scrollName = "render/animation/anim" + std::to_string(i) + "/time";

            m_settingsManager->setAs<bool>(checkboxName.c_str(), false);
            m_settingsManager->setAs<float>(scrollName.c_str(), m_scene->getAnimations()[i].start);
        }
    }

    void checkLoadingComplete()
    {
        if (!m_isLoading)
            return;
        if (m_loadingFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return;

        auto new_scene = m_loadingFuture.get();
        m_isLoading = false;

        if (!new_scene)
            return;

        m_scene = std::move(new_scene);

        oka::Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;
        camera.position = computeSceneFitPosition(camera.fov);
        camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
        camera.updateViewMatrix();
        m_scene->addCamera(camera);

        m_selectedCamera = 0;
        m_cameraDetached = false;

        loadAnimSettings();

        m_sharedCtx = std::make_unique<SharedContext>();

        m_render.reset(RenderFactory::createRender());
        m_render->setSettingsManager(m_settingsManager.get());
        m_render->setSharedContext(m_sharedCtx.get());
        m_render->setScene(m_scene.get());
        m_render->init();

        m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
        m_display->setInputHandler(m_cameraController.get());
    }

    void run()
    {
        // Main render loop
        oka::BufferDesc desc{};
        desc.format = oka::BufferFormat::FLOAT4;
        desc.width = m_settingsManager->getAs<uint32_t>("render/width");
        desc.height = m_settingsManager->getAs<uint32_t>("render/height");

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

            playAnimations(deltaTime);

            auto& selectedCam = m_scene->getCamera(m_selectedCamera);
            if (selectedCam.node != -1 && !m_cameraDetached)
            {
                // GLTF camera in animation mode
                if (m_cameraController->getCamera().moving())
                {
                    // User wants manual control — detach from animation,
                    // keeping the current animated position as starting point
                    m_cameraDetached = true;
                }
                else
                {
                    // Sync CameraController to animated position each frame
                    // so manual takeover starts from the right place.
                    // Only copy position/orientation — preserve key/mouse state.
                    auto& ctrlCam = m_cameraController->getCamera();
                    ctrlCam.position = selectedCam.position;
                    ctrlCam.mOrientation = selectedCam.mOrientation;
                    ctrlCam.updateViewMatrix();
                }
            }

            if (selectedCam.node == -1 || m_cameraDetached)
            {
                m_scene->updateCamera(m_cameraController->getCamera(), m_selectedCamera);
            }

            checkLoadingComplete();

            if (m_resized)
            {
                m_resized = false;
                const uint32_t newW = m_settingsManager->getAs<uint32_t>("render/width");
                const uint32_t newH = m_settingsManager->getAs<uint32_t>("render/height");
                outputBuffer->resize(newW, newH);
                m_sharedCtx->mSubframeIndex = 0;
            }

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

    void playAnimations(const float deltaTime)
    {
        const float speed = m_settingsManager->getAs<float>("render/animation/speed");
        const auto& animations = m_scene->getAnimations();
        for (int i = 0; i < (int)animations.size(); ++i)
        {
            const std::string checkboxNameStr = "render/animation/anim" + std::to_string(i) + "/state";
            bool currAnimEnable = m_settingsManager->getAs<bool>(checkboxNameStr.c_str());

            if (currAnimEnable)
            {
                const std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";
                float currAnimTime = m_settingsManager->getAs<float>(scrollNameStr.c_str());

                const float currAnimStart = animations[i].start;
                const float currAnimEnd = animations[i].end;

                currAnimTime += deltaTime * speed;
                if (currAnimTime > currAnimEnd) currAnimTime -= (currAnimEnd - currAnimStart);
                if (currAnimTime < currAnimStart) currAnimTime = currAnimStart;
                m_settingsManager->setAs<float>(scrollNameStr.c_str(), currAnimTime);
            }
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
            if (ImGui::MenuItem("Open File", nullptr, false, !m_isLoading))
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
            {
                std::string sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
                STRELKA_DEBUG("Resource search path {}", resourceSearchPath);
                m_settingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
                m_pendingResourcePath = resourceSearchPath;

                auto loader = m_sceneLoader.get();
                m_loadingFuture = std::async(std::launch::async, [loader, sceneFile]() -> std::unique_ptr<Scene> {
                    auto scene = std::make_unique<Scene>();
                    if (loader->loadGltf(sceneFile, *scene))
                    {
                        return scene;
                    }
                    return nullptr;
                });
                m_isLoading = true;
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (m_isLoading)
        {
            ImGui::Begin("##Loading", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Loading scene...");
            ImGui::End();
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
            
            const uint32_t renderW = m_settingsManager->getAs<uint32_t>("render/width");
            const uint32_t renderH = m_settingsManager->getAs<uint32_t>("render/height");
            ImVec2 viewportSize = calculateAspectRatioSize(availableSize, renderW, renderH);
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

        const char* debugViewOptions[] = { "None", "Normals", "Motion Blur", "Diffuse AOV", "Specular AOV" };
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

        // Camera selection
        {
            const auto& cameras = m_scene->getCameras();
            int cameraCount = (int)cameras.size();
            if (cameraCount > 0)
            {
                const char* previewName = cameras[m_selectedCamera].name.c_str();
                if (ImGui::BeginCombo("Camera", previewName))
                {
                    for (int n = 0; n < cameraCount; n++)
                    {
                        bool is_selected = (m_selectedCamera == n);
                        if (ImGui::Selectable(cameras[n].name.c_str(), is_selected))
                        {
                            if (m_selectedCamera != n)
                            {
                                m_selectedCamera = n;
                                m_cameraDetached = false; // re-attach to animation
                                m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
                                m_sharedCtx->mSubframeIndex = 0;
                                m_settingsManager->setAs<uint32_t>("render/selectedCamera", m_selectedCamera);
                            }
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
            }
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

            const char* samplerTypeItems[] = { "Halton", "PCG" };
            static int currentSamplerTypeId = 0;
            if (ImGui::BeginCombo("Sampler", samplerTypeItems[currentSamplerTypeId]))
            {
                for (const auto& item : samplerTypeItems)
                {
                    bool is_selected = (item == samplerTypeItems[currentSamplerTypeId]);
                    if (ImGui::Selectable(item, is_selected))
                    {
                        currentSamplerTypeId = &item - samplerTypeItems;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                m_settingsManager->setAs<uint32_t>("render/pt/samplerType", currentSamplerTypeId);
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

        if (ImGui::Begin("Animations"))
        {
            const auto& animations = m_scene->getAnimations();

            if (!animations.empty())
            {
                // Playback controls
                if (ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    // Check if any animation is playing
                    bool anyPlaying = false;
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                        anyPlaying |= m_settingsManager->getAs<bool>(key.c_str());
                    }

                    // Transport bar: |<  <  Play/Pause  >  >|
                    constexpr float frameDuration = 1.0f / 24.0f;

                    // |< Reset to start
                    if (ImGui::Button("|<"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string key = "render/animation/anim" + std::to_string(i) + "/time";
                            m_settingsManager->setAs<float>(key.c_str(), animations[i].start);
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset to start");

                    ImGui::SameLine();

                    // < Step back one frame
                    if (ImGui::Button("<"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string timeKey = "render/animation/anim" + std::to_string(i) + "/time";
                            float t = m_settingsManager->getAs<float>(timeKey.c_str());
                            t = std::max(t - frameDuration, animations[i].start);
                            m_settingsManager->setAs<float>(timeKey.c_str(), t);
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous frame (1/24s)");

                    ImGui::SameLine();

                    // Play / Pause toggle
                    if (anyPlaying)
                    {
                        if (ImGui::Button("Pause"))
                        {
                            for (int i = 0; i < (int)animations.size(); ++i)
                            {
                                std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                                m_settingsManager->setAs<bool>(key.c_str(), false);
                            }
                        }
                    }
                    else
                    {
                        if (ImGui::Button(" Play"))
                        {
                            for (int i = 0; i < (int)animations.size(); ++i)
                            {
                                std::string key = "render/animation/anim" + std::to_string(i) + "/state";
                                m_settingsManager->setAs<bool>(key.c_str(), true);
                            }
                        }
                    }

                    ImGui::SameLine();

                    // > Step forward one frame
                    if (ImGui::Button(">"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string timeKey = "render/animation/anim" + std::to_string(i) + "/time";
                            float t = m_settingsManager->getAs<float>(timeKey.c_str());
                            t = std::min(t + frameDuration, animations[i].end);
                            m_settingsManager->setAs<float>(timeKey.c_str(), t);
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next frame (1/24s)");

                    ImGui::SameLine();

                    // >| Jump to end
                    if (ImGui::Button(">|"))
                    {
                        for (int i = 0; i < (int)animations.size(); ++i)
                        {
                            std::string key = "render/animation/anim" + std::to_string(i) + "/time";
                            m_settingsManager->setAs<float>(key.c_str(), animations[i].end);
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Jump to end");

                    float speed = m_settingsManager->getAs<float>("render/animation/speed");
                    ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f, "%.2fx");
                    m_settingsManager->setAs<float>("render/animation/speed", speed);
                }

                // Motion blur
                if (ImGui::CollapsingHeader("Motion Blur"))
                {
                    bool enableMotionBlur = m_settingsManager->getAs<bool>("render/enableMotionBlur");
                    if (ImGui::Checkbox("Enable Motion Blur", &enableMotionBlur))
                    {
                        m_settingsManager->setAs<bool>("render/enableMotionBlur", enableMotionBlur);
                    }
                    if (enableMotionBlur)
                    {
                        bool isMotionBlurVisible = m_settingsManager->getAs<bool>("render/isMotionBlurVisible");
                        if (ImGui::Checkbox("Show Motion Blur Effect", &isMotionBlurVisible))
                        {
                            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", isMotionBlurVisible);
                        }

                        bool enableCameraMotionBlur = m_settingsManager->getAs<bool>("render/enableCameraMotionBlur");
                        if (ImGui::Checkbox("Camera Motion Blur", &enableCameraMotionBlur))
                        {
                            m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", enableCameraMotionBlur);
                        }

                        float shutterTime = m_settingsManager->getAs<float>("render/motionBlur/shutterTime");
                        if (ImGui::SliderFloat("Shutter Duration (s)", &shutterTime, 0.001f, 1.0f, "%.3f"))
                        {
                            m_settingsManager->setAs<float>("render/motionBlur/shutterTime", shutterTime);
                        }

                        const char* shutterModes[] = { "Centered", "Leading", "Trailing" };
                        int shutterMode = (int)m_settingsManager->getAs<uint32_t>("render/motionBlur/shutterMode");
                        if (ImGui::Combo("Shutter Mode", &shutterMode, shutterModes, IM_ARRAYSIZE(shutterModes)))
                        {
                            m_settingsManager->setAs<uint32_t>("render/motionBlur/shutterMode", (uint32_t)shutterMode);
                        }
                    }
                }

                // Per-animation controls
                if (ImGui::CollapsingHeader("Clips", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    for (int i = 0; i < (int)animations.size(); ++i)
                    {
                        ImGui::PushID(i);
                        std::string checkboxNameStr = "render/animation/anim" + std::to_string(i) + "/state";
                        std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";

                        bool currAnimEnable = m_settingsManager->getAs<bool>(checkboxNameStr.c_str());
                        ImGui::Checkbox(animations[i].name.c_str(), &currAnimEnable);
                        m_settingsManager->setAs<bool>(checkboxNameStr.c_str(), currAnimEnable);

                        float currAnimTime = m_settingsManager->getAs<float>(scrollNameStr.c_str());
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                        ImGui::SliderFloat("##time", &currAnimTime, animations[i].start, animations[i].end, "%.3f s");
                        m_settingsManager->setAs<float>(scrollNameStr.c_str(), currAnimTime);

                        ImGui::PopID();
                    }
                }
            }
            else
            {
                ImGui::TextDisabled("No animations in scene");
            }

            ImGui::End();
        }

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
