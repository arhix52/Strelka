#include "EditorApp.h"

#include <log.h>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>

namespace oka
{

EditorApp::EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath)
    : m_sceneFile(sceneFile), m_resourceSearchPath(resourceSearchPath)
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

void EditorApp::framebufferResize(int newWidth, int newHeight)
{
    m_settingsManager->setAs<uint32_t>("render/width", static_cast<uint32_t>(newWidth));
    m_settingsManager->setAs<uint32_t>("render/height", static_cast<uint32_t>(newHeight));
    m_resized = true;
}

glm::vec3 EditorApp::computeSceneFitPosition(float fovDegrees) const
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

void EditorApp::prepare()
{
    m_sceneLoader->loadGltf(m_sceneFile, *m_scene);

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

void EditorApp::loadSettings()
{
    STRELKA_DEBUG("Resource search path {}", m_resourceSearchPath);

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
    m_settingsManager->setAs<uint32_t>("render/pt/misHeuristic", 0); // 0 = balance, 1 = power
    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 0); // 0 - Halton, 1 - PCG
    m_settingsManager->setAs<bool>("render/enableValidation", false);
    m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", true);
    m_settingsManager->setAs<float>("render/motionBlur/shutterTime", 1.0f / 24.0f);
    m_settingsManager->setAs<uint32_t>("render/motionBlur/shutterMode", 1); // 0=centered, 1=leading, 2=trailing
    m_settingsManager->setAs<float>("render/animation/speed", 1.0f);
    m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
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

void EditorApp::loadAnimSettings()
{
    // Animation settings
    for (int i = 0; i < (int)m_scene->getAnimations().size(); ++i)
    {
        // TODO: need to erase all previous settings like render/animation/anim
        std::string checkboxName = "render/animation/anim" + std::to_string(i) + "/state";
        std::string scrollName = "render/animation/anim" + std::to_string(i) + "/time";

        m_settingsManager->setAs<bool>(checkboxName.c_str(), false);
        m_settingsManager->setAs<float>(scrollName.c_str(), m_scene->getAnimations()[i].start);
    }
}

void EditorApp::checkLoadingComplete()
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

void EditorApp::run()
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
                // User wants manual control -- detach from animation,
                // keeping the current animated position as starting point
                m_cameraDetached = true;
            }
            else
            {
                // Sync CameraController to animated position each frame
                // so manual takeover starts from the right place.
                // Only copy position/orientation -- preserve key/mouse state.
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

void EditorApp::playAnimations(const float deltaTime)
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

void EditorApp::drawUI()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    // --- Main menu bar ---
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

    // --- File dialog handling ---
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

    // --- Panel draw calls (implementations in panels/*.cpp) ---
    drawViewportPanel();
    drawRenderSettingsPanel();
    drawAnimationPanel();

    // Rendering
    ImGui::Render();
}

} // namespace oka
