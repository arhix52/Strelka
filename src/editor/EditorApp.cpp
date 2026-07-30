#include "EditorApp.h"

#include <log.h>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>

#include <tinyexr.h>
#include <stb_image_write.h>

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
    // Display creates its own command queue for independent frame pacing
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
    m_settingsManager->setAs<uint32_t>("render/pt/depth", 8);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 256);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/iteration", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0); // 0 - none, 1 - random, 2 -
                                                                               // stratified sampling, 3 -
                                                                               // optimized stratified sampling
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 1); // 0 - None, 1 - Reinhard, 2 - ACES, 3 - Filmic
    m_settingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    m_settingsManager->setAs<float>("render/cameraSpeed", 1.0f);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<bool>("render/pt/enableTonemap", true);
    m_settingsManager->setAs<bool>("render/pt/isResized", false);
    m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/misHeuristic", 0); // 0 = balance, 1 = power
    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 0); // 0 - Halton, 1 - PCG
    m_settingsManager->setAs<bool>("render/enableValidation", false);
    m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", false);
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
    // Erase all previous per-animation settings to avoid leaking keys from old scenes
    m_settingsManager->eraseByPrefix("render/animation/anim");

    char key[64];
    for (int i = 0; i < (int)m_scene->getAnimations().size(); ++i)
    {
        snprintf(key, sizeof(key), "render/animation/anim%d/state", i);
        m_settingsManager->setAs<bool>(key, false);

        snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
        m_settingsManager->setAs<float>(key, m_scene->getAnimations()[i].start);
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

    // Tear the old renderer down *before* the scene and shared context it points
    // at are replaced. ~MetalRender drains the GPU and waits for in-flight
    // completion handlers; running that after the Scene/SharedContext it holds
    // raw pointers to have already been freed is a use-after-free waiting for
    // the right timing.
    m_render.reset();

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

    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_render->setSettingsManager(m_settingsManager.get());
    m_render->setSharedContext(m_sharedCtx.get());
    m_render->setScene(m_scene.get());
    m_render->init();

    m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
    m_display->setInputHandler(m_cameraController.get());
}

void EditorApp::run()
{
    auto prevTime = std::chrono::high_resolution_clock::now();

    while (!m_display->windowShouldClose())
    {
        m_display->pollEvents();

        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double>(currentTime - prevTime).count();

        const auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
        m_cameraController->update(deltaTime, cameraSpeed);
        prevTime = currentTime;

        playAnimations(deltaTime);

        auto& selectedCam = m_scene->getCamera(m_selectedCamera);
        if (selectedCam.node != -1 && !m_cameraDetached)
        {
            // GLTF camera in animation mode
            if (m_cameraController->getCamera().moving() || m_cameraController->isRotating())
            {
                m_cameraDetached = true;
            }
            else
            {
                auto& ctrlCam = m_cameraController->getCamera();
                ctrlCam.position = selectedCam.position;
                ctrlCam.mOrientation = selectedCam.mOrientation;
                ctrlCam.updateViewMatrix();
            }
        }

        if (selectedCam.node == -1 || m_cameraDetached)
        {
            auto& ctrlCam = m_cameraController->getCamera();
            selectedCam.position = ctrlCam.position;
            selectedCam.mOrientation = ctrlCam.mOrientation;
            selectedCam.matrices = ctrlCam.matrices;
            selectedCam.updated = ctrlCam.updated;
            selectedCam.isDirty = ctrlCam.isDirty;
        }

        checkLoadingComplete();

        if (m_resized)
        {
            m_resized = false;
            m_sharedCtx->mSubframeIndex = 0;
        }

        // getMaxEDR() crosses into AppKit; the value only changes when the window
        // moves between displays, so poll it a few times a second instead of
        // every frame.
        if (std::chrono::duration<double>(currentTime - m_lastEdrQuery).count() > 0.25)
        {
            m_lastEdrQuery = currentTime;
            m_settingsManager->setAs<float>("render/post/tonemapper/maxEDR", m_display->getMaxEDR());
        }

        // Display: always runs at vsync, independent of render
        m_display->onBeginFrame();

        oka::Buffer* readyBuf = m_render->getReadyBuffer();
        if (readyBuf)
        {
            oka::ImageBuffer outputImage;
            outputImage.deviceData = readyBuf->getDevicePointer();
            outputImage.height = readyBuf->height();
            outputImage.width = readyBuf->width();
            outputImage.pixel_format = oka::BufferFormat::FLOAT4;
            outputImage.dataSize = readyBuf->width() * readyBuf->height() * readyBuf->getElementSize();
            m_display->drawFrame(outputImage);
        }

        drawUI();

        // Process pending screenshot save
        if (!m_pendingScreenshotPath.empty() && readyBuf)
        {
            saveScreenshot(readyBuf, m_pendingScreenshotPath);
            m_pendingScreenshotPath.clear();
        }

        m_display->drawUI();
        m_display->onEndFrame();

        // Enqueue the next render pass only after this frame's presentation work
        // has been committed. The renderer runs on its own command queue, but the
        // GPU still executes submissions roughly in arrival order — submitting a
        // multi-second path-trace batch first would push the compositor's work
        // behind it and stall nextDrawable() on the following frame.
        m_render->triggerRenderIfIdle();

        // Window titles go through AppKit; refreshing at vsync is pure overhead
        // and the numbers are unreadable at 60+ Hz anyway.
        if (std::chrono::duration<double>(currentTime - m_lastTitleUpdate).count() > 0.25)
        {
            m_lastTitleUpdate = currentTime;
            char title[128];
            snprintf(title, sizeof(title), "Strelka [render: %.1f ms] [%zu spp]",
                     m_render->getLastRenderTimeMs(), m_sharedCtx->mSubframeIndex);
            m_display->setWindowTitle(title);
        }
    }
}

void EditorApp::playAnimations(const float deltaTime)
{
    const float speed = m_settingsManager->getAs<float>("render/animation/speed");
    const auto& animations = m_scene->getAnimations();
    char key[64];
    for (int i = 0; i < (int)animations.size(); ++i)
    {
        snprintf(key, sizeof(key), "render/animation/anim%d/state", i);
        const bool currAnimEnable = m_settingsManager->getAs<bool>(key);

        if (currAnimEnable)
        {
            snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
            float currAnimTime = m_settingsManager->getAs<float>(key);

            const float currAnimStart = animations[i].start;
            const float currAnimEnd = animations[i].end;

            currAnimTime += deltaTime * speed;
            if (currAnimTime > currAnimEnd) currAnimTime -= (currAnimEnd - currAnimStart);
            if (currAnimTime < currAnimStart) currAnimTime = currAnimStart;
            m_settingsManager->setAs<float>(key, currAnimTime);
        }
    }
}

void EditorApp::saveScreenshot(Buffer* buf, const std::string& path)
{
    const uint32_t w = buf->width();
    const uint32_t h = buf->height();
    const float* data = static_cast<const float*>(buf->getHostPointer());

    auto dotPos = path.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? path.substr(dotPos) : "";

    if (ext == ".exr")
    {
        const char* err = nullptr;
        int ret = SaveEXR(data, w, h, 4, 0, path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to save EXR: {}", err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
        }
        else
        {
            STRELKA_INFO("Screenshot saved: {}", path);
        }
    }
    else if (ext == ".png")
    {
        std::vector<uint8_t> pixels(w * h * 4);
        for (uint32_t i = 0; i < w * h; ++i)
        {
            for (int c = 0; c < 4; ++c)
            {
                float v = std::max(0.0f, std::min(1.0f, data[i * 4 + c]));
                pixels[i * 4 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
            }
        }
        int ret = stbi_write_png(path.c_str(), w, h, 4, pixels.data(), w * 4);
        if (!ret)
        {
            STRELKA_ERROR("Failed to save PNG: {}", path);
        }
        else
        {
            STRELKA_INFO("Screenshot saved: {}", path);
        }
    }
    else
    {
        STRELKA_ERROR("Unsupported screenshot format: {}", ext);
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

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

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
            m_display->requestClose();
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

    // --- Save screenshot dialog handling ---
    if (ImGuiFileDialog::Instance()->Display("SaveScreenshotDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            m_pendingScreenshotPath = ImGuiFileDialog::Instance()->GetFilePathName();
        }
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
