#pragma once

#include <strelka/display/display.h>
#include <strelka/render/render.h>

#include "CameraController.h"

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <memory>
#include <optional>
#include <future>
#include <string>

#include <strelka/sceneloader/gltfloader.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "ImGuizmo.h"
#include "ImGuiFileDialog.h"

namespace oka
{

class EditorApp : public ResizeHandler
{
private:
    bool m_resized = false;
    std::unique_ptr<Display> m_display;
    std::unique_ptr<SettingsManager> m_settingsManager;
    std::unique_ptr<GltfLoader> m_sceneLoader;
    std::unique_ptr<SharedContext> m_sharedCtx;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<CameraController> m_cameraController;

    // Render must be declared after scene/sharedCtx/settings so it is
    // destroyed first (reverse declaration order), since it holds raw
    // pointers to them and its destructor drains the GPU.
    std::unique_ptr<Render> m_render;

    int m_selectedCamera = 0;
    bool m_cameraDetached = false; // true when user takes manual control of a GLTF camera

    std::future<std::unique_ptr<Scene>> m_loadingFuture;
    std::string m_pendingResourcePath;
    bool m_isLoading = false;

    std::string m_sceneFile;
    std::string m_resourceSearchPath;

    std::string m_pendingScreenshotPath;

    // Throttles for AppKit round-trips that do not need per-frame accuracy.
    std::chrono::high_resolution_clock::time_point m_lastTitleUpdate{};
    std::chrono::high_resolution_clock::time_point m_lastEdrQuery{};

public:
    EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath);
    ~EditorApp() = default;

    void framebufferResize(int newWidth, int newHeight) override;

    /// Compute camera position that fits the entire scene in the view frustum.
    glm::vec3 computeSceneFitPosition(float fovDegrees) const;

    void prepare();
    void loadSettings();
    void loadAnimSettings();
    void checkLoadingComplete();
    void run();
    void runReferenceCapture();
    void runBenchmark();
    void playAnimations(float deltaTime);

    // --- UI drawing ---
    void drawUI();

    // Panel draw methods (defined in panels/*.cpp)
    void drawViewportPanel();
    void drawRenderSettingsPanel();
    void drawAnimationPanel();
    void drawPropertyPanel(uint32_t lightId);

    // Screenshot
    void saveScreenshot(Buffer* buf, const std::string& path);

    void showGizmo(Camera& cam, float* matrix, ImGuizmo::OPERATION operation);
};

} // namespace oka
