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
#include <future>
#include <string>
#include <vector>
#include <variant>

#include <strelka/sceneloader/gltfloader.h>

#include "imgui.h"
#include "imgui_internal.h" // DockBuilder / window settings lookup
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

    // Selection (invalid id == -1)
    uint32_t m_selectedNodeId = (uint32_t)-1;
    uint32_t m_selectedInstanceId = (uint32_t)-1;
    uint32_t m_selectedLightId = (uint32_t)-1;
    uint32_t m_selectedMaterialId = (uint32_t)-1;

    ImGuizmo::OPERATION m_gizmoOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE m_gizmoMode = ImGuizmo::LOCAL;

    // Panel visibility, toggled from the Window menu.
    bool m_showOutliner = true;
    bool m_showProperties = true;
    bool m_showMaterials = true;
    bool m_outlinerScrollToSelection = false;
    bool m_layoutRebuildPending = false;

    // Screen rect of the rendered image inside the Viewport panel, refreshed
    // every frame. Picking and the gizmo both need it, and the gizmo is drawn
    // after the panel has already been closed.
    ImVec2 m_viewportRectMin{ 0, 0 };
    ImVec2 m_viewportRectMax{ 0, 0 };

    bool m_documentDirty = false;
    // No exposure in the light sidecar: measure it from the first frame instead.
    bool m_autoExposurePending = false;
    void applyAutoExposure(oka::Buffer* buf);
    bool m_pendingSaveAs = false;

    struct UndoState
    {
        enum class Kind
        {
            Light,
            Node,
            Material
        } kind;
        uint32_t id = 0;
        Scene::UniformLightDesc light{};
        glm::float3 translation{ 0 };
        glm::quat rotation{ 1, 0, 0, 0 };
        glm::float3 scale{ 1 };
        Scene::MaterialDescription material{};
    };
    std::vector<UndoState> m_undoStack;
    std::vector<UndoState> m_redoStack;

    std::future<std::unique_ptr<Scene>> m_loadingFuture;
    std::string m_pendingResourcePath;
    bool m_isLoading = false;

    std::string m_sceneFile;
    std::string m_resourceSearchPath;

    std::string m_pendingScreenshotPath;

    // Throttles for AppKit round-trips that do not need per-frame accuracy.
    std::chrono::high_resolution_clock::time_point m_lastTitleUpdate{};
    std::chrono::high_resolution_clock::time_point m_lastEdrQuery{};

    void setCameraDetached(bool detached);
    void clearSelection();
    void markDocumentDirty();
    void pushUndoLight(uint32_t lightId);
    void pushUndoNode(uint32_t nodeId);
    void pushUndoMaterial(uint32_t materialId);
    void undo();
    void redo();
    bool saveDocument(bool saveAs);
    void applySelectionFromPick(const Scene::PickHit& hit);
    Scene::PickHit pickAtScreenPos(const ImVec2& screenPos);
    void drawSelectionOverlay(Camera& cam);
    void drawBoundsWireframe(const glm::float3& bbMin,
                             const glm::float3& bbMax,
                             const glm::mat4& worldFromLocal,
                             Camera& cam);
    bool computeNodeBounds(const Scene::Node& node,
                           glm::float3& outMin,
                           glm::float3& outMax,
                           glm::mat4& outWorldFromLocal);
    void drawSelectionGizmo(Camera& cam);

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
    void runConvergenceSweep();
    void runBenchmark();
    void runJitterTest();
    void runDenoiseAudit();
    void runLightAudit();
    void playAnimations(float deltaTime);

    // --- UI drawing ---
    void drawUI();
    void buildDefaultDockLayout(ImGuiID dockspaceId);

    // Panel draw methods (defined in panels/*.cpp)
    void drawViewportPanel();
    void drawRenderSettingsPanel();
    void drawAnimationPanel();
    void drawPropertyPanel();
    void drawOutlinerPanel();
    void drawNodeRecursive(int nodeId, const ImGuiTextFilter& filter);
    void drawMaterialPanel();

    // Screenshot
    void saveScreenshot(Buffer* buf, const std::string& path);

    void showGizmo(Camera& cam, float* matrix, ImGuizmo::OPERATION operation);
};

} // namespace oka
