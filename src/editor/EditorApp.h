#pragma once

#include <strelka/display/display.h>

#include <loadprogress.h>
#include <strelka/render/render.h>

#include "CameraController.h"
#include "editor_viewport_layout.h"

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <memory>
#include <future>
#include <string>
#include <vector>

#include <strelka/sceneloader/gltfloader.h>

#include "imgui.h"
#include "ImGuizmo.h"

namespace oka
{

class EditorApp : public ResizeHandler
{
private:
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
    uint32_t m_selectedNodeId = kInvalidIndex;
    uint32_t m_selectedInstanceId = kInvalidIndex;
    uint32_t m_selectedLightId = kInvalidIndex;
    uint32_t m_selectedMaterialId = kInvalidIndex;

    ImGuizmo::OPERATION m_gizmoOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE m_gizmoMode = ImGuizmo::LOCAL;

    // Panel visibility, toggled from the Window menu.
    bool m_showOutliner = true;
    bool m_showProperties = true;
    bool m_showMaterials = true;
    bool m_showMemory = true;
    bool m_outlinerScrollToSelection = false;
    bool m_layoutRebuildPending = false;

    // Screen rect of the rendered image inside the Viewport panel, refreshed
    // every frame. Picking and the gizmo both need it, and the gizmo is drawn
    // after the panel has already been closed.
    ImVec2 m_viewportRectMin{ 0, 0 };
    ImVec2 m_viewportRectMax{ 0, 0 };
    editor_viewport::Layout m_viewportLayout;
    editor_viewport::PresentationMode m_viewportPresentation = editor_viewport::PresentationMode::Fit;
    uint32_t mPresentedPreviewWidth = 0;
    uint32_t mPresentedPreviewHeight = 0;
    /// Which entry of the *current backend's* denoiser list is selected. The list
    /// is not the same on both backends, so this is only meaningful next to the
    /// Ui it was resolved against; clearing the flag makes the panel re-derive it
    /// from settings, which is what everything outside the panel does after
    /// writing those settings itself.
    int mDenoiseModeIndex = 0;
    bool mDenoiseModeInitialized = false;

    bool m_documentDirty = false;
    // Snapshot taken at beginSceneLoad so a failed/cancelled open can restore
    // the previous document instead of leaving Save pointed at a never-loaded path.
    std::string m_sceneFileBeforeLoad;
    bool m_documentDirtyBeforeLoad = false;
    std::string m_attemptedSceneFile;
    std::chrono::steady_clock::time_point m_loadStartedAt{};

    // File → Open Recent. Persisted next to imgui.ini so a rebuild does not
    // wipe the working set, and capped by editor_document::kRecentScenesCapacity.
    std::vector<std::string> m_recentScenes;
    void rememberRecentScene(const std::string& sceneFile);
    void persistRecentScenes();

    // One-shot ImGui modal for open/save/device failures (no toast system).
    std::string m_alertMessage;
    bool m_alertOpen = false;
    bool m_alertOffersRendererRestart = false;
    bool m_rendererRestartRequested = false;
    bool m_frameBudgetConfirmOpen = false;
    uint32_t m_pendingPreviewWidth = 0;
    uint32_t m_pendingPreviewHeight = 0;
    double m_pendingPredictedGpuMs = 0.0;
    float m_pendingRecommendedScale = 0.5f;

    // After Render::deviceError(), stop submitting and show the alert once.
    bool m_deviceErrorLatched = false;
    bool m_renderSubmissionsBlocked = false;

    // No exposure in the light sidecar: measure it from the first frame instead.
    bool m_autoExposurePending = false;
    /// Frames displayed since the scene finished building. Independent of the
    /// accumulator, which resets on camera moves and is off entirely in some
    /// configurations.
    uint32_t m_framesSinceSceneReady = 0;
    size_t m_lastExposureFrameSeen = (size_t)-1;
    void applyAutoExposure(oka::Buffer* buf);
    bool m_pendingSaveAs = false;

    struct UndoState
    {
        enum class Kind
        {
            Light,
            Node,
            Material
        } kind = Kind::Light;
        uint32_t id = 0;
        Scene::UniformLightDesc light{};
        glm::float3 translation{ 0 };
        glm::quat rotation{ 1, 0, 0, 0 };
        glm::float3 scale{ 1 };
        Scene::MaterialDescription material{};
    };
    std::vector<UndoState> m_undoStack;
    std::vector<UndoState> m_redoStack;
    std::vector<UndoState> m_undoStackBeforeLoad;
    std::vector<UndoState> m_redoStackBeforeLoad;

    std::future<std::unique_ptr<Scene>> m_loadingFuture;
    /// Owned by the app rather than by a load, so both the worker and the
    /// renderer can hold a pointer to it for as long as either exists.
    LoadProgress m_loadProgress;
    std::string m_pendingResourcePath;
    bool m_isLoading = false;

    std::string m_sceneFile;
    std::string m_resourceSearchPath;

    std::string m_pendingScreenshotPath;
    // Only reaches an EXR: a PNG is display-referred whatever this says,
    // because 8 bits have nowhere to put the rest. See
    // editor_screenshot::sourceForExtension.
    bool m_screenshotDisplayReferred = false;
    // --need_screenshot: armed before run(), disarmed by the shot it fires so
    // a scene that keeps converging does not write a file per frame.
    bool m_batchScreenshotArmed = false;
    uint32_t m_batchSppTotal = 0;
    bool m_batchScreenshotPending = false;

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
    /// World AABB of the current selection (node, instance, or light). False when
    /// nothing selectable is selected, or the selection has no geometry to frame.
    bool computeSelectionWorldBounds(glm::float3& outMin, glm::float3& outMax);
    /// Dolly / resize the active camera so the selection fills the frame.
    void frameSelectionInView();
    void drawSelectionGizmo(Camera& cam);
    /// Print the viewport as a StrelkaCLI config, so an angle that shows a
    /// defect can be handed to someone else and re-rendered rather than
    /// described. See camera_dump.h.
    void dumpCameraSettings();
    void drawLoadingOverlay();
    void showAlert(const std::string& message);
    void drawAlertModal();
    void drawFrameBudgetModal();
    void ensureValidCameraSelection();
    void handleDeviceError();
    void initializeRendererForCurrentScene();
    void restartRendererAtSafeScale();
    void restoreDocumentAfterFailedLoad(const char* reason);
    void applyPreviewResolution(uint32_t width, uint32_t height);
    void requestPreviewResolution(uint32_t width, uint32_t height);

public:
    EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath);

    /// Render `sppTotal` samples, write one screenshot, close.
    ///
    /// This is what --need_screenshot is for. Call before run(); with
    /// `screenshotOnComplete` false it only sets the sample counts, which is
    /// what --spp_total and --spp_subframe do on their own.
    void setBatchCapture(uint32_t sppTotal, uint32_t sppSubframe, bool screenshotOnComplete);
    /// Cancels a load in flight and waits for it.
    ///
    /// std::future's destructor blocks until the task finishes, so without the
    /// cancel first, closing the window during a load hangs the app for the rest
    /// of that load. Waiting here rather than letting the member destructor do it
    /// also keeps the worker's raw pointer to m_sceneLoader valid for as long as
    /// the worker can still use it.
    ~EditorApp() override;

    void framebufferResize(int newWidth, int newHeight) override;

    /// Compute camera position that fits the entire scene in the view frustum.
    glm::vec3 computeSceneFitPosition(float fovDegrees) const;

    void loadSettings();
    /// Start loading a scene on a worker. Returns immediately; the main loop
    /// picks the result up in checkLoadingComplete(). Used both for the scene
    /// named on the command line and for File -> Open, so startup and reload
    /// cannot drift apart.
    void beginSceneLoad(const std::string& sceneFile, const std::string& resourceSearchPath);
    /// Take the exposure the freshly loaded scene asks for, or arrange to measure
    /// it. Must run after loadSettings(), which writes the photographic defaults.
    void applySceneExposure();
    void loadAnimSettings();
    void checkLoadingComplete();
    void waitForSceneLoad();
    void run();
    void runReferenceCapture();
    void runConvergenceSweep();
    void runBenchmark();
    void runJitterTest();
    void runDenoiseAudit();
    void runPauseBlurCheck();
    void runLightAudit();
    void playAnimations(float deltaTime);

    // --- UI drawing ---
    void drawUI();
    void buildDefaultDockLayout(ImGuiID dockspaceId);

    // Panel draw methods (defined in panels/*.cpp)
    void drawViewportPanel();
    void drawRenderSettingsPanel();
    /// The gamepad block inside the render settings panel. Split out because the
    /// panel is already 1500 lines and this is self-contained.
    void drawGamepadSettings();
    void drawMemoryPanel();
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
