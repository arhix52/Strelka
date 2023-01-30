#include "SimpleRenderTask.h"

#include <pxr/base/gf/gamma.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/tf/stopwatch.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hf/pluginDesc.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hio/imageRegistry.h>
#include <pxr/imaging/hio/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

#include <render/common.h>
#include <render/buffer.h>

#include <display/glfwdisplay.h>

#include <algorithm>
#include <cxxopts.hpp>
#include <iostream>
#include <filesystem>

// #include <cuda_runtime.h>

PXR_NAMESPACE_USING_DIRECTIVE

TF_DEFINE_PRIVATE_TOKENS(_AppTokens, (HdStrelkaDriver)(HdStrelkaRendererPlugin));

HdRendererPluginHandle GetHdStrelkaPlugin()
{
    HdRendererPluginRegistry& registry = HdRendererPluginRegistry::GetInstance();
    const TfToken& pluginId = _AppTokens->HdStrelkaRendererPlugin;
    HdRendererPluginHandle plugin = registry.GetOrCreateRendererPlugin(pluginId);
    return plugin;
}

HdCamera* FindCamera(UsdStageRefPtr& stage, HdRenderIndex* renderIndex, SdfPath& cameraPath)
{
    UsdPrimRange primRange = stage->TraverseAll();
    for (auto prim = primRange.cbegin(); prim != primRange.cend(); prim++)
    {
        if (!prim->IsA<UsdGeomCamera>())
        {
            continue;
        }
        cameraPath = prim->GetPath();
        break;
    }
    HdCamera* camera = (HdCamera*)dynamic_cast<HdCamera*>(renderIndex->GetSprim(HdTokens->camera, cameraPath));
    return camera;
}

std::vector<std::pair<HdCamera*, SdfPath>> FindAllCameras(UsdStageRefPtr& stage, HdRenderIndex* renderIndex)
{
    UsdPrimRange primRange = stage->TraverseAll();
    HdCamera* camera{};
    SdfPath cameraPath{};
    std::vector<std::pair<HdCamera*, SdfPath>> cameras{};
    for (auto prim = primRange.cbegin(); prim != primRange.cend(); prim++)
    {
        if (!prim->IsA<UsdGeomCamera>())
        {
            continue;
        }
        cameraPath = prim->GetPath();
        camera = (HdCamera*)dynamic_cast<HdCamera*>(renderIndex->GetSprim(HdTokens->camera, cameraPath));

        cameras.emplace_back(std::make_pair(camera, cameraPath));
    }

    return cameras;
}

class CameraController : public oka::InputHandler
{
    GfCamera mGfCam;
    GfQuatd mOrientation;
    GfVec3d mPosition;

    float rotationSpeed = 0.025f;
    float movementSpeed = 5.0f;

    double pitch = 0.0;
    double yaw = 0.0;
    double max_pitch_rate = 5;
    double max_yaw_rate = 5;

public:
    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        bool forward = false;
        bool back = false;
    } keys;
    struct MouseButtons
    {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    GfVec2d mMousePos;

public:
    GfVec3d getFront()
    {
        return mOrientation.Transform(GfVec3d(0.0, 0.0, -1.0));
    }
    GfVec3d getUp()
    {
        return mOrientation.Transform(GfVec3d(0.0, 1.0, 0.0));
    }
    GfVec3d getRight()
    {
        return mOrientation.Transform(GfVec3d(1.0, 0.0, 0.0));
    }
    bool moving()
    {
        return keys.left || keys.right || keys.up || keys.down || keys.forward || keys.back || mouseButtons.right ||
               mouseButtons.left || mouseButtons.middle;
    }
    void update(double deltaTime)
    {
        if (moving())
        {
            const float moveSpeed = deltaTime * movementSpeed;
            if (keys.up)
                mPosition += getUp() * moveSpeed;
            if (keys.down)
                mPosition -= getUp() * moveSpeed;
            if (keys.left)
                mPosition -= getRight() * moveSpeed;
            if (keys.right)
                mPosition += getRight() * moveSpeed;
            if (keys.forward)
                mPosition += getFront() * moveSpeed;
            if (keys.back)
                mPosition -= getFront() * moveSpeed;
            updateViewMatrix();
        }
    }

    void ChangePitch(float degrees)
    {
        // Check bounds with the max pitch rate so that we aren't moving too fast
        if (degrees < -max_pitch_rate)
        {
            degrees = -max_pitch_rate;
        }
        else if (degrees > max_pitch_rate)
        {
            degrees = max_pitch_rate;
        }
        pitch += degrees;

        // Check bounds for the camera pitch
        if (pitch > 360.0f)
        {
            pitch -= 360.0f;
        }
        else if (pitch < -360.0f)
        {
            pitch += 360.0f;
        }
    }
    void ChangeHeading(float degrees)
    {
        // Check bounds with the max heading rate so that we aren't moving too fast
        if (degrees < -max_yaw_rate)
        {
            degrees = -max_yaw_rate;
        }
        else if (degrees > max_yaw_rate)
        {
            degrees = max_yaw_rate;
        }
        // This controls how the heading is changed if the camera is pointed straight up or down
        // The heading delta direction changes
        if (pitch > 90 && pitch < 270 || (pitch < -90 && pitch > -270))
        {
            yaw -= degrees;
        }
        else
        {
            yaw += degrees;
        }
        // Check bounds for the camera heading
        if (yaw > 360.0f)
        {
            yaw -= 360.0f;
        }
        else if (yaw < -360.0f)
        {
            yaw += 360.0f;
        }
    }

    void rotate(double rightAngle, double upAngle)
    {
        // GfRotation a(GfVec3d(1.0, 0.0, 0.0), upAngle * rotationSpeed);
        // GfRotation a(GfVec3d(1.0, 0.0, 0.0), pitch);
        GfRotation a(getRight(), upAngle * rotationSpeed);
        // GfRotation b(GfVec3d(0.0, 0.0, 1.0), rightAngle * rotationSpeed);
        GfRotation b(getUp(), rightAngle * rotationSpeed);
        // GfRotation b(GfVec3d(0.0, 1.0, 0.0), yaw);

        // GfRotation b(getUp(), rightAngle * rotationSpeed);

        //GfRotation c = a * b;
        mOrientation = a.GetQuat() * mOrientation * b.GetQuat();
        updateViewMatrix();
        yaw *= .5;
        pitch *= .5;
    }

    void translate(GfVec3d delta)
    {
        mPosition += mOrientation.Transform(delta);
        updateViewMatrix();
    }

    void updateViewMatrix()
    {
        GfMatrix4d view(1.0);
        view.SetRotateOnly(mOrientation);
        view.SetTranslateOnly(mPosition);

        mGfCam.SetTransform(view);
    }

    GfCamera& getCamera()
    {
        return mGfCam;
    }

    CameraController(UsdGeomCamera& cam)
    {
        mGfCam = cam.GetCamera(0.0);
        GfMatrix4d xform = mGfCam.GetTransform();
        xform.Orthonormalize();
        mOrientation = xform.ExtractRotationQuat();
        mOrientation.Normalize();
        mPosition = xform.ExtractTranslation();
    }

    void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
    {
        const bool keyState = ((GLFW_REPEAT == action) || (GLFW_PRESS == action)) ? true : false;
        switch (key)
        {
        case GLFW_KEY_W: {
            keys.forward = keyState;
            break;
        }
        case GLFW_KEY_S: {
            keys.back = keyState;
            break;
        }
        case GLFW_KEY_A: {
            keys.left = keyState;
            break;
        }
        case GLFW_KEY_D: {
            keys.right = keyState;
            break;
        }
        case GLFW_KEY_Q: {
            keys.up = keyState;
            break;
        }
        case GLFW_KEY_E: {
            keys.down = keyState;
            break;
        }
        default:
            break;
        }
    }

    void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods)
    {
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_PRESS)
            {
                mouseButtons.right = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mouseButtons.right = false;
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                mouseButtons.left = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mouseButtons.left = false;
            }
        }
    }

    void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos)
    {
        const float dx = mMousePos[0] - xpos;
        const float dy = mMousePos[1] - ypos;

        // ImGuiIO& io = ImGui::GetIO();
        // bool handled = io.WantCaptureMouse;
        // if (handled)
        //{
        //    camera.mousePos = glm::vec2((float)xpos, (float)ypos);
        //    return;
        //}

        if (mouseButtons.right)
        {
            // rotate(dx, dy);
            ChangeHeading(rotationSpeed * dx);
            ChangePitch(rotationSpeed * dy);
            rotate(dx, dy);
        }
        if (mouseButtons.left)
        {
            translate(GfVec3d(-0.0, 0.0, -dy * .005 * movementSpeed));
        }
        if (mouseButtons.middle)
        {
            translate(GfVec3d(-dx * 0.01, -dy * 0.01, 0.0f));
        }
        mMousePos[0] = xpos;
        mMousePos[1] = ypos;
    }
};

class RenderSurfaceController : public oka::ResizeHandler
{
    uint32_t imageWidth = 800;
    uint32_t imageHeight = 600;
    oka::SettingsManager* mSettingsManager = nullptr;
    std::array<HdRenderBuffer*, 3> mRenderBuffers;
    bool mDirty[3] = { true };
    bool mInUse[3] = { false };

public:
    RenderSurfaceController(oka::SettingsManager* settingsManager, std::array<HdRenderBuffer*, 3>& renderBuffers)
        : mSettingsManager(settingsManager), mRenderBuffers(renderBuffers)
    {
        imageWidth = mSettingsManager->getAs<uint32_t>("render/width");
        imageHeight = mSettingsManager->getAs<uint32_t>("render/height");
    }

    void framebufferResize(int newWidth, int newHeight)
    {
        assert(mSettingsManager);
        mSettingsManager->setAs<uint32_t>("render/width", newWidth);
        mSettingsManager->setAs<uint32_t>("render/height", newHeight);
        imageWidth = newWidth;
        imageHeight = newHeight;
        for (int i = 0; i < 3; ++i)
        {
            mDirty[i] = true;
        }
        // reallocateBuffers();
    }

    bool isDirty(uint32_t index)
    {
        return mDirty[index];
    }

    void acquire(uint32_t index)
    {
        mInUse[index] = true;
    }

    void release(uint32_t index)
    {
        mInUse[index] = false;
    }

    HdRenderBuffer* getRenderBuffer(uint32_t index)
    {
        assert(index < 3);
        if (mDirty[index] && !mInUse[index])
        {
            mRenderBuffers[index]->Allocate(GfVec3i(imageWidth, imageHeight, 1), HdFormatFloat32Vec4, false);
            mDirty[index] = false;
        }
        return mRenderBuffers[index];
    }
};

void setDefaultCamera(UsdGeomCamera& cam)
{
    // y - up, camera looks at (0, 0, 0)
    std::vector<float> r0 = { 0, -1, 0, 0 };
    std::vector<float> r1 = { 0, 0, 1, 0 };
    std::vector<float> r2 = { -1, 0, 0, 0 };
    std::vector<float> r3 = { 0, 0, 0, 1 };

    GfMatrix4d xform(r0, r1, r2, r3);
    GfCamera mGfCam{};

    mGfCam.SetTransform(xform);

    GfRange1f clippingRange = GfRange1f{ 0.1, 1000 };
    mGfCam.SetClippingRange(clippingRange);
    mGfCam.SetVerticalAperture(20.25);
    mGfCam.SetVerticalApertureOffset(0);
    mGfCam.SetHorizontalAperture(36);
    mGfCam.SetHorizontalApertureOffset(0);
    mGfCam.SetFocalLength(50);

    GfCamera::Projection projection = GfCamera::Projection::Perspective;
    mGfCam.SetProjection(projection);

    cam.SetFromCamera(mGfCam, 0.0);
}

bool saveScreenshot(std::string& outputFilePath, float* mappedMem, uint32_t imageWidth, uint32_t imageHeight)
{
    TF_VERIFY(mappedMem != nullptr);

    int pixelCount = imageWidth * imageHeight;

    for (int i = 0; i < pixelCount; i++)
    {
        mappedMem[i * 4 + 0] = GfConvertLinearToDisplay(mappedMem[i * 4 + 0]);
        mappedMem[i * 4 + 1] = GfConvertLinearToDisplay(mappedMem[i * 4 + 1]);
        mappedMem[i * 4 + 2] = GfConvertLinearToDisplay(mappedMem[i * 4 + 2]);
    }

    // Write image to file.
    TfStopwatch timerWrite;
    timerWrite.Start();

    HioImageSharedPtr image = HioImage::OpenForWriting(outputFilePath);

    if (!image)
    {
        fprintf(stderr, "Unable to open output file for writing!\n");
        return false;
    }

    HioImage::StorageSpec storage;
    storage.width = (int)imageWidth;
    storage.height = (int)imageHeight;
    storage.depth = (int)1;
    storage.format = HioFormat::HioFormatFloat32Vec4;
    storage.flipped = false;
    storage.data = mappedMem;

    VtDictionary metadata;
    image->Write(storage, metadata);

    timerWrite.Stop();

    printf("Wrote image (%.3fs)\n", timerWrite.GetSeconds());
    fflush(stdout);

    return true;
}

int main(int argc, const char* argv[])
{
    // config. options
    cxxopts::Options options("Strelka -s <USD Scene path>", "commands");

    options.add_options()("s, scene", "scene path", cxxopts::value<std::string>()->default_value(""))(
        "i, iteration", "Iteration to capture", cxxopts::value<int32_t>()->default_value("-1"))("h, help", "Print usage");

    options.parse_positional({ "s" });
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // check params
    std::string usdFile(result["s"].as<std::string>());
    if (!std::filesystem::exists(usdFile) && !usdFile.empty())
    {
        std::cerr << "usd file doesn't exist";
        exit(0);
    }
    int32_t iterationToCapture(result["i"].as<int32_t>());
    // Init plugin.
    HdRendererPluginHandle pluginHandle = GetHdStrelkaPlugin();

    if (!pluginHandle)
    {
        fprintf(stderr, "HdStrelka plugin not found!\n");
        return EXIT_FAILURE;
    }

    if (!pluginHandle->IsSupported())
    {
        fprintf(stderr, "HdStrelka plugin is not supported!\n");
        return EXIT_FAILURE;
    }

    HdDriverVector drivers;
    // Set up rendering context.
    uint32_t imageWidth = 1024;
    uint32_t imageHeight = 768;

    oka::SharedContext* ctx = new oka::SharedContext(); // &display.getSharedContext();

    ctx->mSettingsManager = new oka::SettingsManager();

    ctx->mSettingsManager->setAs<uint32_t>("render/width", imageWidth);
    ctx->mSettingsManager->setAs<uint32_t>("render/height", imageHeight);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/depth", 6);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/spp", 1);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/iteration", 0);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0); // 0 - none, 1 - random, 2 -
                                                                                   // stratified sampling, 3 - optimized
                                                                                   // stratified sampling
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0); // 0 - reinhard, 1 - aces, 2 - filmic
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    ctx->mSettingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableUpscale", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableAcc", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableTonemap", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", false);

    oka::glfwdisplay display;
    display.init(imageWidth, imageHeight, ctx);

    HdDriver driver;
    driver.name = _AppTokens->HdStrelkaDriver;
    driver.driver = VtValue(ctx);

    drivers.push_back(&driver);

    HdRenderDelegate* renderDelegate = pluginHandle->CreateRenderDelegate();
    TF_VERIFY(renderDelegate);
    renderDelegate->SetDrivers(drivers);

    // Handle cmdline args.
    // Load scene.
    TfStopwatch timerLoad;
    timerLoad.Start();

    // ArGetResolver().ConfigureResolverForAsset(settings.sceneFilePath);
    std::string usdPath = usdFile;

    UsdStageRefPtr stage = UsdStage::Open(usdPath.c_str());

    timerLoad.Stop();

    if (!stage)
    {
        fprintf(stderr, "Unable to open USD stage file.\n");
        return EXIT_FAILURE;
    }

    printf("USD scene loaded (%.3fs)\n", timerLoad.GetSeconds());
    fflush(stdout);

    // Print the up-axis
    std::cout << "Stage up-axis: " << UsdGeomGetStageUpAxis(stage) << std::endl;

    // Print the stage's linear units, or "meters per unit"
    std::cout << "Meters per unit: " << UsdGeomGetStageMetersPerUnit(stage) << std::endl;

    HdRenderIndex* renderIndex = HdRenderIndex::New(renderDelegate, HdDriverVector());
    TF_VERIFY(renderIndex);

    UsdImagingDelegate sceneDelegate(renderIndex, SdfPath::AbsoluteRootPath());
    sceneDelegate.Populate(stage->GetPseudoRoot());
    sceneDelegate.SetTime(0);
    sceneDelegate.SetRefineLevelFallback(4);

    double meterPerUnit = UsdGeomGetStageMetersPerUnit(stage);

    // Init default camera
    SdfPath cameraPath = SdfPath("/defaultCamera");
    UsdGeomCamera cam = UsdGeomCamera::Define(stage, cameraPath);
    setDefaultCamera(cam);

    // Init camera from scene
    cameraPath = SdfPath::EmptyPath();
    HdCamera* camera = FindCamera(stage, renderIndex, cameraPath);
    cam = UsdGeomCamera::Get(stage, cameraPath);
    CameraController cameraController(cam);

    // std::vector<std::pair<HdCamera*, SdfPath>> cameras = FindAllCameras(stage, renderIndex);

    std::array<HdRenderBuffer*, 3> renderBuffers;
    for (int i = 0; i < 3; ++i)
    {
        renderBuffers[i] = (HdRenderBuffer*)renderDelegate->CreateFallbackBprim(HdPrimTypeTokens->renderBuffer);
        renderBuffers[i]->Allocate(GfVec3i(imageWidth, imageHeight, 1), HdFormatFloat32Vec4, false);
    }

    CameraUtilFraming framing;
    framing.dataWindow = GfRect2i(GfVec2i(0, 0), GfVec2i(imageWidth, imageHeight));
    framing.displayWindow = GfRange2f(GfVec2f(0.0f, 0.0f), GfVec2f((float)imageWidth, (float)imageHeight));
    framing.pixelAspectRatio = 1.0f;

    std::pair<bool, CameraUtilConformWindowPolicy> overrideWindowPolicy(false, CameraUtilFit);

    TfTokenVector renderTags(1, HdRenderTagTokens->geometry);
    HdRprimCollection renderCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->refined));
    HdRenderPassSharedPtr renderPass = renderDelegate->CreateRenderPass(renderIndex, renderCollection);

    std::shared_ptr<HdRenderPassState> renderPassState[3];
    std::shared_ptr<SimpleRenderTask> renderTasks[3];
    for (int i = 0; i < 3; ++i)
    {
        renderPassState[i] = std::make_shared<HdRenderPassState>();
        renderPassState[i]->SetCameraAndFraming(camera, framing, overrideWindowPolicy);
        HdRenderPassAovBindingVector aovBindings(1);
        aovBindings[0].aovName = HdAovTokens->color;
        aovBindings[0].renderBuffer = renderBuffers[i];

        renderPassState[i]->SetAovBindings(aovBindings);
        renderTasks[i] = std::make_shared<SimpleRenderTask>(renderPass, renderPassState[i], renderTags);
    }

    // Perform rendering.
    TfStopwatch timerRender;
    timerRender.Start();

    HdEngine engine;

    display.setInputHandler(&cameraController);

    RenderSurfaceController surfaceController(ctx->mSettingsManager, renderBuffers);
    display.setResizeHandler(&surfaceController);

    uint64_t frameCount = 0;

    bool needCopyBuffer = false;
    int32_t waitFramesForScreenshot = -1;

    while (!display.windowShouldClose())
    {
        auto start = std::chrono::high_resolution_clock::now();
        HdTaskSharedPtrVector tasks;
        const uint32_t versionId = frameCount % oka::MAX_FRAMES_IN_FLIGHT;
        // relocation?
        surfaceController.acquire(versionId);
        if (surfaceController.isDirty(versionId))
        {
            HdRenderPassAovBindingVector aovBindings(1);
            aovBindings[0].aovName = HdAovTokens->color;
            surfaceController.release(versionId);
            aovBindings[0].renderBuffer = surfaceController.getRenderBuffer(versionId);
            surfaceController.acquire(versionId);
            renderPassState[versionId]->SetAovBindings(aovBindings);
            renderTasks[versionId] =
                std::make_shared<SimpleRenderTask>(renderPass, renderPassState[versionId], renderTags);
        }
        tasks.push_back(renderTasks[versionId]);
        sceneDelegate.SetTime(1.0f);

        display.pollEvents();

        static auto prevTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double, std::milli>(currentTime - prevTime).count() / 1000.0;
        cameraController.update(deltaTime);
        prevTime = currentTime;

        cam.SetFromCamera(cameraController.getCamera(), 0.0);

        uint32_t iteration = ctx->mSettingsManager->getAs<uint32_t>("render/pt/iteration");
        if (iteration == iterationToCapture)
        {
            ctx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", true);
        }
        bool needScreenshot = ctx->mSettingsManager->getAs<bool>("render/pt/needScreenshot");

        //         if (needScreenshot)
        //         {
        //             if (waitFramesForScreenshot == -1)
        //             {
        //                 waitFramesForScreenshot = oka::MAX_FRAMES_IN_FLIGHT;
        //                 needCopyBuffer = true;
        //             }
        //             else if (waitFramesForScreenshot > 0)
        //             {
        //                 --waitFramesForScreenshot;
        //             }
        //             else if (waitFramesForScreenshot == 0)
        //             {
        // #ifdef WINDOWS
        //                 std::size_t foundSlash = usdPath.find_last_of("/\\");
        // #else
        //                 std::size_t foundSlash = usdPath.find_last_of("/");
        // #endif
        //                 std::size_t foundDot = usdPath.find_last_of(".");
        //                 std::string fileName = usdPath.substr(0, foundDot);
        //                 fileName = fileName.substr(foundSlash + 1);

        //                 std::string outputFilePath =
        //                     fileName + "_" + std::to_string(iteration - oka::MAX_FRAMES_IN_FLIGHT - 1) + "i_" +
        //                     std::to_string(ctx->mSettingsManager->getAs<uint32_t>("render/pt/depth")) + "d_" +
        //                     std::to_string(ctx->mSettingsManager->getAs<uint32_t>("render/pt/spp")) + "spp" + ".png";
        //                 float* mappedMem = (float*)ctx->mResManager->getMappedMemory(screenshotTransferBuffer);

        //                 if (saveScreenshot(outputFilePath, mappedMem, imageWidth, imageHeight))
        //                 {
        //                     waitFramesForScreenshot = -1;
        //                     ctx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", false);
        //                 }
        //             }
        //         }

        display.onBeginFrame();
        engine.Execute(renderIndex, &tasks); // main path tracing rendering in fixed render resolution
        oka::Buffer* outputBuffer = surfaceController.getRenderBuffer(versionId)->GetResource(false).UncheckedGet<oka::Buffer*>();
        oka::ImageBuffer outputImage;
        // upload to host
        outputBuffer->map();
        outputImage.data = outputBuffer->getHostPointer();
        outputImage.dataSize = outputBuffer->getHostDataSize();
        outputImage.height = outputBuffer->height();
        outputImage.width = outputBuffer->width();
        outputImage.pixel_format = outputBuffer->getFormat();

        display.drawFrame(outputImage); // blit rendered image to swapchain
        display.drawUI(); // render ui to swapchain image in window resolution
        display.onEndFrame(); // submit command buffer and present

        auto finish = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration<double, std::milli>(finish - start).count();

        surfaceController.release(versionId);

        display.setWindowTitle((std::string("Strelka") + " [" + std::to_string(frameTime) + " ms]" + " [" +
                                std::to_string(iteration) + " iteration]")
                                   .c_str());
        ++frameCount;
    }

    // renderBuffer->Resolve();
    // TF_VERIFY(renderBuffer->IsConverged());

    timerRender.Stop();

    display.destroy();

    printf("Rendering finished (%.3fs)\n", timerRender.GetSeconds());
    fflush(stdout);

    for (int i = 0; i < 3; ++i)
    {
        renderDelegate->DestroyBprim(renderBuffers[i]);
    }
    return EXIT_SUCCESS;
}
