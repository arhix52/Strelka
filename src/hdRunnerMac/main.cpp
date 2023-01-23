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

#include <cassert>
#include <vector>
#include <fstream>
#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>

#include "ShaderTypes.h"

#include <render.h>

#include <cxxopts.hpp>

#include <simd/simd.h>

static constexpr size_t kMaxFramesInFlight = 3;

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

#pragma region Declarations {

namespace math
{
constexpr simd::float3 add(const simd::float3& a, const simd::float3& b);
constexpr simd_float4x4 makeIdentity();
simd::float4x4 makePerspective();
simd::float4x4 makeXRotate(float angleRadians);
simd::float4x4 makeYRotate(float angleRadians);
simd::float4x4 makeZRotate(float angleRadians);
simd::float4x4 makeTranslate(const simd::float3& v);
simd::float4x4 makeScale(const simd::float3& v);
simd::float3x3 discardTranslation(const simd::float4x4& m);
} // namespace math

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}
class MetalDisplay
{
public:
    MetalDisplay(MTL::Device* pDevice)
    {
        _pDevice = pDevice;
        _pCommandQueue = _pDevice->newCommandQueue();
        _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
        buildShaders();
    }

    MTL::Texture* buildTexture(uint32_t width, uint32_t heigth)
    {
        MTL::TextureDescriptor* pTextureDesc = MTL::TextureDescriptor::alloc()->init();
        pTextureDesc->setWidth(width);
        pTextureDesc->setHeight(heigth);
        pTextureDesc->setPixelFormat(MTL::PixelFormatRGBA32Float);
        pTextureDesc->setTextureType(MTL::TextureType2D);
        pTextureDesc->setStorageMode(MTL::StorageModeManaged);
        pTextureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

        MTL::Texture* pTexture = _pDevice->newTexture(pTextureDesc);

        pTextureDesc->release();

        return pTexture;
    }

    void display(MTK::View* pView, oka::Buffer* result)
    {
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

        bool needRecreate = (result->height() != height || result->width() != width);
        if (needRecreate)
        {
            width = result->width();
            height = result->height();
            mTexture = buildTexture(width, height);
        }
        {
            MTL::Region region = MTL::Region::Make2D(0, 0, width, height);
            mTexture->replaceRegion(region, 0, result->map(), width * result->getElementSize());
        }

        MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
        dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
        MetalDisplay* pRenderer = this;
        pCmd->addCompletedHandler(^void(MTL::CommandBuffer* pCmd) {
          dispatch_semaphore_signal(pRenderer->_semaphore);
        });
        MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
        MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder(pRpd);

        pEnc->setRenderPipelineState(_pPSO);
        pEnc->setFragmentTexture(mTexture, /* index */ 0);
        pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, 0ul, 6ul);

        pEnc->endEncoding();
        pCmd->presentDrawable(pView->currentDrawable());
        pCmd->commit();

        pPool->release();
    }

    void buildShaders()
    {
        using NS::StringEncoding::UTF8StringEncoding;

        std::string shaderSrc;
        readSourceFile(shaderSrc, "./shaders/fullScreen.metal");

        NS::Error* pError = nullptr;
        MTL::Library* pLibrary =
            _pDevice->newLibrary(NS::String::string(shaderSrc.c_str(), UTF8StringEncoding), nullptr, &pError);
        if (!pLibrary)
        {
            __builtin_printf("%s", pError->localizedDescription()->utf8String());
            assert(false);
        }

        MTL::Function* pVertexFn = pLibrary->newFunction(NS::String::string("copyVertex", UTF8StringEncoding));
        MTL::Function* pFragFn = pLibrary->newFunction(NS::String::string("copyFragment", UTF8StringEncoding));

        MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
        pDesc->setVertexFunction(pVertexFn);
        pDesc->setFragmentFunction(pFragFn);
        pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        pDesc->setDepthAttachmentPixelFormat(MTL::PixelFormat::PixelFormatDepth16Unorm);

        _pPSO = _pDevice->newRenderPipelineState(pDesc, &pError);
        if (!_pPSO)
        {
            __builtin_printf("%s", pError->localizedDescription()->utf8String());
            assert(false);
        }

        pVertexFn->release();
        pFragFn->release();
        pDesc->release();
        _pShaderLibrary = pLibrary;
    }

private:
    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pShaderLibrary;
    MTL::RenderPipelineState* _pPSO;
    MTL::Texture* mTexture;
    dispatch_semaphore_t _semaphore;
    uint32_t width = 0, height = 0;
};

class MyMTKViewDelegate : public MTK::ViewDelegate
{
public:
    MyMTKViewDelegate(MTL::Device* pDevice,
                      HdEngine* eng,
                      HdRenderIndex* ri,
                      HdRenderBuffer* rb,
                      HdRenderPassSharedPtr rp,
                      HdRenderPassStateSharedPtr rps);
    virtual ~MyMTKViewDelegate() override;
    virtual void drawInMTKView(MTK::View* pView) override;
    virtual void drawableSizeWillChange(MTK::View* pView, CGSize size) override;

private:
    MTL::Device* mDevice;
    HdEngine* mHydraEngine;
    HdRenderIndex* renderIndex;
    oka::Render* mRender;
    HdRenderBuffer* renderBuffer;
    HdRenderPassSharedPtr renderPass;
    HdRenderPassStateSharedPtr mRenderPassState;
    MetalDisplay* mDisplay;
};

class MyAppDelegate : public NS::ApplicationDelegate
{
public:
    ~MyAppDelegate();

    NS::Menu* createMenuBar();

    virtual void applicationWillFinishLaunching(NS::Notification* pNotification) override;
    virtual void applicationDidFinishLaunching(NS::Notification* pNotification) override;
    virtual bool applicationShouldTerminateAfterLastWindowClosed(NS::Application* pSender) override;

    void setHydraEngine(HdEngine* eng)
    {
        mEngine = eng;
    }

    void setRenderIndex(HdRenderIndex* idx)
    {
        mRenderIndex = idx;
    }

    void setRenderPass(HdRenderPassSharedPtr rp)
    {
        mRenderPass = rp;
    }

    void setRenderBuffer(HdRenderBuffer* rb)
    {
        mRenderBuffer = rb;
    }

    void setRenderPassState(HdRenderPassStateSharedPtr rps)
    {
        mRenderPassState = rps;
    }

private:
    NS::Window* _pWindow;
    MTK::View* _pMtkView;
    MTL::Device* _pDevice;
    HdEngine* mEngine = nullptr;
    HdRenderIndex* mRenderIndex;
    HdRenderBuffer* mRenderBuffer;
    MyMTKViewDelegate* _pViewDelegate = nullptr;
    HdRenderPassSharedPtr mRenderPass;
    HdRenderPassStateSharedPtr mRenderPassState;
};

#pragma endregion Declarations }


int main(int argc, char* argv[])
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

    HdDriver driver;
    driver.name = _AppTokens->HdStrelkaDriver;
    driver.driver = VtValue(ctx);

    drivers.push_back(&driver);

    HdRenderDelegate* renderDelegate = pluginHandle->CreateRenderDelegate();
    TF_VERIFY(renderDelegate);
    renderDelegate->SetDrivers(drivers);

    std::string usdPath = usdFile;

    UsdStageRefPtr stage = UsdStage::Open(usdPath.c_str());
    if (!stage)
    {
        fprintf(stderr, "Unable to open USD stage file.\n");
        return EXIT_FAILURE;
    }
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
    // CameraController cameraController(cam);

    HdRenderBuffer* renderBuffer;
    {
        renderBuffer = (HdRenderBuffer*)renderDelegate->CreateFallbackBprim(HdPrimTypeTokens->renderBuffer);
        renderBuffer->Allocate(GfVec3i(imageWidth, imageHeight, 1), HdFormatFloat32Vec4, false);
    }

    CameraUtilFraming framing;
    framing.dataWindow = GfRect2i(GfVec2i(0, 0), GfVec2i(imageWidth, imageHeight));
    framing.displayWindow = GfRange2f(GfVec2f(0.0f, 0.0f), GfVec2f((float)imageWidth, (float)imageHeight));
    framing.pixelAspectRatio = 1.0f;

    std::pair<bool, CameraUtilConformWindowPolicy> overrideWindowPolicy(false, CameraUtilFit);

    TfTokenVector renderTags(1, HdRenderTagTokens->geometry);
    HdRprimCollection renderCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->refined));
    HdRenderPassSharedPtr renderPass = renderDelegate->CreateRenderPass(renderIndex, renderCollection);
    std::shared_ptr<HdRenderPassState> renderPassState = std::make_shared<HdRenderPassState>();
    renderPassState->SetCameraAndFraming(camera, framing, overrideWindowPolicy);
    HdRenderPassAovBindingVector aovBindings(1);
    aovBindings[0].aovName = HdAovTokens->color;
    aovBindings[0].renderBuffer = renderBuffer;
    renderPassState->SetAovBindings(aovBindings);

    HdEngine engine;

    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    MyAppDelegate del;
    del.setHydraEngine(&engine);
    del.setRenderIndex(renderIndex);
    del.setRenderPass(renderPass);
    del.setRenderBuffer(renderBuffer);
    del.setRenderPassState(renderPassState);

    NS::Application* pSharedApplication = NS::Application::sharedApplication();
    pSharedApplication->setDelegate(&del);
    pSharedApplication->run();

    pAutoreleasePool->release();

    return 0;
}


#pragma mark - AppDelegate
#pragma region AppDelegate {

MyAppDelegate::~MyAppDelegate()
{
    _pMtkView->release();
    _pWindow->release();
    _pDevice->release();
    delete _pViewDelegate;
}

NS::Menu* MyAppDelegate::createMenuBar()
{
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Menu* pMainMenu = NS::Menu::alloc()->init();
    NS::MenuItem* pAppMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pAppMenu = NS::Menu::alloc()->init(NS::String::string("Appname", UTF8StringEncoding));

    NS::String* appName = NS::RunningApplication::currentApplication()->localizedName();
    NS::String* quitItemName = NS::String::string("Quit ", UTF8StringEncoding)->stringByAppendingString(appName);
    SEL quitCb = NS::MenuItem::registerActionCallback("appQuit", [](void*, SEL, const NS::Object* pSender) {
        auto pApp = NS::Application::sharedApplication();
        pApp->terminate(pSender);
    });

    NS::MenuItem* pAppQuitItem = pAppMenu->addItem(quitItemName, quitCb, NS::String::string("q", UTF8StringEncoding));
    pAppQuitItem->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);
    pAppMenuItem->setSubmenu(pAppMenu);

    NS::MenuItem* pWindowMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pWindowMenu = NS::Menu::alloc()->init(NS::String::string("Window", UTF8StringEncoding));

    SEL closeWindowCb = NS::MenuItem::registerActionCallback("windowClose", [](void*, SEL, const NS::Object*) {
        auto pApp = NS::Application::sharedApplication();
        pApp->windows()->object<NS::Window>(0)->close();
    });
    NS::MenuItem* pCloseWindowItem = pWindowMenu->addItem(NS::String::string("Close Window", UTF8StringEncoding),
                                                          closeWindowCb, NS::String::string("w", UTF8StringEncoding));
    pCloseWindowItem->setKeyEquivalentModifierMask(NS::EventModifierFlagCommand);

    pWindowMenuItem->setSubmenu(pWindowMenu);

    pMainMenu->addItem(pAppMenuItem);
    pMainMenu->addItem(pWindowMenuItem);

    pAppMenuItem->release();
    pWindowMenuItem->release();
    pAppMenu->release();
    pWindowMenu->release();

    return pMainMenu->autorelease();
}

void MyAppDelegate::applicationWillFinishLaunching(NS::Notification* pNotification)
{
    NS::Menu* pMenu = createMenuBar();
    NS::Application* pApp = reinterpret_cast<NS::Application*>(pNotification->object());
    pApp->setMainMenu(pMenu);
    pApp->setActivationPolicy(NS::ActivationPolicy::ActivationPolicyRegular);
}

void MyAppDelegate::applicationDidFinishLaunching(NS::Notification* pNotification)
{
    CGRect frame = (CGRect){ { 100.0, 100.0 }, { 1024.0, 768.0 } };

    _pWindow = NS::Window::alloc()->init(
        frame, NS::WindowStyleMaskClosable | NS::WindowStyleMaskTitled, NS::BackingStoreBuffered, false);

    _pDevice = MTL::CreateSystemDefaultDevice();

    _pMtkView = MTK::View::alloc()->init(frame, _pDevice);
    _pMtkView->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
    _pMtkView->setClearColor(MTL::ClearColor::Make(0.1, 0.1, 0.1, 1.0));
    _pMtkView->setDepthStencilPixelFormat(MTL::PixelFormat::PixelFormatDepth16Unorm);
    _pMtkView->setClearDepth(1.0f);

    _pViewDelegate = new MyMTKViewDelegate(_pDevice, mEngine, mRenderIndex, mRenderBuffer, mRenderPass, mRenderPassState);
    _pMtkView->setDelegate(_pViewDelegate);

    _pWindow->setContentView(_pMtkView);
    _pWindow->setTitle(NS::String::string("Strelka Metal", NS::StringEncoding::UTF8StringEncoding));

    _pWindow->makeKeyAndOrderFront(nullptr);

    NS::Application* pApp = reinterpret_cast<NS::Application*>(pNotification->object());
    pApp->activateIgnoringOtherApps(true);
}

bool MyAppDelegate::applicationShouldTerminateAfterLastWindowClosed(NS::Application* pSender)
{
    return true;
}

#pragma endregion AppDelegate }


#pragma mark - ViewDelegate
#pragma region ViewDelegate {

MyMTKViewDelegate::MyMTKViewDelegate(MTL::Device* pDevice,
                                     HdEngine* eng,
                                     HdRenderIndex* ri,
                                     HdRenderBuffer* rb,
                                     HdRenderPassSharedPtr rp,
                                     HdRenderPassStateSharedPtr rps)
{
    mDevice = pDevice;
    mHydraEngine = eng;
    renderIndex = ri;
    renderBuffer = rb;
    renderPass = rp;
    mRenderPassState = rps;
    mDisplay = new MetalDisplay(mDevice);
}

MyMTKViewDelegate::~MyMTKViewDelegate()
{
    // delete _pRenderer;
}

void MyMTKViewDelegate::drawInMTKView(MTK::View* pView)
{
    HdTaskSharedPtrVector tasks;
    std::shared_ptr<SimpleRenderTask> renderTasks;
    TfTokenVector renderTags(1, HdRenderTagTokens->geometry);
    renderTasks = std::make_shared<SimpleRenderTask>(renderPass, mRenderPassState, renderTags);
    tasks.push_back(renderTasks);
    mHydraEngine->Execute(renderIndex, &tasks);
    oka::Buffer* res = renderBuffer->GetResource(false).UncheckedGet<oka::Buffer*>();
    mDisplay->display(pView, res);
}
void MyMTKViewDelegate::drawableSizeWillChange(MTK::View* pView, CGSize size)
{
    // _pRenderer->resize(size);
}

#pragma endregion ViewDelegate }


#pragma mark - Math

namespace math
{
constexpr simd::float3 add(const simd::float3& a, const simd::float3& b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

constexpr simd_float4x4 makeIdentity()
{
    using simd::float4;
    return (simd_float4x4){ (float4){ 1.f, 0.f, 0.f, 0.f }, (float4){ 0.f, 1.f, 0.f, 0.f },
                            (float4){ 0.f, 0.f, 1.f, 0.f }, (float4){ 0.f, 0.f, 0.f, 1.f } };
}

simd::float4x4 makePerspective(float fovRadians, float aspect, float znear, float zfar)
{
    using simd::float4;
    float ys = 1.f / tanf(fovRadians * 0.5f);
    float xs = ys / aspect;
    float zs = zfar / (znear - zfar);
    return simd_matrix_from_rows((float4){ xs, 0.0f, 0.0f, 0.0f }, (float4){ 0.0f, ys, 0.0f, 0.0f },
                                 (float4){ 0.0f, 0.0f, zs, znear * zs }, (float4){ 0, 0, -1, 0 });
}

simd::float4x4 makeXRotate(float angleRadians)
{
    using simd::float4;
    const float a = angleRadians;
    return simd_matrix_from_rows((float4){ 1.0f, 0.0f, 0.0f, 0.0f }, (float4){ 0.0f, cosf(a), sinf(a), 0.0f },
                                 (float4){ 0.0f, -sinf(a), cosf(a), 0.0f }, (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
}

simd::float4x4 makeYRotate(float angleRadians)
{
    using simd::float4;
    const float a = angleRadians;
    return simd_matrix_from_rows((float4){ cosf(a), 0.0f, sinf(a), 0.0f }, (float4){ 0.0f, 1.0f, 0.0f, 0.0f },
                                 (float4){ -sinf(a), 0.0f, cosf(a), 0.0f }, (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
}

simd::float4x4 makeZRotate(float angleRadians)
{
    using simd::float4;
    const float a = angleRadians;
    return simd_matrix_from_rows((float4){ cosf(a), sinf(a), 0.0f, 0.0f }, (float4){ -sinf(a), cosf(a), 0.0f, 0.0f },
                                 (float4){ 0.0f, 0.0f, 1.0f, 0.0f }, (float4){ 0.0f, 0.0f, 0.0f, 1.0f });
}

simd::float4x4 makeTranslate(const simd::float3& v)
{
    using simd::float4;
    const float4 col0 = { 1.0f, 0.0f, 0.0f, 0.0f };
    const float4 col1 = { 0.0f, 1.0f, 0.0f, 0.0f };
    const float4 col2 = { 0.0f, 0.0f, 1.0f, 0.0f };
    const float4 col3 = { v.x, v.y, v.z, 1.0f };
    return simd_matrix(col0, col1, col2, col3);
}

simd::float4x4 makeScale(const simd::float3& v)
{
    using simd::float4;
    return simd_matrix(
        (float4){ v.x, 0, 0, 0 }, (float4){ 0, v.y, 0, 0 }, (float4){ 0, 0, v.z, 0 }, (float4){ 0, 0, 0, 1.0 });
}

simd::float3x3 discardTranslation(const simd::float4x4& m)
{
    return simd_matrix(m.columns[0].xyz, m.columns[1].xyz, m.columns[2].xyz);
}

} // namespace math
