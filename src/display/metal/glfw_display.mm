#include "glfw_display.h"
#include "imgui_style.h"
#include <strelka/render/render.h>

#define IMGUI_IMPL_METAL_CPP
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CGColorSpace.h>

#include <algorithm>
#include <application_paths.h>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <log.h>
#include <paths.h>

using namespace oka;

namespace
{
constexpr uint64_t kCapabilityPollFrames = 15;

const char* outputModeName(oka::display_output::OutputMode mode)
{
    switch (mode)
    {
    case oka::display_output::OutputMode::Auto:
        return "auto";
    case oka::display_output::OutputMode::HDR:
        return "extended";
    case oka::display_output::OutputMode::SDR:
        return "sdr";
    case oka::display_output::OutputMode::ReferenceHDR:
        return "reference";
    }
    return "auto";
}
} // namespace

void GlfwDisplay::setNativeDevice(void* device)
{
    _pDevice = (MTL::Device*) device;
}

void GlfwDisplay::setCommandQueue(void* queue)
{
    _pCommandQueue = (MTL::CommandQueue*) queue;
}

void GlfwDisplay::init(int width, int height, SettingsManager* settings)
{
    mWindowWidth = width;
    mWindowHeight = height;
    mSettings = settings;

    if (!glfwInit())
    {
        STRELKA_FATAL("Failed to init GLFW");
        return;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    if (!mWindow)
    {
        STRELKA_FATAL("Failed to create GLFW Window");
        return;
    }
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, handleMouseMoveCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);

    glfwMakeContextCurrent(mWindow);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // A drag over a panel's body is content interaction, not a window move: the
    // viewport spends every drag on the camera or a gizmo, and an undocked one
    // would otherwise slide across the screen as the user works in it.
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    std::error_code ec;
    const std::filesystem::path supportDirectory = oka::applicationSupportDirectory();
    std::filesystem::create_directories(supportDirectory, ec);
    mIniPath = (supportDirectory / "imgui.ini").string();
    if (!std::filesystem::exists(mIniPath, ec))
    {
        const std::string defaultLayout = oka::resolveResourcePath("default_layout.ini");
        if (std::filesystem::exists(defaultLayout, ec))
        {
            std::filesystem::copy_file(defaultLayout, mIniPath, ec);
            // Braces are required: the STRELKA_* macros expand to a full
            // `if (...) { ... }` statement, so an unbraced if/else around them
            // does not parse.
            if (ec)
            {
                STRELKA_WARNING("Could not seed ImGui layout from {}: {}", defaultLayout, ec.message());
            }
            else
            {
                STRELKA_INFO("Initialised ImGui layout from {}", defaultLayout);
            }
        }
    }
    io.IniFilename = mIniPath.c_str(); // mIniPath must outlive the ImGui context
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    imgui_style::applyGraphiteBlue();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)(_pDevice));

    NSWindow *const nswin = glfwGetCocoaWindow(mWindow);
    layer = CA::MetalLayer::layer()->retain();
    layer->setDevice(_pDevice);
    layer->setPixelFormat(MTL::PixelFormatRGBA16Float);
    auto l = (__bridge CAMetalLayer*)layer;
    nswin.contentView.layer = l;
    nswin.contentView.wantsLayer = YES;

    refreshDisplayCapabilities();
    applyDisplaySettings();

    renderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor()->retain();

    if (!_pCommandQueue)
    {
        _pCommandQueue = _pDevice->newCommandQueue();
        _ownsCommandQueue = true;
    }
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    buildShaders();
}

void* GlfwDisplay::getDisplayNativeTexure()
{
    return mTexture;
}

void GlfwDisplay::resetFrame()
{
    if (mTexture && mOwnsTexture)
    {
        mTexture->release();
    }
    mTexture = nullptr;
    mOwnsTexture = false;
}

float GlfwDisplay::getMaxEDR()
{
    return mOutputCapabilities.appliedHeadroom;
}

display_output::DisplayCapabilities GlfwDisplay::getOutputCapabilities() const
{
    return mOutputCapabilities;
}

display_output::OutputMode GlfwDisplay::requestedOutputMode() const
{
    uint32_t storedMode = 0;

    if (mSettings != nullptr)
    {
        storedMode = mSettings->getAs<uint32_t>("render/post/outputMode");
    }
    storedMode = std::min(storedMode, static_cast<uint32_t>(display_output::OutputMode::ReferenceHDR));
    return static_cast<display_output::OutputMode>(storedMode);
}

void GlfwDisplay::refreshDisplayCapabilities()
{
    if (mWindow == nullptr)
    {
        return;
    }

    NSWindow* const nswin = glfwGetCocoaWindow(mWindow);
    // nil while the window is entirely off-screen or being torn down. The main
    // screen is what AppKit itself falls back to, and answering with last
    // frame's numbers would be worse than answering with another display's.
    NSScreen* screen = nswin.screen;
    if (screen == nil)
    {
        screen = [NSScreen mainScreen];
    }

    display_output::EdrCapabilities edr;
    std::string displayName;
    std::string colorSpaceName;
    float minRefreshRateHz = 0.0f;
    float maxRefreshRateHz = 0.0f;
    float currentRefreshRateHz = 0.0f;

    if (screen != nil)
    {
        // -UTF8String is declared nullable even on a non-nil NSString, and
        // assigning null to a std::string is undefined -- the analyzer is right
        // to insist on the check rather than on a nil test of the NSString.
        const char* const name = screen.localizedName.UTF8String;
        if (name != nullptr)
        {
            displayName = name;
        }
        edr.currentHeadroom = static_cast<float>(screen.maximumExtendedDynamicRangeColorComponentValue);
        edr.potentialHeadroom =
            static_cast<float>(screen.maximumPotentialExtendedDynamicRangeColorComponentValue);
        edr.referenceHeadroom =
            static_cast<float>(screen.maximumReferenceExtendedDynamicRangeColorComponentValue);
        edr.wideGamut = [screen canRepresentDisplayGamut:NSDisplayGamutP3] == YES;

        NSColorSpace* const colorSpace = screen.colorSpace;
        const char* const colorSpaceUtf8 = colorSpace != nil ? colorSpace.localizedName.UTF8String : nullptr;
        if (colorSpaceUtf8 != nullptr)
        {
            colorSpaceName = colorSpaceUtf8;
        }

        const NSTimeInterval shortestInterval = screen.minimumRefreshInterval;
        const NSTimeInterval longestInterval = screen.maximumRefreshInterval;
        if (shortestInterval > 0.0)
        {
            maxRefreshRateHz = static_cast<float>(1.0 / shortestInterval);
        }
        if (longestInterval > 0.0)
        {
            minRefreshRateHz = static_cast<float>(1.0 / longestInterval);
        }
        currentRefreshRateHz = static_cast<float>(screen.maximumFramesPerSecond);
        if (maxRefreshRateHz <= 0.0f)
        {
            maxRefreshRateHz = currentRefreshRateHz;
        }
    }

    if (layer != nullptr)
    {
        CAMetalLayer* const l = (__bridge CAMetalLayer*)layer;
        edr.edrRequested = l.wantsExtendedDynamicRangeContent == YES;
        mOutputCapabilities.maxDrawableCount = static_cast<uint32_t>(l.maximumDrawableCount);
        mOutputCapabilities.displaySync = l.displaySyncEnabled == YES;
    }

    if (mOutputCapabilities.displayName != displayName)
    {
        STRELKA_INFO("Display \"{}\": EDR headroom {:.2f}x now, {:.2f}x potential, {:.2f}x reference; "
                     "refresh {:.1f}-{:.1f} Hz; colour space {}{}",
                     displayName, edr.currentHeadroom, edr.potentialHeadroom, edr.referenceHeadroom,
                     minRefreshRateHz, maxRefreshRateHz,
                     colorSpaceName.empty() ? "unknown" : colorSpaceName, edr.wideGamut ? " (P3 capable)" : "");
    }

    mOutputCapabilities.backend = display_output::DisplayBackend::Metal;
    mOutputCapabilities.edr = edr;
    mOutputCapabilities.displayName = displayName;
    mOutputCapabilities.colorSpaceName = colorSpaceName;
    mOutputCapabilities.minRefreshRateHz = minRefreshRateHz;
    mOutputCapabilities.maxRefreshRateHz = maxRefreshRateHz;
    mOutputCapabilities.currentRefreshRateHz = currentRefreshRateHz;
    mOutputCapabilities.vrrStatus = display_output::interpretRefreshRange(minRefreshRateHz, maxRefreshRateHz);
    mOutputCapabilities.present.vrr = mOutputCapabilities.vrrStatus == display_output::VrrStatus::Supported;
    mOutputCapabilities.output.hdr10 = edr.potentialHeadroom > 1.0f;
}

void GlfwDisplay::applyDisplaySettings()
{
    uint32_t mode = static_cast<uint32_t>(display_output::OutputMode::Auto);
    bool displaySync = true;
    bool tripleBuffering = true;
    float frameRateLimitHz = 0.0f;
    float headroomLimit = 0.0f;

    if (mSettings != nullptr)
    {
        mode = static_cast<uint32_t>(requestedOutputMode());
        displaySync = mSettings->getAs<bool>("display/vsync/enabled");
        tripleBuffering = mSettings->getAs<bool>("display/present/tripleBuffering");
        frameRateLimitHz = mSettings->getAs<float>("display/present/fpsLimit");
        headroomLimit = mSettings->getAs<float>("display/edr/headroomLimit");
    }
    mFrameRateLimitHz = frameRateLimitHz > 0.0f ? frameRateLimitHz : 0.0f;

    // Cheap and pure, so it runs every frame rather than on the AppKit poll: the
    // ceiling is a slider, and a knob that takes a quarter of a second to move
    // the image reads as a knob that does not work.
    mOutputCapabilities.appliedHeadroom = display_output::selectEdrHeadroom(
        static_cast<display_output::OutputMode>(mode), mOutputCapabilities.edr, headroomLimit);
    mOutputCapabilities.output.hdrSelected = mOutputCapabilities.appliedHeadroom > 1.0f;
    mOutputCapabilities.frameRateLimitHz = mFrameRateLimitHz;

    const uint32_t drawableCount = tripleBuffering ? 3U : 2U;
    if (layer == nullptr ||
        (mode == mAppliedOutputMode && displaySync == mAppliedDisplaySync &&
         drawableCount == mAppliedDrawableCount))
    {
        return;
    }

    CAMetalLayer* const l = (__bridge CAMetalLayer*)layer;
    const bool sdr = mode == static_cast<uint32_t>(display_output::OutputMode::SDR);

    const CFStringRef colorSpaceName = sdr ? kCGColorSpaceSRGB : kCGColorSpaceExtendedSRGB;
    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(colorSpaceName);
    l.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);

    l.wantsExtendedDynamicRangeContent = sdr ? NO : YES;
    l.displaySyncEnabled = displaySync ? YES : NO;
    // Two drawables give up the compositor's slack to save a frame of latency.
    // The in-flight semaphore is sized for three either way, so with two it is
    // nextDrawable that applies the back pressure.
    l.maximumDrawableCount = drawableCount;

    mAppliedOutputMode = mode;
    mAppliedDisplaySync = displaySync;
    mAppliedDrawableCount = drawableCount;
    STRELKA_INFO("Display output: mode={} colorspace={} vsync={} drawables={}",
                 outputModeName(static_cast<display_output::OutputMode>(mode)),
                 sdr ? "sRGB" : "extended sRGB", displaySync, drawableCount);
}

void GlfwDisplay::drawFrame(ImageBuffer& result)
{
    if (!mFrameValid || result.deviceData == nullptr || result.width == 0 || result.height == 0)
    {
        return;
    }

    // A renderer that produced a texture has already done this work.
    if (result.deviceTexture)
    {
        MTL::Texture* incoming = (MTL::Texture*)result.deviceTexture;
        if (incoming != mTexture)
        {
            incoming->retain();
            if (mTexture && mOwnsTexture)
            {
                mTexture->release();
            }
            mTexture = incoming;
            mOwnsTexture = true;
        }
        mTexWidth = result.width;
        mTexHeight = result.height;
        return;
    }

    const bool needRecreate = result.height != mTexHeight || result.width != mTexWidth;
    if (needRecreate)
    {
        mTexWidth = result.width;
        mTexHeight = result.height;
        if (mTexture && mOwnsTexture)
        {
            mTexture->release();
        }
        mTexture = buildTexture(mTexWidth, mTexHeight);
        mOwnsTexture = true;
    }

    mBlitEncoder = mCommandBuffer->blitCommandEncoder();

    mBlitEncoder->copyFromBuffer(
        (MTL::Buffer*) result.deviceData, 0,
        oka::Buffer::getElementSize(result.pixel_format) * mTexWidth,
        oka::Buffer::getElementSize(result.pixel_format) * mTexWidth * mTexHeight,
        MTL::Size{mTexWidth, mTexHeight, 1},
        mTexture, 0, 0, MTL::Origin{0, 0, 0});

    mBlitEncoder->endEncoding();
    mBlitEncoder = nullptr;
}

MTL::Texture* GlfwDisplay::buildTexture(uint32_t width, uint32_t heigth)
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

void GlfwDisplay::buildShaders()
{
    using NS::StringEncoding::UTF8StringEncoding;

    const std::string shaderPath = oka::resolveResourcePath("metal/shaders/fullScreen.metallib");
    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newLibrary(NS::String::string(shaderPath.c_str(), UTF8StringEncoding), &pError);
    if (!pLibrary)
    {
        STRELKA_FATAL("Failed to load {}: {}", shaderPath, pError->localizedDescription()->utf8String());
        assert(false);
        return;
    }

    MTL::Function* pVertexFn = pLibrary->newFunction(NS::String::string("copyVertex", UTF8StringEncoding));
    MTL::Function* pFragFn = pLibrary->newFunction(NS::String::string("copyFragment", UTF8StringEncoding));

    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction(pVertexFn);
    pDesc->setFragmentFunction(pFragFn);
    pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA16Float);
    // pDesc->setDepthAttachmentPixelFormat(MTL::PixelFormat::PixelFormatDepth16Unorm);

    _pPSO = _pDevice->newRenderPipelineState(pDesc, &pError);
    if (!_pPSO)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
    }

    pVertexFn->release();
    pFragFn->release();
    pDesc->release();
    _pShaderLibrary = pLibrary;
}

GlfwDisplay::~GlfwDisplay()
{
    @autoreleasepool
    {
        // Qualified: a virtual call from a destructor would skip overrides of
        // derived classes that no longer exist, and clang-analyzer flags it.
        GlfwDisplay::destroy();
    }
}

void GlfwDisplay::destroy()
{
    if (mDestroyed)
    {
        return;
    }
    mDestroyed = true;

    if (mWindow)
    {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    if (_pPSO)
    {
        _pPSO->release();
        _pPSO = nullptr;
    }
    if (_pShaderLibrary)
    {
        _pShaderLibrary->release();
        _pShaderLibrary = nullptr;
    }
    // Only if it is ours. The renderer's texture is retained on adoption, so this
    // release is against a reference this display holds and is safe even though
    // the renderer is destroyed first.
    if (mTexture && mOwnsTexture)
    {
        mTexture->release();
    }
    mTexture = nullptr;
    if (renderPassDescriptor)
    {
        renderPassDescriptor->release();
        renderPassDescriptor = nullptr;
    }
    if (layer)
    {
        layer->release();
        layer = nullptr;
    }
    if (_pCommandQueue && _ownsCommandQueue)
    {
        _pCommandQueue->release();
    }
    _pCommandQueue = nullptr;
    // _pDevice is owned by the renderer — do not release it here.
}

void GlfwDisplay::onBeginFrame()
{
    if ((mFrameIndex % kCapabilityPollFrames) == 0)
    {
        refreshDisplayCapabilities();
    }
    applyDisplaySettings();
    ++mFrameIndex;

    // Bounded like Metal4's 5s waits: a forever wait here freezes the whole
    // editor if a completed-handler never runs (GPU hang / lost device).
    constexpr int64_t kFrameWaitNs = 5LL * 1000LL * 1000LL * 1000LL;
    const dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, kFrameWaitNs);
    if (dispatch_semaphore_wait(_semaphore, deadline) != 0)
    {
        STRELKA_ERROR("Display frame semaphore timed out after 5s — skipping frame");
        mFrameValid = false;
        return;
    }

    mFramePool = NS::AutoreleasePool::alloc()->init();
    mFrameValid = false;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(mWindow, &width, &height);
    // A minimised window reports a 0x0 framebuffer; asking CAMetalLayer for a
    // zero-sized drawable is invalid.
    if (width <= 0 || height <= 0)
    {
        dispatch_semaphore_signal(_semaphore);
        mFramePool->release();
        mFramePool = nullptr;
        return;
    }

    layer->setDrawableSize(CGSizeMake(width, height));
    drawable = layer->nextDrawable();
    if (!drawable)
    {
        // The drawable pool is exhausted (compositor back-pressure). Drop this
        // frame rather than dereferencing null; the next iteration retries.
        dispatch_semaphore_signal(_semaphore);
        mFramePool->release();
        mFramePool = nullptr;
        return;
    }

    const float clear_color[4] = {0.45f, 0.55f, 0.60f, 1.00f};

    mCommandBuffer = _pCommandQueue->commandBuffer();
    if (mRender && !mRender->isRenderBusy() && !mRender->deviceError())
    {
        if (auto* ev = (MTL::Event*)mRender->getNativeFrameEvent())
        {
            const uint64_t v = mRender->frameEventValue();
            if (v != 0)
            {
                mCommandBuffer->encodeWait(ev, v);
            }
        }
    }
    renderPassDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor::Make(clear_color[0] * clear_color[3], clear_color[1] * clear_color[3], clear_color[2] * clear_color[3], clear_color[3]));
    renderPassDescriptor->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    mFrameValid = true;

    // Start the Dear ImGui frame
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)renderPassDescriptor);
}

void GlfwDisplay::onEndFrame()
{
    if (!mFrameValid)
    {
        return;
    }

    if (mFrameRateLimitHz > 0.0f)
    {
        // Not a sleep: the minimum duration is a request to the window server,
        // which on a variable-refresh panel answers it by dropping the panel to
        // a matching rate instead of repeating frames at the maximum one.
        mCommandBuffer->presentDrawableAfterMinimumDuration(
            drawable, static_cast<CFTimeInterval>(1.0F / mFrameRateLimitHz));
    }
    else
    {
        mCommandBuffer->presentDrawable(drawable);
    }

    const dispatch_semaphore_t sem = _semaphore;
    mCommandBuffer->addCompletedHandler(^void(MTL::CommandBuffer* /*cb*/) {
        dispatch_semaphore_signal(sem);
    });

    mCommandBuffer->commit();

    mRenderEncoder = nullptr;
    mCommandBuffer = nullptr;
    drawable = nullptr;
    mFrameValid = false;

    mFramePool->release();
    mFramePool = nullptr;
}

void GlfwDisplay::drawUI()
{
    if (!mFrameValid)
    {
        return;
    }

    mRenderEncoder = mCommandBuffer->renderCommandEncoder(renderPassDescriptor);

    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        (__bridge id<MTLCommandBuffer>)(mCommandBuffer),
        (__bridge id<MTLRenderCommandEncoder>)mRenderEncoder);

    mRenderEncoder->endEncoding();
}
