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
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <log.h>
#include <paths.h>

using namespace oka;

namespace
{
bool readSourceFile(std::string& str, const std::string& filename)
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

/// How often the AppKit probe runs, in frames.
///
/// Every field it reads can change while the editor is running -- the window is
/// dragged to another monitor, the user switches the panel to a reference
/// preset, the compositor lowers the granted headroom because another window
/// wants the backlight -- so it cannot be read once at startup. It is also a
/// dozen ObjC property reads and an NSString, which is not something to spend
/// per frame for values that move on a human timescale.
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

    // ImGui defaults to a bare "imgui.ini" resolved against the *working
    // directory*, so launching the editor from anywhere but the build root meant
    // it neither found nor persisted a layout, and the dockspace came up empty
    // every time. Anchor it to the executable instead, and seed it from the
    // layout shipped in the source tree on first run.
    mIniPath = (oka::getExecutableDir() / "imgui.ini").string();
    std::error_code ec;
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
    // Panels, menus and sliders reachable from the pad. The ImGui GLFW backend
    // feeds it from the same joystick GLFW hands us, so this needs no wiring --
    // but it is deliberately *only* nav: the pad does not move the pointer, so
    // gizmo drags and viewport picking stay on the mouse.
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    imgui_style::applyGraphiteBlue();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)(_pDevice));

    NSWindow *const nswin = glfwGetCocoaWindow(mWindow);
    // CA::MetalLayer::layer() and RenderPassDescriptor::renderPassDescriptor()
    // below are autoreleased factories. Both are stored as members and used for
    // the whole process lifetime, so they must be retained explicitly — they only
    // survived before because nothing ever drained the enclosing pool.
    layer = CA::MetalLayer::layer()->retain();
    layer->setDevice(_pDevice);
    layer->setPixelFormat(MTL::PixelFormatRGBA16Float);
    auto l = (__bridge CAMetalLayer*)layer;
    nswin.contentView.layer = l;
    nswin.contentView.wantsLayer = YES;

    // Colour space, EDR request, vsync and drawable count are all user-visible
    // choices that can change while the editor runs, so they are set from the
    // settings in one place instead of half here and half in the frame loop.
    // The probe runs before the first frame because EditorApp asks for the EDR
    // headroom before it calls onBeginFrame().
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
    // The mode-adjusted headroom, not the raw NSScreen value. SDR output has to
    // report 1.0 or the tone curve keeps mapping into range the layer no longer
    // carries, and the user's headroom ceiling has to reach the tone curve to
    // mean anything at all -- nothing else consumes it.
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

        // AppKit describes the refresh range as intervals, and the shortest
        // interval is the *highest* rate -- the two names read backwards against
        // the hertz they turn into. A fixed-rate panel reports the same interval
        // twice, which is what tells ProMotion apart from a 60 Hz display.
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

    // Fires on the first probe and again whenever the window is dragged to
    // another monitor, which are the two moments the numbers below change
    // wholesale. Logged rather than left to the panel because "the image looks
    // wrong on the second screen" is a report that arrives without a screenshot
    // of the settings.
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
    // Reported through the backend-neutral fields too, so a log line or a future
    // headless probe reading those gets the same answer as the Metal block.
    // Nothing sets surfaceEncoding: this path never produces an HDR10 surface,
    // it hands extended-range values to the window server.
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

    // The renderer and its ACES output matrix produce sRGB primaries. Let
    // ColorSync convert those to the actual panel gamut instead of labelling
    // them as Display P3, which would oversaturate the image. The extended
    // transfer function carries encoded values above SDR white for EDR; the
    // plain one clamps them, which is exactly what SDR output means.
    //
    // The pixel format stays RGBA16Float in SDR as well. An 8-bit layer would
    // need the blit PSO and ImGui's Metal pipeline rebuilt for the new
    // attachment format and buys nothing here -- the window server does the
    // clamping either way.
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
        // Retained, not borrowed. The renderer frees and recreates its display
        // textures whenever the render resolution or the MetalFX usage flags
        // change -- a window resize, an upscaler or the denoiser being switched on
        // -- while this pointer stays live in ImGui's draw list until the frame is
        // encoded, and getDisplayNativeTexure() keeps handing it out on any frame
        // that lands no new one. Borrowing it meant that encode could retain freed
        // memory: a segfault inside setFragmentTexture: with nothing in the log.
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

    const std::string shaderPath = oka::resolveResourcePath("metal/shaders/fullScreen.metal");
    std::string shaderSrc;
    if (!readSourceFile(shaderSrc, shaderPath))
    {
        STRELKA_FATAL("Failed to read {} (looked next to the executable and in the working directory)", shaderPath);
        assert(false);
        return;
    }

    NS::Error* pError = nullptr;
    MTL::Library* pLibrary =
        _pDevice->newLibrary(NS::String::string(shaderSrc.c_str(), UTF8StringEncoding), nullptr, &pError);
    if (!pLibrary)
    {
        STRELKA_FATAL("{}", pError->localizedDescription()->utf8String());
        assert(false);
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
    // Before the semaphore wait and every early-out below it: a frame this
    // display drops is still a frame in which the user may have changed the
    // output mode, and skipping the apply would leave the layer stale until a
    // frame happens to complete.
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
    // Wait for the frame the renderer produced, when it produced it on another
    // queue. While a new trace is in flight the display still owns the other
    // double-buffered slot, which is already complete and needs no wait. Waiting
    // for the new slot here would put every UI command buffer behind a 500 ms
    // trace, exhaust the display semaphore, and freeze the editor while showing
    // pixels that did not depend on that trace.
    //
    // Once the renderer publishes the new slot, Metal orders its visibility
    // across the Metal 4 render queue and this Metal 3 display queue with the
    // completed frame event. Returns null when both share a queue.
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

    // commandBuffer(), nextDrawable() and renderCommandEncoder() all return
    // autoreleased objects — they are owned by mFramePool, not by us. Releasing
    // them explicitly (as the previous code did) was an over-release that only
    // stayed latent because the pool was never drained.
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
