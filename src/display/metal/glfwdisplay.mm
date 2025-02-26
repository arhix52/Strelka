#include "glfwdisplay.h"
#include <render/render.h>

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

#include <fstream>
#include <log.h>

using namespace oka;

void GlfwDisplay::setNativeDevice(void* device)
{
    _pDevice = (MTL::Device*) device;
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
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)(_pDevice));

    NSWindow *nswin = glfwGetCocoaWindow(mWindow);
    layer = CA::MetalLayer::layer();
    layer->setDevice(_pDevice);
    layer->setPixelFormat(MTL::PixelFormatRGBA16Float);
    auto l = (__bridge CAMetalLayer*)layer;
    const CFStringRef name = kCGColorSpaceExtendedDisplayP3;
    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(name);
    l.colorspace = colorspace;
    CGColorSpaceRelease(colorspace);

    l.wantsExtendedDynamicRangeContent = YES;
    nswin.contentView.layer = l;
    nswin.contentView.wantsLayer = YES;

    renderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();

    _pCommandQueue = _pDevice->newCommandQueue();
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    buildShaders();
}

void* GlfwDisplay::getDisplayNativeTexure()
{
    return mTexture;
}

float GlfwDisplay::getMaxEDR()
{
    NSWindow *nswin = glfwGetCocoaWindow(mWindow);
    return nswin.screen.maximumExtendedDynamicRangeColorComponentValue;
}

void GlfwDisplay::drawFrame(ImageBuffer& result)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    const bool needRecreate = result.height != mTexHeight || result.width != mTexWidth;
    if (needRecreate)
    {
        mTexWidth = result.width;
        mTexHeight = result.height;
        if (mTexture)
        {
            mTexture->release();
        }
        mTexture = buildTexture(mTexWidth, mTexHeight);
    }

    mBlitEncoder = mCommandBuffer->blitCommandEncoder();

    mBlitEncoder->copyFromBuffer(
        (MTL::Buffer*) result.deviceData, 0, 
        oka::Buffer::getElementSize(result.pixel_format) * mTexWidth, 
        oka::Buffer::getElementSize(result.pixel_format) * mTexWidth * mTexHeight, 
        MTL::Size{mTexWidth, mTexHeight, 1}, 
        mTexture, 0, 0, MTL::Origin{0, 0, 0});

    mBlitEncoder->endEncoding();

    pPool->release();
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

void GlfwDisplay::buildShaders()
{
    using NS::StringEncoding::UTF8StringEncoding;

    std::string shaderSrc;
    readSourceFile(shaderSrc, "./metal/shaders/fullScreen.metal");

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

void GlfwDisplay::destroy()
{

}

void GlfwDisplay::onBeginFrame()
{
    // NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    int width, height;
    glfwGetFramebufferSize(mWindow, &width, &height);
    layer->setDrawableSize(CGSizeMake(width, height));
    drawable = layer->nextDrawable();

    float clear_color[4] = {0.45f, 0.55f, 0.60f, 1.00f};

    mCommandBuffer = _pCommandQueue->commandBuffer();
    renderPassDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor::Make(clear_color[0] * clear_color[3], clear_color[1] * clear_color[3], clear_color[2] * clear_color[3], clear_color[3]));
    renderPassDescriptor->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    // Start the Dear ImGui frame
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)renderPassDescriptor);
}

void GlfwDisplay::onEndFrame()
{
    mCommandBuffer->presentDrawable(drawable);
    mCommandBuffer->commit();

    mRenderEncoder->release();
    mCommandBuffer->release();
    drawable->release();

    glfwSwapBuffers(mWindow);
}

void GlfwDisplay::drawUI()
{
    mRenderEncoder = mCommandBuffer->renderCommandEncoder(renderPassDescriptor);
@autoreleasepool 
    {

        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
        (__bridge id<MTLCommandBuffer>)(mCommandBuffer),
        (__bridge id<MTLRenderCommandEncoder>)mRenderEncoder);
    }
    mRenderEncoder->endEncoding();
}
