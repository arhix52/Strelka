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

#include <fstream>

using namespace oka;

void glfwdisplay::init(int width, int height, SharedContext* ctx)
{
    _pDevice = (MTL::Device*) ctx->mRender->getNativeDevicePtr();
    mWindowWidth = width;
    mWindowHeight = height;
    mCtx = ctx;

    if (!glfwInit())
    {
        fprintf(stderr, "ERROR: unable to init GLFW\n");
        return;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    if (!mWindow)
    {
        fprintf(stderr, "ERROR: unable to create GLFW Window\n");
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
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)(_pDevice));

    NSWindow *nswin = glfwGetCocoaWindow(mWindow);
    layer = CA::MetalLayer::layer();
    layer->setDevice(_pDevice);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    CAMetalLayer* l = (__bridge CAMetalLayer*)layer;
    nswin.contentView.layer = l;
    nswin.contentView.wantsLayer = YES;

    renderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();


    _pCommandQueue = _pDevice->newCommandQueue();
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    buildShaders();
}

void glfwdisplay::drawFrame(ImageBuffer& result)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();


    bool needRecreate = (result.height != mTexHeight || result.width != mTexWidth);
    if (needRecreate)
    {
        mTexWidth = result.width;
        mTexHeight = result.height;
        mTexture = buildTexture(mTexWidth, mTexHeight);
    }
    {
        MTL::Region region = MTL::Region::Make2D(0, 0, mTexWidth, mTexHeight);
        mTexture->replaceRegion(region, 0, result.data, result.width * oka::Buffer::getElementSize(result.pixel_format));
    }


    renderEncoder->pushDebugGroup(NS::String::string("display", NS::UTF8StringEncoding));

    renderEncoder->setRenderPipelineState(_pPSO);
    // [renderEncoder setRenderPipelineState: _pPSO];
    renderEncoder->setFragmentTexture(mTexture, /* index */ 0);
    // [renderEncoder setFragmentTexture: mTexture atIndex: 0];

    renderEncoder->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, 0ul, 6ul);
    // [renderEncoder drawPrimitives: MTL::PrimitiveTypeTriangle vertexStart: 0 vertexCount:6];

    renderEncoder->popDebugGroup();
}

MTL::Texture* glfwdisplay::buildTexture(uint32_t width, uint32_t heigth)
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

void glfwdisplay::buildShaders()
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
    pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
    // pDesc->setDepthAttachmentPixelFormat(MTL::PixelFormat::PixelFormatDepth16Unorm);

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

void glfwdisplay::destroy()
{

}

void glfwdisplay::onBeginFrame()
{
    int width, height;
    glfwGetFramebufferSize(mWindow, &width, &height);
    layer->setDrawableSize(CGSizeMake(width, height));
    drawable = layer->nextDrawable();

    float clear_color[4] = {0.45f, 0.55f, 0.60f, 1.00f};

    commandBuffer = _pCommandQueue->commandBuffer();
    renderPassDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor::Make(clear_color[0] * clear_color[3], clear_color[1] * clear_color[3], clear_color[2] * clear_color[3], clear_color[3]));
    renderPassDescriptor->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    renderEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);
    
}

void glfwdisplay::onEndFrame()
{
    renderEncoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    // [commandBuffer commit];
    commandBuffer->commit();
    glfwSwapBuffers(mWindow);
}

void glfwdisplay::drawUI()
{
    // Start the Dear ImGui frame
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)renderPassDescriptor);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiIO& io = ImGui::GetIO();

    const char* debugItems[] = { "None", "Normals" };
    static int currentDebugItemId = 0;

    /*
    bool openFD = false;
    static uint32_t showPropertiesId = -1;
    static uint32_t lightId = -1;
    static bool isLight = false;
    static bool openInspector = false;

    const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
    static int currentTonemapItemId = 1;

    const char* stratifiedSamplingItems[] = { "None", "Random", "Stratified", "Optimized" };
    static int currentSamplingItemId = 1;
    */

    ImGui::Begin("Menu:"); // begin window

    if (ImGui::BeginCombo("Debug view", debugItems[currentDebugItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(debugItems); n++)
        {
            bool is_selected = (currentDebugItemId == n);
            if (ImGui::Selectable(debugItems[n], is_selected))
            {
                currentDebugItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    mCtx->mSettingsManager->setAs<uint32_t>("render/pt/debug", currentDebugItemId);

    if (ImGui::TreeNode("Path Tracer"))
    {
        uint32_t maxDepth = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/depth");
        ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);

        uint32_t sppTotal = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/sppTotal");
        ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);

        uint32_t sppSubframe = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/spp");
        ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);

        bool enableAccumulation = mCtx->mSettingsManager->getAs<bool>("render/pt/enableAcc");
        ImGui::Checkbox("Enable Path Tracer Acc", &enableAccumulation);
        mCtx->mSettingsManager->setAs<bool>("render/pt/enableAcc", enableAccumulation);
        /*
        if (ImGui::BeginCombo("Stratified Sampling", stratifiedSamplingItems[currentSamplingItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(stratifiedSamplingItems); n++)
            {
                bool is_selected = (currentSamplingItemId == n);
                if (ImGui::Selectable(stratifiedSamplingItems[n], is_selected))
                {
                    currentSamplingItemId = n;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", currentSamplingItemId);
         */
        ImGui::TreePop();
    }

    if (ImGui::Button("Capture Screen"))
    {
        mCtx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", true);
    }

    float cameraSpeed = mCtx->mSettingsManager->getAs<float>("render/cameraSpeed");
    ImGui::InputFloat("Camera Speed", (float*)&cameraSpeed, 0.5);
    mCtx->mSettingsManager->setAs<float>("render/cameraSpeed", cameraSpeed);
    /*
    if (ImGui::BeginCombo("Tonemap", tonemapItems[currentTonemapItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
        {
            bool is_selected = (currentTonemapItemId == n);
            if (ImGui::Selectable(tonemapItems[n], is_selected))
            {
                currentTonemapItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    mCtx->mSettingsManager->setAs<bool>("render/pt/enableTonemap", currentTonemapItemId != 0 ? true : false);
    mCtx->mSettingsManager->setAs<uint32_t>("render/pt/tonemapperType", currentTonemapItemId - 1);

    bool enableUpscale = mCtx->mSettingsManager->getAs<bool>("render/pt/enableUpscale");
    ImGui::Checkbox("Enable Upscale", &enableUpscale);
    mCtx->mSettingsManager->setAs<bool>("render/pt/enableUpscale", enableUpscale);

    float upscaleFactor = 0.0f;
    if (enableUpscale)
    {
        upscaleFactor = 0.5f;
    }
    else
    {
        upscaleFactor = 1.0f;
    }
    mCtx->mSettingsManager->setAs<float>("render/pt/upscaleFactor", upscaleFactor);

    if (ImGui::Button("Capture Screen"))
    {
        mCtx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", true);
    }

    // bool isRecreate = ImGui::Button("Recreate BVH");
    // renderConfig.recreateBVH = isRecreate ? true : false;
    */

    ImGui::End(); // end window

        // Rendering
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), 
    (__bridge id<MTLCommandBuffer>)(commandBuffer), 
    (__bridge id<MTLRenderCommandEncoder>)renderEncoder);

}

// void glfwdisplay::display(MTK::View* pView, oka::Buffer* result)
// {
//     NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

//     bool needRecreate = (result->height() != height || result->width() != width);
//     if (needRecreate)
//     {
//         width = result->width();
//         height = result->height();
//         mTexture = buildTexture(width, height);
//     }
//     {
//         MTL::Region region = MTL::Region::Make2D(0, 0, width, height);
//         mTexture->replaceRegion(region, 0, result->map(), width * result->getElementSize());
//     }

//     MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
//     dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
//     glfwdisplay* pRenderer = this;
//     pCmd->addCompletedHandler(^void(MTL::CommandBuffer* pCmd) {
//         dispatch_semaphore_signal(pRenderer->_semaphore);
//     });
//     MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
//     MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder(pRpd);

//     pEnc->setRenderPipelineState(_pPSO);
//     pEnc->setFragmentTexture(mTexture, /* index */ 0);
//     pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, 0ul, 6ul);

//     pEnc->endEncoding();
//     pCmd->presentDrawable(pView->currentDrawable());
//     pCmd->commit();

//     pPool->release();
// }
