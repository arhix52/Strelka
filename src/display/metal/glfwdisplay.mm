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

    int width, height;
    glfwGetFramebufferSize(mWindow, &width, &height);
    layer->setDrawableSize(CGSizeMake(width, height));
    CA::MetalDrawable* drawable = layer->nextDrawable();

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

    float clear_color[4] = {0.45f, 0.55f, 0.60f, 1.00f};

    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(_pCommandQueue->commandBuffer());
    renderPassDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor::Make(clear_color[0] * clear_color[3], clear_color[1] * clear_color[3], clear_color[2] * clear_color[3], clear_color[3]));
    renderPassDescriptor->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPassDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPassDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);

    id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder pushDebugGroup:@"display"];

    // pEnc->setRenderPipelineState(_pPSO);
    [renderEncoder setRenderPipelineState: _pPSO];
    // pEnc->setFragmentTexture(mTexture, /* index */ 0);
    [renderEncoder setFragmentTexture: mTexture atIndex: 0];

    // pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, 0ul, 6ul);
    [renderEncoder drawPrimitives: MTL::PrimitiveTypeTriangle vertexStart: 0 vertexCount:6];


    // Start the Dear ImGui frame
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)renderPassDescriptor);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        // ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        // ImGui::Checkbox("Another Window", &show_another_window);

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }
        // Rendering
    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);
    [renderEncoder popDebugGroup];
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
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

}

void glfwdisplay::onEndFrame()
{

}

void glfwdisplay::drawUI()
{

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
