#pragma once
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <strelka/display/display.h>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <string>

namespace oka
{

class GlfwDisplay : public Display
{
public:
    GlfwDisplay() = default;
    ~GlfwDisplay() override;

    void init(int width, int height, SettingsManager* settings) override;
    void setNativeDevice(void* device) override;
    void setCommandQueue(void* queue) override;
    void destroy() override;

    void onBeginFrame() override;
    void onEndFrame() override;

    void* getDisplayNativeTexure() override;

    void drawFrame(ImageBuffer& result) override;
    void drawUI() override;

    float getMaxEDR() override;

private:
    static constexpr size_t kMaxFramesInFlight = 3;

    MTL::Device* _pDevice = nullptr;
    MTL::CommandQueue* _pCommandQueue = nullptr;
    bool _ownsCommandQueue = false;
    MTL::Library* _pShaderLibrary = nullptr;
    MTL::RenderPipelineState* _pPSO = nullptr;
    MTL::Texture* mTexture = nullptr;
    uint32_t mTexWidth = 32;
    uint32_t mTexHeight = 32;
    dispatch_semaphore_t _semaphore = nullptr;
    CA::MetalLayer* layer = nullptr;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;

    MTL::Texture* buildTexture(uint32_t width, uint32_t heigth);
    void buildShaders();

    MTL::CommandBuffer* mCommandBuffer = nullptr;
    MTL::RenderCommandEncoder* mRenderEncoder = nullptr;
    MTL::BlitCommandEncoder* mBlitEncoder = nullptr;
    CA::MetalDrawable* drawable = nullptr;

    // Per-frame autorelease pool. Metal factory methods (commandBuffer(),
    // nextDrawable(), blitCommandEncoder(), ...) return autoreleased objects;
    // without a pool that is drained every frame they accumulate for the whole
    // process lifetime.
    NS::AutoreleasePool* mFramePool = nullptr;

    // True while a drawable could not be acquired this frame; onEndFrame/drawUI
    // must then skip all GPU work instead of dereferencing null.
    bool mFrameValid = false;
    bool mDestroyed = false;

    // Backing storage for ImGuiIO::IniFilename, which keeps the raw pointer.
    std::string mIniPath;
};

} // namespace oka
