#pragma once
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include "Display.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace oka
{

class GlfwDisplay : public Display
{
public:
    GlfwDisplay() = default;
    ~GlfwDisplay() override = default;

    void init(int width, int height, SettingsManager* settings) override;
    void setNativeDevice(void* device) override;
    void destroy() override;

    void onBeginFrame() override;
    void onEndFrame() override;

    void* getDisplayNativeTexure() override;

    void drawFrame(ImageBuffer& result) override;
    void drawUI() override;

private:
    static constexpr size_t kMaxFramesInFlight = 3;

    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pShaderLibrary;
    MTL::RenderPipelineState* _pPSO;
    MTL::Texture* mTexture;
    uint32_t mTexWidth = 32;
    uint32_t mTexHeight = 32;
    dispatch_semaphore_t _semaphore;
    CA::MetalLayer* layer;
    MTL::RenderPassDescriptor *renderPassDescriptor;

    MTL::Texture* buildTexture(uint32_t width, uint32_t heigth);
    void buildShaders();
    
    MTL::CommandBuffer* mCommandBuffer;
    MTL::RenderCommandEncoder* mRenderEncoder;
    MTL::BlitCommandEncoder* mBlitEncoder;
    CA::MetalDrawable* drawable;
};

} // namespace oka
