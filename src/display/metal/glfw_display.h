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
    void resetFrame() override;

    bool isFrameValid() const override
    {
        return mFrameValid;
    }

    float getMaxEDR() override;
    display_output::DisplayCapabilities getOutputCapabilities() const override;

private:
    static constexpr size_t kMaxFramesInFlight = 3;

    /// Reads the NSScreen the window currently sits on. Crosses into AppKit, so
    /// the frame loop calls it on a timer rather than every frame; everything
    /// else reads the cached snapshot.
    void refreshDisplayCapabilities();
    /// Pushes the user's choices onto the CAMetalLayer, and only when one of
    /// them changed -- reassigning the colour space rebuilds the layer's
    /// drawables, which is a visible hitch if done per frame.
    void applyDisplaySettings();
    display_output::OutputMode requestedOutputMode() const;

    MTL::Device* _pDevice = nullptr;
    MTL::CommandQueue* _pCommandQueue = nullptr;
    bool _ownsCommandQueue = false;
    MTL::Library* _pShaderLibrary = nullptr;
    MTL::RenderPipelineState* _pPSO = nullptr;
    MTL::Texture* mTexture = nullptr;
    // False when the texture belongs to the renderer, which is the normal case
    // on Metal; the display must not release what it does not own.
    bool mOwnsTexture = false;
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

    display_output::DisplayCapabilities mOutputCapabilities;
    // UINT32_MAX rather than 0: mode 0 (Auto) is a legal request, so a zero
    // "applied" value would make the first frame skip the layer configuration
    // and leave whatever init() happened to set.
    uint32_t mAppliedOutputMode = UINT32_MAX;
    uint32_t mAppliedDrawableCount = 0;
    bool mAppliedDisplaySync = true;
    // Read every frame instead of latched, because it costs nothing: it only
    // picks which presentDrawable overload onEndFrame calls.
    float mFrameRateLimitHz = 0.0f;
    uint64_t mFrameIndex = 0;
};

} // namespace oka
