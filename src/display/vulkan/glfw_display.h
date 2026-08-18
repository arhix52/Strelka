#pragma once

#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define GLFW_INCLUDE_VULKAN
#include <strelka/display/display.h>

#include "cuda_vulkan_interop.h"

#include <array>
#include <vector>

struct ImGui_ImplVulkanH_Window;

namespace oka
{

class GlfwDisplay : public Display
{
public:
    GlfwDisplay() = default;
    ~GlfwDisplay() override;

    void init(int width, int height, SettingsManager *settings) override;
    void destroy() override;

    void onBeginFrame() override;
    void onEndFrame() override;

    void drawFrame(ImageBuffer& result) override;
    void drawUI() override;
    void resetFrame() override;

    void *getDisplayNativeTexure() override;
    float getMaxEDR() override;
    display_output::DisplayCapabilities getOutputCapabilities() const override;

    bool isFrameValid() const override
    {
        return mFrameValid;
    }

private:
    static constexpr uint32_t kMinImageCount = 2;
    static constexpr uint64_t kFenceTimeoutNs = 5ULL * 1000ULL * 1000ULL * 1000ULL;

    bool createInstance();
    bool createSurface();
    bool selectPhysicalDevice();
    bool createDevice();
    bool createDescriptorPool();
    bool createSwapchain(int width, int height);
    bool initializePresentation(int width, int height);
    bool createCompositionRenderPass();
    bool createCompositionResources(uint32_t width, uint32_t height);
    bool createFinalOutputResources();
    bool createFinalFramebuffers();
    bool loadShader(const char *relativePath, std::vector<uint32_t>& code);
    bool selectMemoryType(uint32_t memoryTypeBits,
                          VkMemoryPropertyFlags properties,
                          uint32_t *memoryTypeIndex) const;
    void destroyCompositionResources();
    void destroyFinalFramebuffers();
    void destroyFinalOutputResources();
    void destroyPresentation();
    bool refreshOutputPolicy();
    bool enumeratePresentModes();
    bool acquireFrame();
    bool recordFrame();
    bool presentFrame();
    bool ensureInterop(uint32_t width, uint32_t height);
    bool createInteropViews();
    bool prepareInteropFrame(ImageBuffer& result);
    void destroyInterop();
    void applyHdrMetadata();
    void refreshPlatformDisplayState();
    void logOutputCapabilities() const;
    GLFWmonitor *currentOutputMonitor() const;
    display_output::OutputMode requestedOutputMode() const;

    VkInstance mInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    VkQueue mQueue = VK_NULL_HANDLE;
    VkSurfaceKHR mSurface = VK_NULL_HANDLE;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    uint32_t mQueueFamily = UINT32_MAX;
    VkRenderPass mCompositionRenderPass = VK_NULL_HANDLE;
    VkImage mCompositionImage = VK_NULL_HANDLE;
    VkDeviceMemory mCompositionMemory = VK_NULL_HANDLE;
    VkImageView mCompositionImageView = VK_NULL_HANDLE;
    VkFramebuffer mCompositionFramebuffer = VK_NULL_HANDLE;
    VkSampler mCompositionSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout mFinalDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet mFinalDescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout mFinalPipelineLayout = VK_NULL_HANDLE;
    VkRenderPass mFinalRenderPass = VK_NULL_HANDLE;
    VkPipeline mFinalPipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> mFinalFramebuffers;
    std::vector<uint32_t> mImGuiFragmentShader;
    std::vector<uint32_t> mFinalVertexShader;
    std::vector<uint32_t> mFinalFragmentShader;
    VkFormat mFinalFormat = VK_FORMAT_UNDEFINED;
    display_output::SurfaceEncoding mFinalEncoding =
        display_output::SurfaceEncoding::SDR;
    uint32_t mCompositionWidth = 0;
    uint32_t mCompositionHeight = 0;
    CudaVulkanInterop mInterop;
    std::array<VkImageView, CudaVulkanInterop::SlotCount> mInteropImageViews{};
    std::array<VkDescriptorSet, CudaVulkanInterop::SlotCount> mInteropDescriptors{};
    uint32_t mInteropWidth = 0;
    uint32_t mInteropHeight = 0;
    size_t mInteropSlot = 0;
    uint64_t mInteropReadyValue = 0;
    uint64_t mInteropRetireValue = 0;
    uint64_t mPresentedFrameSerial = 0;
    bool mInteropSubmissionPending = false;

    ImGui_ImplVulkanH_Window *mWindowData = nullptr;
    void *mViewportTexture = nullptr;
    PFN_vkSetHdrMetadataEXT mSetHdrMetadata = nullptr;
    GLFWmonitor *mOutputMonitor = nullptr;
    VkSurfaceFormatKHR mSelectedSurfaceFormat = {};
    std::vector<VkPresentModeKHR> mAvailablePresentModes;
    display_output::DisplayCapabilities mOutputCapabilities;

    uint32_t mAppliedOutputMode = UINT32_MAX;
    float mAppliedPaperWhiteNits = 0.0f;
    float mAppliedPeakNits = 0.0f;
    bool mAppliedVrrEnabled = false;

    bool mImGuiInitialized = false;
    bool mImGuiContextCreated = false;
    bool mGlfwBackendInitialized = false;
    bool mFrameValid = false;
    bool mFrameRecorded = false;
    bool mSwapchainRebuild = false;
    bool mForcedHdrFallbackWarned = false;
    bool mDestroyed = false;
};

} // namespace oka
