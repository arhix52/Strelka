#pragma once

#define GLM_FORCE_SILENT_WARNINGS
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "common.h"

#include <resourcemanager/resourcemanager.h>
#include <shadermanager/ShaderManager.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };


#ifdef NDEBUG
const bool enableValidationLayers = true; // Enable validation in release
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

namespace oka
{

class VkRender
{
public:
    void initVulkan();
    void cleanup();
    void drawFrame(const uint8_t* outPixels);
    SharedContext& getSharedContext()
    {
        return mSharedCtx;
    }

protected:
    VkInstance mInstance;
    VkDebugUtilsMessengerEXT debugMessenger;

    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice;

    VkQueue mGraphicsQueue;
    VkQueue mPresentQueue;

    SharedContext mSharedCtx;

    virtual void createSurface();
    void initSharedContext();

    std::vector<const char*> mDeviceExtensions = {
// VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef __APPLE__
        "VK_KHR_portability_subset", "VK_KHR_maintenance3", "VK_EXT_descriptor_indexing"
#endif
    };

    static constexpr size_t MAX_UPLOAD_SIZE = 1 << 24; // 16mb
    Buffer* mUploadBuffer[MAX_FRAMES_IN_FLIGHT] = { nullptr, nullptr, nullptr };

    FrameData& getFrameData(uint32_t idx)
    {
        return mSharedCtx.mFramesData[idx % MAX_FRAMES_IN_FLIGHT];
    }

    std::array<bool, MAX_FRAMES_IN_FLIGHT> needImageViewUpdate = { false, false, false };

    //size_t mFrameNumber = 0;
    double msPerFrame = 33.33; // fps counter

    void createInstance();

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

    void setupDebugMessenger();

    void pickPhysicalDevice();

    virtual void createLogicalDevice();

    VkCommandPool createCommandPool();

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags features);

    VkFormat findDepthFormat();

    bool hasStencilComponent(VkFormat format);

    VkDescriptorPool createDescriptorPool();

    void recordBarrier(VkCommandBuffer& cmd,
                       VkImage image,
                       VkImageLayout oldLayout,
                       VkImageLayout newLayout,
                       VkAccessFlags srcAccess,
                       VkAccessFlags dstAccess,
                       VkPipelineStageFlags sourceStage,
                       VkPipelineStageFlags destinationStage,
                       VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
    void recordBufferBarrier(VkCommandBuffer& cmd,
                             Buffer* buff,
                             VkAccessFlags srcAccess,
                             VkAccessFlags dstAccess,
                             VkPipelineStageFlags sourceStage,
                             VkPipelineStageFlags destinationStage);
    void recordImageBarrier(VkCommandBuffer& cmd,
                            Image* image,
                            VkImageLayout newLayout,
                            VkAccessFlags srcAccess,
                            VkAccessFlags dstAccess,
                            VkPipelineStageFlags sourceStage,
                            VkPipelineStageFlags destinationStage,
                            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);

    void createCommandBuffers();

    bool isDeviceSuitable(VkPhysicalDevice device);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    virtual QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    virtual std::vector<const char*> getRequiredExtensions();

    bool checkValidationLayerSupport();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        [[maybe_unused]] void* pUserData)
    {
        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            // std::cout << "Warning: " << pCallbackData->messageIdNumber << ":" << pCallbackData->pMessageIdName << ":"
            // << pCallbackData->pMessage << std::endl;
            std::cout << "Warning: " << pCallbackData->pMessage << std::endl;
        }
        else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            // std::cerr << "Error: " << pCallbackData->messageIdNumber << ":" << pCallbackData->pMessageIdName << ":"
            // << pCallbackData->pMessage << std::endl;
            std::cout << "Error: " << pCallbackData->pMessage << std::endl;
        }
        else
        {
            std::cerr << "Validation: " << pCallbackData->messageIdNumber << ":" << pCallbackData->pMessageIdName << ":"
                      << pCallbackData->pMessage << std::endl;
        }
        return VK_FALSE;
    }

public:
    VkPhysicalDevice getPhysicalDevice()
    {
        return mPhysicalDevice;
    }
    QueueFamilyIndices getQueueFamilies(VkPhysicalDevice mdevice)
    {
        return findQueueFamilies(mdevice);
    }
    VkDevice getDevice()
    {
        return mDevice;
    }
    VkInstance getInstance()
    {
        return mInstance;
    }
    VkQueue getGraphicsQueue()
    {
        return mGraphicsQueue;
    }

    size_t getCurrentFrameIndex()
    {
        return mSharedCtx.mFrameNumber % MAX_FRAMES_IN_FLIGHT;
    }
    FrameData& getCurrentFrameData()
    {
        return mSharedCtx.mFramesData[mSharedCtx.mFrameNumber % MAX_FRAMES_IN_FLIGHT];
    }
};

} // namespace oka
