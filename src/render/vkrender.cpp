#include "vkrender.h"

//#include "debugUtils.h"

#include <chrono>
#include <set>
#include <string>

// profiler
//#include "Tracy.hpp"

[[maybe_unused]] static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                                              const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                                              const VkAllocationCallbacks* pAllocator,
                                                              VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

[[maybe_unused]] static void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                                           VkDebugUtilsMessengerEXT debugMessenger,
                                                           const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

using namespace oka;

void VkRender::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    if (enableValidationLayers)
    {
        // oka::debug::setupDebug(mInstance);
    }
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        mSharedCtx.mFramesData[i].cmdPool = createCommandPool();
    }

    initSharedContext();
    createCommandBuffers();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        mUploadBuffer[i] = mSharedCtx.mResManager->createBuffer(
            MAX_UPLOAD_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
}

void oka::VkRender::initSharedContext()
{
    mSharedCtx.mDevice = mDevice;
    mSharedCtx.mDescriptorPool = createDescriptorPool();
    mSharedCtx.mShaderManager = new ShaderManager;
    mSharedCtx.mResManager =
        new oka::ResourceManager(mDevice, mPhysicalDevice, mInstance, getCurrentFrameData().cmdPool, mGraphicsQueue);
    mSharedCtx.mTextureManager = new oka::TextureManager(mDevice, mPhysicalDevice, mSharedCtx.mResManager);
    mSharedCtx.depthFormat = findDepthFormat();
}

void VkRender::cleanup()
{
    vkDestroyDescriptorPool(mDevice, mSharedCtx.mDescriptorPool, nullptr);

    mSharedCtx.mTextureManager->textureDestroy();
    mSharedCtx.mTextureManager->delTexturesFromQueue();

    for (FrameData& fd : mSharedCtx.mFramesData)
    {
        vkDestroyCommandPool(mDevice, fd.cmdPool, nullptr);
    }

    for (oka::Buffer* buff : mUploadBuffer)
    {
        if (buff)
        {
            mSharedCtx.mResManager->destroyBuffer(buff);
        }
    }

    delete mSharedCtx.mResManager;

    vkDestroyDevice(mDevice, nullptr);

    if (enableValidationLayers)
    {
        DestroyDebugUtilsMessengerEXT(mInstance, debugMessenger, nullptr);
    }

    vkDestroyInstance(mInstance, nullptr);
}

void oka::VkRender::createSurface()
{
}

void VkRender::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport())
    {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "STRELKA";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NoErrorVulkan Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else
    {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &mInstance) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create instance!");
    }
}

void VkRender::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void VkRender::setupDebugMessenger()
{
    if (!enableValidationLayers)
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(mInstance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void VkRender::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

    for (const VkPhysicalDevice& device : devices)
    {
        if (isDeviceSuitable(device))
        {
            mPhysicalDevice = device;
            //break;
            auto props = VkPhysicalDeviceProperties{};
            vkGetPhysicalDeviceProperties(device, &props);
            if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                break; // discrete gpu is preferred
            }
        }
    }

    if (mPhysicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VkRender::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    {
        VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
        indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
        indexingFeatures.pNext = nullptr;

        VkPhysicalDeviceFeatures2 deviceFeatures{};
        deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures.pNext = &indexingFeatures;
        vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &deviceFeatures);

        if (indexingFeatures.descriptorBindingPartiallyBound && indexingFeatures.runtimeDescriptorArray)
        {
            // all set to use unbound arrays of textures
        }
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;

    VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
    indexingFeatures.pNext = nullptr;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    indexingFeatures.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    VkPhysicalDeviceRobustness2FeaturesEXT robustnessFeatures{};
    robustnessFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
    robustnessFeatures.nullDescriptor = VK_TRUE;
    robustnessFeatures.pNext = &indexingFeatures;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &robustnessFeatures;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(mDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = mDeviceExtensions.data();

    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(mDevice, indices.graphicsFamily.value(), 0, &mGraphicsQueue);
}

VkCommandPool VkRender::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(mPhysicalDevice);
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &pool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics command pool!");
    }
    return pool;
}

VkFormat VkRender::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                       VkImageTiling tiling,
                                       VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(mPhysicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VkFormat VkRender::findDepthFormat()
{
    return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
                               VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

bool VkRender::hasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkDescriptorPool VkRender::createDescriptorPool()
{
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10000 },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                                          { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                                          { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                                          { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = sizeof(pool_sizes) / sizeof(VkDescriptorPoolSize);
    poolInfo.pPoolSizes = pool_sizes;
    poolInfo.maxSets = 1000 * poolInfo.poolSizeCount;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &pool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    return pool;
}

void VkRender::recordBarrier(VkCommandBuffer& cmd,
                             VkImage image,
                             VkImageLayout oldLayout,
                             VkImageLayout newLayout,
                             VkAccessFlags srcAccess,
                             VkAccessFlags dstAccess,
                             VkPipelineStageFlags sourceStage,
                             VkPipelineStageFlags destinationStage,
                             VkImageAspectFlags aspectMask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VkRender::recordBufferBarrier(VkCommandBuffer& cmd,
                                   Buffer* buff,
                                   VkAccessFlags srcAccess,
                                   VkAccessFlags dstAccess,
                                   VkPipelineStageFlags sourceStage,
                                   VkPipelineStageFlags destinationStage)
{
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer = mSharedCtx.mResManager->getVkBuffer(buff);
    barrier.offset = 0;
    barrier.size = mSharedCtx.mResManager->getSize(buff);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void VkRender::recordImageBarrier(VkCommandBuffer& cmd,
                                  Image* image,
                                  VkImageLayout newLayout,
                                  VkAccessFlags srcAccess,
                                  VkAccessFlags dstAccess,
                                  VkPipelineStageFlags sourceStage,
                                  VkPipelineStageFlags destinationStage,
                                  VkImageAspectFlags aspectMask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = mSharedCtx.mResManager->getVkImage(image);
    barrier.oldLayout = mSharedCtx.mResManager->getImageLayout(image);
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    mSharedCtx.mResManager->setImageLayout(image, newLayout);

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VkRender::createCommandBuffers()
{
    for (FrameData& fd : mSharedCtx.mFramesData)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = fd.cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(mDevice, &allocInfo, &fd.cmdBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
}

bool VkRender::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && supportedFeatures.samplerAnisotropy;
}

bool VkRender::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(mDeviceExtensions.begin(), mDeviceExtensions.end());

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices VkRender::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = i;
        }

        {
            indices.presentFamily = i;
        }

        if (indices.isComplete())
        {
            break;
        }

        i++;
    }

    return indices;
}

std::vector<const char*> VkRender::getRequiredExtensions()
{
    std::vector<const char*> extensions;
    if (enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif // __APPLE__
    return extensions;
}

bool VkRender::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}
