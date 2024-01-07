#include "OpenXRProgram.h"
#include "OpenXRHelpers.h"

#include <vulkan/vulkan.h>

#include <openxr/openxr_platform.h>

#include <set>
#include <map>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include <fstream>

using namespace oka;

namespace Side
{
const int LEFT = 0;
const int RIGHT = 1;
const int COUNT = 2;
}; // namespace Side


struct SwapchainImage
{
    SwapchainImage(VkPhysicalDevice physicalDevice,
                   VkDevice device,
                   VkRenderPass renderPass,
                   VkCommandPool commandPool,
                   VkDescriptorPool descriptorPool,
                   VkDescriptorSetLayout descriptorSetLayout,
                   const Swapchain* swapchain,
                   XrSwapchainImageVulkanKHR image)
        : device(device), commandPool(commandPool), descriptorPool(descriptorPool), image(image)
    {
        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.image = image.image;
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = swapchain->format;
        imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        VkResult result = vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan image view: ", (int)result);
        }

        VkFramebufferCreateInfo framebufferCreateInfo{};
        framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferCreateInfo.renderPass = renderPass;
        framebufferCreateInfo.attachmentCount = 1;
        framebufferCreateInfo.pAttachments = &imageView;
        framebufferCreateInfo.width = swapchain->width;
        framebufferCreateInfo.height = swapchain->height;
        framebufferCreateInfo.layers = 1;

        result = vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &framebuffer);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan framebuffer: ", (int)result);
        }

        VkBufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        static const size_t bufferSize = sizeof(float) * 4 * 4 * 3;
        createInfo.size = bufferSize;
        createInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        result = vkCreateBuffer(device, &createInfo, nullptr, &buffer);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan buffer: ", (int)result);
        }

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);

        VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        VkPhysicalDeviceMemoryProperties properties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

        uint32_t memoryTypeIndex = 0;

        for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
        {
            if (!(requirements.memoryTypeBits & (1 << i)))
            {
                continue;
            }

            if ((properties.memoryTypes[i].propertyFlags & flags) != flags)
            {
                continue;
            }

            memoryTypeIndex = i;
            break;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        result = vkAllocateMemory(device, &allocateInfo, nullptr, &memory);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to allocate Vulkan memory: ", (int)result);
        }

        vkBindBufferMemory(device, buffer, memory, 0);

        VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to allocate Vulkan command buffers: ", (int)result);
        }

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.descriptorPool = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        result = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to allocate Vulkan command descriptor sets: ", (int)result);
        }

        VkDescriptorBufferInfo descriptorBufferInfo{};
        descriptorBufferInfo.buffer = buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.pBufferInfo = &descriptorBufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }

    ~SwapchainImage()
    {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
    }

    XrSwapchainImageVulkanKHR image;
    VkImageView imageView;
    VkFramebuffer framebuffer;
    VkDeviceMemory memory;
    VkBuffer buffer;
    VkCommandBuffer commandBuffer;
    VkDescriptorSet descriptorSet;

private:
    VkDevice device;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
};

struct OpenXrProgram : IOpenXrProgram
{
    struct InputState
    {
        XrActionSet actionSet{ XR_NULL_HANDLE };
        XrAction grabAction{ XR_NULL_HANDLE };
        XrAction poseAction{ XR_NULL_HANDLE };
        XrAction vibrateAction{ XR_NULL_HANDLE };
        XrAction quitAction{ XR_NULL_HANDLE };
        std::array<XrPath, Side::COUNT> handSubactionPath;
        std::array<XrSpace, Side::COUNT> handSpace;
        std::array<float, Side::COUNT> handScale = { { 1.0f, 1.0f } };
        std::array<XrBool32, Side::COUNT> handActive;
    };

    OpenXrProgram()
        : m_acceptableBlendModes{ XR_ENVIRONMENT_BLEND_MODE_OPAQUE, XR_ENVIRONMENT_BLEND_MODE_ADDITIVE,
                                  XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND }
    {
    }
    ~OpenXrProgram() override
    {
    }

    void CreateInstance() override
    {
        static const char* const applicationName = "OpenXR Strelka";
        static const unsigned int majorVersion = 0;
        static const unsigned int minorVersion = 1;
        static const unsigned int patchVersion = 0;
        static const char* const extensionNames[] = { "XR_KHR_vulkan_enable", "XR_KHR_vulkan_enable2" };

        XrInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.type = XR_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.createFlags = 0;
        strcpy(instanceCreateInfo.applicationInfo.applicationName, applicationName);
        instanceCreateInfo.applicationInfo.applicationVersion = XR_MAKE_VERSION(majorVersion, minorVersion, patchVersion);
        strcpy(instanceCreateInfo.applicationInfo.engineName, applicationName);
        instanceCreateInfo.applicationInfo.engineVersion = XR_MAKE_VERSION(majorVersion, minorVersion, patchVersion);
        instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        instanceCreateInfo.enabledApiLayerCount = 0;
        instanceCreateInfo.enabledApiLayerNames = nullptr;
        instanceCreateInfo.enabledExtensionCount = sizeof(extensionNames) / sizeof(const char*);
        instanceCreateInfo.enabledExtensionNames = extensionNames;

        OPENXR_CHECK(xrCreateInstance(&instanceCreateInfo, &m_instance), "Failed to create OpenXR instance");

        STRELKA_DEBUG("OpenXr instance created");
    }

    void InitializeSystem() override
    {
        assert(m_instance != XR_NULL_HANDLE);
        assert(m_systemId == XR_NULL_SYSTEM_ID);

        XrSystemGetInfo systemInfo{ XR_TYPE_SYSTEM_GET_INFO };
        systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY; // TODO: need to be in settings
        OPENXR_CHECK(xrGetSystem(m_instance, &systemInfo, &m_systemId), "Failed to get system!");

        STRELKA_INFO("Using system {} for form factor {}", m_systemId, to_string(systemInfo.formFactor));
        assert(m_instance != XR_NULL_HANDLE);
        assert(m_systemId != XR_NULL_SYSTEM_ID);
    }

    void InitializeDevice() override
    {
        XrGraphicsRequirementsVulkanKHR graphicsRequirements;
        std::set<std::string> instanceExtensions;
        std::tie(graphicsRequirements, instanceExtensions) = getVulkanInstanceRequirements(m_instance, m_systemId);
        VkInstance vulkanInstance = createVulkanInstance(graphicsRequirements, instanceExtensions);
        VkDebugUtilsMessengerEXT vulkanDebugMessenger = createVulkanDebugMessenger(vulkanInstance);
        VkPhysicalDevice physicalDevice;
        std::set<std::string> deviceExtensions;
        std::tie(physicalDevice, deviceExtensions) = getVulkanDeviceRequirements(m_instance, m_systemId, vulkanInstance);
        int32_t graphicsQueueFamilyIndex = getDeviceQueueFamily(physicalDevice);
        VkDevice device;
        VkQueue queue;
        std::tie(device, queue) = createDevice(physicalDevice, graphicsQueueFamilyIndex, deviceExtensions);
        m_renderPass = createRenderPass(device);
        m_commandPool = createCommandPool(device, graphicsQueueFamilyIndex);
        m_descriptorPool = createDescriptorPool(device);
        m_descriptorSetLayout = createDescriptorSetLayout(device);
        VkShaderModule vertexShader = createShader(device, "./src/XrRunner/vert.spv");
        VkShaderModule fragmentShader = createShader(device, "./src/XrRunner/frag.spv");
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        std::tie(pipelineLayout, pipeline) =
            createPipeline(device, m_renderPass, m_descriptorSetLayout, vertexShader, fragmentShader);

        m_graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR;
        m_graphicsBinding.instance = vulkanInstance;
        m_graphicsBinding.physicalDevice = physicalDevice;
        m_graphicsBinding.device = device;
        m_graphicsBinding.queueFamilyIndex = graphicsQueueFamilyIndex;
        m_graphicsBinding.queueIndex = 0;
    }

    void InitializeSession() override
    {
        XrSessionCreateInfo sessionCreateInfo{};
        sessionCreateInfo.type = XR_TYPE_SESSION_CREATE_INFO;
        sessionCreateInfo.next = &m_graphicsBinding;
        sessionCreateInfo.createFlags = 0;
        sessionCreateInfo.systemId = m_systemId;

        XrResult result = xrCreateSession(m_instance, &sessionCreateInfo, &m_session);
        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to create OpenXR session: {}", to_string(result));
            return;
        }

        XrReferenceSpaceCreateInfo spaceCreateInfo{};
        spaceCreateInfo.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
        spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        spaceCreateInfo.poseInReferenceSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } };

        result = xrCreateReferenceSpace(m_session, &spaceCreateInfo, &m_appSpace);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to create space: {}", to_string(result));
            return;
        }
    }

    void CreateSwapchains() override
    {
        Swapchain* swapchains[eyeCount];
        std::tie(swapchains[0], swapchains[1]) = createSwapchains(m_instance, m_systemId, m_session);

        std::vector<XrSwapchainImageVulkanKHR> swapchainImages[eyeCount];

        for (size_t i = 0; i < eyeCount; i++)
        {
            swapchainImages[i] = getSwapchainImages(swapchains[i]->swapchain);
        }

        std::vector<SwapchainImage*> wrappedSwapchainImages[eyeCount];

        for (size_t i = 0; i < eyeCount; i++)
        {
            wrappedSwapchainImages[i] = std::vector<SwapchainImage*>(swapchainImages[i].size(), nullptr);

            for (size_t j = 0; j < wrappedSwapchainImages[i].size(); j++)
            {
                wrappedSwapchainImages[i][j] = new SwapchainImage(
                    m_graphicsBinding.physicalDevice, m_graphicsBinding.device, m_renderPass, m_commandPool,
                    m_descriptorPool, m_descriptorSetLayout, swapchains[i], swapchainImages[i][j]);
            }
        }
    }

    void PollEvents(bool* exitRenderLoop, bool* requestRestart) override
    {
    }
    bool IsSessionRunning() const override
    {
        return false;
    }
    bool IsSessionFocused() const override

    {
        return false;
    }
    void PollActions()
    {
    }
    void RenderFrame()
    {
    }
    XrEnvironmentBlendMode GetPreferredBlendMode() const override
    {
        return XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    }

private:
    static const size_t eyeCount = 2;
    XrInstance m_instance{ XR_NULL_HANDLE };
    XrSession m_session{ XR_NULL_HANDLE };
    XrSpace m_appSpace{ XR_NULL_HANDLE };
    XrSystemId m_systemId{ XR_NULL_SYSTEM_ID };

    XrGraphicsBindingVulkanKHR m_graphicsBinding{};

    VkRenderPass m_renderPass;
    VkCommandPool m_commandPool;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSetLayout m_descriptorSetLayout;

    std::vector<XrViewConfigurationView> m_configViews;
    std::vector<Swapchain> m_swapchains;
    std::map<XrSwapchain, std::vector<XrSwapchainImageBaseHeader*>> m_swapchainImages;
    std::vector<XrView> m_views;
    int64_t m_colorSwapchainFormat{ -1 };

    std::vector<XrSpace> m_visualizedSpaces;

    // Application's current lifecycle state according to the runtime
    XrSessionState m_sessionState{ XR_SESSION_STATE_UNKNOWN };
    bool m_sessionRunning{ false };

    XrEventDataBuffer m_eventDataBuffer;
    InputState m_input;

    const std::set<XrEnvironmentBlendMode> m_acceptableBlendModes;


        PFN_xrVoidFunction getXRFunction(XrInstance instance, const char* name)
    {
        PFN_xrVoidFunction func;

        XrResult result = xrGetInstanceProcAddr(instance, name, &func);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load OpenXR extension function '{}': {}", name, to_string(result));
            return nullptr;
        }

        return func;
    }

    std::tuple<XrGraphicsRequirementsVulkanKHR, std::set<std::string>> getVulkanInstanceRequirements(XrInstance instance,
                                                                                                     XrSystemId system)
    {
        auto xrGetVulkanGraphicsRequirementsKHR =
            (PFN_xrGetVulkanGraphicsRequirementsKHR)getXRFunction(instance, "xrGetVulkanGraphicsRequirementsKHR");
        auto xrGetVulkanInstanceExtensionsKHR =
            (PFN_xrGetVulkanInstanceExtensionsKHR)getXRFunction(instance, "xrGetVulkanInstanceExtensionsKHR");

        XrGraphicsRequirementsVulkanKHR graphicsRequirements{};
        graphicsRequirements.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR;

        XrResult result = xrGetVulkanGraphicsRequirementsKHR(instance, system, &graphicsRequirements);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan graphics requirements: {}", to_string(result));
            return { graphicsRequirements, {} };
        }

        uint32_t instanceExtensionsSize;

        result = xrGetVulkanInstanceExtensionsKHR(instance, system, 0, &instanceExtensionsSize, nullptr);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan instance extensions: {}", to_string(result));
            return { graphicsRequirements, {} };
        }

        char* instanceExtensionsData = new char[instanceExtensionsSize];

        result = xrGetVulkanInstanceExtensionsKHR(
            instance, system, instanceExtensionsSize, &instanceExtensionsSize, instanceExtensionsData);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan instance extensions: {}", to_string(result));
            return { graphicsRequirements, {} };
        }

        std::set<std::string> instanceExtensions;

        uint32_t last = 0;
        for (uint32_t i = 0; i <= instanceExtensionsSize; i++)
        {
            if (i == instanceExtensionsSize || instanceExtensionsData[i] == ' ')
            {
                instanceExtensions.insert(std::string(instanceExtensionsData + last, i - last));
                last = i + 1;
            }
        }

        delete[] instanceExtensionsData;

        return { graphicsRequirements, instanceExtensions };
    }

    std::tuple<VkPhysicalDevice, std::set<std::string>> getVulkanDeviceRequirements(XrInstance instance,
                                                                                    XrSystemId system,
                                                                                    VkInstance vulkanInstance)
    {
        auto xrGetVulkanGraphicsDeviceKHR =
            (PFN_xrGetVulkanGraphicsDeviceKHR)getXRFunction(instance, "xrGetVulkanGraphicsDeviceKHR");
        auto xrGetVulkanDeviceExtensionsKHR =
            (PFN_xrGetVulkanDeviceExtensionsKHR)getXRFunction(instance, "xrGetVulkanDeviceExtensionsKHR");

        VkPhysicalDevice physicalDevice;

        XrResult result = xrGetVulkanGraphicsDeviceKHR(instance, system, vulkanInstance, &physicalDevice);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan graphics device: {}", to_string(result));
            return { VK_NULL_HANDLE, {} };
        }

        uint32_t deviceExtensionsSize;

        result = xrGetVulkanDeviceExtensionsKHR(instance, system, 0, &deviceExtensionsSize, nullptr);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan device extensions: {}", to_string(result));
            return { VK_NULL_HANDLE, {} };
        }

        char* deviceExtensionsData = new char[deviceExtensionsSize];

        result = xrGetVulkanDeviceExtensionsKHR(
            instance, system, deviceExtensionsSize, &deviceExtensionsSize, deviceExtensionsData);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to get Vulkan device extensions: {}", to_string(result));
            return { VK_NULL_HANDLE, {} };
        }

        std::set<std::string> deviceExtensions;

        uint32_t last = 0;
        for (uint32_t i = 0; i <= deviceExtensionsSize; i++)
        {
            if (i == deviceExtensionsSize || deviceExtensionsData[i] == ' ')
            {
                deviceExtensions.insert(std::string(deviceExtensionsData + last, i - last));
                last = i + 1;
            }
        }

        delete[] deviceExtensionsData;

        return { physicalDevice, deviceExtensions };
    }

    VkInstance createVulkanInstance(XrGraphicsRequirementsVulkanKHR graphicsRequirements,
                                    std::set<std::string> instanceExtensions)
    {
        VkInstance instance;

        size_t extensionCount = 1 + instanceExtensions.size();
        const char** extensionNames = new const char*[extensionCount];

        size_t i = 0;
        static const char* const vulkanExtensionNames[] = { "VK_EXT_debug_utils" };
        extensionNames[i] = vulkanExtensionNames[0];
        i++;

        for (const std::string& instanceExtension : instanceExtensions)
        {
            extensionNames[i] = instanceExtension.c_str();
            i++;
        }

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "XR Strelka render";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        applicationInfo.pEngineName = "XR Vulkan render";
        applicationInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        applicationInfo.apiVersion =
            VK_MAKE_API_VERSION(0, XR_VERSION_MAJOR(graphicsRequirements.minApiVersionSupported),
                                XR_VERSION_MINOR(graphicsRequirements.minApiVersionSupported), 0);


        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensionNames;
        createInfo.enabledLayerCount = 1;
        static const char* const vulkanLayerNames[] = { "VK_LAYER_KHRONOS_validation" };
        createInfo.ppEnabledLayerNames = vulkanLayerNames;

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        delete[] extensionNames;

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan instance: {}", (int)result);
            return VK_NULL_HANDLE;
        }

        return instance;
    }

    static PFN_vkVoidFunction getVKFunction(VkInstance instance, const char* name)
    {
        PFN_vkVoidFunction func = vkGetInstanceProcAddr(instance, name);

        if (!func)
        {
            STRELKA_ERROR("Failed to load VUlkan extension function {}", name);
            return nullptr;
        }

        return func;
    }

    static VkBool32 handleVKError(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                  VkDebugUtilsMessageTypeFlagsEXT type,
                                  const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                  void* userData)
    {
        switch (severity)
        {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            STRELKA_DEBUG("{}", callbackData->pMessage);
            break;
        default:
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            STRELKA_INFO("{}", callbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            STRELKA_WARNING("{}", callbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            STRELKA_ERROR("{}", callbackData->pMessage);
            break;
        }
        return VK_FALSE;
    }

    VkDebugUtilsMessengerEXT createVulkanDebugMessenger(VkInstance instance)
    {
        VkDebugUtilsMessengerEXT debugMessenger;

        VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo{};
        debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugMessengerCreateInfo.pfnUserCallback = handleVKError;

        auto vkCreateDebugUtilsMessengerEXT =
            (PFN_vkCreateDebugUtilsMessengerEXT)getVKFunction(instance, "vkCreateDebugUtilsMessengerEXT");

        VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &debugMessengerCreateInfo, nullptr, &debugMessenger);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan debug messenger: ", (int)result);
            return VK_NULL_HANDLE;
        }

        return debugMessenger;
    }

    int32_t getDeviceQueueFamily(VkPhysicalDevice physicalDevice)
    {
        int32_t graphicsQueueFamilyIndex = -1;

        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        for (int32_t i = 0; i < queueFamilyCount; i++)
        {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                graphicsQueueFamilyIndex = i;
                break;
            }
        }

        if (graphicsQueueFamilyIndex == -1)
        {
            STRELKA_ERROR("No graphics queue found.");
            return graphicsQueueFamilyIndex;
        }

        return graphicsQueueFamilyIndex;
    }

    std::tuple<VkDevice, VkQueue> createDevice(VkPhysicalDevice physicalDevice,
                                               int32_t graphicsQueueFamilyIndex,
                                               std::set<std::string> deviceExtensions)
    {
        VkDevice device;

        size_t extensionCount = deviceExtensions.size();
        const char** extensions = new const char*[extensionCount];

        size_t i = 0;
        for (const std::string& deviceExtension : deviceExtensions)
        {
            extensions[i] = deviceExtension.c_str();
            i++;
        }

        float priority = 1;

        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &priority;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;

        VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);

        delete[] extensions;

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan device: {}", (int)result);
            return { VK_NULL_HANDLE, VK_NULL_HANDLE };
        }

        VkQueue queue;
        vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &queue);

        return { device, queue };
    }

    VkRenderPass createRenderPass(VkDevice device)
    {
        VkRenderPass renderPass;

        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_R8G8B8A8_SRGB;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference attachmentRef{};
        attachmentRef.attachment = 0;
        attachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &attachmentRef;

        VkRenderPassCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.attachmentCount = 1;
        createInfo.pAttachments = &attachment;
        createInfo.subpassCount = 1;
        createInfo.pSubpasses = &subpass;

        VkResult result = vkCreateRenderPass(device, &createInfo, nullptr, &renderPass);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan render pass: {}", (int)result);
            return VK_NULL_HANDLE;
        }

        return renderPass;
    }

    VkCommandPool createCommandPool(VkDevice device, int32_t graphicsQueueFamilyIndex)
    {
        VkCommandPool commandPool;

        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.queueFamilyIndex = graphicsQueueFamilyIndex;

        VkResult result = vkCreateCommandPool(device, &createInfo, nullptr, &commandPool);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan command pool: {}", (int)result);
            return VK_NULL_HANDLE;
        }

        return commandPool;
    }
    VkDescriptorPool createDescriptorPool(VkDevice device)
    {
        VkDescriptorPool descriptorPool;

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = 32;

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        createInfo.maxSets = 32;
        createInfo.poolSizeCount = 1;
        createInfo.pPoolSizes = &poolSize;

        VkResult result = vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan descriptor pool: {}", (int)result);
            return VK_NULL_HANDLE;
        }

        return descriptorPool;
    }
    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device)
    {
        VkDescriptorSetLayout descriptorSetLayout;

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = 1;
        createInfo.pBindings = &binding;

        VkResult result = vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &descriptorSetLayout);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan descriptor set layout: {}", (int)result);
            return VK_NULL_HANDLE;
        }

        return descriptorSetLayout;
    }

    std::string readBinaryFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
        {
            return "";
        }

        // Read the file content into a stringstream
        std::ostringstream content;
        content << file.rdbuf();

        // Close the file
        file.close();

        // Return the content as a string
        return content.str();
    }

    VkShaderModule createShader(VkDevice device, std::string path)
    {
        VkShaderModule shader;

        // std::ifstream file(path);
        // std::string source = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        std::string source = readBinaryFile(path);

        VkShaderModuleCreateInfo shaderCreateInfo{};
        shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderCreateInfo.codeSize = source.size();
        shaderCreateInfo.pCode = (const uint32_t*)source.data();

        VkResult result = vkCreateShaderModule(device, &shaderCreateInfo, nullptr, &shader);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan shader: ", (int)result);
        }

        return shader;
    }

    std::tuple<VkPipelineLayout, VkPipeline> createPipeline(VkDevice device,
                                                            VkRenderPass renderPass,
                                                            VkDescriptorSetLayout descriptorSetLayout,
                                                            VkShaderModule vertexShader,
                                                            VkShaderModule fragmentShader)
    {
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;

        VkPipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCreateInfo.setLayoutCount = 1;
        layoutCreateInfo.pSetLayouts = &descriptorSetLayout;

        VkResult result = vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &pipelineLayout);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan pipeline layout: ", (int)result);
            return { VK_NULL_HANDLE, VK_NULL_HANDLE };
        }

        VkPipelineVertexInputStateCreateInfo vertexInputStage{};
        vertexInputStage.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStage.vertexBindingDescriptionCount = 0;
        vertexInputStage.pVertexBindingDescriptions = nullptr;
        vertexInputStage.vertexAttributeDescriptionCount = 0;
        vertexInputStage.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStage{};
        inputAssemblyStage.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStage.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssemblyStage.primitiveRestartEnable = false;

        VkPipelineShaderStageCreateInfo vertexShaderStage{};
        vertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStage.module = vertexShader;
        vertexShaderStage.pName = "main";

        VkViewport viewport = { 0, 0, 1024, 1024, 0, 1 };

        VkRect2D scissor = { { 0, 0 }, { 1024, 1024 } };

        VkPipelineViewportStateCreateInfo viewportStage{};
        viewportStage.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStage.viewportCount = 1;
        viewportStage.pViewports = &viewport;
        viewportStage.scissorCount = 1;
        viewportStage.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizationStage{};
        rasterizationStage.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStage.depthClampEnable = false;
        rasterizationStage.rasterizerDiscardEnable = false;
        rasterizationStage.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStage.lineWidth = 1;
        rasterizationStage.cullMode = VK_CULL_MODE_NONE;
        rasterizationStage.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationStage.depthBiasEnable = false;
        rasterizationStage.depthBiasConstantFactor = 0;
        rasterizationStage.depthBiasClamp = 0;
        rasterizationStage.depthBiasSlopeFactor = 0;

        VkPipelineMultisampleStateCreateInfo multisampleStage{};
        multisampleStage.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStage.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampleStage.sampleShadingEnable = false;
        multisampleStage.minSampleShading = 0.25;

        VkPipelineDepthStencilStateCreateInfo depthStencilStage{};
        depthStencilStage.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilStage.depthTestEnable = true;
        depthStencilStage.depthWriteEnable = true;
        depthStencilStage.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilStage.depthBoundsTestEnable = false;
        depthStencilStage.minDepthBounds = 0;
        depthStencilStage.maxDepthBounds = 1;
        depthStencilStage.stencilTestEnable = false;

        VkPipelineShaderStageCreateInfo fragmentShaderStage{};
        fragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStage.module = fragmentShader;
        fragmentShaderStage.pName = "main";

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = true;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlendStage{};
        colorBlendStage.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStage.logicOpEnable = false;
        colorBlendStage.logicOp = VK_LOGIC_OP_COPY;
        colorBlendStage.attachmentCount = 1;
        colorBlendStage.pAttachments = &colorBlendAttachment;
        colorBlendStage.blendConstants[0] = 0;
        colorBlendStage.blendConstants[1] = 0;
        colorBlendStage.blendConstants[2] = 0;
        colorBlendStage.blendConstants[3] = 0;

        VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderStage, fragmentShaderStage };

        VkGraphicsPipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        createInfo.stageCount = 2;
        createInfo.pStages = shaderStages;
        createInfo.pVertexInputState = &vertexInputStage;
        createInfo.pInputAssemblyState = &inputAssemblyStage;
        createInfo.pTessellationState = nullptr;
        createInfo.pViewportState = &viewportStage;
        createInfo.pRasterizationState = &rasterizationStage;
        createInfo.pMultisampleState = &multisampleStage;
        createInfo.pDepthStencilState = &depthStencilStage;
        createInfo.pColorBlendState = &colorBlendStage;
        createInfo.pDynamicState = &dynamicState;
        createInfo.layout = pipelineLayout;
        createInfo.renderPass = renderPass;
        createInfo.subpass = 0;
        createInfo.basePipelineHandle = VK_NULL_HANDLE;
        createInfo.basePipelineIndex = -1;

        result = vkCreateGraphicsPipelines(device, nullptr, 1, &createInfo, nullptr, &pipeline);

        if (result != VK_SUCCESS)
        {
            STRELKA_ERROR("Failed to create Vulkan pipeline: ", (int)result);
            return { VK_NULL_HANDLE, VK_NULL_HANDLE };
        }

        return { pipelineLayout, pipeline };
    }

    std::tuple<Swapchain*, Swapchain*> createSwapchains(XrInstance instance, XrSystemId system, XrSession session)
    {
        uint32_t configViewsCount = eyeCount;
        std::vector<XrViewConfigurationView> configViews(configViewsCount, { .type = XR_TYPE_VIEW_CONFIGURATION_VIEW });

        XrResult result = xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                            configViewsCount, &configViewsCount, configViews.data());

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to enumerate view configuration views: {}", to_string(result));
            return { nullptr, nullptr };
        }

        uint32_t formatCount = 0;

        result = xrEnumerateSwapchainFormats(session, 0, &formatCount, nullptr);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to enumerate swapchain formats: {}", to_string(result));
            return { nullptr, nullptr };
        }

        std::vector<int64_t> formats(formatCount);

        result = xrEnumerateSwapchainFormats(session, formatCount, &formatCount, formats.data());

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to enumerate swapchain formats: {}", to_string(result));
            return { nullptr, nullptr };
        }

        int64_t chosenFormat = formats.front();

        for (int64_t format : formats)
        {
            if (format == VK_FORMAT_R8G8B8A8_SRGB)
            {
                chosenFormat = format;
                break;
            }
        }

        XrSwapchain swapchains[eyeCount];

        for (uint32_t i = 0; i < eyeCount; i++)
        {
            XrSwapchainCreateInfo swapchainCreateInfo{};
            swapchainCreateInfo.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
            swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            swapchainCreateInfo.format = chosenFormat;
            swapchainCreateInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
            swapchainCreateInfo.width = configViews[i].recommendedImageRectWidth;
            swapchainCreateInfo.height = configViews[i].recommendedImageRectHeight;
            swapchainCreateInfo.faceCount = 1;
            swapchainCreateInfo.arraySize = 1;
            swapchainCreateInfo.mipCount = 1;

            result = xrCreateSwapchain(session, &swapchainCreateInfo, &swapchains[i]);

            if (result != XR_SUCCESS)
            {
                STRELKA_ERROR("Failed to create swapchain: {}", to_string(result));
                return { nullptr, nullptr };
            }
        }

        return { new Swapchain(swapchains[0], (VkFormat)chosenFormat, configViews[0].recommendedImageRectWidth,
                               configViews[0].recommendedImageRectHeight),
                 new Swapchain(swapchains[1], (VkFormat)chosenFormat, configViews[1].recommendedImageRectWidth,
                               configViews[1].recommendedImageRectHeight) };
    }

    std::vector<XrSwapchainImageVulkanKHR> getSwapchainImages(XrSwapchain swapchain)
    {
        uint32_t imageCount;

        XrResult result = xrEnumerateSwapchainImages(swapchain, 0, &imageCount, nullptr);

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to enumerate swapchain images: {}", to_string(result));

            return {};
        }

        std::vector<XrSwapchainImageVulkanKHR> images(imageCount, { .type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR });

        result =
            xrEnumerateSwapchainImages(swapchain, imageCount, &imageCount, (XrSwapchainImageBaseHeader*)images.data());

        if (result != XR_SUCCESS)
        {
            STRELKA_ERROR("Failed to enumerate swapchain images:: {}", to_string(result));
            return {};
        }

        return images;
    }
};

std::shared_ptr<IOpenXrProgram> oka::CreateOpenXrProgram()
{
    return std::make_shared<OpenXrProgram>();
}
