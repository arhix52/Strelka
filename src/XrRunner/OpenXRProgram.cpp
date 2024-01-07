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

using namespace oka;

namespace Side
{
const int LEFT = 0;
const int RIGHT = 1;
const int COUNT = 2;
}; // namespace Side

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
    }

    void InitializeSession() override
    {
    }
    void CreateSwapchains() override
    {
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
    XrInstance m_instance{ XR_NULL_HANDLE };
    XrSession m_session{ XR_NULL_HANDLE };
    XrSpace m_appSpace{ XR_NULL_HANDLE };
    XrSystemId m_systemId{ XR_NULL_SYSTEM_ID };

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
            STRELKA_ERROR("Failed to create Vulkan device: {}", (int) result);
            return { VK_NULL_HANDLE, VK_NULL_HANDLE };
        }

        VkQueue queue;
        vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &queue);

        return { device, queue };
    }
};

std::shared_ptr<IOpenXrProgram> oka::CreateOpenXrProgram()
{
    return std::make_shared<OpenXrProgram>();
}
