#include "glfw_display.h"
#include "imgui_style.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <strelka/display/output_probe.h>

#include <postprocessing/Tonemappers.h>
#include <paths.h>

#include <cstdlib>
#include <filesystem>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace oka;

namespace
{
bool checkVkResult(VkResult result, const char *operation)
{
    if (result < 0)
    {
        STRELKA_ERROR("{} failed with Vulkan error {}", operation, static_cast<int>(result));
        return false;
    }
    return true;
}

void imguiCheckVkResult(VkResult result)
{
    checkVkResult(result, "ImGui Vulkan backend");
}

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char *name)
{
    size_t index = 0;

    while (index < extensions.size())
    {
        if (std::string(extensions[index].extensionName) == name)
        {
            return true;
        }
        ++index;
    }
    return false;
}

const char *presentModeName(VkPresentModeKHR mode)
{
    if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
    {
        return "IMMEDIATE";
    }
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
        return "MAILBOX";
    }
    if (mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
    {
        return "FIFO_RELAXED";
    }
    if (mode == VK_PRESENT_MODE_FIFO_KHR)
    {
        return "FIFO";
    }
    return "OTHER";
}
} // namespace

GlfwDisplay::~GlfwDisplay()
{
    GlfwDisplay::destroy();
}

void GlfwDisplay::init(int width, int height, SettingsManager *settings)
{
    ImGuiIO *io = nullptr;
    ImGui_ImplVulkan_InitInfo initInfo = {};

    mWindowWidth = width;
    mWindowHeight = height;
    mSettings = settings;

#if defined(__linux__)
    // Point libxkbcommon at the system's compose tables before GLFW loads it.
    //
    // We link Conan's libxkbcommon, which has XLOCALEDIR compiled in as its own
    // package prefix -- and that package ships no share/X11/locale at all. The
    // first keyboard event then prints
    //
    //     xkbcommon: ERROR: couldn't find a Compose file for locale "en_US.UTF-8"
    //
    // on a machine that has the file, at the distribution's path, the whole
    // time. It is not only a line of noise: without a compose table, dead keys
    // and Multi_key sequences produce nothing, so an accented character cannot
    // be typed into any of the editor's text fields.
    //
    // Only when the variable is unset, so anyone pointing it somewhere on
    // purpose keeps their choice, and only when the directory is really there.
    if (std::getenv("XLOCALEDIR") == nullptr && std::filesystem::is_directory("/usr/share/X11/locale"))
    {
        setenv("XLOCALEDIR", "/usr/share/X11/locale", 0);
    }
#endif

    if (!glfwInit())
    {
        STRELKA_FATAL("Failed to initialize GLFW");
        return;
    }
    if (!glfwVulkanSupported())
    {
        STRELKA_FATAL("GLFW did not find a Vulkan loader or Vulkan-capable device");
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    if (mWindow == nullptr)
    {
        STRELKA_FATAL("Failed to create GLFW window");
        return;
    }

    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, handleMouseMoveCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);

    if (!createInstance() || !createSurface() || !selectPhysicalDevice() || !createDevice() ||
        !createDescriptorPool() || !createSwapchain(width, height) ||
        !initializePresentation(width, height))
    {
        STRELKA_FATAL("Failed to initialize the Vulkan display");
        return;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    mImGuiContextCreated = true;
    io = &ImGui::GetIO();
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Panels, menus and sliders reachable from the pad. The ImGui GLFW backend
    // feeds it from the same joystick GLFW hands us, so this needs no wiring --
    // but it is deliberately *only* nav: the pad does not move the pointer, so
    // gizmo drags and viewport picking stay on the mouse.
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io->ConfigWindowsMoveFromTitleBarOnly = true;
    imgui_style::applyGraphiteBlue();

    if (!ImGui_ImplGlfw_InitForVulkan(mWindow, true))
    {
        STRELKA_FATAL("Failed to initialize the ImGui GLFW backend for Vulkan");
        return;
    }
    mGlfwBackendInitialized = true;

    initInfo.ApiVersion = VK_API_VERSION_1_1;
    initInfo.Instance = mInstance;
    initInfo.PhysicalDevice = mPhysicalDevice;
    initInfo.Device = mDevice;
    initInfo.QueueFamily = mQueueFamily;
    initInfo.Queue = mQueue;
    initInfo.DescriptorPool = mDescriptorPool;
    initInfo.MinImageCount = kMinImageCount;
    initInfo.ImageCount = mWindowData->ImageCount;
    initInfo.PipelineInfoMain.RenderPass = mCompositionRenderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CustomShaderFragCreateInfo.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    initInfo.CustomShaderFragCreateInfo.codeSize =
        mImGuiFragmentShader.size() * sizeof(uint32_t);
    initInfo.CustomShaderFragCreateInfo.pCode = mImGuiFragmentShader.data();
    initInfo.CheckVkResultFn = imguiCheckVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo))
    {
        STRELKA_FATAL("Failed to initialize the ImGui Vulkan backend");
        return;
    }
    mImGuiInitialized = true;
}

bool GlfwDisplay::createInstance()
{
    uint32_t extensionCount = 0;
    const char *const *glfwExtensions = nullptr;
    std::vector<const char *> extensions;
    uint32_t availableExtensionCount = 0;
    std::vector<VkExtensionProperties> availableExtensions;
    VkApplicationInfo applicationInfo = {};
    VkInstanceCreateInfo createInfo = {};
    VkResult result = VK_SUCCESS;
    uint32_t index = 0;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (glfwExtensions == nullptr || extensionCount == 0)
    {
        STRELKA_ERROR("GLFW returned no required Vulkan instance extensions");
        return false;
    }

    extensions.reserve(extensionCount);
    while (index < extensionCount)
    {
        extensions.push_back(glfwExtensions[index]);
        ++index;
    }

    result = vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr);
    if (!checkVkResult(result, "vkEnumerateInstanceExtensionProperties"))
    {
        return false;
    }
    availableExtensions.resize(availableExtensionCount);
    result = vkEnumerateInstanceExtensionProperties(
        nullptr, &availableExtensionCount, availableExtensions.data());
    if (!checkVkResult(result, "vkEnumerateInstanceExtensionProperties"))
    {
        return false;
    }
    mOutputCapabilities.output.swapchainColorspace =
        hasExtension(availableExtensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    if (mOutputCapabilities.output.swapchainColorspace)
    {
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }

    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Strelka";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    applicationInfo.pEngineName = "Strelka";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_1;

    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    result = vkCreateInstance(&createInfo, nullptr, &mInstance);
    return checkVkResult(result, "vkCreateInstance");
}

bool GlfwDisplay::createSurface()
{
    VkResult result = VK_SUCCESS;

    result = glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface);
    return checkVkResult(result, "glfwCreateWindowSurface");
}

bool GlfwDisplay::selectPhysicalDevice()
{
    uint32_t deviceCount = 0;
    std::vector<VkPhysicalDevice> devices;
    uint32_t deviceIndex = 0;
    uint32_t queueCount = 0;
    std::vector<VkQueueFamilyProperties> queueProperties;
    uint32_t queueIndex = 0;
    VkBool32 presentSupported = VK_FALSE;
    VkResult result = VK_SUCCESS;
    int cudaDeviceOrdinal = -1;
    cudaDeviceProp cudaProperties = {};
    cudaError_t cudaResult = cudaSuccess;
    VkPhysicalDeviceIDProperties idProperties = {};
    VkPhysicalDeviceProperties2 properties = {};

    if (mRender == nullptr)
    {
        STRELKA_ERROR("Vulkan display requires a renderer before device selection");
        return false;
    }
    cudaDeviceOrdinal = mRender->activeCudaDeviceOrdinal();
    if (cudaDeviceOrdinal < 0)
    {
        STRELKA_ERROR("The active renderer does not expose a CUDA device for Vulkan interop");
        return false;
    }
    cudaResult = cudaGetDeviceProperties(&cudaProperties, cudaDeviceOrdinal);
    if (cudaResult != cudaSuccess)
    {
        STRELKA_ERROR("cudaGetDeviceProperties failed while selecting Vulkan device: {}",
                      cudaGetErrorString(cudaResult));
        return false;
    }

    result = vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
    if (!checkVkResult(result, "vkEnumeratePhysicalDevices") || deviceCount == 0)
    {
        STRELKA_ERROR("No Vulkan physical devices are available");
        return false;
    }

    devices.resize(deviceCount);
    result = vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());
    if (!checkVkResult(result, "vkEnumeratePhysicalDevices"))
    {
        return false;
    }

    while (deviceIndex < deviceCount)
    {
        idProperties = {};
        properties = {};
        idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &idProperties;
        vkGetPhysicalDeviceProperties2(devices[deviceIndex], &properties);
        if (std::memcmp(idProperties.deviceUUID, cudaProperties.uuid.bytes, VK_UUID_SIZE) != 0)
        {
            ++deviceIndex;
            continue;
        }

        queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueCount, nullptr);
        queueProperties.resize(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[deviceIndex], &queueCount, queueProperties.data());
        queueIndex = 0;
        while (queueIndex < queueCount)
        {
            presentSupported = VK_FALSE;
            result = vkGetPhysicalDeviceSurfaceSupportKHR(
                devices[deviceIndex], queueIndex, mSurface, &presentSupported);
            if (!checkVkResult(result, "vkGetPhysicalDeviceSurfaceSupportKHR"))
            {
                return false;
            }
            if ((queueProperties[queueIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
                presentSupported == VK_TRUE)
            {
                mPhysicalDevice = devices[deviceIndex];
                mQueueFamily = queueIndex;
                return true;
            }
            ++queueIndex;
        }
        ++deviceIndex;
    }

    STRELKA_ERROR(
        "No Vulkan physical device matching CUDA device {} has a graphics and presentation queue",
        cudaDeviceOrdinal);
    return false;
}

bool GlfwDisplay::createDevice()
{
    float const queuePriority = 1.0f;
    uint32_t availableExtensionCount = 0;
    std::vector<VkExtensionProperties> availableExtensions;
    std::vector<const char *> deviceExtensions;
    VkDeviceQueueCreateInfo queueInfo = {};
    VkDeviceCreateInfo createInfo = {};
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {};
    VkPhysicalDeviceFeatures2 availableFeatures = {};
    VkResult result = VK_SUCCESS;

    result = vkEnumerateDeviceExtensionProperties(
        mPhysicalDevice, nullptr, &availableExtensionCount, nullptr);
    if (!checkVkResult(result, "vkEnumerateDeviceExtensionProperties"))
    {
        return false;
    }
    availableExtensions.resize(availableExtensionCount);
    result = vkEnumerateDeviceExtensionProperties(
        mPhysicalDevice, nullptr, &availableExtensionCount, availableExtensions.data());
    if (!checkVkResult(result, "vkEnumerateDeviceExtensionProperties"))
    {
        return false;
    }

    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#if defined(_WIN32)
    if (!hasExtension(availableExtensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) ||
        !hasExtension(availableExtensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME))
    {
        STRELKA_ERROR("Vulkan device lacks OPAQUE_WIN32 external interop extensions");
        return false;
    }
    deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    if (!hasExtension(availableExtensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) ||
        !hasExtension(availableExtensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME))
    {
        STRELKA_ERROR("Vulkan device lacks OPAQUE_FD external interop extensions");
        return false;
    }
    deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
    if (!hasExtension(availableExtensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME))
    {
        STRELKA_ERROR("Vulkan device lacks the timeline semaphore extension");
        return false;
    }
    deviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

    timelineFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    availableFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    availableFeatures.pNext = &timelineFeatures;
    vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &availableFeatures);
    if (timelineFeatures.timelineSemaphore != VK_TRUE)
    {
        STRELKA_ERROR("Vulkan device does not support timeline semaphores");
        return false;
    }
    timelineFeatures.timelineSemaphore = VK_TRUE;

    mOutputCapabilities.output.hdrMetadata =
        hasExtension(availableExtensions, VK_EXT_HDR_METADATA_EXTENSION_NAME);
    mOutputCapabilities.presentWait =
        hasExtension(availableExtensions, VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    mOutputCapabilities.presentId =
        hasExtension(availableExtensions, VK_KHR_PRESENT_ID_EXTENSION_NAME);
    mOutputCapabilities.displayTiming =
        hasExtension(availableExtensions, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
    if (mOutputCapabilities.output.hdrMetadata)
    {
        deviceExtensions.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME);
    }

    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = mQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &timelineFeatures;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    result = vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice);
    if (!checkVkResult(result, "vkCreateDevice"))
    {
        return false;
    }

    vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);
    if (mOutputCapabilities.output.hdrMetadata)
    {
        // vkGetDeviceProcAddr yields a generic function pointer.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        mSetHdrMetadata = reinterpret_cast<PFN_vkSetHdrMetadataEXT>(
            vkGetDeviceProcAddr(mDevice, "vkSetHdrMetadataEXT"));
        mOutputCapabilities.output.hdrMetadata = mSetHdrMetadata != nullptr;
    }
    return mQueue != VK_NULL_HANDLE;
}

bool GlfwDisplay::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 3> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
    };
    VkDescriptorPoolCreateInfo poolInfo = {};
    VkResult result = VK_SUCCESS;

    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 2048;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    result = vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool);
    return checkVkResult(result, "vkCreateDescriptorPool");
}

bool GlfwDisplay::loadShader(const char *relativePath, std::vector<uint32_t>& code)
{
    const std::string path = resolveResourcePath(relativePath);
    std::ifstream stream;
    std::streamsize size = 0;

    stream.open(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        STRELKA_ERROR("Failed to open Vulkan shader {}", path);
        return false;
    }
    size = stream.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0)
    {
        STRELKA_ERROR("Vulkan shader {} has an invalid SPIR-V size", path);
        return false;
    }
    code.resize(static_cast<size_t>(size) / sizeof(uint32_t));
    stream.seekg(0, std::ios::beg);
    // std::istream::read needs a char* view over the SPIR-V word buffer.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    if (!stream.read(reinterpret_cast<char *>(code.data()), size))
    {
        STRELKA_ERROR("Failed to read Vulkan shader {}", path);
        return false;
    }
    return true;
}

bool GlfwDisplay::selectMemoryType(uint32_t memoryTypeBits,
                                   VkMemoryPropertyFlags properties,
                                   uint32_t *memoryTypeIndex) const
{
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    uint32_t index = 0;

    if (memoryTypeIndex == nullptr)
    {
        return false;
    }
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memoryProperties);
    while (index < memoryProperties.memoryTypeCount)
    {
        if ((memoryTypeBits & (1u << index)) != 0 &&
            (memoryProperties.memoryTypes[index].propertyFlags & properties) ==
                properties)
        {
            *memoryTypeIndex = index;
            return true;
        }
        ++index;
    }
    STRELKA_ERROR("No Vulkan memory type satisfies composition image requirements");
    return false;
}

bool GlfwDisplay::createCompositionRenderPass()
{
    VkAttachmentDescription attachment = {};
    VkAttachmentReference colorReference = {};
    VkSubpassDescription subpass = {};
    std::array<VkSubpassDependency, 2> dependencies = {};
    VkRenderPassCreateInfo renderPassInfo = {};
    VkResult result = VK_SUCCESS;

    if (mCompositionRenderPass != VK_NULL_HANDLE)
    {
        return true;
    }
    attachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();
    result = vkCreateRenderPass(
        mDevice, &renderPassInfo, nullptr, &mCompositionRenderPass);
    return checkVkResult(result, "vkCreateRenderPass for linear composition");
}

void GlfwDisplay::destroyCompositionResources()
{
    if (mCompositionFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(mDevice, mCompositionFramebuffer, nullptr);
        mCompositionFramebuffer = VK_NULL_HANDLE;
    }
    if (mCompositionImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(mDevice, mCompositionImageView, nullptr);
        mCompositionImageView = VK_NULL_HANDLE;
    }
    if (mCompositionImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(mDevice, mCompositionImage, nullptr);
        mCompositionImage = VK_NULL_HANDLE;
    }
    if (mCompositionMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(mDevice, mCompositionMemory, nullptr);
        mCompositionMemory = VK_NULL_HANDLE;
    }
    mCompositionWidth = 0;
    mCompositionHeight = 0;
}

bool GlfwDisplay::createCompositionResources(uint32_t width, uint32_t height)
{
    VkImageCreateInfo imageInfo = {};
    VkMemoryRequirements memoryRequirements = {};
    VkMemoryAllocateInfo allocationInfo = {};
    VkImageViewCreateInfo viewInfo = {};
    VkFramebufferCreateInfo framebufferInfo = {};
    VkDescriptorImageInfo descriptorImage = {};
    VkWriteDescriptorSet descriptorWrite = {};
    VkResult result = VK_SUCCESS;

    if (width == mCompositionWidth && height == mCompositionHeight &&
        mCompositionFramebuffer != VK_NULL_HANDLE)
    {
        return createFinalFramebuffers();
    }
    destroyCompositionResources();

    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    result = vkCreateImage(mDevice, &imageInfo, nullptr, &mCompositionImage);
    if (!checkVkResult(result, "vkCreateImage for linear composition"))
    {
        return false;
    }

    vkGetImageMemoryRequirements(
        mDevice, mCompositionImage, &memoryRequirements);
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = memoryRequirements.size;
    if (!selectMemoryType(memoryRequirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          &allocationInfo.memoryTypeIndex))
    {
        return false;
    }
    result = vkAllocateMemory(
        mDevice, &allocationInfo, nullptr, &mCompositionMemory);
    if (!checkVkResult(result, "vkAllocateMemory for linear composition"))
    {
        return false;
    }
    result = vkBindImageMemory(
        mDevice, mCompositionImage, mCompositionMemory, 0);
    if (!checkVkResult(result, "vkBindImageMemory for linear composition"))
    {
        return false;
    }

    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = mCompositionImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    result = vkCreateImageView(
        mDevice, &viewInfo, nullptr, &mCompositionImageView);
    if (!checkVkResult(result, "vkCreateImageView for linear composition"))
    {
        return false;
    }

    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = mCompositionRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &mCompositionImageView;
    framebufferInfo.width = width;
    framebufferInfo.height = height;
    framebufferInfo.layers = 1;
    result = vkCreateFramebuffer(
        mDevice, &framebufferInfo, nullptr, &mCompositionFramebuffer);
    if (!checkVkResult(result, "vkCreateFramebuffer for linear composition"))
    {
        return false;
    }

    if (mFinalDescriptorSet != VK_NULL_HANDLE)
    {
        descriptorImage.sampler = mCompositionSampler;
        descriptorImage.imageView = mCompositionImageView;
        descriptorImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = mFinalDescriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrite.pImageInfo = &descriptorImage;
        vkUpdateDescriptorSets(mDevice, 1, &descriptorWrite, 0, nullptr);
    }
    mCompositionWidth = width;
    mCompositionHeight = height;
    return true;
}

bool GlfwDisplay::createFinalFramebuffers()
{
    VkFramebufferCreateInfo framebufferInfo = {};
    VkResult result = VK_SUCCESS;
    size_t index = 0;

    destroyFinalFramebuffers();
    if (mFinalRenderPass == VK_NULL_HANDLE || mWindowData == nullptr)
    {
        return true;
    }
    mFinalFramebuffers.resize(mWindowData->Frames.Size, VK_NULL_HANDLE);
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = mFinalRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.width = static_cast<uint32_t>(mWindowData->Width);
    framebufferInfo.height = static_cast<uint32_t>(mWindowData->Height);
    framebufferInfo.layers = 1;
    while (index < mFinalFramebuffers.size())
    {
        framebufferInfo.pAttachments =
            &mWindowData->Frames[static_cast<int>(index)].BackbufferView;
        result = vkCreateFramebuffer(
            mDevice, &framebufferInfo, nullptr, &mFinalFramebuffers[index]);
        if (!checkVkResult(result, "vkCreateFramebuffer for final output"))
        {
            return false;
        }
        ++index;
    }
    return true;
}

void GlfwDisplay::destroyFinalFramebuffers()
{
    size_t index = 0;

    while (index < mFinalFramebuffers.size())
    {
        if (mFinalFramebuffers[index] != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(mDevice, mFinalFramebuffers[index], nullptr);
        }
        ++index;
    }
    mFinalFramebuffers.clear();
}

void GlfwDisplay::destroyFinalOutputResources()
{
    destroyFinalFramebuffers();
    if (mFinalPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(mDevice, mFinalPipeline, nullptr);
        mFinalPipeline = VK_NULL_HANDLE;
    }
    if (mFinalRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(mDevice, mFinalRenderPass, nullptr);
        mFinalRenderPass = VK_NULL_HANDLE;
    }
    mFinalFormat = VK_FORMAT_UNDEFINED;
}

bool GlfwDisplay::createFinalOutputResources()
{
    VkAttachmentDescription attachment = {};
    VkAttachmentReference colorReference = {};
    VkSubpassDescription subpass = {};
    VkSubpassDependency dependency = {};
    VkRenderPassCreateInfo renderPassInfo = {};
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {};
    std::array<VkShaderModule, 2> shaderModules = {};
    VkShaderModuleCreateInfo shaderModuleInfo = {};
    VkSpecializationMapEntry specializationEntry = {};
    VkSpecializationInfo specializationInfo = {};
    uint32_t hdrOutput = 0;
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    VkPipelineViewportStateCreateInfo viewportState = {};
    VkPipelineRasterizationStateCreateInfo rasterization = {};
    VkPipelineMultisampleStateCreateInfo multisample = {};
    VkPipelineColorBlendAttachmentState blendAttachment = {};
    VkPipelineColorBlendStateCreateInfo blendState = {};
    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    VkResult result = VK_SUCCESS;
    const std::vector<uint32_t> *code = nullptr;
    size_t index = 0;

    destroyFinalOutputResources();
    attachment.format = mSelectedSurfaceFormat.format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    result = vkCreateRenderPass(
        mDevice, &renderPassInfo, nullptr, &mFinalRenderPass);
    if (!checkVkResult(result, "vkCreateRenderPass for final output"))
    {
        return false;
    }

    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    while (index < shaderModules.size())
    {
        code = index == 0 ? &mFinalVertexShader : &mFinalFragmentShader;
        shaderModuleInfo.codeSize = code->size() * sizeof(uint32_t);
        shaderModuleInfo.pCode = code->data();
        result = vkCreateShaderModule(
            mDevice, &shaderModuleInfo, nullptr, &shaderModules[index]);
        if (!checkVkResult(result, "vkCreateShaderModule for final output"))
        {
            while (index > 0)
            {
                --index;
                vkDestroyShaderModule(mDevice, shaderModules[index], nullptr);
            }
            return false;
        }
        ++index;
    }

    hdrOutput = mOutputCapabilities.surfaceEncoding ==
                        display_output::SurfaceEncoding::HDR10
                    ? 1u
                    : 0u;
    specializationEntry.constantID = 0;
    specializationEntry.offset = 0;
    specializationEntry.size = sizeof(hdrOutput);
    specializationInfo.mapEntryCount = 1;
    specializationInfo.pMapEntries = &specializationEntry;
    specializationInfo.dataSize = sizeof(hdrOutput);
    specializationInfo.pData = &hdrOutput;
    shaderStages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = shaderModules[0];
    shaderStages[0].pName = "main";
    shaderStages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = shaderModules[1];
    shaderStages[1].pName = "main";
    shaderStages[1].pSpecializationInfo = &specializationInfo;

    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    rasterization.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    multisample.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.attachmentCount = 1;
    blendState.pAttachments = &blendAttachment;
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount =
        static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &blendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = mFinalPipelineLayout;
    pipelineInfo.renderPass = mFinalRenderPass;
    pipelineInfo.subpass = 0;
    result = vkCreateGraphicsPipelines(
        mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mFinalPipeline);
    index = 0;
    while (index < shaderModules.size())
    {
        vkDestroyShaderModule(mDevice, shaderModules[index], nullptr);
        ++index;
    }
    if (!checkVkResult(result, "vkCreateGraphicsPipelines for final output"))
    {
        return false;
    }
    mFinalFormat = mSelectedSurfaceFormat.format;
    mFinalEncoding = mOutputCapabilities.surfaceEncoding;
    return createFinalFramebuffers();
}

bool GlfwDisplay::initializePresentation(int width, int height)
{
    VkSamplerCreateInfo samplerInfo = {};
    VkDescriptorSetLayoutBinding descriptorBinding = {};
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {};
    VkDescriptorSetAllocateInfo descriptorAllocateInfo = {};
    VkPushConstantRange pushConstantRange = {};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    VkResult result = VK_SUCCESS;

    if (!loadShader("vulkan/shaders/imgui_linear.frag.spv",
                    mImGuiFragmentShader) ||
        !loadShader("vulkan/shaders/final.vert.spv", mFinalVertexShader) ||
        !loadShader("vulkan/shaders/final.frag.spv", mFinalFragmentShader) ||
        !createCompositionRenderPass())
    {
        return false;
    }

    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 0.0f;
    result = vkCreateSampler(
        mDevice, &samplerInfo, nullptr, &mCompositionSampler);
    if (!checkVkResult(result, "vkCreateSampler for final output"))
    {
        return false;
    }

    descriptorBinding.binding = 0;
    descriptorBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorBinding.descriptorCount = 1;
    descriptorBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorLayoutInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutInfo.bindingCount = 1;
    descriptorLayoutInfo.pBindings = &descriptorBinding;
    result = vkCreateDescriptorSetLayout(
        mDevice, &descriptorLayoutInfo, nullptr, &mFinalDescriptorSetLayout);
    if (!checkVkResult(result, "vkCreateDescriptorSetLayout for final output"))
    {
        return false;
    }

    descriptorAllocateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorAllocateInfo.descriptorPool = mDescriptorPool;
    descriptorAllocateInfo.descriptorSetCount = 1;
    descriptorAllocateInfo.pSetLayouts = &mFinalDescriptorSetLayout;
    result = vkAllocateDescriptorSets(
        mDevice, &descriptorAllocateInfo, &mFinalDescriptorSet);
    if (!checkVkResult(result, "vkAllocateDescriptorSets for final output"))
    {
        return false;
    }

    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 2;
    pipelineLayoutInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mFinalDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    result = vkCreatePipelineLayout(
        mDevice, &pipelineLayoutInfo, nullptr, &mFinalPipelineLayout);
    if (!checkVkResult(result, "vkCreatePipelineLayout for final output"))
    {
        return false;
    }
    if (!createCompositionResources(static_cast<uint32_t>(width),
                                    static_cast<uint32_t>(height)))
    {
        return false;
    }
    return createFinalOutputResources();
}

void GlfwDisplay::destroyPresentation()
{
    destroyFinalOutputResources();
    destroyCompositionResources();
    if (mFinalPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(mDevice, mFinalPipelineLayout, nullptr);
        mFinalPipelineLayout = VK_NULL_HANDLE;
    }
    if (mFinalDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(
            mDevice, mFinalDescriptorSetLayout, nullptr);
        mFinalDescriptorSetLayout = VK_NULL_HANDLE;
    }
    mFinalDescriptorSet = VK_NULL_HANDLE;
    if (mCompositionSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(mDevice, mCompositionSampler, nullptr);
        mCompositionSampler = VK_NULL_HANDLE;
    }
    if (mCompositionRenderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(mDevice, mCompositionRenderPass, nullptr);
        mCompositionRenderPass = VK_NULL_HANDLE;
    }
}

display_output::OutputMode GlfwDisplay::requestedOutputMode() const
{
    uint32_t storedMode = 0;

    if (mSettings != nullptr)
    {
        storedMode = mSettings->getAs<uint32_t>("render/post/outputMode");
    }
    storedMode = std::min(storedMode, static_cast<uint32_t>(display_output::OutputMode::SDR));
    return static_cast<display_output::OutputMode>(storedMode);
}

bool GlfwDisplay::refreshOutputPolicy()
{
    const VkSurfaceFormatKHR hdrFormat = {
        VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_COLOR_SPACE_HDR10_ST2084_EXT,
    };
    const VkSurfaceFormatKHR sdrFormat = {
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };
    uint32_t formatCount = 0;
    std::vector<VkSurfaceFormatKHR> formats;
    VkResult result = VK_SUCCESS;
    size_t index = 0;
    bool hdrAvailable = false;
    bool sdrAvailable = false;
    display_output::OutputMode requestedMode = display_output::OutputMode::Auto;
    display_output::SurfaceEncoding selectedEncoding = display_output::SurfaceEncoding::SDR;

    result = vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, nullptr);
    if (!checkVkResult(result, "vkGetPhysicalDeviceSurfaceFormatsKHR") || formatCount == 0)
    {
        return false;
    }

    formats.resize(formatCount);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &formatCount, formats.data());
    if (!checkVkResult(result, "vkGetPhysicalDeviceSurfaceFormatsKHR"))
    {
        return false;
    }

    while (index < formats.size())
    {
        if ((formats[index].format == hdrFormat.format ||
             formats[index].format == VK_FORMAT_UNDEFINED) &&
            formats[index].colorSpace == hdrFormat.colorSpace)
        {
            hdrAvailable = true;
        }
        if ((formats[index].format == sdrFormat.format ||
             formats[index].format == VK_FORMAT_UNDEFINED) &&
            formats[index].colorSpace == sdrFormat.colorSpace)
        {
            sdrAvailable = true;
        }
        ++index;
    }

    hdrAvailable = hdrAvailable && mOutputCapabilities.output.swapchainColorspace;
    mOutputCapabilities.output.hdr10 = hdrAvailable;
    requestedMode = requestedOutputMode();
    selectedEncoding = display_output::selectSurfaceEncoding(requestedMode, mOutputCapabilities.output);
    if (selectedEncoding == display_output::SurfaceEncoding::HDR10)
    {
        mSelectedSurfaceFormat = hdrFormat;
    }
    else
    {
        if (!sdrAvailable)
        {
            STRELKA_ERROR("Vulkan surface does not expose B8G8R8A8_SRGB + SRGB_NONLINEAR");
            return false;
        }
        mSelectedSurfaceFormat = sdrFormat;
    }

    mOutputCapabilities.surfaceEncoding = selectedEncoding;
    mOutputCapabilities.output.hdrSelected =
        selectedEncoding == display_output::SurfaceEncoding::HDR10;
    if (requestedMode == display_output::OutputMode::HDR && !hdrAvailable &&
        !mForcedHdrFallbackWarned)
    {
        STRELKA_WARNING(
            "HDR10 output was requested but A2B10G10R10_UNORM_PACK32 + HDR10_ST2084 is unavailable; using SDR");
        mForcedHdrFallbackWarned = true;
    }
    return enumeratePresentModes();
}

bool GlfwDisplay::enumeratePresentModes()
{
    uint32_t presentModeCount = 0;
    VkResult result = VK_SUCCESS;
    size_t index = 0;
    bool const platformVrr = mOutputCapabilities.present.vrr;
    bool const platformVrrOnFifo = mOutputCapabilities.present.vrrOnFifo;

    result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        mPhysicalDevice, mSurface, &presentModeCount, nullptr);
    if (!checkVkResult(result, "vkGetPhysicalDeviceSurfacePresentModesKHR") ||
        presentModeCount == 0)
    {
        return false;
    }
    mAvailablePresentModes.resize(presentModeCount);
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        mPhysicalDevice, mSurface, &presentModeCount, mAvailablePresentModes.data());
    if (!checkVkResult(result, "vkGetPhysicalDeviceSurfacePresentModesKHR"))
    {
        return false;
    }

    mOutputCapabilities.present = {};
    mOutputCapabilities.present.vrr = platformVrr;
    mOutputCapabilities.present.vrrOnFifo = platformVrrOnFifo;
    while (index < mAvailablePresentModes.size())
    {
        if (mAvailablePresentModes[index] == VK_PRESENT_MODE_FIFO_KHR)
        {
            mOutputCapabilities.present.fifo = true;
        }
        else if (mAvailablePresentModes[index] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
        {
            mOutputCapabilities.present.fifoRelaxed = true;
        }
        else if (mAvailablePresentModes[index] == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            mOutputCapabilities.present.mailbox = true;
        }
        else if (mAvailablePresentModes[index] == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            mOutputCapabilities.present.immediate = true;
        }
        ++index;
    }
    mOutputCapabilities.presentMode = display_output::PresentMode::Fifo;
    return mOutputCapabilities.present.fifo;
}

bool GlfwDisplay::createSwapchain(int width, int height)
{
    const VkPresentModeKHR requestedPresentModes[] = {VK_PRESENT_MODE_FIFO_KHR};
    uint32_t requestedMode = 0;
    bool vrrEnabled = false;
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;

    if (width <= 0 || height <= 0)
    {
        return false;
    }

    if (mWindowData == nullptr)
    {
        mWindowData = new ImGui_ImplVulkanH_Window();
    }
    if (!refreshOutputPolicy())
    {
        return false;
    }
    mWindowData->Surface = mSurface;
    mWindowData->SurfaceFormat = mSelectedSurfaceFormat;
    mWindowData->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        mPhysicalDevice, mSurface, requestedPresentModes, 1);
    mWindowData->ClearValue.color.float32[0] = 0.0f;
    mWindowData->ClearValue.color.float32[1] = 0.0f;
    mWindowData->ClearValue.color.float32[2] = 0.0f;
    mWindowData->ClearValue.color.float32[3] = 1.0f;
    destroyFinalFramebuffers();
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        mInstance, mPhysicalDevice, mDevice, mWindowData, mQueueFamily, nullptr,
        width, height, kMinImageCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    mWindowData->FrameIndex = 0;
    if (mWindowData->Swapchain == VK_NULL_HANDLE)
    {
        return false;
    }
    if (mCompositionRenderPass != VK_NULL_HANDLE)
    {
        if (!createCompositionResources(static_cast<uint32_t>(width),
                                        static_cast<uint32_t>(height)))
        {
            return false;
        }
        if (mFinalRenderPass == VK_NULL_HANDLE ||
            mFinalFormat != mSelectedSurfaceFormat.format ||
            mFinalEncoding != mOutputCapabilities.surfaceEncoding)
        {
            if (!createFinalOutputResources())
            {
                return false;
            }
        }
        else if (!createFinalFramebuffers())
        {
            return false;
        }
    }
    if (mSettings != nullptr)
    {
        requestedMode = mSettings->getAs<uint32_t>("render/post/outputMode");
        vrrEnabled = mSettings->getAs<bool>("display/vrr/enabled");
        paperWhiteNits = mSettings->getAs<float>("render/post/paperWhiteNits");
        peakNits = mSettings->getAs<float>("render/post/peakNits");
    }
    mAppliedOutputMode = requestedMode;
    mAppliedVrrEnabled = vrrEnabled;
    mAppliedPaperWhiteNits = paperWhiteNits;
    mAppliedPeakNits = peakNits;
    mOutputMonitor = currentOutputMonitor();
    refreshPlatformDisplayState();
    mOutputCapabilities.vrrStatus = display_output::interpretVrrStatus(
        vrrEnabled, mOutputCapabilities.presentMode, mOutputCapabilities.present,
        mOutputCapabilities.vrrStatus);
    applyHdrMetadata();
    logOutputCapabilities();
    mSwapchainRebuild = false;
    return true;
}

GLFWmonitor *GlfwDisplay::currentOutputMonitor() const
{
    // Both are returned as the non-const GLFWmonitor* result of this function.
    // NOLINTNEXTLINE(misc-const-correctness)
    GLFWmonitor *fullscreenMonitor = nullptr;
    GLFWmonitor *const *monitors = nullptr;
    // NOLINTNEXTLINE(misc-const-correctness)
    GLFWmonitor *bestMonitor = nullptr;
    const GLFWvidmode *videoMode = nullptr;
    int monitorCount = 0;
    int windowX = 0;
    int windowY = 0;
    int windowWidth = 0;
    int windowHeight = 0;
    int monitorX = 0;
    int monitorY = 0;
    int overlapWidth = 0;
    int overlapHeight = 0;
    int overlapArea = 0;
    int bestOverlapArea = -1;
    int index = 0;

    fullscreenMonitor = glfwGetWindowMonitor(mWindow);
    if (fullscreenMonitor != nullptr)
    {
        return fullscreenMonitor;
    }
    monitors = glfwGetMonitors(&monitorCount);
    if (monitors == nullptr || monitorCount == 0)
    {
        return nullptr;
    }

    glfwGetWindowPos(mWindow, &windowX, &windowY);
    glfwGetWindowSize(mWindow, &windowWidth, &windowHeight);
    while (index < monitorCount)
    {
        glfwGetMonitorPos(monitors[index], &monitorX, &monitorY);
        videoMode = glfwGetVideoMode(monitors[index]);
        if (videoMode != nullptr)
        {
            overlapWidth = std::max(
                0, std::min(windowX + windowWidth, monitorX + videoMode->width) -
                       std::max(windowX, monitorX));
            overlapHeight = std::max(
                0, std::min(windowY + windowHeight, monitorY + videoMode->height) -
                       std::max(windowY, monitorY));
            overlapArea = overlapWidth * overlapHeight;
            if (overlapArea > bestOverlapArea)
            {
                bestOverlapArea = overlapArea;
                bestMonitor = monitors[index];
            }
        }
        ++index;
    }
    return bestMonitor;
}

void GlfwDisplay::refreshPlatformDisplayState()
{
    display_output::PlatformDisplayState state;
    const char *monitorName = nullptr;
    const GLFWvidmode *videoMode = nullptr;
    void *nativeWindow = nullptr;

    if (mOutputMonitor != nullptr)
    {
        monitorName = glfwGetMonitorName(mOutputMonitor);
        videoMode = glfwGetVideoMode(mOutputMonitor);
    }
#if defined(_WIN32)
    nativeWindow = reinterpret_cast<void *>(glfwGetWin32Window(mWindow));
#endif
    state = display_output::probePlatformDisplay(
        monitorName != nullptr ? monitorName : "", nativeWindow);
    if (state.currentRefreshRateHz <= 0.0f && videoMode != nullptr)
    {
        state.currentRefreshRateHz =
            static_cast<float>(videoMode->refreshRate);
    }

    mOutputCapabilities.vrrStatus = state.vrrStatus;
    mOutputCapabilities.minRefreshRateHz = state.minRefreshRateHz;
    mOutputCapabilities.maxRefreshRateHz = state.maxRefreshRateHz;
    mOutputCapabilities.currentRefreshRateHz = state.currentRefreshRateHz;
    mOutputCapabilities.present.vrr =
        state.vrrStatus != display_output::VrrStatus::Unknown;
    mOutputCapabilities.present.vrrOnFifo =
        state.vrrStatus == display_output::VrrStatus::Active;
}

void GlfwDisplay::applyHdrMetadata()
{
    VkHdrMetadataEXT metadata = {};
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;

    if (!mOutputCapabilities.output.hdrSelected ||
        !mOutputCapabilities.output.hdrMetadata || mSetHdrMetadata == nullptr ||
        mWindowData == nullptr || mWindowData->Swapchain == VK_NULL_HANDLE)
    {
        return;
    }
    if (mSettings != nullptr)
    {
        paperWhiteNits = mSettings->getAs<float>("render/post/paperWhiteNits");
        peakNits = mSettings->getAs<float>("render/post/peakNits");
    }
    paperWhiteNits = std::max(1.0f, paperWhiteNits);
    peakNits = std::max(paperWhiteNits, peakNits);

    metadata.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;
    metadata.displayPrimaryRed = {0.708f, 0.292f};
    metadata.displayPrimaryGreen = {0.170f, 0.797f};
    metadata.displayPrimaryBlue = {0.131f, 0.046f};
    metadata.whitePoint = {0.3127f, 0.3290f};
    metadata.maxLuminance = peakNits;
    metadata.minLuminance = 0.001f;
    metadata.maxContentLightLevel = peakNits;
    metadata.maxFrameAverageLightLevel = paperWhiteNits;
    swapchain = mWindowData->Swapchain;
    mSetHdrMetadata(mDevice, 1, &swapchain, &metadata);
}

void GlfwDisplay::logOutputCapabilities() const
{
    std::string presentModes;
    size_t index = 0;

    while (index < mAvailablePresentModes.size())
    {
        if (!presentModes.empty())
        {
            presentModes += ", ";
        }
        presentModes += presentModeName(mAvailablePresentModes[index]);
        ++index;
    }
    STRELKA_INFO(
        "Display output: HDR selected={}, HDR10 pair={}, HDR metadata={}, present modes=[{}], "
        "present wait={}, present id={}, display timing={}, VRR={}, refresh={:.3f} Hz, "
        "VRR range={:.3f}-{:.3f} Hz",
        mOutputCapabilities.output.hdrSelected, mOutputCapabilities.output.hdr10,
        mOutputCapabilities.output.hdrMetadata, presentModes,
        mOutputCapabilities.presentWait, mOutputCapabilities.presentId,
        mOutputCapabilities.displayTiming,
        display_output::vrrStatusName(mOutputCapabilities.vrrStatus),
        mOutputCapabilities.currentRefreshRateHz,
        mOutputCapabilities.minRefreshRateHz,
        mOutputCapabilities.maxRefreshRateHz);
}

void GlfwDisplay::onBeginFrame()
{
    int width = 0;
    int height = 0;
    uint32_t requestedMode = 0;
    bool vrrEnabled = false;
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;
    const GLFWmonitor *outputMonitor = nullptr;
    VkSurfaceFormatKHR previousSurfaceFormat = {};
    bool previousHdrSupported = false;
    bool hdrMetadataChanged = false;
    VkResult result = VK_SUCCESS;

    mFrameValid = false;
    mFrameRecorded = false;
    if (!mImGuiInitialized || mWindowData == nullptr)
    {
        return;
    }

    glfwGetFramebufferSize(mWindow, &width, &height);
    if (width <= 0 || height <= 0 || glfwGetWindowAttrib(mWindow, GLFW_ICONIFIED) != 0)
    {
        return;
    }

    if (mSettings != nullptr)
    {
        requestedMode = mSettings->getAs<uint32_t>("render/post/outputMode");
        vrrEnabled = mSettings->getAs<bool>("display/vrr/enabled");
        paperWhiteNits = mSettings->getAs<float>("render/post/paperWhiteNits");
        peakNits = mSettings->getAs<float>("render/post/peakNits");
    }
    outputMonitor = currentOutputMonitor();
    previousSurfaceFormat = mSelectedSurfaceFormat;
    previousHdrSupported = mOutputCapabilities.output.hdr10;
    if (!refreshOutputPolicy())
    {
        return;
    }
    hdrMetadataChanged = paperWhiteNits != mAppliedPaperWhiteNits ||
                         peakNits != mAppliedPeakNits;
    if (hdrMetadataChanged)
    {
        mAppliedPaperWhiteNits = paperWhiteNits;
        mAppliedPeakNits = peakNits;
        applyHdrMetadata();
    }
    if (requestedMode != mAppliedOutputMode || vrrEnabled != mAppliedVrrEnabled ||
        outputMonitor != mOutputMonitor ||
        previousSurfaceFormat.format != mSelectedSurfaceFormat.format ||
        previousSurfaceFormat.colorSpace != mSelectedSurfaceFormat.colorSpace ||
        previousHdrSupported != mOutputCapabilities.output.hdr10)
    {
        mSwapchainRebuild = true;
    }

    if (mSwapchainRebuild || mWindowData->Width != width || mWindowData->Height != height)
    {
        result = vkDeviceWaitIdle(mDevice);
        if (!checkVkResult(result, "vkDeviceWaitIdle") || !createSwapchain(width, height))
        {
            return;
        }
        ImGui_ImplVulkan_SetMinImageCount(kMinImageCount);
    }

    if (!acquireFrame())
    {
        return;
    }

    ImGui_ImplVulkan_NewFrame();
    mFrameValid = true;
}

bool GlfwDisplay::acquireFrame()
{
    VkSemaphore imageAcquiredSemaphore = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;
    const ImGui_ImplVulkanH_Frame *frame = nullptr;

    imageAcquiredSemaphore =
        mWindowData->FrameSemaphores[static_cast<int>(mWindowData->SemaphoreIndex)].ImageAcquiredSemaphore;
    result = vkAcquireNextImageKHR(
        mDevice, mWindowData->Swapchain, UINT64_MAX, imageAcquiredSemaphore,
        VK_NULL_HANDLE, &mWindowData->FrameIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        mSwapchainRebuild = true;
        return false;
    }
    if (result == VK_SUBOPTIMAL_KHR)
    {
        mSwapchainRebuild = true;
    }
    else if (!checkVkResult(result, "vkAcquireNextImageKHR"))
    {
        return false;
    }

    frame = &mWindowData->Frames[static_cast<int>(mWindowData->FrameIndex)];
    result = vkWaitForFences(mDevice, 1, &frame->Fence, VK_TRUE, kFenceTimeoutNs);
    if (result == VK_TIMEOUT)
    {
        STRELKA_ERROR("Vulkan display frame fence timed out after 5 seconds");
        mSwapchainRebuild = true;
        return false;
    }
    return checkVkResult(result, "vkWaitForFences");
}

void GlfwDisplay::drawFrame(ImageBuffer& result)
{
    if (!mFrameValid || result.deviceData == nullptr || result.width == 0 ||
        result.height == 0 || result.frameSerial == 0)
    {
        return;
    }
    prepareInteropFrame(result);
}

bool GlfwDisplay::ensureInterop(uint32_t width, uint32_t height)
{
    CudaVulkanInterop::CreateInfo createInfo = {};

    if (mInterop.isInitialized() && mInteropWidth == width &&
        mInteropHeight == height)
    {
        return true;
    }

    destroyInterop();
    if (mRender == nullptr || mRender->activeCudaDeviceOrdinal() < 0 ||
        mRender->getNativeCudaStream() == nullptr)
    {
        STRELKA_ERROR("The renderer did not expose CUDA interop state");
        return false;
    }

    createInfo.instance = mInstance;
    createInfo.physicalDevice = mPhysicalDevice;
    createInfo.device = mDevice;
    createInfo.queue = mQueue;
    createInfo.queueFamilyIndex = mQueueFamily;
    createInfo.width = width;
    createInfo.height = height;
    createInfo.cudaDevice = mRender->activeCudaDeviceOrdinal();
    createInfo.cudaStream =
        static_cast<cudaStream_t>(mRender->getNativeCudaStream());
    if (!mInterop.initialize(createInfo))
    {
        STRELKA_ERROR("Failed to initialize CUDA-Vulkan interop: {}",
                      mInterop.lastError());
        destroyInterop();
        return false;
    }
    if (!createInteropViews())
    {
        destroyInterop();
        return false;
    }

    mInteropWidth = width;
    mInteropHeight = height;
    return true;
}

bool GlfwDisplay::createInteropViews()
{
    VkImageViewCreateInfo viewInfo = {};
    VkResult result = VK_SUCCESS;
    size_t slot = 0;

    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = mInterop.format();
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    while (slot < CudaVulkanInterop::SlotCount)
    {
        viewInfo.image = mInterop.image(slot);
        result = vkCreateImageView(
            mDevice, &viewInfo, nullptr, &mInteropImageViews[slot]);
        if (!checkVkResult(result, "vkCreateImageView for CUDA interop"))
        {
            return false;
        }
        mInteropDescriptors[slot] = ImGui_ImplVulkan_AddTexture(
            mInteropImageViews[slot], mInterop.imageLayout());
        if (mInteropDescriptors[slot] == VK_NULL_HANDLE)
        {
            STRELKA_ERROR("ImGui failed to create an interop image descriptor");
            return false;
        }
        ++slot;
    }
    return true;
}

bool GlfwDisplay::prepareInteropFrame(ImageBuffer& result)
{
    cudaStream_t stream = nullptr;
    cudaError_t cudaResult = cudaSuccess;
    uint64_t readyValue = 0;
    uint64_t retireValue = 0;
    size_t slot = 0;

    if (!ensureInterop(result.width, result.height))
    {
        return false;
    }
    if (!shouldTransformFrame(result.frameSerial, mPresentedFrameSerial))
    {
        return true;
    }

    slot = static_cast<size_t>(result.frameSerial % CudaVulkanInterop::SlotCount);
    stream = static_cast<cudaStream_t>(mRender->getNativeCudaStream());
    // Wait for Vulkan's retirement from the previous use before reserving the
    // next retire value. Waiting after reserveRetireValue() waits on the value
    // that only the upcoming Vulkan submission can signal, deadlocking the
    // first displayed frame.
    if (!mInterop.cudaWaitForRetire(slot, stream) ||
        !mInterop.reserveReadyValue(slot, &readyValue) ||
        !mInterop.reserveRetireValue(slot, &retireValue) ||
        readyValue == 0 || retireValue <= readyValue)
    {
        STRELKA_ERROR("Failed to order CUDA-Vulkan interop: {}",
                      mInterop.lastError());
        destroyInterop();
        return false;
    }

    cudaResult = tonemapToSurfaceLinear(
        static_cast<const float4 *>(result.deviceData),
        mInterop.cudaSurface(slot),
        result.width,
        result.height,
        &result.presentation,
        stream);
    if (cudaResult != cudaSuccess)
    {
        STRELKA_ERROR("CUDA surface tonemapping failed: {}",
                      cudaGetErrorString(cudaResult));
        destroyInterop();
        return false;
    }
    if (!mInterop.cudaSignal(slot, readyValue, stream))
    {
        STRELKA_ERROR("Failed to signal the CUDA-ready image: {}",
                      mInterop.lastError());
        destroyInterop();
        return false;
    }

    mInteropSlot = slot;
    mInteropReadyValue = readyValue;
    mInteropRetireValue = retireValue;
    mInteropSubmissionPending = true;
    mPresentedFrameSerial = result.frameSerial;
    // ImGui's ImTextureID carries the Vulkan descriptor handle opaquely.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    mViewportTexture = reinterpret_cast<void *>(mInteropDescriptors[mInteropSlot]);
    return true;
}

void GlfwDisplay::resetFrame()
{
    mViewportTexture = nullptr;
    mPresentedFrameSerial = 0;
    mInteropSubmissionPending = false;
}

void *GlfwDisplay::getDisplayNativeTexure()
{
    return mViewportTexture;
}

float GlfwDisplay::getMaxEDR()
{
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;

    if (!mOutputCapabilities.output.hdrSelected || mSettings == nullptr)
    {
        return 1.0f;
    }
    paperWhiteNits = std::max(
        1.0f, mSettings->getAs<float>("render/post/paperWhiteNits"));
    peakNits = std::max(
        paperWhiteNits, mSettings->getAs<float>("render/post/peakNits"));
    return peakNits / paperWhiteNits;
}

display_output::DisplayCapabilities GlfwDisplay::getOutputCapabilities() const
{
    return mOutputCapabilities;
}

void GlfwDisplay::drawUI()
{
    if (!mFrameValid)
    {
        return;
    }
    mFrameRecorded = recordFrame();
    if (!mFrameRecorded)
    {
        mSwapchainRebuild = true;
    }
}

bool GlfwDisplay::recordFrame()
{
    struct OutputParameters
    {
        float paperWhiteNits;
        float peakNits;
    };

    const ImGui_ImplVulkanH_Frame *frame = nullptr;
    VkCommandBufferBeginInfo commandBufferInfo = {};
    VkRenderPassBeginInfo compositionPassInfo = {};
    VkRenderPassBeginInfo finalPassInfo = {};
    VkClearValue compositionClear = {};
    VkViewport viewport = {};
    VkRect2D scissor = {};
    OutputParameters outputParameters = {};
    std::array<VkPipelineStageFlags, 2> waitStages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    };
    std::array<VkSemaphore, 2> waitSemaphores = {};
    std::array<VkSemaphore, 2> signalSemaphores = {};
    std::array<uint64_t, 2> waitValues = {};
    std::array<uint64_t, 2> signalValues = {};
    VkTimelineSemaphoreSubmitInfo timelineInfo = {};
    VkSubmitInfo submitInfo = {};
    VkSemaphore imageAcquiredSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderCompleteSemaphore = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;

    frame = &mWindowData->Frames[static_cast<int>(mWindowData->FrameIndex)];
    imageAcquiredSemaphore =
        mWindowData->FrameSemaphores[static_cast<int>(mWindowData->SemaphoreIndex)].ImageAcquiredSemaphore;
    renderCompleteSemaphore =
        mWindowData->FrameSemaphores[static_cast<int>(mWindowData->SemaphoreIndex)].RenderCompleteSemaphore;
    waitSemaphores[0] = imageAcquiredSemaphore;
    signalSemaphores[0] = renderCompleteSemaphore;
    if (mInteropSubmissionPending)
    {
        waitSemaphores[1] = mInterop.timelineSemaphore(mInteropSlot);
        signalSemaphores[1] = mInterop.timelineSemaphore(mInteropSlot);
        waitValues[1] = mInteropReadyValue;
        signalValues[1] = mInteropRetireValue;
    }

    result = vkResetCommandPool(mDevice, frame->CommandPool, 0);
    if (!checkVkResult(result, "vkResetCommandPool"))
    {
        return false;
    }

    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(frame->CommandBuffer, &commandBufferInfo);
    if (!checkVkResult(result, "vkBeginCommandBuffer"))
    {
        return false;
    }

    compositionClear.color.float32[3] = 1.0f;
    compositionPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    compositionPassInfo.renderPass = mCompositionRenderPass;
    compositionPassInfo.framebuffer = mCompositionFramebuffer;
    compositionPassInfo.renderArea.extent.width = mCompositionWidth;
    compositionPassInfo.renderArea.extent.height = mCompositionHeight;
    compositionPassInfo.clearValueCount = 1;
    compositionPassInfo.pClearValues = &compositionClear;
    vkCmdBeginRenderPass(
        frame->CommandBuffer, &compositionPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame->CommandBuffer);
    vkCmdEndRenderPass(frame->CommandBuffer);

    viewport.width = static_cast<float>(mWindowData->Width);
    viewport.height = static_cast<float>(mWindowData->Height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    scissor.extent.width = static_cast<uint32_t>(mWindowData->Width);
    scissor.extent.height = static_cast<uint32_t>(mWindowData->Height);
    outputParameters.paperWhiteNits = 203.0f;
    outputParameters.peakNits = 1000.0f;
    if (mSettings != nullptr)
    {
        outputParameters.paperWhiteNits = std::max(
            1.0f, mSettings->getAs<float>("render/post/paperWhiteNits"));
        outputParameters.peakNits = std::max(
            outputParameters.paperWhiteNits,
            mSettings->getAs<float>("render/post/peakNits"));
    }
    finalPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    finalPassInfo.renderPass = mFinalRenderPass;
    finalPassInfo.framebuffer =
        mFinalFramebuffers[static_cast<size_t>(mWindowData->FrameIndex)];
    finalPassInfo.renderArea.extent = scissor.extent;
    vkCmdBeginRenderPass(
        frame->CommandBuffer, &finalPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(
        frame->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mFinalPipeline);
    vkCmdBindDescriptorSets(
        frame->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        mFinalPipelineLayout, 0, 1, &mFinalDescriptorSet, 0, nullptr);
    vkCmdPushConstants(
        frame->CommandBuffer, mFinalPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(outputParameters), &outputParameters);
    vkCmdSetViewport(frame->CommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(frame->CommandBuffer, 0, 1, &scissor);
    vkCmdDraw(frame->CommandBuffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(frame->CommandBuffer);

    result = vkEndCommandBuffer(frame->CommandBuffer);
    if (!checkVkResult(result, "vkEndCommandBuffer"))
    {
        return false;
    }
    result = vkResetFences(mDevice, 1, &frame->Fence);
    if (!checkVkResult(result, "vkResetFences"))
    {
        return false;
    }

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = mInteropSubmissionPending ? 2u : 1u;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frame->CommandBuffer;
    submitInfo.signalSemaphoreCount = mInteropSubmissionPending ? 2u : 1u;
    submitInfo.pSignalSemaphores = signalSemaphores.data();
    if (mInteropSubmissionPending)
    {
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.waitSemaphoreValueCount = submitInfo.waitSemaphoreCount;
        timelineInfo.pWaitSemaphoreValues = waitValues.data();
        timelineInfo.signalSemaphoreValueCount = submitInfo.signalSemaphoreCount;
        timelineInfo.pSignalSemaphoreValues = signalValues.data();
        submitInfo.pNext = &timelineInfo;
    }
    result = vkQueueSubmit(mQueue, 1, &submitInfo, frame->Fence);
    if (!checkVkResult(result, "vkQueueSubmit"))
    {
        return false;
    }
    mInteropSubmissionPending = false;
    return true;
}

void GlfwDisplay::onEndFrame()
{
    if (!mFrameValid || !mFrameRecorded)
    {
        return;
    }
    presentFrame();
    mFrameValid = false;
    mFrameRecorded = false;
}

bool GlfwDisplay::presentFrame()
{
    VkSemaphore renderCompleteSemaphore = VK_NULL_HANDLE;
    VkPresentInfoKHR presentInfo = {};
    VkResult result = VK_SUCCESS;

    if (mSwapchainRebuild)
    {
        return false;
    }

    renderCompleteSemaphore =
        mWindowData->FrameSemaphores[static_cast<int>(mWindowData->SemaphoreIndex)].RenderCompleteSemaphore;
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderCompleteSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &mWindowData->Swapchain;
    presentInfo.pImageIndices = &mWindowData->FrameIndex;
    result = vkQueuePresentKHR(mQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        mSwapchainRebuild = true;
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            return false;
        }
    }
    else if (!checkVkResult(result, "vkQueuePresentKHR"))
    {
        return false;
    }

    mWindowData->SemaphoreIndex =
        (mWindowData->SemaphoreIndex + 1) % mWindowData->SemaphoreCount;
    return true;
}

void GlfwDisplay::destroyInterop()
{
    size_t slot = 0;

    if (mDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(mDevice);
    }
    while (slot < CudaVulkanInterop::SlotCount)
    {
        if (mInteropDescriptors[slot] != VK_NULL_HANDLE && mImGuiInitialized)
        {
            ImGui_ImplVulkan_RemoveTexture(mInteropDescriptors[slot]);
        }
        mInteropDescriptors[slot] = VK_NULL_HANDLE;
        if (mInteropImageViews[slot] != VK_NULL_HANDLE &&
            mDevice != VK_NULL_HANDLE)
        {
            vkDestroyImageView(mDevice, mInteropImageViews[slot], nullptr);
        }
        mInteropImageViews[slot] = VK_NULL_HANDLE;
        ++slot;
    }
    mInterop.shutdown();
    mInteropWidth = 0;
    mInteropHeight = 0;
    mViewportTexture = nullptr;
    mPresentedFrameSerial = 0;
    mInteropSubmissionPending = false;
}

void GlfwDisplay::destroy()
{
    if (mDestroyed)
    {
        return;
    }
    mDestroyed = true;

    if (mDevice != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(mDevice);
    }
    destroyInterop();
    if (mImGuiInitialized)
    {
        ImGui_ImplVulkan_Shutdown();
        mImGuiInitialized = false;
    }
    destroyPresentation();
    if (mGlfwBackendInitialized)
    {
        ImGui_ImplGlfw_Shutdown();
        mGlfwBackendInitialized = false;
    }
    if (mImGuiContextCreated)
    {
        ImGui::DestroyContext();
        mImGuiContextCreated = false;
    }
    if (mWindowData != nullptr && mDevice != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkanH_DestroyWindow(mInstance, mDevice, mWindowData, nullptr);
    }
    if (mSurface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
        mSurface = VK_NULL_HANDLE;
    }
    if (mDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
        mDescriptorPool = VK_NULL_HANDLE;
    }
    if (mDevice != VK_NULL_HANDLE)
    {
        vkDestroyDevice(mDevice, nullptr);
        mDevice = VK_NULL_HANDLE;
    }
    if (mInstance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(mInstance, nullptr);
        mInstance = VK_NULL_HANDLE;
    }

    delete mWindowData;
    mWindowData = nullptr;
    if (mWindow != nullptr)
    {
        glfwDestroyWindow(mWindow);
        mWindow = nullptr;
    }
    glfwTerminate();
}
