#include "cuda_vulkan_interop.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace oka
{

namespace
{

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits ExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits ExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits ExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits ExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

constexpr VkFormat InteropFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags InteropImageUsage =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char *name)
{
    size_t index = 0;

    for (index = 0; index < extensions.size(); ++index) {
        if (std::strcmp(extensions[index].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

CudaVulkanInterop::~CudaVulkanInterop()
{
    destroyResources();
}

bool CudaVulkanInterop::initialize(const CreateInfo& createInfo)
{
    size_t slot = 0;

    lastError_.clear();
    if (initialized_) {
        setError("CUDA-Vulkan interop is already initialized");
        return false;
    }
    if (!validateCreateInfo(createInfo)) {
        return false;
    }

    createInfo_ = createInfo;
    if (createInfo_.queue == VK_NULL_HANDLE) {
        vkGetDeviceQueue(createInfo_.device, createInfo_.queueFamilyIndex, 0,
                         &createInfo_.queue);
    }
    if (createInfo_.queue == VK_NULL_HANDLE) {
        setError("Vulkan queue family has no queue at index zero");
        destroyResources();
        return false;
    }
    if (!selectCudaDevice() || !verifyDeviceUuid() || !probeVulkanSupport() ||
        !loadExportFunctions()) {
        destroyResources();
        return false;
    }

    for (slot = 0; slot < SlotCount; ++slot) {
        if (!createSlot(slot)) {
            destroyResources();
            return false;
        }
    }

    initialized_ = true;
    return true;
}

void CudaVulkanInterop::shutdown()
{
    destroyResources();
}

bool CudaVulkanInterop::isInitialized() const
{
    return initialized_;
}

const std::string& CudaVulkanInterop::lastError() const
{
    return lastError_;
}

VkImage CudaVulkanInterop::image(size_t slot) const
{
    if (slot >= SlotCount) {
        return VK_NULL_HANDLE;
    }
    return slots_[slot].image;
}

VkDeviceMemory CudaVulkanInterop::imageMemory(size_t slot) const
{
    if (slot >= SlotCount) {
        return VK_NULL_HANDLE;
    }
    return slots_[slot].memory;
}

cudaMipmappedArray_t CudaVulkanInterop::cudaMipmappedArray(size_t slot) const
{
    if (slot >= SlotCount) {
        return nullptr;
    }
    return slots_[slot].mipmappedArray;
}

cudaSurfaceObject_t CudaVulkanInterop::cudaSurface(size_t slot) const
{
    if (slot >= SlotCount) {
        return 0;
    }
    return slots_[slot].surface;
}

VkSemaphore CudaVulkanInterop::timelineSemaphore(size_t slot) const
{
    if (slot >= SlotCount) {
        return VK_NULL_HANDLE;
    }
    return slots_[slot].semaphore;
}

uint64_t CudaVulkanInterop::readyValue(size_t slot) const
{
    if (slot >= SlotCount) {
        return 0;
    }
    return slots_[slot].ready;
}

uint64_t CudaVulkanInterop::retireValue(size_t slot) const
{
    if (slot >= SlotCount) {
        return 0;
    }
    return slots_[slot].retire;
}

bool CudaVulkanInterop::reserveReadyValue(size_t slot, uint64_t *value)
{
    Slot *interopSlot = nullptr;

    if (!validateSlot(slot)) {
        return false;
    }
    if (value == nullptr) {
        setError("Ready timeline value output pointer is null");
        return false;
    }

    interopSlot = &slots_[slot];
    if (interopSlot->readyReserved) {
        setError("A ready value is already reserved for this slot");
        return false;
    }
    if (interopSlot->nextValue == std::numeric_limits<uint64_t>::max()) {
        setError("Timeline semaphore value overflow");
        return false;
    }

    ++interopSlot->nextValue;
    interopSlot->ready = interopSlot->nextValue;
    interopSlot->readyReserved = true;
    *value = interopSlot->ready;
    return true;
}

bool CudaVulkanInterop::reserveRetireValue(size_t slot, uint64_t *value)
{
    Slot *interopSlot = nullptr;

    if (!validateSlot(slot)) {
        return false;
    }
    if (value == nullptr) {
        setError("Retire timeline value output pointer is null");
        return false;
    }

    interopSlot = &slots_[slot];
    if (!interopSlot->readyReserved) {
        setError("Reserve a ready value before reserving its retire value");
        return false;
    }
    if (interopSlot->nextValue == std::numeric_limits<uint64_t>::max()) {
        setError("Timeline semaphore value overflow");
        return false;
    }

    ++interopSlot->nextValue;
    interopSlot->retire = interopSlot->nextValue;
    interopSlot->readyReserved = false;
    *value = interopSlot->retire;
    return true;
}

bool CudaVulkanInterop::cudaWait(size_t slot, uint64_t value, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams parameters{};
    cudaExternalSemaphore_t semaphore = nullptr;
    cudaStream_t selectedStream = nullptr;
    cudaError_t cudaStatus = cudaSuccess;

    if (!validateSlot(slot) || !selectCudaDevice()) {
        return false;
    }

    semaphore = slots_[slot].cudaSemaphore;
    selectedStream = stream != nullptr ? stream : createInfo_.cudaStream;
    parameters.params.fence.value = value;
    cudaStatus = cudaWaitExternalSemaphoresAsync(&semaphore, &parameters, 1,
                                                  selectedStream);
    if (cudaStatus != cudaSuccess) {
        setError("cudaWaitExternalSemaphoresAsync", cudaStatus);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::cudaSignal(size_t slot, uint64_t value, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams parameters{};
    cudaExternalSemaphore_t semaphore = nullptr;
    cudaStream_t selectedStream = nullptr;
    cudaError_t cudaStatus = cudaSuccess;

    if (!validateSlot(slot) || !selectCudaDevice()) {
        return false;
    }

    semaphore = slots_[slot].cudaSemaphore;
    selectedStream = stream != nullptr ? stream : createInfo_.cudaStream;
    parameters.params.fence.value = value;
    cudaStatus = cudaSignalExternalSemaphoresAsync(&semaphore, &parameters, 1,
                                                    selectedStream);
    if (cudaStatus != cudaSuccess) {
        setError("cudaSignalExternalSemaphoresAsync", cudaStatus);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::cudaWaitForRetire(size_t slot, cudaStream_t stream)
{
    if (!validateSlot(slot)) {
        return false;
    }
    if (slots_[slot].retire == 0) {
        return true;
    }
    return cudaWait(slot, slots_[slot].retire, stream);
}

bool CudaVulkanInterop::cudaSignalReady(size_t slot, cudaStream_t stream)
{
    if (!validateSlot(slot)) {
        return false;
    }
    if (slots_[slot].ready == 0) {
        setError("No ready timeline value has been reserved for this slot");
        return false;
    }
    return cudaSignal(slot, slots_[slot].ready, stream);
}

VkFormat CudaVulkanInterop::format() const
{
    return InteropFormat;
}

VkImageLayout CudaVulkanInterop::imageLayout() const
{
    return VK_IMAGE_LAYOUT_GENERAL;
}

VkExtent2D CudaVulkanInterop::extent() const
{
    VkExtent2D result{};

    result.width = createInfo_.width;
    result.height = createInfo_.height;
    return result;
}

uint32_t CudaVulkanInterop::queueFamilyIndex() const
{
    return createInfo_.queueFamilyIndex;
}

bool CudaVulkanInterop::validateCreateInfo(const CreateInfo& createInfo)
{
    uint32_t queueFamilyCount = 0;

    if (createInfo.instance == VK_NULL_HANDLE ||
        createInfo.physicalDevice == VK_NULL_HANDLE ||
        createInfo.device == VK_NULL_HANDLE) {
        setError("Vulkan instance, physical device, and device are required");
        return false;
    }
    if (createInfo.width == 0 || createInfo.height == 0) {
        setError("Interop image extent must be nonzero");
        return false;
    }
    if (createInfo.cudaDevice < 0) {
        setError("CUDA device ordinal must be nonnegative");
        return false;
    }

    queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(createInfo.physicalDevice,
                                              &queueFamilyCount, nullptr);
    if (createInfo.queueFamilyIndex >= queueFamilyCount) {
        setError("Vulkan queue family index is out of range");
        return false;
    }
    return true;
}

bool CudaVulkanInterop::verifyDeviceUuid()
{
    VkPhysicalDeviceIDProperties idProperties{};
    VkPhysicalDeviceProperties2 properties{};
    cudaDeviceProp cudaProperties{};
    cudaError_t cudaStatus = cudaSuccess;

    idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &idProperties;
    vkGetPhysicalDeviceProperties2(createInfo_.physicalDevice, &properties);

    cudaStatus = cudaGetDeviceProperties(&cudaProperties, createInfo_.cudaDevice);
    if (cudaStatus != cudaSuccess) {
        setError("cudaGetDeviceProperties", cudaStatus);
        return false;
    }
    if (std::memcmp(idProperties.deviceUUID, cudaProperties.uuid.bytes,
                    VK_UUID_SIZE) != 0) {
        setError("Vulkan physical device UUID does not match the CUDA device UUID");
        return false;
    }
    return true;
}

bool CudaVulkanInterop::probeVulkanSupport()
{
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    VkPhysicalDeviceFeatures2 features{};
    VkPhysicalDeviceExternalImageFormatInfo externalImageInfo{};
    VkPhysicalDeviceImageFormatInfo2 imageInfo{};
    VkExternalImageFormatProperties externalImageProperties{};
    VkImageFormatProperties2 imageProperties{};
    VkPhysicalDeviceExternalSemaphoreInfo semaphoreInfo{};
    VkExternalSemaphoreProperties semaphoreProperties{};
    std::vector<VkExtensionProperties> extensions;
    uint32_t extensionCount = 0;
    VkResult result = VK_SUCCESS;
    bool timelineIsCore = false;

    vkGetPhysicalDeviceProperties(createInfo_.physicalDevice, &properties);
    if (VK_API_VERSION_MAJOR(properties.apiVersion) < 1 ||
        (VK_API_VERSION_MAJOR(properties.apiVersion) == 1 &&
         VK_API_VERSION_MINOR(properties.apiVersion) < 1)) {
        setError("Vulkan 1.1 or newer is required for external interop");
        return false;
    }

    extensionCount = 0;
    result = vkEnumerateDeviceExtensionProperties(createInfo_.physicalDevice,
                                                   nullptr, &extensionCount,
                                                   nullptr);
    if (result != VK_SUCCESS) {
        setError("vkEnumerateDeviceExtensionProperties", result);
        return false;
    }
    extensions.resize(extensionCount);
    result = vkEnumerateDeviceExtensionProperties(createInfo_.physicalDevice,
                                                   nullptr, &extensionCount,
                                                   extensions.data());
    if (result != VK_SUCCESS) {
        setError("vkEnumerateDeviceExtensionProperties", result);
        return false;
    }

#if defined(_WIN32)
    if (!hasExtension(extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) ||
        !hasExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME)) {
        setError("Required Vulkan Win32 external-handle extensions are unavailable");
        return false;
    }
#else
    if (!hasExtension(extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) ||
        !hasExtension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
        setError("Required Vulkan file-descriptor external-handle extensions are unavailable");
        return false;
    }
#endif

    timelineIsCore =
        VK_API_VERSION_MAJOR(properties.apiVersion) > 1 ||
        VK_API_VERSION_MINOR(properties.apiVersion) >= 2;
    if (!timelineIsCore &&
        !hasExtension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
        setError("Vulkan timeline semaphore support is unavailable");
        return false;
    }

    timelineFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &timelineFeatures;
    vkGetPhysicalDeviceFeatures2(createInfo_.physicalDevice, &features);
    if (timelineFeatures.timelineSemaphore != VK_TRUE) {
        setError("Vulkan physical device does not support timeline semaphores");
        return false;
    }

    externalImageInfo.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    externalImageInfo.handleType = ExternalMemoryHandleType;
    imageInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    imageInfo.pNext = &externalImageInfo;
    imageInfo.format = InteropFormat;
    imageInfo.type = VK_IMAGE_TYPE_2D;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = InteropImageUsage;
    imageInfo.flags = 0;
    externalImageProperties.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
    imageProperties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    imageProperties.pNext = &externalImageProperties;
    result = vkGetPhysicalDeviceImageFormatProperties2(createInfo_.physicalDevice,
                                                        &imageInfo,
                                                        &imageProperties);
    if (result != VK_SUCCESS) {
        setError("External RGBA16F optimal image format is unsupported", result);
        return false;
    }
    if ((externalImageProperties.externalMemoryProperties.externalMemoryFeatures &
         VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0) {
        setError("RGBA16F optimal images cannot be exported with the required handle type");
        return false;
    }
    if (createInfo_.width > imageProperties.imageFormatProperties.maxExtent.width ||
        createInfo_.height > imageProperties.imageFormatProperties.maxExtent.height) {
        setError("Interop image extent exceeds Vulkan format limits");
        return false;
    }

    semaphoreInfo.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphoreInfo.handleType = ExternalSemaphoreHandleType;
    semaphoreProperties.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(createInfo_.physicalDevice,
                                                    &semaphoreInfo,
                                                    &semaphoreProperties);
    if ((semaphoreProperties.externalSemaphoreFeatures &
         VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0) {
        setError("Timeline semaphores cannot be exported with the required handle type");
        return false;
    }
    return true;
}

bool CudaVulkanInterop::loadExportFunctions()
{
    // vkGetDeviceProcAddr returns a generic function pointer; reinterpret_cast
    // to the concrete PFN type is the only way to bind these loaders.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
#if defined(_WIN32)
    getMemoryHandle_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(createInfo_.device, "vkGetMemoryWin32HandleKHR"));
    getSemaphoreHandle_ = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(createInfo_.device, "vkGetSemaphoreWin32HandleKHR"));
#else
    getMemoryHandle_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(createInfo_.device, "vkGetMemoryFdKHR"));
    getSemaphoreHandle_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(createInfo_.device, "vkGetSemaphoreFdKHR"));
#endif
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

    if (getMemoryHandle_ == nullptr || getSemaphoreHandle_ == nullptr) {
        setError("Vulkan external-handle export functions are unavailable; "
                 "enable the platform external-memory and external-semaphore device extensions");
        return false;
    }
    return true;
}

bool CudaVulkanInterop::createSlot(size_t slot)
{
    Slot *interopSlot = nullptr;

    interopSlot = &slots_[slot];
    if (!createImageResources(*interopSlot) ||
        !transitionImageToGeneral(*interopSlot) ||
        !importImageToCuda(*interopSlot) ||
        !createSemaphoreResources(*interopSlot) ||
        !importSemaphoreToCuda(*interopSlot)) {
        return false;
    }
    return true;
}

bool CudaVulkanInterop::createImageResources(Slot& slot)
{
    VkExternalMemoryImageCreateInfo externalImageInfo{};
    VkImageCreateInfo imageInfo{};
    VkMemoryRequirements memoryRequirements{};
    VkExportMemoryAllocateInfo exportMemoryInfo{};
    VkMemoryDedicatedAllocateInfo dedicatedMemoryInfo{};
    VkMemoryAllocateInfo allocationInfo{};
    uint32_t memoryTypeIndex = 0;
    VkResult result = VK_SUCCESS;

    externalImageInfo.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalImageInfo.handleTypes = ExternalMemoryHandleType;
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalImageInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = InteropFormat;
    imageInfo.extent.width = createInfo_.width;
    imageInfo.extent.height = createInfo_.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = InteropImageUsage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    result = vkCreateImage(createInfo_.device, &imageInfo, nullptr, &slot.image);
    if (result != VK_SUCCESS) {
        setError("vkCreateImage", result);
        return false;
    }

    vkGetImageMemoryRequirements(createInfo_.device, slot.image,
                                 &memoryRequirements);
    if (!selectMemoryType(memoryRequirements.memoryTypeBits, &memoryTypeIndex)) {
        return false;
    }

    dedicatedMemoryInfo.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedMemoryInfo.image = slot.image;
    exportMemoryInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportMemoryInfo.pNext = &dedicatedMemoryInfo;
    exportMemoryInfo.handleTypes = ExternalMemoryHandleType;
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.pNext = &exportMemoryInfo;
    allocationInfo.allocationSize = memoryRequirements.size;
    allocationInfo.memoryTypeIndex = memoryTypeIndex;
    result = vkAllocateMemory(createInfo_.device, &allocationInfo, nullptr,
                              &slot.memory);
    if (result != VK_SUCCESS) {
        setError("vkAllocateMemory", result);
        return false;
    }

    result = vkBindImageMemory(createInfo_.device, slot.image, slot.memory, 0);
    if (result != VK_SUCCESS) {
        setError("vkBindImageMemory", result);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::transitionImageToGeneral(Slot& slot)
{
    VkCommandPoolCreateInfo poolInfo{};
    VkCommandPool commandPool = nullptr;
    VkCommandBufferAllocateInfo allocationInfo{};
    VkCommandBuffer commandBuffer = nullptr;
    VkCommandBufferBeginInfo beginInfo{};
    VkImageMemoryBarrier barrier{};
    VkSubmitInfo submitInfo{};
    VkResult result = VK_SUCCESS;

    commandPool = VK_NULL_HANDLE;
    commandBuffer = VK_NULL_HANDLE;
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = createInfo_.queueFamilyIndex;
    result = vkCreateCommandPool(createInfo_.device, &poolInfo, nullptr,
                                 &commandPool);
    if (result != VK_SUCCESS) {
        setError("vkCreateCommandPool", result);
        return false;
    }

    allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocationInfo.commandPool = commandPool;
    allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocationInfo.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(createInfo_.device, &allocationInfo,
                                      &commandBuffer);
    if (result != VK_SUCCESS) {
        setError("vkAllocateCommandBuffers", result);
        vkDestroyCommandPool(createInfo_.device, commandPool, nullptr);
        return false;
    }

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        setError("vkBeginCommandBuffer", result);
        vkDestroyCommandPool(createInfo_.device, commandPool, nullptr);
        return false;
    }

    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = slot.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        setError("vkEndCommandBuffer", result);
        vkDestroyCommandPool(createInfo_.device, commandPool, nullptr);
        return false;
    }

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    result = vkQueueSubmit(createInfo_.queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(createInfo_.queue);
    }
    vkDestroyCommandPool(createInfo_.device, commandPool, nullptr);
    if (result != VK_SUCCESS) {
        setError("Vulkan image layout transition", result);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::importImageToCuda(Slot& slot)
{
    VkMemoryRequirements memoryRequirements{};
    cudaExternalMemoryHandleDesc memoryDescription{};
    cudaExternalMemoryMipmappedArrayDesc arrayDescription{};
    cudaChannelFormatDesc channelDescription{};
    cudaArray_t levelArray = nullptr;
    cudaResourceDesc surfaceDescription{};
    cudaError_t cudaStatus = cudaSuccess;
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    HANDLE handle;
    VkResult result = VK_SUCCESS;
#else
    VkMemoryGetFdInfoKHR handleInfo{};
    int handle = 0;
    VkResult result = VK_SUCCESS;
#endif

    vkGetImageMemoryRequirements(createInfo_.device, slot.image,
                                 &memoryRequirements);

#if defined(_WIN32)
    handle = nullptr;
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = slot.memory;
    handleInfo.handleType = ExternalMemoryHandleType;
    result = getMemoryHandle_(createInfo_.device, &handleInfo, &handle);
    if (result != VK_SUCCESS) {
        setError("vkGetMemoryWin32HandleKHR", result);
        return false;
    }
    memoryDescription.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memoryDescription.handle.win32.handle = handle;
#else
    handle = -1;
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handleInfo.memory = slot.memory;
    handleInfo.handleType = ExternalMemoryHandleType;
    result = getMemoryHandle_(createInfo_.device, &handleInfo, &handle);
    if (result != VK_SUCCESS) {
        setError("vkGetMemoryFdKHR", result);
        return false;
    }
    memoryDescription.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryDescription.handle.fd = handle;
#endif

    memoryDescription.size = memoryRequirements.size;
    memoryDescription.flags = cudaExternalMemoryDedicated;
    cudaStatus = cudaImportExternalMemory(&slot.cudaMemory, &memoryDescription);
#if defined(_WIN32)
    CloseHandle(handle);
#else
    if (cudaStatus != cudaSuccess) {
        close(handle);
    }
#endif
    if (cudaStatus != cudaSuccess) {
        setError("cudaImportExternalMemory", cudaStatus);
        return false;
    }

    channelDescription = cudaCreateChannelDesc(16, 16, 16, 16,
                                                cudaChannelFormatKindFloat);
    arrayDescription.offset = 0;
    arrayDescription.formatDesc = channelDescription;
    arrayDescription.extent.width = createInfo_.width;
    arrayDescription.extent.height = createInfo_.height;
    arrayDescription.extent.depth = 0;
    arrayDescription.numLevels = 1;
    arrayDescription.flags = cudaArraySurfaceLoadStore;
    cudaStatus = cudaExternalMemoryGetMappedMipmappedArray(
        &slot.mipmappedArray, slot.cudaMemory, &arrayDescription);
    if (cudaStatus != cudaSuccess) {
        setError("cudaExternalMemoryGetMappedMipmappedArray", cudaStatus);
        return false;
    }

    levelArray = nullptr;
    cudaStatus = cudaGetMipmappedArrayLevel(&levelArray, slot.mipmappedArray, 0);
    if (cudaStatus != cudaSuccess) {
        setError("cudaGetMipmappedArrayLevel", cudaStatus);
        return false;
    }
    surfaceDescription.resType = cudaResourceTypeArray;
    surfaceDescription.res.array.array = levelArray;
    cudaStatus = cudaCreateSurfaceObject(&slot.surface, &surfaceDescription);
    if (cudaStatus != cudaSuccess) {
        setError("cudaCreateSurfaceObject", cudaStatus);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::createSemaphoreResources(Slot& slot)
{
    VkSemaphoreTypeCreateInfo timelineInfo{};
    VkExportSemaphoreCreateInfo exportInfo{};
    VkSemaphoreCreateInfo semaphoreInfo{};
    VkResult result = VK_SUCCESS;

    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.pNext = &timelineInfo;
    exportInfo.handleTypes = ExternalSemaphoreHandleType;
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &exportInfo;
    result = vkCreateSemaphore(createInfo_.device, &semaphoreInfo, nullptr,
                               &slot.semaphore);
    if (result != VK_SUCCESS) {
        setError("vkCreateSemaphore for exported timeline semaphore", result);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::importSemaphoreToCuda(Slot& slot)
{
    cudaExternalSemaphoreHandleDesc semaphoreDescription{};
    cudaError_t cudaStatus = cudaSuccess;
#if defined(_WIN32)
    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    HANDLE handle;
    VkResult result = VK_SUCCESS;
#else
    VkSemaphoreGetFdInfoKHR handleInfo{};
    int handle = 0;
    VkResult result = VK_SUCCESS;
#endif

#if defined(_WIN32)
    handle = nullptr;
    handleInfo.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = slot.semaphore;
    handleInfo.handleType = ExternalSemaphoreHandleType;
    result = getSemaphoreHandle_(createInfo_.device, &handleInfo, &handle);
    if (result != VK_SUCCESS) {
        setError("vkGetSemaphoreWin32HandleKHR", result);
        return false;
    }
    semaphoreDescription.type =
        cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    semaphoreDescription.handle.win32.handle = handle;
#else
    handle = -1;
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    handleInfo.semaphore = slot.semaphore;
    handleInfo.handleType = ExternalSemaphoreHandleType;
    result = getSemaphoreHandle_(createInfo_.device, &handleInfo, &handle);
    if (result != VK_SUCCESS) {
        setError("vkGetSemaphoreFdKHR", result);
        return false;
    }
    semaphoreDescription.type =
        cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    semaphoreDescription.handle.fd = handle;
#endif

    cudaStatus =
        cudaImportExternalSemaphore(&slot.cudaSemaphore, &semaphoreDescription);
#if defined(_WIN32)
    CloseHandle(handle);
#else
    if (cudaStatus != cudaSuccess) {
        close(handle);
    }
#endif
    if (cudaStatus != cudaSuccess) {
        setError("cudaImportExternalSemaphore", cudaStatus);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::selectMemoryType(uint32_t memoryTypeBits,
                                         uint32_t *memoryTypeIndex)
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    uint32_t index = 0;

    if (memoryTypeIndex == nullptr) {
        setError("Memory type output pointer is null");
        return false;
    }
    vkGetPhysicalDeviceMemoryProperties(createInfo_.physicalDevice,
                                        &memoryProperties);
    for (index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if ((memoryTypeBits & (1u << index)) != 0 &&
            (memoryProperties.memoryTypes[index].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            *memoryTypeIndex = index;
            return true;
        }
    }
    setError("No device-local Vulkan memory type is available for the interop image");
    return false;
}

bool CudaVulkanInterop::selectCudaDevice()
{
    cudaError_t cudaStatus = cudaSuccess;

    cudaStatus = cudaSetDevice(createInfo_.cudaDevice);
    if (cudaStatus != cudaSuccess) {
        setError("cudaSetDevice", cudaStatus);
        return false;
    }
    return true;
}

bool CudaVulkanInterop::validateSlot(size_t slot)
{
    if (!initialized_) {
        setError("CUDA-Vulkan interop is not initialized");
        return false;
    }
    if (slot >= SlotCount) {
        setError("Interop slot index is out of range");
        return false;
    }
    return true;
}

void CudaVulkanInterop::destroyResources()
{
    size_t slot = 0;

    if (createInfo_.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(createInfo_.device);
    }
    if (createInfo_.physicalDevice != VK_NULL_HANDLE) {
        if (cudaSetDevice(createInfo_.cudaDevice) == cudaSuccess) {
            cudaDeviceSynchronize();
        }
    }

    for (slot = 0; slot < SlotCount; ++slot) {
        destroySlot(slots_[slot]);
    }

    initialized_ = false;
    createInfo_ = {};
#if defined(_WIN32)
    getMemoryHandle_ = nullptr;
    getSemaphoreHandle_ = nullptr;
#else
    getMemoryHandle_ = nullptr;
    getSemaphoreHandle_ = nullptr;
#endif
}

void CudaVulkanInterop::destroySlot(Slot& slot)
{
    if (slot.surface != 0) {
        cudaDestroySurfaceObject(slot.surface);
    }
    if (slot.mipmappedArray != nullptr) {
        cudaFreeMipmappedArray(slot.mipmappedArray);
    }
    if (slot.cudaMemory != nullptr) {
        cudaDestroyExternalMemory(slot.cudaMemory);
    }
    if (slot.cudaSemaphore != nullptr) {
        cudaDestroyExternalSemaphore(slot.cudaSemaphore);
    }
    if (slot.semaphore != VK_NULL_HANDLE &&
        createInfo_.device != VK_NULL_HANDLE) {
        vkDestroySemaphore(createInfo_.device, slot.semaphore, nullptr);
    }
    if (slot.image != VK_NULL_HANDLE &&
        createInfo_.device != VK_NULL_HANDLE) {
        vkDestroyImage(createInfo_.device, slot.image, nullptr);
    }
    if (slot.memory != VK_NULL_HANDLE &&
        createInfo_.device != VK_NULL_HANDLE) {
        vkFreeMemory(createInfo_.device, slot.memory, nullptr);
    }
    slot = {};
}

void CudaVulkanInterop::setError(const std::string& operation, VkResult result)
{
    std::ostringstream message;

    message << operation << " failed with VkResult "
            << static_cast<int>(result);
    lastError_ = message.str();
}

void CudaVulkanInterop::setError(const std::string& operation,
                                 cudaError_t result)
{
    const char *description = nullptr;
    std::ostringstream message;

    description = cudaGetErrorString(result);
    message << operation << " failed with CUDA error "
            << static_cast<int>(result);
    if (description != nullptr) {
        message << ": " << description;
    }
    lastError_ = message.str();
}

void CudaVulkanInterop::setError(const std::string& message)
{
    lastError_ = message;
}

} // namespace oka
