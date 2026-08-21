#include "cuda_vulkan_interop.h"
#include "postprocessing/Tonemappers.h"

#include <cuda_runtime_api.h>
#include <cmath>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr int SkipResult = 77;
constexpr float ValueTolerance = 0.002f;

struct VulkanContext
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;

    ~VulkanContext()
    {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

struct ReadbackBuffer
{
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;

    ~ReadbackBuffer()
    {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
        }
    }
};

bool hasExtension(const std::vector<VkExtensionProperties>& extensions,
                  const char *name)
{
    size_t index = 0;

    for (index = 0; index < extensions.size(); ++index) {
        if (std::strcmp(extensions[index].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

bool enumerateDeviceExtensions(
    VkPhysicalDevice physicalDevice,
    std::vector<VkExtensionProperties> *extensions)
{
    uint32_t count = 0;
    VkResult result = VK_SUCCESS;

    count = 0;
    result = vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr,
                                                   &count, nullptr);
    if (result != VK_SUCCESS) {
        return false;
    }
    extensions->resize(count);
    result = vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &count, extensions->data());
    return result == VK_SUCCESS;
}

bool selectHostMemoryType(VkPhysicalDevice physicalDevice,
                          uint32_t memoryTypeBits,
                          uint32_t *memoryTypeIndex)
{
    VkPhysicalDeviceMemoryProperties properties{};
    uint32_t index = 0;

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
    for (index = 0; index < properties.memoryTypeCount; ++index) {
        if ((memoryTypeBits & (1u << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            *memoryTypeIndex = index;
            return true;
        }
    }
    return false;
}

bool createReadbackBuffer(const VulkanContext& context,
                          uint32_t width,
                          uint32_t height,
                          ReadbackBuffer *readback,
                          std::string *error)
{
    VkBufferCreateInfo bufferInfo{};
    VkMemoryRequirements requirements{};
    VkMemoryAllocateInfo allocationInfo{};
    uint32_t memoryTypeIndex = 0;
    VkResult result = VK_SUCCESS;

    readback->device = context.device;
    readback->size = static_cast<VkDeviceSize>(width) * height * 8;
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = readback->size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vkCreateBuffer(context.device, &bufferInfo, nullptr,
                            &readback->buffer);
    if (result != VK_SUCCESS) {
        *error = "vkCreateBuffer failed";
        return false;
    }

    vkGetBufferMemoryRequirements(context.device, readback->buffer,
                                  &requirements);
    if (!selectHostMemoryType(context.physicalDevice,
                              requirements.memoryTypeBits,
                              &memoryTypeIndex)) {
        *error = "no host-visible Vulkan memory type";
        return false;
    }

    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = memoryTypeIndex;
    result = vkAllocateMemory(context.device, &allocationInfo, nullptr,
                              &readback->memory);
    if (result != VK_SUCCESS) {
        *error = "vkAllocateMemory for readback failed";
        return false;
    }
    result = vkBindBufferMemory(context.device, readback->buffer,
                                readback->memory, 0);
    if (result != VK_SUCCESS) {
        *error = "vkBindBufferMemory failed";
        return false;
    }
    return true;
}

float halfToFloat(uint16_t value)
{
    uint32_t sign = 0;
    int32_t exponent = 0;
    uint32_t mantissa = 0;
    uint32_t bits = 0;
    float result = NAN;

    sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    exponent = static_cast<int32_t>((value >> 10) & 0x1fu);
    mantissa = value & 0x03ffu;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<uint32_t>(exponent + 112) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float sourceValue(uint32_t x, uint32_t y, uint32_t channel)
{
    uint32_t selector = 0;

    selector = (x * 3u + y * 5u + channel * 7u) % 13u;
    return 0.03125f * static_cast<float>(selector + 1u);
}

void fillSource(std::vector<float4> *pixels, uint32_t width, uint32_t height)
{
    size_t index = 0;
    uint32_t x = 0;
    uint32_t y = 0;

    pixels->resize(static_cast<size_t>(width) * height);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            index = static_cast<size_t>(y) * width + x;
            (*pixels)[index] =
                make_float4(sourceValue(x, y, 0), sourceValue(x, y, 1),
                            sourceValue(x, y, 2), sourceValue(x, y, 3));
        }
    }
}

bool verifyReadback(const ReadbackBuffer& readback,
                    uint32_t width,
                    uint32_t height,
                    const oka::PresentationMetadata& metadata,
                    std::string *error)
{
    VkMappedMemoryRange range{};
    const uint16_t *encoded = nullptr;
    // vkMapMemory writes through &mapped, so it must stay a mutable void*.
    // NOLINTNEXTLINE(misc-const-correctness)
    void *mapped = nullptr;
    size_t pixelIndex = 0;
    size_t channel = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    float expected = NAN;
    float actual = NAN;
    VkResult result = VK_SUCCESS;

    mapped = nullptr;
    result = vkMapMemory(readback.device, readback.memory, 0, VK_WHOLE_SIZE, 0,
                         &mapped);
    if (result != VK_SUCCESS) {
        *error = "vkMapMemory failed";
        return false;
    }
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = readback.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    result = vkInvalidateMappedMemoryRanges(readback.device, 1, &range);
    if (result != VK_SUCCESS) {
        vkUnmapMemory(readback.device, readback.memory);
        *error = "vkInvalidateMappedMemoryRanges failed";
        return false;
    }

    encoded = static_cast<const uint16_t *>(mapped);
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            pixelIndex = static_cast<size_t>(y) * width + x;
            for (channel = 0; channel < 4; ++channel) {
                expected = channel == 3
                               ? 1.0f
                               : sourceValue(x, y,
                                             static_cast<uint32_t>(channel)) *
                                     metadata.exposure[channel];
                actual = halfToFloat(encoded[pixelIndex * 4 + channel]);
                if (std::fabs(actual - expected) > ValueTolerance) {
                    *error = "RGBA16F readback mismatch at (" +
                             std::to_string(x) + ", " + std::to_string(y) +
                             "), channel " + std::to_string(channel) +
                             ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual);
                    vkUnmapMemory(readback.device, readback.memory);
                    return false;
                }
            }
        }
    }
    vkUnmapMemory(readback.device, readback.memory);
    return true;
}

bool recordAndSubmitReadback(const VulkanContext& context,
                             const oka::CudaVulkanInterop& interop,
                             const ReadbackBuffer& readback,
                             uint64_t readyValue,
                             uint64_t retireValue,
                             std::string *error)
{
    VkCommandBufferAllocateInfo allocationInfo{};
    VkCommandBuffer commandBuffer = nullptr;
    VkCommandBufferBeginInfo beginInfo{};
    VkImageMemoryBarrier toTransfer{};
    VkBufferImageCopy copyRegion{};
    VkBufferMemoryBarrier toHost{};
    VkImageMemoryBarrier toGeneral{};
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    VkPipelineStageFlags waitStage = 0;
    VkSemaphore semaphore = nullptr;
    VkSubmitInfo submitInfo{};
    VkResult result = VK_SUCCESS;

    commandBuffer = VK_NULL_HANDLE;
    allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocationInfo.commandPool = context.commandPool;
    allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocationInfo.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(context.device, &allocationInfo,
                                      &commandBuffer);
    if (result != VK_SUCCESS) {
        *error = "vkAllocateCommandBuffers failed";
        return false;
    }

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        *error = "vkBeginCommandBuffer failed";
        vkFreeCommandBuffers(context.device, context.commandPool, 1,
                             &commandBuffer);
        return false;
    }

    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = interop.image(0);
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toTransfer);

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent.width = interop.extent().width;
    copyRegion.imageExtent.height = interop.extent().height;
    copyRegion.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(commandBuffer, interop.image(0),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &copyRegion);

    toHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.buffer = readback.buffer;
    toHost.offset = 0;
    toHost.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &toHost, 0, nullptr);

    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toGeneral.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = interop.image(0);
    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneral.subresourceRange.levelCount = 1;
    toGeneral.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toGeneral);
    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        *error = "vkEndCommandBuffer failed";
        vkFreeCommandBuffers(context.device, context.commandPool, 1,
                             &commandBuffer);
        return false;
    }

    semaphore = interop.timelineSemaphore(0);
    waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = 1;
    timelineInfo.pWaitSemaphoreValues = &readyValue;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &retireValue;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &semaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphore;
    result = vkQueueSubmit(context.queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(context.queue);
    }
    vkFreeCommandBuffers(context.device, context.commandPool, 1,
                         &commandBuffer);
    if (result != VK_SUCCESS) {
        *error = "timeline readback submission failed";
        return false;
    }
    return true;
}

bool runExtent(const VulkanContext& context,
               int cudaDevice,
               cudaStream_t stream,
               uint32_t width,
               uint32_t height,
               std::string *error)
{
    oka::CudaVulkanInterop interop;
    oka::CudaVulkanInterop::CreateInfo createInfo{};
    ReadbackBuffer readback;
    std::vector<float4> hostSource;
    oka::PresentationMetadata metadata{};
    float4 *deviceSource = nullptr;
    size_t sourceSize = 0;
    uint64_t readyValue = 0;
    uint64_t retireValue = 0;
    cudaError_t cudaStatus = cudaSuccess;
    bool success = false;

    deviceSource = nullptr;
    success = false;
    createInfo.instance = context.instance;
    createInfo.physicalDevice = context.physicalDevice;
    createInfo.device = context.device;
    createInfo.queue = context.queue;
    createInfo.queueFamilyIndex = context.queueFamilyIndex;
    createInfo.width = width;
    createInfo.height = height;
    createInfo.cudaDevice = cudaDevice;
    createInfo.cudaStream = stream;
    if (!interop.initialize(createInfo)) {
        *error = interop.lastError();
        return false;
    }
    if (!createReadbackBuffer(context, width, height, &readback, error)) {
        return false;
    }

    fillSource(&hostSource, width, height);
    sourceSize = hostSource.size() * sizeof(float4);
    // cudaMalloc's out-parameter is void**; the typed pointer needs a cast.
    cudaStatus = cudaMalloc(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<void **>(&deviceSource), sourceSize);
    if (cudaStatus != cudaSuccess) {
        *error = std::string("cudaMalloc failed: ") +
                 cudaGetErrorString(cudaStatus);
        return false;
    }
    cudaStatus = cudaMemcpyAsync(deviceSource, hostSource.data(), sourceSize,
                                 cudaMemcpyHostToDevice, stream);
    metadata.exposure[0] = 0.5f;
    metadata.exposure[1] = 1.25f;
    metadata.exposure[2] = 2.0f;
    metadata.maxOutput = 4.0f;
    metadata.gamma = 0.0f;
    metadata.tonemapper = 0;
    metadata.content = oka::PresentationContent::SceneLinear;
    if (cudaStatus == cudaSuccess) {
        cudaStatus = tonemapToSurfaceLinear(
            deviceSource, interop.cudaSurface(0), width, height, &metadata,
            stream);
    }
    if (cudaStatus != cudaSuccess) {
        *error = std::string("CUDA upload/tonemap failed: ") +
                 cudaGetErrorString(cudaStatus);
    } else if (!interop.reserveReadyValue(0, &readyValue) ||
               !interop.reserveRetireValue(0, &retireValue) ||
               readyValue == 0 || retireValue <= readyValue) {
        *error = interop.lastError().empty()
                     ? "invalid ready/retire timeline values"
                     : interop.lastError();
        // NOLINTNEXTLINE(bugprone-branch-clone)
    } else if (!interop.cudaSignalReady(0, stream)) {
        *error = interop.lastError();
    } else if (!recordAndSubmitReadback(context, interop, readback, readyValue,
                                        retireValue, error)) {
    } else if (!interop.cudaWaitForRetire(0, stream)) {
        *error = interop.lastError();
    } else {
        cudaStatus = cudaStreamSynchronize(stream);
        if (cudaStatus != cudaSuccess) {
            *error = std::string("cudaStreamSynchronize failed: ") +
                     cudaGetErrorString(cudaStatus);
        } else {
            success = verifyReadback(readback, width, height, metadata, error);
        }
    }

    cudaFree(deviceSource);
    return success;
}

int initializeVulkanForCuda(int cudaDevice,
                            VulkanContext *context,
                            std::string *reason)
{
    cudaDeviceProp cudaProperties{};
    VkApplicationInfo applicationInfo{};
    VkInstanceCreateInfo instanceInfo{};
    uint32_t physicalDeviceCount = 0;
    std::vector<VkPhysicalDevice> physicalDevices;
    VkPhysicalDeviceIDProperties idProperties{};
    VkPhysicalDeviceProperties2 properties{};
    VkPhysicalDeviceProperties selectedProperties{};
    uint32_t deviceIndex = 0;
    uint32_t queueFamilyCount = 0;
    std::vector<VkQueueFamilyProperties> queueFamilies;
    uint32_t queueIndex = 0;
    std::vector<VkExtensionProperties> extensions;
    std::vector<const char *> enabledExtensions;
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    VkPhysicalDeviceFeatures2 features{};
    float queuePriority = NAN;
    VkDeviceQueueCreateInfo queueInfo{};
    VkDeviceCreateInfo deviceInfo{};
    VkCommandPoolCreateInfo poolInfo{};
    cudaError_t cudaStatus = cudaSuccess;
    VkResult result = VK_SUCCESS;
    bool timelineCore = false;

    cudaStatus = cudaGetDeviceProperties(&cudaProperties, cudaDevice);
    if (cudaStatus != cudaSuccess) {
        *reason = std::string("CUDA device properties unavailable: ") +
                  cudaGetErrorString(cudaStatus);
        return SkipResult;
    }

    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Strelka Vulkan CUDA interop smoke";
    applicationInfo.apiVersion = VK_API_VERSION_1_2;
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &applicationInfo;
    result = vkCreateInstance(&instanceInfo, nullptr, &context->instance);
    if (result != VK_SUCCESS) {
        *reason = "Vulkan 1.2 instance unavailable";
        return SkipResult;
    }

    physicalDeviceCount = 0;
    result = vkEnumeratePhysicalDevices(context->instance,
                                        &physicalDeviceCount, nullptr);
    if (result != VK_SUCCESS || physicalDeviceCount == 0) {
        *reason = "no Vulkan physical device";
        return SkipResult;
    }
    physicalDevices.resize(physicalDeviceCount);
    result = vkEnumeratePhysicalDevices(context->instance,
                                        &physicalDeviceCount,
                                        physicalDevices.data());
    if (result != VK_SUCCESS) {
        *reason = "Vulkan physical-device enumeration failed";
        return SkipResult;
    }
    for (deviceIndex = 0; deviceIndex < physicalDeviceCount; ++deviceIndex) {
        idProperties = {};
        properties = {};
        idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &idProperties;
        vkGetPhysicalDeviceProperties2(physicalDevices[deviceIndex],
                                       &properties);
        if (std::memcmp(idProperties.deviceUUID, cudaProperties.uuid.bytes,
                        VK_UUID_SIZE) == 0) {
            context->physicalDevice = physicalDevices[deviceIndex];
            selectedProperties = properties.properties;
            break;
        }
    }
    if (context->physicalDevice == VK_NULL_HANDLE) {
        *reason = "no Vulkan physical device matches the current CUDA UUID";
        return SkipResult;
    }

    queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice,
                                              &queueFamilyCount, nullptr);
    queueFamilies.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(context->physicalDevice,
                                              &queueFamilyCount,
                                              queueFamilies.data());
    context->queueFamilyIndex = queueFamilyCount;
    for (queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
        if (queueFamilies[queueIndex].queueCount != 0 &&
            (queueFamilies[queueIndex].queueFlags &
             VK_QUEUE_TRANSFER_BIT) != 0) {
            context->queueFamilyIndex = queueIndex;
            break;
        }
    }
    if (context->queueFamilyIndex == queueFamilyCount) {
        *reason = "matching Vulkan device has no transfer queue";
        return SkipResult;
    }
    if (!enumerateDeviceExtensions(context->physicalDevice, &extensions) ||
        !hasExtension(extensions,
                      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) ||
        !hasExtension(extensions,
                      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
        *reason = "required Vulkan external FD extensions unavailable";
        return SkipResult;
    }
    enabledExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    enabledExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    timelineCore =
        VK_API_VERSION_MAJOR(selectedProperties.apiVersion) > 1 ||
        (VK_API_VERSION_MAJOR(selectedProperties.apiVersion) == 1 &&
         VK_API_VERSION_MINOR(selectedProperties.apiVersion) >= 2);
    if (!timelineCore) {
        if (!hasExtension(extensions,
                          VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
            *reason = "Vulkan timeline semaphore extension unavailable";
            return SkipResult;
        }
        enabledExtensions.push_back(
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    timelineFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &timelineFeatures;
    vkGetPhysicalDeviceFeatures2(context->physicalDevice, &features);
    if (timelineFeatures.timelineSemaphore != VK_TRUE) {
        *reason = "Vulkan timeline semaphore feature unavailable";
        return SkipResult;
    }

    queuePriority = 1.0f;
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = context->queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &timelineFeatures;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledExtensions.size());
    deviceInfo.ppEnabledExtensionNames = enabledExtensions.data();
    result = vkCreateDevice(context->physicalDevice, &deviceInfo, nullptr,
                            &context->device);
    if (result != VK_SUCCESS) {
        *reason = "Vulkan device creation with external interop failed";
        return SkipResult;
    }
    vkGetDeviceQueue(context->device, context->queueFamilyIndex, 0,
                     &context->queue);

    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = context->queueFamilyIndex;
    result = vkCreateCommandPool(context->device, &poolInfo, nullptr,
                                 &context->commandPool);
    if (result != VK_SUCCESS) {
        *reason = "Vulkan command-pool creation failed";
        return SkipResult;
    }
    return 0;
}

} // namespace

int main()
{
    VulkanContext context;
    cudaStream_t stream = nullptr;
    int cudaDevice = 0;
    int cudaDeviceCount = 0;
    int initializationResult = 0;
    cudaError_t cudaStatus = cudaSuccess;
    std::string message;

    stream = nullptr;
    cudaDevice = 0;
    cudaDeviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaStatus != cudaSuccess || cudaDeviceCount == 0) {
        std::cout << "SKIP: CUDA device unavailable";
        if (cudaStatus != cudaSuccess) {
            std::cout << ": " << cudaGetErrorString(cudaStatus);
        }
        std::cout << '\n';
        return SkipResult;
    }
    cudaStatus = cudaGetDevice(&cudaDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "SKIP: current CUDA device unavailable: "
                  << cudaGetErrorString(cudaStatus) << '\n';
        return SkipResult;
    }

    initializationResult =
        initializeVulkanForCuda(cudaDevice, &context, &message);
    if (initializationResult != 0) {
        std::cout << "SKIP: " << message << '\n';
        return initializationResult;
    }
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        std::cout << "SKIP: CUDA stream unavailable: "
                  << cudaGetErrorString(cudaStatus) << '\n';
        return SkipResult;
    }

    if (!runExtent(context, cudaDevice, stream, 37, 23, &message)) {
        std::cerr << "FAIL: first extent: " << message << '\n';
        cudaStreamDestroy(stream);
        return 1;
    }
    if (!runExtent(context, cudaDevice, stream, 91, 47, &message)) {
        std::cerr << "FAIL: recreated extent: " << message << '\n';
        cudaStreamDestroy(stream);
        return 1;
    }

    cudaStreamDestroy(stream);
    std::cout << "PASS: Vulkan-CUDA interop smoke completed two extents\n";
    return 0;
}
