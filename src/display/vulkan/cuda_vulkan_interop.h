#pragma once

#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace oka
{

class CudaVulkanInterop
{
public:
    static constexpr size_t SlotCount = 2;

    struct CreateInfo
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        // Queue zero is fetched from queueFamilyIndex when queue is null.
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        int cudaDevice = 0;
        cudaStream_t cudaStream = nullptr;
    };

    CudaVulkanInterop() = default;
    ~CudaVulkanInterop();

    CudaVulkanInterop(const CudaVulkanInterop&) = delete;
    CudaVulkanInterop& operator=(const CudaVulkanInterop&) = delete;
    CudaVulkanInterop(CudaVulkanInterop&&) = delete;
    CudaVulkanInterop& operator=(CudaVulkanInterop&&) = delete;

    bool initialize(const CreateInfo& createInfo);
    void shutdown();

    bool isInitialized() const;
    const std::string& lastError() const;

    VkImage image(size_t slot) const;
    VkDeviceMemory imageMemory(size_t slot) const;
    cudaMipmappedArray_t cudaMipmappedArray(size_t slot) const;
    cudaSurfaceObject_t cudaSurface(size_t slot) const;

    VkSemaphore timelineSemaphore(size_t slot) const;
    uint64_t readyValue(size_t slot) const;
    uint64_t retireValue(size_t slot) const;

    // Reserve ready before retire for each use of a slot. Vulkan waits for
    // ready and signals retire; CUDA signals ready and waits for retire.
    bool reserveReadyValue(size_t slot, uint64_t *value);
    bool reserveRetireValue(size_t slot, uint64_t *value);

    bool cudaWait(size_t slot, uint64_t value, cudaStream_t stream = nullptr);
    bool cudaSignal(size_t slot, uint64_t value, cudaStream_t stream = nullptr);
    bool cudaWaitForRetire(size_t slot, cudaStream_t stream = nullptr);
    bool cudaSignalReady(size_t slot, cudaStream_t stream = nullptr);

    VkFormat format() const;
    VkImageLayout imageLayout() const;
    VkExtent2D extent() const;
    uint32_t queueFamilyIndex() const;

private:
    struct Slot
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        cudaExternalMemory_t cudaMemory = nullptr;
        cudaMipmappedArray_t mipmappedArray = nullptr;
        cudaSurfaceObject_t surface = 0;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        cudaExternalSemaphore_t cudaSemaphore = nullptr;
        uint64_t ready = 0;
        uint64_t retire = 0;
        uint64_t nextValue = 0;
        bool readyReserved = false;
    };

    bool validateCreateInfo(const CreateInfo& createInfo);
    bool verifyDeviceUuid();
    bool probeVulkanSupport();
    bool loadExportFunctions();
    bool createSlot(size_t slot);
    bool createImageResources(Slot& slot);
    bool transitionImageToGeneral(Slot& slot);
    bool importImageToCuda(Slot& slot);
    bool createSemaphoreResources(Slot& slot);
    bool importSemaphoreToCuda(Slot& slot);
    bool selectMemoryType(uint32_t memoryTypeBits, uint32_t *memoryTypeIndex);
    bool selectCudaDevice();
    bool validateSlot(size_t slot);
    void destroyResources();
    void destroySlot(Slot& slot);
    void setError(const std::string& operation, VkResult result);
    void setError(const std::string& operation, cudaError_t result);
    void setError(const std::string& message);

    CreateInfo createInfo_{};
    std::array<Slot, SlotCount> slots_{};
    std::string lastError_;
    bool initialized_ = false;

#if defined(_WIN32)
    PFN_vkGetMemoryWin32HandleKHR getMemoryHandle_ = nullptr;
    PFN_vkGetSemaphoreWin32HandleKHR getSemaphoreHandle_ = nullptr;
#else
    PFN_vkGetMemoryFdKHR getMemoryHandle_ = nullptr;
    PFN_vkGetSemaphoreFdKHR getSemaphoreHandle_ = nullptr;
#endif
};

} // namespace oka
