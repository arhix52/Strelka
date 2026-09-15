#pragma once

#include <strelka/render/buffer.h>

#include <optix_types.h>
#include <stdint.h>

namespace oka
{

class OptixBuffer : public Buffer
{
public:
    // Creates linear buffer
    OptixBuffer(size_t size);
    // Creates two dimensional buffer
    OptixBuffer(void* devicePtr, BufferFormat format, uint32_t width, uint32_t height);
    ~OptixBuffer() override;

    /// Const because the memory report walks every buffer the renderer owns from
    /// a const method, and a size that cannot be read without permission to
    /// modify is a size nobody can report.
    size_t size() const
    {
        return mSizeInBytes;
    }

    bool empty() const
    {
        return mSizeInBytes == 0;
    }

    void resize(uint32_t width, uint32_t height) override;

    void realloc(size_t size);

    void* map() override;
    void unmap() override;

    void* getHostPointer() override
    {
        return map();
    }

    void* getNativePtr()
    {
        return mDeviceData;
    }

    CUdeviceptr getPtr()
    {
        return (CUdeviceptr)mDeviceData;
    }

protected:
    size_t mSizeInBytes;
    uint32_t mDeviceIndex = 0;
};
} // namespace oka
