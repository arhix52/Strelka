#pragma once

#include "buffer.h"

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
    virtual ~OptixBuffer();

    size_t size()
    {
        return mSizeInBytes;
    }

    bool empty()
    {
        return mSizeInBytes == 0;
    }

    void resize(uint32_t width, uint32_t height) override;

    void realloc(size_t size);

    void* map() override;
    void unmap() override;

    void* getNativePtr()
    {
        return mDeviceData;
    }

    CUdeviceptr getPtr()
    {
        return (CUdeviceptr)mDeviceData;
    }

protected:
    void* mDeviceData = nullptr;
    size_t mSizeInBytes;
    uint32_t mDeviceIndex = 0;
};
} // namespace oka
