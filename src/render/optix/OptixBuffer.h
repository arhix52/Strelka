#pragma once

#include "buffer.h"

#include <stdint.h>
#include <vector>

namespace oka
{

class OptixBuffer : public Buffer
{
public:
    OptixBuffer(void* devicePtr, BufferFormat format, uint32_t width, uint32_t height);
    virtual ~OptixBuffer();

    void resize(uint32_t width, uint32_t height) override;

    void* map() override;
    void unmap() override;

    void* getNativePtr()
    {
        return mDeviceData;
    }

protected:
    void* mDeviceData = nullptr;
    uint32_t mDeviceIndex = 0;
};
} // namespace oka
