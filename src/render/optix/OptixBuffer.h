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

    /// Device-to-host copy, then the host copy.
    ///
    /// The base class hands back mHostData.data() unconditionally, which on Metal
    /// is right -- there the buffer is shared memory and the pointer is always
    /// live. Here mHostData stays empty until something calls map(), so a caller
    /// that only ever asked for getHostPointer() -- HeadlessApp::saveOutput, and
    /// therefore every headless render -- read from an empty vector and crashed
    /// on the way out with the image already computed.
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
