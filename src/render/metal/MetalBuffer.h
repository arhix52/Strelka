#pragma once
#include "buffer.h"

#include <Metal/Metal.hpp>

namespace oka
{

class MetalBuffer : public Buffer
{
public:
    MetalBuffer(MTL::Buffer* buff, BufferFormat format, uint32_t width, uint32_t height);
    ~MetalBuffer() override;

    void resize(uint32_t width, uint32_t height) override;

    void* map() override;
    void unmap() override;

    MTL::Buffer* getNativePtr()
    {
        return mBuffer;
    }

    void* getHostPointer() override
    {
        return map();
    }

    size_t getHostDataSize() override
    {
        return mBuffer->length();
    }

protected:
    MTL::Buffer* mBuffer;
};
} // namespace oka