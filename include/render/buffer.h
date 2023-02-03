#pragma once
#include "common.h"
#include <stdint.h>
#include <vector>

namespace oka
{

enum class BufferFormat : char
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct BufferDesc
{
    uint32_t width;
    uint32_t height;
    BufferFormat format;
};

class Buffer
{
public:
    virtual ~Buffer(){};

    virtual void resize(uint32_t width, uint32_t height) = 0;

    virtual void* map() = 0;
    virtual void unmap() = 0;

    uint32_t width() const
    {
        return mWidth;
    }
    uint32_t height() const
    {
        return mHeight;
    }

    // Get output buffer
    void* getHostPointer()
    {
        return mHostData.data();
    }
    size_t getHostDataSize()
    {
        return mHostData.size();
    }

    static size_t getElementSize(BufferFormat format)
    {
        switch (format)
        {
        case BufferFormat::FLOAT4:
            return 4 * sizeof(float);
            break;
        case BufferFormat::FLOAT3:
            return 3 * sizeof(float);
            break;
        case BufferFormat::UNSIGNED_BYTE4:
            return 4 * sizeof(char);
            break;
        default:
            break;
        }
        assert(0);
        return 0;
    }

    size_t getElementSize() const
    {
        return Buffer::getElementSize(mFormat);
    }

    BufferFormat getFormat() const
    {
        return mFormat;
    }

protected:
    uint32_t mWidth = 0u;
    uint32_t mHeight = 0u;
    BufferFormat mFormat;

    std::vector<char> mHostData;
};

struct ImageBuffer
{
    void* data = nullptr;
    size_t dataSize = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    BufferFormat pixel_format;
};

} // namespace oka
