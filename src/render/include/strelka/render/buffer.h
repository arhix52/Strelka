#pragma once
#include <cassert>
#include <cstdint>
#include <vector>

namespace oka
{

enum class BufferFormat : char
{
    UNSIGNED_BYTE,
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
    virtual ~Buffer() = default;

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
    virtual void* getHostPointer()
    {
        return mHostData.data();
    }
    virtual size_t getHostDataSize()
    {
        return mHostData.size();
    }

    void* getDevicePointer()
    {
        return mDeviceData;
    }

    static size_t getElementSize(BufferFormat format)
    {
        switch (format)
        {
        case BufferFormat::UNSIGNED_BYTE:
            return sizeof(uint8_t);
            break;
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
    void* mDeviceData;
    size_t mWidth = 0u;
    size_t mHeight = 0u;
    BufferFormat mFormat;

    std::vector<char> mHostData;
};

struct ImageBuffer
{
    void* data = nullptr;
    void* deviceData = nullptr;
    size_t dataSize = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    BufferFormat pixel_format;
};

} // namespace oka
