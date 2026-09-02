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

enum class PresentationContent : uint32_t
{
    SceneLinear = 0,
    DebugDisplayLinear,
};

enum class PresentationResampling : uint32_t
{
    None = 0,
    Spatial,
};

/// Describes how a scene-linear frame becomes a display image.
///
/// SceneLinear content is multiplied by exposure, passed through tonemapper
/// with maxOutput as the curve's output headroom, then encoded with gamma when
/// gamma is positive. DebugDisplayLinear content bypasses all three operations.
struct PresentationMetadata
{
    float exposure[3] = { 1.0f, 1.0f, 1.0f };
    float maxOutput = 1.0f;
    float gamma = 0.0f;
    uint32_t tonemapper = 0;
    PresentationContent content = PresentationContent::SceneLinear;
    /// Dimensions of the valid scene-linear image at the start of the buffer.
    /// They normally match the published frame. A spatially upscaled frame is
    /// the exception: the buffer allocation is display-sized while the path
    /// tracer deliberately writes only this lower-resolution rectangle.
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    PresentationResampling resampling = PresentationResampling::None;
};

inline bool shouldApplyPresentationTransform(
    const PresentationMetadata& metadata)
{
    return metadata.content == PresentationContent::SceneLinear;
}

inline bool shouldTransformFrame(uint64_t frameSerial,
                                 uint64_t transformedFrameSerial)
{
    return frameSerial != 0 && frameSerial != transformedFrameSerial;
}

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
    void* mDeviceData = nullptr;
    size_t mWidth = 0u;
    size_t mHeight = 0u;
    BufferFormat mFormat = BufferFormat::UNSIGNED_BYTE;

    std::vector<char> mHostData;
};

struct ImageBuffer
{
    void* data = nullptr;
    void* deviceData = nullptr;
    /// A backend that renders straight into a texture puts it here, and the
    /// display uses it instead of copying deviceData into one of its own. Opaque
    /// so this header stays backend-agnostic.
    void* deviceTexture = nullptr;
    size_t dataSize = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    BufferFormat pixel_format = BufferFormat::UNSIGNED_BYTE;
    uint64_t frameSerial = 0;
    PresentationMetadata presentation{};
};

} // namespace oka
