#include "OptixBuffer.h"

using namespace oka;

oka::OptixBuffer::OptixBuffer(void* devicePtr, BufferFormat format, uint32_t width, uint32_t height)
{
    mDeviceData = devicePtr;
    mFormat = format;
    mWidth = width;
    mHeight = height;
}

oka::OptixBuffer::~OptixBuffer()
{
    // TODO:
}

void oka::OptixBuffer::resize(uint32_t width, uint32_t height)
{
}

void* oka::OptixBuffer::map()
{
    return nullptr;
}

void oka::OptixBuffer::unmap()
{
}
