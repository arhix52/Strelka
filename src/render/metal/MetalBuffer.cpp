
#include "MetalBuffer.h"

using namespace oka;

oka::MetalBuffer::MetalBuffer(MTL::Buffer* buff, BufferFormat format, uint32_t width, uint32_t height)
    : mBuffer(buff)
{
    mDeviceData = buff;
    mFormat = format;
    mWidth = width;
    mHeight = height;
}

oka::MetalBuffer::~MetalBuffer()
{
    mBuffer->release();
}

void oka::MetalBuffer::resize(uint32_t width, uint32_t height)
{
}

void* oka::MetalBuffer::map()
{
    return mBuffer->contents();
}

void oka::MetalBuffer::unmap()
{
}
