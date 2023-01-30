#include "OptixBuffer.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

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
    const size_t bufferSize =mWidth * mHeight * getElementSize();
    mHostData.resize(bufferSize);
    cudaMemcpy(static_cast<void*>(mHostData.data()), mDeviceData, bufferSize, cudaMemcpyDeviceToHost);
    return nullptr;
}

void oka::OptixBuffer::unmap()
{
}
