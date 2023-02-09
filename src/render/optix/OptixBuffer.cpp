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
    if (mDeviceData)
    {
        cudaFree(mDeviceData);
    }
}

void oka::OptixBuffer::resize(uint32_t width, uint32_t height)
{
    if (mDeviceData)
    {
        cudaFree(mDeviceData);
    }
    mWidth = width;
    mHeight = height;
    const size_t bufferSize = mWidth * mHeight * getElementSize();
    cudaMalloc(reinterpret_cast<void**>(&mDeviceData), bufferSize);
}

void* oka::OptixBuffer::map()
{
    const size_t bufferSize = mWidth * mHeight * getElementSize();
    mHostData.resize(bufferSize);
    cudaMemcpy(static_cast<void*>(mHostData.data()), mDeviceData, bufferSize, cudaMemcpyDeviceToHost);
    return nullptr;
}

void oka::OptixBuffer::unmap()
{
}
