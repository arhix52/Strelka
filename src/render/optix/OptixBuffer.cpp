#include "OptixBuffer.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_checks.h"

using namespace oka;

oka::OptixBuffer::OptixBuffer(const size_t size)
{
    mFormat = BufferFormat::UNSIGNED_BYTE;
    mWidth = size;
    mHeight = 1;
    mSizeInBytes = size;
    void* devicePtr = nullptr;
    if (size > 0)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
    }
    mDeviceData = devicePtr;
}

oka::OptixBuffer::OptixBuffer(void* devicePtr, BufferFormat format, uint32_t width, uint32_t height)
{
    mDeviceData = devicePtr;
    mFormat = format;
    mWidth = width;
    mHeight = height;
    mSizeInBytes = mWidth * mHeight * getElementSize();
}

oka::OptixBuffer::~OptixBuffer()
{
    // TODO:
    if (mDeviceData)
    {
        CUDA_CHECK(cudaFree(mDeviceData));
    }
}

void oka::OptixBuffer::resize(uint32_t width, uint32_t height)
{
    if (mDeviceData)
    {
        CUDA_CHECK(cudaFree(mDeviceData));
    }
    mWidth = width;
    mHeight = height;
    mSizeInBytes = mWidth * mHeight * getElementSize();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mDeviceData), mSizeInBytes));
}

void oka::OptixBuffer::realloc(size_t size)
{
    if (mDeviceData && mSizeInBytes != size)
    {
        CUDA_CHECK(cudaFree(mDeviceData));
    }
    mSizeInBytes = size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mDeviceData), mSizeInBytes));
}

void* oka::OptixBuffer::map()
{
    mHostData.resize(mSizeInBytes);
    CUDA_CHECK(cudaMemcpy(static_cast<void*>(mHostData.data()), mDeviceData, mSizeInBytes, cudaMemcpyDeviceToHost));
    return mHostData.data();
}

void oka::OptixBuffer::unmap()
{
    assert(mHostData.size() == mSizeInBytes);
    CUDA_CHECK(cudaMemcpy(mDeviceData, static_cast<void*>(mHostData.data()), mSizeInBytes, cudaMemcpyHostToDevice));
}
