#include "OptixBuffer.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_checks.h"

using namespace oka;

oka::OptixBuffer::OptixBuffer(const size_t size) : mSizeInBytes(size)
{
    mFormat = BufferFormat::UNSIGNED_BYTE;
    mWidth = size;
    mHeight = 1;
    void* devicePtr = nullptr;
    if (size > 0)
    {
        CUDA_CHECK(cudaMalloc(&devicePtr, size));
    }
    mDeviceData = devicePtr;
}

oka::OptixBuffer::OptixBuffer(void* devicePtr, BufferFormat format, uint32_t width, uint32_t height)
    : mSizeInBytes(static_cast<size_t>(width) * height)
{
    mDeviceData = devicePtr;
    mFormat = format;
    mWidth = width;
    mHeight = height;
    // getElementSize() reads mFormat, so it cannot run in the initializer list
    // above -- the base's mFormat is not assigned until this line.
    mSizeInBytes *= getElementSize();
}

oka::OptixBuffer::~OptixBuffer()
{
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
    mSizeInBytes = static_cast<size_t>(mWidth) * mHeight * getElementSize();
    CUDA_CHECK(cudaMalloc(&mDeviceData, mSizeInBytes));
}

void oka::OptixBuffer::realloc(size_t size)
{
    if (mSizeInBytes == size && mDeviceData)
        return;
    if (mDeviceData)
    {
        CUDA_CHECK(cudaFree(mDeviceData));
        mDeviceData = nullptr;
    }
    mSizeInBytes = size;
    if (size > 0)
        CUDA_CHECK(cudaMalloc(&mDeviceData, size));
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
