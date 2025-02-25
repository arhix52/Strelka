#pragma once
#include <log/log.h>

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        STRELKA_FATAL("CUDA call ({0}) failed with error: {1} {2}:{3}", call, cudaGetErrorString(error), file, line);
        assert(0);
    }
}

inline void cudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        STRELKA_FATAL("CUDA error on synchronize with error {0} , {1}:{2}", cudaGetErrorString(error), file, line);
        assert(0);
    }
}

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)
#define CUDA_SYNC_CHECK() cudaSyncCheck(__FILE__, __LINE__)

