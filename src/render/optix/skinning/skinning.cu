#include "skinning.h"

// CUDA ядро для сложения двух векторов
__global__ void cuVectorAdd(const float *A, const float *B, float *C, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
    {
        C[index] = A[index] + B[index];
    }
}

void vectorAdd(float* A, float* B, float* C, int N, float* d_A, float* d_B, float* d_C)
{
    // Копирование данных на устройство
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Запуск ядра (1 блок, N потоков)
    cuVectorAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Копирование результата с устройства
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
}