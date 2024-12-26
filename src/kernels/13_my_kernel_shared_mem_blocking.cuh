#pragma once
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void my_sgemm_shared_mem_block(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
}