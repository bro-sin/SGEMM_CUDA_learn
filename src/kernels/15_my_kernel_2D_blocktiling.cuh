#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void my_sgemm2DBlocktiling(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
    const dim3 small_C_dim = {CEIL_DIV(BN, TN), CEIL_DIV(BM, TM)};
    /*
    small_C_dim是这个block要计算的C(M*N)的一个小块(BM*BN)按照TM*TN分块之后的维度
    */

    const uint threadRow = threadIdx.x / small_C_dim.x;
    const uint threadColumn = threadIdx.x % small_C_dim.x;
    /*
    这个线程计算的小矩阵的行数和列数范围是
    行：[threadRow * TM, threadRow * TM + TM)
    列：[threadColumn * TN, threadColumn * TN + TN)
    */
    float threadResults[TM * TN] = {0.0}; // 这个线程计算的小矩阵的所有元素

    const uint currentRow = blockIdx.y;
    const uint currentColumn = blockIdx.x;
    /*
    currentRow和currentColumn是这个block要计算的C(M*N)的一个小块(BM*BN)的这个矩阵的索引
    */

    A += currentRow * BM * K;
    B += currentColumn * BN;
    C += currentRow * BM * N + currentColumn * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColumnA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColumnB = threadIdx.x % BN;

    const uint outer_dot_nums = CEIL_DIV(K, BK);
    for (uint outer_dot_index = 0; outer_dot_index < outer_dot_nums; outer_dot_index++)
    {
        As[innerRowA * BK + innerColumnA] = A[innerRowA * K + innerColumnA];
        Bs[innerRowB * BN + innerColumnB] = B[innerRowB * N + innerColumnB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
        {
            for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
            {
                for (uint inner_dot_index = 0; inner_dot_index < BK; inner_dot_index++)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        As[(threadRow * TM + resIdxM) * BK + inner_dot_index] *
                        Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
        {
            C[(threadRow * TM + resIdxM) * N + threadColumn * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadColumn * TN + resIdxN];
        }
    }
}