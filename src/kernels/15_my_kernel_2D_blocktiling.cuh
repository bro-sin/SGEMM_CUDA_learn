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
    const uint global_A_row = currentRow * BM;
    uint global_A_col = 0;
    B += currentColumn * BN;
    uint global_B_row = 0;
    const uint global_B_col = currentColumn * BN;

    C += currentRow * BM * N + currentColumn * BN;
    const uint global_C_row = currentRow * BM;
    const uint global_C_col = currentColumn * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    /*
    一个block的线程一共有small_C_dim.x * small_C_dim.y个,
    这个值会比BM*BK以及BK*BN小，因此没办法每个线程加载一个元素就将所有元素加载到共享内存中
    可以隔几行加载一个元素，
    但不能隔固定位置加载一个元素，固定的偏差在大矩阵中向下移动的距离是不一样的
    这里隔的行数就是stridA和stridB
    */

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColumnA = threadIdx.x % BK;
    const uint stridA = small_C_dim.x * small_C_dim.y / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColumnB = threadIdx.x % BN;
    const uint stridB = small_C_dim.x * small_C_dim.y / BN;

    const uint outer_dot_nums = CEIL_DIV(K, BK);

    // float register_cache_A;
    // float register_cache_A[TM] = {0};
    // float register_cache_B[TN] = {0};

    for (uint outer_dot_index = 0; outer_dot_index < outer_dot_nums; outer_dot_index++)
    {
        for (uint iter = 0; iter < BM / stridA; iter++)
        {
            const uint _global_A_row_offset = innerRowA + iter * stridA;
            const uint _global_A_col_offset = innerColumnA;
            if (global_A_col + _global_A_col_offset < K && global_A_row + _global_A_row_offset < M)
            {

                As[(_global_A_row_offset)*BK + innerColumnA] =
                    A[(_global_A_row_offset)*K + innerColumnA];
            }
            else
            {
                As[(_global_A_row_offset)*BK + innerColumnA] = 0.0;
            }
        }

        for (uint iter = 0; iter < BK / stridB; iter++)
        {
            // Bs[(innerRowB + iter * stridB) * BN + innerColumnB] =
            //     B[(innerRowB + iter * stridB) * N + innerColumnB];
            const uint _global_B_row_offset = innerRowB + iter * stridB;
            const uint _global_B_col_offset = innerColumnB;
            if (global_B_col + _global_B_col_offset < N && global_B_row + _global_B_row_offset < K)
            {
                Bs[(_global_B_row_offset)*BN + innerColumnB] =
                    B[(_global_B_row_offset)*N + innerColumnB];
            }
            else
            {
                Bs[(_global_B_row_offset)*BN + innerColumnB] = 0.0;
            }
        }

        __syncthreads();

        A += BK;
        global_A_col += BK;
        B += BK * N;
        global_B_row += BK;

        for (uint inner_dot_index = 0; inner_dot_index < BK; inner_dot_index++)
        {
            // for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
            // {
            //     register_cache_B[resIdxN] = Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
            // }

            for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
            {
                // register_cache_A = As[(threadRow * TM + resIdxM) * BK + inner_dot_index];
                // register_cache_A[resIdxM] = As[(threadRow * TM + resIdxM) * BK + inner_dot_index];
                // if (resIdxM == 0)
                // {
                //     for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                //     {
                //         register_cache_B[resIdxN] = Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
                //     }
                // }

                for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                {
                    threadResults[resIdxM * TN + resIdxN] +=
                        As[(threadRow * TM + resIdxM) * BK + inner_dot_index] *
                        // register_cache_A *
                        // register_cache_A[resIdxM] *
                        Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
                    // register_cache_B[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
    {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
        {
            const uint _global_C_row_offset = threadRow * TM + resIdxM;
            const uint _global_C_col_offset = threadColumn * TN + resIdxN;
            // C[(threadRow * TM + resIdxM) * N + threadColumn * TN + resIdxN] =
            //     alpha * threadResults[resIdxM * TN + resIdxN] +
            //     beta * C[(threadRow * TM + resIdxM) * N + threadColumn * TN + resIdxN];
            if (global_C_row + _global_C_row_offset < M && global_C_col + _global_C_col_offset < N)
            {
                C[_global_C_row_offset * N + _global_C_col_offset] =
                    alpha * threadResults[resIdxM * TN + resIdxN] +
                    beta * C[_global_C_row_offset * N + _global_C_col_offset];
            }
        }
    }
}