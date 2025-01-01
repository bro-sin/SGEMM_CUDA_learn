#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

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
    __shared__ float myAs[BM * BK];
    __shared__ float myBs[BK * BN];
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

    const uint outer_dot_nums = K / BK;
    for (uint outer_dot_index = 0; outer_dot_index < outer_dot_nums; outer_dot_index++)
    {
        uint test_count_A = 0;
        uint test_count_myA = 0;

        // 设置四个数组，分别记录myAs和对应A的索引，As和对应A的索引
        const uint iter_nums = 40; // BM / stridA;
        uint myAs_index[iter_nums];
        uint myA_index[iter_nums];
        uint As_index[iter_nums];
        uint A_index[iter_nums];
        for (uint iter = 0; iter < BM / stridA; iter++)
        {
            myAs[(innerRowA + iter * stridA) * BK + innerColumnA] =
                A[(innerRowA + iter * stridA) * K + innerColumnA];
            myAs_index[test_count_myA] = ((innerRowA + iter * stridA) * BK + innerColumnA);
            myA_index[test_count_myA] = ((innerRowA + iter * stridA) * K + innerColumnA);
            test_count_myA++;
        }
        assert(BM / stridA == test_count_myA); // 上面循环中test_count_myA的值等于BM/stridA才停止

        // 记录myBs的索引
        uint myBsIndex[5];
        uint myBIndex[5];
        for (uint iter = 0; iter < BK / stridB; iter++)
        {
            myBs[(innerRowB + iter * stridB) * BN + innerColumnB] =
                B[(innerRowB + iter * stridB) * N + innerColumnB];
            myBsIndex[iter] = (innerRowB + iter * stridB) * BN + innerColumnB;
            myBIndex[iter] = (innerRowB + iter * stridB) * N + innerColumnB;
        }
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += stridA)
        {
            As[(innerRowA + loadOffset) * BK + innerColumnA] =
                A[(innerRowA + loadOffset) * K + innerColumnA];
            As_index[test_count_A] = ((innerRowA + loadOffset) * BK + innerColumnA);
            A_index[test_count_A] = ((innerRowA + loadOffset) * K + innerColumnA);
            test_count_A++;
        }
        assert(BM / stridA == test_count_A);

        uint BsIndex[5];
        uint BIndex[5];
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += stridB)
        {
            Bs[(innerRowB + loadOffset) * BN + innerColumnB] =
                B[(innerRowB + loadOffset) * N + innerColumnB];
            BsIndex[loadOffset / stridB] = (innerRowB + loadOffset) * BN + innerColumnB;
            BIndex[loadOffset / stridB] = (innerRowB + loadOffset) * N + innerColumnB;
        }

        __syncthreads();

        // 检查两个索引是否一致
        for (uint i = 0; i < 5; i++)
        {
            if (myBsIndex[i] != BsIndex[i])
            {
                printf("myBsIndex[%d] = %d, BsIndex[%d] = %d\n", i, myBsIndex[i], i, BsIndex[i]);
            }
            if (myBIndex[i] != BIndex[i])
            {
                printf("myBIndex[%d] = %d, BIndex[%d] = %d\n", i, myBIndex[i], i, BIndex[i]);
            }
        }

        for (uint i = 0; i < test_count_myA; i++)
        {
            if (myAs_index[i] != As_index[i])
            {
                printf("myAs_index[%d] = %d, As_index[%d] = %d\n", i, myAs_index[i], i, As_index[i]);
            }
            if (myA_index[i] != A_index[i])
            {
                printf("myA_index[%d] = %d, A_index[%d] = %d\n", i, myA_index[i], i, A_index[i]);
            }
        }

        // 检查As和Bs是否与myAs和myBs相等
        for (uint i = 0; i < BM * BK; i++)
        {
            if (As[i] != myAs[i])
            {
                printf("As[%d] = %f, myAs[%d] = %f,diff=%f\n", i, As[i], i, myAs[i], As[i] - myAs[i]);
            }
        }
        for (uint i = 0; i < BK * BN; i++)
        {
            if (Bs[i] != myBs[i])
            {
                printf("Bs[%d] = %f, myBs[%d] = %f,diff=%f\n", i, Bs[i], i, myBs[i], Bs[i] - myBs[i]);
            }
        }

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