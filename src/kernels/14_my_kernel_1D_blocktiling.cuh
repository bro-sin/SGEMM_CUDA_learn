#pragma once

#include <cuda_runtime.h>

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void my_sgemm1DBlocktiling(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
    // 搞清楚这个线程在哪里
    const uint threadRow = threadIdx.x / BN;
    const uint threadColumn = threadIdx.x % BN;

    // 搞清楚这个block在哪里
    const uint currentRow = blockIdx.y;
    const uint currentColumn = blockIdx.x;
    // 这个block要计算C(M*N)的一个小块(BM*BN)的这个矩阵的所有元素
    /*
    这个矩阵是需要K/BK个(BM*BN)的小矩阵相加得到
    也就是分块矩阵A的第currentRow行和分块矩阵B的第currentColumn列相乘得到
    */
    const uint dotNums = K / BK;
    /*
    需要循环dotNums次，每次只计算一个(BM*BN)的小矩阵
    */

    // 搞清楚这个线程要计算的是什么
    /*
    这个线程要计算这一个(BM*BN)中的TM个元素
    这TM个元素是这个小矩阵的一列，
    那么需要分块小矩阵小A中的TM行(TM*BK)和分块小矩阵小B中的一列(BK*1)相乘
    也就是将BM再次拆分成了很多个TM
    */
    float threadResults[TM] = {0.0}; // 一个线程只计算一个小矩阵(BM*BN)中的TM个元素

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 将A移动到当前block要计算的分块矩阵的起始行
    A += currentRow * BM * K; // 向下移动currentRow*BM行
    // 将B移动到当前block要计算的分块矩阵的起始列
    B += currentColumn * BN; // 向右移动currentColumn*BN列
    // 将C移动到当前block要计算的分块矩阵的起始位置
    C += currentRow * BM * N + currentColumn * BN;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    // 给每个线程分配加载小A的哪个元素，小B的哪个元素
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColumnA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColumnB = threadIdx.x % BN;

    for (size_t dotindex = 0; dotindex < dotNums; dotindex++)
    {
        // 加载这一个循环用到的小A(BM*BK)和小B(BK*BN)到共享内存
        // 要依靠共享内存完成加载，需要保证线程数量等于BM*BK和BN*BK
        // 不然是加载不完这么多数据的
        As[innerRowA * BK + innerColumnA] = A[innerRowA * K + innerColumnA];
        Bs[innerRowB * BN + innerColumnB] = B[innerRowB * N + innerColumnB];
        __syncthreads(); // 等待所有线程加载完毕

        A += BK;     // 向右移动BK列
        B += BK * N; // 向下移动BK*N行
                     // 将小A和小B移动到下一次要计算的开始位置

        // 开始计算这个小矩阵的TM个元素
        /*
        要计算的实际上是小C矩阵(BM*BN)的第threadRow*TM到第threadRow*TM+TM-1行的第threadColumn列这TM个元素
        */
        for (size_t inner_dot_index = 0; inner_dot_index < BK; inner_dot_index++)
        {
            // 算一个元素的dotproduct,但一次要算TM个元素
            // 就直接把TM个元素的乘积都计算了，并且累加，反正不会互相影响
            float tmpB = Bs[inner_dot_index * BN + threadColumn];
            for (size_t res_index = 0; res_index < TM; res_index++)
            {
                threadResults[res_index] +=
                    As[(threadRow * TM + res_index) * BK + inner_dot_index] *
                    tmpB;
            }
        }
        __syncthreads(); // 等待这一轮dotindex的所有元素计算完成，然后再继续加剩下的元素
    }

    // 加完了之后，threadResults中存放的就是这个小矩阵的TM个元素
    //  将这TM个元素写入到C中
    for (size_t res_index = 0; res_index < TM; res_index++)
    {
        C[(threadRow * TM + res_index) * N + threadColumn] =
            alpha * threadResults[res_index] +
            beta * C[(threadRow * TM + res_index) * N + threadColumn];
    }
}