#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const uint BLOCKSIZE>
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
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_column = threadIdx.x % BLOCKSIZE;

    // 计算这个线程要算C的哪个元素
    // 这里blockdim.x是BLOCKSIZE^2，y是1, z是1
    const uint row = blockIdx.y * BLOCKSIZE + thread_row;
    const uint column = blockIdx.x * BLOCKSIZE + thread_column;
    // 要计算的就是 C[row*N+column]
    // 这个block中要计算的C元素的范围是
    // 行：blockIdx.y*BLOCKSIZE到blockIdx.y*BLOCKSIZE+BLOCKSIZE-1
    // 列：blockIdx.x*BLOCKSIZE到blockIdx.x*BLOCKSIZE+BLOCKSIZE-1

    // 利用这一个block,反复计算A[row][k]*B[k][column]的和

    // 这一个block中所有线程的特点是：
    //  blockIdx完全相同，
    //  threadIdx.x在0-BLOCKSIZE^2之间
    // 可以通过共享内存存下的A的行向量的范围是
    // 行的范围从blockIdx.y*BLOCKSIZE到blockIdx.y*BLOCKSIZE+BLOCKSIZE-1
    // 列的范围从0到K-1，要将K分成多份，每份BLOCKSIZE个元素
    // 实际上每一个线程只负责了获取一个元素，但是一共有BLOCKSIZE^2个线程，
    // 所以获取完成元素之后，所有线程都可以共享这BLOCKSIZE^2个元素

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint numChunks = CEIL_DIV(K, BLOCKSIZE);
    // numChunks表示将K分成了多少份

    float tmp = 0; // 下面循环numChunks次，每次都会计算BLOCKSIZE个元素的和
    for (uint chunk = 0; chunk < numChunks; chunk++)
    {
        // chunk表示当前是第几轮
        As[thread_row * BLOCKSIZE + thread_column] = A[row * K + chunk * BLOCKSIZE + thread_column];
        Bs[thread_row * BLOCKSIZE + thread_column] = B[(chunk * BLOCKSIZE + thread_row) * N + column];
        // 这里同一个block中的线程thread_column和thread_row会在0-BLOCKSIZE之间遍历
        // 这保证了As取到了blockIdx.y*BLOCKSIZE到blockIdx.y*BLOCKSIZE+BLOCKSIZE-1行
        // 以及chunk*BLOCKSIZE到chunk*BLOCKSIZE+BLOCKSIZE-1列的元素
        // 同理Bs取到了chunk*BLOCKSIZE到chunk*BLOCKSIZE+BLOCKSIZE-1行
        // 以及blockIdx.x*BLOCKSIZE到blockIdx.x*BLOCKSIZE+BLOCKSIZE-1列的元素
        __syncthreads();
        // 保证这个block中的所有线程都取到了As和Bs的值

        for (uint k_index = 0; k_index < BLOCKSIZE; k_index++)
        {
            tmp += As[thread_row * BLOCKSIZE + k_index] * Bs[k_index * BLOCKSIZE + thread_column];
        }
        __syncthreads();
        // 保证这个block中的所有线程都计算完了这一轮的BLOCKSIZE个元素
        // 因为下一轮会重新给As和Bs赋值
    }
    // 计算完了，最终tmp就是这个线程累加的K个元素的和
    C[row * N + column] = alpha * tmp + beta * C[row * N + column];
}