#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__device__ uint get_index_row_major(
    const uint row_index,
    const uint col_index,
    const uint row_nums,
    const uint col_nums)
{
    return row_index * col_nums + col_index;
}

__device__ uint get_index_col_major(
    const uint row_index,
    const uint col_index,
    const uint row_nums,
    const uint col_nums)
{
    return col_index * row_nums + row_index;
}

using IndexFunction = uint (*)(const uint, const uint, const uint, const uint);

template <const uint BM,
          const uint BN,
          const uint BK,
          const uint TM,
          const uint TN,
          const IndexFunction get_A_index,
          const IndexFunction get_B_index,
          const IndexFunction get_C_index,
          // 默认内部处理全用行优先
          const IndexFunction get_As_index = get_index_row_major,
          const IndexFunction get_Bs_index = get_index_row_major,
          const IndexFunction get_Cs_index = get_index_row_major>
__global__ void my_sgemm2DBlocktiling_with_index_function(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
    // 矩阵C[M,N]分块为Cs[BM,BN]
    const uint C_and_A_block_row_index_i = blockIdx.y; // 矩阵C分块的行索引
    const uint C_and_B_block_col_index_i = blockIdx.x;
    // i\in [0,I), I=CEIL_DIV(M,BM), 是分块矩阵C的行数和矩阵A分块后的行数
    // j\in [0,J), J=CEIL_DIV(N,BN), 是分块矩阵C的列数和矩阵B分块后的列数

    // 矩阵A分块的列数和矩阵B分块的行数
    const uint U = CEIL_DIV(K, BK);

    const uint2 CS_block_dim(CEIL_DIV(BN, TN), CEIL_DIV(BM, TM));

    const uint Cs_and_As_block_row_index_t = threadIdx.x / CS_block_dim.x;
    const uint Cs_and_Bs_block_col_index_s = threadIdx.x % CS_block_dim.x;

    const uint As_and_Cs_row_index_start = Cs_and_As_block_row_index_t * TM;
    const uint Bs_and_Cs_col_index_start = Cs_and_Bs_block_col_index_s * TN;

    float Cs_m_t_n_s[TM * TN] = {0.0f};

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint C_and_A_row_index_start = C_and_A_block_row_index_i * BM;
    // uint A_col_index_start = 0;

    // uint B_row_index_start = 0;
    const uint C_and_B_col_index_start = C_and_B_block_col_index_i * BN;

    const uint As_row_index_start_for_data_loading_only = threadIdx.x / BK;
    const uint As_col_index_for_data_loading_only = threadIdx.x % BK;
    const uint stride_A_row_for_data_loading_only = blockDim.x / BK; // 一个线程加载的行的间隔

    const uint Bs_row_index_start_for_data_loading_only = threadIdx.x / BN;
    const uint Bs_col_index_for_data_loading_only = threadIdx.x % BN;
    const uint stride_B_col_for_data_loading_only = blockDim.x / BN;

    const uint _B_col_index_for_data_loading_only = C_and_B_col_index_start + Bs_col_index_for_data_loading_only;

    for (uint A_col_index_start_and_B_row_index_start = 0; A_col_index_start_and_B_row_index_start < K; A_col_index_start_and_B_row_index_start += BK)
    {
        // 加载当前block需要使用的部分A的分块矩阵，并且转置后存到As中
        const uint _A_col_index_for_data_loading_only = A_col_index_start_and_B_row_index_start + As_col_index_for_data_loading_only;
        for (uint As_row_offset = 0; As_row_offset < BM; As_row_offset += stride_A_row_for_data_loading_only)
        {
            const uint _A_row_index_for_data_loading_only =
                C_and_A_row_index_start + As_row_index_start_for_data_loading_only + As_row_offset;
            const uint _As_index_for_data_loading_only =
                get_As_index(As_col_index_for_data_loading_only, // 这里因为是转置，所以行列索引要反过来
                             As_row_index_start_for_data_loading_only + As_row_offset,
                             BK, BM); // 期望输出col*BM+row
            if (_A_col_index_for_data_loading_only < K && _A_row_index_for_data_loading_only < M)
            {
                const uint _A_index =
                    get_A_index(_A_row_index_for_data_loading_only,
                                _A_col_index_for_data_loading_only,
                                M, K);
                As[_As_index_for_data_loading_only] = A[_A_index];
            }
            else
            {
                As[_As_index_for_data_loading_only] = 0.0f;
            }
        }

        // 加载当前block需要使用的部分B的分块矩阵，存到Bs中
        //_B_col_index_for_data_loading_only =在循环之外定义，因为这个量与循环变量无关
        for (uint Bs_row_offset = 0; Bs_row_offset < BK; Bs_row_offset += stride_B_col_for_data_loading_only)
        {
            const uint _B_row_index_for_data_loading_only =
                A_col_index_start_and_B_row_index_start + Bs_row_index_start_for_data_loading_only + Bs_row_offset;
            const uint _Bs_index_for_data_loading_only =
                get_Bs_index(Bs_row_index_start_for_data_loading_only + Bs_row_offset,
                             Bs_col_index_for_data_loading_only,
                             BK, BN);
            if (_B_col_index_for_data_loading_only < N && _B_row_index_for_data_loading_only < K)
            {
                const uint _B_index =
                    get_B_index(_B_row_index_for_data_loading_only,
                                _B_col_index_for_data_loading_only,
                                K, N);
                Bs[_Bs_index_for_data_loading_only] = B[_B_index];
            }
            else
            {
                Bs[_Bs_index_for_data_loading_only] = 0.0f;
            }
        }
        __syncthreads();

        for (uint As_col_and_Bs_row = 0; As_col_and_Bs_row < BK; As_col_and_Bs_row++)
        {
            for (uint As_and_Cs_row_offset = 0; As_and_Cs_row_offset < TM; As_and_Cs_row_offset++)
            {
                for (uint Bs_and_Cs_col_offset = 0; Bs_and_Cs_col_offset < TN; Bs_and_Cs_col_offset++)
                {
                    Cs_m_t_n_s[get_Cs_index(As_and_Cs_row_offset, Bs_and_Cs_col_offset, TM, TN)] +=
                        Bs[get_Bs_index(As_col_and_Bs_row, Bs_and_Cs_col_index_start + Bs_and_Cs_col_offset, BK, BN)] *
                        As[get_As_index(As_col_and_Bs_row, As_and_Cs_row_index_start + As_and_Cs_row_offset, BK, BM)]; // 这里因为是转置，所以行列索引要反过来
                }
            }
        }
        __syncthreads();
    }

    // 写回C
    for (uint Cs_row_offset = 0; Cs_row_offset < TM; Cs_row_offset++)
    {
        for (uint Cs_col_offset = 0; Cs_col_offset < TN; Cs_col_offset++)
        {

            const uint C_row = C_and_A_row_index_start + As_and_Cs_row_index_start + Cs_row_offset;
            const uint C_col = C_and_B_col_index_start + Bs_and_Cs_col_index_start + Cs_col_offset;
            if (C_row < M && C_col < N)
            {
                const uint C_index = get_C_index(C_row, C_col, M, N);
                const uint Cs_index = get_Cs_index(Cs_row_offset, Cs_col_offset, TM, TN);
                C[C_index] = alpha * Cs_m_t_n_s[Cs_index] + beta * C[C_index];
            }
        }
    }
}