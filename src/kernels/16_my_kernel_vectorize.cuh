#pragma once

#define VECTOR_SIZE 4
// 矩阵乘法，默认A,B,C是行优先存储，
template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void mysgemmVectorize(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
    // 处理边界条件，考虑使用向量化加速

    // i\in [0,I), M=I\times BM, 是分块矩阵A和C的行数
    // j\in [0,J), N=J\times BN, 是分块矩阵B和C的列数
    const uint C_row_index_i = blockIdx.y; // i
    const uint C_col_index_j = blockIdx.x; // j
    // K=U\times BK,是分块后小矩阵A_{m_i}再次分块的列数，也是分块后小矩阵B_{n_j}再次分块的行数
    // 分块后小矩阵C_{m_in_j}需要U个A_{m_ik_u}*B_{k_un_j}相加
    const uint U = CEIL_DIV(K, BK);

    // BM=T\times TM, BN=S\times TN
    // T是As分块后的行数，S是Bs分块后的列数
    // T和S也是小矩阵Cs分块后的行数和列数
    const dim3 Cs_dim = {CEIL_DIV(BN, TN), CEIL_DIV(BM, TM)}; // S列,T行
    // t是当前线程负责的Cs分块后的小矩阵的行下标
    const uint Cs_row_t = threadIdx.x / Cs_dim.x; //(Idx)/S\in [0,T)
    // s是当前线程负责的Cs分块后的小矩阵的列下标
    const uint Cs_col_s = threadIdx.x % Cs_dim.x; //(Idx)%S\in [0,S)

    // 当前线程负责计算的TM*TN个元素
    float Cs_m_t_n_s[TM * TN] = {0.0};

    // 当前block计算时需要的分块矩阵数据
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动指针到当前block需要计算的C_{m_in_j}的起始位置
    A += C_row_index_i * BM * K;                      // 向下移动i行，每一个小的分块矩阵的行数是BM，大矩阵每一行有K个元素
    B += C_col_index_j * BN;                          // 向右移动j列，每一个小的分块矩阵的列数是BN
    C += C_row_index_i * BM * N + C_col_index_j * BN; // 向右移动j列，每一个小的分块矩阵的列数是BN，向下移动i行，每一个小的分块矩阵的行数是BM,大矩阵每一行有N个元素

    // 下面是给每个线程分配加载数据的任务
    const uint As_row = threadIdx.x / (BK / VECTOR_SIZE); // 最大为VECTOR_SIZE*T*S/(B*K)-1
    const uint As_col = threadIdx.x % (BK / VECTOR_SIZE); // 最大为B*K/VECTOR_SIZE-1
    // 需要重复加载的次数
    const uint strideA = BM * BK / VECTOR_SIZE / (Cs_dim.x * Cs_dim.y); // 数据总数（按照四个元素挨着的那种计算）处以线程总数

    const uint Bs_row = threadIdx.x / (BN / VECTOR_SIZE);
    const uint Bs_col = threadIdx.x % (BN / VECTOR_SIZE);
    const uint strideB = BK * BN / VECTOR_SIZE / (Cs_dim.x * Cs_dim.y);

    // 外层循环，循环U次，每次计算一个小矩阵C_{m_in_j}（对于整个block来说，对于这个线程来说是计算这个小矩阵中的一个小分块）
    for (uint u = 0; u < U; u++)
    {
        // 将这个block计算需要的As和Bs加载到共享内存中
        for (uint loadOffset = 0; loadOffset < strideA; loadOffset++)
        {
            uint aIndex = (As_row + loadOffset) * K + As_col * VECTOR_SIZE;
            // 获取As的第As_row行，第As_col列到第(As_col+3)列的数据
            const float4 tmp = reinterpret_cast<const float4 *>(&A[aIndex])[0];
            // 将数据转置后加载到共享内存中
            As[(As_col * VECTOR_SIZE + 0) * BM + As_row + loadOffset] = tmp.x;
            As[(As_col * VECTOR_SIZE + 1) * BM + As_row + loadOffset] = tmp.y;
            As[(As_col * VECTOR_SIZE + 2) * BM + As_row + loadOffset] = tmp.z;
            As[(As_col * VECTOR_SIZE + 3) * BM + As_row + loadOffset] = tmp.w;
        }
        for (uint loadOffset = 0; loadOffset < strideB; loadOffset++)
        {
            uint bIndex = (Bs_row + loadOffset) * N + Bs_col * VECTOR_SIZE;
            // 获取Bs的第Bs_row行，第Bs_col列到第(Bs_col+3)列的数据
            reinterpret_cast<float4 *>(&Bs[(Bs_row + loadOffset) * BN + Bs_col * VECTOR_SIZE])[0] =
                reinterpret_cast<const float4 *>(&B[bIndex])[0];
        }
        __syncthreads();
        A += BK;     // A向右移动BK
        B += BK * N; // B向下移动BK

        // 来到当前线程要计算的Cs_{m_tn_s}=As_{m_t}Bs_{n_s}
        // As_{m_t}：TM*BK, Bs_{n_s}:BK*TN, Cs_{m_tn_s}:TM*TN
        // for (uint _As_m_t_row = 0; _As_m_t_row < TM; _As_m_t_row++)
        // {
        //     for (uint _Bs_n_s = 0; _Bs_n_s < TN; _Bs_n_s++)
        //     {
        //         for (uint _dot_index = 0; _dot_index < BK; _dot_index++)
        //         {
        //             Cs_m_t_n_s[_As_m_t_row * TN + _Bs_n_s] +=
        //                 As[_dot_index * BM + Cs_row_t * TM + _As_m_t_row] * // As是转置的
        //                 Bs[_dot_index * BN + _Bs_n_s + Cs_col_s * TN];
        //         }
        //     }
        // }

        // 根据上面的逻辑进行改写优化
        for (uint _dot_index = 0; _dot_index < BK; _dot_index++)
        {
            for (uint _As_m_t_row = 0; _As_m_t_row < TM; _As_m_t_row++)
            {
                for (uint _Bs_n_s = 0; _Bs_n_s < TN; _Bs_n_s++)
                {
                    Cs_m_t_n_s[_As_m_t_row * TN + _Bs_n_s] +=
                        As[_dot_index * BM + Cs_row_t * TM + _As_m_t_row] * // As是转置的
                        Bs[_dot_index * BN + _Bs_n_s + Cs_col_s * TN];
                }
            }
        }
        __syncthreads();
    }

    // 将计算结果写回去
    for (uint _res_row = 0; _res_row < TM; _res_row++)
    {
        for (uint _res_col = 0; _res_col < TN; _res_col += VECTOR_SIZE)
        {
            float4 tmp = reinterpret_cast<float4 *>(&C[(Cs_row_t * TM + _res_row) * N + Cs_col_s * TN + _res_col])[0];
            tmp.x = alpha * Cs_m_t_n_s[_res_row * TN + _res_col] + beta * tmp.x;
            tmp.y = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 1] + beta * tmp.y;
            tmp.z = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 2] + beta * tmp.z;
            tmp.w = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(&C[(Cs_row_t * TM + _res_row) * N + Cs_col_s * TN + _res_col])[0] = tmp;
        }
    }
}
