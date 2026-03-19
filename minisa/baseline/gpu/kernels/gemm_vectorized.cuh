#pragma once
#include <cuda_runtime.h>

// Block dimensions for vectorized kernel
#define VEC_BM 128
#define VEC_BN 128
#define VEC_BK 16
#define VEC_TM 8
#define VEC_TN 8
#define VEC_NUM_THREADS ((VEC_BM / VEC_TM) * (VEC_BN / VEC_TN))

// Vectorized GEMM kernel with float4 loads for better memory coalescing
// Uses vectorized memory accesses and double buffering concept
__global__ void gemm_vectorized(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (VEC_BN / VEC_TN);
    const int threadRow = threadIdx.x / (VEC_BN / VEC_TN);

    __shared__ float As[VEC_BM * VEC_BK];
    __shared__ float Bs[VEC_BK * VEC_BN];

    float threadResults[VEC_TM * VEC_TN] = {0.0f};
    float regA[VEC_TM];
    float regB[VEC_TN];

    const int innerRowA = threadIdx.x / (VEC_BK / 4);
    const int innerColA = threadIdx.x % (VEC_BK / 4);
    const int innerRowB = threadIdx.x / (VEC_BN / 4);
    const int innerColB = threadIdx.x % (VEC_BN / 4);

    const int strideA = VEC_NUM_THREADS / (VEC_BK / 4);
    const int strideB = VEC_NUM_THREADS / (VEC_BN / 4);

    const float* baseA = A + cRow * VEC_BM * K;
    const float* baseB = B + cCol * VEC_BN;

    for (int bkIdx = 0; bkIdx < K; bkIdx += VEC_BK) {
        // Load A tile - vectorized when possible
        for (int loadOffset = 0; loadOffset < VEC_BM; loadOffset += strideA) {
            int aRow = cRow * VEC_BM + innerRowA + loadOffset;
            int aCol = bkIdx + innerColA * 4;

            if (aRow < M && aCol + 3 < K) {
                // Vectorized load
                float4 tmp = reinterpret_cast<const float4*>(&baseA[(innerRowA + loadOffset) * K + innerColA * 4])[0];
                As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + 0] = tmp.x;
                As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + 1] = tmp.y;
                As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + 2] = tmp.z;
                As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + 3] = tmp.w;
            } else {
                // Scalar fallback with bounds checking
                for (int i = 0; i < 4; i++) {
                    int col = aCol + i;
                    if (aRow < M && col < K)
                        As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + i] =
                            baseA[(innerRowA + loadOffset) * K + innerColA * 4 + i];
                    else
                        As[(innerRowA + loadOffset) * VEC_BK + innerColA * 4 + i] = 0.0f;
                }
            }
        }

        // Load B tile - vectorized when possible
        for (int loadOffset = 0; loadOffset < VEC_BK; loadOffset += strideB) {
            int bRow = bkIdx + innerRowB + loadOffset;
            int bCol = cCol * VEC_BN + innerColB * 4;

            if (bRow < K && bCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&baseB[(innerRowB + loadOffset) * N + innerColB * 4])[0];
                Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + 0] = tmp.x;
                Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + 1] = tmp.y;
                Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + 2] = tmp.z;
                Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + 3] = tmp.w;
            } else {
                for (int i = 0; i < 4; i++) {
                    int col = bCol + i;
                    if (bRow < K && col < N)
                        Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + i] =
                            baseB[(innerRowB + loadOffset) * N + innerColB * 4 + i];
                    else
                        Bs[(innerRowB + loadOffset) * VEC_BN + innerColB * 4 + i] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute thread tile
        #pragma unroll
        for (int dotIdx = 0; dotIdx < VEC_BK; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < VEC_TM; i++) {
                regA[i] = As[(threadRow * VEC_TM + i) * VEC_BK + dotIdx];
            }
            #pragma unroll
            for (int i = 0; i < VEC_TN; i++) {
                regB[i] = Bs[dotIdx * VEC_BN + threadCol * VEC_TN + i];
            }
            #pragma unroll
            for (int resIdxM = 0; resIdxM < VEC_TM; resIdxM++) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < VEC_TN; resIdxN++) {
                    threadResults[resIdxM * VEC_TN + resIdxN] += regA[resIdxM] * regB[resIdxN];
                }
            }
        }

        __syncthreads();

        baseA += VEC_BK;
        baseB += VEC_BK * N;
    }

    // Write results
    float* baseC = C + cRow * VEC_BM * N + cCol * VEC_BN;
    #pragma unroll
    for (int resIdxM = 0; resIdxM < VEC_TM; resIdxM++) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN < VEC_TN; resIdxN++) {
            int globalRow = cRow * VEC_BM + threadRow * VEC_TM + resIdxM;
            int globalCol = cCol * VEC_BN + threadCol * VEC_TN + resIdxN;
            if (globalRow < M && globalCol < N) {
                baseC[(threadRow * VEC_TM + resIdxM) * N + threadCol * VEC_TN + resIdxN] =
                    threadResults[resIdxM * VEC_TN + resIdxN];
            }
        }
    }
}

inline void launch_gemm_vectorized(const float* A, const float* B, float* C,
                                    int M, int K, int N, cudaStream_t stream = 0) {
    dim3 block(VEC_NUM_THREADS);
    dim3 grid((N + VEC_BN - 1) / VEC_BN, (M + VEC_BM - 1) / VEC_BM);
    gemm_vectorized<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}
