#pragma once
#include <cuda_runtime.h>

// Block tile dimensions
#define BM 128  // Block tile M
#define BN 128  // Block tile N
#define BK 8    // Block tile K

// Thread tile dimensions
#define TM 8    // Thread tile M
#define TN 8    // Thread tile N

// Number of threads per block
#define NUM_THREADS ((BM / TM) * (BN / TN))  // 256 threads

// Optimized GEMM kernel with 2D register tiling
// Each thread computes a TM x TN tile of output
// Uses shared memory for A and B tiles, registers for thread-local computation
__global__ void gemm_optimized(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int K, int N) {
    // Block position
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Thread position within the block's output tile
    const int threadCol = threadIdx.x % (BN / TN);  // 0-15
    const int threadRow = threadIdx.x / (BN / TN);  // 0-15

    // Shared memory for block tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Registers for thread tile computation
    float threadResults[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // Loading indices
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    // Strides for loading
    const int strideA = NUM_THREADS / BK;  // 32
    const int strideB = NUM_THREADS / BN;  // 2

    // Pointers to start of block tiles in global memory
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Main loop over K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A tile into shared memory
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int aRow = cRow * BM + innerRowA + loadOffset;
            int aCol = bkIdx + innerColA;
            if (aRow < M && aCol < K)
                As[(innerRowA + loadOffset) * BK + innerColA] =
                    A[(innerRowA + loadOffset) * K + innerColA];
            else
                As[(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
        }

        // Load B tile into shared memory
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            int bRow = bkIdx + innerRowB + loadOffset;
            int bCol = cCol * BN + innerColB;
            if (bRow < K && bCol < N)
                Bs[(innerRowB + loadOffset) * BN + innerColB] =
                    B[(innerRowB + loadOffset) * N + innerColB];
            else
                Bs[(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
        }

        __syncthreads();

        // Compute thread tile
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Load A values into registers
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            // Load B values into registers
            #pragma unroll
            for (int i = 0; i < TN; i++) {
                regB[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            // Outer product
            #pragma unroll
            for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                    threadResults[resIdxM * TN + resIdxN] += regA[resIdxM] * regB[resIdxN];
                }
            }
        }

        __syncthreads();

        // Advance pointers
        A += BK;
        B += BK * N;
    }

    // Write results to global memory
    #pragma unroll
    for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
            int globalRow = cRow * BM + threadRow * TM + resIdxM;
            int globalCol = cCol * BN + threadCol * TN + resIdxN;
            if (globalRow < M && globalCol < N) {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                    threadResults[resIdxM * TN + resIdxN];
            }
        }
    }
}

// Launch configuration for optimized kernel
inline void launch_gemm_optimized(const float* A, const float* B, float* C,
                                   int M, int K, int N, cudaStream_t stream = 0) {
    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_optimized<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}
