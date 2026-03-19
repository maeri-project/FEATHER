#pragma once
#include <cuda_runtime.h>

// Naive GEMM kernel - each thread computes one element of C
// C = A * B where A is MxK, B is KxN, C is MxN
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch configuration for naive kernel
inline void launch_gemm_naive(const float* A, const float* B, float* C,
                              int M, int K, int N, cudaStream_t stream = 0) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}
