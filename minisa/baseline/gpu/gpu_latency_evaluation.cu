#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "kernels/gemm_naive.cuh"
#include "kernels/gemm_tiled.cuh"
#include "kernels/gemm_optimized.cuh"
#include "kernels/gemm_vectorized.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Profiling configuration
constexpr int NUM_TRIALS = 5;           // Number of independent trials
constexpr int WARMUP_RUNS = 10;         // Warmup iterations per trial
constexpr int TIMED_RUNS_PER_TRIAL = 50; // Timed runs per trial

// Struct to hold GEMM shape information
struct GemmShape {
    std::string category;
    std::string subcategory;
    int M, K, N;
};

// Extended benchmark result with statistics
struct BenchmarkStats {
    float mean_ms;      // Mean latency
    float min_ms;       // Minimum latency
    float max_ms;       // Maximum latency
    float std_ms;       // Standard deviation
    double mean_tflops; // Mean throughput
    std::vector<float> all_latencies; // All trial latencies
};

// Compute statistics from a vector of latencies
BenchmarkStats compute_stats(const std::vector<float>& latencies, double flops) {
    BenchmarkStats stats;
    stats.all_latencies = latencies;

    int n = latencies.size();

    // Mean
    float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    stats.mean_ms = sum / n;

    // Min and Max
    stats.min_ms = *std::min_element(latencies.begin(), latencies.end());
    stats.max_ms = *std::max_element(latencies.begin(), latencies.end());

    // Standard deviation
    float sq_sum = 0.0f;
    for (float lat : latencies) {
        sq_sum += (lat - stats.mean_ms) * (lat - stats.mean_ms);
    }
    stats.std_ms = std::sqrt(sq_sum / n);

    // TFLOPS based on mean latency
    stats.mean_tflops = flops / (stats.mean_ms * 1e9);

    return stats;
}

// Single trial benchmark - returns average latency for this trial
template<typename LaunchFunc>
float benchmark_single_trial(LaunchFunc launch_fn,
                             const float* d_A, const float* d_B, float* d_C,
                             int M, int K, int N) {
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        launch_fn(d_A, d_B, d_C, M, K, N, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_RUNS_PER_TRIAL; i++) {
        launch_fn(d_A, d_B, d_C, M, K, N, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / TIMED_RUNS_PER_TRIAL;
}

// Multi-trial benchmark with statistics
template<typename LaunchFunc>
BenchmarkStats benchmark_kernel_multi_trial(LaunchFunc launch_fn,
                                             const float* d_A, const float* d_B, float* d_C,
                                             int M, int K, int N) {
    std::vector<float> trial_latencies;
    trial_latencies.reserve(NUM_TRIALS);

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        float latency = benchmark_single_trial(launch_fn, d_A, d_B, d_C, M, K, N);
        trial_latencies.push_back(latency);
    }

    double flops = 2.0 * M * N * K;
    return compute_stats(trial_latencies, flops);
}

// Verify kernel correctness against naive implementation
bool verify_result(const float* d_C_ref, const float* d_C_test, int M, int N, float tolerance = 1e-3f) {
    size_t size = (size_t)M * N;
    std::vector<float> h_ref(size), h_test(size);

    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_C_ref, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test.data(), d_C_test, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(h_ref[i] - h_test[i]);
        float maxVal = std::max(std::abs(h_ref[i]), std::abs(h_test[i]));
        if (diff > tolerance * maxVal && diff > tolerance) {
            return false;
        }
    }
    return true;
}

void print_gpu_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "=======================" << std::endl << std::endl;
}

void print_profiling_config() {
    std::cout << "=== Profiling Configuration ===" << std::endl;
    std::cout << "Number of trials: " << NUM_TRIALS << std::endl;
    std::cout << "Warmup runs per trial: " << WARMUP_RUNS << std::endl;
    std::cout << "Timed runs per trial: " << TIMED_RUNS_PER_TRIAL << std::endl;
    std::cout << "Total kernel launches per shape: "
              << NUM_TRIALS * (WARMUP_RUNS + TIMED_RUNS_PER_TRIAL) * 4 << " (4 kernels)" << std::endl;
    std::cout << "================================" << std::endl << std::endl;
}

void print_stats(const std::string& name, const BenchmarkStats& stats, bool ok = true) {
    std::cout << "  " << std::left << std::setw(12) << name << ": "
              << std::fixed << std::setprecision(4) << stats.mean_ms << " ms"
              << " (min=" << stats.min_ms << ", max=" << stats.max_ms
              << ", std=" << std::setprecision(4) << stats.std_ms << ")"
              << " -> " << std::setprecision(2) << stats.mean_tflops << " TFLOPS";
    if (!ok) std::cout << " [FAILED]";
    std::cout << "\n";
}

int main() {
    print_gpu_info();
    print_profiling_config();

    // Define shapes from MINISA_Evaluation_Setup_Full.csv
    std::vector<GemmShape> shapes = {
        // FHE Bootstrapping BConv
        {"FHE", "Bootstrapping BConv", 65536, 40, 88},
        {"FHE", "Bootstrapping BConv", 65536, 40, 92},
        {"FHE", "Bootstrapping BConv", 65536, 40, 84},
        {"FHE", "Bootstrapping BConv", 65536, 40, 120},
        {"FHE", "Bootstrapping BConv", 65536, 40, 116},
        {"FHE", "Bootstrapping BConv", 65536, 44, 100},
        {"FHE", "Bootstrapping BConv", 65536, 44, 96},
        {"FHE", "Bootstrapping BConv", 65536, 44, 128},
        {"FHE", "Bootstrapping BConv", 65536, 44, 104},
        {"FHE", "Bootstrapping BConv", 65536, 44, 136},
        {"FHE", "Bootstrapping BConv", 65536, 44, 132},
        {"FHE", "Bootstrapping BConv", 65536, 48, 112},
        {"FHE", "Bootstrapping BConv", 65536, 48, 108},
        {"FHE", "Bootstrapping BConv", 65536, 48, 140},
        {"FHE", "Bootstrapping BConv", 65536, 48, 132},
        {"FHE", "Bootstrapping BConv", 65536, 48, 148},
        {"FHE", "Bootstrapping BConv", 65536, 48, 144},
        {"FHE", "Bootstrapping BConv", 65536, 52, 124},
        {"FHE", "Bootstrapping BConv", 65536, 52, 120},
        {"FHE", "Bootstrapping BConv", 65536, 52, 152},
        {"FHE", "Bootstrapping BConv", 65536, 52, 128},
        {"FHE", "Bootstrapping BConv", 65536, 52, 160},
        {"FHE", "Bootstrapping BConv", 65536, 52, 156},
        {"FHE", "Bootstrapping BConv", 65536, 56, 136},
        {"FHE", "Bootstrapping BConv", 65536, 56, 132},
        {"FHE", "Bootstrapping BConv", 65536, 60, 144},
        {"FHE", "Bootstrapping BConv", 65536, 56, 140},
        {"FHE", "Bootstrapping BConv", 65536, 60, 152},
        {"FHE", "Bootstrapping BConv", 65536, 60, 148},
        {"FHE", "Bootstrapping BConv", 65536, 28, 84},
        {"FHE", "Bootstrapping BConv", 65536, 28, 80},
        {"FHE", "Bootstrapping BConv", 65536, 32, 92},
        {"FHE", "Bootstrapping BConv", 65536, 28, 88},
        {"FHE", "Bootstrapping BConv", 65536, 32, 100},
        {"FHE", "Bootstrapping BConv", 65536, 32, 96},
        {"FHE", "Bootstrapping BConv", 65536, 36, 76},
        {"FHE", "Bootstrapping BConv", 65536, 36, 72},
        {"FHE", "Bootstrapping BConv", 65536, 36, 104},
        {"FHE", "Bootstrapping BConv", 65536, 36, 80},
        {"FHE", "Bootstrapping BConv", 65536, 36, 112},
        {"FHE", "Bootstrapping BConv", 65536, 36, 108},
        // NTT
        {"FHE", "NTT", 64, 1024, 1024},
        {"FHE", "NTT", 64, 2048, 2048},
        {"FHE", "NTT", 128, 2048, 2048},
        {"FHE", "NTT", 128, 4096, 4096},
        {"FHE", "NTT", 256, 4096, 4096},
        // ChatGPT OSS
        {"ChatGPT OSS", "Q", 256, 2880, 4096},
        {"ChatGPT OSS", "fused QKV", 256, 2880, 5120},
        {"ChatGPT OSS", "attn out", 256, 4096, 2880},
        {"ChatGPT OSS", "scores per group", 256, 64, 2048},
    };

    // Open output CSV file with extended statistics
    std::ofstream csv("gemm_profiling_results.csv");
    csv << "Category,Subcategory,M,K,N,FLOPs,"
        << "Naive_Mean_ms,Naive_Min_ms,Naive_Max_ms,Naive_Std_ms,Naive_TFLOPS,"
        << "Tiled_Mean_ms,Tiled_Min_ms,Tiled_Max_ms,Tiled_Std_ms,Tiled_TFLOPS,"
        << "Optimized_Mean_ms,Optimized_Min_ms,Optimized_Max_ms,Optimized_Std_ms,Optimized_TFLOPS,"
        << "Vectorized_Mean_ms,Vectorized_Min_ms,Vectorized_Max_ms,Vectorized_Std_ms,Vectorized_TFLOPS,"
        << "Best_Kernel,Best_Mean_ms,Best_TFLOPS\n";

    std::cout << "Starting GEMM Profiling (" << shapes.size() << " shapes)\n";
    std::cout << std::string(90, '=') << "\n\n";

    for (size_t idx = 0; idx < shapes.size(); idx++) {
        const auto& shape = shapes[idx];
        int M = shape.M, K = shape.K, N = shape.N;

        std::cout << "[" << (idx + 1) << "/" << shapes.size() << "] "
                  << shape.category << " - " << shape.subcategory
                  << " (M=" << M << ", K=" << K << ", N=" << N << ")\n";

        // Allocate device memory
        size_t size_A = (size_t)M * K * sizeof(float);
        size_t size_B = (size_t)K * N * sizeof(float);
        size_t size_C = (size_t)M * N * sizeof(float);

        float *d_A, *d_B, *d_C, *d_C_ref;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        CUDA_CHECK(cudaMalloc(&d_C, size_C));
        CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));

        // Initialize with random values on host then copy
        std::vector<float> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

        double flops = 2.0 * M * N * K;

        // Benchmark each kernel with multiple trials
        std::cout << "  Running " << NUM_TRIALS << " trials x " << TIMED_RUNS_PER_TRIAL << " iterations each...\n";

        auto naive_stats = benchmark_kernel_multi_trial(launch_gemm_naive, d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaMemcpy(d_C_ref, d_C, size_C, cudaMemcpyDeviceToDevice));

        auto tiled_stats = benchmark_kernel_multi_trial(launch_gemm_tiled, d_A, d_B, d_C, M, K, N);
        bool tiled_ok = verify_result(d_C_ref, d_C, M, N);

        auto optimized_stats = benchmark_kernel_multi_trial(launch_gemm_optimized, d_A, d_B, d_C, M, K, N);
        bool optimized_ok = verify_result(d_C_ref, d_C, M, N);

        auto vectorized_stats = benchmark_kernel_multi_trial(launch_gemm_vectorized, d_A, d_B, d_C, M, K, N);
        bool vectorized_ok = verify_result(d_C_ref, d_C, M, N);

        // Find best kernel based on mean latency
        std::string best_kernel = "Naive";
        float best_ms = naive_stats.mean_ms;
        double best_tflops = naive_stats.mean_tflops;

        if (tiled_ok && tiled_stats.mean_ms < best_ms) {
            best_kernel = "Tiled";
            best_ms = tiled_stats.mean_ms;
            best_tflops = tiled_stats.mean_tflops;
        }
        if (optimized_ok && optimized_stats.mean_ms < best_ms) {
            best_kernel = "Optimized";
            best_ms = optimized_stats.mean_ms;
            best_tflops = optimized_stats.mean_tflops;
        }
        if (vectorized_ok && vectorized_stats.mean_ms < best_ms) {
            best_kernel = "Vectorized";
            best_ms = vectorized_stats.mean_ms;
            best_tflops = vectorized_stats.mean_tflops;
        }

        // Print results with statistics
        print_stats("Naive", naive_stats);
        print_stats("Tiled", tiled_stats, tiled_ok);
        print_stats("Optimized", optimized_stats, optimized_ok);
        print_stats("Vectorized", vectorized_stats, vectorized_ok);
        std::cout << "  >> Best: " << best_kernel << " @ "
                  << std::setprecision(2) << best_tflops << " TFLOPS (mean)\n\n";

        // Write to CSV with full statistics
        csv << shape.category << "," << shape.subcategory << ","
            << M << "," << K << "," << N << ","
            << std::fixed << std::setprecision(0) << flops << ","
            // Naive stats
            << std::setprecision(4) << naive_stats.mean_ms << ","
            << naive_stats.min_ms << "," << naive_stats.max_ms << ","
            << naive_stats.std_ms << "," << naive_stats.mean_tflops << ","
            // Tiled stats
            << tiled_stats.mean_ms << "," << tiled_stats.min_ms << ","
            << tiled_stats.max_ms << "," << tiled_stats.std_ms << ","
            << tiled_stats.mean_tflops << ","
            // Optimized stats
            << optimized_stats.mean_ms << "," << optimized_stats.min_ms << ","
            << optimized_stats.max_ms << "," << optimized_stats.std_ms << ","
            << optimized_stats.mean_tflops << ","
            // Vectorized stats
            << vectorized_stats.mean_ms << "," << vectorized_stats.min_ms << ","
            << vectorized_stats.max_ms << "," << vectorized_stats.std_ms << ","
            << vectorized_stats.mean_tflops << ","
            // Best kernel
            << best_kernel << "," << best_ms << "," << best_tflops << "\n";

        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_C_ref));
    }

    csv.close();
    std::cout << std::string(90, '=') << "\n";
    std::cout << "Profiling complete!\n";
    std::cout << "Configuration: " << NUM_TRIALS << " trials x " << TIMED_RUNS_PER_TRIAL
              << " runs = " << NUM_TRIALS * TIMED_RUNS_PER_TRIAL << " measurements per kernel\n";
    std::cout << "Results saved to: gemm_profiling_results.csv\n";

    return 0;
}
