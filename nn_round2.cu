#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

using std::cout;
using std::vector;
using std::generate;

#define N 8192
#define BLOCK_SIZE 256

// ---------------------------------------------------------------
// NAIVE — baseline reference
// ---------------------------------------------------------------
__global__ void sumReductionNaive(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// ROUND 1 — First Add During Load (kept as stepping stone)
// ---------------------------------------------------------------
__global__ void sumReductionRound1(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float a = (idx < n)               ? input[idx]              : 0.0f;
    float b = (idx + blockDim.x < n)  ? input[idx + blockDim.x] : 0.0f;
    sharedData[tid] = a + b;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// WARP SHUFFLE HELPER
//
// Reduces 32 values held across a warp's registers down to 1
// using __shfl_down_sync() — no shared memory involved at all.
//
// How __shfl_down_sync works:
//   Each thread reads the value of a thread that is `offset` lanes ahead
//   Thread 0 gets thread 0's val + thread 16's val (offset=16)
//   Thread 0 gets thread 0's val + thread 8's val  (offset=8)
//   ... and so on until thread 0 holds the sum of all 32 values
//
// 0xffffffff = all 32 threads participate (bitmask)
// This is pure register-to-register communication — fastest possible
// ---------------------------------------------------------------
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------
// ROUND 2 — First Add During Load + Warp Shuffle
//
// New idea on top of Round 1:
//   Instead of using shared memory for the LAST 5 iterations
//   (when only 1 warp remains active), use warp shuffle instead.
//
// Why it helps:
//   Shared memory has ~20-30 cycle latency even with no bank conflicts
//   Warp shuffle operates on registers directly → ~1-2 cycle latency
//   The last warp's 5 iterations become essentially free
//
// Structure:
//   1. Load 2 elements, add during load (Round 1 trick)
//   2. Shared memory reduction DOWN TO 32 threads (1 warp)
//   3. Switch to warp shuffle for final 5 iterations — no shared mem
// ---------------------------------------------------------------
__global__ void sumReductionRound2(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Round 1 trick: load 2 elements, add immediately
    float a = (idx < n)               ? input[idx]              : 0.0f;
    float b = (idx + blockDim.x < n)  ? input[idx + blockDim.x] : 0.0f;
    sharedData[tid] = a + b;
    __syncthreads();

    // Shared memory reduction — stop at warp boundary (stride > 32)
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // At this point sharedData[0..31] holds 32 partial sums
    // Hand off to warp shuffle — no shared memory from here
    if (tid < 32) {
        float val = sharedData[tid];  // each of 32 threads loads its partial sum
        val = warpReduce(val);        // warp shuffle reduces to thread 0's register
        if (tid == 0) output[blockIdx.x] = val;
    }
}

// ---------------------------------------------------------------
__global__ void applySigmoid(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// ---------------------------------------------------------------
float runNaive(float* d_products) {
    int bpn = N / BLOCK_SIZE;
    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * bpn * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, bpn * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReductionNaive<<<N * bpn, BLOCK_SIZE>>>(d_products, d_partial, N * N);
    sumReductionNaive<<<N, bpn>>>(d_partial, d_h, N * bpn);
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);
    sumReductionNaive<<<bpn, BLOCK_SIZE>>>(d_h, d_partial2, N);
    sumReductionNaive<<<1, bpn>>>(d_partial2, d_y, bpn);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[Naive]  y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_partial); cudaFree(d_h); cudaFree(d_partial2); cudaFree(d_y);
    return t;
}

// ---------------------------------------------------------------
float runRound1(float* d_products) {
    int bpn = N / (BLOCK_SIZE * 2);  // 16 blocks per neuron
    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * bpn * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, bpn * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReductionRound1<<<N * bpn, BLOCK_SIZE>>>(d_products, d_partial, N * N);
    sumReductionRound1<<<N, bpn>>>(d_partial, d_h, N * bpn);
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);
    sumReductionRound1<<<bpn, BLOCK_SIZE>>>(d_h, d_partial2, N);
    sumReductionRound1<<<1, bpn>>>(d_partial2, d_y, bpn);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[Round1] y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_partial); cudaFree(d_h); cudaFree(d_partial2); cudaFree(d_y);
    return t;
}

// ---------------------------------------------------------------
float runRound2(float* d_products) {
    int bpn = N / (BLOCK_SIZE * 2);  // same as Round1 — same load strategy
    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * bpn * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, bpn * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReductionRound2<<<N * bpn, BLOCK_SIZE>>>(d_products, d_partial, N * N);
    sumReductionRound2<<<N, bpn>>>(d_partial, d_h, N * bpn);
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);
    sumReductionRound2<<<bpn, BLOCK_SIZE>>>(d_h, d_partial2, N);
    sumReductionRound2<<<1, bpn>>>(d_partial2, d_y, bpn);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[Round2] y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_partial); cudaFree(d_h); cudaFree(d_partial2); cudaFree(d_y);
    return t;
}

// ---------------------------------------------------------------
int main() {
    cout << "Step1: Generate x\n";
    vector<float> h_x(N);
    generate(h_x.begin(), h_x.end(), []() { return (float)(rand() % 100) / 100.0f; });

    cout << "Step2: CPU element-wise multiply\n";
    vector<float> h_products(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_products[i * N + j] = h_x[j];

    cout << "Step3: Copy to device\n";
    float* d_products;
    cudaMalloc(&d_products, N * N * sizeof(float));
    cudaMemcpy(d_products, h_products.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "\n=== Results ===\n";
    float t0 = runNaive(d_products);
    float t1 = runRound1(d_products);
    float t2 = runRound2(d_products);

    printf("\n=== Summary Table ===\n");
    printf("  %-40s  %10s  %10s\n", "Kernel", "Time (ms)", "Speedup");
    printf("  %-40s  %10.3f  %10.2fx\n", "Naive",                          t0, 1.0f);
    printf("  %-40s  %10.3f  %10.2fx\n", "Round1: First Add During Load",  t1, t0/t1);
    printf("  %-40s  %10.3f  %10.2fx\n", "Round2: + Warp Shuffle",         t2, t0/t2);

    cudaFree(d_products);
    return 0;
}
