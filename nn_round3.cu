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

__global__ void applySigmoid(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// ---------------------------------------------------------------
// WARP SHUFFLE HELPER (same as Round 2)
// ---------------------------------------------------------------
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------
// ROUND 2 kernel (kept as reference)
// ---------------------------------------------------------------
__global__ void sumReductionRound2(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float a = (idx < n)               ? input[idx]              : 0.0f;
    float b = (idx + blockDim.x < n)  ? input[idx + blockDim.x] : 0.0f;
    sharedData[tid] = a + b;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid < 32) {
        float val = sharedData[tid];
        val = warpReduce(val);
        if (tid == 0) output[blockIdx.x] = val;
    }
}

// ---------------------------------------------------------------
// ROUND 3 — Fused Kernel
//
// Problem with all previous versions:
//   The NN pipeline has 5 separate kernel launches:
//     Pass1 reduction  → writes d_partial  to global memory
//     Pass2 reduction  → reads  d_partial, writes d_h to global memory
//     Sigmoid          → reads  d_h,       writes d_h to global memory
//     Final sum pass1  → reads  d_h,       writes d_partial2
//     Final sum pass2  → reads  d_partial2, writes d_y
//   = 4 unnecessary global memory round-trips between kernels
//
// Solution: fuse into 2 kernels only
//
// FUSED KERNEL 1 — reduceAndSigmoid:
//   Each block is responsible for ONE neuron (one row of d_products)
//   It reduces all 8192 elements of that row → applies sigmoid → writes h[neuron]
//   No intermediate d_partial array needed at all
//   Neurons are processed in parallel across blocks
//
//   How: each thread strides across the full row loading+adding,
//   then a shared memory + warp shuffle reduction gives the block sum,
//   thread 0 applies sigmoid and writes directly to d_h
//
// FUSED KERNEL 2 — finalSumShuffle:
//   Reduces all 8192 h[i] values to scalar y
//   Same warp shuffle approach
//   One kernel launch, one global memory write
//
// Net result: 5 launches → 2 launches, 4 global memory round-trips eliminated
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Fused Kernel 1: reduce full row + sigmoid in one kernel
// One block per neuron, threads stride across the full N elements
// ---------------------------------------------------------------
__global__ void reduceAndSigmoid(float* input, float* h_out, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid  = threadIdx.x;
    int neuron = blockIdx.x;  // one block = one neuron

    // Grid-stride across the full row of this neuron
    // Each thread accumulates multiple elements before entering shared mem
    float sum = 0.0f;
    int rowStart = neuron * n;
    for (int i = tid; i < n; i += blockDim.x)
        sum += input[rowStart + i];

    sharedData[tid] = sum;
    __syncthreads();

    // Shared memory reduction down to 1 warp
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // Warp shuffle for final 32 → 1
    if (tid < 32) {
        float val = sharedData[tid];
        val = warpReduce(val);

        // Thread 0 applies sigmoid and writes directly — no intermediate array
        if (tid == 0)
            h_out[neuron] = 1.0f / (1.0f + expf(-val));
    }
}

// ---------------------------------------------------------------
// Fused Kernel 2: reduce all h[i] to scalar y using warp shuffle
// ---------------------------------------------------------------
__global__ void finalSumShuffle(float* h_in, float* y_out, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;

    // Grid-stride accumulation
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += h_in[i];

    sharedData[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid < 32) {
        float val = sharedData[tid];
        val = warpReduce(val);
        if (tid == 0) y_out[0] = val;
    }
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
float runRound2(float* d_products) {
    int bpn = N / (BLOCK_SIZE * 2);
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
float runRound3(float* d_products) {
    float *d_h, *d_y;
    cudaMalloc(&d_h, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Kernel 1: N blocks (one per neuron), each reduces full row + sigmoid
    reduceAndSigmoid<<<N, BLOCK_SIZE>>>(d_products, d_h, N);

    // Kernel 2: single block reduces all h[i] to scalar y
    finalSumShuffle<<<1, BLOCK_SIZE>>>(d_h, d_y, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[Round3] y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_h); cudaFree(d_y);
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
    float t2 = runRound2(d_products);
    float t3 = runRound3(d_products);

    printf("\n=== Summary Table ===\n");
    printf("  %-45s  %10s  %10s\n", "Kernel", "Time (ms)", "Speedup");
    printf("  %-45s  %10.3f  %10.2fx\n", "Naive",                              t0, 1.0f);
    printf("  %-45s  %10.3f  %10.2fx\n", "Round2: Load+Shuffle (5 launches)",  t2, t0/t2);
    printf("  %-45s  %10.3f  %10.2fx\n", "Round3: Fused Kernel (2 launches)",  t3, t0/t3);

    cudaFree(d_products);
    return 0;
}
