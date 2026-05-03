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
// ROUND 1 — First Add During Load
//
// Key idea: instead of loading 1 element per thread then reducing,
// each thread loads 2 elements and adds them immediately at load time.
//
// Why it helps:
//   - Halves the number of blocks needed (each block covers 2*BLOCK_SIZE elements)
//   - Halves total __syncthreads() calls
//   - First reduction step is free — happens during the memory load
//   - Better arithmetic intensity: 1 add per 2 global memory reads
//
// Each thread i handles elements at idx AND idx + blockDim.x
// So one block of 256 threads now reduces 512 elements instead of 256
// ---------------------------------------------------------------
__global__ void sumReductionRound1(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;

    // Each block covers 2 * BLOCK_SIZE elements
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load AND add 2 elements at once — first reduction step is free
    float a = (idx < n)              ? input[idx]              : 0.0f;
    float b = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sharedData[tid] = a + b;
    __syncthreads();

    // Sequential addressing (no divergence, no bank conflicts)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
__global__ void applySigmoid(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// ---------------------------------------------------------------
// Run full NN pipeline — naive version (2 passes per neuron)
// ---------------------------------------------------------------
float runNaive(float* d_products) {
    int blocksPerNeuron = N / BLOCK_SIZE;  // 32 blocks per neuron

    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReductionNaive<<<N * blocksPerNeuron, BLOCK_SIZE>>>(d_products, d_partial, N * N);
    sumReductionNaive<<<N, blocksPerNeuron>>>(d_partial, d_h, N * blocksPerNeuron);
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);
    sumReductionNaive<<<blocksPerNeuron, BLOCK_SIZE>>>(d_h, d_partial2, N);
    sumReductionNaive<<<1, blocksPerNeuron>>>(d_partial2, d_y, blocksPerNeuron);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    printf("[Naive]  y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_partial); cudaFree(d_h);
    cudaFree(d_partial2); cudaFree(d_y);
    return t;
}

// ---------------------------------------------------------------
// Run full NN pipeline — Round 1
// NOTE: each block now covers 2*BLOCK_SIZE elements
// so blocksPerNeuron is halved
// ---------------------------------------------------------------
float runRound1(float* d_products) {
    int blocksPerNeuron = N / (BLOCK_SIZE * 2);  // 16 blocks per neuron (halved!)

    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReductionRound1<<<N * blocksPerNeuron, BLOCK_SIZE>>>(d_products, d_partial, N * N);
    sumReductionRound1<<<N, blocksPerNeuron>>>(d_partial, d_h, N * blocksPerNeuron);
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);
    sumReductionRound1<<<blocksPerNeuron, BLOCK_SIZE>>>(d_h, d_partial2, N);
    sumReductionRound1<<<1, blocksPerNeuron>>>(d_partial2, d_y, blocksPerNeuron);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float t; cudaEventElapsedTime(&t, start, stop);
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    printf("[Round1] y = %f  |  Kernel Time = %f ms\n", h_y, t);

    cudaFree(d_partial); cudaFree(d_h);
    cudaFree(d_partial2); cudaFree(d_y);
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

    printf("\n=== Summary Table ===\n");
    printf("  %-35s  %10s  %10s\n", "Kernel", "Time (ms)", "Speedup");
    printf("  %-35s  %10.3f  %10.2fx\n", "Naive", t0, 1.0f);
    printf("  %-35s  %10.3f  %10.2fx\n", "Round1: First Add During Load", t1, t0/t1);

    cudaFree(d_products);
    return 0;
}
