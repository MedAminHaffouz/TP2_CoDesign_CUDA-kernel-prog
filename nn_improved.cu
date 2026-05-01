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
// NAIVE kernel (reference - from Annex-1)
// Problem 1: tid % (2*stride) causes thread divergence
// Problem 2: strided access causes shared memory bank conflicts
// ---------------------------------------------------------------
__global__ void sumReductionNaive(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          // <-- divergence here
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// IMPROVEMENT 1 — Interleaved Addressing (no divergence)
// Fix: replace tid % (2*stride) with a direct index calculation
// Now threads 0..blockDim.x/2 are ALL active at iteration 1
// threads 0..blockDim.x/4 ALL active at iteration 2, etc.
// No divergence within a warp anymore
// ---------------------------------------------------------------
__global__ void sumReductionImproved1(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // Map tid to the actual data index to process
        int index = 2 * stride * tid;        // <-- no modulo, no divergence
        if (index < blockDim.x)
            sharedData[index] += sharedData[index + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// IMPROVEMENT 2 — Sequential Addressing (no bank conflicts)
// Fix: reverse the stride direction
// Now adjacent threads access adjacent memory locations
// → no shared memory bank conflicts at all
// Also keeps low-indexed threads active (no divergence)
// ---------------------------------------------------------------
__global__ void sumReductionImproved2(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Start stride at half blockDim and go DOWN
    // thread 0 adds sharedData[0] + sharedData[128]
    // thread 1 adds sharedData[1] + sharedData[129]  <- contiguous, no bank conflict
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)                    // <-- only low threads active, contiguous
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// Sigmoid activation (shared by all versions)
// ---------------------------------------------------------------
__global__ void applySigmoid(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

// ---------------------------------------------------------------
// Run full NN pipeline with a given reduction kernel
// Returns kernel execution time in ms
// ---------------------------------------------------------------
float runNN(float* d_products, const char* label,
            void (*reductionKernel)(float*, float*, int)) {

    int blocksPerNeuron = N / BLOCK_SIZE;  // 32

    float *d_partial, *d_h, *d_partial2, *d_y;
    cudaMalloc(&d_partial,  N * blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_h,        N * sizeof(float));
    cudaMalloc(&d_partial2, blocksPerNeuron * sizeof(float));
    cudaMalloc(&d_y,        sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Pass 1: reduce each neuron row (8192 elements → 32 partial sums)
    reductionKernel<<<N * blocksPerNeuron, BLOCK_SIZE>>>(d_products, d_partial, N * N);

    // Pass 2: reduce 32 partial sums → 1 per neuron
    reductionKernel<<<N, blocksPerNeuron>>>(d_partial, d_h, N * blocksPerNeuron);

    // Apply sigmoid
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);

    // Final sum pass 1 and 2
    reductionKernel<<<blocksPerNeuron, BLOCK_SIZE>>>(d_h, d_partial2, N);
    reductionKernel<<<1, blocksPerNeuron>>>(d_partial2, d_y, blocksPerNeuron);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n[%s]\n", label);
    printf("  Output y        = %f\n", h_y);
    printf("  Kernel Time     = %f ms\n", kernel_time);

    cudaFree(d_partial);
    cudaFree(d_h);
    cudaFree(d_partial2);
    cudaFree(d_y);

    return kernel_time;
}

// ---------------------------------------------------------------
int main() {

    cout << "Step1: Generating input vector x\n";
    vector<float> h_x(N);
    generate(h_x.begin(), h_x.end(), []() { return (float)(rand() % 100) / 100.0f; });

    cout << "Step2: CPU element-wise multiply (weights=1)\n";
    vector<float> h_products(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_products[i * N + j] = h_x[j];

    cout << "Step3: Copy to device\n";
    float* d_products;
    cudaMalloc(&d_products, N * N * sizeof(float));
    cudaMemcpy(d_products, h_products.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "\n=== Running all 3 reduction versions ===\n";

    float t0 = runNN(d_products, "Naive",       sumReductionNaive);
    float t1 = runNN(d_products, "Improved 1 - No Divergence",  sumReductionImproved1);
    float t2 = runNN(d_products, "Improved 2 - No Bank Conflicts", sumReductionImproved2);

    printf("\n=== Summary ===\n");
    printf("  Naive       : %f ms  (speedup: 1.00x)\n", t0);
    printf("  Improved 1  : %f ms  (speedup: %.2fx)\n", t1, t0/t1);
    printf("  Improved 2  : %f ms  (speedup: %.2fx)\n", t2, t0/t2);

    cudaFree(d_products);
    return 0;
}
