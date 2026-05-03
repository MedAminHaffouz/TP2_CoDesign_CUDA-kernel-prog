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
// Issues: thread divergence + shared memory bank conflicts
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
// BEST kernel — combines 2 techniques:
//
// Technique 1: Sequential Addressing
//   Stride goes from blockDim/2 DOWN to 32
//   Adjacent threads access adjacent memory → zero bank conflicts
//   Low-indexed threads always active → zero warp divergence
//
// Technique 2: Last Warp Unrolling
//   When stride <= 32, only 1 warp remains active
//   Warps execute in lockstep → __syncthreads() is unnecessary
//   Manual unroll of last 6 iterations removes sync overhead
// ---------------------------------------------------------------
__global__ void sumReductionBest(float* input, float* output, int n) {
    __shared__ volatile float sharedData[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Technique 1: sequential addressing, stop before last warp
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    // Technique 2: unroll last warp (no __syncthreads needed)
    if (tid < 32) {
        sharedData[tid] += sharedData[tid + 32];
        sharedData[tid] += sharedData[tid + 16];
        sharedData[tid] += sharedData[tid + 8];
        sharedData[tid] += sharedData[tid + 4];
        sharedData[tid] += sharedData[tid + 2];
        sharedData[tid] += sharedData[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
// Sigmoid activation
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

    // Pass 1: reduce each neuron's 8192 inputs → 32 partial sums
    reductionKernel<<<N * blocksPerNeuron, BLOCK_SIZE>>>(d_products, d_partial, N * N);

    // Pass 2: reduce 32 partial sums → 1 value per neuron → d_h[i]
    reductionKernel<<<N, blocksPerNeuron>>>(d_partial, d_h, N * blocksPerNeuron);

    // Apply sigmoid to all hidden neuron outputs
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);

    // Final sum: reduce all h[i] → scalar output y (2 passes)
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
    printf("  Output y     = %f\n", h_y);
    printf("  Kernel Time  = %f ms\n", kernel_time);

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

    cout << "Step2: CPU element-wise multiply (all weights = 1)\n";
    vector<float> h_products(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_products[i * N + j] = h_x[j];

    cout << "Step3: Copy data to device\n";
    float* d_products;
    cudaMalloc(&d_products, N * N * sizeof(float));
    cudaMemcpy(d_products, h_products.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "\n=== Running Naive vs Best ===\n";
    float t_naive = runNN(d_products, "Naive",                        sumReductionNaive);
    float t_best  = runNN(d_products, "Best (Seq. Addr + Warp Unroll)", sumReductionBest);

    printf("\n=== Summary Table ===\n");
    printf("  %-40s  %10s  %10s\n", "Kernel Implementation", "Perf (ms)", "Speedup");
    printf("  %-40s  %10.3f  %10.2fx\n", "Naive",                        t_naive, 1.0f);
    printf("  %-40s  %10.3f  %10.2fx\n", "Best (Seq. Addr + Warp Unroll)", t_best, t_naive / t_best);

    cudaFree(d_products);
    return 0;
}