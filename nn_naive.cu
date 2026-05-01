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
// Naive Sum Reduction (from Annex-1, unchanged)
// Each block reduces BLOCK_SIZE elements into 1 partial sum
// ---------------------------------------------------------------
__global__ void sumReductionNaive(float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared
    if (idx < n)
        sharedData[tid] = input[idx];
    else
        sharedData[tid] = 0.0f;

    __syncthreads();

    // Naive reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
        output[blockIdx.x] = sharedData[0];
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
// Final sum of hidden layer outputs (to get y)
// Single block reduction of 8192 h values
// ---------------------------------------------------------------
__global__ void finalSum(float* h, float* y, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        sharedData[tid] = h[idx];
    else
        sharedData[tid] = 0.0f;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        y[blockIdx.x] = sharedData[0];
}

// ---------------------------------------------------------------
int main() {

    // ---- Step 1: Generate input vector x (host) ----
    cout << "Step1: Generating input vector x and weight matrix W\n";
    vector<float> h_x(N);
    generate(h_x.begin(), h_x.end(), []() { return (float)(rand() % 100) / 100.0f; });

    // ---- Step 2: CPU element-wise multiply x * W ----
    // All weights = 1, so result[i][j] = x[j] for all neurons i
    // Result is N*N matrix stored row-major (neuron i = row i)
    cout << "Step2: CPU element-wise multiply (x * W, all weights=1)\n";
    vector<float> h_products(N * N);
    for (int i = 0; i < N; i++)           // for each neuron
        for (int j = 0; j < N; j++)       // for each input element
            h_products[i * N + j] = h_x[j]; // w[i][j] = 1

    // ---- Step 3: Allocate device memory ----
    cout << "Step3: Device memory allocation\n";
    float *d_products, *d_partial, *d_h, *d_partial2, *d_y;

    cudaMalloc(&d_products, N * N * sizeof(float));  // input to GPU
    cudaMalloc(&d_partial,  N * N / BLOCK_SIZE * sizeof(float)); // partial sums per neuron
    cudaMalloc(&d_h,        N * sizeof(float));      // hidden layer outputs
    cudaMalloc(&d_partial2, N / BLOCK_SIZE * sizeof(float));     // partial sums for final sum
    cudaMalloc(&d_y,        sizeof(float));           // output y

    // ---- Step 4: Timing ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float h2d_time, kernel_time, d2h_time;
    cudaEvent_t e1, e2, e3;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);

    cudaEventRecord(start, 0);

    // ---- Step 5: Copy products to device ----
    cout << "Step4: Copy data to device\n";
    cudaMemcpy(d_products, h_products.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(e1, 0);

    // ---- Step 6: GPU Computation ----
    cout << "Step5: GPU computation (reduction + sigmoid + output sum)\n";

    // Pass 1: reduce each row of 8192 elements into 32 partial sums
    // Launch: N blocks (one per neuron), each block has BLOCK_SIZE threads
    // Each neuron's row is reduced in 32 steps → 32 partial sums per neuron
    int blocksPerNeuron = N / BLOCK_SIZE;  // 8192/256 = 32

    // We launch blocksPerNeuron * N threads total
    // Trick: reshape — process all neurons' pass1 in one big launch
    sumReductionNaive<<<N * blocksPerNeuron, BLOCK_SIZE>>>(d_products, d_partial, N * N);

    // Pass 2: reduce 32 partial sums per neuron → 1 value = h[i]
    // Launch N blocks, each with 32 threads
    sumReductionNaive<<<N, blocksPerNeuron>>>(d_partial, d_h, N * blocksPerNeuron);

    // Apply sigmoid to all hidden layer outputs
    applySigmoid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_h, N);

    // Final sum: reduce all h[i] into y
    // Pass 1: 32 partial sums
    finalSum<<<blocksPerNeuron, BLOCK_SIZE>>>(d_h, d_partial2, N);
    // Pass 2: 1 final value
    finalSum<<<1, blocksPerNeuron>>>(d_partial2, d_y, blocksPerNeuron);

    cudaDeviceSynchronize();
    cudaEventRecord(e2, 0);

    // ---- Step 7: Copy result back ----
    float h_y = 0.0f;
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(e3, 0);

    cudaEventSynchronize(e3);

    cudaEventElapsedTime(&h2d_time,    start, e1);
    cudaEventElapsedTime(&kernel_time, e1,    e2);
    cudaEventElapsedTime(&d2h_time,    e2,    e3);
    float total = h2d_time + kernel_time + d2h_time;

    printf("\n--- Naive Sum Reduction Neural Network ---\n");
    printf("Output y = %f\n\n", h_y);
    printf("Host to Device Transfer : %f ms\n", h2d_time);
    printf("GPU Kernel Execution    : %f ms\n", kernel_time);
    printf("Device to Host Transfer : %f ms\n", d2h_time);
    printf("Total Time              : %f ms\n", total);

    // Cleanup
    cudaFree(d_products);
    cudaFree(d_partial);
    cudaFree(d_h);
    cudaFree(d_partial2);
    cudaFree(d_y);

    return 0;
}
