#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;
using std::cout;
using std::vector;
using std::generate;

// TensorCore tile size (fixed by wmma API)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ---------------------------------------------------------------
// Conversion kernel: float -> half on device
// ---------------------------------------------------------------
__global__ void convertFloatToHalf(const float* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ---------------------------------------------------------------
// TensorCore kernel
// Each warp computes one 16x16 tile of C
// ---------------------------------------------------------------
__global__ void matmul_tensorcore(const half* a, const half* b, float* c, int N) {

    // Warp position in the output matrix (in units of 16x16 tiles)
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN =  blockIdx.y * blockDim.y + threadIdx.y;

    // Declare wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in steps of WMMA_K (16)
    for (int k = 0; k < N; k += WMMA_K) {

        int aRow = warpM * WMMA_M;  // row in A
        int aCol = k;               // col in A
        int bRow = k;               // row in B
        int bCol = warpN * WMMA_N;  // col in B

        if (aRow < N && aCol < N && bRow < N && bCol < N) {
            // Load 16x16 tiles from global memory into fragments
            wmma::load_matrix_sync(a_frag, a + aRow * N + aCol, N);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // TensorCore multiply-accumulate: c_frag += a_frag * b_frag
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result tile back to global memory
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < N && cCol < N) {
        wmma::store_matrix_sync(c + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------
int main() {
    int N = 8192;
    size_t bytes_f = N * N * sizeof(float);
    size_t bytes_h = N * N * sizeof(half);
    float Nbr_GFLOPS = 2 * N/1000.0 * N/1000.0 * N/1000.0;

    // Host vectors (float)
    vector<float> h_a(N * N);
    vector<float> h_b(N * N);
    vector<float> h_c(N * N);

    cout << "Step1 : h_a and h_b generation\n";
    generate(h_a.begin(), h_a.end(), []() { return (float)(rand() % 100); });
    generate(h_b.begin(), h_b.end(), []() { return (float)(rand() % 100); });

    cout << "Step2 : Mem Allocation on device\n";
    // Float device buffers (for input/output transfer)
    float *d_a_f, *d_b_f, *d_c;
    cudaMalloc(&d_a_f, bytes_f);
    cudaMalloc(&d_b_f, bytes_f);
    cudaMalloc(&d_c,   bytes_f);

    // Half device buffers (for TensorCore input)
    half *d_a_h, *d_b_h;
    cudaMalloc(&d_a_h, bytes_h);
    cudaMalloc(&d_b_h, bytes_h);

    cout << "Step3 : Launch Events to measure Time\n";
    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    cudaEvent_t start, Host2dev, KernelExec, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Copy float data to device
    cout << "Step3 : Copy Data To Device\n";
    cudaMemcpy(d_a_f, h_a.data(), bytes_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_f, h_b.data(), bytes_f, cudaMemcpyHostToDevice);

    // Convert float -> half on device
    int totalElements = N * N;
    int convThreads = 256;
    int convBlocks = (totalElements + convThreads - 1) / convThreads;
    convertFloatToHalf<<<convBlocks, convThreads>>>(d_a_f, d_a_h, totalElements);
    convertFloatToHalf<<<convBlocks, convThreads>>>(d_b_f, d_b_h, totalElements);

    cudaEventRecord(Host2dev, 0);

    // Launch TensorCore kernel
    // Each warp handles one 16x16 tile
    // blockDim: 128 threads = 4 warps along M dimension, 4 along N
    const int WARP_SIZE = 32;
    dim3 blockDim(128, 4);
    dim3 gridDim((N / WMMA_M + (blockDim.x / WARP_SIZE) - 1) / (blockDim.x / WARP_SIZE),
                 (N / WMMA_N + blockDim.y - 1) / blockDim.y);

    matmul_tensorcore<<<gridDim, blockDim>>>(d_a_h, d_b_h, d_c, N);

    cudaEventRecord(KernelExec, 0);

    // Copy result back
    cout << "Step4 : Copy Result Back To Host\n";
    cudaMemcpy(h_c.data(), d_c, bytes_f, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time,  start, Host2dev);
    cudaEventElapsedTime(&Kernel_time,    Host2dev, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time,  KernelExec, stop);

    printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
    printf("Time elapsed on TensorCore matmul on GPU: %f ms.\n\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total_gpu_time);

    float Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
    printf("TensorCore Performance: %f GFLOPS.\n\n", Perf_GFLOPS);

    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_a_f); cudaFree(d_b_f); cudaFree(d_c);
    cudaFree(d_a_h); cudaFree(d_b_h);

    return 0;
}