#include <iostream>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using std::cout;
using std::vector;
using std::generate;

int main() {
    int N = 8192;
    size_t bytes = N * N * sizeof(float);
    float Nbr_GFLOPS = 2 * N/1000.0 * N/1000.0 * N/1000.0;

    // Host vectors
    vector<float> h_a(N * N);
    vector<float> h_b(N * N);
    vector<float> h_c(N * N);

    cout << "Step1 : h_a and h_b generation\n";
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    cout << "Step2 : Mem Allocation on device\n";
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    cout << "Step3 : Launch Events to measure Time\n";
    float Total_gpu_time, Host2Dev_time, Kernel_time, Dev2Host_time;
    cudaEvent_t start, Host2dev, KernelExec, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&Host2dev);
    cudaEventCreate(&KernelExec);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cout << "Step3 : Copy Data To Device\n";
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(Host2dev, 0);

    // cuBLAS SGEMM: C = alpha*A*B + beta*C
    // NOTE: cuBLAS is column-major. To compute row-major A*B,
    // we call cublasSgemm with B and A swapped (equivalent transpose trick)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_b, N,   // B first (transpose trick)
                d_a, N,   // A second
                &beta,
                d_c, N);

    cudaEventRecord(KernelExec, 0);

    cout << "Step4 : Copy Result Back To Host\n";
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&Total_gpu_time, start, stop);
    cudaEventElapsedTime(&Host2Dev_time,  start, Host2dev);
    cudaEventElapsedTime(&Kernel_time,    Host2dev, KernelExec);
    cudaEventElapsedTime(&Dev2Host_time,  KernelExec, stop);

    printf("Time elapsed on Host To Device Transfer: %f ms.\n\n", Host2Dev_time);
    printf("Time elapsed on matrix multiplication on GPU (cuBLAS): %f ms.\n\n", Kernel_time);
    printf("Time elapsed on Device To Host Transfer: %f ms.\n\n", Dev2Host_time);
    printf("Total Time: %f ms.\n\n", Total_gpu_time);

    float Perf_GFLOPS = Nbr_GFLOPS * 1000 / Kernel_time;
    printf("cuBLAS Performance: %f GFLOPS.\n\n", Perf_GFLOPS);

    cout << "COMPLETED SUCCESSFULLY\n";

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}