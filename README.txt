Compile commands (Linux/nvcc):
kernel.cu        : nvcc kernel.cu -o matmul -std=c++14 -ccbin g++-10
matmul_cublas.cu : nvcc matmul_cublas.cu -o cublas_matmul -std=c++14 -ccbin g++-10 -lcublas
matmul_tensorcore.cu : nvcc matmul_tensorcore.cu -o matmul_tc -std=c++14 -ccbin g++-10 -arch=sm_75
nn_improved.cu   : nvcc nn_improved.cu -o nn_improved -std=c++14 -ccbin g++-10

Tested on: RTX 4060, CUDA 11.x, Pop!_OS Linux