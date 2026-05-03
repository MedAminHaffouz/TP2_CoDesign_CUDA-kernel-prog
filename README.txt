Compile commands (Linux/nvcc):
kernel.cu        : nvcc kernel.cu -o matmul -std=c++14 -ccbin g++-10
matmul_cublas.cu : nvcc matmul_cublas.cu -o cublas_matmul -std=c++14 -ccbin g++-10 -lcublas
matmul_tensorcore.cu : nvcc matmul_tensorcore.cu -o matmul_tc -std=c++14 -ccbin g++-10 -arch=sm_75
nn_improved.cu   : nvcc nn_improved.cu -o nn_improved -std=c++14 -ccbin g++-10
nn_final.cu : nvcc nn_final.cu -o nn_final -std=c++14 -ccbin g++-10

Tested on: RTX 4060, CUDA 11.x, Pop!_OS Linux

----------------

================================================================
  CUDA TP - GL3 Codesign
  Run these on your machine and note down the outputs
================================================================

----------------------------------------------------------------
STEP 0 — One-time setup (Arch Linux + RTX 3050)
----------------------------------------------------------------

1. Install CUDA toolkit:
   sudo pacman -S cuda

2. Check it works:
   nvcc --version
   (should show CUDA 12.x)

3. Check your GPU is detected:
   nvidia-smi
   (should show RTX 3050)

4. Find your compatible g++ version:
   ls /usr/bin/g++*
   (use whatever version appears, e.g. g++-12 or g++-11)

Note: if nvcc gives errors about g++ version, add -ccbin g++-XX
to every compile command below, where XX is your version number.

----------------------------------------------------------------
PART 1 — Matrix Multiplication (Float + Int)
----------------------------------------------------------------

Folder: part1/

The file kernel.cu contains 2 kernels: matrixMulXrow and matrixMulYrow.
By default it runs matrixMulXrow. You need to run it 4 times total.

FLOAT runs (default, no changes needed):

  Step A — compile:
    cd part1
    nvcc kernel.cu -o matmul -std=c++14

  Step B — run MatmulXrow (already set by default):
    ./matmul
    → press Enter when it pauses
    → NOTE DOWN: Total Time value

  Step C — open kernel.cu, find this line (around line 131):
    matrixMulXrow<<<blocks, threads>>>(d_a, d_b, d_c, N);
    change to:
    matrixMulYrow<<<blocks, threads>>>(d_a, d_b, d_c, N);

  Step D — recompile and run:
    nvcc kernel.cu -o matmul -std=c++14
    ./matmul
    → NOTE DOWN: Total Time value

INT runs — open kernel.cu and replace ALL occurrences of:
  - vector<float>  →  vector<int>
  - sizeof(float)  →  sizeof(int)
  - const float*   →  const int*
  - float*         →  int* (for d_a, d_b, d_c only)

Then repeat Steps B, C, D for both kernels.

What to note for each run:
  → Total Time: XXXX ms   (that's all that goes in the table)

----------------------------------------------------------------
PART 2 — cuBLAS and TensorCore
----------------------------------------------------------------

Folder: part2/

--- cuBLAS ---
  nvcc matmul_cublas.cu -o cublas_matmul -std=c++14 -lcublas
  ./cublas_matmul
  → NOTE DOWN: Kernel Time + GFLOPS values

--- TensorCore ---
  nvcc matmul_tensorcore.cu -o matmul_tc -std=c++14 -arch=sm_80
  ./matmul_tc
  → NOTE DOWN: Kernel Time + GFLOPS values

Note: -arch=sm_80 is for Ampere (RTX 3050).
If it fails try -arch=sm_75.

----------------------------------------------------------------
PART 3 — Sum Reduction Neural Network
----------------------------------------------------------------

Folder: part3/

  nvcc nn_final.cu -o nn_final -std=c++14
  ./nn_final

  → NOTE DOWN the full Summary Table printed at the end:
    Naive time + Speedup
    Round2 time + Speedup
    Round3 time + Speedup

That's all. Just send me screenshots of all terminal outputs.
================================================================