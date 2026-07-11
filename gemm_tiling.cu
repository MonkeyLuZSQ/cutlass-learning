#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// 统一参数 —— 三个 kernel 使用完全相同的 grid/block
// ============================================================
// Block tile: 每个 thread block 负责 BM × BN 的输出
#define BM 64
#define BN 64
#define BK 16
// Thread tile: 每个线程负责 TM × TN 的输出
#define TM 4
#define TN 4
// 线程块维度: (BM/TM) × (BN/TN) = 16×16 = 256 threads
#define THREADS_M (BM / TM)   // 16
#define THREADS_N (BN / TN)   // 16
// Shared memory padding
#define SMEM_PAD 1

// Warmup & benchmark 参数
#define WARMUP_ITERS 2
#define BENCH_ITERS 5

// ============================================================
// 朴素 GEMM kernel（统一 grid/block 版本）
//
// 与 CUTLASS 外积使用完全相同的 grid 和 block：
//   block = (16, 16) = 256 threads
//   grid  = (N/64, M/64)
//
// 区别：每个线程独立计算 TM×TN=4×4 个输出元素，
//       不使用 shared memory，每个元素都从 global memory 直接读取。
//       16 个输出元素 × K 次乘加 = 16K 次 global memory 读取（每元素）
// ============================================================
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    // 当前线程负责的 TM×TN 输出块的起始全局坐标
    int base_row = blockIdx.y * BM + ty * TM;  // 每线程起始行
    int base_col = blockIdx.x * BN + tx * TN;  // 每线程起始列

    // 对 TM×TN 个输出元素逐个计算（朴素：无共享，无寄存器复用）
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = base_row + i;
            int col = base_col + j;
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                    sum += A[row * K + k] * B[k * N + col];
                C[row * N + col] = sum;
            }
        }
    }
}

// ============================================================
// 内积分块 kernel（统一 grid/block 版本）
//
// 与 CUTLASS 外积使用完全相同的 grid 和 block，
// 同时使用相同的 shared memory tile 大小 (BM×BK, BK×BN)。
//
// 区别：每个线程计算 TM×TN=4×4 个输出元素，
//       但计算方式是"内积"——对每个输出元素独立做 K 维度的点积。
//       即：C[row][col] = Σ_k As[row][k] * Bs[k][col]
//
// 对比 CUTLASS 外积：
//   内积：外层循环 K，每个输出元素独立累加 → 寄存器无法跨元素复用
//   外积：外层循环 K，每次取 A 列 ⊗ B 行 → 寄存器复用 A/B fragment
// ============================================================
__global__ void gemm_inner_product(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // 与 CUTLASS 外积相同的 shared memory 布局（含 padding）
    __shared__ float As[BM][BK + SMEM_PAD];      // 64 × 17
    __shared__ float Bs[BK][BN + SMEM_PAD];      // 16 × 65

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    // 当前线程负责的 TM×TN 输出块的起始全局坐标
    int base_row = blockIdx.y * BM + ty * TM;
    int base_col = blockIdx.x * BN + tx * TN;

    // 每个线程的 TM×TN 个累加器
    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            regC[i][j] = 0.0f;

    // 主循环：沿 K 维度分块
    for (int kk = 0; kk < K; kk += BK) {

        // ---- 协作加载 shared memory（与 CUTLASS 外积完全相同）----
        int tid = ty * blockDim.x + tx;  // 线性 ID 0..255

        // 加载 As[64][16] = 1024 元素，256 线程每人 4 个
        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += THREADS_M * THREADS_N) {
            int smem_row = idx / BK;
            int smem_col = idx % BK;
            int gmem_row = blockIdx.y * BM + smem_row;
            int gmem_col = kk + smem_col;
            As[smem_row][smem_col] = (gmem_row < M && gmem_col < K)
                                     ? A[gmem_row * K + gmem_col] : 0.0f;
        }

        // 加载 Bs[16][64] = 1024 元素，256 线程每人 4 个
        #pragma unroll
        for (int idx = tid; idx < BK * BN; idx += THREADS_M * THREADS_N) {
            int smem_row = idx / BN;
            int smem_col = idx % BN;
            int gmem_row = kk + smem_row;
            int gmem_col = blockIdx.x * BN + smem_col;
            Bs[smem_row][smem_col] = (gmem_row < K && gmem_col < N)
                                     ? B[gmem_row * N + gmem_col] : 0.0f;
        }

        __syncthreads();

        // ============ 内积计算 ============
        // 每个线程独立计算自己的 TM×TN 个输出元素
        // 对每个 (i,j)，做 BK 次乘加：Σ_k As[base_row+i][k] * Bs[k][base_col+j]
        //
        // 关键区别：这里每个输出元素需要独立访问 As 和 Bs，
        // 无法像外积那样复用 regA/ regB fragment。
        // 总访问次数：TM × TN × BK = 4 × 4 × 16 = 256 次 shared memory 读取
        // 而外积只需：BK × (TM + TN) = 16 × 8 = 128 次（减半！）

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < BK; ++k)
                    sum += As[ty * TM + i][k] * Bs[k][tx * TN + j];
                regC[i][j] += sum;
            }
        }

        __syncthreads();
    }

    // ---- 写回结果 ----
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = base_row + i;
            int c = base_col + j;
            if (r < M && c < N)
                C[r * N + c] = regC[i][j];
        }
}

// ============================================================
// CUTLASS 风格外积 kernel（不变）
// 关键区别：在 BK 循环内部，先取 A 列向量 regA[TM] 和 B 行向量 regB[TN]，
// 再做外积 regA ⊗ regB → TM×TN 次 F
// 这里虽然按照cutlass的风格做了外积实现，但是cutlass的实现是更复杂的，使用了更多的优化手段，比如warp-level primitives，双缓冲机制和更复杂的内存访问模式。这个kernel只是一个简化版，主要用于教学和对比。
// ============================================================
__global__ void gemm_outer_product_cutlass(const float* A, const float* B, float* C,
                                           int M, int N, int K) {
    __shared__ float As[BM][BK + SMEM_PAD];      // 64 × 17
    __shared__ float Bs[BK][BN + SMEM_PAD];      // 16 × 65

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    int base_row = blockIdx.y * BM + ty * TM;
    int base_col = blockIdx.x * BN + tx * TN;

    float regC[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            regC[i][j] = 0.0f;

    for (int kk = 0; kk < K; kk += BK) {

        // 协作加载 shared memory（与内积完全相同）
        int tid = ty * blockDim.x + tx;

        #pragma unroll
        for (int idx = tid; idx < BM * BK; idx += THREADS_M * THREADS_N) {
            int smem_row = idx / BK;
            int smem_col = idx % BK;
            int gmem_row = blockIdx.y * BM + smem_row;
            int gmem_col = kk + smem_col;
            As[smem_row][smem_col] = (gmem_row < M && gmem_col < K)
                                     ? A[gmem_row * K + gmem_col] : 0.0f;
        }

        #pragma unroll
        for (int idx = tid; idx < BK * BN; idx += THREADS_M * THREADS_N) {
            int smem_row = idx / BN;
            int smem_col = idx % BN;
            int gmem_row = kk + smem_row;
            int gmem_col = blockIdx.x * BN + smem_col;
            Bs[smem_row][smem_col] = (gmem_row < K && gmem_col < N)
                                     ? B[gmem_row * N + gmem_col] : 0.0f;
        }

        __syncthreads();

        // ============ 外积计算 ============
        // 关键区别：在 BK 循环内部，先取 A 列向量 regA[TM] 和 B 行向量 regB[TN]，
        // 再做外积 regA ⊗ regB → TM×TN 次 FMA
        //
        // 总 shared memory 访问次数：BK × (TM + TN) = 16 × 8 = 128 次
        // 而内积需要：TM × TN × BK = 4 × 4 × 16 = 256 次
        // 外积的 shared memory 访问量减半！

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regA[i] = As[ty * TM + i][k];

            float regB[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                regB[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    regC[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = base_row + i;
            int c = base_col + j;
            if (r < M && c < N)
                C[r * N + c] = regC[i][j];
        }
}

// ============================================================
// Host 辅助函数
// ============================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = (float)(rand() % 100) / 100.0f;
}

bool verify(const float* ref, const float* test, int M, int N, float eps = 1e-2) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(ref[i] - test[i]) > eps) {
            printf("  Mismatch at %d: ref=%f, test=%f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// 通用 benchmark：warmup + 多次迭代取中位数
float benchmark_kernel(void (*kernel)(const float*, const float*, float*, int, int, int),
                       dim3 grid, dim3 block,
                       const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K,
                       cudaEvent_t start, cudaEvent_t stop) {
    // warmup：让 GPU 升到 boost clock
    for (int i = 0; i < WARMUP_ITERS; ++i)
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 多次迭代
    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // 排序取中位数
    for (int i = 0; i < BENCH_ITERS - 1; ++i)
        for (int j = i + 1; j < BENCH_ITERS; ++j)
            if (times[j] < times[i]) {
                float t = times[i]; times[i] = times[j]; times[j] = t;
            }
    return times[BENCH_ITERS / 2];
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C_ref, *h_C_naive, *h_C_inner, *h_C_cutlass;
    float *d_A, *d_B, *d_C_naive, *d_C_inner, *d_C_cutlass;

    h_A = (float*)malloc(sA);
    h_B = (float*)malloc(sB);
    h_C_ref = (float*)malloc(sC);
    h_C_naive = (float*)malloc(sC);
    h_C_inner = (float*)malloc(sC);
    h_C_cutlass = (float*)malloc(sC);

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // 分配设备内存
    cudaMalloc(&d_A, sA);
    cudaMalloc(&d_B, sB);
    cudaMalloc(&d_C_naive, sC);
    cudaMalloc(&d_C_inner, sC);
    cudaMalloc(&d_C_cutlass, sC);
    cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice);

    // ============================================================
    // 三个 kernel 使用完全相同的 grid 和 block！
    // ============================================================
    dim3 block(THREADS_N, THREADS_M);  // (16, 16) = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // (8, 8) = 64 blocks

    printf("Grid = (%d, %d), Block = (%d, %d) = %d threads\n",
           grid.x, grid.y, block.x, block.y, block.x * block.y);
    printf("Block tile = %dx%d, Thread tile = %dx%d\n\n", BM, BN, TM, TN);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ============================================================
    // GPU benchmark
    // ============================================================
    printf("Running GPU benchmarks (warmup=%d, iters=%d, median)...\n",
           WARMUP_ITERS, BENCH_ITERS);

    float ms_naive   = benchmark_kernel(gemm_naive, grid, block,
                                        d_A, d_B, d_C_naive, M, N, K, start, stop);
    float ms_inner   = benchmark_kernel(gemm_inner_product, grid, block,
                                        d_A, d_B, d_C_inner, M, N, K, start, stop);
    float ms_cutlass = benchmark_kernel(gemm_outer_product_cutlass, grid, block,
                                        d_A, d_B, d_C_cutlass, M, N, K, start, stop);

    // ============================================================
    // CPU 参考计算
    // ============================================================
    printf("Computing CPU reference for verification...\n");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k)
                s += h_A[i * K + k] * h_B[k * N + j];
            h_C_ref[i * N + j] = s;
        }

    // 拷贝结果回 host
    cudaMemcpy(h_C_naive, d_C_naive, sC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_inner, d_C_inner, sC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cutlass, d_C_cutlass, sC, cudaMemcpyDeviceToHost);

    // 验证
    bool pass_naive   = verify(h_C_ref, h_C_naive, M, N);
    bool pass_inner   = verify(h_C_ref, h_C_inner, M, N);
    bool pass_cutlass = verify(h_C_ref, h_C_cutlass, M, N);

    printf("\n");
    printf("============================================================\n");
    printf("  GEMM Benchmark (Unified Grid/Block)  |  Matrix: %dx%dx%d\n", M, N, K);
    printf("============================================================\n");
    printf("  Grid = (%d, %d), Block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    printf("  Block tile = %dx%d, Thread tile = %dx%d\n", BM, BN, TM, TN);
    printf("------------------------------------------------------------\n");
    printf("  Naive kernel:          %s   %8.4f ms\n",
           pass_naive ? "PASS" : "FAIL", ms_naive);
    printf("  Inner product kernel:  %s   %8.4f ms\n",
           pass_inner ? "PASS" : "FAIL", ms_inner);
    printf("  CUTLASS outer kernel:  %s   %8.4f ms\n",
           pass_cutlass ? "PASS" : "FAIL", ms_cutlass);
    printf("------------------------------------------------------------\n");
    printf("  Speedup naive/inner:    %.2fx\n", ms_naive / ms_inner);
    printf("  Speedup naive/cutlass:  %.2fx\n", ms_naive / ms_cutlass);
    printf("  Speedup inner/cutlass:  %.2fx\n", ms_inner / ms_cutlass);
    printf("============================================================\n");

    // 清理
    free(h_A); free(h_B); free(h_C_ref);
    free(h_C_naive); free(h_C_inner); free(h_C_cutlass);
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C_naive); cudaFree(d_C_inner); cudaFree(d_C_cutlass);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
