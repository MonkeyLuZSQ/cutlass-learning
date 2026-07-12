#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace nvcuda;

// ============================================================
// WMMA GEMM 参数 (SM 75, Turing: GTX 1660 SUPER)
//
// WMMA 指令形状: m16 × n16 × k16
//   → 一条 mma_sync 指令完成 16×16×16 = 4096 次乘加
//   → 输入 FP16, 累加器 FP32
//
// 三层 tiling:
//   Block tile:  64×64   (一个 thread block 的输出)
//   Warp tile:   32×32   (一个 warp 的输出, 4 warps/block)
//   Instr tile:  16×16×16 (一次 WMMA 指令的输出)
//   每个 warp 做 2×2 = 4 次 WMMA 覆盖 32×32
//
// K 分块: BK = 16 (匹配 WMMA 的 k 维度)
// ============================================================
#define BM 64
#define BN 64
#define BK 16         // 匹配 WMMA m16n16k16 (SM 75 Turing 支持的形状)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_M 2     // BM / (2 * WMMA_M) = 64 / 32 = 2
#define WARPS_N 2     // BN / (2 * WMMA_N) = 64 / 32 = 2
#define NUM_WARPS (WARPS_M * WARPS_N)  // 4
#define THREADS_PER_BLOCK (NUM_WARPS * 32)  // 128

// 对比用: 之前的外积 kernel 参数
#define OP_BM 64
#define OP_BN 64
#define OP_BK 16
#define OP_TM 8
#define OP_TN 8
#define OP_THREADS_M (OP_BM / OP_TM)
#define OP_THREADS_N (OP_BN / OP_TN)
#define SMEM_PAD 1

#define WARMUP_ITERS 5
#define BENCH_ITERS 20

// ============================================================
// WMMA GEMM kernel
//
// 全局内存: FP32 (A, B, C 都是 float)
// 共享内存: FP16 (As, Bs 用 half 存储, 节省带宽)
// 计算:     Tensor Core (FP16 输入, FP32 累加)
//
// 流程: FP32 → load & convert → FP16 smem → WMMA → FP32 C
// ============================================================
__global__ void gemm_wmma(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    // ---- Shared memory: FP16, 带 padding 消除 bank conflict ----
    __shared__ half As[BM][BK];       // 64 × 16
    __shared__ half Bs[BK][BN];       // 16 × 64

    int tid = threadIdx.x;
    int warp_id = tid / 32;           // 0..3
    int lane_id = tid % 32;           // 0..31

    // warp 在 block tile 中的位置 (2×2 网格)
    int warp_m = warp_id / WARPS_N;   // 0 or 1
    int warp_n = warp_id % WARPS_N;   // 0 or 1

    // ============================================================
    // 初始化 WMMA 累加器（必须在 K 循环外部！）
    // 每个 warp 负责 32×32 输出，分为 2×2 个 16×16 WMMA 操作
    // ============================================================
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // ============================================================
    // 主循环: 沿 K 维度, 每次处理 BK=16 列
    // ============================================================
    for (int kk = 0; kk < K; kk += BK) {

        // ---- 协作加载 shared memory ----
        // 128 线程加载 As[64][16] = 1024 个元素, 每人 8 个
        // 加载 Bs[16][64] = 1024 个元素, 每人 8 个
        // 同时完成 FP32 → FP16 转换

        for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
            int r = idx / BK;
            int c = idx % BK;
            int gr = blockIdx.y * BM + r;
            int gc = kk + c;
            As[r][c] = __float2half((gr < M && gc < K) ? A[gr * K + gc] : 0.0f);
        }

        for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = kk + r;
            int gc = blockIdx.x * BN + c;
            Bs[r][c] = __float2half((gr < K && gc < N) ? B[gr * N + gc] : 0.0f);
        }

        __syncthreads();

        // ============================================================
        // WMMA 计算: 2×2 个 mma_sync
        //
        //   warp (0,0): A[0:16,:] × B[:,0:16]  → c_frag[0][0]
        //               A[0:16,:] × B[:,16:32] → c_frag[0][1]
        //   warp (0,1): A[0:16,:] × B[:,32:48] → c_frag[0][0] (warp 视角)
        //               ...
        //
        // 每个 mma_sync: 16×16 × 16×16 → 16×16
        //   一条指令完成 16×16×16 = 4096 次乘加!
        // ============================================================
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                // A fragment: As 中 16 行 × 16 列
                // row_major: 同一行内列连续, leading dimension = BK = 16
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &As[warp_m * 32 + i * WMMA_M][0],
                    BK);

                // B fragment: Bs 中 16 行 × 16 列
                // Bs[K][N] 是 C row-major 存储, N 维连续
                // row_major: leading dimension = BN = 64
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    &Bs[0][warp_n * 32 + j * WMMA_N],
                    BN);                                  // leading dimension = BN = 64

                // Tensor Core 矩阵乘累加
                wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // ============================================================
    // 写回结果（在 K 循环外部，所有 K-tile 累加完成后）
    // ============================================================
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int out_row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
            int out_col = blockIdx.x * BN + warp_n * 32 + j * WMMA_N;

            if (out_row < M && out_col < N) {
                wmma::store_matrix_sync(
                    &C[out_row * N + out_col],
                    c_frag[i][j],
                    N,
                    wmma::mem_row_major
                );
            }
        }
    }
}

// ============================================================
// 对比用: 之前的外积 kernel (FP32, 无 Tensor Core)
// ============================================================
__global__ void gemm_outer_product_fp32(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    __shared__ float As[OP_BM][OP_BK + SMEM_PAD];
    __shared__ float Bs[OP_BK][OP_BN + SMEM_PAD];

    int tx = threadIdx.x, ty = threadIdx.y;
    int base_row = blockIdx.y * OP_BM + ty * OP_TM;
    int base_col = blockIdx.x * OP_BN + tx * OP_TN;

    float regC[OP_TM][OP_TN];
    #pragma unroll
    for (int i = 0; i < OP_TM; i++)
        #pragma unroll
        for (int j = 0; j < OP_TN; j++)
            regC[i][j] = 0.0f;

    for (int kk = 0; kk < K; kk += OP_BK) {
        int tid = ty * blockDim.x + tx;
        for (int idx = tid; idx < OP_BM * OP_BK; idx += OP_THREADS_M * OP_THREADS_N) {
            int r = idx / OP_BK, c = idx % OP_BK;
            int gr = blockIdx.y * OP_BM + r, gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = tid; idx < OP_BK * OP_BN; idx += OP_THREADS_M * OP_THREADS_N) {
            int r = idx / OP_BN, c = idx % OP_BN;
            int gr = kk + r, gc = blockIdx.x * OP_BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < OP_BK; k++) {
            float regA[OP_TM], regB[OP_TN];
            #pragma unroll
            for (int i = 0; i < OP_TM; i++) regA[i] = As[ty * OP_TM + i][k];
            #pragma unroll
            for (int j = 0; j < OP_TN; j++) regB[j] = Bs[k][tx * OP_TN + j];
            #pragma unroll
            for (int i = 0; i < OP_TM; i++)
                #pragma unroll
                for (int j = 0; j < OP_TN; j++)
                    regC[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < OP_TM; i++)
        #pragma unroll
        for (int j = 0; j < OP_TN; j++) {
            int r = base_row + i, c = base_col + j;
            if (r < M && c < N) C[r * N + c] = regC[i][j];
        }
}

// ============================================================
// Host 辅助函数
// ============================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (float)(rand() % 100) / 100.0f;
}

// mode 0: 绝对误差 (FP32 kernel), mode 1: 相对误差 (WMMA FP16)
bool verify(const float* ref, const float* test, int M, int N,
            float eps, bool relative = false) {
    int mismatches = 0;
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(ref[i] - test[i]);
        float err = relative ? diff / (fabs(ref[i]) + 1e-8f) : diff;
        if (relative && err > max_rel_err) max_rel_err = err;
        if (err > eps) {
            if (mismatches < 5)
                printf("  Mismatch at %d: ref=%f, test=%f (%s_err=%f)\n",
                       i, ref[i], test[i],
                       relative ? "rel" : "abs", err);
            mismatches++;
        }
    }
    if (mismatches > 0)
        printf("  Total mismatches: %d / %d (%.2f%%)\n",
               mismatches, M * N, 100.0f * mismatches / (M * N));
    if (relative)
        printf("  Max relative error: %.6f\n", max_rel_err);
    return mismatches == 0;
}

float benchmark_kernel(void (*kernel)(const float*, const float*, float*, int, int, int),
                       dim3 grid, dim3 block,
                       const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K,
                       cudaEvent_t start, cudaEvent_t stop) {
    for (int i = 0; i < WARMUP_ITERS; i++)
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; i++) {
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    for (int i = 0; i < BENCH_ITERS - 1; i++)
        for (int j = i + 1; j < BENCH_ITERS; j++)
            if (times[j] < times[i]) {
                float t = times[i]; times[i] = times[j]; times[j] = t;
            }
    return times[BENCH_ITERS / 2];
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C_ref, *h_C_wmma, *h_C_outer;
    float *d_A, *d_B, *d_C_wmma, *d_C_outer;

    h_A = (float*)malloc(sA);
    h_B = (float*)malloc(sB);
    h_C_ref = (float*)malloc(sC);
    h_C_wmma = (float*)malloc(sC);
    h_C_outer = (float*)malloc(sC);

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, sA);
    cudaMalloc(&d_B, sB);
    cudaMalloc(&d_C_wmma, sC);
    cudaMalloc(&d_C_outer, sC);
    cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice);

    // WMMA kernel: block = (128, 1, 1), grid = (8, 8)
    dim3 block_wmma(THREADS_PER_BLOCK);
    dim3 grid_wmma((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 外积 kernel: block = (8, 8), grid = (8, 8)
    dim3 block_outer(OP_THREADS_N, OP_THREADS_M);
    dim3 grid_outer((N + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);

    printf("WMMA kernel:    Grid=(%d,%d), Block=%d threads (%d warps)\n",
           grid_wmma.x, grid_wmma.y, THREADS_PER_BLOCK, NUM_WARPS);
    printf("                  WMMA shape: %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Outer product:  Grid=(%d,%d), Block=(%d,%d)=%d threads\n\n",
           grid_outer.x, grid_outer.y, OP_THREADS_N, OP_THREADS_M,
           OP_THREADS_N * OP_THREADS_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running benchmarks...\n");

    float ms_wmma = benchmark_kernel(gemm_wmma, grid_wmma, block_wmma,
                                     d_A, d_B, d_C_wmma, M, N, K, start, stop);
    float ms_outer = benchmark_kernel(gemm_outer_product_fp32, grid_outer, block_outer,
                                      d_A, d_B, d_C_outer, M, N, K, start, stop);

    printf("Computing CPU reference...\n");
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += h_A[i * K + k] * h_B[k * N + j];
            h_C_ref[i * N + j] = s;
        }

    cudaMemcpy(h_C_wmma, d_C_wmma, sC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_outer, d_C_outer, sC, cudaMemcpyDeviceToHost);

    printf("\nVerifying WMMA result (FP16 precision, rel_eps=5%%)...\n");
    bool pass_wmma = verify(h_C_ref, h_C_wmma, M, N, 0.05f, true);
    printf("Verifying outer product result (FP32 precision, abs_eps=1e-2)...\n");
    bool pass_outer = verify(h_C_ref, h_C_outer, M, N, 1e-2, false);

    printf("\n");
    printf("================================================================\n");
    printf("  GEMM Benchmark: WMMA (Tensor Core) vs Outer Product (FP32)\n");
    printf("  Matrix: %dx%dx%d\n", M, N, K);
    printf("================================================================\n");
    printf("  WMMA kernel (FP16 in, FP32 acc):   %s   %8.4f ms\n",
           pass_wmma ? "PASS" : "FAIL", ms_wmma);
    printf("  Outer product (FP32):              %s   %8.4f ms\n",
           pass_outer ? "PASS" : "FAIL", ms_outer);
    printf("----------------------------------------------------------------\n");
    printf("  Tensor Core speedup: %.2fx\n", ms_outer / ms_wmma);
    printf("================================================================\n");

    free(h_A); free(h_B); free(h_C_ref); free(h_C_wmma); free(h_C_outer);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_wmma); cudaFree(d_C_outer);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
