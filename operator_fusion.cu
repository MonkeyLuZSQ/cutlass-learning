#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Operator fusion teaching code: reduction + softmax + GEMM-softmax fusion.
 *
 * Attention-style computation:
 *
 *   Scores[M, N] = A[M, K] * B[K, N]
 *   Prob[M, N]   = softmax(Scores, dim=N)
 *
 * Two paths are kept intentionally:
 *
 *   1. unfused path:
 *        GEMM writes Scores to global memory, then softmax reads Scores.
 *
 *   2. fused teaching path:
 *        GEMM accumulator stays inside the CTA, moves to shared memory,
 *        then the same CTA applies row softmax and writes Prob.
 *
 * Important limitation:
 *   Softmax needs the max and sum of a full row. The fused kernel below is
 *   only valid when one CTA covers the full N dimension, so N <= OP_BN.
 *   For large N, a real implementation needs multi-stage reductions or a
 *   FlashAttention-style online softmax.
 */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Same outer-product tiling style as gemm_tiling.cu.
#define OP_BM 64
#define OP_BN 64
#define OP_BK 16
#define OP_TM 8
#define OP_TN 8
#define OP_THREADS_M (OP_BM / OP_TM)
#define OP_THREADS_N (OP_BN / OP_TN)
#define SMEM_PAD 1

#define SOFTMAX_THREADS 256
#define WARMUP_ITERS 5
#define BENCH_ITERS 20

// ------------------------------------------------------------
// 1. Reduction helpers
// ------------------------------------------------------------

__device__ float block_reduce_sum(float value) {
    __shared__ float smem[SOFTMAX_THREADS];
    int tid = threadIdx.x;

    smem[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

__device__ float block_reduce_max(float value) {
    __shared__ float smem[SOFTMAX_THREADS];
    int tid = threadIdx.x;

    smem[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

// Each block outputs one partial sum. Larger reductions can launch this kernel
// repeatedly, or use CUB/cooperative groups in production code.
__global__ void reduce_sum_partial_kernel(const float* x, float* partial, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += x[i];
    }

    sum = block_reduce_sum(sum);
    if (tid == 0) {
        partial[blockIdx.x] = sum;
    }
}

// ------------------------------------------------------------
// 2. Standalone row-wise softmax
// ------------------------------------------------------------

/*
 * One block computes one row:
 *
 *   y = exp(x - row_max) / sum(exp(x - row_max))
 *
 * The subtraction by row_max is the standard numerical-stability trick.
 */
__global__ void row_softmax_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, x[row * cols + col]);
    }
    float row_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += __expf(x[row * cols + col] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);

    for (int col = tid; col < cols; col += blockDim.x) {
        y[row * cols + col] = __expf(x[row * cols + col] - row_max) / row_sum;
    }
}

void launch_row_softmax(const float* d_x, float* d_y, int rows, int cols) {
    row_softmax_kernel<<<rows, SOFTMAX_THREADS>>>(d_x, d_y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------
// 3. Outer-product GEMM baseline: materialize Scores
// ------------------------------------------------------------

__global__ void outer_gemm_scores_kernel(const float* A,
                                         const float* B,
                                         float* Scores,
                                         int M,
                                         int N,
                                         int K) {
    __shared__ float As[OP_BM][OP_BK + SMEM_PAD];
    __shared__ float Bs[OP_BK][OP_BN + SMEM_PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = OP_THREADS_M * OP_THREADS_N;

    int base_row = blockIdx.y * OP_BM + ty * OP_TM;
    int base_col = blockIdx.x * OP_BN + tx * OP_TN;

    float acc[OP_TM][OP_TN] = {0.0f};

    for (int kk = 0; kk < K; kk += OP_BK) {
        for (int idx = tid; idx < OP_BM * OP_BK; idx += num_threads) {
            int r = idx / OP_BK;
            int c = idx % OP_BK;
            int gr = blockIdx.y * OP_BM + r;
            int gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        for (int idx = tid; idx < OP_BK * OP_BN; idx += num_threads) {
            int r = idx / OP_BN;
            int c = idx % OP_BN;
            int gr = kk + r;
            int gc = blockIdx.x * OP_BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < OP_BK; ++k) {
            float a[OP_TM];
            float b[OP_TN];

            for (int i = 0; i < OP_TM; ++i) {
                a[i] = As[ty * OP_TM + i][k];
            }
            for (int j = 0; j < OP_TN; ++j) {
                b[j] = Bs[k][tx * OP_TN + j];
            }

            for (int i = 0; i < OP_TM; ++i) {
                for (int j = 0; j < OP_TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < OP_TM; ++i) {
        for (int j = 0; j < OP_TN; ++j) {
            int row = base_row + i;
            int col = base_col + j;
            if (row < M && col < N) {
                Scores[row * N + col] = acc[i][j];
            }
        }
    }
}

void launch_unfused_gemm_softmax(const float* d_A,
                                 const float* d_B,
                                 float* d_scores,
                                 float* d_prob,
                                 int M,
                                 int N,
                                 int K) {
    dim3 block(OP_THREADS_N, OP_THREADS_M);
    dim3 grid((N + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);

    outer_gemm_scores_kernel<<<grid, block>>>(d_A, d_B, d_scores, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    launch_row_softmax(d_scores, d_prob, M, N);
}

// ------------------------------------------------------------
// 4. Fused teaching kernel: GEMM accumulator -> softmax
// ------------------------------------------------------------

__global__ void fused_outer_gemm_softmax_kernel(const float* A,
                                                const float* B,
                                                float* Prob,
                                                int M,
                                                int N,
                                                int K) {
    __shared__ float As[OP_BM][OP_BK + SMEM_PAD];
    __shared__ float Bs[OP_BK][OP_BN + SMEM_PAD];
    __shared__ float ScoreTile[OP_BM][OP_BN + SMEM_PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = OP_THREADS_M * OP_THREADS_N;

    int tile_row = blockIdx.y * OP_BM;
    int base_row = tile_row + ty * OP_TM;
    int base_col = tx * OP_TN;

    float acc[OP_TM][OP_TN] = {0.0f};

    for (int kk = 0; kk < K; kk += OP_BK) {
        for (int idx = tid; idx < OP_BM * OP_BK; idx += num_threads) {
            int r = idx / OP_BK;
            int c = idx % OP_BK;
            int gr = tile_row + r;
            int gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        for (int idx = tid; idx < OP_BK * OP_BN; idx += num_threads) {
            int r = idx / OP_BN;
            int c = idx % OP_BN;
            int gr = kk + r;
            Bs[r][c] = (gr < K && c < N) ? B[gr * N + c] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < OP_BK; ++k) {
            float a[OP_TM];
            float b[OP_TN];

            for (int i = 0; i < OP_TM; ++i) {
                a[i] = As[ty * OP_TM + i][k];
            }
            for (int j = 0; j < OP_TN; ++j) {
                b[j] = Bs[k][tx * OP_TN + j];
            }

            for (int i = 0; i < OP_TM; ++i) {
                for (int j = 0; j < OP_TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Fusion point: do not write Scores to global memory. Keep the CTA tile
    // on chip, then run softmax on ScoreTile.
    for (int i = 0; i < OP_TM; ++i) {
        for (int j = 0; j < OP_TN; ++j) {
            ScoreTile[ty * OP_TM + i][tx * OP_TN + j] = acc[i][j];
        }
    }
    __syncthreads();

    // Simple teaching version: one thread handles one row. Faster versions use
    // warp/block reductions so each row is processed by many threads.
    if (tid < OP_BM) {
        int local_row = tid;
        int global_row = tile_row + local_row;

        if (global_row < M) {
            float row_max = -INFINITY;
            for (int col = 0; col < N; ++col) {
                row_max = fmaxf(row_max, ScoreTile[local_row][col]);
            }

            float row_sum = 0.0f;
            for (int col = 0; col < N; ++col) {
                row_sum += __expf(ScoreTile[local_row][col] - row_max);
            }

            for (int col = 0; col < N; ++col) {
                Prob[global_row * N + col] =
                    __expf(ScoreTile[local_row][col] - row_max) / row_sum;
            }
        }
    }
}

void launch_fused_gemm_softmax(const float* d_A,
                               const float* d_B,
                               float* d_prob,
                               int M,
                               int N,
                               int K) {
    if (N > OP_BN) {
        fprintf(stderr, "fused kernel requires N <= %d, got N=%d\n", OP_BN, N);
        exit(EXIT_FAILURE);
    }

    dim3 block(OP_THREADS_N, OP_THREADS_M);
    dim3 grid(1, (M + OP_BM - 1) / OP_BM);
    fused_outer_gemm_softmax_kernel<<<grid, block>>>(d_A, d_B, d_prob, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------
// 5. Benchmark helpers: warmup + average time
// ------------------------------------------------------------

float average_time_ms(const float* times) {
    float sum = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        sum += times[i];
    }
    return sum / (float)BENCH_ITERS;
}

float benchmark_unfused_gemm_softmax(const float* d_A,
                                     const float* d_B,
                                     float* d_scores,
                                     float* d_prob,
                                     int M,
                                     int N,
                                     int K,
                                     cudaEvent_t start,
                                     cudaEvent_t stop) {
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch_unfused_gemm_softmax(d_A, d_B, d_scores, d_prob, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_unfused_gemm_softmax(d_A, d_B, d_scores, d_prob, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    return average_time_ms(times);
}

float benchmark_fused_gemm_softmax(const float* d_A,
                                   const float* d_B,
                                   float* d_prob,
                                   int M,
                                   int N,
                                   int K,
                                   cudaEvent_t start,
                                   cudaEvent_t stop) {
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch_fused_gemm_softmax(d_A, d_B, d_prob, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_fused_gemm_softmax(d_A, d_B, d_prob, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    return average_time_ms(times);
}

// ------------------------------------------------------------
// Minimal smoke test
// ------------------------------------------------------------

void init_matrix(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
}

int main() {
    const int M = 512;
    const int N = 64;
    const int K = 512;

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_unfused = (float*)malloc(bytes_C);
    float* h_fused = (float*)malloc(bytes_C);

    float *d_A, *d_B, *d_scores, *d_unfused, *d_fused;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_scores, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_unfused, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_fused, bytes_C));

    srand(0);
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ms_unfused = benchmark_unfused_gemm_softmax(d_A,
                                                      d_B,
                                                      d_scores,
                                                      d_unfused,
                                                      M,
                                                      N,
                                                      K,
                                                      start,
                                                      stop);
    float ms_fused = benchmark_fused_gemm_softmax(d_A,
                                                  d_B,
                                                  d_fused,
                                                  M,
                                                  N,
                                                  K,
                                                  start,
                                                  stop);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_unfused, d_unfused, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fused, d_fused, bytes_C, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = fmaxf(max_diff, fabsf(h_unfused[i] - h_fused[i]));
    }

    printf("operator_fusion.cu smoke test finished.\n");
    printf("unfused path: GEMM -> global Scores -> softmax\n");
    printf("fused path:   GEMM accumulator -> shared tile -> softmax\n");
    printf("timing: %d warmup launches, then average of %d timed launches\n",
           WARMUP_ITERS,
           BENCH_ITERS);
    printf("unfused GEMM + softmax: %.4f ms\n", ms_unfused);
    printf("fused GEMM + softmax:   %.4f ms\n", ms_fused);
    printf("max abs diff: %.8f\n", max_diff);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_unfused));
    CUDA_CHECK(cudaFree(d_fused));
    free(h_A);
    free(h_B);
    free(h_unfused);
    free(h_fused);

    return 0;
}
