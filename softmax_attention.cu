#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

//外积分块gemm
// 一级分块 CTA tile: OP_BM x OP_BN
#define OP_BM 64
#define OP_BN 64
#define OP_BK 16

//二级分块 Warp tile: OP_TM x OP_TN
#define OP_TM 8
#define OP_TN 8

// 三级分块 Thread tile: OP_THREADS_M x OP_THREADS_N
#define OP_THREADS_M (OP_BM / OP_TM)
#define OP_THREADS_N (OP_BN / OP_TN)

//shared memory padding to avoid bank conflicts
#define PAD 1

#define WARMUP_ITERS 5
#define BENCH_ITERS 20

#ifndef USE_OUTER_GEMM
#define USE_OUTER_GEMM 1
#endif

#if USE_OUTER_GEMM
#define USE_SELECTED_OUTER_GEMM
#endif

struct KernelTimes {
    float transpose_ms;
    float qk_gemm_ms;
    float softmax_ms;
    float cv_gemm_ms;
};

void reset_kernel_times(KernelTimes* times) {
    if (!times) return;
    times->transpose_ms = 0.0f;
    times->qk_gemm_ms = 0.0f;
    times->softmax_ms = 0.0f;
    times->cv_gemm_ms = 0.0f;
}

void record_kernel_time(cudaEvent_t start, cudaEvent_t stop, float* dst) {
    if (!dst) return;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    *dst += elapsed_ms;
}

// Transpose K: input B is M x N, output A is N x M.
__global__ void transpose_kernel(float* A, const float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // B's row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // B's col
    if (row < M && col < N) {
        A[col * M + row] = B[row * N + col];
    }
}

// Plain GEMM kernel. Keep the original accumulation logic: C += A * B.
__global__ void gemm_kernel(float* C,
                            const float* A,
                            const float* B,
                            int M,
                            int N,
                            int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        for (int i = 0; i < K; ++i) {
            C[row * N + col] += A[row * K + i] * B[i * N + col];
        }
    }
}

__global__ void gemm_kernel_opt(float* C,
                              const float* A,
                              const float* B,
                              int M,
                              int N,
                              int K) {
    __shared__ float As[OP_BM][OP_BK + PAD];
    __shared__ float Bs[OP_BK][OP_BN + PAD];

    int tx = threadIdx.x;  //当前线程在y方向的第几个tile
    int ty = threadIdx.y;

    int row = blockIdx.y * OP_BM + ty * OP_TM;  //当前线程在全局矩阵中的行号
    int col = blockIdx.x * OP_BN + tx * OP_TN;

    int tid = ty * blockDim.x + tx;  //当前线程在block中的编号
    int num_threads = OP_THREADS_M * OP_THREADS_N;  //每个block中线程总数

    float regC[OP_TM][OP_TN];
    #pragma unroll
    for(int i = 0; i < OP_TM; ++i) {
        #pragma unroll
        for(int j = 0; j < OP_TN; ++j) {
            regC[i][j] = 0.0f;
        }
    }

    //外积循环
    for(int kk = 0; kk < K; kk += OP_BK)
    {
        #pragma unroll
        for(int idx = tid; idx < OP_BM * OP_BK; idx += num_threads)
        {
            int r = idx / OP_BK;
            int c = idx % OP_BK;
            int gr = blockIdx.y * OP_BM + r;
            int gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        #pragma unroll
		for (int idx = tid; idx < OP_BK * OP_BN; idx += num_threads)
		{
			int r = idx / OP_BN;
			int c = idx % OP_BN;
			int gr = kk + r;                   // 全局行号
			int gc = blockIdx.x * OP_BN + c;   // 全局列号
			Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
		}

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < OP_BK; ++k)
        {
            float regA[OP_TM];
            #pragma unroll
            for(int i = 0; i < OP_TM; ++i)
            {
                regA[i] = As[ty * OP_TM + i][k];
            }
            float regB[OP_TN];
            #pragma unroll
            for(int j = 0; j < OP_TN; ++j)
            {
                regB[j] = Bs[k][tx * OP_TN + j];
            }

            #pragma unroll
			for (int i = 0; i < OP_TM; ++i)
				#pragma unroll
				for (int j = 0; j < OP_TN; ++j)
					regC[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    #pragma unroll
	for (int i = 0; i < OP_TM; ++i)
		#pragma unroll
		for (int j = 0; j < OP_TN; ++j)
		{
			int r = row + i;
			int c = col + j;
			if (r < M && c < N)
				C[r * N + c] = regC[i][j];
		}
}

// Row-wise softmax. One thread handles one row.
__global__ void softmax_kernel(float* A, int d, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float temp = sqrtf((float)d);
    float* row_ptr = A + row * N;

    for (int j = 0; j < N; ++j) {
        row_ptr[j] /= temp;
    }

    float max_val = -INFINITY;
    for (int j = 0; j < N; ++j) {
        if (row_ptr[j] > max_val) max_val = row_ptr[j];
    }

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        float e = expf(row_ptr[j] - max_val);
        row_ptr[j] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; ++j) {
        row_ptr[j] *= inv_sum;
    }
}

void solve_impl(const float* Q,
                const float* K,
                const float* V,
                float* output,
                int M,
                int N,
                int d,
                KernelTimes* times,
                cudaEvent_t start,
                cudaEvent_t stop) {
    float* K_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&K_trans, (size_t)N * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(K_trans, 0, (size_t)N * d * sizeof(float)));

    dim3 blocksize(16, 16);
    dim3 gridsize_trans((d + 16 - 1) / 16, (N + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    transpose_kernel<<<gridsize_trans, blocksize>>>(K_trans, K, N, d);
    CUDA_CHECK(cudaGetLastError());
    if (times) record_kernel_time(start, stop, &times->transpose_ms);

    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C, 0, (size_t)M * N * sizeof(float)));

#ifdef USE_SELECTED_OUTER_GEMM
    dim3 blocksize_opt(OP_THREADS_N, OP_THREADS_M);
    dim3 gridsize_QK((N + OP_BN - 1) / OP_BN, (M + OP_TM - 1) / OP_BM);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel_opt<<<gridsize_QK, blocksize_opt>>>(C, Q, K_trans, M, N, d);
    CUDA_CHECK(cudaGetLastError());
#else
    dim3 gridsize_QK((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel<<<gridsize_QK, blocksize>>>(C, Q, K_trans, M, N, d);
    CUDA_CHECK(cudaGetLastError());
#endif
    if (times) record_kernel_time(start, stop, &times->qk_gemm_ms);

    gridsize_QK = dim3((M + 255) / 256, 1);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel<<<gridsize_QK, 256>>>(C, d, M, N);
    CUDA_CHECK(cudaGetLastError());
    if (times) record_kernel_time(start, stop, &times->softmax_ms);

#ifdef USE_SELECTED_OUTER_GEMM
    dim3 gridsize_CV((d + OP_BN - 1) / OP_BN, (M + OP_TM - 1) / OP_BM);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel_opt<<<gridsize_CV, blocksize_opt>>>(output, C, V, M, d, N);
    CUDA_CHECK(cudaGetLastError());
#else
    dim3 gridsize_CV((d + 16 - 1) / 16, (M + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel<<<gridsize_CV, blocksize>>>(output, C, V, M, d, N);
    CUDA_CHECK(cudaGetLastError());
#endif
    if (times) record_kernel_time(start, stop, &times->cv_gemm_ms);

    CUDA_CHECK(cudaFree(K_trans));
    CUDA_CHECK(cudaFree(C));
}

extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d) {
    solve_impl(Q, K, V, output, M, N, d, nullptr, nullptr, nullptr);
}

void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
}

void cpu_reference(const float* Q,
                   const float* K,
                   const float* V,
                   float* output,
                   int M,
                   int N,
                   int d) {
    float* scores = (float*)malloc((size_t)M * N * sizeof(float));
    if (!scores) {
        fprintf(stderr, "CPU reference malloc failed\n");
        exit(EXIT_FAILURE);
    }

    float scale = 1.0f / sqrtf((float)d);

    for (int row = 0; row < M; ++row) {
        float max_val = -INFINITY;

        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += Q[row * d + k] * K[col * d + k];
            }
            scores[row * N + col] = sum * scale;
            if (scores[row * N + col] > max_val) {
                max_val = scores[row * N + col];
            }
        }

        float exp_sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            scores[row * N + col] = expf(scores[row * N + col] - max_val);
            exp_sum += scores[row * N + col];
        }

        for (int col = 0; col < N; ++col) {
            scores[row * N + col] /= exp_sum;
        }
    }

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < d; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += scores[row * N + k] * V[k * d + col];
            }
            output[row * d + col] = sum;
        }
    }

    free(scores);
}

float max_abs_diff(const float* ref, const float* test, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

float benchmark_solve(const float* d_Q,
                      const float* d_K,
                      const float* d_V,
                      float* d_output,
                      int M,
                      int N,
                      int d,
                      cudaEvent_t start,
                      cudaEvent_t stop,
                      KernelTimes* per_kernel_times) {
    size_t output_bytes = (size_t)M * d * sizeof(float);
    reset_kernel_times(per_kernel_times);

    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    if (per_kernel_times) {
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
        solve(d_Q, d_K, d_V, d_output, M, N, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
        CUDA_CHECK(cudaEventRecord(start));
        solve_impl(d_Q, d_K, d_V, d_output, M, N, d,
                   per_kernel_times, kernel_start, kernel_stop);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }

    if (per_kernel_times) {
        per_kernel_times->transpose_ms /= (float)BENCH_ITERS;
        per_kernel_times->qk_gemm_ms /= (float)BENCH_ITERS;
        per_kernel_times->softmax_ms /= (float)BENCH_ITERS;
        per_kernel_times->cv_gemm_ms /= (float)BENCH_ITERS;
        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
    }

    return total_ms / (float)BENCH_ITERS;
}

int main(int argc, char** argv) {
    int M = 128;
    int N = 128;
    int d = 64;

    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        d = atoi(argv[3]);
    }

    size_t bytes_Q = (size_t)M * d * sizeof(float);
    size_t bytes_K = (size_t)N * d * sizeof(float);
    size_t bytes_V = (size_t)N * d * sizeof(float);
    size_t bytes_output = (size_t)M * d * sizeof(float);

    float* h_Q = (float*)malloc(bytes_Q);
    float* h_K = (float*)malloc(bytes_K);
    float* h_V = (float*)malloc(bytes_V);
    float* h_output = (float*)malloc(bytes_output);
    float* h_ref = (float*)malloc(bytes_output);

    if (!h_Q || !h_K || !h_V || !h_output || !h_ref) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    srand(0);
    init_matrix(h_Q, M * d);
    init_matrix(h_K, N * d);
    init_matrix(h_V, N * d);

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Q, bytes_Q));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_K));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_V));
    CUDA_CHECK(cudaMalloc(&d_output, bytes_output));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes_Q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes_K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes_V, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    KernelTimes per_kernel_times;
    float ms = benchmark_solve(d_Q, d_K, d_V, d_output, M, N, d,
                               start, stop, &per_kernel_times);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost));

    cpu_reference(h_Q, h_K, h_V, h_ref, M, N, d);
    float diff = max_abs_diff(h_ref, h_output, M * d);

    printf("Softmax Attention Example\n");
    printf("  Shape: M=%d, N=%d, d=%d\n", M, N, d);
    printf("  Flow: transpose(K) -> Q*K^T -> softmax -> Prob*V\n");
#ifdef USE_SELECTED_OUTER_GEMM
    printf("  GEMM kernel: outer-product tiled GEMM (USE_OUTER_GEMM=1)\n");
#else
    printf("  GEMM kernel: naive GEMM (USE_OUTER_GEMM=0)\n");
#endif
    printf("  Timing: %d warmup runs, then average of %d timed runs\n",
           WARMUP_ITERS,
           BENCH_ITERS);
    printf("  Average time: %.4f ms\n", ms);
    printf("  Kernel average times:\n");
    printf("    transpose(K): %.4f ms\n", per_kernel_times.transpose_ms);
    printf("    Q*K^T GEMM:   %.4f ms\n", per_kernel_times.qk_gemm_ms);
    printf("    softmax:      %.4f ms\n", per_kernel_times.softmax_ms);
    printf("    Prob*V GEMM:  %.4f ms\n", per_kernel_times.cv_gemm_ms);
    printf("  Max abs diff vs CPU reference: %.8f\n", diff);
    printf("  Result: %s\n", diff < 1e-3f ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);
    free(h_ref);

    return 0;
}
