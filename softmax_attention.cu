#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

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

#define WARMUP_ITERS 2
#define BENCH_ITERS 5

#ifndef USE_OUTER_GEMM
#define USE_OUTER_GEMM 1
#endif

#ifndef USE_QKT_SOFTMAX_FUSED
#define USE_QKT_SOFTMAX_FUSED 0
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

// QK^T outer-product GEMM.
// A is Q[M, K], B is original K[N, K], C is Scores[M, N].
// This kernel embeds the logical transpose of K into the B tile load:
//
//   C[row, col] += Q[row, k] * K[col, k]
//
// It removes the external transpose_kernel + K_trans temporary buffer from the
// outer-product GEMM path. It is not used for Prob * V, because that second GEMM
// uses V[N, d] as a normal row-major B matrix.
__global__ void qkt_gemm_kernel_opt(float* C,
                                    const float* A,
                                    const float* B,
                                    int M,
                                    int N,
                                    int K) {
    __shared__ float As[OP_BM][OP_BK + PAD];
    __shared__ float Bs[OP_BK][OP_BN + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * OP_BM + ty * OP_TM;
    int col = blockIdx.x * OP_BN + tx * OP_TN;

    int tid = ty * blockDim.x + tx;
    int num_threads = OP_THREADS_M * OP_THREADS_N;

    float regC[OP_TM][OP_TN];
    #pragma unroll
    for(int i = 0; i < OP_TM; ++i) {
        #pragma unroll
        for(int j = 0; j < OP_TN; ++j) {
            regC[i][j] = 0.0f;
        }
    }

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
            int gr = kk + r;
            int gc = blockIdx.x * OP_BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gc * K + gr] : 0.0f;
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
// opt: one block one row
__global__ void softmax_kernel(float* A, int d, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float temp = sqrtf((float)d);
    float* row_ptr = A + row * N;

    // each thread has its own local max
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = row_ptr[j] / temp;
        row_ptr[j] = val;
        local_max = fmaxf(local_max, val);
    }

    //block reduce
    __shared__ float smem[256];
    smem[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = smem[0];

    //exp + sum
    float local_sum = 0.0f;
    for(int j = tid; j < N; j += blockDim.x)
    {
        float e = expf(row_ptr[j] - max_val);
        row_ptr[j] = e;
        local_sum += e;
    }

    //block reduce
    smem[tid] = local_sum;
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float sum = smem[0];

    float inv_sum = 1.0f / sum;

    for (int j = tid; j < N; j += blockDim.x) {
        row_ptr[j] *= inv_sum;
    }

}

#if USE_QKT_SOFTMAX_FUSED
// Teaching TODO:
// Implement this kernel yourself. The solver below expects this exact interface.
//
// Required behavior:
//   Q[M, d] and K[N, d] are row-major.
//   C[M, N] should be written as softmax(Q * K^T / sqrt(d)).
//
// Suggested launch model:
//   one CUDA block handles one row of Q.
//   blockDim.x threads compute all N scores for that row, reduce max/sum inside
//   the block, and write the normalized probability row to C.
//
// Dynamic shared memory provided by solve_qkt_softmax_fused:
//   (N + blockDim.x) * sizeof(float)
// You can use the first N floats for scores and the rest for block reduction.
__global__ void qkt_softmax_fused_kernel(float* C,
                                         const float* Q, //[M, d]
                                         const float* K, //[N, d]
                                         int M,
                                         int N,
                                         int d)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduce = shared + N;

    if(row >= M)
    {
        return;
    }

    float scale = rsqrtf((float)d);

    //1. 计算当前Q行和所有K行的点积
    for(int col = tid; col < N; col += blockDim.x)
    {
        float acc = 0.0f;
        for(int k = 0; k < d; ++k)
        {
            acc += Q[row * d + k] * K[col * d + k];
        }
        scores[col] = acc * scale;
    }
    __syncthreads();

    // 2. 求softmax的行最大值
    float local_max = -FLT_MAX;
    for(int col = tid; col < N; col += blockDim.x)
    {
        local_max = fmaxf(local_max, scores[col]);
    }
    reduce[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = reduce[0];

    //3 计算exp(score - max)并求和
    float local_sum = 0.0f;
    for(int col = tid; col < N; col += blockDim.x)
    {
        float value = expf(scores[col] - row_max);
        scores[col] = value;
        local_sum += value;
    }

    reduce[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = reduce[0];

    // 4. 写回 softmax 结果
    for (int col = tid; col < N; col += blockDim.x) {
        C[row * N + col] = scores[col] / row_sum;
    }

}
#endif

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
#ifndef USE_SELECTED_OUTER_GEMM
    float* K_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&K_trans, (size_t)N * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(K_trans, 0, (size_t)N * d * sizeof(float)));

    dim3 blocksize(16, 16);
    dim3 gridsize_trans((d + 16 - 1) / 16, (N + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    transpose_kernel<<<gridsize_trans, blocksize>>>(K_trans, K, N, d);
    CUDA_CHECK(cudaGetLastError());
    if (times) record_kernel_time(start, stop, &times->transpose_ms);
#else
    dim3 blocksize(16, 16);
    if (times) times->transpose_ms = 0.0f;
#endif

    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C, 0, (size_t)M * N * sizeof(float)));

#ifdef USE_SELECTED_OUTER_GEMM
    dim3 blocksize_opt(OP_THREADS_N, OP_THREADS_M);
    dim3 gridsize_QK((N + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    qkt_gemm_kernel_opt<<<gridsize_QK, blocksize_opt>>>(C, Q, K, M, N, d);
    CUDA_CHECK(cudaGetLastError());
#else
    dim3 gridsize_QK((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel<<<gridsize_QK, blocksize>>>(C, Q, K_trans, M, N, d);
    CUDA_CHECK(cudaGetLastError());
#endif
    if (times) record_kernel_time(start, stop, &times->qk_gemm_ms);

    dim3 grid_softmax(M);
    dim3 block_softmax(256);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel<<<grid_softmax, block_softmax>>>(C, d, M, N);
    CUDA_CHECK(cudaGetLastError());
    if (times) record_kernel_time(start, stop, &times->softmax_ms);

#ifdef USE_SELECTED_OUTER_GEMM
    dim3 gridsize_CV((d + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);
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

#ifndef USE_SELECTED_OUTER_GEMM
    CUDA_CHECK(cudaFree(K_trans));
#endif
    CUDA_CHECK(cudaFree(C));
}

#if USE_QKT_SOFTMAX_FUSED
void solve_qkt_softmax_fused(const float* Q,
                             const float* K,
                             const float* V,
                             float* output,
                             int M,
                             int N,
                             int d,
                             KernelTimes* times,
                             cudaEvent_t start,
                             cudaEvent_t stop) {
    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * sizeof(float)));

    dim3 grid_qkt_softmax(M);
    dim3 block_qkt_softmax(256);
    size_t shared_bytes =
        ((size_t)N + (size_t)block_qkt_softmax.x) * sizeof(float);

    int device = 0;
    int max_shared_bytes = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_bytes,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      device));
    if (shared_bytes > (size_t)max_shared_bytes) {
        fprintf(stderr,
                "qkt_softmax_fused_kernel needs %zu bytes shared memory, "
                "but this device allows %d bytes per block. Reduce N or "
                "rewrite the kernel with tiled/online softmax.\n",
                shared_bytes,
                max_shared_bytes);
        exit(EXIT_FAILURE);
    }

    // Fused stage:
    //   QK^T GEMM + row-wise softmax -> C[M, N]
    //
    // Because QK^T and softmax are now in the same CUDA kernel, their time
    // cannot be measured separately. For reporting, qk_gemm_ms records the
    // fused QK^T+softmax time, and softmax_ms is kept at 0.
    if (times) CUDA_CHECK(cudaEventRecord(start));
    qkt_softmax_fused_kernel<<<grid_qkt_softmax,
                               block_qkt_softmax,
                               shared_bytes>>>(C, Q, K, M, N, d);
    CUDA_CHECK(cudaGetLastError());
    if (times) {
        record_kernel_time(start, stop, &times->qk_gemm_ms);
        times->softmax_ms += 0.0f;
        times->transpose_ms += 0.0f;
    }

#ifdef USE_SELECTED_OUTER_GEMM
    dim3 blocksize_opt(OP_THREADS_N, OP_THREADS_M);
    dim3 gridsize_CV((d + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel_opt<<<gridsize_CV, blocksize_opt>>>(output, C, V, M, d, N);
    CUDA_CHECK(cudaGetLastError());
#else
    dim3 blocksize(16, 16);
    dim3 gridsize_CV((d + 16 - 1) / 16, (M + 16 - 1) / 16);
    if (times) CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel<<<gridsize_CV, blocksize>>>(output, C, V, M, d, N);
    CUDA_CHECK(cudaGetLastError());
#endif
    if (times) record_kernel_time(start, stop, &times->cv_gemm_ms);

    CUDA_CHECK(cudaFree(C));
}
#endif

extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d) {

#if USE_QKT_SOFTMAX_FUSED
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    solve_qkt_softmax_fused(Q, K, V, output, M, N, d, nullptr, start, stop);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
#else
    solve_impl(Q, K, V, output, M, N, d, nullptr, nullptr, nullptr);
#endif
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
#if USE_QKT_SOFTMAX_FUSED
        solve_qkt_softmax_fused(d_Q, d_K, d_V, d_output, M, N, d,
                                per_kernel_times, kernel_start, kernel_stop);
#else
        solve_impl(d_Q, d_K, d_V, d_output, M, N, d,
                   per_kernel_times, kernel_start, kernel_stop);
#endif
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
#if USE_QKT_SOFTMAX_FUSED
    printf("  Flow: fused Q*K^T + softmax -> Prob*V\n");
    printf("  Fused kernel: qkt_softmax_fused_kernel (student implementation)\n");
#else
    printf("  Flow: transpose(K) -> Q*K^T -> softmax -> Prob*V\n");
#endif
#ifdef USE_SELECTED_OUTER_GEMM
#if USE_QKT_SOFTMAX_FUSED
    printf("  GEMM kernel: fused QK^T+softmax + normal outer GEMM (USE_OUTER_GEMM=1)\n");
    printf("  K transpose: not used; qkt_softmax_fused_kernel reads K[N,d] directly\n");
#else
    printf("  GEMM kernel: QK^T outer GEMM + normal outer GEMM (USE_OUTER_GEMM=1)\n");
    printf("  K transpose: embedded in qkt_gemm_kernel_opt\n");
#endif
#else
#if USE_QKT_SOFTMAX_FUSED
    printf("  GEMM kernel: fused QK^T+softmax + naive ProbV GEMM (USE_OUTER_GEMM=0)\n");
    printf("  K transpose: not used; qkt_softmax_fused_kernel reads K[N,d] directly\n");
#else
    printf("  GEMM kernel: naive GEMM (USE_OUTER_GEMM=0)\n");
    printf("  K transpose: external transpose_kernel\n");
#endif
#endif
    printf("  Timing: %d warmup runs, then average of %d timed runs\n",
           WARMUP_ITERS,
           BENCH_ITERS);
    printf("  Average time: %.4f ms\n", ms);
    printf("  Kernel average times:\n");
#if USE_QKT_SOFTMAX_FUSED
    printf("    transpose(K): not used\n");
    printf("    Q*K^T + softmax: %.4f ms\n", per_kernel_times.qk_gemm_ms);
    printf("    softmax:      fused above\n");
#else
#ifdef USE_SELECTED_OUTER_GEMM
    printf("    transpose(K): embedded, no standalone kernel\n");
#else
    printf("    transpose(K): %.4f ms\n", per_kernel_times.transpose_ms);
#endif
    printf("    Q*K^T GEMM:   %.4f ms\n", per_kernel_times.qk_gemm_ms);
    printf("    softmax:      %.4f ms\n", per_kernel_times.softmax_ms);
#endif
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
