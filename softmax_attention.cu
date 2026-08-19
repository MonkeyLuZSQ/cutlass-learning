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

#define WARMUP_ITERS 5
#define BENCH_ITERS 20

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

extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d) {
    float* K_trans = nullptr;
    CUDA_CHECK(cudaMalloc(&K_trans, (size_t)N * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(K_trans, 0, (size_t)N * d * sizeof(float)));

    dim3 blocksize(16, 16);
    dim3 gridsize_trans((d + 16 - 1) / 16, (N + 16 - 1) / 16);
    transpose_kernel<<<gridsize_trans, blocksize>>>(K_trans, K, N, d);
    CUDA_CHECK(cudaGetLastError());

    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(C, 0, (size_t)M * N * sizeof(float)));

    dim3 gridsize_QK((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    gemm_kernel<<<gridsize_QK, blocksize>>>(C, Q, K_trans, M, N, d);
    CUDA_CHECK(cudaGetLastError());

    gridsize_QK = dim3((M + 255) / 256, 1);
    softmax_kernel<<<gridsize_QK, 256>>>(C, d, M, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 gridsize_CV((d + 16 - 1) / 16, (M + 16 - 1) / 16);
    gemm_kernel<<<gridsize_CV, blocksize>>>(output, C, V, M, d, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(K_trans));
    CUDA_CHECK(cudaFree(C));
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
                      cudaEvent_t stop) {
    size_t output_bytes = (size_t)M * d * sizeof(float);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
        solve(d_Q, d_K, d_V, d_output, M, N, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));
        CUDA_CHECK(cudaEventRecord(start));
        solve(d_Q, d_K, d_V, d_output, M, N, d);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
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

    float ms = benchmark_solve(d_Q, d_K, d_V, d_output, M, N, d, start, stop);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost));

    cpu_reference(h_Q, h_K, h_V, h_ref, M, N, d);
    float diff = max_abs_diff(h_ref, h_output, M * d);

    printf("Softmax Attention Example\n");
    printf("  Shape: M=%d, N=%d, d=%d\n", M, N, d);
    printf("  Flow: transpose(K) -> Q*K^T -> softmax -> Prob*V\n");
    printf("  Timing: %d warmup runs, then average of %d timed runs\n",
           WARMUP_ITERS,
           BENCH_ITERS);
    printf("  Average time: %.4f ms\n", ms);
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
