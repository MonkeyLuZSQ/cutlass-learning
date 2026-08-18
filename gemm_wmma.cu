#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

// ========================= CUTLASS official headers =========================
// These headers come from the official CUTLASS repository.
//
// Important: the #include lines intentionally use logical CUTLASS header names
// such as "cutlass/gemm/device/gemm.h". The physical disk location is supplied
// by nvcc with -I include search paths.
//
// Current learning project:
//   E:\Program Files\cutlass\cutlass-learing
//
// In the current workspace, the official CUTLASS repository root is:
//   E:\Program Files\cutlass\cutlass
//
// Compile from the learning project with:
//   -I..\cutlass\include
//   -I..\cutlass\tools\util\include
//
// Official source files to compare with:
//   include/cutlass/cutlass.h
//       Base CUTLASS definitions, cutlass::Status, and common macros.
//
//   include/cutlass/half.h
//       Defines cutlass::half_t. CUTLASS device::Gemm commonly uses
//       cutlass::half_t instead of CUDA runtime half for template operands.
//
//   include/cutlass/gemm/device/gemm.h
//       Defines cutlass::gemm::device::Gemm, the real official API used below.
//
//   include/cutlass/layout/matrix.h
//       Defines cutlass::layout::RowMajor and ColumnMajor.
//
//   include/cutlass/numeric_types.h
//       Defines helpers such as cutlass::sizeof_bits<T>.
//
// Official examples/tests to compare with:
//   examples/00_basic_gemm/basic_gemm.cu
//       Minimal cutlass::gemm::device::Gemm host call flow.
//
//   examples/07_volta_tensorop_gemm/volta_tensorop_gemm.cu
//   examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu
//       Tensor Core / TensorOp GEMM template parameter style.
//
//   test/unit/gemm/device/gemm_f16t_f16t_f32t_tensor_op_f32_sm75.cu
//       SM75 reference for half RowMajor x half RowMajor -> float RowMajor,
//       TensorOp, f32 accumulator.
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

/*
 * 文件目标:
 *
 *   C[M, N] = A[M, K] * B[K, N]
 *
 * 本文件同时保留三条路径，方便把“自己写 WMMA”和“调用官方 CUTLASS”放在一起看:
 *
 *   1. Hand WMMA:
 *      手写 CUDA kernel，显式使用 nvcuda::wmma::fragment 和 wmma::mma_sync。
 *      这条路径适合观察 Tensor Core 在 kernel 里的最低层使用位置。
 *
 *   2. Official CUTLASS:
 *      真实调用 cutlass::gemm::device::Gemm。你不再自己写 mainloop、
 *      shared memory swizzle、warp-level MMA 和 epilogue，CUTLASS 模板会生成这些代码。
 *
 *   3. FP32 outer product:
 *      不使用 Tensor Core，用作正确性和性能对照。
 *
 * CUTLASS device::Gemm 的标准 host 调用框架:
 *
 *   using Gemm = cutlass::gemm::device::Gemm<...>;   // 1. 声明 kernel 类型
 *   Gemm gemm_op;                                    // 2. 构造 operator 对象
 *   Gemm::Arguments args{...};                       // 3. 构造参数
 *   gemm_op.can_implement(args);                     // 4. 检查 shape/layout/alignment 是否支持
 *   Gemm::get_workspace_size(args);                  // 5. 查询临时 workspace
 *   gemm_op.initialize(args, workspace, stream);     // 6. 初始化 kernel 参数
 *   gemm_op(stream);                                 // 7. launch CUTLASS kernel
 *
 * 上面的框架对应官方:
 *   cutlass/include/cutlass/gemm/device/gemm.h
 *   cutlass/examples/00_basic_gemm/basic_gemm.cu
 */

// ============================================================
// Hand WMMA tile shapes. These names mirror CUTLASS concepts.
// ============================================================
// ThreadblockShape(M, N, K):
//   一个 CUDA thread block/CTA 一次负责 C 的 BM x BN 输出 tile，
//   并且每轮 mainloop 从 K 维搬运 BK 深度的数据。
//   对应 CUTLASS 模板参数:
//     cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>
#define BM 64
#define BN 64
#define BK 16

// InstructionShape for CUDA WMMA API:
//   nvcuda::wmma 的这个 kernel 使用 m16n16k16 fragment。
//   注意官方 CUTLASS 的 SM75 TensorOp 通常使用更贴近 PTX/hardware 的 16x8x8，
//   所以这里和下面 CutlassOfficialGemm 的 InstructionShape 不完全一样。
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// WarpShape:
//   本教学 kernel 让 1 个 warp 计算 32x32 输出 tile。
//   32x32 = 2x2 个 16x16 WMMA accumulator fragment。
//   一个 block 有 2x2 个 warp，所以覆盖 64x64 C tile。
#define WARPS_M 2
#define WARPS_N 2
#define NUM_WARPS (WARPS_M * WARPS_N)
#define THREADS_PER_BLOCK (NUM_WARPS * 32)

// FP32 outer-product comparison kernel shapes.
// 这个 kernel 不对应官方 CUTLASS API，只是你前面学习 tiling/outer product 的对照组。
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

// CUTLASS official API switch:
//
//   USE_CUTLASS_MINIMAL_API = 1
//     Use the short style from official examples/00_basic_gemm/basic_gemm.cu:
//       cutlass::Status status = gemm_op(args);
//
//   USE_CUTLASS_MINIMAL_API = 0
//     Use the explicit engineering/teaching style:
//       can_implement -> get_workspace_size -> initialize -> operator()
//
// Compile examples:
//   nvcc ... gemm_wmma.cu -o gemm_wmma
//   nvcc ... -DUSE_CUTLASS_MINIMAL_API=1 gemm_wmma.cu -o gemm_wmma_minimal
#ifndef USE_CUTLASS_MINIMAL_API
#define USE_CUTLASS_MINIMAL_API 0
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================
// 1. Hand-written WMMA Tensor Core kernel
// ============================================================
__global__ void gemm_wmma_tensor_core(const float* A, const float* B, float* C,
                                      int M, int N, int K,
                                      int lda, int ldb, int ldc) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // CUTLASS calls this part the mainloop: copy A/B tiles, then do tiled MMA.
    for (int kk = 0; kk < K; kk += BK) {
        // Simplified TiledCopy: all 128 threads cooperatively copy A to shared memory.
        for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
            int r = idx / BK;
            int c = idx % BK;
            int gr = blockIdx.y * BM + r;
            int gc = kk + c;
            float x = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
            As[r][c] = __float2half(x);
        }

        // Simplified TiledCopy: all 128 threads cooperatively copy B to shared memory.
        for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
            int r = idx / BN;
            int c = idx % BN;
            int gr = kk + r;
            int gc = blockIdx.x * BN + c;
            float x = (gr < K && gc < N) ? B[gr * ldb + gc] : 0.0f;
            Bs[r][c] = __float2half(x);
        }

        __syncthreads();

        // Simplified TiledMma: one warp owns a 32x32 tile, made of 2x2 WMMA ops.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major>
                    a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major>
                    b_frag;

                int a_row = warp_m * 32 + i * WMMA_M;
                int b_col = warp_n * 32 + j * WMMA_N;

                wmma::load_matrix_sync(a_frag, &As[a_row][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[0][b_col], BN);

                // Tensor Core instruction through CUDA WMMA API.
                wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // CUTLASS calls this part the epilogue: store accumulator fragments to C/D.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
            int out_col = blockIdx.x * BN + warp_n * 32 + j * WMMA_N;
            if (out_row < M && out_col < N) {
                wmma::store_matrix_sync(&C[out_row * ldc + out_col],
                                        c_frag[i][j], ldc,
                                        wmma::mem_row_major);
            }
        }
    }
}

// ============================================================
// 2. CUTLASS-like wrapper around the hand WMMA kernel
// ============================================================
// 这一节不是官方 CUTLASS，只是把手写 WMMA kernel 包装成类似 CUTLASS 的使用框架。
//
// 为什么保留它:
//   新手第一次看 CUTLASS 时，经常分不清:
//     - host 端调用流程: Arguments / can_implement / initialize / run
//     - device 端 kernel 结构: mainloop / tiled copy / tiled mma / epilogue
//
//   这个 wrapper 用很少的代码模拟 CUTLASS host 端框架，方便和后面的官方
//   cutlass::gemm::device::Gemm 调用一一对照。
//
// 对应关系:
//   WmmaGemmArguments
//     类似 CUTLASS Gemm::Arguments。
//
//   WmmaGemmCutlassLike::can_implement()
//     类似 CUTLASS gemm_op.can_implement(arguments)。
//
//   WmmaGemmCutlassLike::initialize()
//     类似 CUTLASS gemm_op.initialize(arguments, workspace, stream)。
//
//   WmmaGemmCutlassLike::run()
//     类似 CUTLASS gemm_op(stream)，但这里实际 launch 的是我们自己的
//     gemm_wmma_tensor_core kernel。
enum class GemmStatus {
    kSuccess,
    kErrorInvalidProblem,
    kErrorNotInitialized,
    kErrorCuda
};

const char* status_name(GemmStatus status) {
    switch (status) {
        case GemmStatus::kSuccess:
            return "success";
        case GemmStatus::kErrorInvalidProblem:
            return "invalid problem";
        case GemmStatus::kErrorNotInitialized:
            return "not initialized";
        case GemmStatus::kErrorCuda:
            return "cuda error";
    }
    return "unknown";
}

struct GemmCoord {
    int m;
    int n;
    int k;
};

struct WmmaGemmArguments {
    GemmCoord problem_size;
    const float* A;
    int lda;
    const float* B;
    int ldb;
    float* C;
    int ldc;
};

class WmmaGemmCutlassLike {
public:
    static GemmStatus can_implement(const WmmaGemmArguments& args) {
        int m = args.problem_size.m;
        int n = args.problem_size.n;
        int k = args.problem_size.k;

        if (!args.A || !args.B || !args.C || m <= 0 || n <= 0 || k <= 0) {
            return GemmStatus::kErrorInvalidProblem;
        }
        if (args.lda < k || args.ldb < n || args.ldc < n) {
            return GemmStatus::kErrorInvalidProblem;
        }

        // Teaching restriction. Real CUTLASS uses predicated iterators for edges.
        if ((m % WMMA_M) != 0 || (n % WMMA_N) != 0 || (k % WMMA_K) != 0) {
            return GemmStatus::kErrorInvalidProblem;
        }

        return GemmStatus::kSuccess;
    }

    GemmStatus initialize(const WmmaGemmArguments& args,
                          void* workspace = nullptr,
                          cudaStream_t stream = nullptr) {
        (void)workspace;
        (void)stream;

        GemmStatus status = can_implement(args);
        if (status != GemmStatus::kSuccess) {
            initialized_ = false;
            return status;
        }

        args_ = args;
        initialized_ = true;
        return GemmStatus::kSuccess;
    }

    GemmStatus run(cudaStream_t stream = nullptr) const {
        if (!initialized_) {
            return GemmStatus::kErrorNotInitialized;
        }

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((args_.problem_size.n + BN - 1) / BN,
                  (args_.problem_size.m + BM - 1) / BM);

        gemm_wmma_tensor_core<<<grid, block, 0, stream>>>(
            args_.A, args_.B, args_.C,
            args_.problem_size.m, args_.problem_size.n, args_.problem_size.k,
            args_.lda, args_.ldb, args_.ldc);

        cudaError_t err = cudaGetLastError();
        return err == cudaSuccess ? GemmStatus::kSuccess : GemmStatus::kErrorCuda;
    }

    GemmStatus operator()(cudaStream_t stream = nullptr) const {
        return run(stream);
    }

    static void print_teaching_summary(const WmmaGemmArguments& args) {
        dim3 grid((args.problem_size.n + BN - 1) / BN,
                  (args.problem_size.m + BM - 1) / BM);

        printf("\nTeaching GEMM configuration\n");
        printf("  ProblemShape:          M=%d, N=%d, K=%d\n",
               args.problem_size.m, args.problem_size.n, args.problem_size.k);
        printf("  Hand ThreadblockShape: %dx%dx%d\n", BM, BN, BK);
        printf("  Hand WarpShape:        32x32x16, %d warps per block\n", NUM_WARPS);
        printf("  Hand WMMA shape:       %dx%dx%d through nvcuda::wmma\n",
               WMMA_M, WMMA_N, WMMA_K);
        printf("  Hand Grid/Block:       grid=(%d,%d), block=%d threads\n",
               grid.x, grid.y, THREADS_PER_BLOCK);
        printf("  Official CUTLASS type: device::Gemm<half,row; half,row; float,row>\n");
        printf("  CUTLASS SM75 op shape: 16x8x8 TensorOp\n");
#if USE_CUTLASS_MINIMAL_API
        printf("  CUTLASS call style:    minimal gemm_op(args)\n");
#else
        printf("  CUTLASS call style:    full can_implement + initialize + gemm_op()\n");
#endif
    }

private:
    WmmaGemmArguments args_{};
    bool initialized_ = false;
};

GemmStatus run_cutlass_like_wmma_gemm(const WmmaGemmArguments& args,
                                      cudaStream_t stream = nullptr) {
    WmmaGemmCutlassLike gemm_op;

    GemmStatus status = WmmaGemmCutlassLike::can_implement(args);
    if (status != GemmStatus::kSuccess) {
        return status;
    }

    status = gemm_op.initialize(args, nullptr, stream);
    if (status != GemmStatus::kSuccess) {
        return status;
    }

    status = gemm_op.run(stream);
    if (status != GemmStatus::kSuccess) {
        return status;
    }

    cudaError_t err = cudaStreamSynchronize(stream);
    return err == cudaSuccess ? GemmStatus::kSuccess : GemmStatus::kErrorCuda;
}

// ============================================================
// 3. Real official CUTLASS GEMM call
// ============================================================
// 这一节是真正使用 CUTLASS 官方库。它不是“模仿 CUTLASS 流程”，而是实例化并调用:
//
//   cutlass::gemm::device::Gemm
//
// 官方定义位置:
//   cutlass/include/cutlass/gemm/device/gemm.h
//
// 本文件选用的计算:
//   A: half, row-major, shape [M, K]
//   B: half, row-major, shape [K, N]
//   D: float, row-major, shape [M, N]
//   accumulator: float
//   operator class: TensorOp, 即使用 Tensor Core
//   target arch: Sm75, 适合 Turing，例如 GTX 1660 SUPER
//
// 注意数据类型:
//   手写 WMMA kernel 为了方便学习，global memory 中 A/B 仍然是 float，
//   在 shared memory 里转成 half。
//
//   官方 CUTLASS TensorOp GEMM 的 A/B 模板类型直接声明为 cutlass::half_t，
//   所以 main() 中会先调用 convert_float_to_cutlass_half，把同一份输入转成 half。
using CutlassElementInputA = cutlass::half_t;
using CutlassElementInputB = cutlass::half_t;
using CutlassElementOutput = float;
using CutlassElementAccumulator = float;
using CutlassElementCompute = float;

// Layout 类型对应官方 cutlass/include/cutlass/layout/matrix.h。
// RowMajor 表示同一行的列方向连续:
//   A[row, col] address = base + row * lda + col
//   B[row, col] address = base + row * ldb + col
//   C[row, col] address = base + row * ldc + col
using CutlassLayoutA = cutlass::layout::RowMajor;
using CutlassLayoutB = cutlass::layout::RowMajor;
using CutlassLayoutC = cutlass::layout::RowMajor;

// CutlassOfficialGemm 是官方 device-level GEMM 的完整类型声明。
//
// 模板参数可对照官方:
//   cutlass/include/cutlass/gemm/device/gemm.h
//   cutlass/test/unit/gemm/device/gemm_f16t_f16t_f32t_tensor_op_f32_sm75.cu
//
// 从上到下解释:
//
//   ElementA, LayoutA:
//     A 矩阵元素类型与存储布局。
//
//   ElementB, LayoutB:
//     B 矩阵元素类型与存储布局。
//
//   ElementC/LayoutC:
//     输出矩阵 D 的元素类型与存储布局。device::Gemm 里参数名常叫 C/D:
//       C 是旧值输入，用于 beta * C
//       D 是最终输出
//     本例 beta=0，所以 C 的旧值不会影响结果，C 和 D 指向同一块 d_C_cutlass。
//
//   ElementAccumulator:
//     Tensor Core 每个 MMA 的累加类型。本例 half x half -> float accumulate。
//
//   cutlass::arch::OpClassTensorOp:
//     告诉 CUTLASS 使用 Tensor Core 路径。如果改成 OpClassSimt，就是普通 CUDA core。
//
//   cutlass::arch::Sm75:
//     生成适合 Turing SM75 的 kernel。你的 GTX 1660 SUPER 属于 SM75。
//
//   GemmShape<128, 128, 32>:
//     ThreadblockShape。一个 CTA 负责 128x128 的 C tile，每轮处理 K=32。
//
//   GemmShape<64, 64, 32>:
//     WarpShape。一个 warp group/warp-level MMA tile 的逻辑形状。
//
//   GemmShape<16, 8, 8>:
//     InstructionShape。SM75 TensorOp 的底层 MMA 指令形状。
//     这和 nvcuda::wmma 的 16x16x16 API 形状不同，但都使用 Tensor Core。
//
//   LinearCombination:
//     CUTLASS epilogue。它做:
//       D = alpha * accumulator + beta * C
//     本例 alpha=1, beta=0，所以 D = accumulator。
//
//   GemmIdentityThreadblockSwizzle:
//     thread block 到 C tile 的默认映射方式。
//
//   NumStages=2:
//     mainloop pipeline stage 数。更高级的 CUTLASS kernel 会用多 stage 隐藏访存延迟。
using CutlassOfficialGemm = cutlass::gemm::device::Gemm<
    CutlassElementInputA,
    CutlassLayoutA,
    CutlassElementInputB,
    CutlassLayoutB,
    CutlassElementOutput,
    CutlassLayoutC,
    CutlassElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        CutlassElementOutput,
        128 / cutlass::sizeof_bits<CutlassElementOutput>::value,
        CutlassElementAccumulator,
        CutlassElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

__global__ void convert_float_to_cutlass_half(const float* src,
                                              cutlass::half_t* dst,
                                              int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = cutlass::half_t(src[idx]);
    }
}

// run_cutlass_official_gemm:
//   这是本文件最核心的“官方 CUTLASS 调用函数”。
//
// 参数说明:
//   A/B:
//     device pointer，类型必须和 CutlassOfficialGemm 的 ElementA/ElementB 对齐，
//     即 cutlass::half_t*。
//
//   C:
//     device pointer，类型必须和 ElementOutput 对齐，即 float*。
//
//   M/N/K:
//     GEMM problem size，含义和数学公式 C[M,N] = A[M,K] * B[K,N] 一致。
//
//   lda/ldb/ldc:
//     leading dimension。因为这里是 RowMajor:
//       lda = A 的逻辑列数 K
//       ldb = B 的逻辑列数 N
//       ldc = C/D 的逻辑列数 N
//
//   stream:
//     CUDA stream。CUTLASS 的 initialize/operator() 都可以接收 stream。
//
// 返回值:
//   这里用自定义 GemmStatus 包一下 cutlass::Status 和 cudaError_t，
//   便于和手写 WMMA wrapper 使用同一种错误处理方式。
GemmStatus run_cutlass_official_gemm(const cutlass::half_t* A,
                                     const cutlass::half_t* B,
                                     float* C,
                                     int M,
                                     int N,
                                     int K,
                                     int lda,
                                     int ldb,
                                     int ldc,
                                     cudaStream_t stream = nullptr) {
    // 1. 构造 CUTLASS operator 对象。
    //
    // 官方 examples/00_basic_gemm/basic_gemm.cu 也采用这个模式:
    //   CutlassGemm gemm_operator;
    //
    // gemm_op 本身很轻，真正的 kernel 类型已经在 CutlassOfficialGemm 的模板参数中确定。
    CutlassOfficialGemm gemm_op;

    // 2. 构造 Arguments。
    //
    // 对应官方 cutlass::gemm::device::Gemm::Arguments。
    // Arguments 是 host 端可构造的参数包，CUTLASS 会把 problem shape、tensor refs、
    // epilogue scalars、split_k_slices 等信息打包给 kernel。
    //
    // 这也是 CUTLASS 学习里最重要的模式之一:
    //   先定义 kernel 类型，再定义 Arguments，再 initialize/run。
    CutlassOfficialGemm::Arguments args{
        cutlass::gemm::GemmCoord(M, N, K),  // problem_size
        {A, lda},                           // tensor A ref: pointer + leading dimension
        {B, ldb},                           // tensor B ref: pointer + leading dimension
        {C, ldc},                           // source C ref for beta * C
        {C, ldc},                           // destination D ref
        {1.0f, 0.0f},                       // epilogue scalars: alpha, beta
        1                                   // split_k_slices, 1 means no split-K
    };

    // 3. can_implement()。
    //
    // CUTLASS 会检查当前 problem 是否满足这个 kernel 的要求，例如:
    //   - layout 是否匹配
    //   - alignment 是否满足 vectorized load/store
    //   - arch/operator/tile shape 是否支持
    //
    // 对应官方示例:
    //   status = gemm_op.can_implement(arguments);
#if USE_CUTLASS_MINIMAL_API
    // Minimal API, same style as official examples/00_basic_gemm/basic_gemm.cu:
    //
    //   cutlass::Status status = gemm_operator(args);
    //
    // This convenience overload performs the required initialization and launch
    // in one call. It is the shortest official CUTLASS GEMM usage.
    (void)stream;
    cutlass::Status status = gemm_op(args);
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (status != cutlass::Status::kSuccess || sync_err != cudaSuccess) {
        return GemmStatus::kErrorCuda;
    }
    return GemmStatus::kSuccess;
#else
    // Full API: can_implement -> get_workspace_size -> initialize -> operator().
    // This expands what the minimal gemm_op(args) call hides.
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return GemmStatus::kErrorInvalidProblem;
    }

    // 4. 查询并分配 workspace。
    //
    // 某些 CUTLASS GEMM 需要额外临时空间，例如 split-K reduction。
    // 本例 split_k_slices=1，通常 workspace_size 为 0，但仍按官方框架写完整。
    //
    // 对应官方示例:
    //   size_t workspace_size = Gemm::get_workspace_size(arguments);
    size_t workspace_size = CutlassOfficialGemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            return GemmStatus::kErrorCuda;
        }
    }

    // 5. initialize()。
    //
    // initialize 会把 Arguments 转换成 kernel launch 所需的 params，
    // 并记录 workspace/stream 等信息。这样 operator() 时只需要触发 launch。
    //
    // 对应官方示例:
    //   status = gemm_op.initialize(arguments, workspace.get());
    status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
        // 6. operator()(stream)。
        //
        // 这一步真正 launch CUTLASS kernel。
        // 官方 device::Gemm 同时支持 gemm_op() 和 gemm_op(stream)。
        status = gemm_op(stream);
    }

    // 7. 同步与清理。
    //
    // benchmark 函数里会用 cudaEvent 计时，所以不会在每次 launch 后强制同步；
    // 这个“一次性调用函数”是教学流程函数，调用结束后同步，方便立即检查错误和结果。
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (workspace) {
        cudaFree(workspace);
    }

    if (status != cutlass::Status::kSuccess || sync_err != cudaSuccess) {
        return GemmStatus::kErrorCuda;
    }

    return GemmStatus::kSuccess;
#endif
}

// ============================================================
// 4. FP32 outer-product comparison kernel
// ============================================================
__global__ void gemm_outer_product_fp32(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    __shared__ float As[OP_BM][OP_BK + SMEM_PAD];
    __shared__ float Bs[OP_BK][OP_BN + SMEM_PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = OP_THREADS_M * OP_THREADS_N;

    int base_row = blockIdx.y * OP_BM + ty * OP_TM;
    int base_col = blockIdx.x * OP_BN + tx * OP_TN;

    float regC[OP_TM][OP_TN];
#pragma unroll
    for (int i = 0; i < OP_TM; ++i) {
#pragma unroll
        for (int j = 0; j < OP_TN; ++j) {
            regC[i][j] = 0.0f;
        }
    }

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

#pragma unroll
        for (int k = 0; k < OP_BK; ++k) {
            float regA[OP_TM];
            float regB[OP_TN];

#pragma unroll
            for (int i = 0; i < OP_TM; ++i) {
                regA[i] = As[ty * OP_TM + i][k];
            }

#pragma unroll
            for (int j = 0; j < OP_TN; ++j) {
                regB[j] = Bs[k][tx * OP_TN + j];
            }

#pragma unroll
            for (int i = 0; i < OP_TM; ++i) {
#pragma unroll
                for (int j = 0; j < OP_TN; ++j) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < OP_TM; ++i) {
#pragma unroll
        for (int j = 0; j < OP_TN; ++j) {
            int r = base_row + i;
            int c = base_col + j;
            if (r < M && c < N) {
                C[r * N + c] = regC[i][j];
            }
        }
    }
}

// ============================================================
// Host helpers
// ============================================================
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

void cpu_reference_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify(const float* ref, const float* test, int M, int N,
            float eps, bool relative = false) {
    int mismatches = 0;
    float max_rel_err = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        float err = relative ? diff / (fabsf(ref[i]) + 1e-8f) : diff;

        if (relative && err > max_rel_err) {
            max_rel_err = err;
        }

        if (err > eps) {
            if (mismatches < 5) {
                printf("  Mismatch at %d: ref=%f, test=%f, %s_err=%f\n",
                       i, ref[i], test[i],
                       relative ? "rel" : "abs", err);
            }
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        printf("  Total mismatches: %d / %d (%.2f%%)\n",
               mismatches, M * N, 100.0f * mismatches / (M * N));
    }
    if (relative) {
        printf("  Max relative error: %.6f\n", max_rel_err);
    }

    return mismatches == 0;
}

float average_time_ms(const float* times) {
    // Benchmark policy:
    //   1. Run WARMUP_ITERS launches first. These launches are not timed.
    //      Warmup lets CUDA finish lazy initialization, cache/JIT setup, and GPU clock ramp-up.
    //
    //   2. Run BENCH_ITERS timed launches.
    //
    //   3. Return arithmetic mean time in milliseconds.
    //      This matches the common "warmup + repeat + average" microbenchmark style.
    float sum = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        sum += times[i];
    }
    return sum / (float)BENCH_ITERS;
}

float benchmark_cutlass_like_wmma(const WmmaGemmArguments& args,
                                  cudaEvent_t start,
                                  cudaEvent_t stop) {
    WmmaGemmCutlassLike gemm_op;
    GemmStatus status = gemm_op.initialize(args);
    if (status != GemmStatus::kSuccess) {
        printf("Hand WMMA initialize failed: %s\n", status_name(status));
        return -1.0f;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        status = gemm_op.run();
        if (status != GemmStatus::kSuccess) {
            printf("Hand WMMA warmup failed: %s\n", status_name(status));
            return -1.0f;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        status = gemm_op.run();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (status != GemmStatus::kSuccess) {
            printf("Hand WMMA benchmark failed: %s\n", status_name(status));
            return -1.0f;
        }
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    return average_time_ms(times);
}

float benchmark_cutlass_official(const cutlass::half_t* d_A,
                                 const cutlass::half_t* d_B,
                                 float* d_C,
                                 int M,
                                 int N,
                                 int K,
                                 cudaEvent_t start,
                                 cudaEvent_t stop) {
    // benchmark 版本和 run_cutlass_official_gemm 的调用流程相同，
    // 但有一个重要区别:
    //
    //   run_cutlass_official_gemm:
    //     为了教学完整性，每次函数调用内部分配 workspace、initialize、launch、sync、free。
    //
    //   benchmark_cutlass_official:
    //     workspace 分配和 initialize 放在计时前，只计时 gemm_op() 的 kernel launch。
    //
    // 真实项目里通常也是:
    //   1. 初始化/选择 kernel
    //   2. 分配 workspace
    //   3. 重复调用 operator() 处理多批数据
    //
    // 这样可以避免把一次性的 setup 开销算进 GEMM kernel 性能。
    CutlassOfficialGemm gemm_op;

    CutlassOfficialGemm::Arguments args{
        cutlass::gemm::GemmCoord(M, N, K),
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1.0f, 0.0f},
        1
    };

#if USE_CUTLASS_MINIMAL_API
    // Minimal benchmark mode:
    //   warmup: gemm_op(args)
    //   timing: gemm_op(args)
    //
    // This intentionally matches the short official basic_gemm.cu style.
    // It may include more per-call setup overhead than the full initialized mode.
    cutlass::Status status;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            printf("Official CUTLASS minimal warmup failed.\n");
            return -1.0f;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        status = gemm_op(args);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (status != cutlass::Status::kSuccess) {
            printf("Official CUTLASS minimal benchmark failed.\n");
            return -1.0f;
        }
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    return average_time_ms(times);
#else
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("Official CUTLASS can_implement failed.\n");
        return -1.0f;
    }

    size_t workspace_size = CutlassOfficialGemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) {
            CUDA_CHECK(cudaFree(workspace));
        }
        printf("Official CUTLASS initialize failed.\n");
        return -1.0f;
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            if (workspace) {
                CUDA_CHECK(cudaFree(workspace));
            }
            printf("Official CUTLASS warmup failed.\n");
            return -1.0f;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        status = gemm_op();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (status != cutlass::Status::kSuccess) {
            if (workspace) {
                CUDA_CHECK(cudaFree(workspace));
            }
            printf("Official CUTLASS benchmark failed.\n");
            return -1.0f;
        }
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }

    return average_time_ms(times);
#endif
}

float benchmark_outer_product(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K,
                              cudaEvent_t start,
                              cudaEvent_t stop) {
    dim3 block(OP_THREADS_N, OP_THREADS_M);
    dim3 grid((N + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        gemm_outer_product_fp32<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_outer_product_fp32<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    return average_time_ms(times);
}

void print_device_info() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s, compute capability sm_%d%d\n",
           prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        printf("Warning: Tensor Core WMMA path requires Volta(sm_70) or newer.\n");
    }
}

int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;

    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    print_device_info();

    size_t sA = (size_t)M * K * sizeof(float);
    size_t sB = (size_t)K * N * sizeof(float);
    size_t sC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(sA);
    float* h_B = (float*)malloc(sB);
    float* h_C_ref = (float*)malloc(sC);
    float* h_C_wmma = (float*)malloc(sC);
    float* h_C_cutlass = (float*)malloc(sC);
    float* h_C_outer = (float*)malloc(sC);

    if (!h_A || !h_B || !h_C_ref || !h_C_wmma || !h_C_cutlass || !h_C_outer) {
        fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_wmma = nullptr;
    float* d_C_cutlass = nullptr;
    float* d_C_outer = nullptr;
    cutlass::half_t* d_A_cutlass = nullptr;
    cutlass::half_t* d_B_cutlass = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, sC));
    CUDA_CHECK(cudaMalloc(&d_C_cutlass, sC));
    CUDA_CHECK(cudaMalloc(&d_C_outer, sC));

    // 官方 CUTLASS TensorOp GEMM 的 A/B 模板类型是 cutlass::half_t，
    // 因此需要单独准备 half 版本输入。
    //
    // 手写 WMMA kernel 则保留 d_A/d_B 为 float，并在 shared memory 中转 half，
    // 这样你可以在 kernel 里明确看到 float -> half -> Tensor Core 的路径。
    CUDA_CHECK(cudaMalloc(&d_A_cutlass, (size_t)M * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_B_cutlass, (size_t)K * N * sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_wmma, 0, sC));
    CUDA_CHECK(cudaMemset(d_C_cutlass, 0, sC));
    CUDA_CHECK(cudaMemset(d_C_outer, 0, sC));

    // 把同一份 float 输入转成 cutlass::half_t。
    // 这样 CPU reference、手写 WMMA、官方 CUTLASS 都使用同一组随机输入，
    // 区别只在于:
    //   - CPU reference 用 float 计算
    //   - hand WMMA / official CUTLASS 用 FP16 输入 + FP32 accumulate
    //
    // 所以后面 verify 对 Tensor Core 路径使用相对误差 5%。
    int convert_threads = 256;
    int convert_a_blocks = (M * K + convert_threads - 1) / convert_threads;
    int convert_b_blocks = (K * N + convert_threads - 1) / convert_threads;
    convert_float_to_cutlass_half<<<convert_a_blocks, convert_threads>>>(
        d_A, d_A_cutlass, M * K);
    convert_float_to_cutlass_half<<<convert_b_blocks, convert_threads>>>(
        d_B, d_B_cutlass, K * N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // 手写 WMMA wrapper 的 Arguments。
    // 它故意模仿 CUTLASS Gemm::Arguments 的结构:
    //   problem size + A ref + B ref + C/D ref + leading dimensions。
    WmmaGemmArguments wmma_args{
        {M, N, K},
        d_A, K,
        d_B, N,
        d_C_wmma, N
    };

    WmmaGemmCutlassLike::print_teaching_summary(wmma_args);

    GemmStatus status = WmmaGemmCutlassLike::can_implement(wmma_args);
    if (status != GemmStatus::kSuccess) {
        printf("\nThis teaching WMMA kernel requires positive M/N/K multiples of 16.\n");
        printf("Current M=%d, N=%d, K=%d, status=%s\n",
               M, N, K, status_name(status));
        return EXIT_FAILURE;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 路径 1: 手写 WMMA。
    // 这个函数不是官方 CUTLASS，只是用 CUTLASS 风格 host flow 包装我们自己的 kernel。
    printf("\nRunning one hand WMMA wrapper call...\n");
    status = run_cutlass_like_wmma_gemm(wmma_args);
    if (status != GemmStatus::kSuccess) {
        printf("run_cutlass_like_wmma_gemm failed: %s\n", status_name(status));
        return EXIT_FAILURE;
    }

    // 路径 2: 官方 CUTLASS。
    // 这里会进入真正的 cutlass::gemm::device::Gemm:
    //   run_cutlass_official_gemm()
    //     -> CutlassOfficialGemm::Arguments
    //     -> gemm_op.can_implement()
    //     -> CutlassOfficialGemm::get_workspace_size()
    //     -> gemm_op.initialize()
    //     -> gemm_op(stream)
    printf("Running one official CUTLASS GEMM call...\n");
    status = run_cutlass_official_gemm(d_A_cutlass, d_B_cutlass, d_C_cutlass,
                                       M, N, K, K, N, N);
    if (status != GemmStatus::kSuccess) {
        printf("run_cutlass_official_gemm failed: %s\n", status_name(status));
        return EXIT_FAILURE;
    }

    // benchmark 阶段:
    //   benchmark_cutlass_like_wmma:
    //     计时手写 WMMA kernel，不包含 CPU reference 和数据转换。
    //
    //   benchmark_cutlass_official:
    //     计时官方 CUTLASS kernel。workspace 和 initialize 在计时前完成，
    //     这样计时更接近真实 GEMM kernel launch 的耗时。
    //
    //   benchmark_outer_product:
    //     计时普通 FP32 CUDA core 路径。
    printf("Running benchmarks...\n");
    float ms_wmma = benchmark_cutlass_like_wmma(wmma_args, start, stop);
    float ms_cutlass = benchmark_cutlass_official(d_A_cutlass, d_B_cutlass,
                                                  d_C_cutlass, M, N, K,
                                                  start, stop);
    float ms_outer = benchmark_outer_product(d_A, d_B, d_C_outer,
                                             M, N, K, start, stop);

    printf("Computing CPU reference...\n");
    cpu_reference_gemm(h_A, h_B, h_C_ref, M, N, K);

    CUDA_CHECK(cudaMemcpy(h_C_wmma, d_C_wmma, sC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cutlass, d_C_cutlass, sC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_outer, d_C_outer, sC, cudaMemcpyDeviceToHost));

    printf("\nVerifying hand WMMA result (FP16 input, FP32 accumulate, rel_eps=5%%)...\n");
    bool pass_wmma = verify(h_C_ref, h_C_wmma, M, N, 0.05f, true);

    printf("Verifying official CUTLASS result (FP16 input, FP32 accumulate, rel_eps=5%%)...\n");
    bool pass_cutlass = verify(h_C_ref, h_C_cutlass, M, N, 0.05f, true);

    printf("Verifying outer product result (FP32, abs_eps=1e-2)...\n");
    bool pass_outer = verify(h_C_ref, h_C_outer, M, N, 1e-2f, false);

    printf("\n");
    printf("================================================================\n");
    printf("  GEMM Benchmark: Hand WMMA vs Official CUTLASS vs FP32 Outer\n");
    printf("  Matrix: %dx%dx%d\n", M, N, K);
    printf("  Timing: %d warmup launches, then average of %d timed launches\n",
           WARMUP_ITERS, BENCH_ITERS);
    printf("================================================================\n");
    printf("  Hand WMMA Tensor Core:   %s   %8.4f ms\n",
           pass_wmma ? "PASS" : "FAIL", ms_wmma);
    printf("  Official CUTLASS GEMM:   %s   %8.4f ms\n",
           pass_cutlass ? "PASS" : "FAIL", ms_cutlass);
    printf("  FP32 outer product:      %s   %8.4f ms\n",
           pass_outer ? "PASS" : "FAIL", ms_outer);
    printf("----------------------------------------------------------------\n");
    if (ms_wmma > 0.0f) {
        printf("  Hand WMMA speedup over FP32 outer: %.2fx\n", ms_outer / ms_wmma);
    }
    if (ms_cutlass > 0.0f) {
        printf("  Official CUTLASS speedup over FP32 outer: %.2fx\n",
               ms_outer / ms_cutlass);
    }
    printf("================================================================\n");

    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_wmma);
    free(h_C_cutlass);
    free(h_C_outer);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaFree(d_C_cutlass));
    CUDA_CHECK(cudaFree(d_C_outer));
    CUDA_CHECK(cudaFree(d_A_cutlass));
    CUDA_CHECK(cudaFree(d_B_cutlass));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
