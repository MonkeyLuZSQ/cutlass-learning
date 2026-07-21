#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*C_{M,N} = A_{M,K}\times B_{K,N}*/

// block size /*我理解的cutlass gemm的第一级分块*/ 
#define BM 16
#define BN 16
#define BK 16

// 外积 kernel 中每个线程负责的输出元素数 /*第二级分块*/
#define TM 4
#define TN 4

// ---- 优化版外积 kernel 参数 (对齐 CUTLASS 逻辑) ----
// 第一级分块: CTA tile (类比 CUTLASS ThreadblockShape)
#define OP_BM 64
#define OP_BN 64
#define OP_BK 16
// 第二级分块: Thread tile (类比 CUTLASS WarpShape/InstructionShape)
#define OP_TM 8
#define OP_TN 8
// 线程网格: 每个线程都是计算线程, 无空闲线程
#define OP_THREADS_M (OP_BM / OP_TM)  // 64/8 = 8
#define OP_THREADS_N (OP_BN / OP_TN)  // 64/8 = 8
// shared memory padding 消除 bank conflict
#define SMEM_PAD 1

// Warmup & benchmark 参数
#define WARMUP_ITERS 4
#define BENCH_ITERS 10

//朴素gemm kernel，不使用shared memory，不分块
__global__ void gemm_naive_product(const float* A, const float* B, float* C, int M, int N, int K)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < M && col < N)
	{
		float sum = 0.0f;
		for(int k = 0; k < K; ++k)
		{
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}


//  朴素的内积分块kernel，每个线程负责1个C元素，循环K个维度，每次从shared memory加载A的一行和B的一列
__global__ void gemm_inner_product(const float* A, const float* B, float* C, int M, int N, int K)
{
	__shared__ float As[BM][BK];
	__shared__ float Bs[BK][BN];
	
	int bx = blockIdx.x; //线程块block在grid中的索引
	int by = blockIdx.y;
	int tx = threadIdx.x; //当前线程在其线程块内的索引，列索引
	int ty = threadIdx.y;
	
	int row = by * BM + ty; //当前线程所负责的C元素在全局矩阵中的行号
	int col = bx * BN + tx; 
	
	float sum = 0.0f;
	for (int k = 0; k < K; k += BK) //K维度切成大小为BK的小块，每次只需要将A_{BM,BK}, B_{BK,BN}大小的子矩阵加载到共享内存，避免直接加载A_{BM,K}和B_{K,BN}的数据量过大，共享内存无法一次性加载
	{
		// 加载A tile
		if(row < M && k + tx < K)
		{
			As[ty][tx] = A[row * K + k + tx]; //加载A的row行，k+tx列，把第k到k+BK列加载到共享内存里，算完后在加载下一个BK块的列
		}
		else
		{
			As[ty][tx] = 0.0f;
		}
		
		// 加载B tile
		if(k + ty < K && col < N)
		{
			Bs[ty][tx] = B[(k + ty) * N + col];
		}
		else
		{
			Bs[ty][tx] = 0.0f;
		}
		
		__syncthreads();
		
		//内积计算
		#pragma unroll
		for(int i = 0; i < BK; ++i)
		{
			sum += As[ty][i] * Bs[i][tx];
		}
		
		__syncthreads();
		
	}
	if(row < M && col < N)
	{
		C[row * N + col] = sum;
	}
	
}

//---------------------------------------------------------
//外积分块 kernel，CUTLASS都是进行外积的运算，减少共享内存中A和B的加载次数
//每个线程负责 TM * TN个C元素，从shared memory 加载A的一小列和B的一小行
//所以会有计算线程，其余的线程只负责搬运数据
__global__ void gemm_outer_product(const float* A, const float* B, float* C, int M, int N, int K)
{
	__shared__ float As[BM][BK];
	__shared__ float Bs[BK][BN];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//当前线程负责的C元素在全局矩阵中的位置
	int row = by * BM + ty;
	int col = bx * BN + tx;
	
	//判断当前线程是否属于计算线程组
	//计算线程组: ty必须是TM的倍数，tx必须是TN的倍数，且ty/TM<BM/TM, tx/TN<BN/TN
	bool is_compute_thread = (ty % TM == 0) && (tx % TN == 0) && (ty / TM < BM / TM) && (tx / TN < BN / TN);
	
	//寄存器累加器
	float regC[TM][TN] = {{0.0f}};
	
	//主循环
	for(int kk = 0; kk < K; kk += BK)
	{
		// 加载A tile: A的row行，kk+tx列，把第kk到kk+BK列加载到共享内存
		if(row < M && kk + tx < K)
		{
			//注意这里有bug, 当BK大小不等于BN和BM时, 索引便会出现偏差
			As[ty][tx] = A[row * K + kk + tx];
		}
		else
		{
			As[ty][tx] = 0.0f;
		}
		
		// 加载B tile: B的kk+ty行，col列，把第kk到kk+BK行加载到共享内存
		if(kk + ty < K && col < N)
		{
			Bs[ty][tx] = B[(kk + ty) * N + col];
		}
		else
		{
			Bs[ty][tx] = 0.0f;
		}
		
		__syncthreads();
		
		if(is_compute_thread)
		{
			//在当前的BK块内，循环K维度
			for(int k = 0; k < BK; ++k)
			{
				//加载A片段: 计算线程的ty本身就是TM的倍数,
				//所以它负责的行就是 ty, ty+1, ..., ty+TM-1
				float regA[TM];
				for(int i = 0; i < TM; ++i)
				{
					regA[i] = As[ty + i][k];
				}
				
				//加载B片段: 同理, 负责的列就是 tx, tx+1, ..., tx+TN-1
				float regB[TN];
				for(int j = 0; j < TN; ++j)
				{
					regB[j] = Bs[k][tx + j];
				}
				
				//计算外积：regC[i][j] += regA[i] * regB[j];
				#pragma unroll
				for(int i = 0; i < TM; ++i)
				{
					#pragma unroll
					for(int j = 0; j < TN; ++j)
					{
						regC[i][j] += regA[i] * regB[j];
					}
				}
			}
		}
		__syncthreads();
	}
	
	//写回结果: 计算线程的全局起始行=by*BM+ty, 起始列=bx*BN+tx
	if(is_compute_thread)
	{
		for(int i = 0; i < TM; ++i)
		{
			for(int j = 0; j < TN; ++j)
			{
				int r = by * BM + ty + i;
				int c = bx * BN + tx + j;
				if(r < M && c < N)
				{
					C[r * N + c] = regC[i][j];
				}
			}
		}
	}
	
}

//---------------------------------------------------------
// 优化版外积 kernel —— 对齐 CUTLASS 底层逻辑
//
// 原版 outer 虽然使用了外积思想，但有三个工程缺陷导致它比 naive 还慢:
//   缺陷1: 256个线程中只有16个(6.25%)参与计算，其余全部空转
//   缺陷2: tile 只有 16×16，每个 block 计算量太小，无法隐藏访存延迟
//   缺陷3: As[16][16] 列访问产生 shared memory bank conflict

//   修复1: 紧凑线程映射 —— block 只有 64 个线程，每个都是计算线程
//   修复2: tile 放大到 64×64 —— 用 64 个线程覆盖 4096 个输出
//   修复3: padding —— As[64][17] 消除 bank conflict
//
// 【核心思想: 线程数与 tile 尺寸解耦】
// 内积模式: 1个线程算1个C元素 → 想要大tile就必须多线程 → 撞上1024上限
// 外积模式: 1个线程算TM×TN个C元素 → 少量线程就能覆盖大tile → 不受限
//
//   OP_BM×OP_BN (64×64)  → ThreadblockShape: 一个线程块负责的输出区域
//   OP_TM×OP_TN (8×8)    → WarpShape/InstructionShape: 一个线程负责的输出子块
//   协作加载 stride loop  → TiledCopy: 所有线程均匀搬运数据
//   regA⊗regB 外积累加    → MMA_Atom: 最小计算单元
//---------------------------------------------------------
__global__ void gemm_outer_product_opt(const float* A, const float* B, float* C,
                                       int M, int N, int K)
{
	// ==================== Shared Memory 声明 ====================
	// padding: 第二维多加1个float，使得相邻行的同列元素不再落在同一个bank上
	// 原理: shared memory 有32个bank，每4字节一个bank
	//   无padding: As[i][k] 地址 = i*16+k, 行0和行2的同列地址差32 → 同一bank → 冲突
	//   有padding: As[i][k] 地址 = i*17+k, 行0和行1地址差17 → 不同bank → 无冲突
	__shared__ float As[OP_BM][OP_BK + SMEM_PAD];  // [64][17], 存A的一个64×16子块
	__shared__ float Bs[OP_BK][OP_BN + SMEM_PAD];  // [16][65], 存B的一个16×64子块
	
	// ==================== 线程映射 ====================
	// block 大小 = (8, 8) = 64 个线程
	// tx: 当前线程在N方向上是第几个thread-tile (0..7)
	// ty: 当前线程在M方向上是第几个thread-tile (0..7)
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// 紧凑映射: 每个线程直接对应一个 8×8 输出子块的左上角
	// 不需要 is_compute_thread 判断 —— 所有64个线程都是计算线程
	// 对比原版: 原版256线程中只有 ty%4==0 && tx%4==0 的16个线程能计算
	int base_row = blockIdx.y * OP_BM + ty * OP_TM;  // 我负责的8行的起始全局行号
	int base_col = blockIdx.x * OP_BN + tx * OP_TN;  // 我负责的8列的起始全局列号
	
	// 线性化线程ID: 把2D的(tx,ty)映射到1D的tid，用于后面的协作加载
	int tid = ty * blockDim.x + tx;  // 0..63
	int num_threads = OP_THREADS_M * OP_THREADS_N;  // 64
	
	// ==================== 寄存器累加器 ====================
	// 每个线程在寄存器中持有 8×8 = 64 个输出值
	// 这些值在整个K循环中持续累加，最后一次性写回global memory
	// 寄存器是GPU上最快的存储，访问延迟为0个时钟周期
	float regC[OP_TM][OP_TN];
	#pragma unroll
	for (int i = 0; i < OP_TM; ++i)
		#pragma unroll
		for (int j = 0; j < OP_TN; ++j)
			regC[i][j] = 0.0f;
	
	// ==================== K 维度主循环 ====================
	// 把 K=512 切成 512/16=32 轮，每轮只搬 64×16 + 16×64 = 2048 个float到shared memory
	// 数据流: Global Memory → Shared Memory → Register → FMA计算
	for (int kk = 0; kk < K; kk += OP_BK)
	{
		// ---- 第一步: 协作加载 (Global → Shared) ----
		// 64个线程搬运 64×16=1024 个A元素: 每个线程搬 1024/64=16 个
		// 使用 stride loop: tid, tid+64, tid+128, ... 直到覆盖所有元素
		// 这对应 CUTLASS 的 TiledCopy: 把搬运工作均匀分配给所有线程
		#pragma unroll
		for (int idx = tid; idx < OP_BM * OP_BK; idx += num_threads)
		{
			int r = idx / OP_BK;       // 在tile内的行号 (0..63)
			int c = idx % OP_BK;       // 在tile内的列号 (0..15)
			int gr = blockIdx.y * OP_BM + r;  // 全局行号
			int gc = kk + c;                   // 全局列号
			As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
		}
		
		// 同理搬运 B 的 16×64=1024 个元素
		#pragma unroll
		for (int idx = tid; idx < OP_BK * OP_BN; idx += num_threads)
		{
			int r = idx / OP_BN;       // 在tile内的行号 (0..15)
			int c = idx % OP_BN;       // 在tile内的列号 (0..63)
			int gr = kk + r;                   // 全局行号
			int gc = blockIdx.x * OP_BN + c;   // 全局列号
			Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
		}
		
		// 屏障: 确保所有线程都完成加载后，才开始计算
		// 否则可能读到其他线程还没写入的数据
		__syncthreads();
		
		// ---- 第二步: 外积计算 (Shared → Register → FMA) ----
		// 在当前 16 步的K范围内，逐步累加
		#pragma unroll
		for (int k = 0; k < OP_BK; ++k)
		{
			// 从 shared memory 加载 A 片段到寄存器:
			// 取 As 中我负责的那8行、第k列 → 一个长度为8的列向量
			// 注意: 这里是沿列方向读取 As[ty*8+0][k], As[ty*8+1][k], ...
			// 由于 padding，相邻行地址差17个float，不会撞bank
			float regA[OP_TM];
			#pragma unroll
			for (int i = 0; i < OP_TM; ++i)
				regA[i] = As[ty * OP_TM + i][k];
			
			// 从 shared memory 加载 B 片段到寄存器:
			// 取 Bs 中第k行、我负责的那8列 → 一个长度为8的行向量
			// 沿行方向读取，天然连续，无bank conflict
			float regB[OP_TN];
			#pragma unroll
			for (int j = 0; j < OP_TN; ++j)
				regB[j] = Bs[k][tx * OP_TN + j];
			
			// 外积累加: regA (8×1列向量) ⊗ regB (1×8行向量) → 8×8 矩阵
			// 一次加载 8+8=16 个值，产生 8×8=64 次 FMA
			// 算术强度 = 64 FMA / 16 loads = 4 (内积只有 0.5)
			#pragma unroll
			for (int i = 0; i < OP_TM; ++i)
				#pragma unroll
				for (int j = 0; j < OP_TN; ++j)
					regC[i][j] += regA[i] * regB[j];
		}
		
		// 屏障: 确保所有线程计算完毕后，再进入下一轮加载
		// 防止下一轮的写入覆盖本轮还在读取的数据
		__syncthreads();
	}
	
	// ==================== 写回结果 (Register → Global) ====================
	// 每个线程把自己寄存器中的 8×8 = 64 个结果写回 global memory
	#pragma unroll
	for (int i = 0; i < OP_TM; ++i)
		#pragma unroll
		for (int j = 0; j < OP_TN; ++j)
		{
			int r = base_row + i;
			int c = base_col + j;
			if (r < M && c < N)
				C[r * N + c] = regC[i][j];
		}
}

// ------------------------------------------------------------
// Host 辅助函数
// ------------------------------------------------------------
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = (float)(rand() % 100) / 100.0f;
}

bool verify(const float* ref, const float* test, int M, int N, float eps = 1e-2) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(ref[i] - test[i]) > eps) {
            printf("Mismatch at %d: ref=%f, test=%f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

//主函数调用
int main()
{
	int M = 512;
	int N = 512;
	int K = 512;
	
	size_t size_A = M * K * sizeof(float);
	size_t size_B = K * N * sizeof(float);
	size_t size_C = M * N * sizeof(float);
	
	float *h_A, *h_B, *h_C_ref, *h_C_inner, *h_C_outer, *h_C_outer_opt, *h_C_naive;
	float *d_A, *d_B, *d_C_inner, *d_C_outer, *d_C_outer_opt, *d_C_naive;
	
	//分配host内存
	h_A = new float[size_A];
	h_B = new float[size_B];
	h_C_ref = new float[size_C];
	h_C_inner = new float[size_C];
	h_C_outer = new float[size_C];
	h_C_outer_opt = new float[size_C];
	h_C_naive = new float[size_C];
	
	srand((unsigned)time(NULL));
	init_matrix(h_A, M, K);
	init_matrix(h_B, K, N);
	
	//CPU 参考结果
	for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
	
	//分配device内存
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C_inner, size_C);
	cudaMalloc(&d_C_outer, size_C);
	cudaMalloc(&d_C_outer_opt, size_C);
	cudaMalloc(&d_C_naive, size_C);
	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

	//定义kernel启动参数
    dim3 block_naive(BN, BM);
    dim3 grid_naive((N + BN - 1) / BN, (M + BM - 1) / BM);
	
	dim3 block_inner(BN, BM);
	dim3 grid_inner((N + BN - 1) / BN, (M + BM - 1) / BM);
	
	//定义外积kernel
	dim3 block_outer(BN, BM);
	dim3 grid_outer((N + BN - 1) / BN, (M + BM - 1) / BM);
	
	//定义优化版外积kernel: 紧凑线程映射, block=(8,8)=64线程
	dim3 block_outer_opt(OP_THREADS_N, OP_THREADS_M);  // (8, 8)
	dim3 grid_outer_opt((N + OP_BN - 1) / OP_BN, (M + OP_BM - 1) / OP_BM);  // (8, 8)
	
	//创建计时事件
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//------------朴素kernel-----------------
	// Warmup: 让GPU完成首次运行的初始化开销（缓存预热、频率提升等）
	for (int i = 0; i < WARMUP_ITERS; ++i)
	{
		gemm_naive_product<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, M, N, K);
	}
	cudaDeviceSynchronize();
	
	// Benchmark: 多次运行取最短时间
	float ms_naive = 1e9f;
	for (int i = 0; i < BENCH_ITERS; ++i)
	{
		cudaEventRecord(start);
		gemm_naive_product<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, M, N, K);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float t;
		cudaEventElapsedTime(&t, start, stop);
		if (t < ms_naive) ms_naive = t;
	}
	
	//------------内积kernel-----------------
	for (int i = 0; i < WARMUP_ITERS; ++i)
	{
		gemm_inner_product<<<grid_inner, block_inner>>>(d_A, d_B, d_C_inner, M, N, K);
	}
	cudaDeviceSynchronize();
	
	float ms_inner = 1e9f;
	for (int i = 0; i < BENCH_ITERS; ++i)
	{
		cudaEventRecord(start);
		gemm_inner_product<<<grid_inner, block_inner>>>(d_A, d_B, d_C_inner, M, N, K);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float t;
		cudaEventElapsedTime(&t, start, stop);
		if (t < ms_inner) ms_inner = t;
	}
	
	//-------------外积kernel--------------
	for (int i = 0; i < WARMUP_ITERS; ++i)
	{
		gemm_outer_product<<<grid_outer, block_outer>>>(d_A, d_B, d_C_outer, M, N, K);
	}
	cudaDeviceSynchronize();
	
	float ms_outer = 1e9f;
	for (int i = 0; i < BENCH_ITERS; ++i)
	{
		cudaEventRecord(start);
		gemm_outer_product<<<grid_outer, block_outer>>>(d_A, d_B, d_C_outer, M, N, K);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float t;
		cudaEventElapsedTime(&t, start, stop);
		if (t < ms_outer) ms_outer = t;
	}
	
	//-------------优化版外积kernel--------------
	for (int i = 0; i < WARMUP_ITERS; ++i)
	{
		gemm_outer_product_opt<<<grid_outer_opt, block_outer_opt>>>(d_A, d_B, d_C_outer_opt, M, N, K);
	}
	cudaDeviceSynchronize();
	
	float ms_outer_opt = 1e9f;
	for (int i = 0; i < BENCH_ITERS; ++i)
	{
		cudaEventRecord(start);
		gemm_outer_product_opt<<<grid_outer_opt, block_outer_opt>>>(d_A, d_B, d_C_outer_opt, M, N, K);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float t;
		cudaEventElapsedTime(&t, start, stop);
		if (t < ms_outer_opt) ms_outer_opt = t;
	}
	
	cudaMemcpy(h_C_inner, d_C_inner, size_C, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C_outer, d_C_outer, size_C, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C_outer_opt, d_C_outer_opt, size_C, cudaMemcpyDeviceToHost);
	
	 // 验证结果
    bool pass_inner = verify(h_C_ref, h_C_inner, M, N);
    bool pass_outer = verify(h_C_ref, h_C_outer, M, N);
    bool pass_outer_opt = verify(h_C_ref, h_C_outer_opt, M, N);

    printf("========================================\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tile sizes: BM=%d, BN=%d, BK=%d\n", BM, BN, BK);
    printf("Outer product thread tile: TM=%d, TN=%d\n", TM, TN);
    printf("Outer-opt: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d, threads=%d\n",
           OP_BM, OP_BN, OP_BK, OP_TM, OP_TN, OP_THREADS_M * OP_THREADS_N);
    printf("Warmup iters: %d, Benchmark iters: %d (min time)\n", WARMUP_ITERS, BENCH_ITERS);
    printf("----------------------------------------\n");
    printf("Naive kernel:              time = %.3f ms\n", ms_naive);
    printf("----------------------------------------\n");
    printf("Inner kernel (16x16):      %s, time = %.3f ms\n",
           pass_inner ? "PASS" : "FAIL", ms_inner);
    printf("Outer kernel (16x16):      %s, time = %.3f ms\n",
           pass_outer ? "PASS" : "FAIL", ms_outer);
    printf("Outer-opt kernel (64x64):  %s, time = %.3f ms\n",
           pass_outer_opt ? "PASS" : "FAIL", ms_outer_opt);
    printf("----------------------------------------\n");
    printf("Speedup (outer-opt vs inner):  %.2fx\n", ms_inner / ms_outer_opt);
    printf("Speedup (outer-opt vs outer):  %.2fx\n", ms_outer / ms_outer_opt);
    printf("Speedup (outer-opt vs naive):  %.2fx\n", ms_naive / ms_outer_opt);
    printf("========================================\n");

	//清理
	delete[] h_A;
	delete[] h_B;
	delete[] h_C_ref;
	delete[] h_C_inner;
	delete[] h_C_outer;
	delete[] h_C_outer_opt;
	delete[] h_C_naive;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C_inner);
	cudaFree(d_C_outer);
	cudaFree(d_C_outer_opt);
	cudaFree(d_C_naive);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
	
}

