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
	
	// 每个计算线程负责的TM*TN输出块的起始坐标
	int base_row = row - (ty % TM);  //对齐到TM的倍数
	int base_col = col - (tx % TN);  //对齐到TN的倍数
	
	//寄存器累加器
	float regC[TM][TN] = {{0.0f}};
	
	//主循环
	for(int kk = 0; kk < K; kk += BK)
	{
		// 加载A tile
		if(row < M && kk + tx < K)
		{
			As[ty][tx] = A[row * K + kk + tx]; //加载A的row行，k+tx列，把第k到k+BK列加载到共享内存里，算完后在加载下一个BK块的列
		}
		else
		{
			As[ty][tx] = 0.0f;
		}
		
		// 加载B tile
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
				//加载A的片段：TM个元素，来自As[base_row_local + i][k]
				float regA[TM];
				for(int i = 0; i < TM; ++i)
				{
					int local_row = (base_row - by * BM) + i;   //在As中的行索引
					regA[i] = As[local_row][k];
				}
				
				//加载B的片段：TN个元素，来自Bs[k][base_col_local + j]
				float regB[TN];
				for(int j = 0; j < TN; ++j)
				{
					int local_col = (base_col - bx * BN) + j; //在Bs中的列索引
					regB[j] = Bs[k][local_col];
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
	
	//写回结果
	if(is_compute_thread)
	{
		for(int i = 0; i < TM; ++i)
		{
			for(int j = 0; j < TN; ++j)
			{
				int r = base_row + i;
				int c = base_col + j;
				if(r < M && c < N)
				{
					C[r * N + c] = regC[i][j];
				}
			}
		}
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
	
	float *h_A, *h_B, *h_C_ref, *h_C_inner, *h_C_outer, * h_C_naive;
	float *d_A, *d_B, *d_C_inner, *d_C_outer, * d_C_naive;
	
	//分配host内存
	h_A = new float[size_A];
	h_B = new float[size_B];
	h_C_ref = new float[size_C];
	h_C_inner = new float[size_C];
	h_C_outer = new float[size_C];
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
	
	//创建计时事件
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//------------朴素kernel-----------------
	cudaEventRecord(start);
	gemm_naive_product<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, M, N, K);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms_naive;
	cudaEventElapsedTime(&ms_naive, start, stop);
	
	//------------内积kernel-----------------
	cudaEventRecord(start);
	gemm_inner_product<<<grid_inner, block_inner>>>(d_A, d_B, d_C_inner, M, N, K);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms_inner;
	cudaEventElapsedTime(&ms_inner, start, stop);
	
	//-------------外积kernel--------------
	cudaEventRecord(start);
	gemm_outer_product<<<grid_outer, block_outer>>>(d_A, d_B, d_C_outer, M, N, K);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms_outer;
    cudaEventElapsedTime(&ms_outer, start, stop);
	
	cudaMemcpy(h_C_inner, d_C_inner, size_C, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C_outer, d_C_outer, size_C, cudaMemcpyDeviceToHost);
	
	 // 验证结果
    bool pass_inner = verify(h_C_ref, h_C_inner, M, N);
    bool pass_outer = verify(h_C_ref, h_C_outer, M, N);

    printf("========================================\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tile sizes: BM=%d, BN=%d, BK=%d\n", BM, BN, BK);
    printf("Outer product thread tile: TM=%d, TN=%d\n", TM, TN);
 printf("----------------------------------------\n");
    printf("Naive product kernel: %s, time = %.3f ms\n",
           pass_inner ? "PASS" : "FAIL", ms_naive);
    printf("----------------------------------------\n");
    printf("Inner product kernel: %s, time = %.3f ms\n",
           pass_inner ? "PASS" : "FAIL", ms_inner);
    printf("Outer product kernel: %s, time = %.3f ms\n",
           pass_outer ? "PASS" : "FAIL", ms_outer);
    printf("Speedup (inner/outer): %.2fx\n", ms_inner / ms_outer);
    printf("========================================\n");

	//清理
	delete[] h_A;
	delete[] h_B;
	delete[] h_C_ref;
	delete[] h_C_inner;
	delete[] h_C_outer;
	delete[] h_C_naive;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C_inner);
	cudaFree(d_C_outer);
	cudaFree(d_C_naive);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
	
}

