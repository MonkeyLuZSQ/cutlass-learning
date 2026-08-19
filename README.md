# GEMM Tiling Kernels — 从朴素到分块优化

## 四种 Kernel 设计思想

**Naive (朴素)**: 每线程计算一个 C 元素，直接对全局内存做 K 维点积。无 shared memory，无分块。A 的每行被 N 个线程重复读取，B 的每列被 M 个线程重复读取，全局内存访问量 O(MNK)。

**Inner Product (内积)**: K 维分块 (BK)，每轮将 A[BM,BK] 和 B[BK,BN] 加载到 shared memory，每线程对 As 的一行和 Bs 的一列做长度为 BK 的点积。A/B 各元素被同 block 内 BN/BM 个线程复用，全局内存访问量降至 O(MN·K/BK)。

**Outer Product (外积, 原版)**: 每线程从 As 加载长度为 TM 的列向量、从 Bs 加载长度为 TN 的行向量，做外积 `regC[TM][TN] += regA[TM] ⊗ regB[TN]`。单次加载产生 TM×TN 次 FMA，数据复用率是内积的 TM×TN 倍。CUTLASS 的 warp-level GEMM 本质上采用此模式。但原版实现有工程缺陷（见下文），导致性能反而最差。

**Outer Product Opt (外积优化版)**: 修复原版三个工程缺陷后的正确实现，对齐 CUTLASS 底层逻辑。性能为四者最优。

## NCU Profiling 瓶颈定位 (512×512×512, SM75)

| 指标 | naive | inner | outer |
|------|-------|-------|-------|
| Duration | 814 us | **523 us** | 1230 us |
| Compute SM % | 61.65 | 73.31 | 83.99 |
| Memory BW | 4.47 GB/s | 6.98 GB/s | 23.03 GB/s |
| L1 Hit Rate | **87.44%** | 11.58% | 14.76% |
| IPC | 0.73 | 0.96 | **1.97** |
| Regs/thread | 52 | **39** | 64 |

**Naive**: L1 cache 命中率 87.44% 掩盖了无分块的缺陷，512² 矩阵可被 L1 有效缓存，性能不差。

**Inner (原版最快)**: shared memory 替代 L1 提供低延迟访问，全部 256 线程参与计算，寄存器开销最低 (39)。瓶颈: No Eligible 76%，warp 调度不充分，计算密度 (1 element/thread) 仍有提升空间。

**Outer (原版最慢)**: IPC 和 Compute 指标最优，但 Duration 反而最差。两个根因:

1. **Shared Memory Bank Conflict**: `As[16][16]` 中 `float[16]` 占 64B = 2 个 bank width，同列不同行 (row, row+2) 落入同一 bank → 2-way conflict，有效带宽减半。
2. **线程利用率极低**: `is_compute_thread = (ty%4==0) && (tx%4==0)` 使 256 线程中仅 16 个参与计算 (6.25%)，其余在计算阶段完全空闲，Memory BW 高达 23 GB/s 但有效计算不足。

## Benchmark 结果 (512×512×512, GTX 1660 SUPER)

```
Naive kernel:              time = 0.828 ms
Inner kernel (16x16):      PASS, time = 0.520 ms
Outer kernel (16x16):      PASS, time = 0.835 ms
Outer-opt kernel (64x64):  PASS, time = 0.182 ms

Speedup (outer-opt vs inner):  2.85x
Speedup (outer-opt vs outer):  4.58x
Speedup (outer-opt vs naive):  4.54x
```

---

## Outer-opt 优化详解 (面向初学者)

原版 outer 的**算法思想是正确的**（外积），但**实现有三个致命问题**：

#### 缺陷 1: 线程利用率 6.25%

原版使用 16×16 = 256 个线程的 block，但只有满足 `(ty%4==0) && (tx%4==0)` 的线程才能参与计算。满足条件的只有 4×4 = 16 个线程。

#### 缺陷 2: Tile 太小，计算量不足

原版 tile 只有 16×16 = 256 个输出元素。对于 512×512 的矩阵，需要 32×32 = 1024 个 block。每个 block 的工作量太小，GPU 花大量时间在"启动 block → 加载数据 → 同步 → 计算一点点 → 结束"的开销上，真正的计算占比很低。

#### 缺陷 3: Shared Memory Bank Conflict

Shared memory 由 32 个 bank 组成（可以理解为 32 个并行的存储通道）。当同一个 warp 内的多个线程同时访问同一个 bank 的不同地址时，访问会被串行化（排队），这就是 bank conflict。

原版 `As[16][16]`：元素 `As[i][j]` 的地址是 `i*16 + j`。当外积计算沿列方向读取 `As[0][k], As[1][k], As[2][k], ...` 时：
- `As[0][k]` 地址 = k，bank = k % 32
- `As[2][k]` 地址 = 32 + k，bank = (32+k) % 32 = k % 32

行 0 和行 2 落在同一个 bank！产生 2-way conflict，有效带宽减半。

### 优化

#### 修复 1: 紧凑线程映射 —— 所有线程都是计算线程

```
原版: block = 16×16 = 256 线程, 其中 16 个计算 (6.25%)
优化: block = 8×8 = 64 线程, 全部 64 个计算 (100%)
```

关键思想：**先确定每个线程负责多大的输出子块（8×8），再反推 block 需要多少线程**。

- 每个线程负责 TM×TN = 8×8 = 64 个 C 元素
- block 需要覆盖 64×64 的输出区域
- 所以线程数 = (64/8) × (64/8) = 8×8 = 64

线程映射方式：
```
threadIdx.x = tx (0..7) → 负责第 tx 个列方向的 8 列
threadIdx.y = ty (0..7) → 负责第 ty 个行方向的 8 行

线程 (tx, ty) 负责的输出区域:
  行: [blockIdx.y*64 + ty*8,  blockIdx.y*64 + ty*8 + 7]
  列: [blockIdx.x*64 + tx*8,  blockIdx.x*64 + tx*8 + 7]
```

不需要 `is_compute_thread` 判断，因为每个线程天然就是一个计算单元。

#### 修复 2: Tile 放大到 64×64 —— 线程数与 tile 尺寸解耦

这是外积模式相对于内积模式的**结构性优势**：

```
内积: 1 线程 = 1 个输出 → 想要 64×64 tile 需要 4096 线程 → 超过 CUDA 上限 1024！
外积: 1 线程 = 64 个输出 → 64×64 tile 只需 64 线程 → 远低于上限
```

大 tile 的好处：
- 每个 block 产出 64×64 = 4096 个 C 元素，计算量充足
- 从 global memory 加载一次数据后，可以被更多次计算复用
- 512×512 矩阵只需 8×8 = 64 个 block（原版需要 1024 个），减少调度开销

#### 修复 3: Padding 消除 Bank Conflict

```
原版: __shared__ float As[16][16]   → 行间距 16 个 float
优化: __shared__ float As[64][17]   → 行间距 17 个 float (多加 1 个 padding)
```

原理：shared memory 有 32 个 bank，每个 bank 宽 4 字节（1 个 float）。

- 无 padding：`As[i][k]` 地址 = `i*16 + k`，行 0 和行 2 的同列地址差 32 → 同一 bank
- 有 padding：`As[i][k]` 地址 = `i*17 + k`，任意两行的同列地址差 17 的倍数 → 17 与 32 互素 → 永远不会撞 bank

代价：每个 tile 多占 64×1 = 64 个 float = 256 字节，几乎可以忽略。

### 线程索引：加载与计算的双映射

Outer-opt 的核心设计是**加载和计算使用不同的线程映射**。这正是 CUTLASS 将 `TiledCopy`（搬运）和 `TiledMMA`（计算）解耦的底层逻辑。

**计算阶段：2D 映射 → 每个线程拥有一个 8×8 输出子块**

```
C 矩阵的一个 64×64 block tile:

        tx=0   tx=1   tx=2   tx=3   tx=4   tx=5   tx=6   tx=7
       ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
ty=0   │8×8 │ │8×8 │ │8×8 │ │8×8 │ │8×8 │ │8×8 │ │8×8 │ │8×8 │
       └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
ty=1   │ ...                                           ... │
       └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
  ...
ty=7   │ ...                                           ... │
       └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
       列0-7  列8-15 列16-23 列24-31 列32-39 列40-47 列48-55 列56-63

线程(tx,ty) 的输出子块左上角:
  base_row = blockIdx.y*64 + ty*8    (行方向, 步长8)
  base_col = blockIdx.x*64 + tx*8    (列方向, 步长8)
```

**加载阶段：线性化 → stride loop 均匀搬运**

加载时不关心"谁算哪块"，只关心 64 个线程如何均匀搬完 1024 个元素。将 2D 线程压成 1D：

```
tid = ty * 8 + tx    (0..63)

A tile [64×16] = 1024 个元素, 每线程搬 1024/64 = 16 个:
  for (idx = tid; idx < 1024; idx += 64)
      As[idx/16][idx%16] = A[全局地址]

tid=0 搬: idx = 0, 64, 128, ..., 960  (共16个, 分散在不同行)
tid=1 搬: idx = 1, 65, 129, ..., 961
...
tid=63搬: idx = 63, 127, 191, ..., 1023
```

**两阶段的衔接：搬运者 ≠ 使用者**

```
┌────────────────────────────────────────────────────────────┐
│  加载 (TiledCopy): tid 线性化, stride loop                  │
│  tid=0 搬入 As[0][0], As[4][0], As[8][0] ...              │
│  这些元素将来被 ty=0, ty=4, ty=8 的线程读取                │
├──────────────────── __syncthreads() ───────────────────────┤
│  计算 (TiledMMA): (tx,ty) 2D 映射                          │
│  线程(tx=2,ty=3) 读 As[24..31][k] 和 Bs[k][16..23]        │
│  不管数据是谁搬来的, syncthreads 后全部可见                 │
└────────────────────────────────────────────────────────────┘
```

这种解耦的好处：搬运策略（vectorized load、double buffering）和计算策略（MMA 指令形状）可以独立优化，互不影响。

### 数据流：从 Global Memory 到计算结果

理解 outer-opt 的关键是理解数据在三级存储之间的流动：

```
┌─────────────────────────────────────────────────────────────────┐
│  Global Memory (显存, ~400GB/s带宽, ~400周期延迟)                │
│  存放完整的 A[512×512], B[512×512], C[512×512]                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 协作加载 (64线程各搬16个float)
                           │ 每轮搬 64×16 + 16×64 = 2048 个float
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Shared Memory (片上, ~19TB/s带宽, ~20周期延迟)                  │
│  As[64][17]: A 的一个 64×16 子块                                 │
│  Bs[16][65]: B 的一个 16×64 子块                                 │
│  所有 64 个线程共享, 通过 __syncthreads() 保证可见性              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 每个线程加载自己的片段
                           │ regA[8]: As中我负责的8行第k列
                           │ regB[8]: Bs中第k行我负责的8列
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Register (寄存器, 无限带宽, 0周期延迟)                           │
│  regC[8][8]: 我负责的 64 个输出值, 在整个K循环中持续累加          │
│  regA[8], regB[8]: 当前K步的输入片段                             │
│  外积: regC[i][j] += regA[i] * regB[j], 共 64 次 FMA            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ K循环结束后, 一次性写回
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Global Memory: C[base_row+i][base_col+j] = regC[i][j]          │
└─────────────────────────────────────────────────────────────────┘
```

### 算术强度：外积为什么比内积快？

**算术强度** = 每次从 shared memory 加载一个数据，能产生多少次计算（FMA）。

内积模式：
```
每步K: 加载 1个A + 1个B = 2次加载, 产生 1次FMA
算术强度 = 1/2 = 0.5
```

外积模式 (TM=8, TN=8)：
```
每步K: 加载 8个A + 8个B = 16次加载, 产生 8×8 = 64次FMA
算术强度 = 64/16 = 4
```

外积的算术强度是内积的 **8 倍**。这意味着：同样的 shared memory 带宽下，外积能做 8 倍多的计算。当计算足够密集时，GPU 的计算单元才能被充分利用，而不是大部分时间在等数据。

### 与 CUTLASS 的对应关系

| 本代码中的概念 | CUTLASS 中的概念 | 作用 |
|--------------|-----------------|------|
| OP_BM×OP_BN = 64×64 | `ThreadblockShape` | 一个线程块负责的 C 输出区域 |
| OP_TM×OP_TN = 8×8 | 无直接等价参数，可粗略理解为 thread-level micro tile | 手写 kernel 中一个线程负责的输出子块 |
| OP_THREADS_M×OP_THREADS_N = 8×8 | CTA 内线程布局 | 手写 kernel 的 block 线程组织方式，不等同于 CUTLASS 的 `WarpShape` |
| stride loop 协作加载 | `TiledCopy` + `Copy_Atom` | 所有线程均匀搬运 global→shared |
| regA⊗regB 外积累加 | `MMA_Atom` / `TiledMMA` | 最小计算单元，一次 load 最大 FMA |
| SMEM_PAD = 1 | CuTe `Swizzle` / padding | 消除 shared memory bank conflict |
| K 维度 for 循环 | Mainloop (K-tile iterator) | 流水线化 global→shared 的搬运 |

### 原版 vs 优化版 对比总结

| 对比项 | 原版 Outer | 优化版 Outer-opt |
|--------|-----------|-----------------|
| Block 线程数 | 16×16 = 256 | 8×8 = 64 |
| 计算线程占比 | 16/256 = 6.25% | 64/64 = 100% |
| Block tile 大小 | 16×16 = 256 个输出 | 64×64 = 4096 个输出 |
| Thread tile 大小 | TM=4, TN=4 (16个输出/线程) | TM=8, TN=8 (64个输出/线程) |
| Shared memory | As[16][16], 有 bank conflict | As[64][17], 无 bank conflict |
| 加载方式 | 每线程固定搬1个 (256线程搬256个) | stride loop (64线程各搬16个) |
| Grid 大小 (512²) | 32×32 = 1024 blocks | 8×8 = 64 blocks |
| 实测性能 | 0.835 ms | **0.182 ms (4.58x)** |

## 编译运行

```
nvcc -std=c++17 -arch=sm_75 gemm_tiling.cu -o gemm_tiling.exe
.\gemm_tiling.exe
```

---

## WMMA + CUTLASS 编译运行（WSL）

`gemm_wmma.cu` 同时包含手写 WMMA Tensor Core kernel 和官方 CUTLASS `cutlass::gemm::device::Gemm` 调用，因此编译时需要让 `nvcc` 找到 CUTLASS 官方仓库的头文件。

如果已经在 WSL 中写入环境变量：

```bash
echo 'export CUTLASS_PATH="/mnt/e/Program Files/cutlass/cutlass"' >> ~/.bashrc
source ~/.bashrc
```

之后进入当前项目目录：

```bash
cd "/mnt/e/Program Files/cutlass/cutlass-learing"
```

先确认 CUTLASS 头文件路径有效：

```bash
echo "$CUTLASS_PATH"
ls "$CUTLASS_PATH/include/cutlass/cutlass.h"
```

编译 `gemm_wmma.cu`：

```bash
nvcc -std=c++17 -arch=sm_75 --expt-relaxed-constexpr \
  -I"$CUTLASS_PATH/include" \
  -I"$CUTLASS_PATH/tools/util/include" \
  gemm_wmma.cu -o gemm_wmma
```

默认编译使用完整 CUTLASS 调用流程：

```cpp
can_implement -> get_workspace_size -> initialize -> gemm_op()
```

若需切换到简洁调用风格：

```cpp
gemm_op(args)
```

可以编译时打开宏：

```bash
nvcc -std=c++17 -arch=sm_75 --expt-relaxed-constexpr \
  -I"$CUTLASS_PATH/include" \
  -I"$CUTLASS_PATH/tools/util/include" \
  -DUSE_CUTLASS_MINIMAL_API=1 \
  gemm_wmma.cu -o gemm_wmma_minimal
```

### `cutlass::gemm::device::Gemm` 参数说明

完整 GEMM 模板参数可概括为：

```cpp
cutlass::gemm::device::Gemm<
  ElementA, LayoutA,
  ElementB, LayoutB,
  ElementC, LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  AlignmentA,
  AlignmentB,
  SplitKSerial,
  Operator
>
```

| 参数 | 含义 |
|---|---|
| `ElementA/LayoutA` | A 矩阵元素类型与布局 |
| `ElementB/LayoutB` | B 矩阵元素类型与布局 |
| `ElementC/LayoutC` | C/D 矩阵元素类型与布局 |
| `ElementAccumulator` | MMA 累加器类型 |
| `OperatorClass` | 计算单元类型，`OpClassTensorOp` 表示 Tensor Core |
| `ArchTag` | 目标 GPU 架构，如 `Sm75` |
| `ThreadblockShape` | CTA/block 级 GEMM tile |
| `WarpShape` | warp 级 GEMM tile |
| `InstructionShape` | 底层 MMA 指令 tile |
| `EpilogueOutputOp` | 输出阶段操作，如 `D = alpha * acc + beta * C` |
| `ThreadblockSwizzle` | thread block 到输出 tile 的映射方式 |
| `Stages` | mainloop pipeline stage 数 |
| `AlignmentA/AlignmentB` | A/B 全局内存访问对齐 |
| `SplitKSerial` | 是否启用 serial split-K |
| `Operator` | 底层 multiply-add 操作，通常使用默认值 |

当前 `gemm_wmma.cu` 显式设置三层 `GemmShape`：

```cpp
cutlass::gemm::GemmShape<128, 128, 32>  // ThreadblockShape
cutlass::gemm::GemmShape<64, 64, 32>    // WarpShape
cutlass::gemm::GemmShape<16, 8, 8>      // InstructionShape
```

这是为了展示 CUTLASS GEMM 的层级划分：

```text
Threadblock tile -> Warp tile -> Tensor Core instruction tile
```

官方 `basic_gemm.cu` 通常只写前几个参数，是因为后续参数有默认配置；本项目显式写出这些参数，是为了教学中清楚展示 Tensor Core 路径和 tiling 层级。

运行：

```bash
./gemm_wmma
```

也可以指定矩阵大小：

```bash
./gemm_wmma 512 512 512
```
