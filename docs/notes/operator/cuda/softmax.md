---
title: softmax 算子优化
---

# `softmax`

## 写在之前

`softmax`这个函数的重要性不言而喻，是非常基础的算子，他的思路可以参考上个规约函数，第一印象只是多了指数的计算。所以先给出第一印象的解题思路（向量化加载+规约计算）：

1. 为了向量化计算先做16字节对齐，先处理没有对齐的前面几个数据
2. 剩下的数据向量化加载，求指数并且做两件事情：保存到当前寄存器并且进行单个`thread`的求和，为后续`warp`规约准备数据
3. 原子加得到全局的和，就像`reduction`那样
4. 再次利用向量化处理最后的归一化，然后把结果写入输出

下面是我第一次解题的完整步骤，写了一版之后发现了很多的问题：

```c
#include <cuda_runtime.h>
#include <stdint.h> 

#define  mask  0xffffffff  // 选择wrap中的哪些lane参与计算
#define blockDIM 256

// 用三个全局变量：和、完成累加的 block 数、已读完和的 block 数
__device__ float sum_all = 0.f;
__device__ unsigned int g_count = 0;
__device__ unsigned int g_done  = 0;

__inline__ __device__ float warp_reduce_sum(float sum){
    sum += __shfl_down_sync(mask,sum,16);
    sum += __shfl_down_sync(mask,sum,8);
    sum += __shfl_down_sync(mask,sum,4);
    sum += __shfl_down_sync(mask,sum,2);
    sum += __shfl_down_sync(mask,sum,1);
    return sum;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 1.先做16字节对齐，处理没对齐的元素
    const int misaligin=(int)reinterpret_cast<uintptr_t>(input) & 15u;
    const int misalign_num =misaligin ? (16 - misaligin)/4 : 0 ;
    float no_align = 0.0f;
    if(idx < misalign_num){
        no_align = __expf(input[idx]);
    }
    // 2.向量化加载，求e^x，然后保存到寄存器变量。
    const float4* v= reinterpret_cast<const float4*>(input + misalign_num);
    const int core_num = (N - misalign_num ) /4;
    float sum = no_align;
    float4 tmp ;
    float tail[3]= {0.f};
    if(idx == core_num  && (N - misalign_num) % 4 != 0 ){//处理尾部core
 
        for(int i = 0;i<(N - misalign_num) % 4;i++){
            tail[i] = __expf(input[4*idx + i+misalign_num]);
            sum += tail[i];
        }
    }
    else if(idx < core_num){
        float4 input_a =  v[idx];
        tmp.x = __expf(input_a.x);
        tmp.y = __expf(input_a.y);
        tmp.z = __expf(input_a.z);
        tmp.w = __expf(input_a.w);
        sum += (tmp.x + tmp.y + tmp.z + tmp.w);
    }
    // 3.除了保存到共享内存，各个thread进行求和。规约到lane0，然后lane0存到共享内存
    sum = warp_reduce_sum(sum);
    int lane = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    const int warps = blockDIM / 32;
    __shared__ float warp1[blockDIM/32];
    if(lane == 0) warp1[warpid] = sum;
    __syncthreads();

    __shared__ float s_block_sum;
    float block_sum = 0.f;
    if (warpid == 0) {
        block_sum = (threadIdx.x < warps) ? warp1[lane] : 0.f;
        block_sum = warp_reduce_sum(block_sum);
        // 4.原子加进行串行求和
        if (lane == 0) {
            atomicAdd(&sum_all, block_sum);
            //全局同步
            __threadfence(); // 保证 sum_all 对其他 block 可见
            atomicAdd(&g_count, 1u); 
            s_block_sum = block_sum; 
        }
    }
    //自旋等待所有block累加
    if (threadIdx.x == 0) {
        while (atomicAdd(&g_count, 0u) < gridDim.x) { /* spin */ }
    }
    __syncthreads();
    // 5.每个thread拿到求和结果，再用向量化加载共享内存的内容进行计算除法，结果向量化写道output
    float tmp_sum = sum_all;
    //head
    if(idx < misalign_num){
        output[idx] = no_align / tmp_sum;
    }
    //tail
    if(idx == core_num  && (N - misalign_num) % 4 != 0 ){
        for(int i = 0;i<(N - misalign_num) % 4;i++){
            tail[i] /= tmp_sum;
            output[4*idx + i +misalign_num] = tail[i];
        }
    }
    //main
    else if(idx < core_num){
        float tmp_sum = sum_all;
        tmp.x /=  tmp_sum;
        tmp.y /=  tmp_sum;
        tmp.z /=  tmp_sum;
        tmp.w /=  tmp_sum;
        // output 对齐检查：如果 output 首地址 16 字节对齐，则 float4 写安全
        uintptr_t out_addr = reinterpret_cast<uintptr_t>(output + misalign_num);
        if ((out_addr & 15u) == 0) {
            reinterpret_cast<float4*>(output + misalign_num)[idx] = tmp;
        } else {
            // 退化成分量写，保证正确性
            float* op = output + misalign_num + 4*idx;
            op[0] = tmp.x; op[1] = tmp.y; op[2] = tmp.z; op[3] = tmp.w;
        }
    }
    // 7.所有 block 读完 sum_all 后，最后一个 block 重置全局变量
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int d = atomicAdd(&g_done, 1u) + 1u;
        if (d == gridDim.x) {
            sum_all = 0.f;
            g_count = 0u;
            g_done  = 0u;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

```

## 重新思考

这个题乍一眼看很像归约，只是在原先求和的基础上多加了归一化。

但是坑点在于：

1. 原先的碎片化处理写起来太复杂了：
   1. 16字节没有对齐的数据怎么存储，怎么计算，用哪个核心进行处理，其他核怎么办
   2. 向量化加载的尾部数据也要有上面的考虑
2. 求和过程中要先求幂计算，直接求的话会溢出，我上面的这个思路就没有处理溢出。所以需要额外先求出最大值，所有`input`减去最大值在求幂，这样并不影响结果，且不会溢出。(`softmax`的本质其实是两次归约＋一次写回)
3. 跨block同步如果要在单核内做的话，需要自旋等待，有死锁风险。
   1. 为什么？从 **CUDA 的线程模型和硬件执行模型** 说起，`__syncthreads()` 能工作的前提是所有线程一定在运行，硬件单元是以block级别调度到SM上的，但是block之间不能保证同时运行。
   2. 于是：
      - 在跑的 block 卡在 barrier 上不肯结束；
      - 没被调度的 block 因为 SM 被占满，**永远无法启动**；
      - **死锁**。
4. 有多种方法可以改进，比如`online softmax`，这也是后续`flashAttention`的核心组件。

三种优化版本的实现：

1. 防止数值溢出——`safe softmax`
2. 防止内存溢出——`tile`切片
3. 优化访存——`online softmax`

**需要改进的点：**

- 原先设置的原子计数器+自旋的方案：上面解释过了，容易死锁。我们选择方案1
  - 方案1：多阶段 kernel：拆成多个kernel函数，天然同步。
    - <u>缺点：多阶段有额外开销，并且需要重复读取数据。</u>
  - 方案2：`Grid Sync`：限制 grid 数量
    - 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算出每个 SM 能放多少 block；
    - 乘以 SM 总数，得到上限 `maxBlocks`；
    - 启动时使用 `cudaLaunchCooperativeKernel`，grid 大小**不能超过** `maxBlocks`。
    - <u>缺点：需要**grid-stride loop**</u>
  - 这个问题太经典了，属于**GPU硬件的不可能三角**，它们本质上互相冲突：
    - **任意大的 grid**（可扩展性）
    - **kernel 内全局同步**（所有 block 对齐）
    - **零额外开销**（无额外 kernel、无驻留限制、无额外内存）

## 优化点一：多阶段kernel

分成多阶段，虽然对显存的访问次数增加，但是却大大减少了开发的复杂度。同时这种解耦的思想也非常有助于debug。

前面说了，`softmax`的本质其实是两次归约＋一次写回。一次规约求最大值，一次规约求和。然后进行除法计算，写回output。

自然而然想到开辟三个kernel

### 1.`block_max_kernel`

```c
__inline__ __device__ float warp_reduce_max(float v){
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

__global__ void block_max_kernel(const float* input, float* max, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    //1.向量化加载，规约求最大值
    const int core_num = (N) /4;
    const float4* v= reinterpret_cast<const float4*>(input);
    const int warps = blockDIM / 32;
    __shared__ float warp_max[blockDIM/32];
    float thread_max = -FLT_MAX;
    int tail_num = (N) % 4 ;
    if(idx < core_num){
        float4 tmp =  v[idx];
        thread_max = fmaxf(fmaxf(tmp.x, tmp.y), fmaxf(tmp.z, tmp.w));
    }
    else if(idx == core_num  && tail_num != 0 ){
        int base = 4 * idx;
        for (int i = 0; i < tail_num; ++i)
            thread_max = fmaxf(thread_max, input[base + i]);
    }
    thread_max = warp_reduce_max(thread_max);
    if(lane == 0) warp_max[warpid] = thread_max;
    __syncthreads();
    if (warpid == 0) {
        float block_max = (threadIdx.x < warps) ? warp_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        // 4.原子加进行串行求最大值
        if (lane == 0) {
            atomicMaxFloat(max, block_max);
        }
    }
}
```

写法非常简单，需要额外注意的是`atomicMAX`函数需要自己定义：用 **CAS（Compare-And-Swap）循环** 在 float 上模拟出原子 `max` 操作。因为CUDA 只对 `int`/`unsigned int`/`unsigned long long` 提供 `atomicCAS`，没有 float 版本的原子 max，所以要"曲线救国"。

`atomicCAS(addr, expected, desired)` 的语义是：

> 如果 `*addr == expected`，就把 `*addr` 写成 `desired`，返回**旧的** `*addr`；否则不改，也返回旧的 `*addr`。

所以"读-改-写"要靠循环自己实现：

```cpp
int old = *addr_as_i, assumed;
do {
    assumed = old;                                    // 记下我这次读到的值
    old = atomicCAS(addr_as_i, assumed,               // 尝试：如果还是 assumed，就换成新值
        __float_as_int(fmaxf(value, __int_as_float(assumed))));
} while (assumed != old);                             // 如果 old 变了，说明被别人抢先改了，重试
```

### 2.`block_sum_kernel`

这个阶段需要减去最大值，所以要传入第一个kernel的输出值。

```cpp
__inline__ __device__ float warp_reduce_sum(float sum){
    sum += __shfl_down_sync(mask,sum,16);
    sum += __shfl_down_sync(mask,sum,8);
    sum += __shfl_down_sync(mask,sum,4);
    sum += __shfl_down_sync(mask,sum,2);
    sum += __shfl_down_sync(mask,sum,1);
    return sum;
}

__global__ void block_sum_kernel(const float* input, float* output,float* SUM,float* MAX, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    const int warps = blockDIM / 32;
    //1.向量化加载，求e^x，然后保存到寄存器变量。
    const int core_num = (N) /4;
    const int tail_num = (N) % 4 ;
    const float4* v= reinterpret_cast<const float4*>(input);
    float sum = 0.f;
    float4 tmp ={-FLT_MAX}; 
    if(idx < core_num){
        float4 input_a = v[idx];
        tmp.x = __expf(input_a.x - *MAX);
        tmp.y = __expf(input_a.y - *MAX);
        tmp.z = __expf(input_a.z - *MAX);
        tmp.w = __expf(input_a.w - *MAX);
        reinterpret_cast<float4*>(output)[idx] = tmp;
        sum += (tmp.x + tmp.y + tmp.z + tmp.w);
    }
    else if(idx == core_num && tail_num != 0 ){
        if(tail_num == 1){
            tmp.x = __expf(input[4*idx] - *MAX);
            sum += tmp.x;
            output[4*idx] = tmp.x;
        }
        else if(tail_num == 2){
            tmp.x = __expf(input[4*idx] - *MAX);
            tmp.y = __expf(input[4*idx + 1] - *MAX);
            sum += (tmp.x + tmp.y);
            output[4*idx] = tmp.x;
            output[4*idx+1] = tmp.y;
        }
        else if(tail_num == 3){
            tmp.x = __expf(input[4*idx] - *MAX);
            tmp.y = __expf(input[4*idx + 1] - *MAX);
            tmp.z = __expf(input[4*idx + 2] - *MAX);
            sum += (tmp.x + tmp.y + tmp.z);
            output[4*idx] = tmp.x;
            output[4*idx+1] = tmp.y;
            output[4*idx+2] = tmp.z;
        }

    }
    // 3.除了保存到共享内存，各个thread进行求和。规约到lane0，然后lane0存到共享内存
    sum = warp_reduce_sum(sum);
    __shared__ float warp1[blockDIM/32];
    if(lane == 0) warp1[warpid] = sum;
    __syncthreads();
    float block_sum = 0.f;
    if (warpid == 0) {
        block_sum = (threadIdx.x < warps) ? warp1[lane] : 0.f;
        block_sum = warp_reduce_sum(block_sum);
        // 4.原子加进行串行求和
        if (lane == 0) {
            atomicAdd(SUM, block_sum);
        }
    }
}
```

### 3. `softmax_kernel`

只是完成除法操作：

```cpp
__global__ void softmax_kernel(const float* input, float* output, float* SUM ,int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //1.向量化加载，求e^x，然后保存到寄存器变量。
    const int core_num = (N) /4;
    const int tail_num = (N) % 4 ;
    const float4* v= reinterpret_cast<const float4*>(input);
    if(idx < core_num){
        float4 input_a = v[idx];
        input_a.x /= *SUM;
        input_a.y /= *SUM;
        input_a.z /= *SUM;
        input_a.w /= *SUM;
        reinterpret_cast<float4*>(output)[idx] = input_a;
    }
    else if(idx == core_num && tail_num != 0 ){
        for (int i = 0; i < tail_num; ++i){
            float input_a = input[4*idx+i];
            input_a /= *SUM;
            output[4*idx+i] = input_a;
        }
        
    }
   
}
```

在核函数调用过程中，只需要额外再外部申请几个变量，就可以逐步调用，完成softmax操作。

### 4.调用

```cpp

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 4 * threadsPerBlock - 1) / (4 * threadsPerBlock);

    float* d_max = nullptr;
    float* d_sum = nullptr;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    // 1. max
    block_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, N);
    // 2. exp + sum（同时把 exp 结果写入 output）
    block_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_sum, d_max, N);
    // 3. divide
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, output, d_sum, N);

    cudaFree(d_max);
    cudaFree(d_sum);
}
```

这个优化相较于原版总结一下：

优点：

1. 完成了`safe softmax`
2. 进行多阶段，避免block间同步，消除了原先的死锁的可能

缺点：

1. 频繁`launch kernel`

   1. 目前` launch `了 3 个 kernel，每个都要把 `input`/`output` 完整读一遍：

      - `max kernel`：读 N
      - `sum kernel`：读 N + 写 N（exp 结果）
      - `softmax kernel`：读 N + 写 N

      **总共 ~5N 的访存**。

怎么样可以节省访存呢？这就说到第二个优化点了:`**One-pass online softmax**`

## 优化点二：`online softmax`

这个的原理网上很多，这里就不仔细介绍了。核心思想就是因为指数的运算法则：**对于输入的线性平移可以看作整体的放缩**。

```
exp(xi - m_new) = exp(xi - m_old) * exp(m_old - m_new)
```

这样就可以很方便的维护两个核心变量：

- `m` = 目前见过的最大值
- `s` = 目前见过的 `Σ exp(xi - m)`

这样就不需要多次对原数据的读取归约就可以一次性完成计算操作。

但是理想很丰满，现实很骨感；为了避免block同步以及简化开发难度，我们还是采用多kernel的方案，搭配在线计算完成算子设计；当然这个算子的上限可以单kernel完成，就是需要按照我们之前说的那样，**进行cooperative 的block数量限制，并且辅助gride-stride loop实现**（如果不考虑N很大的）；为了方便，我们适当偷懒一下。哈哈哈。

先说一下思路：

我们分为两个kernel进行：

- Kernel 1：online 求全局 `(M, S)`（每个线程 online 扫自己的元素，block 归约，atomic 合并）
  - 对比：之前求最大值和求和是分开的，这里其实是一步完成
- Kernel 2：读 input，`out[i] = exp(in[i] - M) / S`

访存：K1 读 N，K2 读 N + 写 N = **3N**，但其中 2N 在 L2 里，实际 HBM 流量 ≈ 读 1N + 写 1N。

在庖丁解牛理一下K1的思路：

1. 每个thread向量化读取，并且维护自己的`(M, S)`

2. 向量化成员的最大值求解，以及计算对应的指数和，更新维护的`(M, S)`

3. block内部进行求和求最大值，更新block的`(M, S)`

4. block之间进行一次`atomicMAX`,同时更新`(M, S)`，获得最终结果

   只有一次全局操作，所以不需要同步！！！



再啰嗦两句，这个题我想了很久，明明仍然需要求最大值，明明需要求和，那不还是两次归约？为什么说合并成一次了？

——其实，优化的是两次内存读取变成了一次，原先的两次规约前后依赖，所以会涉及到block同步，所以才拆开两个kernel。这个是由下至上，只有最后一步涉及到全局，两次规约都是block内的归约，可以同步的。

| 视角               | max 和 sum 的依赖发生在           | 结果                                           |
| :----------------- | :-------------------------------- | :--------------------------------------------- |
| 由上至下（朴素）   | **全局层**：全局 sum 依赖全局 max | 两次全局规约之间必须同步                       |
| 由下至上（online） | **本地层**：本地 S_t 依赖本地 M_t | 本地寄存器顺序执行，天然同步；全局只需一次合并 |

于是我按照上面的思路写K1，发现了新的问题:

```cpp
__global__ void online_softmax_kernel_MaxAndSum(const float* input, float* output, int N) {
......
// 4. block之间进行一次`atomicMAX`,同时更新`(M, S)`，获得最终结果
	if(threadIdx.x == 0 ){
        atomicMaxFloat(MAX,block_max);
        block_sum *= __expf(block_max - MAX);
        atomicAdd(SUM,block_sum);
    }

}
```

第四步这么写还是需要同步！

**正确的 online：`(M, S)` 必须打包成一次 CAS**

online 的核心不是"先 `atomicMax` 再 `atomicAdd`"，而是：

> **把 `(M, S)` 当成一个整体，用一次 64 位 `atomicCAS` 同时更新。**

```cpp
__device__ __forceinline__ void atomic_merge_pair(unsigned long long* addr,
                                                   float m, float s) {
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        float om = __int_as_float((int)(assumed & 0xffffffffu));
        float os = __int_as_float((int)(assumed >> 32));
        float nm = fmaxf(om, m);
        float ns = os * __expf(om - nm) + s * __expf(m - nm);
		//!!!!注意
		unsigned long long packed =
            ((unsigned long long)(unsigned int)__float_as_int(ns) << 32) |
            (unsigned long long)(unsigned int)__float_as_int(nm);   // ← 加 (unsigned int) 中转
			//否则负数测试失败
        old = atomicCAS(addr, assumed, packed);
    } while (assumed != old);
}
```

这里为什么要用long呢？

**在 GPU 上用 `atomicCAS` 做无锁更新时，硬件只提供 32/64 位整数的 CAS，没有“同时原子更新两个 float”的原语，所以你必须把两个 float 塞进一个 64 位整数里。**

完整的K1:

```CPP
__global__ void online_softmax_kernel_MaxAndSum(const float* input,unsigned long long* g_pair ,int N) {
// 1. 每个thread向量化读取，并且维护自己的`(M, S)`
    int idx = blockDim.x * blockIdx.x + threadIdx.x ;
    const int core_num = N /4;
    const int tail_num = N % 4;
    float max = -FLT_MAX;
    float sum = 0.f;
    // 2. 向量化成员的最大值求解，以及计算对应的指数和，更新维护的`(M, S)`
    if(idx < core_num){
        float4 input_a = reinterpret_cast<const float4*>(input)[idx];
        max = fmaxf(fmaxf(input_a.x, input_a.y), fmaxf(input_a.z, input_a.w));
        sum += (__expf(input_a.x - max) + __expf(input_a.y - max) +__expf(input_a.z - max) +__expf(input_a.w - max));
    }
    else if(idx == core_num && tail_num != 0 ){ //tail
        for(int i =0;i < tail_num ; i++){
            float tmp = input[4*idx +i];
            float max_tmp = max;
            max = max > tmp ? max : tmp;
            if(max_tmp < tmp) sum = sum*__expf(max_tmp - tmp) +__expf(tmp - max) ;//第一次online sum
            else sum += __expf(tmp - max) ;
        }
    }
    __syncthreads();
// 3. block内部进行求和求最大值，更新block的`(M, S)`
    __shared__ float block_max;
    __shared__ float block_sum;

    float tmp_m = block_reduce_max(max);
    if(threadIdx.x == 0) block_max = tmp_m;
    __syncthreads();
    sum = sum * __expf(max - block_max); // 第二次online sum
    float tmp_s  = block_reduce_sum(sum);
    if(threadIdx.x == 0) block_sum = tmp_s;
    __syncthreads();
// 4. block之间进行一次`atomicMAX`,同时更新`(M, S)`，获得最终结果
    if(threadIdx.x == 0 ){
        atomic_merge_pair(g_pair,block_max,block_sum);
    }
}
```

K2:

```CPP
__global__ void online_softmax_kernel_Div(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const unsigned long long* __restrict__ g_pair,
                                          int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int core_num = N / 4;
    const int tail_num = N % 4;

    // 1. 拆包 (M, S)
    unsigned long long p = *g_pair;
    float M = __int_as_float((int)(p & 0xffffffffu));   // 低 32 位
    float S = __int_as_float((int)(p >> 32));           // 高 32 位
    float inv_S = 1.f / S;                               // 除法变乘法

    // 2. 主分支：float4 向量化
    if (idx < core_num) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        float4 o;
        o.x = __expf(v.x - M) * inv_S;
        o.y = __expf(v.y - M) * inv_S;
        o.z = __expf(v.z - M) * inv_S;
        o.w = __expf(v.w - M) * inv_S;
        reinterpret_cast<float4*>(output)[idx] = o;
    }
    // 3. tail 分支：处理最后 1~3 个元素
    else if (idx == core_num && tail_num != 0) {
        int base = 4 * idx;
        for (int i = 0; i < tail_num; ++i) {
            float x = input[base + i];
            output[base + i] = __expf(x - M) * inv_S;
        }
    }
}
```

是不是非常简洁，同时额外的辅助函数也有更新：

```cpp
__device__ __forceinline__ void atomic_merge_pair(unsigned long long* addr,
                                                   float m, float s) {
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        float om = __int_as_float((int)(assumed & 0xffffffffu));
        float os = __int_as_float((int)(assumed >> 32));
        float nm = fmaxf(om, m);
        float ns = os * __expf(om - nm) + s * __expf(m - nm);
        unsigned long long packed =
            ((unsigned long long)(unsigned int)__float_as_int(ns) << 32) |
            (unsigned long long)(unsigned int)__float_as_int(nm);   // ← 加 (unsigned int) 中转
        old = atomicCAS(addr, assumed, packed);
    } while (assumed != old);
}


__inline__ __device__ float block_reduce_max(float v) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;

    // 1. warp 内归约
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 2));
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 1));

    // 2. 每个 warp 的 lane 0 写共享内存
    __shared__ float smem[32];       // 最多 32 个 warp
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    // 3. warp 0 读所有 warp 的结果，再归约一次
    if (warp == 0) {
        v = (lane < nwarps) ? smem[lane] : -FLT_MAX;
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 16));
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 8));
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 4));
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 2));
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, 1));
    }
    return v;   // 有效值只在 warp 0 的 lane 0
}

__inline__ __device__ float block_reduce_sum(float sum) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;

    // 1. warp 内归约
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // 2. lane 0 写共享内存
    __shared__ float smem[32];
    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    // 3. warp 0 再归约
    if (warp == 0) {
        sum = (lane < nwarps) ? smem[lane] : 0.f;   // 注意：sum 的空位补 0
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    return sum;   // 有效值只在 warp 0 的 lane 0
}
```

这样一个完整的`online softmax`就完成了！！！



## 总结

这个题我们尝试了三种方法：

1. 传统的`softmax`，遇到了溢出以及死锁的问题
2. `safe softmax`+ `muti-kernel`，但是对显存访问压力过大
3. `online softmax`，优化了访存两次归约一次完成



但是这个算子的优化点远不止于此，我们手写的算子性能远远比不上官方的算子，他们在维度、布局等等方面都会进行优化，并且会借助`profing`工具进行调优。只能说需要做的还有很多，但是对于初学入门已经足够了！