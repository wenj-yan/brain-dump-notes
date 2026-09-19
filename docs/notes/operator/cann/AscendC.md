---
title: Ascend C编程范式与完整流程
---

# Ascend C

## 1.抽象硬件架构

AI Core是昇腾AI处理器的计算核心，昇腾AI处理器内部有多个AI Core。本章节将介绍AI Core的并行计算架构抽象，**该抽象架构屏蔽了不同硬件之间的差异**。使用Ascend C进行编程时，基于抽象硬件架构，可以简化硬件细节，显著降低开发门槛。

- 计算单元
  - Cube计算单元，矩阵运算
  - Vector计算单元，向量运算
  - Scalar计算单元，地址计算等标量计算工作。
- 存储单元
  - `Local Memory`，对应的数据类型为`LocalTensor`
  - `Global Memory`，对应的数据类型为`GlobalTensor`
- 搬移单元
  - DMA，负责数据搬移。

**计算过程：**

- AI Core内部的异步并行计算过程：Scalar计算单元读取指令序列，并把向量计算、矩阵计算、数据搬运指令发射给对应单元的指令队列，向量计算单元、矩阵计算单元、数据搬运单元异步的并行执行接收到的指令。该过程可以参考图1中蓝色箭头所示的指令流。
- 不同的指令间有可能存在依赖关系，为了保证不同指令队列间的指令按照正确的逻辑关系执行，Scalar计算单元也会给对应单元下发同步指令。各单元之间的同步过程可以参考图1中的绿色箭头所示的同步信号流。
- AI Core内部数据处理的基本过程：DMA搬入单元将数据从Global Memory搬运到Local Memory，Vector/Cube计算单元完成数据计算，并把计算结果写回Local Memory，DMA搬出单元把处理好的数据从Local Memory搬运回Global Memory。该过程可以参考图1中的红色箭头所示的数据流。

## 2.编程模型

### 2.1 SPMD模型

单指令多数据编程方式

具体到Ascend C编程模型中的应用，是将需要处理的数据拆分并同时在多个计算核心（类比于上文介绍中的多个进程）上运行，从而获取更高的性能。**多个AI Core共享相同的****指令****代码，每个核上的运行实例唯一的区别是block_idx不同**，每个核通过不同的block_idx来识别自己的身份。block的概念类似于上文中进程的概念，**block_idx就是标识进程唯一性的进程ID**。并行计算过程如下图所示。

### 2.2 核函数

核函数（Kernel Function）是Ascend C算子**设备侧实现的入口**。Ascend C允许用户使用C/C++函数的语法扩展来编写设备端的运行代码，**用户在核函数中进行数据访问和计算操作，由此实现该算子的所有功能**。

！！！区别于普通的C++函数调用时仅执行一次，**当核函数被调用时，多个核都执行相同的核函数代码，具有相同的函数入参，并行执行。**！！！

核函数定义时需要使用函数类型限定符__global__和__aicore__；其指针入参变量需要增加变量类型限定符__gm__，表明该指针变量指向Global Memory上某处内存地址；使用<<<>>>内核调用符调用执行核函数，并指定调用时的执行核数。

例子：

```C++
// 实现核函数
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
{
    // 初始化算子类，算子类提供算子初始化和核心处理等方法
    KernelAdd op;
    // 初始化函数，获取该核函数需要处理的输入输出地址，同时完成必要的内存初始化工作
    op.Init(x, y, z);
    // 核心处理函数，完成算子的数据搬运与计算等核心逻辑
    op.Process();
}
// 调用核函数
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}
```

#### **核函数的定义与调用**

- 函数类型限定符，除了需要按照C/C++函数声明的方式定义核函数之外，还要为核函数加上额外的函数类型限定符，包含global和aicore。
  - global，核函数，可以被<<<>>>调用。
  - aicore，标识该核函数在设备端AI Core上执行

```C
__global__ __aicore__ void kernel_name(argument list);
```

编程中使用到的函数可以分为三类：**核函数（device侧执行）、host侧执行函数、device侧执行函数（除核函数之外的）**。下图以Kernel直调算子开发方式为例描述三者的调用关系：

- host侧执行函数可以调用同类的host执行函数，也就是通用C/C++编程中的函数调用；也可以通过<<<>>>调用核函数。
- device侧执行函数（除核函数之外的）可以调用同类的device侧执行函数。
- 核函数可以调用device侧执行函数（除核函数之外的）。

| 特性         | 核函数                                                       | Device侧执行函数（核函数除外）                               |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **调用者**   | **只能由Host侧代码** 通过 `kernel_name<<<grid, block>>>(args)` 调用。 | **只能由Device侧代码**调用，即由**核函数**或其他**Device侧函数**调用。 |
| **执行模式** | **并行执行**。启动后，会创建大量线程（由grid和block定义），所有线程都执行该核函数的代码。 | **串行执行**。在被调用时，就像普通C/C++函数一样，在**当前线程**中顺序执行。 |
| **关键字**   | 使用 `__global__` 关键字定义。                               | 使用 `__aicore__` 关键字定义。                               |
| **返回值**   | **必须**是 `void` 类型。                                     | 可以有返回值（任何类型）。                                   |
| **作用**     | **入口点与任务分发**。它是Host与Device之间的桥梁，负责定义并行任务的规模和启动并行计算。 | **功能实现与代码复用**。它封装了在GPU上执行的具体计算逻辑，避免代码重复，使核函数更清晰。 |

- 变量类型限定符
  - 指针入参变量需要增加变量类型限定符__gm__，表明该指针变量指向Global Memory上某处内存地址。
- 其他
  - 1规则：核函数必须具有void返回类型。
  - 2规则：仅支持入参为指针或C/C++内置数据类型（Primitive data types），如：half* s0、float* s1、int32_t c。
  - 3建议：为了统一表达，建议使用GM_ADDR宏来修饰入参，GM_ADDR宏定义如下：

```C++
//定义
#define GM_ADDR __gm__ uint8_t*
//使用
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
```

- 这里统一使用uint8_t类型的指针，在后续的使用中需要将其转化为实际的指针类型。

**核函数的调用：**

- 核函数使用内核调用符<<<...>>>这种语法形式，来规定核函数的执行配置：
  - 内核调用符仅可在NPU侧编译时调用，CPU侧编译无法识别该符号。

```C
kernel_name<<<blockDim, l2ctrl, stream>>>(argument list);
```

- 参数：
  - `blockDim`，规定了核函数将会在几个核上执行。每个执行该核函数的核会被分配一个逻辑ID，即block_idx，可以在核函数的实现中调用[GetBlockIdx](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0185.html)来获取block_idx；为了充分利用硬件资源，一般设置为物理核的核数或其倍数。
  - `l2ctrl`，保留参数，暂时设置为固定值nullptr，开发者无需关注，无用
  - `stream`，类型为aclrtStream，stream**用于维护一些异步操作的执行顺序**，确保按照应用程序中的代码调用顺序在device上执行。

核函数的调用是异步的，核函数的调用结束后，控制权立刻返回给主机端，可以调用以下[aclrtSynchronizeStream](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/appdevgapi/aclcppdevg_03_0076.html)函数来强制主机端程序等待所有核函数执行完毕。

```C
aclError aclrtSynchronizeStream(aclrtStream stream);
```

#### 模板核函数的定义与调用

核函数定义示例如下，它有两个模板参数：a和T。a是一个非类型模板参数，T是一个类型模板参数。

```C
template<int a, typename T>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
...
    AscendC::printf("Print Template a: %d\n", a);
...
    xGm.SetGlobalBuffer((__gm__T*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__T*)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__T*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
...
}
```

模板核函数的调用方式如下：add_custom<20, float>这部分代码调用了名为add_custom的核函数，并为其模板参数提供了具体值。

```C
add_custom<20, float><<<blockDim, nullptr, stream>>>(x, y, z);
```

### 2.3 编程范式

**编程范式描述了算子实现的固定流程，基于编程范式进行编程，可以快速搭建算子实现的代码框架。**

AI Core内部的执行单元异步并行地执行接收到的指令。如下图所示，从输入数据到输出数据需要经过3个阶段任务的处理（T1、T2、T3），多个执行单元并行处理，每个执行单元只会专注于一个任务的处理，会处理所有的数据分片。

Ascend C编程范式就是这样一种流水线式的编程范式，把算子核内的处理程序，分成多个**流水任务**，通过队列（Queue）完成**任务间通信和同步**，并通过统一的**资源管理**模块（Pipe）来统一管理内存、事件等资源。

#### Vector编程范式

- **CopyIn**负责搬入操作：将输入数据从Global Memory搬运到Local Memory（VECIN用于表达矢量计算搬入数据的存放位置），完成搬运后执行入队列操作；
- **Compute**负责矢量指令计算操作：完成队列出队后，从Local Memory获取数据并计算，计算完成后执行入队操作；
- **CopyOut**负责搬出操作：完成队列出队后，将计算结果从Local Memory（VECOUT用于表达矢量计算搬出数据的存放位置）搬运到Global Memory。

1. 申请内存
2. 数据搬移H2D
3. 计算
4. 释放内存

参考计算模板：

```C++
AscendC::TPipe pipe;   //创建全局的资源管理   
AscendC::TQue<AscendC::TPosition::VecIn, 1> queIn;  //创建CopyIn阶段的队列
AscendC::TQue<AscendC::TPosition::VecOut, 1> queOut; //创建CopyOut阶段的队列
// Init 阶段：
pipe.InitBuffer(queIn, 2, 1024);  // 开启double buffer,将待处理的数据一分为二,实现流水并行
pipe.InitBuffer(queOut, 2, 1024);
for-loop {
    //CopyIn 阶段{
    auto tensor = queIn.AllocTensor<half>();     //从Que上申请资源, 长度1024   !!!获得内存块A的使用权
    AscendC::DataCopy(tensor, gm, 1024);          //搬运数据从GM到VECIN   向内存块A写入数据
    queIn.EnQue(tensor);                        // 声明：内存块A已准备好，可以用于计算
    }
    //Compute阶段{   
    auto tensor = queIn.DeQue<half>();           // 获得内存块A的使用权
    auto tensorOut = queOut.AllocTensor<half>();    
    AscendC::Abs(tensorOut, tensor, 1024);        // 使用内存块A进行计算
    queIn.FreeTensor(tensor);                      // 释放内存块A的使用权
    queOut.EnQue(tensorOut);
    }
    //CopyOut 阶段{
    auto tensor = queOut.DeQue<half>();
    AscendC::DataCopy(gmOut, tensor, 1024);
    queOut.FreeTensor(tensor);
    }
}
```

为什么使用`VECIN``CECOUT`队列而**不是直接拷贝数据**？

- 从预分配的内存池获取，不是malloc
- **队列作为缓冲区**：使得不同阶段可以同时工作
- **隐藏数据搬运延迟**：计算单元不用等待数据拷贝
- 进入队列的不是数据本身，而是存储空间的使用权。

#### Cube编程范式

```C
// 创建Matmul对象 创建对象时需要传入A、B、C、Bias的参数类型信息， 类型信息通过MatmulType来定义，包括：内存逻辑位置、数据格式、数据类型。
typedef MatmulType<TPosition::GM, CubeFormat::ND, half> aType; 
typedef MatmulType<TPosition::GM, CubeFormat::ND, half> bType; 
typedef MatmulType<TPosition::GM, CubeFormat::ND, float> cType; 
typedef MatmulType<TPosition::GM, CubeFormat::ND, float> biasType; 
Matmul<aType, bType, cType, biasType> mm; 

REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化
// CopyIn阶段：完成从GM到LocalMemory的搬运
mm.SetTensorA(gm_a);    // 设置左矩阵A
mm.SetTensorB(gm_b);    // 设置右矩阵B
mm.SetBias(gm_bias);    // 设置Bias
// Compute阶段：完成矩阵乘计算
while (mm.Iterate()) { 
    // CopyOut阶段：完成从LocalMemory到GM的搬运
    mm.GetTensorC(gm_c); 
}
// 结束矩阵乘操作
mm.End();
```

#### 融合算子编程范式

支持Vector与Cube混合计算的算子称之为融合算子。

- Cube的输出可以作为Vector的输入：CO2->VECIN
- Vector的输出可以作为Cube的输入：VECOUT->A1->A2、VECOUT->B1->B2

伪代码：

```C
template<typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process()
{
    // 步骤1：初始化一个MatMul对象，将输入数据从Global Memory搬运到Cube核上。
    uint32_t computeRound = 0;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj);
    matmulObj.Init(&tiling);
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    
    while (matmulObj.template Iterate<true>()) { // 步骤2：进行MatMul内部的计算。
        // 步骤3：将MatMul的计算结果搬运到Vector核上。
        reluOutLocal = reluOutQueue_.AllocTensor<cType>();
        matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
       // 步骤4：进行Vector矢量计算。
        AscendC::LeakyRelu(reluOutLocal, reluOutLocal, (cType)alpha, tiling.baseM * tiling.baseN);
        reluOutQueue_.EnQue(reluOutLocal);
        // 步骤5：将输出结果搬运到Global Memory上
        reluOutQueue_.DeQue<cType>();
        ...
        AscendC::DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
        reluOutQueue_.FreeTensor(reluOutLocal);

        computeRound++;
    }
    matmulObj.End();
}
```



## 3.算子实现

Ascend C的算子实现主要包含两个部分：

- Host侧Tiling实现
- 由于NPU中AI Core内部存储无法完全容纳算子输入输出的所有数据，**需要每次搬运一部分输入数据进行计算然后搬出，再搬运下一部分输入数据进行计算，这个过程就称之为Tiling**。切分数据的算法称为Tiling算法或者Tiling策略。根据算子的shape等信息来确定数据切分算法相关参数（比如每次搬运的块大小，以及总共循环多少次）的计算程序，称之为Tiling实现，也叫Tiling函数（Tiling Function）。由于Tiling实现中完成的均为标量计算，AI Core并不擅长，所以我们将其独立出来放在Host侧CPU上执行。
- Device侧Kernel实现
- Kernel实现即算子核函数实现，在Kernel函数内部通过解析Host侧传入的Tiling结构体获取Tiling信息，根据Tiling信息控制数据搬入搬出Local Memory的流程；通过调用计算、数据搬运、内存管理、任务同步API，实现算子逻辑。其核心逻辑基本上都为计算密集型任务，需要在NPU上执行。

### 2.1 矢量编程（add为例）

目录如下：

- [基础矢量算子](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0033.html)：开发一个简单的Add矢量算子。
- [TBuf的使用](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10003.html)：在算子计算过程中使用临时空间存储运算的中间结果。
- [多核Tiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0035.html)：算子在AI处理器的多个核上运行，所有核的计算数据量相等且32字节对齐。
- [尾块Tiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00009.html)：算子在AI处理器的多个核上运行，所有核的计算数据量相等，每个核上除最后一个数据块（尾块）外，其余数据块的数据量相等，每个核都需要处理尾块数据的计算。
- [尾核Tiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10008.html)：算子在AI处理器的多个核上运行，数据无法平均分配到每个核。将所有核分为多个整核和多个尾核，整核的计算数据量相等，尾核的计算数据量相等。
- [尾核&尾块](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10009.html)：算子在AI处理器的多个核上运行，数据无法平均分配到每个核，同时每个核内的数据无法均分，除最后一个数据块（尾块）外，其余数据块的数据量相等，每个核都需要单独处理尾块数据的计算。
- [DoubleBuffer场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10010.html)：使能double buffer，算子中的多条流水并行执行。
- [Broadcast场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10011.html)：算子中两个输入的shape（形状）不相等，需要将一个输入的shape进行Broadcast（广播）后，再执行计算。
- [非对齐场景](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0034.html)：更多数据非32字节对齐场景的处理方案。

#### 基础矢量算子

- 算子分析：分析算子的数学表达式、输入、输出以及计算逻辑的实现，明确需要调用的Ascend C接口。
- 核函数定义：定义Ascend C算子入口函数。
- 根据[矢量编程范式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html#ZH-CN_TOPIC_0000002336176954__section116515238815)实现算子类：完成核函数的内部实现，包括3个基本任务：CopyIn，Compute，CopyOut。

**算子分析**

1. 算子分析

```Plaintext
z = x + y
```

1. 计算逻辑：输入数据需要先从外部存储（Global Memory）搬运进片上存储（Unified Buffer），然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。
2. 明确输入与输出
   1. 输入：x,y
   2. 输出：z
   3. 数据类型：half
   4. 输入输出shape：[1,2048]
   5. format：ND
3. 确定核函数名称与参数
   1. 名称：`add_custom`
   2. 参数：`x`,`y`,`z`，均为global memory内存地址
4. 确定算子实现所需要的接口
   1. `datacopy`完成数据搬移
   2. `add`算子实现运算
   3. `queue`队列管理计算。

**核函数****定义**

1. 函数原型定义

```C
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
}
```

1. 调用算子类的初始化和处理函数
2. 算子类的Init函数，完成内存初始化相关工作，Process函数完成算子实现的核心逻辑

```C
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
```

1. 核函数调用封装
2. 对核函数的调用进行封装，得到add_custom_do函数，便于主程序调用。

```C
#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
}
#endif
```

**算子类实现**

比较简单，这里不做赘述。

算子类主要包含实现上述计算图逻辑：

```C
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z){}
    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process(){}

private:
    // 搬入函数，完成CopyIn阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyIn(int32_t progress){}
    // 计算函数，完成Compute阶段的处理，被核心Process函数调用
    __aicore__ inline void Compute(int32_t progress){}
    // 搬出函数，完成CopyOut阶段的处理，被核心Process函数调用
    __aicore__ inline void CopyOut(int32_t progress){}

private:
    AscendC::TPipe pipe;  // Pipe内存管理对象
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;  // 输入数据Queue队列管理对象，TPosition为VECIN
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;  // 输入数据Queue队列管理对象，TPosition为VECIN
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;  // 输出数据Queue队列管理对象，TPosition为VECOUT  
    AscendC::GlobalTensor<half> xGm;  // 管理输入输出Global Memory内存地址的对象，其中xGm, yGm为输入，zGm为输出
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};
```

- 初始化函数

  - 设置输入输出Global Tensor的Global Memory内存地址。

  - ```C
    xGm.SetGlobalBuffer((__gm__ half *)x, TOTAL_LENGTH);
    ```

  - pipe内存管理对象为输入输出队列分配内存

  - ```C
    pipe.InitBuffer(inQueueX, 1, TOTAL_LENGTH * sizeof(half))
    ```

```C
constexpr int32_t TOTAL_LENGTH = 1 * 2048;  // 数据总长
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // 设置Global Memory的起始地址以及长度
    xGm.SetGlobalBuffer((__gm__ half *)x, TOTAL_LENGTH);
    yGm.SetGlobalBuffer((__gm__ half *)y, TOTAL_LENGTH);
    zGm.SetGlobalBuffer((__gm__ half *)z, TOTAL_LENGTH);

    // 通过Pipe内存管理对象为输入输出Queue分配内存
    pipe.InitBuffer(inQueueX, 1, TOTAL_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, 1, TOTAL_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, 1, TOTAL_LENGTH * sizeof(half));
}
```

- 核函数

核函数的实现分为3个基本任务：CopyIn，Compute，CopyOut。Process函数中通过如下方式调用这三个函数。

```C
__aicore__ inline void Process()
{
    CopyIn();
    Compute();
    CopyOut();
}
```

1. copyin

```C
__aicore__ inline void CopyIn(int32_t progress)
{
    // 从Que中为LocalTensor分配内存
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    // 将GlobalTensor数据拷贝到LocalTensor
    AscendC::DataCopy(xLocal, xGm, TOTAL_LENGTH);
    AscendC::DataCopy(yLocal, yGm, TOTAL_LENGTH);
    // LocalTensor放入VECIN的Queue中
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```

1. compute

```C
__aicore__ inline void Compute(int32_t progress)
{
    // 将Input从VECIN的Queue中取出
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    // 调用Add算子进行计算
    AscendC::Add(zLocal, xLocal, yLocal, TOTAL_LENGTH);
    // 将计算结果LocalTensor放入到VECOUT的Queue中
    outQueueZ.EnQue<half>(zLocal);
    // 释放LocalTensor
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
```

1. copyout

```C
 __aicore__ inline void CopyOut(int32_t progress)
{
    // 将计算结果从到VECOUT的Queue中取出
    AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    // 将计算结果从LocalTensor数据拷贝到GlobalTensor
    AscendC::DataCopy(zGm, zLocal, TOTAL_LENGTH);
    // 释放LocalTensor
    outQueueZ.FreeTensor(zLocal);
}
```

#### TBuf使用

存储中间结果。

这些中间结果以临时变量表示，临时变量占用的内存可以使用**TBuf数据结构**来管理

例子:

```C
__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t> ();
    AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t> ();
    AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t> ();
 
    AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
    AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
 
    AscendC::Add(tmpTensor2, tmpTensor0, tmpTensor1, TOTAL_LENGTH);
    AscendC::Cast(zLocal, tmpTensor2, AscendC::RoundMode::CAST_RINT, TOTAL_LENGTH);
 
    outQueueZ.EnQue<bfloat16_t>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
```

#### 多核/Tiling切分

为了提高算子的执行效率，**通常在算子中实现多核并行计算**，即对输入数据进行切分，并将不同的数据块分配到不同的核上处理。

由于单个核上内部存储Local Memory大小有限，存在无法一次完整地容纳算子的输入和输出数据的场景，因此需要每次搬运一部分输入进行计算然后搬出，再搬运下一部分输入进行计算，直到获得最终的完整结果，这个数据切分、分块计算的过程称之为**Tiling**。切分数据的算法称为Tiling算法或者Tiling策略。根据算子的shape等信息来确定数据切分算法相关参数（比如每次搬运的块大小，以及总共循环多少次）的计算程序，称之为**Tiling实现**，也叫Tiling函数（Tiling Function）。由于Tiling实现中完成的均为标量计算，AI Core并不擅长，所以我们将其独立出来放在Host侧CPU上执行。核函数内部通过解析Host侧传入的Tiling结构体获取Tiling信息，根据Tiling信息控制数据搬入、搬出Local Memory的流程；通过调用计算、数据搬运、内存管理、任务同步API，实现算子逻辑。

由于硬件限制，在对输入数据进行数据切分时应遵循以下几个原则：

1. 由于AI Core中Unified Buffer上的物理限制，要求Unified Buffer上的数据存储空间必须保持32字节对齐。
   1. 输入数据不满足32字节对齐时，需要取输入数据长度向上对齐到32字节的长度作为输入数据总长度。
   2. 进行Tiling有关计算时，以32字节为最小单位进行计算。
2. 尽可能最大利用Unified Buffer空间。
3. AI Core与外部存储交互时会产生性能开销，频繁的进行数据搬运会导致性能瓶颈，因此应尽可能充分利用Unified Buffer空间，减少从Global Memory上搬运数据的次数。
4. 昇腾AI处理器包含多个AI Core，应该充分均衡利用多核计算能力，将计算均衡分配到多个AI Core上。

多种场景：

- 核间均分，核内均分
- 核间均分，核内不均分
  - 每个核处理的数据量相同，核内各数据块的数据量不完全相同。此场景基于多核Tiling，核内数据不能切分为多个数据量相同且32字节对齐的数据块，需要通过[尾块Tiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00009.html)处理尾块数据的计算。
- 核间不均分，核内均分
- 核间不均分，核内不均分

##### 多核Tiling

本样例为输入数据在核间均分、核内均分场景。本样例的Tiling策略为：

数据整体长度TOTAL_LENGTH为8 * 2048

数据平均分配到8个核上运行

每个核上计算的数据长度BLOCK_LENGTH为2048

将单核上的数据切分成16块（此处切分成16块仅用来作为Tiling的样例，并不代表性能最佳，仅供参考）

每块数据的长度TILE_LENGTH为128。

基于本节的切分策略，Tiling需要定义如下参数：

- blockLength：每个核的计算数据长度；
- tileNum：每个核需要计算的数据块个数；
- tileLength：每个核内每个数据块的长度。

使用C++语法定义TilingData结构体，代码如下。该头文件命名为*“`算子名称_tiling.h”`*。

```C
struct AddCustomTilingData {
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
}
```

接下来，创建一个与Tiling结构体头文件对应的cpp文件`add_custom_tiling.cpp`，并在该文件内完成Tiling参数的计算。

由于每个核内数据被切分为16块，根据使用的核数和核内切分数，计算Tiling参数，并写入到Tiling结构体内。代码示例如下：

```C
#include "add_custom_tiling.h"
constexpr int32_t CORE_NUM = 8;                             // 使用的核数
constexpr int32_t TILE_NUM = 16;                             // 核内切分数量
void GenerateTilingData(uint8_t* tilingBuf)
{
    uint32_t totalLength;
    // 此处省略如何获取数据总长TOTAL_LENGTH，可以根据具体情况实现。本章节仅介绍Tiling相关内容。

    AddCustomTilingData *tiling = reinterpret_cast<AddCustomTilingData *>(tilingBuf);
    uint32_t blockLength = TOTAL_LENGTH / CORE_NUM;
    uint32_t tileNum = TILE_NUM;
    uint32_t tileLength = blockLength / tileNum;

    tiling->blockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
}
```

最后，在Host侧调用程序中，调用上述Tiling参数计算函数，计算出相关参数，然后传递到Kernel侧核函数。

### 2.2 矩阵编程（matmul）

#### a.数据流

在了解矩阵乘法数据流之前，需要先回顾一下几个重要的存储**逻辑位置**的概念：

- 搬入数据的存放位置：A1，用于存放整块A矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：B1，用于存放整块B矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：C1，用于存放整块的矩阵乘偏置Bias矩阵，可类比CPU多级缓存中的二级缓存；
- 搬入数据的存放位置：A2，用于存放切分后的小块A矩阵，可类比CPU多级缓存中的一级缓存；
- 搬入数据的存放位置：B2，用于存放切分后的小块B矩阵，可类比CPU多级缓存中的一级缓存；
- 搬入数据的存放位置：C2，用于存放切分后的小块矩阵乘偏置Bias矩阵，可类比CPU多级缓存中的一级缓存；
- 结果数据的存放位置：CO1，用于存放小块结果C矩阵，可理解为Cube Out；
- 结果数据的存放位置：CO2，用于存放整块结果C矩阵，可理解为Cube Out；
- 搬入数据的存放位置：VECCALC，一般在计算需要临时变量时使用此位置。

#### b.基础知识

**注意****数据类型**：

- ND
- NZ

具体要使用哪一种数据类型，要注意转换。

**tiling**：

- 多核切分（M,K）->(singleCoreM,singleCoreK)
- 核内切分(singleCoreM,singleCoreK)->（baseM,baseK）。这里是有累加操作的。
- 其他参数
  - iterateOrder，偏移方向
  - depthA1，depthB1：A1、B1上存储的矩阵片全载A2/B2的份数，A2、B2存储大小分别是baseM * baseK，baseN * baseK，即depthA1是A1矩阵切片含有baseM * baseK块的个数，depthB1是B1矩阵切片含有baseN * baseK块的个数。
  - stepM，stepN：stepM为左矩阵在A1中缓存的buffer M方向上baseM的倍数，stepN为右矩阵在B1中缓存的buffer N方向上baseN的倍数。
  - stepKa，stepKb：stepKa为左矩阵在A1中缓存的buffer K方向上baseK的倍数，stepKb为右矩阵在B1中缓存的buffer K方向上baseK的倍数。

#### c.算子实现

Ascend C提供一组Matmul高阶API，**封装了这些常用的切分和数据搬运、计算的算法逻辑**，方便用户快速实现Matmul矩阵乘法的运算操作。开发者在host侧通过调用API自动获取Tiling参数，该参数传递到kernel侧后，在初始化操作时传入，通过几个简单的API即可完成矩阵乘操作。

### 2.3 工程化算子开发

工程化算子开发是指基于自动生成的**自定义算子工程**完成算子实现、编译部署、单算子调用代码自动生成等一系列流程。

该开发流程是标准的开发流程，建议开发者按照该流程进行算子开发。该方式下，算子开发的代码会更规范、统一、易于维护；同时该方式考虑了单算子API调用、算子入图、AI框架调用等功能的集成，使得开发者易于借助CANN框架实现上述功能。

#### 创建算子工程

自定义算子工程生成工具**msOpGen**

可基于算子原型定义输出算子工程：包括**算子host侧代码实现文件**、**算子kernel侧实现文件**以及**工程编译配置文件等**。

编写算子的原型定义json文件，用于生成算子开发工程。

注意：！！！！

opname需要规范，命名为驼峰形式

```JSON
[
    {
        "op": "AddCustom",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            }
        ]
    }
]
```

使用msOpGen工具生成算子的开发工程:

```Shell
${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i $HOME/sample/add_custom.json -c ai_core-<soc_version> -lan cpp -out $HOME/sample/AddCustom

/home/aaa/Ascend/ascend-toolkit/8.2.RC1/python/site-packages/bin/msopgen gen -i matmul_single.json -c ai_core-Ascend910B3 -lan cpp -out ./Matmul_single
```

- **${INSTALL_DIR}**为CANN软件安装后文件存储路径
- -i：指定算子原型定义文件*add_custom*.json所在路径，请根据实际情况修改。
- -c：ai_core-*<soc_version>*代表算子在AI Core上执行，*<soc_version>*为昇腾AI处理器的型号。
- -lan： 参数cpp代表算子基于Ascend C编程框架，使用C/C++编程语言开发。
- -out：生成文件所在路径，可配置为绝对路径或者相对路径，并且工具执行用户对路径具有可读写权限。若不配置，则默认生成在执行命令的当前路径。

执行完成后，会在指定目录生成算子工程目录：

```Plaintext
AddCustom
├── build.sh         // 编译入口脚本
├── cmake            // 算子工程编译所需脚本及公共编译文件存放目录
├── CMakeLists.txt    // 算子工程的CMakeLists.txt
├── CMakePresets.json    // 编译配置项
├── framework        // AI框架适配时，算子插件实现文件目录
├── op_host                      // host侧实现文件
│   ├── add_custom_tiling.h    // 算子tiling定义文件
│   ├── add_custom.cpp         // 算子原型注册、shape推导、信息库、tiling实现等内容文件
│   ├── CMakeLists.txt
├── op_kernel                   // kernel侧实现文件
│   ├── CMakeLists.txt   
│   ├── add_custom.cpp        // 算子代码实现文件 
└── scripts                     // 自定义算子工程打包相关脚本所在目录
```

#### 算子核函数实现

#### 算子Host Tiling实现

#### 算子编译

编译AddCustom工程，生成自定义算子安装包，并将其安装到算子库中。

- 编译自定义算子工程，构建生成自定义算子包。

  - ```Plaintext
    ./build.sh
    ```

  - 编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包**custom_opp.run**，例如**“custom_opp_ubuntu_x86_64.run”**。

- 自定义算子安装包部署

  - 在自定义算子包所在路径下，执行如下命令，安装自定义算子包。

  - ```Plaintext
    ./custom_opp_<target os>_<target architecture>.run
    ```

  - 命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中

  - 目录`/home/aaa/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/`

  - ```Plaintext
    ├── opp    // 算子库目录
    │   ├── built-in     // 内置算子所在目录
    │   ├── vendors     // 自定义算子所在目录
    │       ├── config.ini
    │       └── vendor_name1   // 自定义算子所在目录，若不指定路径安装，默认为“customize”
    │           ├── framework     //自定义算子插件库
    │           ├── op_impl
    │           │   └── ai_core
    │           │       └── tbe
    │           │           ├── config
    │           │           │   └── ${soc_version}     //昇腾AI处理器类型
    │           │           │       └── aic-${soc_version}-ops-info.json     //自定义算子信息库文件
    │           │           ├── vendor_name1_impl    //自定义算子实现代码文件
    │           │           │   └── dynamic
    │           │           │       ├── xx.cpp
    │           │           │       └── xx.py
    │           │           ├── kernel     //自定义算子二进制文件
    │           │           │   └── ${soc_version}     //昇腾AI处理器类型
    │           │           │   └── config
    │           │           └── op_tiling
    │           │               ├── lib
    │           │               └── liboptiling.so 
    │           └── op_proto     //自定义算子原型库所在目录
    │               ├── inc
    │               │   └── op_proto.h
    │               └── lib
    │       ├── vendor_name2   // 存储厂商vendor_name2部署的自定义算子
    ```

#### 算子ST测试

- 创建算子ST测试用例定义文件“AddCustom_case.json”

  - ```Plain
    msopst create -i Op_implementation/add_custom.cpp -out ./output
    ```

- 设置环境变量

  - 如下为设置环境变量的示例。$${INSTALL_DIR}表示CANN软件安装目录，例如$$HOME/Ascend/ascend-toolkit/latest。*{arch-os}*为运行环境的架构和操作系统，*arch*表示操作系统架构，*os*表示操作系统，例如x86_64-linux或aarch64-linux。

  - ```Plaintext
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/arm64-linux/
    export NPU_HOST_LIB=$HOME/Ascend/ascend-toolkit/8.2.RC1/arm64-linux/devlib
    ```

- 进入msOpST工具所在目录，执行如下命令生成并执行测试用例。

  - 进入msOpST工具所在目录。`cd $HOME/Ascend/ascend-toolkit/latest/python/site-packages/bin`
  - 生成测试用例文件并执行,`./msopst run -i $HOME/wenjie/learn/addCustom/AddCustom_case.json -soc Ascend910B3 -out $HOME/wenjie/learn/addCustom/AddCustom_st`

- 此命令执行完成后，会输出类似如下打屏结果：

  - ```Plaintext
    ------------------------------------------------------------------------
    - test case count: 1
    - success count: 1
    - failed count: 0
    ------------------------------------------------------------------------
    2023-08-28 20:20:40 (25058) - [INFO] Process finished!
    2023-08-28 20:20:40 (25058) - [INFO] The st report saved in:  xxxx/AddCustom_st/20230828202015/st_report.json.
    ```

  - 也可以查看上述屏显信息提示的“st_report.json”文件，查看详细运行结果。

### 2.4 融合算子编程

## 4.tiling

切分可分为**核间切分**和**核内切分**两块内容。

1. 核间切分是将数据分配给NPU的多个Aicore；
2. 分配到某个核计算的数据，也需要分批处理，这就是核内切分。

**Tiling过程**从编程实践上，在算子工程的op_host和op_kernel目录下的三个文件中：

- op_host文件夹下**“算子Tiling结构定义头文件”**以及**“算子host实现cpp文件”**的Tiling实现函数里
  - “算子名称_tiling.h”
    - TilingData数据结构（切分算法相关参数）的定义和注册；
  - “算子名称.cpp”
    - 实现部分：根据算子的shape等信息来确定数据切分算法相关参数（比如每次搬运的块大小，以及总共循环多少次）的计算程序**。**由于Tiling实现中完成的均为标量计算，AI Core并不擅长，所以将其独立出来放在host CPU上执行。
- op_kernel目录下的算子device侧实现“**算子名称.cpp**”，根据TilingData传入的参数，结合API实现数据切分操作。
  - 核间切分：体现在算子类的init函数中，通过SetGlobalBuffer，用来设置每个核需要处理的数据在Global Memmory上的起始地址。
  - 核内切分，体现在算子类的三个函数中：
    - **init函数的pipe.InitBuffer**为TQue进行Local Memory内存分配
    - **CopyIn函数**中，DataCopy将输入向量从Global Memory拷贝到Local Memory，进行运算
    - **CopyOut函数**中，DataCopy将计算结果从Local Memory拷贝到 Global Memory

### 实现逻辑

首先，我们需要调用Ascend C “Host侧实现API”中的“PlatformAscendC类”的有关函数，获取与“Host侧的Tiling函数”有关的硬件平台的信息。常用的有获取当前硬件平台的类型，可用的Vector和Cube核心数，以及ub的存储容量等。

```C++
#include <tiling/platform/platform_ascendc.h>    //添加头文件

ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    LOG_PRINT("get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).\n",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}
```

接着，调用API获取输入向量的shape和数据类型，并计算32字节对齐的数据量。比如输入的数据类型是float16，占用2个字节，这样16个float16作为最小的分配和计算单位，在代码中以ALIGN_NUM表示。

如果输入的shape不满足32字节对齐，还需要先进行32字节对齐，对齐后再进行Tiling计算。

### Matmul tiling API

方便用户获取[Matmul kernel计算](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0614.html#ZH-CN_TOPIC_0000002370251917__li5878185413338)时所需的Tiling参数。用户只需要传入A/B/C矩阵的Position位置、Format格式和DType数据类型等信息，调用API接口，即可获取到[Init](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0630.html)中TCubeTiling结构体中的相关参数。

Matmul Tiling API分为Matmul单核Tiling接口、多核Tiling接口和BatchMatmul Tiling接口，分别用于Matmul单核计算、多核计算和BatchMatmul计算场景。获取Tiling参数的流程如下：

1. 创建一个单核Tiling对象，或多核Tiling对象，或BatchMatmul Tiling对象。
2. 设置A、B、C、Bias的参数类型信息；M、N、Ka、Kb形状信息等。
3. 调用[GetTiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/API/ascendcopapi/atlasascendc_api_07_0692.html)接口，获取Tiling信息。

## 5.AI框架算子适配

AI框架调用时，除了需要提供CANN框架调用时需要的代码实现文件，还需要进行插件适配开发。

参考：[适配开发-Ascend Extension for PyTorch7.1.0-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/Frameworkfeatures/featuresguide_00021.html)

### 环境配置

因为我们之前的torch_npu是使用二进制软件包方式安装的。适配前需执行如下命令拉取torch_npu仓对应分支的代码并进入OpPlugin目录。

```Plain
git clone https://gitcode.com/ascend/pytorch.git -b v2.6.0-7.1.0 --recursive
cd pytorch/third_party/op-plugin
```

- *2.6.0*为PyTorch版本，用户需根据实际情况指定PyTorch版本。
- 7.1.0为Ascend Extension for PyTorch软件版本。

适配文件结构：

```Python
.
├── op_plugin
│   ├── config                         # 算子配置文件目录
│   │   ├── derivatives.yaml          # 算子前反向绑定配置文件
│   │   └── op_plugin_functions.yaml  # 算子对外接口配置文件
│   ├── ops                            # 算子适配文件目录
│   │   ├── aclops                    # aclop算子适配目录
│   │   │   ├── AbsKernelNpu.cpp
│   │   │   └── ...
│   │   └── opapi                     # aclnn算子适配目录
│   │       ├── sparse                # sparse相关算子适配目录
│   │       │   └── SparseTensorUtils.h
│   │       ├── AbsKernelNpuOpApi.cpp
│   │       └── ...
│   ├── OpInterface.h                   # 编译PyTorch框架后自动生成op_plugin对外接口的头文件，用于框架侧调用算子
│   ├── OpInterface.cpp               # 编译PyTorch框架后自动生成op_plugin对外接口路由实现，内部实现不同类型算子分支选择代码
│   ├── AclOpsInterface.h             # 编译PyTorch框架后自动生成aclop算子插件适配所对应头文件 
│   ├── OpApiInterface.h              # 编译PyTorch框架后自动生成aclnn算子插件适配所对应头文件
│   ├── ...    
```

执行如下命令打开op_plugin_functions.yaml文件进行算子yaml配置。

```Plain
vim op_plugin/config/op_plugin_functions.yaml
```

添加接口信息：注意版本问题

**一定要注意****tab****对齐!!!!!!**

```Plain
custom: 
  - func: npu_add_custom(Tensor x, Tensor y) -> Tensor 
    op_api: v2.6
  - func: npu_add_custom_backward(Tensor grad) -> (Tensor, Tensor)
    op_api: v2.6
```

打开derivatives.yaml文件，进行自定义算子的前反向注册绑定。

```Plain
vim op_plugin/config/derivatives.yaml
```

将如下信息拷贝至derivatives.yaml文件的backward节点中。

```Plain
- name: npu_add_custom(Tensor x, Tensor y) -> Tensor
  x, y: npu_add_custom_backward(grad)
  version: v2.6
```

！！！在op_plugin/ops/opapi目录下，创建AddCustomKernelOpApi.cpp文件并实现算子适配主体函数npu_add_custom和npu_add_custom_backward。其核心逻辑为调用EXEC_NPU_CMD接口，完成输出结果的计算，EXEC_NPU_CMD第一个入参格式为aclnn+Optype（算子类型），之后的参数分别为输入输出。其中由于add操作的反向计算相对简单，因此不需要调用算子进行计算。**接口部分需注意！！！若维度变化，则相应需要在接口处进行修改**

```C++
#include "op_plugin/OpApiInterface.h" 
#include "op_plugin/utils/op_api_common.h" 

namespace op_api { 
using npu_preparation = at_npu::native::OpPreparation; 

// 正向接口，可选操作，已结构化适配，可无需添加
at::Tensor npu_add_custom(const at::Tensor& x, const at::Tensor& y)
{ 
    // 构造输出tensor 
    at::Tensor result = npu_preparation::apply_tensor_without_format(x); 
    // 计算输出结果
    // 调用EXEC_NPU_CMD接口，完成输出结果的计算
    // 第一个入参格式为aclnn+Optype，之后的参数分别为输入输出
    EXEC_NPU_CMD(aclnnAddCustom, x, y, result); 
    return result; 
}

// 反向接口
std::tuple<at::Tensor, at::Tensor> npu_add_custom_backward(const at::Tensor& grad)
{
    // 构造输出tensor
    at::Tensor result = npu_preparation::apply_tensor_without_format(grad);
    result.copy_(grad);
    // 计算输出结果
    return {result, result};
}
}  // namespace op_api
```

编译Ascend Extension for PyTorch插件并安装。

(需要在docker环境下进行编译，实际测试本机编译可能由于gcc,cmake版本问题，安装之后无法引入该模块)

docker操作：

```Plain
cd ~/ascend/pytorch/ci/docker/ARM
docker build -t manylinux-builder:v2 .
(上面只需执行一次)
#每次编译前进入docker环境再编译

docker run -it -v /home/aaa/ascend/pytorch:/home/pytorch manylinux-builder:v2 bash
cd /home/pytorch
bash ci/build.sh --python=3.9

pip3 install --upgrade dist/torch_npu-2.6.0.post13-cp39-cp39-linux_aarch64.whl --user
```

然后进行pytorch测试：

```Python
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

class TestCustomAdd(TestCase):

    def test_add_custom(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)
        print(x, '\n', y)

        output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()

        print(output)
        self.assertRtolEqual(output, x + y)


if __name__ == "__main__":
    run_tests()
```

## 6.算子测试

### AddCustom测试

- 设置环境变量

  - ```Bash
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest/arm64-linux/
    export NPU_HOST_LIB=$HOME/Ascend/ascend-toolkit/8.2.RC1/arm64-linux/devlib
    ```

- 生成并执行测试用例

  - 运行命令`msopst run -i $HOME/wenjie/learn/addCustom/AddCustom_case.json -soc Ascend910B3 -out $HOME/wenjie/learn/addCustom/AddCustom_st`
  - msOpST工具无需进入其所在目录使用

- 测试完成

  - 得到结果

  - ```Bash
    ========================================================================
    run command: /home/aaa/Ascend/ascend-toolkit/latest/bin/msopst run -i /home/aaa/b/learn/addCustom/AddCustom_case.json -soc Ascend910B3 -out /home/aaa/b/learn/addCustom/AddCustom_st
    ------------------------------------------------------------------------
    - test case count: 1
    - success count: 1
    - failed count: 0
    ------------------------------------------------------------------------
    ========================================================================
    ```

## 算子入图开发

在图模式下用户首先将模型的计算过程构造成一张图，然后通过GE将图下发到昇腾硬件执行。相对于单个算子依次下发的方式，图模式下，GE可以通过计算图优化、多流并行、内存复用、模型下沉等技术手段，加速模型执行效率，减少模型内存占用。

开发流程（区别）：

- 完成算子入图（GE图）开发，需要提供shape推导等算子入图适配函数的实现。
- 基于图IR执行算子
- 需要额外交付算子入图的代码文件