---
title: AMCT 量化工具实践记录
---

# AMCT

## 0.项目简介

项目地址：`https://gitcode.com/cann/amct/`

昇腾 NPU 原生的模型量化压缩工具，支持多种方法。

**亮点为：**

- **🎯 硬件亲和** —— 量化结果直接对接昇腾 NPU 低比特运算单元
- **🔢 多精度全栈** —— INT8 / INT4 / MXFP8 / MXFP4 / HiFloat8 任选
- **🚀 大模型就绪** —— 原生支持 DeepSeek-V3.2 / V4 等前沿模型

**✨ 核心特性**

| 特性类别           | 简介                                                         |
| ------------------ | ------------------------------------------------------------ |
| **PTQ 量化算法**   | Min-Max / AWQ / GPTQ / SmoothQuant 等训练后量化算法 |
| **HiFloat8 量化**  | 华为自研 8-bit 浮点格式，锥形精度 + 大动态范围|
| **NPU 自定义算子** | 基于NPU的自研算子，Ascend C kernel 实现 |
| **大模型量化**     | DeepSeek-V3.2 / V4 量化方案，详见 |

**📊 性能收益**

量化显著降低部署成本：

| 精度格式     | 仅权重（W）              | 全量化（W+A）            | 收益                       |
| ------------ | ------------------------ | ------------------------ | -------------------------- |
| **INT8**     | ✅ Min-Max / AWQ / GPTQ   | ✅ Min-Max / SmoothQuant  | 体积 **↓50%** · 吞吐 ↑     |
| **INT4**     | ✅ AWQ / GPTQ             | ✅ FlatQuant              | 体积 **↓75%** · 低带宽友好 |
| **HiFloat8** | ✅ Cast / Quantile / OFMR | ✅ Cast / Quantile / OFMR | 体积 **↓50%** · 大动态范围 |
| **MXFP8**    | ✅ MXQuant                | ✅ MXQuant                | 体积 **↓50%** · 高精度     |
| **MXFP4**    | ✅ MXQuant                | ✅ MXQuant                | 体积 **↓75%** · 微缩浮点   |

## 1.环境配置

1.单独安装`torch_cpu`(正常pip install找不到路径)

```
git clone https://gitcode.com/cann/amct.git  #拉取源码

pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cpu
```

2.安装剩余包

```
pip install -r requirement.txt
```

请按照`readme`进行安装，需要注意的地方会提醒。

## 2.量化

### AWQ

案例使用：

```
python3 src/run_qwen_samples.py --model_path=/data/home/5110075_01/scow/ai/appData/ascend-k8s-devHost-20260708-112448/Qwen2.5-7B-Instruct
```

**报错：**

```
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.7.1+cpu).

OSError: /usr/local/python3.12.13/lib/python3.12/site-packages/torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl
```

验证发现，是`torchaudio`库和`torch`版本不兼容，但是`requirement`里面已经写了版本`2.7.1`，主要是他在README里面还强调后端加速依赖这个CPU版本的`torch`.

```
python -c "import torchaudio; print('OK')"
# 报错：undefined symbol: torch_library_impl
```

解决方法：

```
# 卸载当前的 torchaudio
pip uninstall torchaudio -y

# 安装与 torch 2.7.1+cpu 匹配的版本
# 注意：需要指定与 torch 版本严格匹配的 torchaudio
pip install torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

量化后测试

报错：

```
huggingface_hub.errors.HfUriError: Invalid HF URI 'hf://datasets/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3/.huggingface.yaml'. Repository id must be 'namespace/name', got 'wikitext'.
```

新版 `huggingface_hub` 要求仓库 ID 必须包含命名空间（如 `namespace/dataset_name`），但 `datasets` 库在解析 `wikitext` 时只传了 `wikitext`，没有命名空间部分。

这通常发生在：

- `datasets` 版本较旧（< 2.16.0）
- `huggingface_hub` 版本较新（>= 0.20.0）

解决方法：

```
#降级 huggingface_hub
#这个方法不对，必须升级，降级之后会有新的问题
pip install huggingface_hub==0.19.4

#正确方法：
# 先卸载旧版本
pip uninstall datasets huggingface_hub -y
# 安装兼容的版本组合
pip install huggingface_hub==0.26.0 datasets==3.0.0

#还是不行，有更多的依赖问题
# 清理并安装兼容版本
pip uninstall transformers huggingface_hub datasets accelerate compressed-tensors -y
# 安装满足所有要求的版本
# compressed-tensors 需要 transformers>=4.45.0
# accelerate 1.14.0 需要 huggingface_hub>=0.21.0
pip install transformers==4.46.3 huggingface_hub==0.26.5 datasets==3.1.0 accelerate==1.1.0 compressed-tensors==0.15.0.1
```

实验结果：

```
root@dev-260708-192420-1783509896-worker-0:/project/toolkits/amct/examples/algorithms/awq# python3 src/run_qwen_samples.py --model_path=/data/home/5110075_01/scow/ai/appData/ascend-k8s-devHost-20260708-112448/Qwen2.5-7B-Instruct
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.7.1+cpu).
Getting official pretrained /data/home/5110075_01/scow/ai/appData/ascend-k8s-devHost-20260708-112448/Qwen2.5-7B-Instruct
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.65s/it]
Loading dataset: pileval
Repo card metadata block was not found. Setting CardData to empty.
 * Split into 490 blocks
2026-07-08 21:03:27,081 - INFO - [AMCT]:[Optimizer]: Do <class 'amct_pytorch.classic.optimizer.insert_quantize_op_pass.InsertQuantizeModulePass'>
[W708 21:03:28.089185099 ToKernelNpu.cpp:41] Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
Calibration time taken:  0.0 min  39.76632118225098 s
2026-07-08 21:04:07,302 - INFO - [AMCT]:[Optimizer]: Do <class 'amct_pytorch.classic.optimizer.replace_npu_quant_pass.ReplaceNpuQuantModulePass'>
Loading dataset: Wikitext2
test-00000-of-00001.parquet: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 733k/733k [00:00<00:00, 1.17MB/s]
train-00000-of-00001.parquet: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.36M/6.36M [00:00<00:00, 12.5MB/s]
validation-00000-of-00001.parquet: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 657k/657k [00:00<00:00, 11.4MB/s]
Generating test split: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 4358/4358 [00:00<00:00, 337508.34 examples/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:00<00:00, 748639.89 examples/s]
Generating validation split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 3760/3760 [00:00<00:00, 633077.08 examples/s]
Token indices sequence length is longer than the specified maximum sequence length for this model (299078 > 131072). Running this sequence through the model will result in indexing errors
evaluating...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 146/146 [00:32<00:00,  4.47it/s]
Test time taken:  0.0 min  32.634257555007935 s
Score:  7.814728736877441(未压缩Score:  7.457449436187744)(GPU :7.75)
```

但是我不用这个工具测未压缩的PPL也不一样，

`GPU/NPU-scripts`:

```
==================================================
Model: /data/model_weight/Qwen2.5-7B-Instruct
Device: cuda
Dtype: torch.float16
Dataset: wikitext
Samples: 2891
Loss: 2.5882
Perplexity: 13.31
==================================================
```

改成滑窗方式测试：

`GPU/NPU-scripts_windows`:

```
==================================================
Model: /scow/ai/appData/ascend-k8s-devHost-20260708-112448/Qwen2.5-7B-Instruct
Device: npu
Dtype: torch.float16
Dataset: wikitext
Loss: 1.9980
Perplexity: 7.37
==================================================
```

基本对齐。
