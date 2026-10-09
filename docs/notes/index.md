---
title: 笔记
outline: false
---

<script setup>
import { data } from './recent.data'
</script>

# 笔记索引

这里是我的学习笔记，按主题分类整理，持续更新中。

## 算子设计

### CUDA

- [reduction 算子优化](/notes/operator/cuda/reduction)
- [softmax 算子优化](/notes/operator/cuda/softmax)

### CANN

- [Ascend C 编程范式与完整流程](/notes/operator/cann/AscendC)

## 量化

### 工具

- [AMCT 量化工具实践记录](/notes/quant/tools/amct)

## vllm-omni

### PR思路学习

- [MiniMax-H3 的 MXFP8 量化与 SwiGLU 融合优化——PR#7988](/notes/vllm-omni/PR思路学习/minimax-h3-mxfp8-pr-7988)

## 最近更新

<PostList :posts="data" compact />
