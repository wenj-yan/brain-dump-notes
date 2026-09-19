---
title: 笔记
outline: false
---

<script setup>
import { data } from './recent.data'
import { withBase } from 'vitepress'
</script>

# 笔记索引

这里是我的学习笔记，按主题分类整理，持续更新中。

## 算子设计

### CUDA

- [reduction 算子优化](/notes/operator/cuda/reduction)
- [softmax 算子优化](/notes/operator/cuda/softmax)

### CANN

- [Ascend C编程范式与完整流程](/notes/operator/cann/AscendC)

## 量化

### 工具

- [AMCT 量化工具实践记录](/notes/quant/tools/amct)

## 最近更新

<div v-for="p in data" :key="p.url" style="margin: 6px 0">
  {{ p.date }} · <a :href="withBase(p.url)">{{ p.title }}</a>
</div>