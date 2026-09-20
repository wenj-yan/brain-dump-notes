---
title: Blog
outline: false
---

<script setup>
import { data } from './posts.data'
</script>

# Blog

成篇的技术文章与随笔，比笔记更完整，比日志更有条理。

<PostList :posts="data">
  <template #empty>
    <p>还没有文章。在 <code>docs/blog/</code> 下新建一个 <code>.md</code> 文件，写好下面的 frontmatter，刷新页面就会自动出现在这里：</p>
    <pre><code>---
title: 文章标题
date: 2026-09-20
tags: [CUDA, 性能优化]
description: 一句话摘要，会显示在列表里
---</code></pre>
  </template>
</PostList>

## 怎么整理

| 内容 | 放哪 |
| --- | --- |
| 成篇的技术文章、踩坑记录、随笔 | `docs/blog/` |
| 日常碎片、当天做了什么 | `docs/daily/` |
| 系统性的主题知识 | `docs/notes/` |
