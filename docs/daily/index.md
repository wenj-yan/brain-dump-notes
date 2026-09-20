---
title: Daily
outline: false
---

<script setup>
import { data } from './posts.data'
</script>

# Daily

日常记录。科研、刷题、读书、生活，想到什么写什么。

<PostList :posts="data">
  <template #empty>
    <p>还没有日志。在 <code>docs/daily/</code> 下按 <code>年/月-日.md</code> 建文件（例如 <code>docs/daily/2026/09-20.md</code>），date 会用来排序：</p>
    <pre><code>---
title: 今天啃完了 softmax 的 warp 规约
date: 2026-09-20
tags: [日常]
---</code></pre>
  </template>
</PostList>
