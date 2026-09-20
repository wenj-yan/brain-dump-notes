import { createContentLoader } from 'vitepress'
import { byDateDesc, toPost } from '../.vitepress/utils/posts'

export default createContentLoader('notes/**/*.md', {
  includeSrc: true, // 不带上 src 就拿不到 h1 兜底
  transform(raw) {
    return raw
      .filter((p) => !p.url.endsWith('/')) // 排除 index.md 自身
      .map((p) => toPost(p))
      .filter((p) => p.date)
      .sort(byDateDesc) // 新的在前
      .slice(0, 10) // 只显示最近 10 篇
  },
})
