import { createContentLoader } from 'vitepress'
import { byDateDesc, toPost } from '../.vitepress/utils/posts'

export default createContentLoader('blog/**/*.md', {
  includeSrc: true, // 没有 description 时从正文抽摘要
  transform(raw) {
    return raw
      .filter((p) => !p.url.endsWith('/')) // 排除 index.md 自身
      .map((p) => toPost(p))
      .filter((p) => p.date)
      .sort(byDateDesc)
  },
})
