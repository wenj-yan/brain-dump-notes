import { createContentLoader } from 'vitepress'
import { byDateDesc, toPost } from './.vitepress/utils/posts'

/** 首页「最近更新」覆盖的板块 */
const sections = [
  { prefix: '/notes/', label: '笔记' },
  { prefix: '/blog/', label: '博客' },
  { prefix: '/daily/', label: '日志' },
]

export default createContentLoader(['notes/**/*.md', 'blog/**/*.md', 'daily/**/*.md'], {
  includeSrc: true,
  transform(raw) {
    return raw
      .filter((p) => !p.url.endsWith('/')) // 排除各板块的 index.md
      .map((p) => ({
        ...toPost(p),
        section: sections.find((s) => p.url.startsWith(s.prefix))?.label ?? '',
      }))
      .filter((p) => p.date)
      .sort(byDateDesc)
      .slice(0, 8)
  },
})
