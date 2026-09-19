import { createContentLoader } from 'vitepress'
import { execSync } from 'node:child_process'

function gitDate(file: string): string {
  try {
    return execSync(`git log -1 --format=%cI -- "${file}"`, { encoding: 'utf-8' }).trim().slice(0, 10)
  } catch {
    return ''
  }
}

export default createContentLoader('notes/**/*.md', {
  includeSrc: true,                                 // 不带上 src 就拿不到 h1 兜底
  transform(raw) {
    return raw
      .filter(p => !p.url.endsWith('/'))            // 排除 index.md 自身
      .map(p => {
        const file = `docs${p.url.replace(/\.html$/, '')}.md`   // url 可能带 .html 后缀
        return {
          title: p.frontmatter.title
            ?? p.src?.match(/^#\s+(.+)$/m)?.[1]?.replace(/`/g, '')   // 没写 title 就取第一个 h1，去掉反引号
            ?? p.url,
          url: p.url,
          date: gitDate(file),
        }
      })
      .filter(p => p.date)
      .sort((a, b) => (a.date < b.date ? 1 : -1))   // 新的在前
      .slice(0, 10)                                  // 只显示最近 10 篇
  },
})