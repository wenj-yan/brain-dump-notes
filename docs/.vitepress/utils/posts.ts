import { execSync } from 'node:child_process'
import { statSync } from 'node:fs'

/** 数据加载器里每条原始记录的最小结构 */
export interface RawPage {
  url: string
  src?: string
  frontmatter: Record<string, any>
}

/** 列表卡片需要的文章元信息 */
export interface Post {
  title: string
  url: string
  date: string
  description: string
  tags: string[]
  section?: string
}

/** 页面 URL 转源文件相对路径：/blog/hello.html -> docs/blog/hello.md */
export function sourceFile(url: string): string {
  const clean = url.replace(/\.html$/, '')
  return `docs${clean.endsWith('/') ? `${clean}index` : clean}.md`
}

/** 取文件的最后一次提交日期，拿不到就返回空串 */
function gitDate(file: string): string {
  try {
    return execSync(`git log -1 --format=%cI --follow -- "${file}"`, { encoding: 'utf-8' })
      .trim()
      .slice(0, 10)
  } catch {
    return ''
  }
}

/** 取文件修改时间兜底，用于还没提交的新文章 */
function fileDate(file: string): string {
  try {
    return statSync(file).mtime.toISOString().slice(0, 10)
  } catch {
    return ''
  }
}

/** frontmatter 里的 date 归一化成 YYYY-MM-DD */
function normalizeDate(value: unknown): string {
  if (!value) return ''
  const date = value instanceof Date ? value : new Date(String(value))
  return Number.isNaN(date.getTime()) ? '' : date.toISOString().slice(0, 10)
}

/** 从正文里抽一段纯文本摘要（跳过标题、代码块、HTML 标签行） */
function excerpt(src: string, limit = 88): string {
  const lines = src.replace(/^---[\s\S]*?\n---\n/, '').split('\n')
  const picked: string[] = []
  let inCode = false

  for (const line of lines) {
    const text = line.trim()
    if (text.startsWith('```')) {
      inCode = !inCode
      continue
    }
    if (inCode) continue
    if (!text) {
      if (picked.length) break
      continue
    }
    if (text.startsWith('#') || text.startsWith('<') || text.startsWith(':::')) continue
    picked.push(text)
    if (picked.join(' ').length >= limit) break
  }

  const text = picked
    .join(' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/[*_>#]/g, '')
    .trim()

  return text.length > limit ? `${text.slice(0, limit)}…` : text
}

/** 把一条原始记录转成卡片数据 */
export function toPost(page: RawPage): Post {
  const fm = page.frontmatter ?? {}
  const file = sourceFile(page.url)

  return {
    title:
      fm.title ??
      page.src?.match(/^#\s+(.+)$/m)?.[1]?.replace(/`/g, '') ??
      page.url,
    url: page.url,
    date: normalizeDate(fm.date) || gitDate(file) || fileDate(file),
    description: fm.description ?? (page.src ? excerpt(page.src) : ''),
    tags: Array.isArray(fm.tags) ? fm.tags : [],
  }
}

/** 按日期倒序，新的在前 */
export function byDateDesc(a: Post, b: Post): number {
  return a.date < b.date ? 1 : a.date > b.date ? -1 : 0
}
