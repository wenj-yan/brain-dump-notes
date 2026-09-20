import { createContentLoader } from 'vitepress'
import { byDateDesc, toPost } from '../.vitepress/utils/posts'

export default createContentLoader('daily/**/*.md', {
  includeSrc: true,
  transform(raw) {
    return raw
      .filter((p) => !p.url.endsWith('/'))
      .map((p) => toPost(p))
      .filter((p) => p.date)
      .sort(byDateDesc)
  },
})
