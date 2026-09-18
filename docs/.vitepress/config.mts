import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Brain Dump Notes",
  description: "记录成长点滴，我的进阶之旅",
  base: '/brain-dump-notes/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'home', link: '/' },
      { text: 'notes', link: '/notes/' },
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: {
      '/notes/': [
        {
          text: '入门',
          items: [
            { text: '第一篇笔记', link: '/notes/first' },
          ],
        },
      ],
    },

    search: { provider: 'local' },          // 免配置全文搜索
    outline: { level: [2, 3], label: '本页目录' },
    editLink: {
      pattern: 'https://github.com/wenj-yan/brain-dump-notes/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },
    lastUpdatedText: '最近更新',
    docFooter: { prev: '上一篇', next: '下一篇' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/wenj-yan/brain-dump-notes' },
    ],

    
  }
})
