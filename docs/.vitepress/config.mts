import { defineConfig } from 'vitepress'

const base = '/brain-dump-notes/'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Brain Dump Notes",
  description: "记录成长点滴，我的进阶之旅",
  base,
  lastUpdated: true,
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}terminal.svg` }]],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '笔记', link: '/notes/' },
      { text: '博客', link: '/blog/' },
      { text: '日志', link: '/daily/' },
      { text: '关于', link: '/about' }
    ],

    sidebar: {
      '/notes/': [
        {
          text: '算子设计',
          collapsed: false,
          items: [
            {
              text: 'CUDA',
              collapsed: false,
              items: [
                { text: '[算子][CUDA]reduction算子优化', link: '/notes/operator/cuda/reduction' },
                { text: '[算子][CUDA]softmax算子优化', link: '/notes/operator/cuda/softmax' },
              ]
            },
            {
              text: 'CANN',
              collapsed: false,
              items: [
                { text: '[CANN]AscendC算子开发全流程记录', link: '/notes/operator/cann/AscendC' },
              ]
            }
          ],
        },
        {
          text: '量化',
          collapsed: false,
          items: [
            {
              text: '工具',
              collapsed: false,
              items: [
                { text: '[量化][工具]AMCT量化工具实践记录', link: '/notes/quant/tools/amct' },
              ]
            },
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
    footer: {
      message: '记录成长点滴，我的进阶之旅',
      copyright: 'Powered by VitePress',
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/wenj-yan/brain-dump-notes' },
    ],
  }
})
