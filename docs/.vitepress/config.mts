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
      { text: 'notes', link: '/notes/index.md' },
      { text: 'Examples', link: '/markdown-examples' }
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
              items:[
                { text: '[算子][CUDA]reduction算子优化', link: '/notes/reduction' },
              { text: '[算子][CUDA]softmax算子优化', link: '/notes/softmax' },
              ]
            },
            {
              text: 'CANN',
              collapsed: false, 
              items:[
                { text: '[CANN]AscendC算子开发全流程记录', link: '/notes/AscendC' },

              ]
            }
            

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
