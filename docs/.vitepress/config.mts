import { defineConfig } from 'vitepress'

const base = '/brain-dump-notes/'

export default defineConfig({
  title: 'Brain Dump Notes',
  description: '学习笔记目录',
  base,
  lastUpdated: true,
  themeConfig: {
    nav: [{ text: '笔记目录', link: '/' }],
    sidebar: {
      '/': [
        {
          text: 'paper read',
          items: [],
        },
        {
          text: 'verl-omni',
          collapsed: false,
          items: [
            {
              text: 'PR思路学习',
              collapsed: false,
              items: [
                { text: '如何为 verl-omni 增加 DGPO 算法', link: '/notes/verl-omni/PR思路学习/verl-omni-dgpo-pr-733' },
              ]
            },
          ],
        },
        {
          text: 'vllm-omni',
          collapsed: false,
          items: [
            {
              text: 'PR思路学习',
              collapsed: false,
              items: [
                { text: 'MiniMax-H3 的 MXFP8 量化与 SwiGLU 融合优化——PR#7988', link: '/notes/vllm-omni/PR思路学习/minimax-h3-mxfp8-pr-7988' },
              ]
            },
          ],
        },
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
    search: { provider: 'local' },
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
  },
})
