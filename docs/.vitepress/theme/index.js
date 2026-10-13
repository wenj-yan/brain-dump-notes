import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import PostList from './PostList.vue'
import VisitStats from './VisitStats.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-bottom': () => h(VisitStats),
    })
  },
  enhanceApp({ app }) {
    app.component('PostList', PostList)
  },
}
