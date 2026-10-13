import DefaultTheme from 'vitepress/theme'
import PostList from './PostList.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('PostList', PostList)
  },
}
