import DefaultTheme from 'vitepress/theme'
import Terminal from './Terminal.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Terminal', Terminal)
  },
}