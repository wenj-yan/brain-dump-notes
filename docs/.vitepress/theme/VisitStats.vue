<script setup>
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()
const local = ref(false)
let script
let stopWatching
let disposed = false

async function countVisit() {
  await nextTick()
  if (disposed || local.value) return

  if (window.bszCaller && window.bszTag) {
    const path = route.path
    window.bszCaller.fetch(
      'https://busuanzi.ibruce.info/busuanzi?jsonpCallback=BusuanziCallback',
      (data) => {
        if (disposed || route.path !== path) return
        window.bszTag.texts(data)
        window.bszTag.shows()
      },
    )
    return
  }

  if (script) return
  script = document.createElement('script')
  script.src = 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js'
  script.async = true
  script.onerror = () => {
    script.remove()
    script = undefined
  }
  document.head.appendChild(script)
}

onMounted(() => {
  local.value = ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)
  countVisit()
  stopWatching = watch(() => route.path, countVisit)
})

onUnmounted(() => {
  disposed = true
  stopWatching?.()
  script?.remove()
})
</script>

<template>
  <div class="visit-stats" aria-label="网站访问统计">
    <template v-if="local">访客人数 / 浏览次数：上线后统计</template>
    <template v-else>
      <span id="busuanzi_container_site_uv">访客人数 <span id="busuanzi_value_site_uv">—</span> 人</span>
      <span aria-hidden="true"> · </span>
      <span id="busuanzi_container_site_pv">浏览次数 <span id="busuanzi_value_site_pv">—</span> 次</span>
    </template>
  </div>
</template>

<style scoped>
.visit-stats {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  font-size: 12px;
  line-height: 1.8;
  text-align: center;
}
</style>
