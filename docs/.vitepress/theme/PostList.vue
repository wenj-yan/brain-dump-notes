<template>
  <div v-if="posts.length" class="post-list" :class="{ compact }">
    <a v-for="p in posts" :key="p.url" class="post-card" :href="withBase(p.url)">
      <div class="post-head">
        <span v-if="p.section" class="post-section">{{ p.section }}</span>
        <span class="post-date">{{ p.date }}</span>
      </div>

      <component :is="compact ? 'span' : 'h3'" class="post-title">{{ p.title }}</component>

      <p v-if="!compact && p.description" class="post-desc">{{ p.description }}</p>

      <div v-if="!compact && p.tags && p.tags.length" class="post-tags">
        <span v-for="t in p.tags" :key="t" class="post-tag">#{{ t }}</span>
      </div>
    </a>
  </div>

  <div v-else class="post-empty">
    <slot name="empty">这里还没有内容。</slot>
  </div>
</template>

<script setup>
import { withBase } from 'vitepress'

defineProps({
  posts: { type: Array, default: () => [] },
  /** 紧凑模式：单行列表，用在首页「最近更新」 */
  compact: { type: Boolean, default: false },
})
</script>

<style scoped>
.post-list {
  display: grid;
  gap: 12px;
  margin: 20px 0;
}

.post-card {
  display: block;
  padding: 14px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}

.post-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
}

.post-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.post-section {
  padding: 1px 6px;
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 4px;
  color: var(--vp-c-brand-1);
}

.post-title {
  display: block;
  margin: 6px 0 0;
  padding: 0;
  border: none;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.5;
  color: var(--vp-c-text-1);
  transition: color 0.25s;
}

.post-card:hover .post-title {
  color: var(--vp-c-brand-1);
}

.post-desc {
  margin: 6px 0 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}

.post-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.post-tag {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--vp-c-text-2);
}

/* 紧凑模式 */
.post-list.compact {
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  overflow: hidden;
}

.compact .post-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  border: none;
  border-radius: 0;
  border-bottom: 1px solid var(--vp-c-divider);
  background: transparent;
  transform: none;
}

.compact .post-card:last-child {
  border-bottom: none;
}

.compact .post-card:hover {
  background: var(--vp-c-bg-mute);
  box-shadow: none;
}

.compact .post-head {
  flex-shrink: 0;
}

.compact .post-title {
  flex: 1;
  margin: 0;
  overflow: hidden;
  font-size: 14px;
  font-weight: 500;
  white-space: nowrap;
  text-overflow: ellipsis;
}

@media (max-width: 640px) {
  .compact .post-card {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }

  .compact .post-title {
    white-space: normal;
  }
}

.post-empty {
  margin: 20px 0;
  padding: 24px;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
}
</style>
