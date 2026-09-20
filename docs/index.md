---
layout: home

hero:
  name: "Brain Dump Notes"
  text: "记录成长点滴，我的进阶之旅"
  tagline: "> 一个计算机研究生的笔记 / 博客 / 日志"
  actions:
    - theme: brand
      text: $ cd notes
      link: /notes/
    - theme: alt
      text: $ ls blog
      link: /blog/

features:
  - icon: "📒"
    title: Notes
    details: "系统整理的学习笔记：算子设计、量化、模型部署……"
    link: /notes/
    linkText: cd notes
  - icon: "✍️"
    title: Blog
    details: "成篇的技术文章与随笔，记录踩过的坑和想清楚的事。"
    link: /blog/
    linkText: 读文章
  - icon: "🗓️"
    title: Daily
    details: "科研、刷题、读书的日常碎片，按时间沉淀。"
    link: /daily/
    linkText: 看日志
  - icon: "🫧"
    title: About
    details: "我是谁、在做什么、怎么联系我。"
    link: /about
    linkText: 认识一下
---

<script setup>
import { data } from './recent.data'
</script>

## 最近更新

<PostList :posts="data" compact />

## 终端

<Terminal />
