<template>
  <div class="terminal" @click="focusInput">
    <div class="terminal-bar">
      <span class="dot red"></span>
      <span class="dot yellow"></span>
      <span class="dot green"></span>
      <span class="terminal-title">brain-dump — zsh</span>
    </div>

    <div class="terminal-body" ref="bodyRef">
      <div v-for="(line, i) in lines" :key="i" class="line">
        <span v-if="line.type === 'cmd'" class="prompt">➜ ~ </span>
        <span v-if="line.type === 'cmd'" class="cmd">{{ line.text }}</span>
        <span v-else class="out" v-html="line.text"></span>
      </div>

      <div class="line input-line">
        <span class="prompt">➜ ~ </span>
        <input
          ref="inputRef"
          v-model="input"
          @keydown.enter="run"
          @keydown.up.prevent="historyUp"
          @keydown.down.prevent="historyDown"
          spellcheck="false"
          autocomplete="off"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'

const lines = ref([
  { type: 'out', text: 'Welcome to <b>brain-dump-notes</b> terminal.' },
  { type: 'out', text: 'Type <span class="hl">help</span> to see available commands.' },
  { type: 'out', text: '&nbsp;' },
])

const input = ref('')
const inputRef = ref(null)
const bodyRef = ref(null)
const history = ref([])
const historyIndex = ref(-1)

const commands = {
  help: () => [
    'Available commands:',
    '  <span class="hl">help</span>      show this help',
    '  <span class="hl">ls</span>        list sections',
    '  <span class="hl">cat</span> &lt;file&gt;  read a section',
    '  <span class="hl">whoami</span>    about me',
    '  <span class="hl">date</span>      current time',
    '  <span class="hl">clear</span>     clear screen',
  ],
  ls: () => [
    '<span class="dir">markdown-examples/</span>  <span class="dir">api-examples/</span>  <span class="file">README.md</span>',
  ],
  whoami: () => ['brain-dump-notes — 记录成长点滴，我的进阶之旅'],
  date: () => [new Date().toString()],
  cat: (args) => {
    const target = args[0]
    if (!target) return ['cat: missing file operand']
    const files = {
      'README.md': '记录成长点滴，我的进阶之旅。',
      'markdown-examples': 'Markdown 示例，见 /markdown-examples',
      'api-examples': 'API 示例，见 /api-examples',
    }
    return files[target]
      ? [files[target]]
      : [`cat: ${target}: No such file or directory`]
  },
}

async function run() {
  const raw = input.value.trim()
  if (!raw) return

  lines.value.push({ type: 'cmd', text: raw })
  history.value.push(raw)
  historyIndex.value = -1
  input.value = ''

  const [cmd, ...args] = raw.split(/\s+/)

  if (cmd === 'clear') {
    lines.value = []
  } else if (commands[cmd]) {
    const out = commands[cmd](args)
    out.forEach((t) => lines.value.push({ type: 'out', text: t }))
  } else {
    lines.value.push({
      type: 'out',
      text: `<span class="err">command not found: ${cmd}</span>`,
    })
  }

  await nextTick()
  bodyRef.value.scrollTop = bodyRef.value.scrollHeight
}

function historyUp() {
  if (!history.value.length) return
  if (historyIndex.value === -1) historyIndex.value = history.value.length
  historyIndex.value = Math.max(0, historyIndex.value - 1)
  input.value = history.value[historyIndex.value]
}

function historyDown() {
  if (historyIndex.value === -1) return
  historyIndex.value = Math.min(history.value.length - 1, historyIndex.value + 1)
  input.value = history.value[historyIndex.value]
}

function focusInput() {
  inputRef.value?.focus()
}

onMounted(() => {
  inputRef.value?.focus()
})
</script>

<style scoped>
.terminal {
  max-width: 720px;
  margin: 0 auto;
  border-radius: 10px;
  overflow: hidden;
  background: #1e1e2e;
  border: 1px solid #313244;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  text-align: left;
  cursor: text;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}

.terminal-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: #313244;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
.red { background: #f38ba8; }
.yellow { background: #fab387; }
.green { background: #a6e3a1; }

.terminal-title {
  margin-left: 8px;
  font-size: 12px;
  color: #7f849c;
}

.terminal-body {
  padding: 16px;
  height: 320px;
  overflow-y: auto;
  font-size: 14px;
  line-height: 1.6;
  color: #cdd6f4;
}

.line {
  white-space: pre-wrap;
  word-break: break-word;
}

.prompt {
  color: #a6e3a1;
}

.cmd {
  color: #cdd6f4;
}

.out :deep(.hl) { color: #89b4fa; }
.out :deep(.err) { color: #f38ba8; }
.out :deep(.dir) { color: #89b4fa; }
.out :deep(.file) { color: #f9e2af; }
.out :deep(b) { color: #cba6f7; }

.input-line {
  display: flex;
  align-items: center;
}

.input-line input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: #cdd6f4;
  font-family: inherit;
  font-size: inherit;
  caret-color: #a6e3a1;
}

.terminal-body::-webkit-scrollbar {
  width: 8px;
}
.terminal-body::-webkit-scrollbar-thumb {
  background: #45475a;
  border-radius: 4px;
}
</style>