const $ = (selector) => document.querySelector(selector);

const state = { audioFile: null, audioUrl: null, lyricText: '', parsedLines: [], offset: 0, duration: 0, result: null };

const audioInput = $('#audioInput');
const audioDropzone = $('#audioDropzone');
const audioPlayer = $('#audioPlayer');
const lyricsInput = $('#lyricsInput');
const lyricsFileInput = $('#lyricsFileInput');
const lyricsDropzone = $('#lyricsDropzone');
let lyricsImportVersion = 0;

function formatTime(seconds) {
  const centiseconds = Math.round(Math.max(0, seconds) * 100);
  const minutes = Math.floor(centiseconds / 6000);
  const remainder = centiseconds % 6000;
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(remainder / 100)).padStart(2, '0')}.${String(remainder % 100).padStart(2, '0')}`;
}

function parseLyrics(text) {
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).map((line, index) => {
    if (/^\[[^\]]+\]$/.test(line)) return null;
    const match = line.match(/^(?:\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\])?\s*(.*)$/);
    if (!match || !match[4]) return null;
    const [, minutes, seconds, fraction, lyric] = match;
    const importedSeconds = minutes == null ? null : Number(minutes) * 60 + Number(seconds) + Number(`0.${(fraction || '0').padEnd(2, '0')}`);
    return { index, text: lyric.trim(), importedSeconds };
  }).filter(Boolean);
}

function renderPreview() {
  const offsetSeconds = Number($('#offsetSlider').value);
  state.offset = offsetSeconds;
  $('#offsetValue').textContent = `${offsetSeconds >= 0 ? '+' : ''}${offsetSeconds.toFixed(2)}s`;
  const detectedOffset = state.result ? state.result.global_offset_ms / 1000 : offsetSeconds;
  $('#metricOffset').textContent = `${detectedOffset >= 0 ? '+' : ''}${detectedOffset.toFixed(2)}s`;
  const lines = state.result?.lines || state.parsedLines.map((line, index) => ({
    text: line.text,
    start_ms: line.importedSeconds == null ? null : Math.round(line.importedSeconds * 1000),
    timing_confidence: null,
    content_confidence: null,
    review_status: 'local_preview',
  }));
  $('#lyricsPreview').innerHTML = lines.map((line) => {
    const seconds = line.start_ms == null ? null : Math.max(0, line.start_ms / 1000 + offsetSeconds);
    const confidence = line.timing_confidence == null ? '' : ` · 内容 ${(line.content_confidence * 100).toFixed(0)}% · 时间 ${(line.timing_confidence * 100).toFixed(0)}%`;
    const issueClass = line.review_status === 'needs_review' ? ' active' : '';
    return `<div class="lyric-line${issueClass}" ${seconds == null ? '' : `data-time="${seconds.toFixed(3)}"`}><span class="lyric-time">${seconds == null ? '--:--.--' : formatTime(seconds)}</span><span class="lyric-text">${escapeHtml(line.text)}<small class="line-confidence">${confidence}</small></span></div>`;
  }).join('');
  $('#metricLines').textContent = lines.length;
}

async function requestAlignment(onStatus) {
  const form = new FormData();
  form.append('audio', state.audioFile);
  form.append('lyrics', lyricsInput.value);
  const importedLrc = state.lyricText.match(/^\s*\[\d{1,3}:\d{2}/m) ? state.lyricText : '';
  if (importedLrc) form.append('imported_lrc', importedLrc);
  form.append('language', 'en');
  form.append('mode', $('#modeSelect').value);
  form.append('duration_ms', String(Math.round(state.duration * 1000)));
  const taskId = crypto.randomUUID().replaceAll('-', '');
  form.append('task_id', taskId);
  let polling = true;
  const pollStatus = async () => {
    while (polling) {
      const statusResponse = await fetch(`/api/status/${taskId}`);
      const status = await statusResponse.json();
      if (status.message) onStatus(status.message);
      if (status.stage === 'completed' || status.stage === 'failed') break;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  };
  const pollingTask = pollStatus();
  const response = await fetch('/api/align', { method: 'POST', body: form });
  polling = false;
  await pollingTask;
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || '对齐请求失败');
  return payload;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function setAudio(file) {
  if (!file) return;
  state.audioFile = file;
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = URL.createObjectURL(file);
  audioPlayer.src = state.audioUrl;
  audioPlayer.hidden = false;
  $('#audioTitle').textContent = file.name;
  $('#audioMeta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 正在读取时长`;
  audioPlayer.onloadedmetadata = () => {
    state.duration = audioPlayer.duration;
    $('#audioMeta').textContent = `${formatTime(state.duration)} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $('#metricDuration').textContent = formatTime(state.duration);
    if (state.parsedLines.length) renderPreview();
  };
}

function setLyrics(text, filename = '') {
  const changed = state.lyricText !== text;
  state.lyricText = text;
  lyricsInput.value = text;
  state.parsedLines = parseLyrics(text);
  $('#lineCount').textContent = `${state.parsedLines.length} 行`;
  if (filename) $('#lyricsFileMeta').textContent = `${filename} · 已读取 ${state.parsedLines.length} 行`;
  if (changed) {
    state.result = null;
    $('#resultStatus').textContent = '歌词已更新，等待生成对齐';
    renderPreview();
  }
}

function setLyricsSource(source) {
  document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item.dataset.source === source));
  $('#textSource').classList.toggle('hidden', source !== 'text');
  $('#fileSource').classList.toggle('hidden', source !== 'file');
}

async function importLyricsFile(file) {
  if (!file) return;
  const version = ++lyricsImportVersion;
  const status = $('#lyricsFileMeta');
  status.classList.remove('error');
  try {
    if (!/\.(txt|lrc|srt|vtt|zip)$/i.test(file.name)) {
      throw new Error('请拖入 TXT、LRC、SRT、VTT 或笔记 ZIP 文件。');
    }
    if (file.size > 2 * 1024 * 1024) throw new Error('歌词文件不能超过 2 MB。');
    status.textContent = `正在读取 ${file.name}…`;
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/lyrics/import', { method: 'POST', body: form });
    const payload = await response.json();
    if (version !== lyricsImportVersion) return;
    if (!response.ok) throw new Error(payload.detail || '歌词文件读取失败，请重新尝试。');
    setLyrics(payload.text, file.name);
    const format = payload.format === 'lrc' ? '含时间标签的歌词' : '纯文本歌词';
    status.textContent = `${file.name} · ${format} · 已读取 ${payload.line_count} 行`;
    setLyricsSource('text');
  } catch (error) {
    if (version !== lyricsImportVersion) return;
    status.textContent = error.message;
    status.classList.add('error');
  } finally {
    if (version === lyricsImportVersion) lyricsFileInput.value = '';
  }
}

audioInput.addEventListener('change', (event) => setAudio(event.target.files[0]));
['dragenter', 'dragover'].forEach((eventName) => audioDropzone.addEventListener(eventName, (event) => { event.preventDefault(); audioDropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => audioDropzone.addEventListener(eventName, (event) => { event.preventDefault(); audioDropzone.classList.remove('dragging'); }));
audioDropzone.addEventListener('drop', (event) => setAudio(event.dataTransfer.files[0]));

lyricsInput.addEventListener('input', (event) => setLyrics(event.target.value));
lyricsFileInput.addEventListener('change', async (event) => {
  await importLyricsFile(event.target.files[0]);
});

[lyricsDropzone, lyricsInput].forEach((target) => {
  ['dragenter', 'dragover'].forEach((name) => target.addEventListener(name, (event) => {
    if (![...event.dataTransfer.types].includes('Files')) return;
    event.preventDefault();
    event.stopPropagation();
    target.classList.add('dragging');
  }));
  target.addEventListener('dragleave', (event) => {
    if (![...event.dataTransfer.types].includes('Files')) return;
    event.preventDefault();
    event.stopPropagation();
    target.classList.remove('dragging');
  });
  target.addEventListener('drop', async (event) => {
    const files = [...event.dataTransfer.files];
    if (!files.length) return;
    event.preventDefault();
    event.stopPropagation();
    target.classList.remove('dragging');
    if (files.length !== 1) {
      $('#lyricsFileMeta').textContent = '一次请拖入一个歌词文件。';
      $('#lyricsFileMeta').classList.add('error');
      return;
    }
    await importLyricsFile(files[0]);
  });
});

document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
  setLyricsSource(tab.dataset.source);
}));

$('#offsetSlider').addEventListener('input', renderPreview);
$('#analyzeButton').addEventListener('click', async () => {
  setLyrics(lyricsInput.value);
  if (!state.audioFile) return alert('请先添加音频文件。');
  if (!state.parsedLines.length) return alert('请先输入或导入歌词。');
  const button = $('#analyzeButton');
  button.disabled = true;
  button.innerHTML = '正在识别与对齐 <span>…</span>';
  $('#resultStatus').textContent = `${$('#modeSelect').selectedOptions[0].textContent} 处理中`;
  try {
    state.result = await requestAlignment((message) => {
      $('#resultStatus').textContent = message;
    });
    $('#emptyState').classList.add('hidden');
    $('#resultState').classList.remove('hidden');
    const modeNames = { fast: 'Fast', rap: 'Rap', quality: 'Quality' };
    const actualMode = modeNames[state.result.transcription_mode] || state.result.transcription_mode;
    $('#resultStatus').textContent = `${actualMode} · ${state.result.status === 'approved' ? '自动通过' : `需要审核 · ${state.result.issue_count} 个问题`}`;
    $('#metricOffset').textContent = `${state.result.global_offset_ms >= 0 ? '+' : ''}${(state.result.global_offset_ms / 1000).toFixed(2)}s`;
    // Result timestamps already come from the audio. The LRC delta is diagnostic.
    $('#offsetSlider').value = '0';
    renderPreview();
  } catch (error) {
    $('#resultStatus').textContent = '处理失败';
    alert(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = '重新生成对齐 <span>→</span>';
  }
});

$('#clearButton').addEventListener('click', () => window.location.reload());
$('#downloadButton').addEventListener('click', () => {
  const lines = [...document.querySelectorAll('.lyric-line[data-time]')].map((element) => {
    const textElement = element.querySelector('.lyric-text');
    const text = textElement.firstChild?.textContent || '';
    return `[${element.querySelector('.lyric-time').textContent}]${text}`;
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${state.audioFile?.name.replace(/\.[^.]+$/, '') || 'lyrics'}.lrc`;
  link.click();
  URL.revokeObjectURL(link.href);
});

audioPlayer.addEventListener('timeupdate', () => {
  const timedLines = [...document.querySelectorAll('.lyric-line[data-time]')];
  timedLines.forEach((line, index) => line.classList.toggle('active',
    Number(line.dataset.time) <= audioPlayer.currentTime &&
    audioPlayer.currentTime < Number(timedLines[index + 1]?.dataset.time || Infinity)));
});
