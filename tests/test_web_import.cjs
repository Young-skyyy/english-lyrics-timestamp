const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map();
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    value: '0', listeners: {}, classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(name, handler) { this.listeners[name] = handler; },
  });
  return elements.get(selector);
}
let requestCount = 0;
const context = vm.createContext({
  document: { querySelector: element, querySelectorAll: () => [] },
  window: {},
  FormData: class { append() {} },
  fetch: async (url) => {
    requestCount++;
    assert.equal(url, '/api/lyrics/import');
    return { ok: true, json: async () => ({ text: '[Chorus]\nsilver stars\nwe walk home', format: 'text', line_count: 2 }) };
  },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8'), context);
const event = (files) => ({ dataTransfer: { files, types: ['Files'] }, preventDefault() { this.prevented = true; }, stopPropagation() {} });
(async () => {
  vm.runInContext("state.result = {lines: [{text: 'old result'}]}", context);
  const drop = event([{ name: 'notes.zip', size: 120 }]);
  await element('#lyricsDropzone').listeners.drop(drop);
  assert.equal(drop.prevented, true);
  assert.equal(element('#lyricsInput').value, '[Chorus]\nsilver stars\nwe walk home');
  assert.match(element('#lyricsFileMeta').textContent, /notes.zip.*2 行/);
  assert.equal(vm.runInContext('state.result', context), null);
  assert.equal(vm.runInContext('state.parsedLines.length', context), 2);
  const textDrop = event([{name:'lyrics.lrc',size:100}]);
  await element('#lyricsInput').listeners.drop(textDrop);
  assert.equal(textDrop.prevented, true);
  const before = requestCount;
  await element('#lyricsInput').listeners.drop(event([{name:'audio.mp3',size:100}]));
  assert.equal(requestCount, before);
  assert.match(element('#lyricsFileMeta').textContent, /TXT/);
  assert.equal(element('#lyricsInput').value, '[Chorus]\nsilver stars\nwe walk home');
  console.log('Lyric drop checks passed (ZIP/TXT drop, format feedback, section labels, stale-result reset, invalid file retention).');
})().catch(error => { console.error(error); process.exitCode = 1; });
