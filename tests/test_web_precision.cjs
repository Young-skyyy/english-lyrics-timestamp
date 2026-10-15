const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = new Map();
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {
    value: selector === '#offsetSlider' ? '0' : '',
    selectedOptions: [{ textContent: 'Rap' }],
    classList: { add() {}, remove() {}, toggle() {} },
    listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
  });
  return elements.get(selector);
}
const context = vm.createContext({
  document: { querySelector: element, querySelectorAll: () => [] },
  window: { location: { reload() {} } },
  alert(message) { throw new Error(message); },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8'), context);
assert.equal(vm.runInContext('formatTime(59.999)', context), '01:00.00');
assert.equal(vm.runInContext('formatTime(60.001)', context), '01:00.00');

(async () => {
  vm.runInContext(`
    state.audioFile = { name: 'test.wav' };
    lyricsInput.value = 'silver stars';
    requestAlignment = async () => ({
      global_offset_ms: 500, transcription_mode: 'rap', status: 'approved', issue_count: 0,
      lines: [{text: 'silver stars', start_ms: 1123, content_confidence: .98,
               timing_confidence: .98, review_status: 'auto_pass'}]
    });
  `, context);
  await element('#analyzeButton').listeners.click();
  assert.equal(element('#offsetSlider').value, '0');
  assert.match(element('#lyricsPreview').innerHTML, /data-time="1.123"/);
  assert.equal(element('#metricOffset').textContent, '+0.50s');
  element('#offsetSlider').value = '0.01';
  vm.runInContext('renderPreview()', context);
  assert.match(element('#lyricsPreview').innerHTML, /data-time="1.133"/);
  assert.equal(element('#offsetValue').textContent, '+0.01s');
  assert.equal(element('#metricOffset').textContent, '+0.50s');
  console.log('Web precision checks passed (rollover, no double offset, millisecond preview, manual adjustment).');
})().catch(error => { console.error(error); process.exitCode = 1; });
